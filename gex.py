# cTrader Screener - a self-hosted stock screener sourcing data from cTrader.
# Copyright (C) 2026 Janne Savolainen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. It is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for details:
# <https://www.gnu.org/licenses/>.

"""
gex.py - dealer gamma exposure from the free CBOE delayed-quotes chain.

Source: https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json
No key, no account, 15-minute delay. Indices take an underscore prefix (_SPX,
_NDX); the bare index symbol returns 403.

CBOE ships a `gamma` field per contract, so nothing here re-derives greeks from
Black-Scholes. That removes a whole class of error - our IV assumptions cannot
disagree with theirs, because we do not have any.

WHAT THIS IS NOT
----------------
GEX is inference, not measurement, and it is worth being blunt about where the
softness is:

  * Dealer positioning is ASSUMED. The standard convention - dealers long calls,
    short puts - is a heuristic, not an observation. Nobody publishes what
    dealers actually hold.
  * Open interest updates once a day, after the close. Intraday GEX moves only
    because spot and IV moved, not because positioning did.
  * Methodologies differ enough that two GEX charts of the same underlying on
    the same day can disagree on the sign. There is no canonical number to be
    right about.
  * It means nothing on names with thin chains. A stock with 200 contracts
    outstanding has no dealer hedging flow worth modelling.

So this is treated as context, never as a gate. It flags, it does not reject.

NOT IN THE BACKTEST
-------------------
The free endpoint serves only the current chain. Historical options data costs
money, so a backtest that consulted GEX would be scoring past trades against
levels that did not exist on those dates. `backtest.py` never imports this
module, and it must stay that way.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; ctrader-screener)"}

# cTrader/broker names -> CBOE tickers. Indices need the underscore.
INDEX_MAP = {
    "US500": "_SPX", "SPX500": "_SPX", "SP500": "_SPX", "US.500": "_SPX",
    "NAS100": "_NDX", "USTEC": "_NDX", "NASDAQ100": "_NDX",
    "US2000": "_RUT", "RUSSELL2000": "_RUT",
    "US30": "_DJX", "VIX": "_VIX",
}
SUFFIXES = (".US", ".NAS", ".NYSE", ".CFD", ".ARCA")

# OCC symbol: ROOT + YYMMDD + C|P + strike * 1000, zero padded to 8
OCC = re.compile(r"^(?P<root>[A-Z0-9\.]+?)(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})"
                 r"(?P<cp>[CP])(?P<strike>\d{8})$")


def cboe_ticker(symbol: str) -> str:
    """AAPL.US -> AAPL, US500 -> _SPX."""
    s = symbol.upper().strip()
    if s in INDEX_MAP:
        return INDEX_MAP[s]
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return INDEX_MAP.get(s, s)


@dataclass
class Contract:
    strike: float
    expiry: date
    is_call: bool
    gamma: float
    oi: float
    dte: int


def parse_chain(payload: dict, spot_hint: float | None = None) -> tuple[float, list[Contract]]:
    data = payload.get("data") or {}
    spot = _num(data.get("close")) or _num(data.get("current_price")) or spot_hint or 0.0
    today = datetime.now(timezone.utc).date()
    out: list[Contract] = []
    for o in data.get("options") or []:
        m = OCC.match(str(o.get("option", "")))
        if not m:
            continue
        g, oi = _num(o.get("gamma")), _num(o.get("open_interest"))
        if not g or not oi:
            continue                      # no gamma or nobody holds it
        try:
            exp = date(2000 + int(m["y"]), int(m["m"]), int(m["d"]))
        except ValueError:
            continue
        out.append(Contract(strike=int(m["strike"]) / 1000.0, expiry=exp,
                            is_call=m["cp"] == "C", gamma=float(g), oi=float(oi),
                            dte=(exp - today).days))
    return spot, out


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def compute(spot: float, contracts: list[Contract], *, strike_range_pct: float = 12.0,
            max_dte: int = 45, min_oi: float = 1.0) -> dict:
    """Aggregate per-strike gamma exposure and pull the levels off it.

    GEX per contract = gamma * OI * 100 * spot^2 * 0.01
      - 100 is the contract multiplier
      - spot^2 * 0.01 converts gamma per $1 into dollars per 1% move
      - calls positive, puts negative: the dealers-long-calls convention
    """
    if not spot or not contracts:
        return {"ok": False, "why": "no spot or no contracts"}

    lo, hi = spot * (1 - strike_range_pct / 100), spot * (1 + strike_range_pct / 100)
    used = [c for c in contracts
            if lo <= c.strike <= hi and 0 <= c.dte <= max_dte and c.oi >= min_oi]
    if not used:
        return {"ok": False, "why": f"no contracts within {strike_range_pct}% "
                                    f"and {max_dte} DTE"}

    mult = 100 * spot * spot * 0.01
    by_strike: dict[float, dict] = {}
    for c in used:
        v = c.gamma * c.oi * mult * (1 if c.is_call else -1)
        b = by_strike.setdefault(c.strike, {"call": 0.0, "put": 0.0, "net": 0.0, "oi": 0.0})
        b["call" if c.is_call else "put"] += v
        b["net"] += v
        b["oi"] += c.oi

    strikes = sorted(by_strike)
    net = sum(b["net"] for b in by_strike.values())

    # the flip: where cumulative gamma crosses zero, walking strikes upward
    cum, flip = 0.0, None
    prev_s, prev_c = None, 0.0
    for s in strikes:
        cum += by_strike[s]["net"]
        if prev_s is not None and (prev_c < 0 <= cum or prev_c > 0 >= cum):
            span = cum - prev_c
            flip = s if abs(span) < 1e-9 else prev_s + (s - prev_s) * (-prev_c / span)
            break
        prev_s, prev_c = s, cum

    call_wall = max(strikes, key=lambda s: by_strike[s]["call"], default=None)
    put_wall = min(strikes, key=lambda s: by_strike[s]["put"], default=None)
    top = sorted(strikes, key=lambda s: abs(by_strike[s]["net"]), reverse=True)[:6]

    # A missing flip is information, not an error: it means cumulative gamma
    # keeps the same sign right across the sampled strikes, so the nearest
    # regime change is further out than we looked.
    flip_note = None
    if flip is None:
        flip_note = (f"no gamma flip within {strike_range_pct:.0f}% of spot - "
                     f"cumulative gamma stays {'positive' if net > 0 else 'negative'} "
                     f"across the whole sampled range")

    return {
        "ok": True,
        "spot": round(spot, 4),
        "net_gex": round(net, 0),
        "regime": "positive" if net > 0 else "negative",
        "regime_note": ("dealers dampen moves - mean reversion, ranges hold"
                        if net > 0 else
                        "dealers amplify moves - trends extend, ranges break"),
        "flip": round(flip, 2) if flip else None,
        "flip_dist_pct": round(100 * (spot - flip) / spot, 2) if flip else None,
        "flip_note": flip_note,
        "call_wall": round(call_wall, 2) if call_wall else None,
        "call_wall_pct": round(100 * (call_wall - spot) / spot, 2) if call_wall else None,
        "put_wall": round(put_wall, 2) if put_wall else None,
        "put_wall_pct": round(100 * (put_wall - spot) / spot, 2) if put_wall else None,
        "contracts_used": len(used),
        "total_oi": int(sum(b["oi"] for b in by_strike.values())),
        "levels": [{"strike": s, "net": round(by_strike[s]["net"], 0),
                    "pct": round(100 * (s - spot) / spot, 2)} for s in sorted(top)],
        "params": {"strike_range_pct": strike_range_pct, "max_dte": max_dte},
    }


# ---------------------------------------------------------------------------
# fetching, with a cache sized to the 15-minute delay
# ---------------------------------------------------------------------------

class GexFeed:
    def __init__(self, cfg: dict, store):
        self.cfg = cfg or {}
        self.store = store
        self.ttl = float(self.cfg.get("cache_ttl_s", 900))   # matches the delay
        self._sem = asyncio.Semaphore(int(self.cfg.get("concurrency", 3)))

    async def one(self, symbol: str) -> dict:
        """Levels for one symbol. Cached, because the data only moves every 15m."""
        tick = cboe_ticker(symbol)
        key = f"gex:{tick}"
        hit = self.store.get(key, max_age_s=self.ttl)
        if hit:
            return hit | {"cached": True}

        async with self._sem:
            try:
                async with httpx.AsyncClient(timeout=60, follow_redirects=True,
                                             headers=UA) as c:
                    r = await c.get(CBOE.format(tick))
                if r.status_code == 404:
                    out = {"ok": False, "symbol": symbol, "cboe": tick,
                           "why": "no listed options for this symbol"}
                    self.store.put(key, out)      # cache the miss too
                    return out
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                return {"ok": False, "symbol": symbol, "cboe": tick,
                        "why": f"{type(e).__name__}: {str(e)[:120]}"}

        spot, contracts = parse_chain(payload)
        out = compute(spot, contracts,
                      strike_range_pct=float(self.cfg.get("strike_range_pct", 12)),
                      max_dte=int(self.cfg.get("max_dte", 45)),
                      min_oi=float(self.cfg.get("min_oi", 1)))
        out |= {"symbol": symbol, "cboe": tick,
                "as_of": datetime.now(timezone.utc).isoformat(),
                "delay": "CBOE delayed quotes, roughly 15 minutes"}
        self.store.put(key, out)
        return out

    async def many(self, symbols: list[str]) -> dict[str, dict]:
        cap = int(self.cfg.get("max_symbols", 40))
        picked = symbols[:cap]
        results = await asyncio.gather(*(self.one(s) for s in picked),
                                       return_exceptions=True)
        out = {}
        for s, r in zip(picked, results):
            out[s] = r if isinstance(r, dict) else {"ok": False, "why": str(r)[:120]}
        return out


# ---------------------------------------------------------------------------
# what it means for a setup that already passed the gates
# ---------------------------------------------------------------------------

def setup_notes(row: dict, g: dict) -> list[dict]:
    """Check the trade's own levels against the gamma structure.

    Flags only. GEX never rejects a setup - it is inference, and the gates are
    measurement.
    """
    if not g or not g.get("ok"):
        return []
    notes: list[dict] = []
    entry = row.get("entry")
    target, stop = row.get("target"), row.get("stop")
    cw, pw, flip = g.get("call_wall"), g.get("put_wall"), g.get("flip")
    if not entry:
        return []

    if cw and target and entry < cw < target:
        notes.append({"key": "target_beyond_call_wall", "weight": 2,
                      "text": f"the call wall sits at {cw} ({g['call_wall_pct']:+.1f}%), "
                              f"between entry and the {target} target - price often "
                              f"stalls there, so the reward leg may not pay in full"})
    if pw and stop and stop < pw < entry:
        notes.append({"key": "stop_below_put_wall", "weight": 1,
                      "text": f"the put wall at {pw} ({g['put_wall_pct']:+.1f}%) sits "
                              f"above the {stop} stop - that support has to fail first"})
    if pw and stop and pw < stop:
        notes.append({"key": "stop_above_put_wall", "weight": 2,
                      "text": f"the stop {stop} is above the put wall {pw}, so it sits "
                              f"in front of the support rather than behind it"})

    # Proximity matters as much as ordering. A stop a fraction below a wall is
    # the worst place for it: price wicks to the level that would have held,
    # takes the stop on the way, and turns. This was missed by the ordering
    # checks above, which only looked at what sits BETWEEN entry and target.
    near = float(row.get("_wall_near_pct") or 0.6)
    if pw and stop and abs(100 * (stop - pw) / pw) <= near:
        notes.append({"key": "stop_on_put_wall", "weight": 2,
                      "text": f"the stop {stop} sits within {near}% of the put wall {pw} - "
                              f"a wick to that support would take you out just before "
                              f"the level that might have held it"})
    if cw and target and abs(100 * (target - cw) / cw) <= near:
        notes.append({"key": "target_on_call_wall", "weight": 1,
                      "text": f"the target {target} lands on the call wall {cw} - that is "
                              f"where price tends to stall, so the level is realistic but "
                              f"getting filled at it is not guaranteed"})
    if entry and cw and abs(100 * (entry - cw) / cw) <= near:
        notes.append({"key": "entry_on_call_wall", "weight": 2,
                      "text": f"entry {entry} is right at the call wall {cw} - buying into "
                              f"the heaviest gamma concentration, where moves tend to die"})
    if g.get("regime") == "negative":
        notes.append({"key": "negative_gamma", "weight": 1,
                      "text": "net gamma is negative: dealer hedging amplifies moves, "
                              "so expect wider swings and more stop noise"})
    if flip and g.get("flip_dist_pct") is not None and abs(g["flip_dist_pct"]) < 1.0:
        notes.append({"key": "at_flip", "weight": 2,
                      "text": f"price is within 1% of the gamma flip at {flip}, where "
                              f"hedging behaviour inverts - direction is unstable here"})
    return notes


# ---------------------------------------------------------------------------
# what these levels tend to do
# ---------------------------------------------------------------------------

def level_guide(g: dict) -> list[dict]:
    """Plain explanations of each level, phrased for where price actually is.

    Written as tendencies, not rules. All of this rests on the assumption that
    dealers are long calls and short puts, which nobody verifies because nobody
    publishes dealer inventory.
    """
    if not g or not g.get("ok"):
        return []
    spot = g["spot"]
    out = []

    cw, cwp = g.get("call_wall"), g.get("call_wall_pct")
    if cw:
        below = spot < cw
        out.append({
            "level": "Call wall", "price": cw, "pct": cwp,
            "what": ("The strike holding the most positive gamma. Dealers hedging a long "
                     "gamma position sell into strength and buy into weakness, which bleeds "
                     "energy out of moves as price approaches."),
            "likely": (f"Price is {abs(cwp):.1f}% below it. Rallies tend to slow or stall "
                       f"here, and price often drifts toward it and sits there into expiry. "
                       f"A target above this level has to get through it first."
                       if below else
                       f"Price is {cwp:+.1f}% past it, so it has already broken. Once above, "
                       f"the same hedging that capped the move can chase it, and moves "
                       f"extend more easily."),
            "watch": "Walls are recomputed daily and move sharply after a monthly expiry.",
        })

    pw, pwp = g.get("put_wall"), g.get("put_wall_pct")
    if pw:
        out.append({
            "level": "Put wall", "price": pw, "pct": pwp,
            "what": ("The strike holding the heaviest put positioning. This is the level "
                     "where the two common readings disagree, and it is worth knowing that "
                     "rather than picking one: it is usually described as support, because "
                     "large put open interest clusters where people defend, but under the "
                     "standard dealer convention that same strike is SHORT gamma, and short "
                     "gamma hedging sells into weakness rather than buying it."),
            "likely": (f"Price is {abs(pwp):.1f}% above it. Expect it to act as a magnet and "
                       f"a place where declines pause. But if it gives way decisively, the "
                       f"hedging below tends to accelerate the move rather than cushion it, "
                       f"so it is a poor place to hide a stop."
                       if spot > pw else
                       f"Price is {pwp:+.1f}% through it, which is the uncomfortable side. "
                       f"Below the heaviest put strike, hedging flow amplifies moves."),
            "watch": "Support here is a tendency, not a floor. Size for it failing.",
        })

    fl, flp = g.get("flip"), g.get("flip_dist_pct")
    if fl:
        above = spot > fl
        out.append({
            "level": "Gamma flip", "price": fl, "pct": flp,
            "what": ("Where cumulative dealer gamma crosses zero, and hedging behaviour "
                     "inverts. Above it dealers are net long gamma and trade against the "
                     "move, damping volatility. Below it they are net short gamma and trade "
                     "with the move, feeding it."),
            "likely": (f"Price sits {abs(flp):.1f}% above the flip: the mean-reverting side. "
                       f"Ranges tend to hold, dips get bought, and breakouts fail more often "
                       f"than they run."
                       if above else
                       f"Price sits {abs(flp):.1f}% below the flip: the trending side. "
                       f"Expect wider swings, more follow-through, and stops taken by noise "
                       f"that would not have mattered above it."),
            "watch": ("Behaviour is least reliable within about 1% of the flip, where it is "
                      "not clearly in either regime."),
        })
    elif g.get("flip_note"):
        out.append({"level": "Gamma flip", "price": None, "pct": None,
                    "what": g["flip_note"],
                    "likely": (f"The whole sampled range is on the "
                               f"{g.get('regime')}-gamma side, so there is no nearby level "
                               f"where the regime changes."),
                    "watch": "Widen strike_range_pct if you need to find where it sits."})
    return out
