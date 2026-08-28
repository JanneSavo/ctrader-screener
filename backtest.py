# cTrader Screener - a self-hosted stock screener sourcing data from cTrader.
# Copyright (C) 2026 JanneSavo
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. It is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for details:
# <https://www.gnu.org/licenses/>.

"""
backtest.py — the missing half. Walk the gates forward and see whether they paid.

Everything else in this project answers "is there a setup". This answers "did
that kind of setup make money", which is a different and much less flattering
question.

What it does NOT use, on purpose: news, earnings, social chatter, the LLM
review. None of those are available at a point in time — the headlines table
only holds what was collected during a recent scan, so replaying them into
2024 would be pure lookahead. The backtest is price and volume only, which is
also the only part of the stack that is honestly reproducible.

Rules of the simulation, all deliberately pessimistic:

  * A signal on bar i is FILLED AT THE OPEN OF BAR i+1. You cannot buy the
    close you just used to decide.
  * Stop and target are the levels the strategy itself computed. If the fill
    gaps past either one, the trade is skipped rather than counted as an
    instant winner.
  * If a bar touches both the stop and the target, the stop wins. Daily bars
    do not say which came first, and assuming the good one is how backtests
    lie.
  * Costs are charged on both sides in basis points, spread included.
  * The regime filter is recomputed from the index bars as of that date, so a
    bull-only strategy does not get to trade the 2022 tape in hindsight.

What it still cannot fix: the universe is the symbol list your broker shows
TODAY, so anything delisted is missing and the results are survivorship-biased
upward. And you are testing gates you already tuned by looking at this data.
Both are printed in the warnings rather than buried here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from strategies import Ctx


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class BTConfig:
    strategy: str = "pullback50"
    symbols: int = 120           # sample size; 0 = every cached symbol
    history: int = 0             # bars of history per symbol; 0 = all cached
    hold_max: int = 40           # time stop, in bars
    use_regime: bool = True      # honour the strategy's regime requirement
    cost_bps: float = 5.0        # per side, spread + commission
    risk_pct: float = 1.0        # of equity, per trade
    max_open: int = 8            # concurrent positions in the portfolio sim
    equity: float = 10_000.0
    seed: int = 7                # symbol sampling, so runs are repeatable

    def clean(self) -> "BTConfig":
        self.symbols = max(0, int(self.symbols))
        self.history = max(0, int(self.history))
        self.hold_max = max(1, min(500, int(self.hold_max)))
        self.cost_bps = max(0.0, min(200.0, float(self.cost_bps)))
        self.risk_pct = max(0.01, min(10.0, float(self.risk_pct)))
        self.max_open = max(1, min(100, int(self.max_open)))
        self.equity = max(100.0, float(self.equity))
        return self


# ---------------------------------------------------------------------------
# regime, as of each bar
# ---------------------------------------------------------------------------


def regime_series(idx: pd.DataFrame, fast: int = 50, slow: int = 200) -> dict[str, bool]:
    """close > 50DMA and 50DMA > 200DMA, per date. Same test as strategy.regime,
    vectorised so the walk does not recompute it 200,000 times."""
    if idx is None or idx.empty:
        return {}
    f = idx["Close"].rolling(fast).mean()
    s = idx["Close"].rolling(slow).mean()
    ok = (idx["Close"] > f) & (f > s)
    return {ts.date().isoformat(): bool(v) for ts, v in ok.items() if pd.notna(v)}


# ---------------------------------------------------------------------------
# one symbol, forward
# ---------------------------------------------------------------------------


def walk(strat, symbol: str, df: pd.DataFrame, ctx: Ctx, cfg: BTConfig,
         reg: dict[str, bool], rejects: dict[str, int] | None = None
         ) -> tuple[list[dict], int]:
    """Every bar where the gates all passed. Returns (signals, bars_walked).

    `rejects` accumulates which gate said no, across every bar of every symbol.
    Without it a run that fires nothing is a shrug; with it you can see that one
    gate rejected 99% of the history and the other six never mattered.

    The evaluation window is clipped to what the strategy actually needs. A
    strategy that wants 265 bars gets 275, never the whole frame — otherwise
    the cost of the walk grows with the square of the history and a five-year
    test never finishes.
    """
    need = int(strat.min_bars)
    win = need + 10
    n = len(df)
    if n < need + 2:
        return [], 0

    gated = cfg.use_regime and getattr(strat, "needs_regime", None) in ("bull", "bear")
    want_bull = getattr(strat, "needs_regime", None) == "bull"

    out, walked, evaluated = [], 0, 0
    # stop one bar early: a signal on the last bar has no next open to fill at
    for i in range(need - 1, n - 1):
        walked += 1
        day = df.index[i].date().isoformat()
        if gated:
            bull = reg.get(day)
            if bull is None or bull != want_bull:
                continue
        sub = df.iloc[max(0, i - win + 1): i + 1]
        try:
            r = strat.evaluate(symbol, sub, ctx)
        except Exception:
            continue
        if not r:
            continue
        evaluated += 1
        if rejects is not None:
            rejects["_evaluated"] = rejects.get("_evaluated", 0) + 1
        if not r.get("pass"):
            if rejects is not None:
                for name in r.get("failed") or []:
                    rejects[name] = rejects.get(name, 0) + 1
            continue
        out.append({"i": i, "date": day, "entry_ref": float(r["entry"]),
                    "stop": float(r["stop"]), "target": float(r["target"]),
                    "rr": float(r.get("rr") or 0)})
    return out, walked


# ---------------------------------------------------------------------------
# one signal, resolved
# ---------------------------------------------------------------------------


def resolve(df: pd.DataFrame, sig: dict, cfg: BTConfig, direction: str = "long") -> dict:
    """Fill at the next open, then walk bars until stop, target or time."""
    i = sig["i"]
    n = len(df)
    j0 = i + 1
    if j0 >= n:
        return {"skip": "no bar to fill on"}

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    cost = cfg.cost_bps / 10_000.0
    long = direction != "short"
    raw_fill = float(o[j0])
    if not math.isfinite(raw_fill) or raw_fill <= 0:
        return {"skip": "no open price"}
    fill = raw_fill * (1 + cost) if long else raw_fill * (1 - cost)
    stop, target = sig["stop"], sig["target"]

    if long and (fill <= stop):
        return {"skip": "gapped through the stop"}
    if long and (fill >= target):
        return {"skip": "gapped past the target"}
    if not long and (fill >= stop):
        return {"skip": "gapped through the stop"}
    if not long and (fill <= target):
        return {"skip": "gapped past the target"}

    risk_ps = abs(fill - stop)
    if risk_ps <= 0:
        return {"skip": "no risk per share"}

    last = min(j0 + cfg.hold_max - 1, n - 1)
    reason, exit_px, jx = "", 0.0, last
    for j in range(j0, last + 1):
        if long:
            hit_stop = l[j] <= stop
            hit_tgt = h[j] >= target
        else:
            hit_stop = h[j] >= stop
            hit_tgt = l[j] <= target
        if hit_stop:                      # stop first, always. See module docstring.
            exit_px = min(stop, float(o[j])) if long else max(stop, float(o[j]))
            reason, jx = "stop", j
            break
        if hit_tgt:
            exit_px = max(target, float(o[j])) if long else min(target, float(o[j]))
            reason, jx = "target", j
            break
    if not reason:
        exit_px, jx = float(c[last]), last
        reason = "time" if last < n - 1 or (last - j0 + 1) >= cfg.hold_max else "open"

    exit_net = exit_px * (1 - cost) if long else exit_px * (1 + cost)
    pnl_ps = (exit_net - fill) if long else (fill - exit_net)
    return {
        "symbol": sig.get("symbol", ""),
        "signal_date": sig["date"],
        "entry_date": df.index[j0].date().isoformat(),
        "exit_date": df.index[jx].date().isoformat(),
        "entry": round(fill, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "exit": round(exit_net, 4),
        "reason": reason,
        "bars": int(jx - j0 + 1),
        "r": round(pnl_ps / risk_ps, 3),
        "pct": round(100 * pnl_ps / fill, 3),
        "risk_ps": risk_ps,
        "planned_rr": sig.get("rr"),
        "slip_pct": round(100 * (fill - sig["entry_ref"]) / sig["entry_ref"], 3),
    }


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------


def portfolio(trades: list[dict], cfg: BTConfig) -> dict:
    """Take signals in date order under a concurrency cap and a risk budget.

    Signal-level stats treat every trade as if you could always take it. You
    cannot: on the good days forty names fire at once and you have eight slots.
    This is the part that turns an R distribution into an equity curve.
    """
    taken, skipped_cap, skipped_dupe = [], 0, 0
    equity = cfg.equity
    peak = equity
    curve = [{"date": trades[0]["entry_date"] if trades else "", "equity": round(equity, 2)}]
    open_pos: list[dict] = []
    maxdd = 0.0

    def close_through(day: str) -> None:
        nonlocal equity, peak, maxdd
        still = []
        for p in sorted(open_pos, key=lambda x: x["exit_date"]):
            if p["exit_date"] <= day:
                equity += p["r"] * p["risk_amt"]
                peak = max(peak, equity)
                maxdd = max(maxdd, (peak - equity) / peak if peak > 0 else 0.0)
                curve.append({"date": p["exit_date"], "equity": round(equity, 2)})
            else:
                still.append(p)
        open_pos[:] = still

    for t in sorted(trades, key=lambda x: (x["entry_date"], x["symbol"])):
        close_through(t["entry_date"])
        if any(p["symbol"] == t["symbol"] for p in open_pos):
            skipped_dupe += 1
            continue
        if len(open_pos) >= cfg.max_open:
            skipped_cap += 1
            continue
        risk_amt = equity * cfg.risk_pct / 100.0
        rec = dict(t)
        rec["risk_amt"] = round(risk_amt, 2)
        rec["pnl"] = round(t["r"] * risk_amt, 2)
        taken.append(rec)
        open_pos.append({"symbol": t["symbol"], "exit_date": t["exit_date"],
                         "r": t["r"], "risk_amt": risk_amt})
    close_through("9999-12-31")

    return {"taken": taken, "skipped_cap": skipped_cap, "skipped_dupe": skipped_dupe,
            "equity_start": round(cfg.equity, 2), "equity_end": round(equity, 2),
            "return_pct": round(100 * (equity / cfg.equity - 1), 2),
            "max_dd_pct": round(100 * maxdd, 2),
            "curve": _thin(curve, 400)}


def _thin(curve: list[dict], keep: int) -> list[dict]:
    if len(curve) <= keep:
        return curve
    step = len(curve) / keep
    out = [curve[int(k * step)] for k in range(keep)]
    out[-1] = curve[-1]
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def stats(trades: list[dict]) -> dict:
    """Signal-level. Every trade at one unit of risk, no capital constraint."""
    done = [t for t in trades if t["reason"] != "open"]
    if not done:
        return {"trades": 0}
    rs = np.array([t["r"] for t in done], dtype=float)
    wins, losses = rs[rs > 0], rs[rs <= 0]
    gross_w, gross_l = float(wins.sum()), float(-losses.sum())
    reasons: dict[str, int] = {}
    for t in done:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    by_month: dict[str, float] = {}
    for t in done:
        m = t["exit_date"][:7]
        by_month[m] = round(by_month.get(m, 0.0) + t["r"], 2)
    return {
        "trades": len(done),
        "still_open": len(trades) - len(done),
        "wins": int((rs > 0).sum()),
        "win_rate": round(100 * float((rs > 0).mean()), 1),
        "avg_r": round(float(rs.mean()), 3),
        "median_r": round(float(np.median(rs)), 3),
        "expectancy_r": round(float(rs.mean()), 3),
        "total_r": round(float(rs.sum()), 2),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None,
        "best_r": round(float(rs.max()), 2),
        "worst_r": round(float(rs.min()), 2),
        "avg_bars": round(float(np.mean([t["bars"] for t in done])), 1),
        "stdev_r": round(float(rs.std(ddof=1)), 3) if len(rs) > 1 else 0.0,
        # Below ten trades a t-stat is arithmetic, not evidence. Two trades that
        # both lost 1R give t = -255, which would be the most confident garbage
        # on the page.
        "t_stat": round(float(rs.mean() / (rs.std(ddof=1) / math.sqrt(len(rs)))), 2)
                  if len(rs) >= 10 and rs.std(ddof=1) > 0 else None,
        "exits": reasons,
        "by_month": dict(sorted(by_month.items())),
    }


def warnings_for(cfg: BTConfig, cov: dict, sig: dict, pf: dict, span: dict) -> list[str]:
    out = []
    n = sig.get("trades", 0)
    worst = (cov.get("rejected_by") or [{}])[0]
    if n == 0:
        out.append("No trade closed. Either the gates never fired on this history, "
                   "or every fill gapped past a level. Check the coverage numbers.")
        if worst.get("gate"):
            out.append(f"The gate that said no most often was “{worst['gate']}” — "
                       f"{worst['pct']}% of the {cov.get('evaluated', 0)} bars it "
                       f"looked at. If nothing ever fires, that is the one to loosen.")
        return out
    if worst.get("pct", 0) >= 95 and cov.get("signals", 0) < 50:
        out.append(f"“{worst['gate']}” rejected {worst['pct']}% of every bar examined. "
                   f"This strategy is that one gate; the rest are decoration.")
    if n < 30:
        out.append(f"{n} trades is not a sample, it is an anecdote. Nothing below "
                   f"this line means anything yet.")
    elif n < 100:
        out.append(f"{n} trades. Enough to notice a pattern, not enough to trust one.")
    if span.get("years", 0) < 1.5:
        out.append(f"The test covers about {span.get('years', 0):.1f} years. That is "
                   f"one market mood, not a cycle.")
    if sig.get("t_stat") is not None and abs(sig["t_stat"]) < 2:
        out.append(f"Average R of {sig['avg_r']} has a t-stat of {sig['t_stat']}. "
                   f"Under 2, that is indistinguishable from luck.")
    if sig.get("exits", {}).get("time", 0) > 0.5 * n:
        out.append("Over half the trades ended on the time stop rather than at a "
                   "level. The target is out of reach on this timeframe.")
    if pf.get("skipped_cap", 0) > len(pf.get("taken", [])):
        out.append(f"The concurrency cap turned away {pf['skipped_cap']} signals — "
                   f"more than it took. The portfolio result is mostly a story "
                   f"about {cfg.max_open} slots, not about the gates.")
    if cfg.cost_bps <= 0:
        out.append("Costs are set to zero. Real spreads on this universe will eat "
                   "a meaningful part of a sub-0.2R edge.")
    out.append("Survivorship: the universe is the symbol list your broker shows "
               "today. Names that went to zero or were delisted are not in it, so "
               "every number here is flattered.")
    out.append("In-sample: these gates were chosen while looking at this market. "
               "A backtest of a strategy you already tuned is a description, not "
               "a prediction.")
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run(store, strat, cfg: BTConfig, index_symbol: str = "US500",
        progress=None) -> dict:
    """Walk every sampled symbol, resolve every signal, then build a portfolio.

    `progress` is called with (done, total, symbol) so the server can stream it.
    Runs synchronously — the caller puts it on a thread.
    """
    cfg.clean()
    started = datetime.now(timezone.utc)

    universe = sorted(store.symbols_with_bars(strat.min_bars + 2))
    universe = [s for s in universe if s != index_symbol]
    total_universe = len(universe)
    if cfg.symbols and cfg.symbols < len(universe):
        rng = np.random.default_rng(cfg.seed)
        pick = rng.choice(len(universe), size=cfg.symbols, replace=False)
        universe = sorted(universe[i] for i in pick)

    reg = regime_series(store.bars(index_symbol, 5000)) if cfg.use_regime else {}
    ctx = Ctx(regime={}, equity=cfg.equity, risk_per_trade=cfg.risk_pct / 100.0)

    limit = cfg.history or 5000
    trades: list[dict] = []
    skips: dict[str, int] = {}
    rejects: dict[str, int] = {}
    walked = signals = 0
    # Span is the MEDIAN symbol's tradable window, not the union. One symbol
    # with four years of history should not make a run over 285-bar caches
    # claim it covered four years.
    firsts: list[str] = []
    last_day = "0000"
    per_symbol: dict[str, int] = {}

    for k, sym in enumerate(universe, 1):
        if progress:
            progress(k, len(universe), sym)
        df = store.bars(sym, limit)
        if df is None or df.empty or len(df) < strat.min_bars + 2:
            continue
        firsts.append(df.index[min(int(strat.min_bars) - 1, len(df) - 1)].date().isoformat())
        last_day = max(last_day, df.index[-1].date().isoformat())
        sigs, w = walk(strat, sym, df, ctx, cfg, reg, rejects)
        walked += w
        signals += len(sigs)
        for s in sigs:
            s["symbol"] = sym
            t = resolve(df, s, cfg, getattr(strat, "direction", "long"))
            if "skip" in t:
                skips[t["skip"]] = skips.get(t["skip"], 0) + 1
                continue
            t["symbol"] = sym
            trades.append(t)
            per_symbol[sym] = per_symbol.get(sym, 0) + 1

    span_days, first_day, earliest = 0, "", ""
    if firsts and last_day != "0000":
        firsts.sort()
        first_day = firsts[len(firsts) // 2]
        earliest = firsts[0]
        span_days = (datetime.fromisoformat(last_day) - datetime.fromisoformat(first_day)).days
    span = {"from": first_day, "to": last_day if last_day != "0000" else "",
            "earliest": earliest, "days": span_days,
            "years": round(span_days / 365.25, 2)}

    sig_stats = stats(trades)
    pf = portfolio(trades, cfg)
    pf_stats = stats(pf["taken"])

    bench = _benchmark(store, index_symbol, span, cfg.history)
    evaluated = rejects.pop("_evaluated", 0)
    cov = {"symbols_cached": total_universe, "symbols_tested": len(universe),
           "symbols_with_trades": len(per_symbol), "bars_walked": walked,
           "evaluated": evaluated,
           "signals": signals, "signals_per_symbol_per_year":
               round(signals / max(1, len(universe)) / max(0.01, span["years"]), 2),
           "skipped": sorted(skips.items(), key=lambda kv: -kv[1]),
           "rejected_by": [{"gate": k, "n": v,
                            "pct": round(100 * v / max(1, evaluated), 1)}
                           for k, v in sorted(rejects.items(), key=lambda kv: -kv[1])]}

    top = sorted(per_symbol.items(), key=lambda kv: -kv[1])[:12]
    return {
        "config": asdict(cfg),
        "strategy": {"key": strat.key, "label": getattr(strat, "label", strat.key),
                     "direction": getattr(strat, "direction", "long"),
                     "needs_regime": getattr(strat, "needs_regime", None),
                     "min_bars": int(strat.min_bars)},
        "span": span,
        "coverage": cov,
        "signal_stats": sig_stats,
        "portfolio_stats": pf_stats,
        "portfolio": {k: v for k, v in pf.items() if k != "taken"},
        "benchmark": bench,
        "trades": sorted(pf["taken"], key=lambda t: t["entry_date"], reverse=True)[:400],
        "busiest": top,
        "warnings": warnings_for(cfg, cov, sig_stats, pf, span),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }


def _benchmark(store, symbol: str, span: dict, limit: int) -> dict:
    """Index buy-and-hold over the same window. The number to beat."""
    df = store.bars(symbol, limit or 5000)
    if df is None or df.empty or not span.get("from"):
        return {}
    d = df[(df.index >= span["from"]) & (df.index <= span["to"] + " 23:59")]
    if len(d) < 2:
        return {}
    ret = 100 * (float(d["Close"].iloc[-1]) / float(d["Close"].iloc[0]) - 1)
    roll = d["Close"].cummax()
    dd = float(((roll - d["Close"]) / roll).max())
    return {"symbol": symbol, "return_pct": round(ret, 2),
            "max_dd_pct": round(100 * dd, 2), "bars": len(d)}
