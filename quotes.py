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
quotes.py — the fast clock.

The screener runs on three separate clocks, because one refresh interval is the
wrong shape for a daily-bar strategy:

  scan      full universe, full history. Nothing new appears until a daily bar
            closes, so running this every minute finds exactly the same setups
            it found a minute ago. Default: manual, or once after the close.

  forming   re-runs the gates on the CURRENT, still-open daily bar for symbols
            already on screen. This is the one that actually earns a fast
            interval: the rules say enter at the close of the bounce day, so at
            21:45 EET you need to know which names are shaping up while you can
            still act. Default: 60s.

  quote     just the price, for rows already on screen. Has the entry level run
            away? Has it broken the stop intraday? Default: 5s.

Forming-bar results REPAINT. A bounce that looks clean at 16:00 can close red.
Rows built from an open bar are marked provisional and must be treated as a
heads-up, not a signal.

Ceiling: MCP is request/response, so "live" here means polling as fast as
cTrader will answer — realistically ~1s. Genuine streaming needs the FIX price
feed. Your FIX credentials are useless for history but are exactly right for
this, and the two are complementary: MCP for bars and account, FIX for ticks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Clocks:
    quote_interval: float = 5.0      # 0 = off
    forming_interval: float = 60.0   # 0 = off
    scan_interval: float = 0.0       # 0 = manual only
    max_chase_pct: float = 1.5       # how far past entry before it is a chase


@dataclass
class Watchlist:
    """Symbols the fast clocks care about — whatever is currently on screen."""
    symbols: set[str] = field(default_factory=set)
    entries: dict[str, dict] = field(default_factory=dict)
    strategy_of: dict[str, str] = field(default_factory=dict)

    def load(self, rows: list[dict]) -> None:
        self.symbols = {r["symbol"] for r in rows}
        self.entries = {r["symbol"]: {"entry": r["entry"], "stop": r["stop"],
                                      "target": r["target"]} for r in rows}
        self.strategy_of = {r["symbol"]: r.get("strategy", "") for r in rows}


def price_state(levels: dict, price: float, max_chase_pct: float) -> dict:
    """What the live price means against a setup that is already defined."""
    entry, stop, target = levels["entry"], levels["stop"], levels["target"]
    from_entry = 100 * (price - entry) / entry
    if price <= stop:
        state, note = "stopped", "below the stop level"
    elif price >= target:
        state, note = "target", "at or through the target"
    elif from_entry > max_chase_pct:
        state, note = "chase", f"{from_entry:+.2f}% past the entry"
    elif from_entry < -0.25:
        state, note = "better", f"{from_entry:+.2f}% under the entry"
    else:
        state, note = "at_entry", f"{from_entry:+.2f}% from the entry"
    return {"price": round(price, 4), "from_entry": round(from_entry, 2),
            "state": state, "note": note, "ts": time.time()}


async def poll_quotes(mcp, symbols: list[str]) -> dict[str, float]:
    """One price per symbol. Uses a bulk tool if cTrader exposes one."""
    if not symbols or "quotes" not in mcp.resolved:
        return {}
    out: dict[str, float] = {}

    async def one(sym: str):
        try:
            raw = await mcp.call("quotes", symbol=sym)
            p = _extract_price(raw)
            if p:
                out[sym] = p
        except RuntimeError:
            pass

    await asyncio.gather(*(one(s) for s in symbols))
    return out


def _extract_price(raw) -> float | None:
    """bid/ask/last/close, wrapped in whatever the server felt like."""
    def walk(o):
        if isinstance(o, dict):
            low = {str(k).lower(): v for k, v in o.items()}
            bid, ask = _num(low.get("bid")), _num(low.get("ask"))
            if bid and ask:
                return (bid + ask) / 2
            for k in ("last", "price", "close", "lastprice", "mid", "c"):
                v = _num(low.get(k))
                if v:
                    return v
            for v in o.values():
                got = walk(v)
                if got:
                    return got
        elif isinstance(o, list):
            for v in o:
                got = walk(v)
                if got:
                    return got
        return None
    return walk(raw)


def _num(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


async def ensure_subscribed(mcp, symbols: list[str], name: str = "Screener") -> dict:
    """Put the current setups in a watchlist so cTrader will quote them.

    Live quotes are only served for SUBSCRIBED symbols. A name with no open
    chart and no watchlist entry answers "symbol is unsubscribed", which is why
    only the two symbols that happened to have charts were updating.
    """
    need = {"watchlist_get", "watchlist_add"}
    if not need <= set(mcp.resolved):
        return {"ok": False, "why": "no watchlist tools resolved"}

    existing: set[str] = set()
    have_list = False
    try:
        raw = await mcp.call("watchlist_get")
        for wl in _walk_watchlists(raw):
            if str(wl.get("name", "")).lower() == name.lower():
                have_list = True
                existing = {str(s).split("-")[0].upper() for s in (wl.get("symbols") or [])}
                break
    except RuntimeError:
        pass

    if not have_list and "watchlist_create" in mcp.resolved:
        try:
            await mcp.call("watchlist_create", name=name)
        except RuntimeError:
            pass

    added, failed = 0, []
    for s in symbols:
        if s.split("-")[0].upper() in existing:
            continue
        try:
            await mcp.call("watchlist_add", name=name, symbol=s)
            added += 1
        except RuntimeError as e:
            failed.append(f"{s}: {str(e)[:80]}")
    return {"ok": True, "watchlist": name, "added": added,
            "already": len(existing), "failed": failed}


def _walk_watchlists(raw) -> list[dict]:
    if isinstance(raw, dict):
        for k in ("watchlists", "data", "result"):
            if isinstance(raw.get(k), list):
                return [w for w in raw[k] if isinstance(w, dict)]
    return raw if isinstance(raw, list) else []
