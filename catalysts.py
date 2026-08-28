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
catalysts.py — earnings dates and news headlines.

Two hard rules, both learned the expensive way:

  1. Earnings dates are fetched, never generated. No model is allowed anywhere
     near a date field. A hallucinated earnings date puts you in a position
     through a print you thought you had avoided.
  2. The blackout is a deterministic gate that runs BEFORE any LLM sees the
     setup. The LLM can add caution, never remove it.

Earnings: one bulk Finnhub call covers the whole universe for a date range,
so 500 symbols cost one request. yfinance is the per-symbol fallback.
News: cTrader's own news tool if your local MCP exposes one, else Finnhub.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

FINNHUB = "https://finnhub.io/api/v1"


@dataclass
class Earnings:
    symbol: str
    date: str | None
    when: str | None          # bmo / amc / dmh
    days_out: int | None
    source: str

    @property
    def known(self) -> bool:
        return self.date is not None


@dataclass
class Story:
    headline: str
    source: str
    url: str
    published: str

    def brief(self) -> str:
        return f"[{self.published[:10]}] {self.headline} ({self.source})"


# ---------------------------------------------------------------------------
# earnings
# ---------------------------------------------------------------------------


class EarningsCalendar:
    """Bulk-fetched, cached for a day, keyed by bare ticker (no .US suffix)."""

    def __init__(self, cfg: dict, store):
        self.cfg = cfg
        self.store = store
        self.map: dict[str, Earnings] = {}

    async def load(self, horizon_days: int = 45) -> str:
        cached = self.store.get("earnings_cal", max_age_s=self.cfg.get("ttl_s", 43200))
        if cached:
            self.map = {k: Earnings(**v) for k, v in cached.items()}
            return f"cached ({len(self.map)} symbols)"

        key = self.cfg.get("finnhub_key")
        if key:
            try:
                n = await self._finnhub(key, horizon_days)
                self.store.put("earnings_cal", {k: vars(v) for k, v in self.map.items()})
                return f"finnhub ({n} symbols)"
            except (httpx.HTTPError, ValueError, KeyError) as e:
                return f"finnhub failed: {e}"
        return "no provider configured"

    async def _finnhub(self, key: str, horizon: int) -> int:
        today = date.today()
        params = {"from": today.isoformat(),
                  "to": (today + timedelta(days=horizon)).isoformat(), "token": key}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{FINNHUB}/calendar/earnings", params=params)
            r.raise_for_status()
            rows = r.json().get("earningsCalendar", [])
        for row in rows:
            sym = str(row.get("symbol", "")).upper()
            d = row.get("date")
            if not sym or not d:
                continue
            prev = self.map.get(sym)
            if prev and prev.date and prev.date <= d:
                continue                       # keep the nearest print
            self.map[sym] = Earnings(sym, d, row.get("hour"),
                                     (datetime.fromisoformat(d).date() - today).days,
                                     "finnhub")
        return len(self.map)

    def lookup(self, symbol: str) -> Earnings:
        base = _base(symbol)
        hit = self.map.get(base)
        if hit:
            # days_out was computed at fetch time; refresh against today
            try:
                hit.days_out = (datetime.fromisoformat(hit.date).date() - date.today()).days
            except (ValueError, TypeError):
                pass
            return hit
        return Earnings(base, None, None, None, "unknown")

    def yfinance_fallback(self, symbol: str) -> Earnings:
        """Per-symbol, only worth calling for the handful that survived the gates."""
        base = _base(symbol)
        try:
            import yfinance as yf
            cal = yf.Ticker(base).calendar
            dt = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if isinstance(dt, list) and dt:
                dt = dt[0]
            if dt is None:
                return Earnings(base, None, None, None, "unknown")
            d = dt.date() if hasattr(dt, "date") else datetime.fromisoformat(str(dt)).date()
            return Earnings(base, d.isoformat(), None, (d - date.today()).days, "yfinance")
        except Exception:
            return Earnings(base, None, None, None, "unknown")


def earnings_gate(e: Earnings, blackout_days: int, hold_days: int,
                  block_unknown: bool) -> tuple[bool, str]:
    """The deterministic gate. Returns (ok, human-readable reason)."""
    if not e.known:
        if block_unknown:
            return False, f"no earnings date found ({e.source})"
        return True, "no earnings date found — not blocked, verify manually"
    d = e.days_out
    if d is None or d < 0:
        return True, f"last print {e.date}"
    if d <= blackout_days:
        return False, f"reports in {d} days ({e.date}{', ' + e.when if e.when else ''})"
    if d <= hold_days:
        return True, f"reports in {d} days — inside the typical hold, size down"
    return True, f"clear until {e.date} ({d} days)"


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------


class NewsFeed:
    def __init__(self, cfg: dict, mcp=None):
        self.cfg = cfg
        self.mcp = mcp

    async def fetch(self, symbol: str, days: int = 7, limit: int = 8) -> list[Story]:
        if self.cfg.get("prefer") == "ctrader" and self.mcp and "news" in self.mcp.resolved:
            try:
                return _normalize_stories(await self.mcp.call("news", symbol=symbol), limit)
            except RuntimeError:
                pass
        key = self.cfg.get("finnhub_key")
        if not key:
            return []
        today = date.today()
        params = {"symbol": _base(symbol), "from": (today - timedelta(days=days)).isoformat(),
                  "to": today.isoformat(), "token": key}
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(f"{FINNHUB}/company-news", params=params)
                r.raise_for_status()
                return _normalize_stories(r.json(), limit)
        except (httpx.HTTPError, ValueError):
            return []


def _normalize_stories(raw, limit: int) -> list[Story]:
    rows = raw if isinstance(raw, list) else _dig(raw)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        low = {str(k).lower(): v for k, v in r.items()}
        head = low.get("headline") or low.get("title") or low.get("text")
        if not head:
            continue
        ts = low.get("datetime") or low.get("date") or low.get("published") or low.get("time")
        if isinstance(ts, (int, float)):
            ts = datetime.utcfromtimestamp(float(ts)).isoformat()
        out.append(Story(str(head)[:220], str(low.get("source") or low.get("provider") or "—"),
                         str(low.get("url") or ""), str(ts or "")))
    out.sort(key=lambda s: s.published, reverse=True)
    return out[:limit]


def _dig(obj):
    if isinstance(obj, dict):
        for k in ("news", "articles", "data", "result", "items", "stories"):
            if isinstance(obj.get(k), list):
                return obj[k]
        for v in obj.values():
            if isinstance(v, list):
                return v
    return []


def _base(symbol: str) -> str:
    """AAPL.US -> AAPL"""
    s = symbol.upper()
    for suf in (".US", ".NAS", ".NYSE", ".CFD"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


# ---------------------------------------------------------------------------

async def enrich(rows: list[dict], cal: EarningsCalendar, news: NewsFeed,
                 cfg: dict) -> list[dict]:
    """Attach earnings + news to the rows that survived the technical gates.

    Runs only on survivors, so this is a handful of requests, not hundreds.
    """
    blackout = cfg.get("blackout_days", 10)
    hold = cfg.get("typical_hold_days", 20)
    block_unknown = cfg.get("block_unknown", False)
    use_fallback = cfg.get("yfinance_fallback", True)

    async def one(r: dict):
        e = cal.lookup(r["symbol"])
        if not e.known and use_fallback:
            e = await asyncio.to_thread(cal.yfinance_fallback, r["symbol"])
        ok, why = earnings_gate(e, blackout, hold, block_unknown)
        r["earnings"] = {"date": e.date, "days_out": e.days_out, "when": e.when,
                         "source": e.source, "ok": ok, "why": why}
        if not ok:
            r["pass"] = False
            r["failed"] = list(r.get("failed", [])) + ["Earnings"]
        r["gates"] = list(r.get("gates", [])) + [
            {"name": "Earnings", "ok": ok, "detail": why}]
        stories = await news.fetch(r["symbol"], cfg.get("news_days", 7),
                                   cfg.get("news_limit", 8))
        r["news"] = [vars(s) for s in stories]

    await asyncio.gather(*(one(r) for r in rows))
    return rows
