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
social.py — retail chatter.

Read this before turning it on. Social sentiment is not a trading signal here
and is deliberately not wired up as one. Retail chatter is thin, adversarial and
trivially manufactured — anyone can pay for a hundred bullish posts. If the
screener started buying names because the internet was excited about them, it
would be a pump detector pointed the wrong way.

Two things chatter is genuinely good for, and this module does only those:

  1. VOLUME, not direction. A z-score spike in mentions means something happened.
     If the news feed shows nothing to explain it, that is a reason to look
     harder before entering, not a reason to buy.
  2. PROMOTION PATTERNS. Coordinated posting, price targets with no reasoning,
     brand-new accounts — the LLM reads for this and flags it.

Providers are pluggable because access keeps moving:
  stocktwits  cashtag stream, lowest friction
  reddit      needs an approved OAuth client. Self-service registration closed
              under the Responsible Builder Policy in late 2025 — new clients go
              through a manual ticket queue that can be silently rejected, so do
              not count on having this working today. 100 QPM once approved.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx


@dataclass
class Post:
    text: str
    author: str
    source: str
    published: str
    score: int = 0

    def brief(self) -> str:
        return f"[{self.source}] {self.text[:200]}"


@dataclass
class Chatter:
    symbol: str
    count: int
    baseline: float
    z: float
    posts: list[Post]
    source: str

    @property
    def spiking(self) -> bool:
        return self.z >= 2.0

    def summary(self) -> str:
        if not self.count:
            return "no chatter found"
        if self.baseline <= 0:
            return f"{self.count} posts (no baseline yet)"
        return (f"{self.count} posts vs {self.baseline:.0f} typical "
                f"({self.z:+.1f} sigma){' — spike' if self.spiking else ''}")


class SocialFeed:
    def __init__(self, cfg: dict, store):
        self.cfg = cfg
        self.store = store

    async def fetch(self, symbol: str, limit: int = 15) -> Chatter:
        base = _base(symbol)
        provider = self.cfg.get("provider", "stocktwits")
        posts: list[Post] = []
        try:
            if provider == "stocktwits":
                posts = await self._stocktwits(base, limit)
            elif provider == "reddit":
                posts = await self._reddit(base, limit)
        except (httpx.HTTPError, ValueError, KeyError):
            posts = []
        z, baseline = self._baseline(base, len(posts))
        return Chatter(base, len(posts), baseline, z, posts[:limit], provider)

    # -- providers ---------------------------------------------------------

    async def _stocktwits(self, sym: str, limit: int) -> list[Post]:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": _UA}) as c:
            r = await c.get(url, params={"limit": min(limit, 30)})
            r.raise_for_status()
            data = r.json()
        out = []
        for m in data.get("messages", [])[:limit]:
            out.append(Post(
                text=str(m.get("body", ""))[:400],
                author=str((m.get("user") or {}).get("username", "—")),
                source="stocktwits",
                published=str(m.get("created_at", "")),
                score=int((m.get("likes") or {}).get("total", 0) or 0),
            ))
        return out

    async def _reddit(self, sym: str, limit: int) -> list[Post]:
        cid, secret = self.cfg.get("reddit_client_id"), self.cfg.get("reddit_secret")
        if not (cid and secret):
            return []
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": _UA}) as c:
            tok = await c.post("https://www.reddit.com/api/v1/access_token",
                               data={"grant_type": "client_credentials"},
                               auth=(cid, secret))
            tok.raise_for_status()
            access = tok.json()["access_token"]
            subs = "+".join(self.cfg.get("subreddits", ["stocks", "investing"]))
            r = await c.get(f"https://oauth.reddit.com/r/{subs}/search",
                            params={"q": sym, "restrict_sr": "true", "sort": "new",
                                    "t": "week", "limit": limit},
                            headers={"Authorization": f"Bearer {access}", "User-Agent": _UA})
            r.raise_for_status()
            data = r.json()
        out = []
        for ch in data.get("data", {}).get("children", [])[:limit]:
            d = ch.get("data", {})
            out.append(Post(
                text=f"{d.get('title','')} {d.get('selftext','')}".strip()[:400],
                author=str(d.get("author", "—")),
                source=f"r/{d.get('subreddit','')}",
                published=datetime.utcfromtimestamp(d.get("created_utc", 0)).isoformat(),
                score=int(d.get("score", 0) or 0),
            ))
        return out

    # -- baseline ----------------------------------------------------------

    def _baseline(self, sym: str, count: int) -> tuple[float, float]:
        """Rolling daily mention counts -> z-score. Needs a few days to mean anything."""
        key = f"chatter:{sym}"
        hist = self.store.get(key) or {}
        today = date.today().isoformat()
        hist[today] = count
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        hist = {k: v for k, v in hist.items() if k >= cutoff}
        self.store.put(key, hist)

        prior = [v for k, v in hist.items() if k != today]
        if len(prior) < 5:
            return 0.0, 0.0
        mean = statistics.fmean(prior)
        # A run of identical counts gives zero variance, which would divide a
        # normal day into a 30-sigma event. Floor the spread at Poisson noise.
        sd = max(statistics.pstdev(prior), mean ** 0.5, 1.0)
        return round((count - mean) / sd, 2), round(mean, 1)


_UA = "ctrader-web-screener/1.0 (personal research)"


def _base(symbol: str) -> str:
    s = symbol.upper()
    for suf in (".US", ".NAS", ".NYSE", ".CFD"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


async def attach(rows: list[dict], feed: SocialFeed, cfg: dict) -> list[dict]:
    """Attach chatter to survivors. Never changes pass/fail on its own."""
    if not cfg.get("enabled"):
        for r in rows:
            r["social"] = None
        return rows

    sem = asyncio.Semaphore(int(cfg.get("concurrency", 3)))

    async def one(r: dict):
        async with sem:
            ch = await feed.fetch(r["symbol"], cfg.get("limit", 15))
        r["social"] = {
            "count": ch.count, "baseline": ch.baseline, "z": ch.z,
            "spiking": ch.spiking, "source": ch.source, "summary": ch.summary(),
            "posts": [vars(p) for p in ch.posts],
        }
        # A spike with nothing in the news is the case worth surfacing.
        if ch.spiking and not (r.get("news") or []):
            r["social"]["unexplained"] = True

    await asyncio.gather(*(one(r) for r in rows))
    return rows
