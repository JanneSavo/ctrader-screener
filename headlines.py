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
headlines.py — per-headline classification, deterministic aggregation.

The holistic review asks one big question: "read all this and decide." Every
model tested failed the same way on it — given another company's headlines it
cited them as if they belonged to this symbol. A prompt rule ("check the
subject") helped and did not hold.

So stop asking. Each headline gets its own call, scoped to one company and one
sentence of text, answering three narrow questions. The verdict is then computed
in Python from the answers. A headline the model marks as being about a
different company contributes nothing, structurally — there is no path by which
its content reaches the verdict, because the verdict is arithmetic over labels,
not prose written by a model that saw everything at once.

This is more calls but much shorter ones, and small models are reliable at
narrow classification in a way they are not at holistic judgement.
"""

from __future__ import annotations

import asyncio
import json
import re

CLASSIFY_SYSTEM = """You label a single news headline for a stock screener.

You are given ONE company and ONE headline. Answer three questions about it.

1. about: is this headline primarily about THAT company? Sector round-ups, ETF
   commentary, "top 10 stocks" listicles and articles mainly about a competitor
   are NOT about the company, even if it is mentioned in passing. If the
   headline is about a different company, about=false.

2. catalyst: what kind of event does it describe?
   none        - no specific corporate event (opinion, rating, price move, listicle)
   m&a         - takeover, merger, acquisition, going private
   regulatory  - investigation, probe, lawsuit from a regulator, sanction
   litigation  - private lawsuit, court ruling, settlement
   guidance    - cut or raised outlook, warning, pre-announcement
   insider     - insider or executive share sales or purchases
   management  - CEO/CFO departure, sudden resignation
   earnings    - results released or about to be
   other       - a real corporate event that fits none of the above

3. impact: how much should this make a swing trader hesitate over the next month?
   none   - no reason to hesitate (this is the normal answer)
   minor  - worth knowing, not disqualifying
   major  - a real reason to stand aside

Analyst ratings, price targets, "top pick" mentions and momentum commentary are
always catalyst=none and impact=none. Ordinary volatility is not a catalyst.

Respond with ONLY this JSON, nothing else:
{"about": true|false, "catalyst": "...", "impact": "none|minor|major", "why": "under 12 words"}"""


def _parse_one(txt: str) -> dict:
    s = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S | re.I).strip()
    s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return {"about": False, "catalyst": "none", "impact": "none",
                "why": "unparseable", "error": True}
    cat = str(d.get("catalyst", "none")).lower()[:20]
    imp = str(d.get("impact", "none")).lower()
    return {"about": bool(d.get("about")),
            "catalyst": cat if cat else "none",
            "impact": imp if imp in ("none", "minor", "major") else "none",
            "why": str(d.get("why", ""))[:120]}


async def classify(analyst, symbol: str, company: str, news: list[dict],
                   concurrency: int = 1) -> list[dict]:
    """One call per headline. Each call sees one company and one headline."""
    sem = asyncio.Semaphore(max(1, concurrency))
    out: list[dict] = [None] * len(news)

    async def one(i: int, h: dict):
        prompt = (f"Company: {company} ({symbol})\n"
                  f"Headline: {h.get('headline', '')}\n"
                  f"Source: {h.get('source', '')}")
        async with sem:
            try:
                txt = await analyst._complete(prompt, system=CLASSIFY_SYSTEM)
                lab = _parse_one(txt)
            except Exception as e:
                lab = {"about": False, "catalyst": "none", "impact": "none",
                       "why": str(e)[:80], "error": True}
        lab["headline"] = h.get("headline", "")[:200]
        lab["source"] = h.get("source", "")
        out[i] = lab

    await asyncio.gather(*(one(i, h) for i, h in enumerate(news)))
    return [o for o in out if o]


# impact -> the verdict a single headline justifies on its own
_LADDER = {"major": "avoid", "minor": "caution", "none": "clear"}
_RANK = {"clear": 0, "caution": 1, "avoid": 2}


def aggregate(symbol: str, labels: list[dict], row: dict) -> dict:
    """Compute the verdict in Python. No model sees the whole picture.

    Only headlines the model marked as being ABOUT this company can move the
    verdict, so a foreign headline cannot contribute no matter what it says.
    """
    from llm import severity

    relevant = [l for l in labels if l.get("about") and not l.get("error")]
    ignored = [l for l in labels if not l.get("about") and not l.get("error")]
    errors = [l for l in labels if l.get("error")]

    verdict, drivers = "clear", []
    for l in relevant:
        v = _LADDER.get(l["impact"], "clear")
        if _RANK[v] > _RANK[verdict]:
            verdict = v
        if l["impact"] != "none":
            drivers.append(l)

    cats = [l["catalyst"] for l in drivers if l["catalyst"] != "none"]
    reasons = [f"{l['catalyst']}: {l['why']}" for l in drivers[:3]]
    if not drivers:
        reasons = [f"{len(relevant)} headlines about {symbol}, none decision-relevant"
                   if relevant else
                   f"no headlines about {symbol} ({len(ignored)} were about other companies)"]

    # confidence is evidence count, not a model's self-report
    conf = min(1.0, 0.4 + 0.25 * len(drivers)) if drivers else 0.0

    bear = ""
    if drivers:
        bear = ("Reasons to stand aside, from headlines about this company: "
                + "; ".join(f"\"{l['headline'][:90]}\"" for l in drivers[:2]) + ".")
    elif relevant:
        bear = ("Nothing in this company's own headlines argues against the setup; "
                f"{len(relevant)} were checked.")
    else:
        bear = (f"No company-specific news was found. {len(ignored)} headlines were "
                f"about other companies and were excluded.")

    out = {"verdict": verdict, "confidence": round(conf, 2), "reasons": reasons,
           "catalyst": cats[0] if cats else "none",
           "social_note": "", "technical_note": "",
           "bear_case": bear,
           "headlines": {"checked": len(labels), "about_company": len(relevant),
                         "ignored_other_company": len(ignored), "errors": len(errors)},
           "labels": [{k: l[k] for k in ("headline", "about", "catalyst", "impact", "why")}
                      for l in labels]}
    out["severity"] = severity(out, row)
    return out
