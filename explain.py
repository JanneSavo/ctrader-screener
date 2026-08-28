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
explain.py — why is this one moving?

Causal attribution on price moves is where analysis usually goes wrong: a move
happens, a headline exists nearby, and the two get welded together. So the work
is split, and the model gets the smaller half.

Python computes, first, the question that comes before "why":

    HOW MUCH OF THIS MOVE IS THE STOCK AT ALL?

If the index fell 2.1% and this fell 2.4%, there is nothing about the company to
explain, and any headline you attach is decoration. The residual - the move left
over after removing beta times the index move - is the only part worth
explaining. It is computed here, and when it is small the verdict is decided in
Python and the model is not consulted.

Only when a move is genuinely idiosyncratic does the model get asked to match it
against a timeline of headlines, and it is required to answer "nothing here
explains it" when nothing does. An unexplained 8% drop is a real and useful
finding: it usually means the news has not surfaced yet.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# what actually happened
# ---------------------------------------------------------------------------


def profile(df: pd.DataFrame, index_df: pd.DataFrame | None,
            lookback: int = 20) -> dict:
    """Measure the move, and split it into market and stock-specific parts."""
    if df is None or len(df) < 60:
        return {}
    d = df.tail(max(lookback + 130, 200)).copy()
    close = d["Close"]

    # the move: peak to trough inside the lookback window
    win = close.tail(lookback)
    peak_i = int(np.argmax(win.values))
    trough_i = int(np.argmin(win.values[peak_i:])) + peak_i
    peak, trough = float(win.iloc[peak_i]), float(win.iloc[trough_i])
    move_pct = 100 * (trough - peak) / peak if peak else 0.0
    start_day = win.index[peak_i].date().isoformat()
    end_day = win.index[trough_i].date().isoformat()
    bars = max(trough_i - peak_i, 0)

    # single-day shocks vs a grind
    rets = close.pct_change().tail(lookback)
    worst_day = float(rets.min() * 100)
    worst_day_on = rets.idxmin().date().isoformat() if len(rets) else None
    gaps = (d["Open"] / close.shift(1) - 1).tail(lookback)
    worst_gap = float(gaps.min() * 100) if len(gaps) else 0.0
    shape = ("one-day shock" if worst_day <= 0.6 * move_pct and move_pct < 0
             else "steady decline" if bars >= 4 else "short slide")

    vol = d["Volume"].tail(lookback).mean() / max(d["Volume"].tail(90).mean(), 1e-9)

    out = {
        "window_days": lookback,
        "move_pct": round(move_pct, 2),
        "from": start_day, "to": end_day, "bars": bars,
        "shape": shape,
        "worst_day_pct": round(worst_day, 2), "worst_day_on": worst_day_on,
        "worst_gap_pct": round(worst_gap, 2),
        "volume_vs_90d": round(float(vol), 2),
    }
    out.update(_vs_market(d, index_df, start_day, end_day))
    return out


def _vs_market(d: pd.DataFrame, index_df: pd.DataFrame | None,
               start_day: str, end_day: str) -> dict:
    """Beta-adjusted residual. The part of the move that is about this company."""
    if index_df is None or len(index_df) < 120:
        return {"market": None}

    a = d["Close"].pct_change().dropna()
    b = index_df["Close"].pct_change().dropna()
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["stock", "index"]
    if len(joined) < 60:
        return {"market": None}

    hist = joined.tail(120)
    var = float(hist["index"].var())
    beta = float(hist.cov().iloc[0, 1] / var) if var > 1e-12 else 1.0

    seg = joined.loc[(joined.index >= start_day) & (joined.index <= end_day)]
    if seg.empty:
        seg = joined.tail(5)
    stock_move = float((1 + seg["stock"]).prod() - 1) * 100
    index_move = float((1 + seg["index"]).prod() - 1) * 100
    explained = beta * index_move
    residual = stock_move - explained

    return {"market": {
        "beta": round(beta, 2),
        "stock_move_pct": round(stock_move, 2),
        "index_move_pct": round(index_move, 2),
        "explained_by_market_pct": round(explained, 2),
        "residual_pct": round(residual, 2),
        "share_market": (round(min(1.0, abs(explained) / abs(stock_move)), 2)
                         if abs(stock_move) > 0.01 else None),
    }}


def peers(profile_d: dict, universe_moves: list[float]) -> dict:
    """Where this move sits against everything else screened today."""
    if not universe_moves:
        return {}
    arr = np.array([m for m in universe_moves if m is not None])
    if arr.size < 10:
        return {}
    mv = profile_d.get("move_pct")
    if mv is None:
        return {}
    return {"universe_median_move_pct": round(float(np.median(arr)), 2),
            "percentile": round(float((arr < mv).mean() * 100), 1),
            "note": "percentile 5 means only 5% of the universe fell more"}


# ---------------------------------------------------------------------------
# the deterministic verdict, taken before any model is consulted
# ---------------------------------------------------------------------------

def pre_verdict(p: dict, pr: dict) -> dict | None:
    """Decide in Python where the numbers already decide it.

    A model asked "why did this fall" will always find a reason. So the cases
    where there is nothing company-specific to explain never reach it.
    """
    if not p:
        return {"verdict": "no_data", "detail": "not enough history"}

    mv = p.get("move_pct") or 0.0
    if mv > -2.0:
        return {"verdict": "no_meaningful_dip",
                "detail": f"largest drawdown in {p['window_days']} sessions is "
                          f"{mv:.1f}% - that is noise, not a dip"}

    mk = p.get("market") or {}
    share = mk.get("share_market")
    resid = mk.get("residual_pct")
    if share is not None and resid is not None:
        if share >= 0.75 and abs(resid) < 2.0:
            return {"verdict": "market_wide",
                    "detail": f"the index moved {mk['index_move_pct']:.1f}% and at "
                              f"beta {mk['beta']} that explains {mk['explained_by_market_pct']:.1f}% "
                              f"of this {mk['stock_move_pct']:.1f}% move. Residual "
                              f"{resid:+.1f}% - nothing company-specific to explain."}
    return None


def timeline(p: dict, news: list[dict], earnings: dict | None) -> list[dict]:
    """Headlines with their position relative to the move, so the model cannot
    silently use an article published after the drop to explain the drop."""
    if not p:
        return []
    start, end = p.get("from"), p.get("to")
    out = []
    for h in news or []:
        day = str(h.get("published", ""))[:10]
        if not day:
            continue
        where = ("before the move" if day < start
                 else "during the move" if day <= end
                 else "after the move - cannot have CAUSED it, but may REPORT on it")
        out.append({"date": day, "headline": h.get("headline", "")[:220],
                    "source": h.get("source", ""), "timing": where})
    out.sort(key=lambda x: x["date"])
    if earnings and earnings.get("date"):
        out.append({"date": earnings["date"], "headline": "(scheduled earnings date)",
                    "source": "calendar",
                    "timing": "upcoming" if earnings.get("days_out", 0) > 0 else "past"})
    return out


PROMPT = """You are explaining a specific price move in one stock.

You are given a measured move, how much of it the market already explains, and a
timeline of headlines with their timing relative to the move.

Your only job is to say which of these it is:

  news_driven        a specific headline dated BEFORE or DURING the move plausibly
                     accounts for it
  earnings_related   the move lines up with an earnings event
  sector_or_market   the move mostly tracks something broader
  unexplained        nothing in the supplied text accounts for it

Rules that matter more than the answer:

- A headline dated AFTER the move cannot have CAUSED it. But it may REPORT on
  it: "Guidance Slashed: Inside the 30% Drop" published two days later tells you
  the cause was guidance, even though the article is not the cause. Use such a
  headline as EVIDENCE of the cause, say explicitly that you are doing so, and
  set "cited_is_retrospective": true. Do not treat a mere price-move recap or a
  post-drop analyst note as evidence of anything - those follow every large move
  and explain nothing.
- Do not explain a move with a headline that is merely nearby in time. Ask
  whether that news would plausibly move a stock this much. An analyst note
  would not cause a 9% drop; a guidance cut would.
- "unexplained" is a good answer and often the correct one. An unexplained drop
  usually means the news has not surfaced yet, which is worth knowing. Do not
  reach for a weak explanation to avoid saying it.
- You have no information beyond what is supplied. No general knowledge about
  this company. If the timeline is empty, the answer is "unexplained".
- Never invent a number. The measurements given are the only numbers.

Respond with ONLY this JSON:
{"verdict":"news_driven|earnings_related|sector_or_market|unexplained",
 "confidence":0.0-1.0,
 "cited_headline":"the exact headline you are relying on, or empty",
 "cited_is_retrospective":true|false,
 "explanation":"two sentences at most",
 "what_to_check":"what the user should verify, or empty"}"""


def _parse(txt: str) -> dict:
    s = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S | re.I).strip()
    s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        d = json.loads(m.group(0)) if m else {}
    v = str(d.get("verdict", "")).lower()
    ok = ("news_driven", "earnings_related", "sector_or_market", "unexplained")
    return {"verdict": v if v in ok else "unexplained",
            "confidence": max(0.0, min(1.0, float(d.get("confidence") or 0))),
            "cited_headline": str(d.get("cited_headline", ""))[:220],
            "cited_is_retrospective": bool(d.get("cited_is_retrospective")),
            "explanation": str(d.get("explanation", ""))[:400],
            "what_to_check": str(d.get("what_to_check", ""))[:220]}


async def explain(analyst, symbol: str, df, index_df, news: list[dict],
                  earnings: dict | None, universe_moves: list[float] | None = None,
                  lookback: int = 20) -> dict:
    p = profile(df, index_df, lookback)
    pv = pre_verdict(p, peers(p, universe_moves or []))
    tl = timeline(p, news, earnings)
    base = {"symbol": symbol, "move": p, "peers": peers(p, universe_moves or []),
            "timeline": tl}

    if pv:
        return base | {"decided_by": "computation", **pv}

    if not tl:
        return base | {"decided_by": "computation", "verdict": "unexplained",
                       "detail": "a company-specific move with no headlines "
                                 "collected for it. The news may not have surfaced."}

    mk = p.get("market") or {}
    brief = (
        f"Stock: {symbol}\n"
        f"Move: {p['move_pct']}% from {p['from']} to {p['to']} ({p['shape']}), "
        f"worst single day {p['worst_day_pct']}% on {p['worst_day_on']}, "
        f"volume {p['volume_vs_90d']}x its 90-day average\n"
        + (f"Market: index moved {mk.get('index_move_pct')}%, beta {mk.get('beta')}, "
           f"so {mk.get('explained_by_market_pct')}% is market. "
           f"Company-specific residual: {mk.get('residual_pct')}%\n" if mk else "")
        + "\nTimeline:\n"
        + "\n".join(f"- [{t['date']}, {t['timing']}] {t['headline']} ({t['source']})"
                    for t in tl))
    txt = await analyst._complete(brief, system=PROMPT)
    out = _parse(txt)

    # a cited headline must exist in the timeline and predate the move's end
    cited = out.get("cited_headline", "")
    if cited:
        match = next((t for t in tl if cited[:60].lower() in t["headline"].lower()), None)
        if not match:
            out["cited_headline"] = ""
            out["integrity"] = "cited headline was not in the timeline - dropped"
            out["verdict"] = "unexplained"
        elif match["timing"].startswith("after"):
            if out.get("cited_is_retrospective"):
                # legitimate: a later article reporting what caused the earlier move
                out["integrity"] = ("cause inferred from reporting published after the "
                                    "move - the article describes the cause, it is not "
                                    "the cause")
            else:
                out["integrity"] = ("cited a post-move headline as the cause without "
                                    "marking it retrospective - rejected")
                out["verdict"] = "unexplained"
    return base | {"decided_by": "model", **out}
