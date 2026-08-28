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
tape.py — the numbers the technical review is allowed to reason about.

Everything here is computed. The model never reads a chart, never estimates a
level, and never gets a number it did not receive from this module. That split
is the whole design: Python computes, the model argues.

The reason for the split is the Reddit post this came from. The bull case there
is a valuation argument wearing a technical-analysis label — share count times
price, a P/S ratio, a target derived from a multiple. Every number is real and
the conclusion still does not follow. The reply is the better analysis, and it
does not add a single number: it asks why the market prices it that way, and
points out that revenue means nothing without profit.

That is the job. Not "is this oversold" — the gates already answer that,
deterministically, and better than a language model can. The job is the
question the gates cannot ask: given that everything passed, what is the
strongest reason this still fails?

FLAGS below are objective conditions, evaluated in Python. The model weighs
them and writes the case against. It cannot invent one, and it cannot clear one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import atr_wilder


def digest(row: dict, df: pd.DataFrame, universe_atr: list[float] | None = None,
           universe_turnover: list[float] | None = None) -> dict:
    """A compact, factual snapshot of the tape behind one signal."""
    if df is None or len(df) < 60:
        return {}
    d = df.copy()
    d["atr"] = atr_wilder(d, 14)
    last = d.iloc[-1]
    close = float(last["Close"])
    atr = float(last["atr"]) if pd.notna(last["atr"]) else 0.0

    hi_252 = float(d["High"].iloc[-252:].max())
    lo_252 = float(d["Low"].iloc[-252:].min())
    ma200 = float(d["Close"].rolling(200).mean().iloc[-1]) if len(d) >= 200 else float("nan")
    vol20 = float(d["Volume"].iloc[-20:].mean())
    vol60 = float(d["Volume"].iloc[-60:].mean())

    rets = d["Close"].pct_change()
    gaps = (d["Open"] / d["Close"].shift(1) - 1).abs()

    out = {
        "price": round(close, 4),
        "atr_pct": round(100 * atr / close, 2) if close else None,
        "from_52w_high_pct": round(100 * (close / hi_252 - 1), 1),
        "from_52w_low_pct": round(100 * (close / lo_252 - 1), 1),
        "vs_200dma_pct": round(100 * (close / ma200 - 1), 1) if ma200 == ma200 else None,
        "ret_1m_pct": round(100 * (close / float(d["Close"].iloc[-21]) - 1), 1) if len(d) > 21 else None,
        "ret_3m_pct": round(100 * (close / float(d["Close"].iloc[-63]) - 1), 1) if len(d) > 63 else None,
        "vol_trend_pct": round(100 * (vol20 / vol60 - 1), 1) if vol60 else None,
        "turnover": round(close * vol20, 0),
        "down_days_10": int((rets.iloc[-10:] < 0).sum()),
        "gap_days_60": int((gaps.iloc[-60:] > 0.03).sum()),
        "max_1d_drop_60_pct": round(100 * float(rets.iloc[-60:].min()), 1),
        "worst_dd_1y_pct": round(100 * float(
            (d["Close"] / d["Close"].cummax() - 1).iloc[-252:].min()), 1),
    }
    if universe_atr:
        med = float(np.median([x for x in universe_atr if x]))
        out["atr_vs_universe"] = round((out["atr_pct"] or 0) / med, 2) if med else None
    if universe_turnover:
        med = float(np.median([x for x in universe_turnover if x]))
        out["activity_vs_universe"] = (round(out["turnover"] / med, 2)
                                       if med else None)
    return out


# ---------------------------------------------------------------------------
# objective red flags
# ---------------------------------------------------------------------------

def flags(row: dict, tp: dict) -> list[dict]:
    """Conditions that are true or false, not opinions. Each carries its number."""
    out = []

    def add(key, text, weight=1):
        out.append({"key": key, "text": text, "weight": weight})

    if not tp:
        return out

    if (tp.get("from_52w_high_pct") or 0) <= -35:
        add("deep_drawdown",
            f"{abs(tp['from_52w_high_pct']):.0f}% below its 52-week high — the trend "
            f"gates pass on a stock the market has repriced hard", 2)

    if tp.get("vs_200dma_pct") is not None and tp["vs_200dma_pct"] < 0:
        add("below_200",
            f"{abs(tp['vs_200dma_pct']):.1f}% under its 200-day average", 2)

    if (tp.get("atr_vs_universe") or 0) >= 2.0:
        add("high_vol",
            f"ATR is {tp['atr_vs_universe']:.1f}x the universe median — the stop "
            f"distance is wide and the position will be small and noisy", 1)

    if (tp.get("price") or 99) < 5:
        add("low_price",
            f"trades at {tp['price']} — spread and tick size matter at this level", 1)

    # cTrader reports TICK volume, not share volume, so an absolute currency
    # threshold here is meaningless. Compare against the universe instead.
    if (tp.get("activity_vs_universe") or 1.0) < 0.25:
        add("thin",
            f"trading activity is {tp['activity_vs_universe']:.2f}x the universe "
            f"median — quiet enough that fills may be poor", 2)

    if (tp.get("gap_days_60") or 0) >= 6:
        add("gappy",
            f"{tp['gap_days_60']} gaps over 3% in 60 sessions — stops are suggestions "
            f"on a name that moves like this", 1)

    if (tp.get("max_1d_drop_60_pct") or 0) <= -12:
        add("crash_risk",
            f"a single {abs(tp['max_1d_drop_60_pct']):.0f}% down day in the last 60 "
            f"sessions", 1)

    if (tp.get("ret_3m_pct") or 0) <= -25:
        add("falling_knife",
            f"down {abs(tp['ret_3m_pct']):.0f}% over three months — a bounce inside a "
            f"decline is not the same setup as a pullback inside an uptrend", 2)

    if (tp.get("vol_trend_pct") or 0) <= -35:
        add("drying_up",
            f"20-day volume is {abs(tp['vol_trend_pct']):.0f}% below the 60-day average "
            f"— participation is leaving", 1)

    return out


def brief(row: dict, tp: dict, fl: list[dict]) -> str:
    """The technical block handed to the model. Facts only, no framing.

    Anything unknown is OMITTED rather than sent as "unknown" or "None". A model
    handed a missing liquidity figure will invent a liquidity problem from it -
    qwen2.5:14b cautioned on six of eight setups citing "thin volume" that was
    never in the data. Silence is safer than a null.
    """
    if not tp:
        return ""

    def has(*keys) -> bool:
        return all(tp.get(k) is not None for k in keys)

    lines = ["", "Tape (all computed, do not restate anything not listed here):"]
    if has("price"):
        atr = f", ATR {tp['atr_pct']}% of price" if has("atr_pct") else ""
        rel = (f", {tp['atr_vs_universe']}x the universe median ATR"
               if has("atr_vs_universe") else "")
        lines.append(f"- price {tp['price']}{atr}{rel}")
    if has("from_52w_high_pct", "from_52w_low_pct"):
        lines.append(f"- {tp['from_52w_high_pct']}% from the 52-week high, "
                     f"{tp['from_52w_low_pct']}% above the low")
    if has("ret_1m_pct") or has("ret_3m_pct"):
        lines.append(f"- 1-month return {tp.get('ret_1m_pct')}%, "
                     f"3-month {tp.get('ret_3m_pct')}%")
    if has("vs_200dma_pct"):
        lines.append(f"- versus the 200-day average: {tp['vs_200dma_pct']}%")
    if has("worst_dd_1y_pct", "max_1d_drop_60_pct"):
        lines.append(f"- worst drawdown in a year {tp['worst_dd_1y_pct']}%, "
                     f"largest single down day in 60 sessions {tp['max_1d_drop_60_pct']}%")
    if has("gap_days_60"):
        vol = (f", 20-day volume {tp['vol_trend_pct']}% versus the 60-day"
               if has("vol_trend_pct") else "")
        lines.append(f"- {tp['gap_days_60']} gaps over 3% in 60 sessions{vol}")
    if has("activity_vs_universe"):
        lines.append(f"- relative trading activity: {tp['activity_vs_universe']}x the "
                     f"universe median (from tick counts, NOT share volume - never "
                     f"quote it as shares, and do not treat it as a liquidity figure)")

    if fl:
        lines += ["", "Conditions already flagged by the screener:"]
        lines += [f"- {f['text']}" for f in fl]
    return "\n".join(lines)
