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

"""Oversold snapback inside a long-term uptrend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Ctx, Gate, Strategy, atr_wilder, rsi


class OversoldSnapback(Strategy):
    key = "meanrev"
    label = "Oversold snapback"
    description = ("Above the 200DMA but washed out short term: RSI(2) in the "
                   "basement, several down days, target is the 10-day average.")
    direction = "long"
    needs_regime = None          # deliberately regime-agnostic
    defaults = {
        "sma_slow": 200, "ma_len": 10, "atr_len": 14, "vol_len": 20,
        "rsi_len": 2, "rsi_max": 10.0,
        "down_days_min": 2, "max_drop_atr": 4.0,
        "atr_stop_mult": 2.5, "min_rr": 1.0,
    }
    trigger_gates = frozenset({"Oversold", "Down streak"})
    rank_weights = {"rr": 0.30, "oversold": 0.40, "turnover": 0.30}

    @property
    def min_bars(self) -> int:
        return self.p["sma_slow"] + 40

    def evaluate(self, symbol, df, ctx: Ctx):
        p = self.p
        if df is None or len(df) < self.min_bars:
            return None
        d = df.copy()
        d["ma"] = d["Close"].rolling(p["ma_len"]).mean()
        d["ma_slow"] = d["Close"].rolling(p["sma_slow"]).mean()
        d["atr"] = atr_wilder(d, p["atr_len"])
        d["rsi"] = rsi(d["Close"], p["rsi_len"])
        d["vol_avg"] = d["Volume"].rolling(p["vol_len"]).mean()
        last = d.iloc[-1]
        if not np.isfinite([last["ma"], last["ma_slow"], last["atr"]]).all():
            return None
        atr = float(last["atr"])
        if atr <= 0 or pd.isna(last["rsi"]):
            return None

        entry = float(last["Close"])
        r = float(last["rsi"])
        downs = int((d["Close"].diff().iloc[-4:] < 0).sum())
        drop_atr = float((float(d["High"].iloc[-10:].max()) - entry) / atr)

        target = float(last["ma"])
        stop = entry - p["atr_stop_mult"] * atr
        rr = (target - entry) / max(entry - stop, 1e-9)

        gates = [
            Gate("Long-term uptrend", bool(entry > last["ma_slow"]),
                 f"{(entry / float(last['ma_slow']) - 1) * 100:+.1f}% vs the 200DMA"),
            Gate("Oversold", r <= p["rsi_max"], f"RSI({p['rsi_len']}) at {r:.1f}"),
            Gate("Down streak", downs >= p["down_days_min"], f"{downs} of the last 4 days lower"),
            Gate("Not a collapse", drop_atr <= p["max_drop_atr"],
                 f"{drop_atr:.1f} ATR below the 10-day high"),
            Gate("Target above entry", target > entry,
                 f"10-day average at {target:.2f} vs {entry:.2f}"),
            Gate("Reward", rr >= p["min_rr"], f"{rr:.2f}R to the 10-day average"),
        ]
        vol_avg = float(last["vol_avg"]) if pd.notna(last["vol_avg"]) else 0.0
        return self.signal(
            symbol, d, gates, entry=entry, stop=stop, target=target, ctx=ctx,
            zone={"start": d.index[-5].date().isoformat(),
                  "end": d.index[-1].date().isoformat(),
                  "low": round(float(d["Low"].iloc[-5:].min()), 4),
                  "ma_at_low": round(target, 4),
                  "bounce": d.index[-1].date().isoformat(),
                  "swing_high": round(float(d["High"].iloc[-10:].max()), 4)},
            extras={"rsi": round(r, 1), "oversold": round(max(0.0, 100 - r) / 100, 3),
                    "down_days": downs, "drop_atr": round(drop_atr, 2),
                    "trend_frac": round(float((d["Close"].iloc[-60:] > d["ma_slow"].iloc[-60:]).mean()), 3),
                    "vol_ratio": 1.0,
                    "turnover": round(float(entry * vol_avg), 0)})
