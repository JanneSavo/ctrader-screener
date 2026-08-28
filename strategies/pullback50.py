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

"""Pullback into the 50DMA, in an established uptrend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Ctx, Gate, Strategy, atr_wilder, close_position


class Pullback50(Strategy):
    key = "pullback50"
    label = "50DMA pullback"
    description = ("Uptrending name dips into its 50-day average and closes "
                   "back above it.")
    direction = "long"
    needs_regime = "bull"
    defaults = {
        "sma_fast": 50, "sma_slow": 200, "atr_len": 14, "vol_len": 20,
        "trend_window": 60, "trend_min_frac": 0.70,
        "pullback_lookback": 10, "touch_atr": 0.60, "break_atr": 1.00,
        "depth_min": 0.025, "depth_max": 0.150, "swing_high_window": 40,
        "close_pos_min": 0.55, "vol_ratio_min": 0.90, "max_extension_atr": 1.25,
        "atr_stop_mult": 1.75, "target_pct": 0.10, "min_rr": 1.8,
        "fixed_stop_pct": 0.05,
    }
    rank_weights = {"trend_frac": 0.20, "rr": 0.30, "tightness": 0.25,
                    "vol_ratio": 0.15, "turnover": 0.10}

    @property
    def min_bars(self) -> int:
        return self.p["sma_slow"] + self.p["trend_window"] + 5

    def evaluate(self, symbol, df, ctx: Ctx):
        p = self.p
        if df is None or len(df) < self.min_bars:
            return None
        d = df.copy()
        d["ma"] = d["Close"].rolling(p["sma_fast"]).mean()
        d["ma_slow"] = d["Close"].rolling(p["sma_slow"]).mean()
        d["atr"] = atr_wilder(d, p["atr_len"])
        d["vol_avg"] = d["Volume"].rolling(p["vol_len"]).mean()
        last, prev = d.iloc[-1], d.iloc[-2]
        if not np.isfinite([last["ma"], last["ma_slow"], last["atr"]]).all():
            return None
        atr = float(last["atr"])
        if atr <= 0:
            return None

        win = d.iloc[-p["trend_window"]:]
        trend_frac = float((win["Close"] > win["ma"]).mean())
        trend_ok = bool(last["Close"] > last["ma_slow"] and last["ma"] > last["ma_slow"]
                        and trend_frac >= p["trend_min_frac"])

        look = d.iloc[-p["pullback_lookback"]:]
        nearest = float(((look["Low"] - look["ma"]) / look["atr"]).min())
        worst = float(((look["Close"] - look["ma"]) / look["atr"]).min())
        swing_high = float(d["High"].iloc[-p["swing_high_window"]:].max())
        swing_low = float(look["Low"].min())
        depth = (swing_high - swing_low) / swing_high

        cpos = close_position(last)
        vol_avg = float(last["vol_avg"]) if pd.notna(last["vol_avg"]) else 0.0
        vratio = float(last["Volume"] / vol_avg) if vol_avg else 1.0
        bounce = bool(last["Close"] > last["Open"] and last["Close"] > prev["Close"]
                      and last["Close"] > last["ma"] and cpos >= p["close_pos_min"]
                      and vratio >= p["vol_ratio_min"])
        ext = float((last["Close"] - last["ma"]) / atr)

        entry = float(last["Close"])
        stop = min(swing_low - 0.25 * atr, entry - p["atr_stop_mult"] * atr)
        target = entry * (1 + p["target_pct"])
        rr = (target - entry) / max(entry - stop, 1e-9)

        gates = [
            Gate("Trend", trend_ok, f"{trend_frac:.0%} of {p['trend_window']} bars above the 50DMA"),
            Gate("Pullback", nearest <= p["touch_atr"], f"low came within {nearest:.2f} ATR of the 50DMA"),
            Gate("Held the line", worst >= -p["break_atr"], f"deepest close {worst:.2f} ATR from the 50DMA"),
            Gate("Depth", p["depth_min"] <= depth <= p["depth_max"],
                 f"{depth:.1%} off the {p['swing_high_window']}-bar high"),
            Gate("Bounce", bounce, f"closed at {cpos:.0%} of range on {vratio:.2f}x volume"),
            Gate("Not extended", ext <= p["max_extension_atr"], f"{ext:.2f} ATR above the 50DMA"),
            Gate("Reward", rr >= p["min_rr"], f"{rr:.2f}R to a {p['target_pct']:.0%} target"),
        ]
        return self.signal(
            symbol, d, gates, entry=entry, stop=stop, target=target, ctx=ctx,
            zone={"start": look.index[0].date().isoformat(),
                  "end": look.index[-1].date().isoformat(),
                  "low": round(swing_low, 4), "ma_at_low": round(float(look["ma"].min()), 4),
                  "bounce": d.index[-1].date().isoformat(),
                  "swing_high": round(swing_high, 4)},
            extras={"trend_frac": round(trend_frac, 3), "depth": round(depth, 4),
                    "extension": round(ext, 2), "tightness": round(1 / (1 + abs(ext)), 3),
                    "vol_ratio": round(vratio, 2),
                    "turnover": round(float(entry * vol_avg), 0),
                    "stop_fixed5": round(entry * (1 - p["fixed_stop_pct"]), 4)})
