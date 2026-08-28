"""Breakout above an N-day high after a volatility contraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Ctx, Gate, Strategy, atr_wilder, close_position


class RangeBreakout(Strategy):
    key = "breakout"
    label = "Range breakout"
    description = ("Price coils into a tight range, then closes above the "
                   "N-day high on expanding volume.")
    direction = "long"
    needs_regime = "bull"
    defaults = {
        "lookback": 40, "ma_len": 50, "atr_len": 14, "vol_len": 20,
        "squeeze_window": 15, "squeeze_max": 0.65,   # recent ATR vs its own average
        "vol_ratio_min": 1.30, "close_pos_min": 0.60,
        "max_extension_atr": 1.00,                   # how far past the breakout level
        "atr_stop_mult": 2.0, "target_r": 2.5, "min_rr": 2.0,
    }
    rank_weights = {"rr": 0.25, "squeeze": 0.30, "vol_ratio": 0.25, "turnover": 0.20}

    @property
    def min_bars(self) -> int:
        return max(self.p["ma_len"], self.p["lookback"]) + self.p["vol_len"] + 30

    def evaluate(self, symbol, df, ctx: Ctx):
        p = self.p
        if df is None or len(df) < self.min_bars:
            return None
        d = df.copy()
        d["ma"] = d["Close"].rolling(p["ma_len"]).mean()
        d["atr"] = atr_wilder(d, p["atr_len"])
        d["vol_avg"] = d["Volume"].rolling(p["vol_len"]).mean()
        last = d.iloc[-1]
        if not np.isfinite([last["ma"], last["atr"]]).all() or last["atr"] <= 0:
            return None
        atr = float(last["atr"])

        prior = d.iloc[-(p["lookback"] + 1):-1]
        level = float(prior["High"].max())
        broke = bool(last["Close"] > level)

        recent_atr = float(d["atr"].iloc[-p["squeeze_window"]:].mean())
        base_atr = float(d["atr"].iloc[-p["lookback"]:].mean())
        squeeze = recent_atr / base_atr if base_atr else 1.0

        vol_avg = float(last["vol_avg"]) if pd.notna(last["vol_avg"]) else 0.0
        vratio = float(last["Volume"] / vol_avg) if vol_avg else 1.0
        cpos = close_position(last)

        entry = float(last["Close"])
        ext = (entry - level) / atr
        stop = min(level - 0.25 * atr, entry - p["atr_stop_mult"] * atr)
        risk = max(entry - stop, 1e-9)
        target = entry + p["target_r"] * risk
        rr = (target - entry) / risk

        gates = [
            Gate("Above the 50DMA", bool(entry > last["ma"]),
                 f"{(entry / float(last['ma']) - 1) * 100:+.1f}% vs the 50DMA"),
            Gate("Breakout", broke, f"closed {entry:.2f} vs the {p['lookback']}-day high {level:.2f}"),
            Gate("Squeeze", squeeze <= p["squeeze_max"],
                 f"recent ATR is {squeeze:.0%} of its {p['lookback']}-bar average"),
            Gate("Volume", vratio >= p["vol_ratio_min"], f"{vratio:.2f}x average volume"),
            Gate("Closed strong", cpos >= p["close_pos_min"], f"closed at {cpos:.0%} of range"),
            Gate("Not extended", ext <= p["max_extension_atr"],
                 f"{ext:.2f} ATR above the breakout level"),
            Gate("Reward", rr >= p["min_rr"], f"{rr:.2f}R at a {p['target_r']}R target"),
        ]
        return self.signal(
            symbol, d, gates, entry=entry, stop=stop, target=target, ctx=ctx,
            zone={"start": prior.index[0].date().isoformat(),
                  "end": prior.index[-1].date().isoformat(),
                  "low": round(float(prior["Low"].min()), 4), "ma_at_low": round(level, 4),
                  "bounce": d.index[-1].date().isoformat(), "swing_high": round(level, 4)},
            extras={"level": round(level, 4), "squeeze": round(1 - squeeze, 3),
                    "extension": round(ext, 2), "vol_ratio": round(vratio, 2),
                    "trend_frac": round(float((d["Close"].iloc[-60:] > d["ma"].iloc[-60:]).mean()), 3),
                    "turnover": round(float(entry * vol_avg), 0)})
