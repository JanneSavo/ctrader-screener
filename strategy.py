"""
strategy.py — pullback-into-the-50DMA continuation setup.

Every gate always reports pass/fail plus the measured value, because the useful
question at 9am is not "is there a signal" but "which gate killed it and by how
much". The UI reads these directly.

Deviation from the source writeup, on purpose: the 5% stop is kept only as a
reference number. Sizing runs off ATR, because 5% is ~4 ATR on a sleepy name
and under 1 ATR on a volatile one — same number, completely different trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    sma_fast: int = 50
    sma_slow: int = 200
    atr_len: int = 14
    vol_len: int = 20

    trend_window: int = 60
    trend_min_frac: float = 0.70

    pullback_lookback: int = 10
    touch_atr: float = 0.60
    break_atr: float = 1.00
    depth_min: float = 0.025
    depth_max: float = 0.150
    swing_high_window: int = 40

    close_pos_min: float = 0.55
    vol_ratio_min: float = 0.90
    max_extension_atr: float = 1.25

    fixed_stop_pct: float = 0.05
    atr_stop_mult: float = 1.75
    target_pct: float = 0.10
    min_rr: float = 1.8

    risk_per_trade: float = 0.01

    weights: dict = field(default_factory=lambda: {
        "trend_frac": 0.20, "rr": 0.30, "tightness": 0.25,
        "vol_ratio": 0.15, "turnover": 0.10,
    })

    @property
    def min_bars(self) -> int:
        return self.sma_slow + self.trend_window + 5


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    d["sma_f"] = d["Close"].rolling(p.sma_fast).mean()
    d["sma_s"] = d["Close"].rolling(p.sma_slow).mean()
    d["atr"] = atr_wilder(d, p.atr_len)
    d["vol_avg"] = d["Volume"].rolling(p.vol_len).mean()
    return d


def regime(df: pd.DataFrame, p: Params) -> dict:
    """Step 1 of the strategy: the index itself has to be in an uptrend."""
    if df is None or len(df) < p.sma_slow + 2:
        return {"ok": False, "note": "not enough index history", "bars": len(df or [])}
    d = indicators(df, p)
    last = d.iloc[-1]
    above_f = bool(last["Close"] > last["sma_f"])
    stacked = bool(last["sma_f"] > last["sma_s"])
    return {
        "ok": above_f and stacked,
        "above_fast": above_f,
        "stacked": stacked,
        "close": round(float(last["Close"]), 2),
        "sma_fast": round(float(last["sma_f"]), 2),
        "sma_slow": round(float(last["sma_s"]), 2),
        "asof": d.index[-1].date().isoformat(),
        "note": ("index above its 50 and 50 over 200" if above_f and stacked
                 else "index below its 50DMA" if not above_f
                 else "50DMA still under the 200"),
    }


def evaluate(symbol: str, df: pd.DataFrame, p: Params, equity: float,
             provisional: bool = False) -> dict | None:
    """Full gate evaluation. Returns None only when there is not enough data.

    provisional=True means the last bar is still open. The gates are identical;
    the result is flagged so the UI can show it as a heads-up rather than a
    signal, because an open bar repaints.
    """
    if df is None or len(df) < p.min_bars:
        return None
    d = indicators(df, p)
    last, prev = d.iloc[-1], d.iloc[-2]
    if not np.isfinite([last["sma_f"], last["sma_s"], last["atr"]]).all():
        return None
    atr = float(last["atr"])
    if atr <= 0:
        return None

    # --- trend -----------------------------------------------------------
    win = d.iloc[-p.trend_window:]
    trend_frac = float((win["Close"] > win["sma_f"]).mean())
    trend_ok = bool(last["Close"] > last["sma_s"] and last["sma_f"] > last["sma_s"]
                    and trend_frac >= p.trend_min_frac)

    # --- pullback --------------------------------------------------------
    look = d.iloc[-p.pullback_lookback:]
    nearest = float(((look["Low"] - look["sma_f"]) / look["atr"]).min())
    worst_close = float(((look["Close"] - look["sma_f"]) / look["atr"]).min())
    swing_high = float(d["High"].iloc[-p.swing_high_window:].max())
    swing_low = float(look["Low"].min())
    depth = (swing_high - swing_low) / swing_high

    touch_ok = nearest <= p.touch_atr
    hold_ok = worst_close >= -p.break_atr
    depth_ok = p.depth_min <= depth <= p.depth_max

    # --- bounce candle ---------------------------------------------------
    rng = float(last["High"] - last["Low"])
    close_pos = 0.5 if rng <= 0 else float((last["Close"] - last["Low"]) / rng)
    vol_avg = float(last["vol_avg"]) if pd.notna(last["vol_avg"]) and last["vol_avg"] else 0.0
    vol_ratio = float(last["Volume"] / vol_avg) if vol_avg else 1.0
    bounce_ok = bool(
        last["Close"] > last["Open"] and last["Close"] > prev["Close"]
        and last["Close"] > last["sma_f"] and close_pos >= p.close_pos_min
        and vol_ratio >= p.vol_ratio_min)

    extension = float((last["Close"] - last["sma_f"]) / atr)
    ext_ok = extension <= p.max_extension_atr

    # --- risk ------------------------------------------------------------
    entry = float(last["Close"])
    stop = min(swing_low - 0.25 * atr, entry - p.atr_stop_mult * atr)
    stop_fixed = entry * (1 - p.fixed_stop_pct)
    target = entry * (1 + p.target_pct)
    risk_ps = max(entry - stop, 1e-9)
    rr = (target - entry) / risk_ps
    rr_ok = rr >= p.min_rr

    gates = [
        ("Trend", trend_ok, f"{trend_frac:.0%} of {p.trend_window} bars above the 50DMA"),
        ("Pullback", touch_ok, f"low came within {nearest:.2f} ATR of the 50DMA"),
        ("Held the line", hold_ok, f"deepest close {worst_close:.2f} ATR from the 50DMA"),
        ("Depth", depth_ok, f"{depth:.1%} off the {p.swing_high_window}-bar high"),
        ("Bounce", bounce_ok, f"closed at {close_pos:.0%} of range on {vol_ratio:.2f}x volume"),
        ("Not extended", ext_ok, f"{extension:.2f} ATR above the 50DMA"),
        ("Reward", rr_ok, f"{rr:.2f}R to a {p.target_pct:.0%} target"),
    ]
    failed = [g[0] for g in gates if not g[1]]

    risk_amt = equity * p.risk_per_trade
    units = risk_amt / risk_ps

    look_idx = look.index
    zone = {
        "start": look_idx[0].date().isoformat(),
        "end": look_idx[-1].date().isoformat(),
        "low": round(swing_low, 4),
        "ma_at_low": round(float(look["sma_f"].min()), 4),
        "bounce": d.index[-1].date().isoformat(),
        "swing_high": round(swing_high, 4),
    }
    spark = _spark(d)

    return {
        "symbol": symbol,
        "asof": d.index[-1].date().isoformat(),
        "provisional": bool(provisional),
        "pass": not failed,
        "failed": failed,
        "gates": [{"name": n, "ok": bool(o), "detail": t} for n, o, t in gates],
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "stop_fixed5": round(stop_fixed, 4),
        "target": round(target, 4),
        "stop_pct": round(100 * risk_ps / entry, 2),
        "rr": round(rr, 2),
        "units": round(units, 2),
        "risk_amt": round(risk_amt, 2),
        "trend_frac": round(trend_frac, 3),
        "depth": round(depth, 4),
        "extension": round(extension, 2),
        "tightness": round(1 / (1 + abs(extension)), 3),
        "vol_ratio": round(vol_ratio, 2),
        "turnover": round(float(entry * vol_avg), 0),
        "atr_pct": round(100 * atr / entry, 2),
        "spark": spark,
        "zone": zone,
    }


def _spark(d: pd.DataFrame, n: int = 70) -> dict:
    tail = d.iloc[-n:]
    return {
        "c": [round(float(x), 4) for x in tail["Close"]],
        "m": [None if pd.isna(x) else round(float(x), 4) for x in tail["sma_f"]],
        "lo": int(np.argmin(tail["Low"].values[-10:]) + len(tail) - 10),
    }


def rank(rows: list[dict], p: Params) -> list[dict]:
    """Percentile-rank each factor across today's candidates, then weight."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    total = pd.Series(0.0, index=df.index)
    for col, w in p.weights.items():
        if col in df:
            total += df[col].rank(pct=True) * w
    df["score"] = (100 * total / sum(p.weights.values())).round(1)
    df = df.sort_values("score", ascending=False)
    out = []
    for r, row in enumerate(df.to_dict("records"), 1):
        row["rank"] = r
        out.append(row)
    return out
