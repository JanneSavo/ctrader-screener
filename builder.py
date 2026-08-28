"""
builder.py — compile a declarative spec into a working strategy.

A spec is JSON: a list of gate rules over a fixed feature vocabulary, plus
entry/stop/target rules. No Python is evaluated, ever. There is no eval(), no
exec() and no lambda coming from the UI — the vocabulary below is the entire
surface area, so a saved recipe cannot do anything a coded strategy could not.

The part that makes this worth building is not the gate composer. It is
`preview()`, which reports:

  - how many symbols pass right now
  - WHICH GATE rejected the most candidates
  - how often the spec fired historically, per symbol per year

The middle one tells you which gate is load-bearing and which is decoration.
The last one is the honest one: a spec that fires twice a year has no sample
size, however good today's list looks.

A builder makes overfitting effortless. Nudging thresholds until today's screen
looks clean is curve-fitting to one cross-section of one day. Frequency and
rejection stats are shown precisely so that is visible while you do it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from strategies.base import Ctx, Gate, Strategy, atr_wilder, close_position, rsi

# ---------------------------------------------------------------------------
# feature vocabulary
# ---------------------------------------------------------------------------


def _ma(d, length=50, **_):
    return d["Close"].rolling(int(length)).mean()


def _ema(d, length=20, **_):
    return d["Close"].ewm(span=int(length), adjust=False).mean()


def _atr(d, length=14, **_):
    return atr_wilder(d, int(length))


def _dist_ma_atr(d, length=50, atr_len=14, **_):
    return (d["Close"] - _ma(d, length)) / _atr(d, atr_len)


def _dist_ma_pct(d, length=50, **_):
    return 100 * (d["Close"] / _ma(d, length) - 1)


def _low_dist_ma_atr(d, length=50, atr_len=14, lookback=10, **_):
    dist = (d["Low"] - _ma(d, length)) / _atr(d, atr_len)
    return dist.rolling(int(lookback)).min()


def _pct_from_high(d, length=40, **_):
    return 100 * (1 - d["Close"] / d["High"].rolling(int(length)).max())


def _pct_from_low(d, length=40, **_):
    return 100 * (d["Close"] / d["Low"].rolling(int(length)).min() - 1)


def _vol_ratio(d, length=20, **_):
    avg = d["Volume"].rolling(int(length)).mean()
    return d["Volume"] / avg.replace(0, np.nan)


def _turnover(d, length=20, **_):
    return d["Close"] * d["Volume"].rolling(int(length)).mean()


def _close_pos(d, **_):
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    return (d["Close"] - d["Low"]) / rng


def _frac_above_ma(d, length=50, window=60, **_):
    return (d["Close"] > _ma(d, length)).rolling(int(window)).mean()


def _ma_slope_pct(d, length=50, window=10, **_):
    m = _ma(d, length)
    return 100 * (m / m.shift(int(window)) - 1)


def _breakout_atr(d, length=40, atr_len=14, **_):
    prior = d["High"].rolling(int(length)).max().shift(1)
    return (d["Close"] - prior) / _atr(d, atr_len)


def _squeeze(d, short=15, long=40, atr_len=14, **_):
    a = _atr(d, atr_len)
    return a.rolling(int(short)).mean() / a.rolling(int(long)).mean()


def _down_days(d, window=4, **_):
    return (d["Close"].diff() < 0).rolling(int(window)).sum()


def _rsi(d, length=14, **_):
    return rsi(d["Close"], int(length))


def _atr_pct(d, length=14, **_):
    return 100 * _atr(d, length) / d["Close"]


def _gap_pct(d, **_):
    return 100 * (d["Open"] / d["Close"].shift(1) - 1)


def _ret_pct(d, window=5, **_):
    return 100 * (d["Close"] / d["Close"].shift(int(window)) - 1)


FEATURES: dict[str, dict[str, Any]] = {
    "rsi":            {"fn": _rsi, "args": {"length": 14}, "label": "RSI", "unit": ""},
    "dist_ma_atr":    {"fn": _dist_ma_atr, "args": {"length": 50, "atr_len": 14},
                       "label": "Distance from MA (ATR)", "unit": "ATR"},
    "dist_ma_pct":    {"fn": _dist_ma_pct, "args": {"length": 50},
                       "label": "Distance from MA (%)", "unit": "%"},
    "low_dist_ma_atr": {"fn": _low_dist_ma_atr, "args": {"length": 50, "atr_len": 14, "lookback": 10},
                        "label": "Nearest low to MA over lookback", "unit": "ATR"},
    "pct_from_high":  {"fn": _pct_from_high, "args": {"length": 40},
                       "label": "Below the N-bar high", "unit": "%"},
    "pct_from_low":   {"fn": _pct_from_low, "args": {"length": 40},
                       "label": "Above the N-bar low", "unit": "%"},
    "frac_above_ma":  {"fn": _frac_above_ma, "args": {"length": 50, "window": 60},
                       "label": "Share of bars above the MA", "unit": "0-1"},
    "ma_slope_pct":   {"fn": _ma_slope_pct, "args": {"length": 50, "window": 10},
                       "label": "MA slope", "unit": "%"},
    "breakout_atr":   {"fn": _breakout_atr, "args": {"length": 40, "atr_len": 14},
                       "label": "Past the N-bar high", "unit": "ATR"},
    "squeeze":        {"fn": _squeeze, "args": {"short": 15, "long": 40, "atr_len": 14},
                       "label": "Volatility squeeze ratio", "unit": "x"},
    "vol_ratio":      {"fn": _vol_ratio, "args": {"length": 20},
                       "label": "Volume vs average", "unit": "x"},
    "turnover":       {"fn": _turnover, "args": {"length": 20},
                       "label": "Average turnover", "unit": "currency"},
    "close_pos":      {"fn": _close_pos, "args": {},
                       "label": "Close within the bar range", "unit": "0-1"},
    "down_days":      {"fn": _down_days, "args": {"window": 4},
                       "label": "Down days in window", "unit": "days"},
    "atr_pct":        {"fn": _atr_pct, "args": {"length": 14},
                       "label": "ATR as % of price", "unit": "%"},
    "gap_pct":        {"fn": _gap_pct, "args": {}, "label": "Opening gap", "unit": "%"},
    "ret_pct":        {"fn": _ret_pct, "args": {"window": 5},
                       "label": "Return over window", "unit": "%"},
}

OPS: dict[str, Callable[[float, Any], bool]] = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">":  lambda v, t: v > t,
    "<":  lambda v, t: v < t,
    "between": lambda v, t: t[0] <= v <= t[1],
}

OP_WORDS = {">=": "at least", "<=": "at most", ">": "above", "<": "below",
            "between": "between"}


def vocabulary() -> dict:
    """What the UI renders as the gate builder."""
    return {
        "features": [{"key": k, "label": v["label"], "unit": v["unit"],
                      "args": v["args"]} for k, v in sorted(FEATURES.items())],
        "ops": [{"key": k, "label": OP_WORDS[k]} for k in OPS],
        "stops": [{"key": "atr", "label": "ATR multiple", "args": {"mult": 2.0}},
                  {"key": "pct", "label": "Percent", "args": {"pct": 5.0}},
                  {"key": "swing_low", "label": "Swing low", "args": {"lookback": 10, "pad_atr": 0.25}}],
        "targets": [{"key": "r", "label": "R multiple", "args": {"mult": 2.0}},
                    {"key": "pct", "label": "Percent", "args": {"pct": 10.0}},
                    {"key": "ma", "label": "Moving average", "args": {"length": 10}}],
        "regimes": [{"key": "bull", "label": "Only when the index is in an uptrend"},
                    {"key": "bear", "label": "Only when the index is not"},
                    {"key": None, "label": "Any regime"}],
    }


# ---------------------------------------------------------------------------
# spec validation
# ---------------------------------------------------------------------------


class SpecError(ValueError):
    pass


def validate(spec: dict) -> dict:
    if "key" not in spec:
        raise SpecError(
            "spec has no 'key' field. The identifier field is called 'key' (not "
            "'name' or 'id'): a short slug like my_dip_buy. 'label' is the "
            "human-readable title and is optional."
            + (f" You supplied: {', '.join(sorted(spec))}." if spec else ""))
    key = str(spec.get("key") or "").strip()
    if not key.replace("_", "").isalnum() or not key:
        raise SpecError(f"key {key!r} must be alphanumeric with underscores, "
                        f"e.g. oversold_above_200")
    gates = spec.get("gates") or []
    if not gates:
        raise SpecError("a strategy needs at least one gate")
    for i, g in enumerate(gates, 1):
        if g.get("feature") not in FEATURES:
            raise SpecError(f"gate {i}: unknown feature {g.get('feature')!r}")
        if g.get("op") not in OPS:
            raise SpecError(f"gate {i}: unknown operator {g.get('op')!r}")
        v = g.get("value")
        if g["op"] == "between":
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise SpecError(f"gate {i}: 'between' needs two values")
        elif not isinstance(v, (int, float)):
            raise SpecError(f"gate {i}: value must be a number")
        for arg in (g.get("args") or {}):
            if arg not in FEATURES[g["feature"]]["args"]:
                raise SpecError(f"gate {i}: {g['feature']} has no argument {arg!r}")
    if (spec.get("stop") or {}).get("kind", "atr") not in ("atr", "pct", "swing_low"):
        raise SpecError("stop.kind must be atr, pct or swing_low")
    if (spec.get("target") or {}).get("kind", "r") not in ("r", "pct", "ma"):
        raise SpecError("target.kind must be r, pct or ma")
    if spec.get("needs_regime") not in ("bull", "bear", None):
        raise SpecError("needs_regime must be bull, bear or null")
    return spec


# ---------------------------------------------------------------------------
# the compiled strategy
# ---------------------------------------------------------------------------


class Composite(Strategy):
    """A saved recipe, behaving exactly like a hand-written strategy."""

    def __init__(self, spec: dict):
        validate(spec)
        self.spec = spec
        self.key = spec["key"]
        self.label = spec.get("label") or spec["key"]
        self.description = spec.get("description", "built in the strategy builder")
        self.direction = spec.get("direction", "long")
        self.needs_regime = spec.get("needs_regime", "bull")
        self.rank_weights = spec.get("rank_weights") or {"rr": 1.0}
        self.p = spec
        self.built = True

    @property
    def min_bars(self) -> int:
        longest = 0
        for g in self.spec["gates"]:
            args = {**FEATURES[g["feature"]]["args"], **(g.get("args") or {})}
            longest = max([longest] + [int(v) for v in args.values()
                                       if isinstance(v, (int, float))])
        return max(120, longest * 2 + 60)

    # -- evaluation --------------------------------------------------------

    def _series(self, d: pd.DataFrame, g: dict) -> pd.Series:
        f = FEATURES[g["feature"]]
        args = {**f["args"], **(g.get("args") or {})}
        return f["fn"](d, **args)

    def check(self, d: pd.DataFrame, i: int = -1) -> list[Gate]:
        gates = []
        for g in self.spec["gates"]:
            s = self._series(d, g)
            val = s.iloc[i]
            f = FEATURES[g["feature"]]
            if pd.isna(val):
                gates.append(Gate(g.get("label") or f["label"], False, "not enough history"))
                continue
            ok = bool(OPS[g["op"]](float(val), g["value"]))
            tgt = (f"{g['value'][0]}–{g['value'][1]}" if g["op"] == "between"
                   else g["value"])
            gates.append(Gate(
                g.get("label") or f["label"], ok,
                f"{float(val):.2f}{f['unit'] and ' ' + f['unit']} "
                f"({OP_WORDS[g['op']]} {tgt})"))
        return gates

    def _levels(self, d: pd.DataFrame, i: int = -1) -> tuple[float, float, float]:
        entry = float(d["Close"].iloc[i])
        atr = float(atr_wilder(d, 14).iloc[i])
        st = self.spec.get("stop") or {"kind": "atr", "mult": 2.0}
        if st.get("kind", "atr") == "atr":
            stop = entry - float(st.get("mult", 2.0)) * atr
        elif st.get("kind") == "pct":
            stop = entry * (1 - float(st.get("pct", 5.0)) / 100)
        else:
            lb = int(st.get("lookback", 10))
            window = d["Low"].iloc[max(0, len(d) + i + 1 - lb):len(d) + i + 1]
            stop = float(window.min()) - float(st.get("pad_atr", 0.25)) * atr
        risk = max(entry - stop, 1e-9)

        tg = self.spec.get("target") or {"kind": "r", "mult": 2.0}
        if tg.get("kind", "r") == "r":
            target = entry + float(tg.get("mult", 2.0)) * risk
        elif tg.get("kind") == "pct":
            target = entry * (1 + float(tg.get("pct", 10.0)) / 100)
        else:
            target = float(d["Close"].rolling(int(tg.get("length", 10))).mean().iloc[i])
        return entry, stop, target

    def evaluate(self, symbol, df, ctx: Ctx):
        if df is None or len(df) < self.min_bars:
            return None
        d = df.copy()
        d["atr"] = atr_wilder(d, 14)
        d["ma"] = d["Close"].rolling(50).mean()
        gates = self.check(d)
        entry, stop, target = self._levels(d)
        if not np.isfinite([entry, stop, target]).all() or entry <= stop:
            return None
        min_rr = float(self.spec.get("min_rr", 0) or 0)
        rr = (target - entry) / max(entry - stop, 1e-9)
        if min_rr:
            gates.append(Gate("Reward", rr >= min_rr, f"{rr:.2f}R against a {min_rr}R floor"))

        extras = {"turnover": float(_turnover(d).iloc[-1] or 0),
                  "vol_ratio": float(_vol_ratio(d).iloc[-1] or 1),
                  "trend_frac": float(_frac_above_ma(d).iloc[-1] or 0)}
        return self.signal(symbol, d, gates, entry=entry, stop=stop, target=target,
                           ctx=ctx, zone={
                               "start": d.index[-10].date().isoformat(),
                               "end": d.index[-1].date().isoformat(),
                               "low": round(float(d["Low"].iloc[-10:].min()), 4),
                               "ma_at_low": round(float(d["ma"].iloc[-1]), 4),
                               "bounce": d.index[-1].date().isoformat(),
                               "swing_high": round(float(d["High"].iloc[-40:].max()), 4),
                           }, extras=extras)


# ---------------------------------------------------------------------------
# preview — the reason the builder is worth having
# ---------------------------------------------------------------------------


def preview(spec: dict, frames: dict[str, pd.DataFrame], ctx: Ctx,
            history_bars: int = 500, sample: int = 40) -> dict:
    """Run a candidate spec over cached bars and report where it stands.

    Returns the pass list, a per-gate rejection count, and how often the spec
    fired historically. The rejection count shows which gate is load-bearing;
    a gate that never rejects anything is decoration, and one that rejects
    everything is the whole strategy wearing six other gates as a disguise.
    """
    strat = Composite(spec)
    rejects: dict[str, int] = {}
    passes, evaluated = [], 0

    for sym, df in frames.items():
        r = strat.evaluate(sym, df, ctx)
        if r is None:
            continue
        evaluated += 1
        for g in r["gates"]:
            if not g["ok"]:
                rejects[g["name"]] = rejects.get(g["name"], 0) + 1
        if r["pass"]:
            passes.append(r)

    # historical firing rate: walk back over a sample of symbols
    fired = bars_walked = 0
    for sym, df in list(frames.items())[:sample]:
        if df is None or len(df) < strat.min_bars + 20:
            continue
        d = df.copy()
        span = min(history_bars, len(d) - strat.min_bars)
        for i in range(-span, 0):
            sub = d.iloc[: len(d) + i + 1]
            if len(sub) < strat.min_bars:
                continue
            bars_walked += 1
            if all(g.ok for g in strat.check(sub)):
                fired += 1

    per_year = (fired / bars_walked * 252) if bars_walked else 0.0
    return {
        "ok": True,
        "evaluated": evaluated,
        "passes": len(passes),
        "hit_rate": round(100 * len(passes) / evaluated, 2) if evaluated else 0.0,
        "rejected_by": sorted(rejects.items(), key=lambda kv: -kv[1]),
        "history": {"bars_walked": bars_walked, "signals": fired,
                    "per_symbol_per_year": round(per_year, 2)},
        "sample": [{k: r[k] for k in ("symbol", "entry", "stop", "target", "rr",
                                      "stop_pct", "units")} for r in passes[:12]],
        "warnings": _warnings(spec, len(passes), evaluated, per_year, rejects),
    }


def _warnings(spec, passes, evaluated, per_year, rejects) -> list[str]:
    out = []
    if evaluated and passes / evaluated > 0.25:
        out.append(f"{100 * passes / evaluated:.0f}% of the universe passes — "
                   f"the gates are barely filtering anything.")
    if evaluated and passes == 0:
        out.append("Nothing passes. Check the rejection counts to see which gate is fatal.")
    if 0 < per_year < 2:
        out.append(f"This fired about {per_year:.1f} times per symbol per year "
                   f"historically. That is too little to ever prove it works.")
    dead = [g.get("label") or FEATURES[g["feature"]]["label"] for g in spec.get("gates", [])
            if (g.get("label") or FEATURES[g["feature"]]["label"]) not in rejects]
    if dead and evaluated:
        out.append(f"Rejected nothing: {', '.join(dead[:4])}. "
                   f"Those gates are not doing any work.")
    return out
