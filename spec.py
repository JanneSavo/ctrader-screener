"""
spec.py — strategies as data.

A spec is JSON. It is interpreted, never exec'd, so nothing the builder UI can
produce is arbitrary code. Terms resolve to pandas Series over the bar frame;
conditions compare them; the shared risk block does the rest.

    {"key":"my_setup","label":"My setup","direction":"long","needs_regime":"bull",
     "gates":[
       {"label":"Trend","left":{"fn":"close"},"op":">","right":{"fn":"ma","n":200},
        "window":{"mode":"frac","n":60,"min_frac":0.7}}
     ],
     "entry":{"fn":"close"},
     "stop":{"fn":"sub","a":{"fn":"close"},
             "b":{"fn":"mul","a":{"fn":"atr","n":14},"b":{"const":1.75}}},
     "target":{"fn":"mul","a":{"fn":"close"},"b":{"const":1.10}}}

A saved spec loads as a real strategy alongside the hand-written plugins and is
indistinguishable downstream — same gates, same sizing, same plotting.

A WORD ON WHAT THIS IS NOT. The preview tells you how many symbols pass and
which gate is the bottleneck. It says nothing about whether the setup makes
money. Nudging thresholds until the preview looks good is curve-fitting with a
mouse, and it is faster and more comfortable than doing it by hand, which makes
it more dangerous, not less. A spec that survives preview is a hypothesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Ctx, Gate, Strategy, atr_wilder, rsi

# ---------------------------------------------------------------------------
# vocabulary — every term the builder can offer, and nothing else
# ---------------------------------------------------------------------------

VOCAB = {
    "close":     {"label": "Close", "args": []},
    "open":      {"label": "Open", "args": []},
    "high":      {"label": "High", "args": []},
    "low":       {"label": "Low", "args": []},
    "volume":    {"label": "Volume", "args": []},
    "ma":        {"label": "Moving average", "args": ["n"], "defaults": {"n": 50}},
    "ema":       {"label": "Exponential MA", "args": ["n"], "defaults": {"n": 21}},
    "atr":       {"label": "ATR", "args": ["n"], "defaults": {"n": 14}},
    "rsi":       {"label": "RSI", "args": ["n"], "defaults": {"n": 14}},
    "vol_avg":   {"label": "Average volume", "args": ["n"], "defaults": {"n": 20}},
    "high_n":    {"label": "Highest high of N bars", "args": ["n"], "defaults": {"n": 40}},
    "low_n":     {"label": "Lowest low of N bars", "args": ["n"], "defaults": {"n": 40}},
    "close_pos": {"label": "Close position in range (0-1)", "args": []},
    "prev":      {"label": "Previous bar's value of", "args": ["a"]},
    "turnover":  {"label": "Price x average volume", "args": ["n"], "defaults": {"n": 20}},
    "ratio":     {"label": "A / B", "args": ["a", "b"]},
    "atr_dist":  {"label": "(A - B) in ATRs", "args": ["a", "b"], "defaults": {"n": 14}},
    "pct_diff":  {"label": "(A - B) / B as %", "args": ["a", "b"]},
    "add":       {"label": "A + B", "args": ["a", "b"]},
    "sub":       {"label": "A - B", "args": ["a", "b"]},
    "mul":       {"label": "A x B", "args": ["a", "b"]},
    "div":       {"label": "A / B", "args": ["a", "b"]},
}

OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "cross_above": None,   # handled separately, needs the previous bar
    "cross_below": None,
}

WINDOWS = {
    "now": "the latest bar",
    "any": "at least once in the last N bars",
    "all": "on every one of the last N bars",
    "frac": "on at least X% of the last N bars",
}

MAX_LOOKBACK = 500


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------------------
# term evaluation
# ---------------------------------------------------------------------------


def term(d: pd.DataFrame, t: dict) -> pd.Series:
    """Resolve a term to a Series aligned with d. Recursive, bounded, total."""
    if not isinstance(t, dict):
        raise SpecError(f"term must be an object, got {type(t).__name__}")
    if "const" in t:
        return pd.Series(float(t["const"]), index=d.index)
    fn = t.get("fn")
    if fn not in VOCAB:
        raise SpecError(f"unknown term {fn!r}. Known: {', '.join(sorted(VOCAB))}")

    def n_of(default=14) -> int:
        v = int(t.get("n", default))
        if not 1 <= v <= MAX_LOOKBACK:
            raise SpecError(f"{fn}: n must be 1..{MAX_LOOKBACK}, got {v}")
        return v

    if fn == "close":
        return d["Close"]
    if fn == "open":
        return d["Open"]
    if fn == "high":
        return d["High"]
    if fn == "low":
        return d["Low"]
    if fn == "volume":
        return d["Volume"]
    if fn == "ma":
        return d["Close"].rolling(n_of(50)).mean()
    if fn == "ema":
        return d["Close"].ewm(span=n_of(21), adjust=False).mean()
    if fn == "atr":
        return atr_wilder(d, n_of(14))
    if fn == "rsi":
        return rsi(d["Close"], n_of(14))
    if fn == "vol_avg":
        return d["Volume"].rolling(n_of(20)).mean()
    if fn == "high_n":
        return d["High"].rolling(n_of(40)).max()
    if fn == "low_n":
        return d["Low"].rolling(n_of(40)).min()
    if fn == "close_pos":
        rng = (d["High"] - d["Low"]).replace(0, np.nan)
        return ((d["Close"] - d["Low"]) / rng).fillna(0.5)
    if fn == "turnover":
        return d["Close"] * d["Volume"].rolling(n_of(20)).mean()
    if fn == "prev":
        return term(d, t["a"]).shift(1)

    a, b = term(d, t["a"]), term(d, t["b"])
    if fn == "add":
        return a + b
    if fn == "sub":
        return a - b
    if fn == "mul":
        return a * b
    if fn in ("div", "ratio"):
        return a / b.replace(0, np.nan)
    if fn == "pct_diff":
        return 100 * (a - b) / b.replace(0, np.nan)
    if fn == "atr_dist":
        return (a - b) / atr_wilder(d, int(t.get("n", 14))).replace(0, np.nan)
    raise SpecError(f"term {fn!r} is in the vocabulary but not implemented")


def describe(t: dict) -> str:
    """Human-readable term, for gate detail lines and the exported source."""
    if not isinstance(t, dict):
        return str(t)
    if "const" in t:
        return f"{t['const']:g}"
    fn = t.get("fn", "?")
    if fn in ("close", "open", "high", "low", "volume", "close_pos"):
        return fn
    if fn in ("ma", "ema", "atr", "rsi", "vol_avg", "high_n", "low_n", "turnover"):
        return f"{fn}({t.get('n', '')})"
    if fn == "prev":
        return f"prev({describe(t.get('a', {}))})"
    sym = {"add": "+", "sub": "-", "mul": "x", "div": "/", "ratio": "/",
           "pct_diff": "%diff", "atr_dist": "ATRs from"}.get(fn, fn)
    return f"({describe(t.get('a', {}))} {sym} {describe(t.get('b', {}))})"


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------


def gate(d: pd.DataFrame, g: dict) -> Gate:
    left, right = term(d, g["left"]), term(d, g["right"])
    op = g.get("op", ">")
    label = g.get("label") or f"{describe(g['left'])} {op} {describe(g['right'])}"

    if op in ("cross_above", "cross_below"):
        prev_l, prev_r = left.shift(1), right.shift(1)
        series = ((prev_l <= prev_r) & (left > right)) if op == "cross_above" \
            else ((prev_l >= prev_r) & (left < right))
    elif op in OPS and OPS[op]:
        series = OPS[op](left, right)
    else:
        raise SpecError(f"unknown operator {op!r}. Known: {', '.join(OPS)}")

    w = g.get("window") or {"mode": "now"}
    mode = w.get("mode", "now")
    n = int(w.get("n", 1))
    if not 1 <= n <= MAX_LOOKBACK:
        raise SpecError(f"{label}: window n must be 1..{MAX_LOOKBACK}")

    lv, rv = float(left.iloc[-1]), float(right.iloc[-1])
    vals = f"{lv:.4g} vs {rv:.4g}"

    if mode == "now":
        ok = bool(series.iloc[-1])
        detail = f"{describe(g['left'])} {op} {describe(g['right'])} — {vals}"
    elif mode == "any":
        ok = bool(series.iloc[-n:].any())
        detail = f"{'held' if ok else 'never held'} in the last {n} bars — now {vals}"
    elif mode == "all":
        ok = bool(series.iloc[-n:].all())
        detail = f"{'held' if ok else 'broke'} on all of the last {n} bars — now {vals}"
    elif mode == "frac":
        frac = float(series.iloc[-n:].mean())
        need = float(w.get("min_frac", 0.7))
        ok = frac >= need
        detail = f"{frac:.0%} of the last {n} bars (needs {need:.0%})"
    else:
        raise SpecError(f"unknown window mode {mode!r}. Known: {', '.join(WINDOWS)}")

    return Gate(label, ok, detail)


def validate(spec: dict) -> list[str]:
    """Everything wrong with a spec, all at once — not just the first problem."""
    errs: list[str] = []
    if not spec.get("key", "").strip():
        errs.append("key is required")
    elif not spec["key"].replace("_", "").isalnum():
        errs.append("key must be letters, digits and underscores only")
    if not spec.get("gates"):
        errs.append("at least one gate is required")
    if spec.get("direction", "long") not in ("long", "short"):
        errs.append("direction must be long or short")
    if spec.get("needs_regime") not in (None, "bull", "bear"):
        errs.append("needs_regime must be bull, bear or null")

    probe = _probe_frame()
    for i, g in enumerate(spec.get("gates") or [], 1):
        try:
            gate(probe, g)
        except (SpecError, KeyError, TypeError, ValueError) as e:
            errs.append(f"gate {i} ({g.get('label', 'unnamed')}): {e}")
    for field in ("entry", "stop", "target"):
        if field not in spec:
            errs.append(f"{field} is required")
            continue
        try:
            term(probe, spec[field])
        except (SpecError, KeyError, TypeError, ValueError) as e:
            errs.append(f"{field}: {e}")
    return errs


def _probe_frame(n: int = 320) -> pd.DataFrame:
    """A synthetic frame just for validation, so a bad spec is caught before a scan."""
    rng = np.random.default_rng(0)
    c = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    d = pd.DataFrame({"Close": c})
    d["Open"] = d["Close"].shift(1).fillna(c[0])
    d["High"] = np.maximum(d.Open, d.Close) * 1.005
    d["Low"] = np.minimum(d.Open, d.Close) * 0.995
    d["Volume"] = rng.integers(1_000_000, 2_000_000, n).astype(float)
    d.index = pd.bdate_range("2024-01-01", periods=n)
    return d


# ---------------------------------------------------------------------------
# the interpreted strategy
# ---------------------------------------------------------------------------


class SpecStrategy(Strategy):
    """Wraps a spec so it behaves exactly like a hand-written plugin."""

    def __init__(self, spec: dict, params: dict | None = None):
        errs = validate(spec)
        if errs:
            raise SpecError("; ".join(errs))
        self.spec = spec
        self.key = spec["key"]
        self.label = spec.get("label") or spec["key"]
        self.description = spec.get("description", "")
        self.direction = spec.get("direction", "long")
        self.needs_regime = spec.get("needs_regime", "bull")
        self.rank_weights = spec.get("rank_weights") or {"rr": 1.0}
        self.source = "spec"
        super().__init__(params)

    @property
    def min_bars(self) -> int:
        return int(self.spec.get("min_bars") or _needed(self.spec))

    def evaluate(self, symbol, df, ctx: Ctx):
        if df is None or len(df) < self.min_bars:
            return None
        d = df.copy()
        d["atr"] = atr_wilder(d, int(self.spec.get("atr_len", 14)))
        d["ma"] = d["Close"].rolling(int(self.spec.get("display_ma", 50))).mean()
        try:
            gates = [gate(d, g) for g in self.spec["gates"]]
            entry = float(term(d, self.spec["entry"]).iloc[-1])
            stop = float(term(d, self.spec["stop"]).iloc[-1])
            target = float(term(d, self.spec["target"]).iloc[-1])
        except (SpecError, KeyError, IndexError):
            return None
        if not all(np.isfinite([entry, stop, target])) or entry <= 0:
            return None

        min_rr = float(self.spec.get("min_rr", 0) or 0)
        if min_rr:
            rr = abs(target - entry) / max(abs(entry - stop), 1e-9)
            gates.append(Gate("Reward", rr >= min_rr, f"{rr:.2f}R (needs {min_rr}R)"))

        look = d.iloc[-10:]
        return self.signal(
            symbol, d, gates, entry=entry, stop=stop, target=target, ctx=ctx,
            zone={"start": look.index[0].date().isoformat(),
                  "end": look.index[-1].date().isoformat(),
                  "low": round(float(look["Low"].min()), 4),
                  "ma_at_low": round(float(look["ma"].min()), 4)
                  if look["ma"].notna().any() else round(entry, 4),
                  "bounce": d.index[-1].date().isoformat(),
                  "swing_high": round(float(look["High"].max()), 4)},
            extras={"trend_frac": _frac_above(d),
                    "vol_ratio": _vol_ratio(d),
                    "turnover": round(float(d["Close"].iloc[-1]
                                            * d["Volume"].rolling(20).mean().iloc[-1]), 0)
                    if len(d) >= 20 else 0.0})


def _needed(spec: dict) -> int:
    """Deepest lookback anything in the spec asks for, plus headroom."""
    deepest = 0

    def walk(o):
        nonlocal deepest
        if isinstance(o, dict):
            if isinstance(o.get("n"), int):
                deepest = max(deepest, o["n"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(spec)
    return deepest + 30


def _frac_above(d: pd.DataFrame) -> float:
    if d["ma"].iloc[-60:].isna().all():
        return 0.0
    return round(float((d["Close"].iloc[-60:] > d["ma"].iloc[-60:]).mean()), 3)


def _vol_ratio(d: pd.DataFrame) -> float:
    if len(d) < 20:
        return 1.0
    avg = float(d["Volume"].rolling(20).mean().iloc[-1])
    return round(float(d["Volume"].iloc[-1] / avg), 2) if avg else 1.0


# ---------------------------------------------------------------------------
# export — graduate a spec into a hand-editable plugin
# ---------------------------------------------------------------------------


def to_python(spec: dict) -> str:
    import json
    gates_doc = "\n".join(
        f"#   {g.get('label') or '?'}: {describe(g['left'])} {g.get('op', '>')} "
        f"{describe(g['right'])}" for g in spec.get("gates", []))
    return f'''"""{spec.get('label', spec['key'])} — exported from the builder.

{spec.get('description', '')}

Gates:
{gates_doc}

This still interprets the spec below. Rewrite evaluate() by hand when you want
something the vocabulary cannot express.
"""

from __future__ import annotations

from spec import SpecStrategy

SPEC = {json.dumps(spec, indent=4)}


class {_cls(spec['key'])}(SpecStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__(SPEC, params)
'''


def _cls(key: str) -> str:
    return "".join(p.capitalize() or "_" for p in key.split("_")) or "Custom"
