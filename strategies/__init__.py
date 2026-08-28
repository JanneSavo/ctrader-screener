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
strategies/__init__.py — discovery and ranking.

Drop a file in this folder with a Strategy subclass and it shows up in config,
in the dashboard filter and in the plotter. Nothing else needs editing.

RANKING NOTE, and it matters: scores are percentile ranks computed WITHIN one
strategy, never across strategies. A breakout scoring 90 and a snapback scoring
90 are each "best of what that strategy found today" — they are not comparable
to each other. Percentile-ranking a mixed pool would silently invent a ordering
between different edges based on factors they do not share.

Comparing strategies to each other needs realised expectancy, which means a
backtest. Until that exists, the dashboard groups by strategy rather than
pretending one ordering exists.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pandas as pd

from .base import Ctx, Gate, Strategy, atr_wilder, close_position, rsi

__all__ = ["Ctx", "Gate", "Strategy", "atr_wilder", "close_position", "rsi",
           "registry", "build", "rank_within", "regime_ok_for"]

_CACHE: dict[str, type[Strategy]] = {}


def registry() -> dict[str, type[Strategy]]:
    """Every Strategy subclass in this package, keyed by .key."""
    if _CACHE:
        return _CACHE
    for mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if mod.name.startswith("_") or mod.name == "base":
            continue
        m = importlib.import_module(f"{__name__}.{mod.name}")
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == m.__name__:
                if obj.key in _CACHE and _CACHE[obj.key] is not obj:
                    raise RuntimeError(f"two strategies claim key {obj.key!r}")
                _CACHE[obj.key] = obj
    return _CACHE


def saved_specs(folder: str = "specs") -> dict[str, dict]:
    """Specs built in the UI. Loaded as strategies, same as the code plugins."""
    import json
    out: dict[str, dict] = {}
    d = Path(__file__).parent.parent / folder
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            spec = json.loads(f.read_text())
            if spec.get("key"):
                out[spec["key"]] = spec
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build(cfg: dict) -> list[Strategy]:
    """Instantiate the enabled strategies with their configured params."""
    from spec import SpecStrategy
    reg = registry()
    specs = saved_specs()
    out = []
    for key, entry in (cfg or {}).items():
        entry = entry or {}
        if not entry.get("enabled", False) or key not in specs or key in reg:
            continue
        out.append(SpecStrategy(specs[key],
                                {k: v for k, v in entry.items() if k != "enabled"}))
    for key, entry in (cfg or {}).items():
        entry = entry or {}
        if not entry.get("enabled", False):
            continue
        if key in specs and key not in reg:
            continue                       # already built above
        if key not in reg:
            raise RuntimeError(
                f"config enables strategy {key!r}, which does not exist. "
                f"Available: {', '.join(sorted(set(reg) | set(specs)))}")
        out.append(reg[key]({k: v for k, v in entry.items() if k != "enabled"}))
    return out


def regime_ok_for(strat: Strategy, regime: dict) -> tuple[bool, str]:
    """A strategy declares what market it wants. Regime-agnostic ones always run."""
    if strat.needs_regime is None:
        return True, "runs in any regime"
    bull = bool((regime or {}).get("ok"))
    if strat.needs_regime == "bull":
        return bull, (regime or {}).get("note", "no index data")
    if strat.needs_regime == "bear":
        return (not bull), (regime or {}).get("note", "no index data")
    return True, "unknown regime requirement"


def rank_within(rows: list[dict], strategies: list[Strategy]) -> list[dict]:
    """Percentile-rank inside each strategy, then concatenate. Never across."""
    if not rows:
        return []
    weights = {s.key: s.rank_weights for s in strategies}
    out: list[dict] = []
    df = pd.DataFrame(rows)
    for key, grp in df.groupby("strategy", sort=False):
        w = weights.get(key) or {"rr": 1.0}
        total = pd.Series(0.0, index=grp.index)
        used = 0.0
        for col, weight in w.items():
            if col in grp:
                total += grp[col].rank(pct=True) * weight
                used += weight
        grp = grp.assign(score=(100 * total / (used or 1)).round(1))
        grp = grp.sort_values("score", ascending=False)
        for i, rec in enumerate(grp.to_dict("records"), 1):
            # A DataFrame round-trip fills any key that only SOME rows have with
            # NaN - e.g. "flag", which the review sets only on flagged setups.
            # NaN survives json.dumps into SQLite but makes the HTTP layer throw
            # "Out of range float values are not JSON compliant", so /api/results
            # returned 500 and the dashboard showed an empty table.
            rec = {k: (None if isinstance(v, float) and pd.isna(v) else v)
                   for k, v in rec.items()}
            rec["rank"] = i
            out.append(rec)
    out.sort(key=lambda r: (r["rank"], -r["score"]))
    return out
