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
strategies/base.py — the seam every strategy plugs into.

A strategy is a file in this folder. It declares what it needs, reads a bar
frame, and returns gates plus levels. Everything else — bar caching, earnings
blackout, news, chatter, LLM review, live quotes, chart plotting, the dashboard
— is shared and strategy-agnostic.

What a strategy MUST NOT do:
  - fetch its own data (bars arrive already cached and shared across strategies)
  - decide its own position size (the risk block is common, so sizing stays
    consistent and comparable across strategies)
  - rank itself against other strategies (see ranking note in registry.py)

Add a strategy by dropping a file here with a Strategy subclass. It is
discovered automatically and appears in config, the UI filter and the plotter
with no changes anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# shared primitives
# ---------------------------------------------------------------------------


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str


@dataclass
class Ctx:
    """Everything a strategy is allowed to know beyond its own bars."""
    regime: dict = field(default_factory=dict)
    equity: float = 10_000.0
    risk_per_trade: float = 0.01


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def close_position(bar) -> float:
    rng = float(bar["High"] - bar["Low"])
    return 0.5 if rng <= 0 else float((bar["Close"] - bar["Low"]) / rng)


# ---------------------------------------------------------------------------
# the base class
# ---------------------------------------------------------------------------


class Strategy:
    key: str = "base"
    label: str = "Base"
    description: str = ""
    direction: str = "long"          # long | short
    needs_regime: str | None = "bull"  # bull | bear | None (regime-agnostic)
    defaults: dict[str, Any] = {}
    # Gates that describe THE ENTRY BAR rather than the structure leading to it.
    # A setup failing only these is not a failure - it is a setup that has not
    # triggered yet. Without this split the screener only ever shows a name on
    # the single evening its trigger bar prints, and shows nothing the day after.
    trigger_gates: frozenset[str] = frozenset()
    rank_weights: dict[str, float] = {"rr": 0.4, "trend_frac": 0.3, "turnover": 0.3}

    def __init__(self, params: dict | None = None):
        self.p = {**self.defaults, **(params or {})}

    # -- to implement ------------------------------------------------------

    @property
    def min_bars(self) -> int:
        raise NotImplementedError

    def evaluate(self, symbol: str, df: pd.DataFrame, ctx: Ctx) -> dict | None:
        raise NotImplementedError

    # -- shared risk block -------------------------------------------------

    def signal(self, symbol: str, d: pd.DataFrame, gates: list[Gate], *,
               entry: float, stop: float, target: float, ctx: Ctx,
               zone: dict | None = None, extras: dict | None = None,
               provisional: bool = False) -> dict:
        """Common sizing, R and payload shape. Every strategy ends here so the
        numbers mean the same thing whichever gates produced them."""
        risk_ps = max(abs(entry - stop), 1e-9)
        reward = abs(target - entry)
        rr = reward / risk_ps
        risk_amt = ctx.equity * ctx.risk_per_trade
        failed = [g.name for g in gates if not g.ok]
        structural = [f for f in failed if f not in self.trigger_gates]
        # triggered: everything passes. watching: only the trigger is missing.
        status = ("triggered" if not failed
                  else "watching" if not structural else "rejected")
        atr = float(d["atr"].iloc[-1]) if "atr" in d else float("nan")

        row = {
            "symbol": symbol,
            "strategy": self.key,
            "strategy_label": self.label,
            "direction": self.direction,
            "asof": d.index[-1].date().isoformat(),
            "provisional": bool(provisional),
            "pass": not failed,
            "status": status,
            "watching": status == "watching",
            "awaiting": [f for f in failed if f in self.trigger_gates],
            "failed": failed,
            "gates": [{"name": g.name, "ok": g.ok, "detail": g.detail} for g in gates],
            "entry": round(float(entry), 4),
            "stop": round(float(stop), 4),
            "target": round(float(target), 4),
            "stop_pct": round(100 * risk_ps / entry, 2),
            "rr": round(rr, 2),
            "units": round(risk_amt / risk_ps, 2),
            "risk_amt": round(risk_amt, 2),
            "atr_pct": round(100 * atr / entry, 2) if atr == atr else None,
            "zone": zone or {},
            "spark": _spark(d),
        }
        row.update(extras or {})
        return row


def _spark(d: pd.DataFrame, n: int = 70) -> dict:
    tail = d.iloc[-n:]
    ma = tail["ma"] if "ma" in tail else tail["Close"].rolling(20).mean()
    return {
        "c": [round(float(x), 4) for x in tail["Close"]],
        "m": [None if pd.isna(x) else round(float(x), 4) for x in ma],
        "lo": int(np.argmin(tail["Low"].values[-10:]) + len(tail) - 10),
    }
