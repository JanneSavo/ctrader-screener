"""
plot.py — push the setup back onto the cTrader chart.

Rewritten against the real cTrader Desktop 2.0.0 toolset. The first version was
written against assumed tools and got the shape wrong in three ways that all
mattered:

  1. There is no per-object-type tool. ONE tool, add_chart_object, draws
     everything via an object_type argument.
  2. Objects have no symbol scoping and no name argument. They land on whatever
     chart is ACTIVE, and come back only as an objectId. So idempotency cannot
     work by name — we record the ids we created and delete exactly those.
  3. cTrader exposes clear_chart_objects, which deletes EVERY drawing on the
     chart. Auto-discovery matched it for deletion. We use delete_chart_object,
     one id at a time, and never touch an id we did not create.

Because objects go to the active chart, every draw first focuses the right
chart: reuse an open tab for that symbol if there is one, otherwise open it.
Blindly calling open_chart would leave a tab per scan.

The nicest find: cTrader has a native risk_reward object taking entry, stop and
target as one block. That replaces three loose horizontal lines with the thing
the platform already knows how to render.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

PREFIX = "SCR"


@dataclass
class Op:
    """One add_chart_object call, already shaped for the real schema."""
    kind: str            # our label, for the event log
    args: dict           # object_type + points, exactly as cTrader wants them

    def label(self) -> str:
        return self.kind


def _iso(day: str) -> str:
    """'2026-08-27' -> ISO 8601 UTC. Times must be full timestamps."""
    return f"{day}T00:00:00Z" if len(day) == 10 else day


def plan(row: dict, cfg: dict) -> list[Op]:
    """Turn one screener row into the objects that belong on its chart."""
    z = row.get("zone") or {}
    colors = cfg.get("colors") or {}
    bounce = _iso(z.get("bounce") or row["asof"])
    ops: list[Op] = []

    # The native risk/reward block: entry, stop and target in one object.
    # Note it renders its own Lots figure from cTrader's default risk settings,
    # which has nothing to do with the size the screener calculated.
    if cfg.get("draw_risk_block", True):
        ops.append(Op("risk_reward", {
            "object_type": "risk_reward",
            "side": "sell" if row.get("direction") == "short" else "buy",
            "price1": row["entry"], "price2": row["stop"], "price3": row["target"],
            "time1": bounce,
        }))

    # Plain horizontal lines as well, not instead: risk_reward is an interactive
    # object, so cTrader only paints its labels while the mouse is over it.
    # These stay visible and put a permanent tag on the price axis.
    if cfg.get("draw_levels", True):
        for name, price, col in (("entry", row["entry"], colors.get("entry", "#7E8DA0")),
                                 ("stop", row["stop"], colors.get("stop", "#E2685F")),
                                 ("target", row["target"], colors.get("target", "#6FD3B5"))):
            ops.append(Op(f"hline_{name}", {
                "object_type": "horizontal_line", "price1": price, "color": col}))

    # the pullback window, as a filled rectangle
    if z and cfg.get("draw_zone", True):
        ops.append(Op("zone", {
            "object_type": "rectangle",
            "time1": _iso(z["start"]), "price1": z["low"],
            "time2": _iso(z["end"]), "price2": z.get("ma_at_low", row["entry"]),
            "color": colors.get("zone", "#E8B94A"), "fill": True,
        }))
        ops.append(Op("bounce_marker", {
            "object_type": "up_triangle" if row.get("direction") != "short" else "down_triangle",
            "time1": bounce, "price1": z.get("low", row["entry"]),
            "color": colors.get("bounce", "#6FD3B5"),
        }))

    # one text object carrying everything the screener knows
    if cfg.get("draw_label", True):
        llm = row.get("llm") or {}
        bits = [f"{row['rr']}R  {row['units']}u  stop {row['stop_pct']}%",
                f"{row.get('strategy_label') or row.get('strategy', '')}"
                f" #{row.get('rank', '?')} score {row.get('score', '')}"]
        e = row.get("earnings") or {}
        if e.get("why"):
            bits.append(e["why"])
        flags = row.get("tape_flags") or []
        if flags:
            bits.append("tape: " + "; ".join(f["text"][:60] for f in flags[:2]))
        if llm.get("verdict") and llm["verdict"] not in ("off", "clear"):
            bits.append(f"{llm['verdict'].upper()}: "
                        f"{'; '.join(llm.get('reasons') or [])[:80]}")
        # Anchor the label at the LEFT edge of the pullback window, not at the
        # bounce bar. The bounce is the newest bar, so text placed there runs
        # off the right edge of the viewport and is unreadable.
        ops.append(Op("label", {
            "object_type": "text", "time1": _iso(z.get("start") or row["asof"]),
            "price1": row["target"],
            "text": " | ".join(bits), "color": colors.get("label", "#E7ECF3"),
        }))
    return ops


def _object_id(raw) -> str | None:
    """Pull the objectId out of an add_chart_object response."""
    def walk(o):
        if isinstance(o, dict):
            low = {str(k).lower(): v for k, v in o.items()}
            for k in ("objectid", "id", "object_id"):
                if low.get(k):
                    return str(low[k])
            for v in o.values():
                got = walk(v)
                if got:
                    return got
        elif isinstance(o, list):
            for v in o:
                got = walk(v)
                if got:
                    return got
        return None
    return walk(raw)


def _live_ids(raw) -> set[str]:
    """objectIds currently on the active chart, from get_chart_objects."""
    found: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            low = {str(k).lower(): v for k, v in o.items()}
            for k in ("objectid", "id", "object_id"):
                if low.get(k):
                    found.add(str(low[k]))
                    break
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(raw)
    return found


class Plotter:
    def __init__(self, mcp, cfg: dict, store=None):
        self.mcp = mcp
        self.cfg = cfg
        self.store = store          # where the objectIds we created are recorded

    # -- id bookkeeping ----------------------------------------------------

    def _key(self, symbol: str, strategy: str | None) -> str:
        return f"plot:{strategy or 'x'}:{symbol}"

    def _mine(self, symbol: str, strategy: str | None) -> list[str]:
        if not self.store:
            return []
        return list(self.store.get(self._key(symbol, strategy)) or [])

    def _remember(self, symbol: str, strategy: str | None, ids: list[str]) -> None:
        if self.store:
            self.store.put(self._key(symbol, strategy), ids)

    def missing_tools(self) -> list[str]:
        need = ["object_add", "chart_open"]
        if self.cfg.get("clear_before_draw", True):
            need.append("object_delete")
        return [t for t in need if t not in self.mcp.resolved]

    # -- chart focus -------------------------------------------------------

    async def focus(self, symbol: str) -> dict:
        """Reuse an open tab for this symbol, else open one. Never both."""
        tf = self.cfg.get("timeframe", "d1")
        chart_id = None
        if "chart_list" in self.mcp.resolved:
            try:
                charts = await self.mcp.call("chart_list")
                chart_id = _find_chart(charts, symbol)
            except RuntimeError:
                chart_id = None

        if chart_id and "chart_focus" in self.mcp.resolved:
            await self.mcp.call("chart_focus", chart_id=chart_id)
            if "chart_timeframe" in self.mcp.resolved:
                try:
                    await self.mcp.call("chart_timeframe", period=tf)
                except RuntimeError:
                    pass
            return {"reused": True, "chart_id": chart_id}

        await self.mcp.call("chart_open", symbol=symbol, period=tf)
        return {"reused": False, "chart_id": None}

    # -- clearing ----------------------------------------------------------

    async def clear(self, symbol: str, strategy: str | None = None,
                    focus_first: bool = True) -> int:
        """Delete only the ids we recorded for this symbol+strategy.

        Anything the user drew themselves has an id we never stored, so it is
        untouchable by construction rather than by a name convention.
        """
        if "object_delete" not in self.mcp.resolved:
            return 0
        mine = self._mine(symbol, strategy)
        if not mine:
            return 0
        if focus_first:
            await self.focus(symbol)

        live = None
        if "object_list" in self.mcp.resolved:
            try:
                live = _live_ids(await self.mcp.call("object_list"))
            except RuntimeError:
                live = None

        removed, left = 0, []
        for oid in mine:
            if live is not None and oid not in live:
                continue                      # already gone; drop it silently
            try:
                await self.mcp.call("object_delete", object_id=oid)
                removed += 1
            except RuntimeError:
                left.append(oid)              # keep it so we retry next time
        self._remember(symbol, strategy, left)
        return removed


    # -- drawing -----------------------------------------------------------

    async def draw(self, row: dict) -> dict:
        sym, strat = row["symbol"], row.get("strategy")
        missing = self.missing_tools()
        if missing:
            return {"symbol": sym, "ok": False, "drawn": 0,
                    "error": f"cTrader exposes no tool for: {', '.join(missing)}. "
                             f"Run --dump-tools and pin them under ctrader.tools."}

        focused = await self.focus(sym)
        cleared = 0
        if self.cfg.get("clear_before_draw", True):
            cleared = await self.clear(sym, strat, focus_first=False)

        ids, done, failed = [], [], []
        for op in plan(row, self.cfg):
            try:
                raw = await self.mcp.call("object_add", **op.args)
                oid = _object_id(raw)
                if oid:
                    ids.append(oid)
                done.append(op.label())
            except (RuntimeError, KeyError) as e:
                failed.append(f"{op.label()}: {str(e)[:140]}")

        if self.cfg.get("draw_indicators", False):
            for name in self.cfg.get("indicators", ["Simple Moving Average"]):
                try:
                    await self.mcp.call("indicator_add", name=name)
                    done.append(f"indicator:{name}")
                except RuntimeError as e:
                    failed.append(f"indicator {name}: {str(e)[:100]}")

        self._remember(sym, strat, self._mine(sym, strat) + ids)
        return {"symbol": sym, "strategy": strat, "ok": not failed,
                "cleared": cleared, "drawn": len(done), "objects": done,
                "ids": ids, "reused_chart": focused.get("reused"), "failed": failed}

    async def draw_many(self, rows: list[dict], top_n: int = 5) -> list[dict]:
        """Serial on purpose. Drawing switches the active chart, so parallel
        calls would race over which chart the next object lands on."""
        out = []
        for r in rows[:top_n]:
            out.append(await self.draw(r))
            await asyncio.sleep(self.cfg.get("delay_s", 0.2))
        return out


def _find_chart(raw, symbol: str) -> str | None:
    """chartId of an already-open tab showing this symbol, if any."""
    want = symbol.upper()
    hit = None

    def walk(o):
        nonlocal hit
        if hit:
            return
        if isinstance(o, dict):
            low = {str(k).lower(): v for k, v in o.items()}
            sym = low.get("symbolname") or low.get("symbol")
            cid = low.get("chartid") or low.get("id")
            if sym and cid and str(sym).upper() == want:
                hit = str(cid)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(raw)
    return hit
