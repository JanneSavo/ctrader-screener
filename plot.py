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


# how many sessions to push each label right, so they do not stack up
_LABEL_SLOT = {"gex_call_wall": 2, "gex_put_wall": 4, "gex_flip": 6}


def _shift(day: str, days: int) -> str:
    """Move a label anchor along by N days, keeping the ISO shape."""
    from datetime import datetime, timedelta
    base = str(day)[:10]
    try:
        d = datetime.strptime(base, "%Y-%m-%d") + timedelta(days=days)
        return d.strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        return _iso(day)


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

    # The pullback window as a box around the bars it actually happened in.
    #
    # This used to run from the swing low up to the lowest value of the moving
    # average inside the window, which is a sliver sitting under the candles and
    # reads as a stray block rather than a highlight. Boxing low-to-high over
    # the same bars is what the eye expects: it frames the dip.
    if z and cfg.get("draw_zone", True):
        top = z.get("window_high") or z.get("ma_at_low") or row["entry"]
        ops.append(Op("zone", {
            "object_type": "rectangle",
            "time1": _iso(z["start"]), "price1": z["low"],
            "time2": _iso(z["end"]), "price2": top,
            "color": colors.get("zone", "#E8B94A"), "fill": True,
        }))
        ops.append(Op("bounce_marker", {
            "object_type": "up_triangle" if row.get("direction") != "short" else "down_triangle",
            "time1": bounce, "price1": z.get("low", row["entry"]),
            "color": colors.get("bounce", "#6FD3B5"),
        }))

    # gamma levels: walls act as magnets, the flip is where hedging inverts
    g = row.get("gex") or {}
    if cfg.get("draw_gex", True) and g.get("ok"):
        for key, price, col, tag in (
                ("gex_call_wall", g.get("call_wall"), colors.get("call_wall", "#4FB6A5"),
                 f"call wall {g.get('call_wall')} ({g.get('call_wall_pct'):+}%)"),
                ("gex_put_wall", g.get("put_wall"), colors.get("put_wall", "#C46A62"),
                 f"put wall {g.get('put_wall')} ({g.get('put_wall_pct'):+}%)"),
                ("gex_flip", g.get("flip"), colors.get("flip", "#9B8CC4"),
                 f"gamma flip {g.get('flip')}")):
            if not price:
                continue
            ops.append(Op(key, {"object_type": "horizontal_line",
                                "price1": price, "color": col}))
            if cfg.get("label_gex", True):
                # Stagger the anchors. Every label used to hang off the same
                # timestamp, so three gamma labels plus the setup label piled up
                # at the left edge and overlapped each other into a smear.
                ops.append(Op(key + "_label", {
                    "object_type": "text",
                    "time1": _shift(z.get("start") or row["asof"], _LABEL_SLOT[key]),
                    "price1": price, "text": tag, "color": col}))

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
        # Anchor the label well LEFT of the pullback window. Anchoring at the
        # window start still put it near the right edge - the window is recent
        # by definition - so the text ran off the viewport and was cut mid-word.
        # Text renders rightward from its anchor, so it needs room to its right.
        ops.append(Op("label", {
            "object_type": "text",
            "time1": _shift(z.get("start") or row["asof"],
                            -int(cfg.get("label_lead_days", 30))),
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

    def _geom_key(self, symbol: str, strategy: str | None) -> str:
        return f"plotgeom:{strategy or 'x'}:{symbol}"

    def _row_key(self, symbol: str, strategy: str | None) -> str:
        return f"plotrow:{strategy or 'x'}:{symbol}"

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
            tf_ok, tf_why = True, None
            if "chart_timeframe" in self.mcp.resolved:
                try:
                    await self.mcp.call("chart_timeframe", period=tf)
                except RuntimeError as e:
                    # Swallowing this silently meant daily-derived zone and
                    # label times were drawn onto a 4h chart, where they land in
                    # the wrong place. Surface it instead.
                    tf_ok, tf_why = False, str(e)[:140]
            return {"reused": True, "chart_id": chart_id,
                    "timeframe": tf, "timeframe_ok": tf_ok, "timeframe_why": tf_why}

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

        ids, done, failed, drawn_ops = [], [], [], []
        for op in plan(row, self.cfg):
            try:
                raw = await self.mcp.call("object_add", **op.args)
                oid = _object_id(raw)
                if oid:
                    ids.append(oid)
                    drawn_ops.append(op)
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

        # keep the intended coordinates alongside the ids: cTrader exposes no
        # lock flag, so objects CAN be dragged. Storing the geometry is what
        # makes restore() possible.
        self._remember(sym, strat, self._mine(sym, strat) + ids)
        if self.store:
            self.store.put(self._geom_key(sym, strat),
                           [{"id": i, "kind": o.kind, "args": o.args}
                            for i, o in zip(ids, drawn_ops)])
            # the row is the source of truth for a redraw: chart objects cannot
            # be edited in place, so restoring means drawing them again
            self.store.put(self._row_key(sym, strat), row)
        return {"symbol": sym, "strategy": strat, "ok": not failed,
                "cleared": cleared, "drawn": len(done), "objects": done,
                "ids": ids, "reused_chart": focused.get("reused"),
                "timeframe": focused.get("timeframe"),
                "timeframe_ok": focused.get("timeframe_ok"),
                "timeframe_why": focused.get("timeframe_why"), "failed": failed}

    # -- putting things back ----------------------------------------------

    async def restore(self, symbol: str, strategy: str | None = None) -> dict:
        """Move this tool's objects back to where it drew them.

        cTrader exposes no lock or read-only flag, so anything drawn can be
        dragged by accident. The first version of this used update_chart_object
        to rewrite the coordinates in place, which would have been the tidy fix.
        It does not work: on cTrader Desktop 2.0.0 that tool answers "Invalid
        parameters" for every MCP-created object, for every argument
        combination tried, including a colour-only change. The identifier is not
        the problem - get_chart_objects returns name == objectId, which is what
        was passed.

        So restore deletes this tool's objects and draws them again from the
        snapshot taken at plot time. The object ids change, which is why the
        snapshot is kept: the geometry does not depend on reading anything back
        off the chart.
        """
        snap = (self.store.get(self._row_key(symbol, strategy))
                if self.store else None)
        if not snap:
            return {"ok": False, "why": "nothing recorded for this symbol - plot it first"}
        res = await self.draw(snap)
        return res | {"restored_from": "snapshot", "note": "objects were redrawn, "
                      "so their ids changed"}

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
