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
ctrader_mcp.py — thin, schema-agnostic MCP client for cTrader AI Agent Connect.

There is no LLM in this path. MCP is just JSON-RPC over stdio or HTTP, so we
call the tools directly and treat cTrader as a data provider.

Spotware publishes what the servers *can do* (candles, symbols, account,
positions) but not the tool names or argument schemas. So this module:
  1. discovers tools at runtime  (session.list_tools)
  2. resolves them by regex, or by explicit override in config.yaml
  3. normalizes whatever comes back into DataFrames / dicts

Run `python ctrader_mcp.py --dump-tools` once against your own cTrader,
paste the real names + arg names into config.yaml, and the guessing stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# SDK renamed this between releases; accept either.
try:
    from mcp.client.streamable_http import streamable_http_client as _http_client
except ImportError:  # older SDKs
    from mcp.client.streamable_http import streamablehttp_client as _http_client


# ---------------------------------------------------------------------------
# tool resolution
# ---------------------------------------------------------------------------

PATTERNS = {
    "candles":  [r"candle", r"\bbars?\b", r"ohlc", r"histor.*(price|data|quote)"],
    "symbols":  [r"symbol.*(list|available|all)", r"list.*symbol", r"^get_symbols$"],
    "account":  [r"account.*(info|summary|balance|state)", r"\bbalance\b", r"\bequity\b"],
    "details":  [r"symbol.*(detail|info|spec)", r"instrument.*info"],
    "positions": [r"position", r"open.*trade"],
    "news":     [r"news", r"headline", r"market.*(story|article)"],
    "quotes":   [r"(live|current|latest).*(price|quote)", r"^get_price", r"\bquote\b", r"\btick\b"],
    # chart write-back (local MCP only). cTrader draws every object type
    # through ONE tool, so there is a single object_add key.
    "chart_open":     [r"open.*chart"],
    "chart_list":     [r"list.*charts?"],
    "chart_focus":    [r"focus.*chart"],
    "chart_timeframe": [r"chart.*timeframe", r"change.*timeframe"],
    "indicator_add":  [r"add.*indicator", r"indicator.*add", r"apply.*indicator"],
    "object_add":     [r"add.*chart.*object", r"add.*drawing", r"draw.*object"],
    # NEVER auto-match deletion: cTrader also exposes clear_chart_objects, which
    # wipes every drawing on the chart. Pin this in config or leave it unset.
    "object_delete":  [],
    "object_list":    [r"get.*chart.*objects?", r"list.*chart.*objects?"],
    "order_place":    [],
}


@dataclass
class ToolSpec:
    """How to call one logical operation. Overridable from config.yaml."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)   # template, {placeholders}

    def build(self, **kw) -> dict:
        # An empty template means "send exactly what the caller passed". Tools
        # like add_chart_object take a different argument set per object type,
        # so templating them is impossible; without this the args are dropped
        # and cTrader answers "Missing required parameter".
        if not self.args:
            return {k: v for k, v in kw.items() if v is not None}
        out = {}
        for k, v in self.args.items():
            if isinstance(v, str) and "{" in v:
                s = v.format(**kw)
                out[k] = int(s) if s.lstrip("-").isdigit() else s
            else:
                out[k] = v
        return out


class CTraderMCP:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session: ClientSession | None = None
        self.tools: list[Any] = []
        self.resolved: dict[str, ToolSpec] = {}
        self._stack: contextlib.AsyncExitStack | None = None
        self._sem = asyncio.Semaphore(int(cfg.get("max_concurrency", 4)))

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "CTraderMCP":
        try:
            return await self._connect()
        except BaseException:
            # If __aenter__ raises, __aexit__ is never called and the exit
            # stack is left for the garbage collector to unwind in whatever
            # task happens to be running — which is exactly the "exit cancel
            # scope in a different task" crash. Tear it down here instead.
            if self._stack is not None:
                with contextlib.suppress(BaseException):
                    await self._stack.aclose()
                self._stack = None
            self.session = None
            raise

    async def _connect(self) -> "CTraderMCP":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        mode = self.cfg.get("mode", "local")

        if mode == "local":
            p = self.cfg["local"]
            params = StdioServerParameters(
                command=p["command"], args=p.get("args", []), env=p.get("env") or None
            )
            read, write, *_ = await self._stack.enter_async_context(stdio_client(params))
        else:
            r = self.cfg["remote"]
            # cTrader Desktop serves its MCP on plain localhost with no auth, so
            # only send a bearer token when one is actually configured. An empty
            # "Bearer " header is rejected by some servers.
            headers = {}
            if r.get("token"):
                headers["Authorization"] = f"Bearer {r['token']}"
            # SDK <2.1 took headers=; 2.1+ takes a preconfigured http_client.
            try:
                cm = _http_client(r["url"], headers=headers or None)
            except TypeError:
                client = None
                if headers:
                    from mcp.client.streamable_http import create_mcp_http_client
                    client = await self._stack.enter_async_context(
                        create_mcp_http_client(headers=headers))
                cm = _http_client(r["url"], http_client=client)
            read, write, *_ = await self._stack.enter_async_context(cm)

        timeout = float(self.cfg.get("connect_timeout_s", 20))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(self.session.initialize(), timeout)
        self.tools = (await asyncio.wait_for(self.session.list_tools(), timeout)).tools
        self._resolve()
        return self

    async def __aexit__(self, *exc):
        if self._stack:
            await self._stack.__aexit__(*exc)
        self.session = None

    # -- resolution --------------------------------------------------------

    def _resolve(self) -> None:
        overrides = self.cfg.get("tools") or {}
        names = {t.name: t for t in self.tools}

        # Walk config keys too, not just PATTERNS: a pinned tool with no
        # discovery pattern (a deliberately un-guessable one) must still resolve.
        for key in list(PATTERNS) + [k for k in overrides if k not in PATTERNS]:
            pats = PATTERNS.get(key, [])
            ov = overrides.get(key)
            if ov and ov.get("name") in names:
                self.resolved[key] = ToolSpec(ov["name"], ov.get("args", {}))
                continue
            if ov and ov.get("name"):
                raise RuntimeError(
                    f"config.yaml pins tools.{key}.name={ov['name']!r} but cTrader "
                    f"does not expose it. Run --dump-tools to see the real names."
                )
            hit = self._match(pats)
            if hit:
                self.resolved[key] = ToolSpec(hit.name, self._guess_args(hit, key))

    def _match(self, pats: list[str]):
        for p in pats:
            for t in self.tools:
                blob = f"{t.name} {getattr(t, 'description', '') or ''}".lower()
                if re.search(p, t.name.lower()) or re.search(p, blob):
                    return t
        return None

    @staticmethod
    def _schema(tool) -> dict:
        """MCP SDK 2.1 renamed inputSchema -> input_schema. Accept both."""
        return (getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None) or {})

    def _guess_args(self, tool, key: str) -> dict:
        """Best-effort arg template from the tool's own JSON schema."""
        schema = self._schema(tool)
        props = list((schema.get("properties") or {}).keys())
        args: dict[str, Any] = {}
        if key != "candles":
            for p in props:
                if re.fullmatch(r"symbol(name|_name|id)?", p, re.I):
                    args[p] = "{symbol}"
            return args
        for p in props:
            pl = p.lower()
            if re.fullmatch(r"symbol(name|_name|id)?", pl):
                args[p] = "{symbol}"
            elif pl in ("period", "timeframe", "interval", "resolution", "granularity"):
                args[p] = self.cfg.get("timeframe", "Daily")
            elif pl in ("count", "limit", "bars", "numbars", "num_bars", "maxcount"):
                args[p] = "{count}"
        return args

    def describe(self) -> str:
        lines = [f"{len(self.tools)} tools exposed by cTrader:", ""]
        for t in sorted(self.tools, key=lambda x: x.name):
            props = (self._schema(t).get("properties") or {})
            req = set(self._schema(t).get("required") or [])
            sig = ", ".join(f"{k}{'*' if k in req else ''}" for k in props)
            lines.append(f"  {t.name}({sig})")
            d = (getattr(t, "description", "") or "").strip().splitlines()
            if d:
                lines.append(f"      {d[0][:110]}")
        lines += ["", "resolved mapping:"]
        for k in PATTERNS:
            s = self.resolved.get(k)
            lines.append(f"  {k:<10} -> {s.name + ' ' + json.dumps(s.args) if s else '(none found)'}")
        return "\n".join(lines)

    # -- calling -----------------------------------------------------------

    async def call(self, key: str, **kw) -> Any:
        spec = self.resolved.get(key)
        if not spec:
            raise RuntimeError(f"no cTrader tool resolved for {key!r}; see --dump-tools")
        async with self._sem:
            timeout = float(self.cfg.get("call_timeout_s", 30))
            try:
                res = await asyncio.wait_for(
                    self.session.call_tool(spec.name, spec.build(**kw)), timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"{spec.name} timed out after {timeout:.0f}s. cTrader's MCP "
                    f"server serialises requests - if this repeats, check that "
                    f"ctrader.max_concurrency is 1.")
        # SDK 2.1 renamed isError -> is_error. Check both.
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise RuntimeError(f"{spec.name} failed: {_text(res)[:300]}")
        payload = _payload(res)
        # cTrader also reports some failures as a plain text body with is_error
        # unset, which would otherwise look like a successful empty result.
        if isinstance(payload, str) and re.match(
                r"\s*(missing required|invalid |error[: ]|failed|not found)",
                payload, re.I):
            raise RuntimeError(f"{spec.name} failed: {payload[:200]}")
        return payload

    # cTrader's get_trendbars takes a DATE RANGE, not a bar count, and its
    # timeframe codes are m1/h1/d1 rather than "Daily". Callers still think in
    # bars, so the translation happens here.
    TF = {"daily": "d1", "d1": "d1", "weekly": "w1", "w1": "w1",
          "monthly": "month1", "h4": "h4", "h1": "h1", "hourly": "h1",
          "m30": "m30", "m15": "m15", "m5": "m5", "m1": "m1"}

    async def candles(self, symbol: str, count: int) -> pd.DataFrame | None:
        tf = self.TF.get(str(self.cfg.get("timeframe", "Daily")).lower(), "d1")
        # widen for weekends and holidays, and respect the server's 1000 cap
        span = 3 if tf in ("d1", "w1", "month1") else 1
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(int(count) * span, 7))
        raw = await self.call("candles", symbol=symbol, count=min(int(count), 1000),
                              **{"from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "timeframe": tf})
        return normalize_candles(raw)

    async def symbols(self) -> list[str]:
        raw = await self.call("symbols")
        return normalize_symbols(raw)

    async def account(self) -> dict:
        raw = await self.call("account")
        return normalize_account(raw)


# ---------------------------------------------------------------------------
# response normalization — MCP results are content blocks, not typed objects
# ---------------------------------------------------------------------------


def _text(res) -> str:
    parts = []
    for c in getattr(res, "content", []) or []:
        if getattr(c, "type", None) == "text":
            parts.append(c.text)
    return "\n".join(parts)


def _payload(res) -> Any:
    """structuredContent if the server sends it, else parsed JSON, else text."""
    sc = getattr(res, "structuredContent", None)
    if sc:
        return sc
    txt = _text(res).strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return txt


def _rows(obj: Any) -> list[dict]:
    """Dig a list-of-dicts out of whatever wrapper the server used."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for k in ("candles", "bars", "data", "result", "results", "items",
                  "symbols", "values", "content", "ohlc"):
            if k in obj:
                got = _rows(obj[k])
                if got:
                    return got
        for v in obj.values():                       # last resort: any list child
            if isinstance(v, (list, dict)):
                got = _rows(v)
                if got:
                    return got
    return []


_ALIASES = {
    "open": ["open", "o", "openprice", "open_price"],
    "high": ["high", "h", "highprice", "high_price"],
    "low": ["low", "l", "lowprice", "low_price"],
    "close": ["close", "c", "closeprice", "close_price"],
    "volume": ["volume", "v", "tickvolume", "tick_volume", "vol"],
    "ts": ["time", "timestamp", "date", "datetime", "opentime", "open_time", "t"],
}


def normalize_candles(raw: Any) -> pd.DataFrame | None:
    rows = _rows(raw)
    if not rows:
        return None
    lower = [{str(k).lower().replace(" ", "_"): v for k, v in r.items()} for r in rows]

    def pick(r: dict, field: str):
        for a in _ALIASES[field]:
            if a in r:
                return r[a]
        return None

    recs = []
    for r in lower:
        c = pick(r, "close")
        t = pick(r, "ts")
        if c is None or t is None:
            continue
        recs.append({
            "ts": _parse_ts(t),
            "Open": float(pick(r, "open") if pick(r, "open") is not None else c),
            "High": float(pick(r, "high") if pick(r, "high") is not None else c),
            "Low": float(pick(r, "low") if pick(r, "low") is not None else c),
            "Close": float(c),
            "Volume": float(pick(r, "volume") or 0.0),
        })
    if not recs:
        return None
    df = pd.DataFrame(recs).dropna(subset=["ts"])
    df = df.drop_duplicates("ts").sort_values("ts").set_index("ts")
    return df


def _parse_ts(v: Any) -> pd.Timestamp | None:
    try:
        if isinstance(v, (int, float)):
            # epoch seconds vs milliseconds
            unit = "ms" if float(v) > 1e11 else "s"
            return pd.Timestamp(datetime.fromtimestamp(float(v) / (1000 if unit == "ms" else 1),
                                                       tz=timezone.utc)).tz_localize(None)
        return pd.Timestamp(str(v)).tz_localize(None) if pd.Timestamp(str(v)).tzinfo is None \
            else pd.Timestamp(str(v)).tz_convert("UTC").tz_localize(None)
    except (ValueError, TypeError, OverflowError):
        return None


def normalize_symbols(raw: Any) -> list[str]:
    rows = _rows(raw)
    out = []
    for r in rows:
        low = {str(k).lower(): v for k, v in r.items()}
        for k in ("symbolname", "symbol_name", "symbol", "name", "code"):
            if low.get(k):
                out.append(str(low[k]))
                break
    if not out and isinstance(raw, list):
        out = [str(x) for x in raw if isinstance(x, str)]
    return sorted(set(out))


def normalize_account(raw: Any) -> dict:
    src = raw if isinstance(raw, dict) else (_rows(raw)[:1] or [{}])[0]
    low = {str(k).lower().replace(" ", "_"): v for k, v in (src or {}).items()}
    for k in ("account", "accountinfo", "summary"):
        if isinstance(low.get(k), dict):
            low = {str(a).lower(): b for a, b in low[k].items()}
            break

    def num(*keys):
        for k in keys:
            if k in low:
                try:
                    return float(str(low[k]).replace(",", ""))
                except ValueError:
                    pass
        return None

    return {
        "balance": num("balance", "accountbalance"),
        "equity": num("equity", "accountequity"),
        "margin_free": num("freemargin", "free_margin", "marginfree", "availablemargin"),
        # cTrader calls these depositAsset and traderId
        "currency": (low.get("currency") or low.get("depositcurrency")
                     or low.get("deposit_currency") or low.get("depositasset")),
        "account_id": (low.get("accountid") or low.get("account_id")
                       or low.get("login") or low.get("traderid")),
        "broker": low.get("brokername"),
        "leverage": low.get("leverage"),
    }


# ---------------------------------------------------------------------------

class MCPOwner:
    """Keeps one session alive with its lifecycle pinned to a single task.

    stdio_client runs an anyio task group. Entering it in a request handler and
    exiting it from a different task raises "attempted to exit cancel scope in a
    different task" and takes the server down. So open and close both happen
    inside _run; everyone else just borrows the live session, which is safe to
    call concurrently from any task.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mcp: CTraderMCP | None = None
        self.error: BaseException | None = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()

    async def _run(self):
        try:
            async with CTraderMCP(self.cfg) as mcp:
                self.mcp, self.error = mcp, None
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:      # noqa: BLE001 - surfaced to callers
            self.error = e
            self._ready.set()
        finally:
            self.mcp = None

    async def get(self) -> CTraderMCP:
        if self.mcp is not None:
            return self.mcp
        if self._task is None or self._task.done():
            self._ready = asyncio.Event()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(),
                                   float(self.cfg.get("connect_timeout_s", 20)) + 5)
        except asyncio.TimeoutError:
            self._task.cancel()
            raise ConnectionError("timed out starting the cTrader MCP server")
        if self.mcp is None:
            raise ConnectionError(
                f"cTrader MCP is not reachable: {self.error}. "
                f"Is cTrader running, and is ctrader.local in config.yaml correct?")
        return self.mcp

    async def close(self):
        self._stop.set()
        if self._task:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=5)
        self._task, self.mcp = None, None


async def _dump(cfg_path: str):
    import yaml
    cfg = yaml.safe_load(open(cfg_path))["ctrader"]
    async with CTraderMCP(cfg) as c:
        print(c.describe())
        if "candles" in c.resolved:
            sym = cfg.get("index_symbol", "US500")
            print(f"\nsample candles for {sym}:")
            df = await c.candles(sym, 5)
            print(df if df is not None else "  (could not normalize — paste the raw shape to fix)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-tools", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    if a.dump_tools:
        asyncio.run(_dump(a.config))
    else:
        print("nothing to do — try --dump-tools")
