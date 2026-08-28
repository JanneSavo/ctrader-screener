"""
server.py — the screener service. FastAPI + SSE, no build step, no cloud.

    python server.py            # http://127.0.0.1:8790
    python server.py --scan     # headless run, prints the table, exits
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from assistant import Deps as ChatDeps, chat as run_chat
from backtest import BTConfig
from backtest import run as run_backtest
from bots import DEFAULT_CAPS, EXITS, BotState, CodegenError, PaperEngine, Position, generate_cbot, validate_bot
from builder import Composite, SpecError, preview, validate, vocabulary
from catalysts import EarningsCalendar, NewsFeed, enrich
from explain import explain as explain_move
from ctrader_mcp import CTraderMCP, MCPOwner
from llm import SEVERITY_LABEL, Analyst
from quotes import Clocks, Watchlist, ensure_subscribed, poll_quotes, price_state
from plot import Plotter
from social import SocialFeed, attach
from store import Store
from tape import brief as tape_brief, digest as tape_digest, flags as tape_flags
from strategies import Ctx, build, rank_within, regime_ok_for, registry
from strategy import Params, regime

ROOT = Path(__file__).parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
PARAMS = Params(**(CFG.get("strategy") or {}))
STORE = Store(CFG.get("db", str(ROOT / "screener.db")))

def load_strategies() -> list:
    """Coded strategies from config, plus enabled recipes from the builder."""
    out = build(CFG.get("strategies") or {"pullback50": {"enabled": True}})
    for r in STORE.recipes(enabled_only=True):
        try:
            out.append(Composite(r["spec"]))
        except SpecError as e:
            print(f"skipping saved recipe {r['key']}: {e}", file=sys.stderr)
    return out


STRATS = load_strategies()
MIN_BARS = max([s.min_bars for s in STRATS], default=300)


def reload_strategies() -> None:
    global STRATS, MIN_BARS
    STRATS = load_strategies()
    MIN_BARS = max([s.min_bars for s in STRATS], default=300)

STATE: dict = {"scanning": False, "done": 0, "total": 0, "symbol": "", "error": None}
QUEUE: asyncio.Queue | None = None

CLOCKS = Clocks(**(CFG.get("refresh") or {}))
WATCH = Watchlist()
LIVE: dict = {}            # symbol -> latest price_state
OWNER = MCPOwner(CFG["ctrader"])
MCP_LOCK = asyncio.Lock()


async def get_mcp() -> CTraderMCP:
    """One long-lived session shared by the scan, the clocks and the plotter.

    Opening a stdio subprocess per poll would be absurd at a 5s interval. The
    lock only serialises connect attempts; a live session is handed straight
    back so one stalled connect cannot block every other request.
    """
    if OWNER.mcp is not None:
        return OWNER.mcp
    async with MCP_LOCK:
        return await OWNER.get()


async def drop_mcp() -> None:
    async with MCP_LOCK:
        await OWNER.close()


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

def _wanted() -> set[str] | None:
    """S&P 500 membership list, if you supplied one. None = take everything."""
    f = CFG.get("universe_file")
    if not f:
        return None
    path = ROOT / f
    if not path.exists():
        return None
    return {l.strip().upper() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")}


def _match_universe(broker_symbols: list[str]) -> list[str]:
    """cTrader names US stock CFDs like AAPL.US — strip the suffix to compare."""
    want = _wanted()
    suffixes = CFG.get("symbol_suffixes", [".US", ".NAS", ".NYSE", ""])
    if want is None:
        return [s for s in broker_symbols
                if any(s.upper().endswith(x) for x in suffixes if x)]
    out = []
    for s in broker_symbols:
        base = s.upper()
        for x in suffixes:
            if x and base.endswith(x):
                base = base[: -len(x)]
                break
        if base.replace("-", ".") in want or base in want:
            out.append(s)
    return sorted(out)


async def _symbol_universe(mcp: CTraderMCP) -> list[str]:
    cached = STORE.get("universe", max_age_s=CFG.get("universe_ttl_s", 86400))
    if cached:
        return cached
    syms = await mcp.symbols()
    matched = _match_universe(syms)
    STORE.put("universe", matched)
    STORE.put("broker_symbol_count", len(syms))
    return matched


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

async def _ensure_bars(mcp: CTraderMCP, symbol: str) -> None:
    have = STORE.count(symbol)
    need = MIN_BARS + 20
    count = need if have < need else CFG.get("incremental_bars", 10)
    df = await mcp.candles(symbol, count)
    if df is not None:
        STORE.upsert(symbol, df)


async def run_scan() -> dict:
    started = datetime.now(timezone.utc).isoformat()
    STATE.update(scanning=True, done=0, total=0, symbol="", error=None)
    await _emit({"phase": "connecting"})

    try:
        if True:
            mcp = await get_mcp()
            missing = [k for k in ("candles", "symbols") if k not in mcp.resolved]
            if missing:
                raise RuntimeError(
                    f"cTrader did not expose a tool for: {', '.join(missing)}. "
                    f"Run `python ctrader_mcp.py --dump-tools` and pin the real "
                    f"names under `tools:` in config.yaml.")

            # account -> equity for sizing
            equity = CFG.get("fallback_equity", 10000.0)
            acct = {}
            if "account" in mcp.resolved:
                try:
                    acct = await mcp.account()
                    equity = acct.get("equity") or acct.get("balance") or equity
                except RuntimeError as e:
                    acct = {"error": str(e)}
            STORE.put("account", acct)

            # regime on the index
            idx = CFG["ctrader"].get("index_symbol", "US500")
            await _emit({"phase": "regime", "symbol": idx})
            await _ensure_bars(mcp, idx)
            reg = regime(STORE.bars(idx), PARAMS)
            reg["symbol"] = idx
            STORE.put("regime", reg)
            notes = {"earnings": "disabled", "llm": "disabled", "reviewed": 0}

            symbols = await _symbol_universe(mcp)
            STATE.update(total=len(symbols))
            await _emit({"phase": "scanning", "total": len(symbols)})

            rows, errors = [], []
            sem = asyncio.Semaphore(int(CFG["ctrader"].get("max_concurrency", 4)))

            ctx = Ctx(regime=reg, equity=equity,
                      risk_per_trade=PARAMS.risk_per_trade)
            active = [(st, ok) for st in STRATS
                      for ok, _why in [regime_ok_for(st, reg)]]
            runnable = [st for st, ok in active if ok]
            skipped = [st.key for st, ok in active if not ok]
            notes["strategies"] = {"ran": [st.key for st in runnable],
                                   "skipped_on_regime": skipped}

            async def one(sym: str):
                async with sem:
                    try:
                        await _ensure_bars(mcp, sym)
                        bars = STORE.bars(sym)
                        for st in runnable:
                            r = st.evaluate(sym, bars, ctx)
                            if r and r["pass"]:
                                rows.append(r)
                    except Exception as e:  # one bad symbol must not kill the scan
                        errors.append({"symbol": sym, "error": str(e)[:200]})
                    finally:
                        STATE["done"] += 1
                        STATE["symbol"] = sym
                        await _emit({"phase": "scanning", "done": STATE["done"],
                                     "total": STATE["total"], "symbol": sym})

            await asyncio.gather(*(one(s) for s in symbols))

            # --- catalysts + review, on survivors only -----------------
            cat_cfg = CFG.get("catalysts") or {}
            if rows and cat_cfg.get("enabled"):
                await _emit({"phase": "catalysts", "count": len(rows)})
                cal = EarningsCalendar(cat_cfg, STORE)
                notes["earnings"] = await cal.load()
                feed = NewsFeed(cat_cfg, mcp if "news" in mcp.resolved else None)
                rows = await enrich(rows, cal, feed, cat_cfg)

                soc_cfg = CFG.get("social") or {}
                if soc_cfg.get("enabled"):
                    await _emit({"phase": "social", "count": len(rows)})
                    rows = await attach(rows, SocialFeed(soc_cfg, STORE), soc_cfg)
                    notes["social"] = soc_cfg.get("provider", "stocktwits")
                else:
                    rows = await attach(rows, SocialFeed(soc_cfg, STORE), soc_cfg)

                for r in rows:
                    STORE.add_stories(r["symbol"], "news", r.get("news") or [])
                    STORE.add_stories(r["symbol"], "social",
                                      (r.get("social") or {}).get("posts") or [])
                blocked = [r for r in rows if not r["pass"]]
                rows = [r for r in rows if r["pass"]]
                STORE.put("blocked_by_earnings",
                          [{"symbol": r["symbol"], "why": r["earnings"]["why"]} for r in blocked])

            # tape digest for every survivor: computed here, never by the model
            if rows:
                atr_pool = [r.get("atr_pct") for r in rows if r.get("atr_pct")]
                turn_pool = [r.get("turnover") for r in rows if r.get("turnover")]
                for r in rows:
                    tp = tape_digest(r, STORE.bars(r["symbol"]), atr_pool, turn_pool)
                    fl = tape_flags(r, tp)
                    r["tape"] = tp
                    r["tape_flags"] = fl
                    r["tape_brief"] = tape_brief(r, tp, fl)
                    if fl:
                        r["gates"] = list(r.get("gates", [])) + [{
                            "name": "Tape check", "ok": sum(f["weight"] for f in fl) < 4,
                            "detail": "; ".join(f["text"] for f in fl[:3])}]

            analyst = Analyst(CFG.get("llm") or {}, STORE)
            if rows and analyst.enabled:
                await _emit({"phase": "review", "count": len(rows)})
                rows = await analyst.review(rows)
                notes["llm"] = f"{analyst.model} / {analyst.mode}"
                notes["reviewed"] = len(rows)
                for r in rows:
                    if (r.get("llm") or {}).get("verdict") not in (None, "off"):
                        STORE.add_analysis(r["symbol"], r["llm"])
            STORE.put("stage_notes", notes)

            for r in rows:
                r.pop("tape_brief", None)      # prompt input, not a stored result
            ranked = rank_within(rows, STRATS)
            STORE.put("errors", errors[:50])
            plot_cfg = CFG.get("plot") or {}
            if ranked and plot_cfg.get("auto"):
                await _emit({"phase": "plot", "count": min(len(ranked), plot_cfg.get("top_n", 5))})
                res = await Plotter(mcp, plot_cfg, STORE).draw_many(ranked, plot_cfg.get("top_n", 5))
                STORE.put("plotted", res)
                notes["plotted"] = sum(1 for r in res if r["ok"])

            WATCH.load(ranked)
            LIVE.clear()
            if ranked:
                sub = await ensure_subscribed(
                    mcp, sorted(WATCH.symbols),
                    (CFG.get('refresh') or {}).get('watchlist', 'Screener'))
                STORE.put('subscription', sub)
                notes['subscribed'] = sub
            sid = STORE.save_scan(reg["ok"], reg["note"], len(symbols), ranked, started)
            await _emit({"phase": "done", "scan_id": sid, "hits": len(ranked),
                         "errors": len(errors)})
            return {"scan_id": sid, "hits": len(ranked), "errors": len(errors),
                    **(STORE.get("stage_notes") or {})}

    except Exception as e:
        STATE["error"] = str(e)
        await _emit({"phase": "error", "message": str(e)})
        await drop_mcp()
        raise
    finally:
        STATE["scanning"] = False


async def quote_loop() -> None:
    """Fast clock: price only, for rows already on screen."""
    while True:
        iv = CLOCKS.quote_interval
        if iv <= 0 or not WATCH.symbols or STATE["scanning"]:
            await asyncio.sleep(1)
            continue
        try:
            mcp = await get_mcp()
            prices = await poll_quotes(mcp, sorted(WATCH.symbols))
            missing = sorted(WATCH.symbols - set(prices))
            if missing != (STORE.get('unquoted') or []):
                STORE.put('unquoted', missing)
            for sym, px in prices.items():
                LIVE[sym] = price_state(WATCH.entries[sym], px, CLOCKS.max_chase_pct)
            if prices:
                await _emit({"phase": "quotes", "live": LIVE})
        except Exception as e:
            await _emit({"phase": "quote_error", "message": str(e)[:160]})
            await drop_mcp()
            await asyncio.sleep(5)
        await asyncio.sleep(iv)


async def forming_loop() -> None:
    """Medium clock: re-run the gates against today's still-open daily bar."""
    while True:
        iv = CLOCKS.forming_interval
        if iv <= 0 or not WATCH.symbols or STATE["scanning"]:
            await asyncio.sleep(2)
            continue
        try:
            mcp = await get_mcp()
            acct = STORE.get("account") or {}
            equity = acct.get("equity") or acct.get("balance") or CFG.get("fallback_equity", 10000.0)
            changed = []
            for sym in sorted(WATCH.symbols):
                df = await mcp.candles(sym, 3)     # includes the open bar
                if df is not None:
                    STORE.upsert(sym, df)
                st = next((x for x in STRATS if x.key == WATCH.strategy_of.get(sym)), None)
                if st is None:
                    continue
                r = st.evaluate(sym, STORE.bars(sym), Ctx(
                    regime=STORE.get("regime") or {}, equity=equity,
                    risk_per_trade=PARAMS.risk_per_trade))
                if r:
                    r["provisional"] = True
                    changed.append({"symbol": sym, "strategy": r["strategy"], "pass": r["pass"],
                                    "failed": r["failed"], "entry": r["entry"],
                                    "stop": r["stop"], "rr": r["rr"],
                                    "vol_ratio": r["vol_ratio"]})
            if changed:
                STORE.put("forming", changed)
                await _emit({"phase": "forming", "rows": changed})
        except Exception as e:
            await _emit({"phase": "forming_error", "message": str(e)[:160]})
            await drop_mcp()
            await asyncio.sleep(10)
        await asyncio.sleep(iv)


async def scan_loop() -> None:
    """Slow clock: full universe. Off by default — nothing new until a bar closes."""
    while True:
        iv = CLOCKS.scan_interval
        if iv <= 0:
            await asyncio.sleep(5)
            continue
        await asyncio.sleep(iv)
        if not STATE["scanning"]:
            with contextlib.suppress(Exception):
                await run_scan()


# ---------------------------------------------------------------------------
# bots
# ---------------------------------------------------------------------------

BOT_STATE: dict[str, BotState] = {}


def _known_strategies() -> set[str]:
    return set(registry()) | {r["key"] for r in STORE.recipes()}


def _bot_state(rec: dict, equity: float) -> BotState:
    st = BOT_STATE.get(rec["key"])
    if st is None:
        raw = rec.get("state") or {}
        st = BotState(key=rec["key"], equity_start=equity,
                      realised=float(raw.get("realised", 0.0)),
                      opened_today=int(raw.get("opened_today", 0)),
                      day=raw.get("day", ""), halted=raw.get("halted", ""))
        st.positions = [Position(**p) for p in raw.get("positions", [])]
        BOT_STATE[rec["key"]] = st
    return st


def _persist(st: BotState) -> None:
    STORE.set_bot(st.key, state={"realised": round(st.realised, 2),
                                 "opened_today": st.opened_today, "day": st.day,
                                 "halted": st.halted,
                                 "positions": [p.to_dict() for p in st.positions]})


async def bot_loop() -> None:
    """Paper bots tick on the same cadence as the forming clock."""
    while True:
        iv = float((CFG.get("bots") or {}).get("tick_s", 60))
        if iv <= 0 or STATE["scanning"]:
            await asyncio.sleep(2)
            continue
        try:
            recs = [b for b in STORE.bots(enabled_only=True) if b["mode"] == "paper"]
            if recs:
                acct = STORE.get("account") or {}
                equity = acct.get("equity") or CFG.get("fallback_equity", 10000.0)
                scan = STORE.latest_scan()
                rows = scan["rows"] if scan else []
                prices = {s: v["price"] for s, v in LIVE.items()}
                now = datetime.now(timezone.utc)
                for rec in recs:
                    eng = PaperEngine(rec["spec"], STORE, equity)
                    st = _bot_state(rec, equity)
                    eng.roll_day(st, now)
                    for p in st.positions:
                        p.bars_held += 1
                    events = eng.manage(st, prices, now)
                    for row in rows:
                        if eng.considers(row):
                            ev = eng.open(st, row, now)
                            if ev and ev["event"] != "skipped":
                                events.append(ev)
                    STORE.log_bot(rec["key"], events)
                    _persist(st)
                    if events:
                        await _emit({"phase": "bot", "bot": rec["key"], "events": events})
        except Exception as e:
            await _emit({"phase": "bot_error", "message": str(e)[:200]})
            await asyncio.sleep(10)
        await asyncio.sleep(iv)


async def _emit(evt: dict) -> None:
    if QUEUE:
        await QUEUE.put(evt)


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global QUEUE
    QUEUE = asyncio.Queue()
    # Rehydrate the watchlist from the last saved scan. It lives in memory, so
    # without this a restart leaves the quote clock with nothing to poll and
    # prices sit frozen until you run a new scan.
    _scan = STORE.latest_scan()
    if _scan and _scan.get("rows"):
        WATCH.load(_scan["rows"])
        print(f"watching {len(WATCH.symbols)} symbols from scan {_scan['id']}",
              file=sys.stderr)
    tasks = [asyncio.create_task(t())
             for t in (quote_loop, forming_loop, scan_loop, bot_loop)]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await drop_mcp()


app = FastAPI(title="cTrader pullback screener", lifespan=lifespan)


@app.get("/api/state")
async def api_state():
    scan = STORE.latest_scan()
    return {
        "scanning": STATE["scanning"],
        "progress": {"done": STATE["done"], "total": STATE["total"], "symbol": STATE["symbol"]},
        "error": STATE["error"],
        "regime": STORE.get("regime"),
        "account": STORE.get("account"),
        "universe": len(STORE.get("universe") or []),
        "broker_symbols": STORE.get("broker_symbol_count"),
        "risk_pct": PARAMS.risk_per_trade,
        "scan": {k: scan[k] for k in ("id", "finished", "scanned", "hits")} if scan else None,
        "history": STORE.scan_history(20),
        "errors": STORE.get("errors") or [],
        "stages": STORE.get("stage_notes") or {},
        "clocks": vars(CLOCKS),
        "assistant_model": (CFG.get("llm") or {}).get("assistant_model")
                            or (CFG.get("llm") or {}).get("model"),
        "strategies": [{"key": k, "label": v.label, "description": v.description,
                        "direction": v.direction, "needs_regime": v.needs_regime,
                        "enabled": any(s.key == k for s in STRATS), "built": False}
                       for k, v in sorted(registry().items())]
                      + [{"key": r["key"], "label": r["label"],
                          "description": r["spec"].get("description", ""),
                          "direction": r["spec"].get("direction", "long"),
                          "needs_regime": r["spec"].get("needs_regime"),
                          "enabled": r["enabled"], "built": True}
                         for r in STORE.recipes()],
        "live": LIVE,
        "unquoted": STORE.get("unquoted") or [],
        "subscription": STORE.get("subscription") or {},
        "forming": STORE.get("forming") or [],
        "blocked_by_earnings": STORE.get("blocked_by_earnings") or [],
        "plotted": STORE.get("plotted") or [],
    }


@app.get("/api/results")
async def api_results():
    scan = STORE.latest_scan()
    return {"rows": scan["rows"] if scan else [], "scan": scan and
            {k: scan[k] for k in ("id", "finished", "regime_ok", "regime_note", "scanned", "hits")}}


@app.post("/api/scan")
async def api_scan():
    if STATE["scanning"]:
        raise HTTPException(409, "a scan is already running")
    asyncio.create_task(run_scan())
    return {"started": True}


async def _mcp_or_503() -> CTraderMCP:
    try:
        return await get_mcp()
    except Exception as e:
        await drop_mcp()
        raise HTTPException(503, str(e)[:300])


@app.post("/api/plot/{symbol}")
async def api_plot(symbol: str):
    """Draw one setup onto its cTrader chart. Local MCP only."""
    scan = STORE.latest_scan()
    row = next((r for r in (scan["rows"] if scan else []) if r["symbol"] == symbol), None)
    if not row:
        raise HTTPException(404, f"{symbol} is not in the latest scan")
    mcp = await _mcp_or_503()
    res = await Plotter(mcp, CFG.get("plot") or {}, STORE).draw(row)
    if not res["ok"] and res.get("error"):
        raise HTTPException(503, res["error"])
    return res


@app.post("/api/plot")
async def api_plot_top(body: dict | None = None):
    scan = STORE.latest_scan()
    rows = scan["rows"] if scan else []
    if not rows:
        raise HTTPException(404, "nothing scanned yet")
    cfg = CFG.get("plot") or {}
    n = int((body or {}).get("top_n") or cfg.get("top_n", 5))
    res = await Plotter(await _mcp_or_503(), cfg, STORE).draw_many(rows, n)
    STORE.put("plotted", res)
    return {"results": res}


@app.delete("/api/plot/{symbol}")
async def api_unplot(symbol: str):
    """Remove only this screener's objects for one symbol."""
    scan_l = STORE.latest_scan()
    strat_l = next((r.get("strategy") for r in (scan_l["rows"] if scan_l else [])
                    if r["symbol"] == symbol), None)
    removed = await Plotter(await _mcp_or_503(), CFG.get("plot") or {},
                            STORE).clear(symbol, strat_l)
    return {"symbol": symbol, "removed": removed}


@app.get("/api/feed")
async def api_feed(kind: str | None = None, symbol: str | None = None, limit: int = 120):
    """kind=news | social | omitted for both."""
    if kind not in (None, "news", "social"):
        raise HTTPException(400, "kind must be news or social")
    return {"items": STORE.feed(kind, symbol, min(limit, 500))}


@app.get("/api/analysis")
async def api_analysis(min_severity: int = 0, limit: int = 100):
    """min_severity=2 is the Important view."""
    return {"items": STORE.analyses(max(0, min(3, min_severity)), min(limit, 300)),
            "labels": SEVERITY_LABEL}


@app.get("/api/llm/health")
async def api_llm_health():
    """Is the model server reachable and does it have the configured model?"""
    return await Analyst(CFG.get("llm") or {}, STORE).health()


@app.post("/api/llm/test")
async def api_llm_test():
    """Run the review over the latest scan's top row and return the raw verdict."""
    scan = STORE.latest_scan()
    rows = (scan["rows"] if scan else [])[:1]
    if not rows:
        raise HTTPException(404, "nothing scanned yet")
    a = Analyst(CFG.get("llm") or {}, STORE)
    if not a.enabled:
        raise HTTPException(409, "llm.enabled is false in config.yaml")
    r = dict(rows[0])
    r["tape_brief"] = tape_brief(r, r.get("tape") or {}, r.get("tape_flags") or [])
    import time as _t
    t0 = _t.time()
    out = await a.review([r])
    return {"symbol": r["symbol"], "seconds": round(_t.time() - t0, 1),
            "verdict": out[0].get("llm")}


async def _explain(symbol: str, lookback: int = 20) -> dict:
    """Why is this one moving? Market share of the move is computed, not judged."""
    df = STORE.bars(symbol, 400)
    if df is None or df.empty:
        return {"error": f"no cached bars for {symbol} - run a scan first"}
    idx = STORE.bars(CFG["ctrader"].get("index_symbol", "US500"), 400)

    news = [{"headline": i["headline"], "source": i["source"],
             "published": i["published"]}
            for i in STORE.feed("news", symbol, 30)]
    if not news:
        # headlines are only stored for scan survivors, so an arbitrary symbol
        # has none. Fetch on demand rather than reporting "unexplained" purely
        # because nobody had collected anything for it.
        cat = CFG.get("catalysts") or {}
        try:
            stories = await NewsFeed(cat, None).fetch(symbol, 30, 20)
            news = [vars(s) for s in stories]
            STORE.add_stories(symbol, "news", news)
        except Exception:
            news = []

    scan = STORE.latest_scan()
    row = next((r for r in ((scan or {}).get("rows") or [])
                if r["symbol"].upper() == symbol.upper()), None)
    earnings = (row or {}).get("earnings")

    # how everything else screened today moved, for percentile context
    moves = []
    for other in (STORE.get("universe") or [])[:200]:
        b = STORE.bars(other, 40)
        if b is not None and len(b) > 21:
            c = b["Close"]
            moves.append(float(100 * (c.iloc[-1] / c.iloc[-21] - 1)))

    lcfg = CFG.get("llm") or {}
    analyst = Analyst({**lcfg, "model": lcfg.get("assistant_model") or lcfg.get("model")},
                      STORE)
    return await explain_move(analyst, symbol, df, idx, news, earnings, moves, lookback)


@app.get("/api/explain/{symbol}")
async def api_explain(symbol: str, lookback: int = 20):
    res = await _explain(symbol, max(5, min(120, lookback)))
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


def _normalise_spec(spec: dict) -> dict:
    """Accept the shapes a model naturally writes.

    The schema calls the identifier "key"; models write "name" or "id" and then
    cannot recover, because the validation error described a malformed key
    rather than a missing field. Meet them halfway instead of burning steps.
    """
    if not isinstance(spec, dict):
        return spec
    spec = dict(spec)
    if "key" not in spec:
        raw = spec.get("name") or spec.get("id") or spec.get("label")
        if raw:
            spec["key"] = re.sub(r"[^a-z0-9]+", "_", str(raw).lower()).strip("_")
            spec.setdefault("label", str(raw))
    for side in ("stop", "target"):
        blk = spec.get(side)
        if isinstance(blk, dict):
            blk = dict(blk)
            if "kind" not in blk and blk.get("type"):
                blk["kind"] = blk.pop("type")
            inner = blk.pop("args", None)
            if isinstance(inner, dict):
                blk.update(inner)          # {"args": {"mult": 2}} -> {"mult": 2}
            spec[side] = blk
    for g in spec.get("gates") or []:
        if isinstance(g, dict) and "args" not in g:
            # models often inline a feature argument next to the comparison
            extra = {k: v for k, v in g.items()
                     if k not in ("feature", "op", "value", "label")}
            if extra:
                g["args"] = extra
                for k in extra:
                    g.pop(k, None)
    return spec


def _chat_deps() -> ChatDeps:
    """Tool implementations for the assistant. Module level so the bench and the
    endpoint use exactly the same wiring."""

    def _setups():
        scan = STORE.latest_scan()
        return [{k: r.get(k) for k in
                 ("symbol", "strategy", "score", "rank", "entry", "stop", "target",
                  "rr", "stop_pct", "units", "atr_pct")}
                | {"verdict": (r.get("llm") or {}).get("verdict"),
                   "earnings": (r.get("earnings") or {}).get("why")}
                for r in ((STORE.latest_scan() or {}).get("rows") or [])]

    def _setup(symbol: str):
        scan = STORE.latest_scan()
        row = next((r for r in ((scan or {}).get("rows") or [])
                    if r["symbol"].upper() == symbol.upper()), None)
        if not row:
            return {"error": f"{symbol} is not in the latest scan",
                    "available": [r["symbol"] for r in ((scan or {}).get("rows") or [])]}
        return {k: row.get(k) for k in
                ("symbol", "strategy", "score", "entry", "stop", "target", "rr",
                 "stop_pct", "units", "gates", "tape", "tape_flags", "earnings",
                 "llm", "zone")}

    def _strategies():
        return {"coded": [{"key": k, "label": v.label, "description": v.description,
                           "needs_regime": v.needs_regime,
                           "enabled": any(x.key == k for x in STRATS),
                           "params": next((x.p for x in STRATS if x.key == k), None)}
                          for k, v in sorted(registry().items())],
                "recipes": [{"key": r["key"], "label": r["label"],
                             "enabled": r["enabled"],
                             "gates": len(r["spec"].get("gates") or [])}
                            for r in STORE.recipes()]}

    def _recipe(key: str):
        r = next((x for x in STORE.recipes() if x["key"] == key), None)
        return r["spec"] if r else {"error": f"no recipe {key!r}"}

    def _context():
        """What this installation actually is, right now."""
        scan = STORE.latest_scan()
        acct = STORE.get("account") or {}
        reg = STORE.get("regime") or {}
        return {
            "broker": acct.get("broker"), "currency": acct.get("currency"),
            "equity": acct.get("equity"),
            "risk_per_trade_pct": PARAMS.risk_per_trade * 100,
            "universe_size": len(STORE.get("universe") or []),
            "universe_note": ("all US-suffixed broker symbols (includes ETFs); "
                              "no sp500.txt filter is in place"
                              if not CFG.get("universe_file") or
                              not (ROOT / str(CFG.get("universe_file"))).exists()
                              else f"filtered by {CFG.get('universe_file')}"),
            "index_symbol": CFG["ctrader"].get("index_symbol"),
            "regime": {"ok": reg.get("ok"), "note": reg.get("note")},
            "enabled_strategies": [x.key for x in STRATS],
            "last_scan": {"id": (scan or {}).get("id"),
                          "finished": (scan or {}).get("finished"),
                          "scanned": (scan or {}).get("scanned"),
                          "hits": (scan or {}).get("hits")},
            # units in the key names: a bare number invites the model to guess,
            # and it reported "quote interval 5 minutes" for a 5-second clock
            "clocks_seconds": {"quote": CLOCKS.quote_interval,
                               "forming_bar": CLOCKS.forming_interval,
                               "full_scan": CLOCKS.scan_interval,
                               "note": "0 means disabled / manual only"},
            "max_chase_pct": CLOCKS.max_chase_pct,
            "review": {"enabled": (CFG.get("llm") or {}).get("enabled"),
                       "model": (CFG.get("llm") or {}).get("model"),
                       "style": (CFG.get("llm") or {}).get("style")},
            "catalysts": {"earnings_provider": "finnhub" if
                          (CFG.get("catalysts") or {}).get("finnhub_key") else "none",
                          "blackout_days": (CFG.get("catalysts") or {}).get("blackout_days")},
            "social_enabled": (CFG.get("social") or {}).get("enabled"),
            "bots_live_allowed": (CFG.get("bots") or {}).get("allow_live"),
            "known_limitations": [
                "cTrader Volume is TICK COUNT, not share volume - absolute turnover "
                "figures are meaningless, only relative comparisons are valid",
                "cTrader only quotes SUBSCRIBED symbols, so some rows show no live price",
                "the backtest uses price and volume only - news, earnings and the "
                "review are not point-in-time and would be lookahead",
                "scores are percentile ranks WITHIN one strategy and are not "
                "comparable across strategies",
            ],
        }

    async def _preview(spec: dict):
        spec = _normalise_spec(spec)
        try:
            validate(spec)
        except SpecError as e:
            v = vocabulary()
            return {"error": str(e),
                    "valid_features": [f["key"] for f in v["features"]],
                    "valid_ops": [o["key"] for o in v["ops"]],
                    "valid_stops": [k["key"] for k in v["stops"]],
                    "valid_targets": [k["key"] for k in v["targets"]],
                    "hint": "Use these exact feature keys. A gate is "
                            '{"feature": <key>, "op": <op>, "value": <number>}. '
                            "Every feature is compared to a NUMBER, so express "
                            '"above the 200-day average" as '
                            '{"feature": "dist_ma_pct", "args": {"length": 200}, '
                            '"op": ">", "value": 0}.'}
        universe = (STORE.get("universe") or [])[:150]
        frames = {x: STORE.bars(x) for x in universe}
        frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
        if not frames:
            return {"error": "no cached bars yet - run a scan first"}
        acct = STORE.get("account") or {}
        ctx = Ctx(regime=STORE.get("regime") or {},
                  equity=acct.get("equity") or CFG.get("fallback_equity", 10000.0),
                  risk_per_trade=PARAMS.risk_per_trade)
        return preview(spec, frames, ctx)

    def _save(spec: dict):
        spec = _normalise_spec(spec)
        try:
            validate(spec)
        except SpecError as e:
            return {"error": str(e)}
        if spec.get("key") in registry():
            return {"error": f"{spec['key']!r} is the key of a coded strategy"}
        STORE.save_recipe(spec, enabled=False)
        reload_strategies()
        return {"saved": spec["key"], "enabled": False,
                "note": "saved disabled - enable it yourself in the Builder tab"}

    return ChatDeps(
        list_setups=_setups, get_setup=_setup, list_strategies=_strategies,
        get_recipe=_recipe, vocabulary=vocabulary, preview_recipe=_preview,
        save_recipe=_save, get_context=_context,
        list_backtests=lambda: STORE.backtests(25),
        get_backtest=lambda i: (STORE.backtest(int(i)) if str(i).isdigit()
                                else {"error": "backtest id must be a number"}),
        search_news=lambda sym: STORE.feed(None, sym or None, 25),
        explain_move=lambda sym: _explain(str(sym)),
    )


@app.post("/api/chat")
async def api_chat(body: dict):
    """Talk to the local model with tools over this project's own data.

    Read-only except for save_recipe, which stores a recipe DISABLED. The
    assistant cannot scan, plot, trade or arm a bot - see assistant.py.
    """
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages is required")
    lcfg = CFG.get("llm") or {}
    # the assistant may run a different model from the review: drafting needs
    # multi-step self-correction, classification needs speed
    analyst = Analyst({**lcfg, "model": lcfg.get("assistant_model") or lcfg.get("model")},
                      STORE)
    if not analyst.enabled:
        raise HTTPException(409, "llm.enabled is false in config.yaml")
    return await run_chat(analyst, messages, _chat_deps())


@app.get("/api/builder/vocabulary")
async def api_vocabulary():
    return vocabulary()


@app.get("/api/builder/recipes")
async def api_recipes():
    return {"recipes": STORE.recipes()}


@app.post("/api/builder/preview")
async def api_preview(spec: dict):
    """Run a candidate spec over cached bars. Does not touch cTrader."""
    try:
        validate(spec)
    except SpecError as e:
        raise HTTPException(400, str(e))
    universe = (STORE.get("universe") or [])[:int(spec.get("_sample") or 150)]
    frames = {s: STORE.bars(s) for s in universe}
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not frames:
        raise HTTPException(409, "no cached bars yet — run a scan first")
    acct = STORE.get("account") or {}
    ctx = Ctx(regime=STORE.get("regime") or {},
              equity=acct.get("equity") or CFG.get("fallback_equity", 10000.0),
              risk_per_trade=PARAMS.risk_per_trade)
    return preview(spec, frames, ctx)


@app.post("/api/builder/save")
async def api_save_recipe(body: dict):
    spec, enabled = body.get("spec") or {}, bool(body.get("enabled"))
    try:
        validate(spec)
    except SpecError as e:
        raise HTTPException(400, str(e))
    if spec["key"] in registry():
        raise HTTPException(409, f"{spec['key']!r} is the key of a coded strategy")
    STORE.save_recipe(spec, enabled)
    reload_strategies()
    return {"saved": spec["key"], "enabled": enabled,
            "active": [s.key for s in STRATS]}


@app.post("/api/builder/enable/{key}")
async def api_enable_recipe(key: str, body: dict | None = None):
    if not STORE.set_recipe_enabled(key, bool((body or {}).get("enabled", True))):
        raise HTTPException(404, f"no saved recipe {key!r}")
    reload_strategies()
    return {"active": [s.key for s in STRATS]}


@app.delete("/api/builder/recipes/{key}")
async def api_delete_recipe(key: str):
    if not STORE.delete_recipe(key):
        raise HTTPException(404, f"no saved recipe {key!r}")
    reload_strategies()
    return {"deleted": key, "active": [s.key for s in STRATS]}


# ---------------------------------------------------------------------------
# backtest
#
# Two long jobs live here and both are deliberately single-flight. The walk is
# CPU-bound pandas, so it goes on a thread and reports through BT_STATE rather
# than awaiting anything; the backfill is 700-odd MCP round-trips, so it stays
# on the loop and is bounded by the same concurrency limit the scan uses.
# ---------------------------------------------------------------------------

BT_STATE: dict = {"running": False, "done": 0, "total": 0, "symbol": "",
                  "error": None, "id": None, "phase": ""}
BF_STATE: dict = {"running": False, "done": 0, "total": 0, "symbol": "",
                  "error": None, "added": 0}


def _strategy_by_key(key: str):
    """Coded strategies and saved recipes both, whether or not they are enabled
    for scanning. You should be able to backtest something before you turn it
    on, which is rather the point."""
    reg = registry()
    if key in reg:
        return reg[key]()
    rec = next((r for r in STORE.recipes() if r["key"] == key), None)
    if rec:
        return Composite(rec["spec"])
    raise HTTPException(404, f"no strategy {key!r}")


@app.get("/api/backtest/meta")
async def api_backtest_meta():
    """Everything the tab needs to draw itself before a run."""
    strats = [{"key": k, "label": v.label, "direction": v.direction,
               "needs_regime": v.needs_regime, "min_bars": v({}).min_bars,
               "built": False} for k, v in sorted(registry().items())]
    for r in STORE.recipes():
        try:
            c = Composite(r["spec"])
        except SpecError:
            continue
        strats.append({"key": c.key, "label": c.label, "direction": c.direction,
                       "needs_regime": c.needs_regime, "min_bars": c.min_bars,
                       "built": True})
    cov = STORE.bar_coverage()
    return {"strategies": strats, "coverage": cov,
            "defaults": vars(BTConfig()),
            "index_symbol": CFG["ctrader"].get("index_symbol", "US500"),
            "state": BT_STATE, "backfill": BF_STATE,
            "runs": STORE.backtests(25)}


@app.get("/api/backtest/state")
async def api_backtest_state():
    return {"backtest": BT_STATE, "backfill": BF_STATE}


@app.post("/api/backtest/run")
async def api_backtest_run(body: dict | None = None):
    if BT_STATE["running"]:
        raise HTTPException(409, "a backtest is already running")
    body = body or {}
    cfg = BTConfig(**{k: v for k, v in body.items() if k in BTConfig().__dict__})
    strat = _strategy_by_key(body.get("strategy") or cfg.strategy)
    cfg.strategy = strat.key
    idx = CFG["ctrader"].get("index_symbol", "US500")

    def _progress(done: int, total: int, symbol: str) -> None:
        BT_STATE.update(done=done, total=total, symbol=symbol)

    async def _pump() -> None:
        """The walk cannot await, so a second task publishes its progress."""
        while BT_STATE["running"]:
            await _emit({"phase": "backtest", "done": BT_STATE["done"],
                         "total": BT_STATE["total"], "symbol": BT_STATE["symbol"]})
            await asyncio.sleep(1.0)

    async def _go() -> None:
        BT_STATE.update(running=True, done=0, total=0, symbol="", error=None,
                        id=None, phase="walking")
        pump = asyncio.create_task(_pump())
        try:
            res = await asyncio.to_thread(run_backtest, STORE, strat, cfg, idx, _progress)
            bt_id = STORE.save_backtest(strat.key, vars(cfg), res)
            BT_STATE.update(id=bt_id, phase="done")
            await _emit({"phase": "backtest_done", "id": bt_id,
                         "trades": (res.get("signal_stats") or {}).get("trades", 0)})
        except Exception as e:
            BT_STATE.update(error=str(e)[:300], phase="error")
            await _emit({"phase": "backtest_error", "message": str(e)[:200]})
        finally:
            BT_STATE["running"] = False
            pump.cancel()

    asyncio.create_task(_go())
    return {"started": True, "strategy": strat.key, "config": vars(cfg)}


@app.get("/api/backtest/runs")
async def api_backtest_runs(limit: int = 25):
    return {"runs": STORE.backtests(min(limit, 100))}


@app.get("/api/backtest/result/{bt_id}")
async def api_backtest_result(bt_id: int):
    r = STORE.backtest(bt_id)
    if not r:
        raise HTTPException(404, f"no backtest {bt_id}")
    return r


@app.get("/api/backtest/latest")
async def api_backtest_latest():
    runs = STORE.backtests(1)
    if not runs:
        raise HTTPException(404, "nothing backtested yet")
    return STORE.backtest(runs[0]["id"])


@app.delete("/api/backtest/{bt_id}")
async def api_backtest_delete(bt_id: int):
    if not STORE.delete_backtest(bt_id):
        raise HTTPException(404, f"no backtest {bt_id}")
    return {"deleted": bt_id}


@app.post("/api/backfill")
async def api_backfill(body: dict | None = None):
    """Deepen the bar cache so there is something to walk.

    A normal scan keeps ~285 bars per symbol, which is barely more than the
    200DMA warmup — about twenty testable days. cTrader will serve up to 1000
    daily bars, roughly four years, and that is the difference between a
    backtest and a rounding error.
    """
    if BF_STATE["running"]:
        raise HTTPException(409, "a backfill is already running")
    if STATE["scanning"]:
        raise HTTPException(409, "a scan is running — let it finish first")
    body = body or {}
    want = max(100, min(1000, int(body.get("bars") or 1000)))
    only_short = bool(body.get("only_short", True))

    async def _go() -> None:
        BF_STATE.update(running=True, done=0, total=0, symbol="", error=None, added=0)
        try:
            mcp = await get_mcp()
            idx = CFG["ctrader"].get("index_symbol", "US500")
            syms = sorted(set(STORE.get("universe") or []) | {idx})
            if not syms:
                syms = sorted(set(STORE.symbols_with_bars(1)) | {idx})
            if only_short:
                syms = [s for s in syms if STORE.count(s) < want * 0.9]
            BF_STATE["total"] = len(syms)
            sem = asyncio.Semaphore(int(CFG["ctrader"].get("max_concurrency", 4)))

            async def one(sym: str) -> None:
                async with sem:
                    try:
                        df = await mcp.candles(sym, want)
                        if df is not None:
                            BF_STATE["added"] += STORE.upsert(sym, df)
                    except Exception:
                        pass          # one dead symbol must not stop the backfill
                    finally:
                        BF_STATE["done"] += 1
                        BF_STATE["symbol"] = sym
                        if BF_STATE["done"] % 10 == 0:
                            await _emit({"phase": "backfill", "done": BF_STATE["done"],
                                         "total": BF_STATE["total"], "symbol": sym})

            await asyncio.gather(*(one(s) for s in syms))
            await _emit({"phase": "backfill_done", "symbols": len(syms),
                         "rows": BF_STATE["added"]})
        except Exception as e:
            BF_STATE["error"] = str(e)[:300]
            await _emit({"phase": "backfill_error", "message": str(e)[:200]})
            await drop_mcp()
        finally:
            BF_STATE["running"] = False

    asyncio.create_task(_go())
    return {"started": True, "bars": want}


@app.get("/api/bots")
async def api_bots():
    return {"bots": STORE.bots(), "exits": EXITS, "caps": DEFAULT_CAPS,
            "strategies": sorted(_known_strategies()),
            "kill_switch": bool(STORE.get("kill_switch")),
            "allow_live": bool((CFG.get("bots") or {}).get("allow_live", False))}


@app.post("/api/bots/save")
async def api_save_bot(body: dict):
    spec, enabled = body.get("spec") or {}, bool(body.get("enabled"))
    try:
        validate_bot(spec, _known_strategies())
    except SpecError as e:
        raise HTTPException(400, str(e))
    if spec.get("mode") == "live":
        raise HTTPException(400, "save as paper first, then arm it explicitly")
    STORE.save_bot(spec, enabled)
    BOT_STATE.pop(spec["key"], None)
    return {"saved": spec["key"], "enabled": enabled, "mode": "paper"}


@app.post("/api/bots/{key}/arm")
async def api_arm_bot(key: str, body: dict):
    """Switch a bot to live. Deliberately awkward."""
    if not (CFG.get("bots") or {}).get("allow_live", False):
        raise HTTPException(403, "live trading is off. Set bots.allow_live in config.yaml.")
    rec = next((b for b in STORE.bots() if b["key"] == key), None)
    if not rec:
        raise HTTPException(404, f"no bot {key!r}")
    if body.get("confirm") != key:
        raise HTTPException(400, f"type the bot key {key!r} to confirm")
    if "order_place" not in (OWNER.mcp.resolved if OWNER.mcp else {}):
        raise HTTPException(503, "cTrader exposes no order tool that this can use")
    STORE.set_bot(key, armed=True, mode="live")
    STORE.log_bot(key, [{"event": "armed", "mode": "live"}])
    return {"armed": key}


@app.post("/api/bots/{key}/enable")
async def api_enable_bot(key: str, body: dict | None = None):
    if not STORE.set_bot(key, enabled=bool((body or {}).get("enabled", True))):
        raise HTTPException(404, f"no bot {key!r}")
    return {"ok": True}


@app.delete("/api/bots/{key}")
async def api_delete_bot(key: str):
    if not STORE.delete_bot(key):
        raise HTTPException(404, f"no bot {key!r}")
    BOT_STATE.pop(key, None)
    return {"deleted": key}


@app.get("/api/bots/{key}/events")
async def api_bot_events(key: str, limit: int = 80):
    return {"events": STORE.bot_events(key, min(limit, 300))}


@app.get("/api/bots/{key}/cbot")
async def api_export_cbot(key: str):
    """Generate the cTrader cBot source for this bot."""
    rec = next((b for b in STORE.bots() if b["key"] == key), None)
    if not rec:
        raise HTTPException(404, f"no bot {key!r}")
    src_key = rec["spec"]["entry_strategy"]
    recipe = next((r for r in STORE.recipes() if r["key"] == src_key), None)
    if not recipe:
        raise HTTPException(400, "cBot export needs a builder recipe as the entry "
                                 "source; coded strategies are already Python")
    try:
        code = generate_cbot(rec["spec"], recipe["spec"])
    except CodegenError as e:
        raise HTTPException(422, str(e))
    return {"filename": f"{rec['key']}.cs", "code": code}


@app.post("/api/kill")
async def api_kill(body: dict | None = None):
    on = bool((body or {}).get("on", True))
    STORE.put("kill_switch", on)
    STORE.log_bot("*", [{"event": "kill_switch", "on": on}])
    return {"kill_switch": on}


@app.post("/api/clocks")
async def api_clocks(body: dict):
    """Change any of the three intervals at runtime. Seconds; 0 disables."""
    for k in ("quote_interval", "forming_interval", "scan_interval", "max_chase_pct"):
        if k in body:
            try:
                setattr(CLOCKS, k, max(0.0, float(body[k])))
            except (TypeError, ValueError):
                raise HTTPException(400, f"{k} must be a number")
    return vars(CLOCKS)


@app.get("/api/events")
async def api_events():
    async def gen():
        yield f"data: {json.dumps({'phase': 'hello'})}\n\n"
        while True:
            try:
                evt = await asyncio.wait_for(QUEUE.get(), timeout=20)
                yield f"data: {json.dumps(evt)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/symbol/{symbol}")
async def api_symbol(symbol: str):
    df = STORE.bars(symbol, 400)
    if df.empty:
        raise HTTPException(404, f"no cached bars for {symbol}")
    acct = STORE.get("account") or {}
    equity = acct.get("equity") or acct.get("balance") or CFG.get("fallback_equity", 10000.0)
    # pick the strategy that produced this row, else the first enabled one
    scan = STORE.latest_scan()
    owner = next((r.get("strategy") for r in (scan["rows"] if scan else [])
                  if r["symbol"] == symbol), None)
    strat = next((s for s in STRATS if s.key == owner), None) or (STRATS[0] if STRATS else None)
    if strat is None:
        raise HTTPException(503, "no strategies are enabled")
    detail = strat.evaluate(symbol, df, Ctx(
        regime=STORE.get("regime") or {}, equity=equity,
        risk_per_trade=PARAMS.risk_per_trade))
    tp = tape_digest(detail or {}, df)
    if detail:
        detail["tape"] = tp
        detail["tape_flags"] = tape_flags(detail, tp)
    tail = df.iloc[-160:]
    return {"detail": detail, "bars": [
        {"t": ts.date().isoformat(), "o": r.Open, "h": r.High, "l": r.Low, "c": r.Close}
        for ts, r in tail.iterrows()]}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="run one scan and exit")
    ap.add_argument("--backtest", metavar="STRATEGY",
                    help="walk one strategy over the cached bars and exit")
    ap.add_argument("--symbols", type=int, default=120,
                    help="backtest sample size; 0 = every cached symbol")
    ap.add_argument("--port", type=int, default=CFG.get("port", 8790))
    a = ap.parse_args()

    if a.backtest:
        # The Windows console is cp1252 and this output has arrows and dashes.
        with contextlib.suppress(AttributeError, ValueError):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        strat = _strategy_by_key(a.backtest)
        cfg = BTConfig(strategy=strat.key, symbols=a.symbols)
        res = run_backtest(STORE, strat, cfg,
                           CFG["ctrader"].get("index_symbol", "US500"),
                           lambda d, t, s: print(f"\r{d}/{t} {s:<14}", end="",
                                                 file=sys.stderr))
        print(file=sys.stderr)
        s, p = res["signal_stats"], res["portfolio"]
        print(f"{res['strategy']['label']}  {res['span']['from']} → {res['span']['to']} "
              f"({res['span']['years']}y, {res['coverage']['symbols_tested']} symbols)")
        print(f"  signals {res['coverage']['signals']}  trades {s.get('trades', 0)}  "
              f"win {s.get('win_rate', 0)}%  avgR {s.get('avg_r', 0)}  "
              f"PF {s.get('profit_factor')}  t {s.get('t_stat')}")
        print(f"  portfolio {p.get('return_pct')}% vs index "
              f"{(res.get('benchmark') or {}).get('return_pct')}%  "
              f"maxDD {p.get('max_dd_pct')}%")
        for w in res["warnings"]:
            print(f"  ! {w}")
        STORE.save_backtest(strat.key, vars(cfg), res)
        return

    if a.scan:
        res = asyncio.run(run_scan())
        scan = STORE.latest_scan()
        for r in (scan["rows"] if scan else [])[:25]:
            print(f"{r['rank']:>3}  {r['score']:>5.1f}  {r['symbol']:<12} "
                  f"entry {r['entry']:<10} stop {r['stop']:<10} "
                  f"{r['stop_pct']:>5.2f}%  {r['rr']:>4.2f}R  units {r['units']}")
        print(json.dumps(res))
        return
    uvicorn.run(app, host=CFG.get("host", "127.0.0.1"), port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
