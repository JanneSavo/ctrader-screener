"""
store.py — local bar cache.

A 500-symbol scan is 500 MCP round-trips. Doing that from cold every run is
slow and hammers cTrader, so bars land in SQLite and each run only asks for
the tail. First sight of a symbol backfills; after that it's ~5 bars.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL, ts TEXT NOT NULL,
  o REAL, h REAL, l REAL, c REAL, v REAL,
  PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started TEXT, finished TEXT, regime_ok INTEGER,
  regime_note TEXT, scanned INTEGER, hits INTEGER, rows TEXT
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS stories (
  id TEXT PRIMARY KEY, symbol TEXT, kind TEXT, headline TEXT,
  source TEXT, url TEXT, published TEXT, seen TEXT
);
CREATE INDEX IF NOT EXISTS ix_stories ON stories(published DESC);
CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, ts TEXT,
  verdict TEXT, severity INTEGER, confidence REAL, catalyst TEXT,
  reasons TEXT, social_note TEXT, sources TEXT, mode TEXT, model TEXT,
  technical_note TEXT DEFAULT '', bear_case TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_analyses ON analyses(ts DESC);
CREATE TABLE IF NOT EXISTS bots (
  key TEXT PRIMARY KEY, label TEXT, spec TEXT, enabled INTEGER,
  mode TEXT, armed INTEGER DEFAULT 0, state TEXT, created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS bot_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, bot TEXT, ts TEXT, event TEXT,
  symbol TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_bot_events ON bot_events(ts DESC);
CREATE TABLE IF NOT EXISTS recipes (
  key TEXT PRIMARY KEY, label TEXT, spec TEXT, enabled INTEGER,
  created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS backtests (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, strategy TEXT,
  config TEXT, result TEXT
);
CREATE INDEX IF NOT EXISTS ix_backtests ON backtests(ts DESC);
"""


class Store:
    def __init__(self, path: str = "screener.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- bars --------------------------------------------------------------

    def last_ts(self, symbol: str) -> str | None:
        r = self.db.execute("SELECT MAX(ts) FROM bars WHERE symbol=?", (symbol,)).fetchone()
        return r[0]

    def count(self, symbol: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM bars WHERE symbol=?", (symbol,)).fetchone()[0]

    def upsert(self, symbol: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        rows = [(symbol, ts.isoformat(), float(r.Open), float(r.High),
                 float(r.Low), float(r.Close), float(r.Volume))
                for ts, r in df.iterrows()]
        self.db.executemany(
            "INSERT INTO bars(symbol,ts,o,h,l,c,v) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol,ts) DO UPDATE SET o=excluded.o,h=excluded.h,"
            "l=excluded.l,c=excluded.c,v=excluded.v", rows)
        self.db.commit()
        return len(rows)

    def bars(self, symbol: str, limit: int = 600) -> pd.DataFrame:
        q = ("SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? "
             "ORDER BY ts DESC LIMIT ?")
        df = pd.read_sql_query(q, self.db, params=(symbol, limit))
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").set_index("ts")
        return df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                  "c": "Close", "v": "Volume"})

    def symbols_with_bars(self, min_count: int = 1) -> list[str]:
        """Every symbol the cache holds enough history for. The backtest
        universe is this, not the broker's list — you can only test what you
        have bars for."""
        return [r[0] for r in self.db.execute(
            "SELECT symbol FROM bars GROUP BY symbol HAVING COUNT(*) >= ?",
            (int(min_count),)).fetchall()]

    def bar_coverage(self) -> dict:
        """How deep the cache actually goes. The backtest tab shows this before
        you run anything, because a 20-bar window is not a backtest."""
        r = self.db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(ts), MAX(ts) FROM bars").fetchone()
        med = self.db.execute(
            "SELECT AVG(n) FROM (SELECT COUNT(*) n FROM bars GROUP BY symbol)").fetchone()
        return {"bars": r[0] or 0, "symbols": r[1] or 0,
                "first": (r[2] or "")[:10], "last": (r[3] or "")[:10],
                "avg_per_symbol": round(med[0] or 0)}

    # -- backtests ---------------------------------------------------------

    def save_backtest(self, strategy: str, config: dict, result: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO backtests(ts,strategy,config,result) VALUES (?,?,?,?)",
            (_now(), strategy, json.dumps(_clean(config)), json.dumps(_clean(result))))
        self.db.commit()
        return cur.lastrowid

    def backtests(self, limit: int = 25) -> list[dict]:
        """Index only — the full result blob is big, so the list view gets a
        summary and you fetch one run by id."""
        rs = self.db.execute(
            "SELECT id,ts,strategy,config,result FROM backtests ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        out = []
        for i, ts, strat, cfg, res in rs:
            r = json.loads(res)
            s = r.get("signal_stats") or {}
            out.append({"id": i, "ts": ts, "strategy": strat,
                        "config": json.loads(cfg),
                        "trades": s.get("trades", 0),
                        "avg_r": s.get("avg_r"),
                        "win_rate": s.get("win_rate"),
                        "return_pct": (r.get("portfolio") or {}).get("return_pct"),
                        "span": r.get("span") or {}})
        return out

    def backtest(self, bt_id: int) -> dict | None:
        r = self.db.execute("SELECT id,ts,strategy,result FROM backtests WHERE id=?",
                            (bt_id,)).fetchone()
        if not r:
            return None
        return {"id": r[0], "ts": r[1], "strategy": r[2], **json.loads(r[3])}

    def delete_backtest(self, bt_id: int) -> bool:
        cur = self.db.execute("DELETE FROM backtests WHERE id=?", (bt_id,))
        self.db.commit()
        return cur.rowcount > 0

    # -- scans -------------------------------------------------------------

    def save_scan(self, regime_ok: bool, note: str, scanned: int, rows: list[dict],
                  started: str) -> int:
        cur = self.db.execute(
            "INSERT INTO scans(started,finished,regime_ok,regime_note,scanned,hits,rows) "
            "VALUES (?,?,?,?,?,?,?)",
            (started, _now(), int(regime_ok), note, scanned, len(rows),
             json.dumps(_clean(rows))))
        self.db.commit()
        return cur.lastrowid

    def latest_scan(self) -> dict | None:
        r = self.db.execute(
            "SELECT id,started,finished,regime_ok,regime_note,scanned,hits,rows "
            "FROM scans ORDER BY id DESC LIMIT 1").fetchone()
        if not r:
            return None
        return {"id": r[0], "started": r[1], "finished": r[2], "regime_ok": bool(r[3]),
                "regime_note": r[4], "scanned": r[5], "hits": r[6], "rows": json.loads(r[7])}

    def scan_history(self, n: int = 30) -> list[dict]:
        rs = self.db.execute(
            "SELECT id,finished,regime_ok,scanned,hits FROM scans ORDER BY id DESC LIMIT ?",
            (n,)).fetchall()
        return [{"id": a, "finished": b, "regime_ok": bool(c), "scanned": d, "hits": e}
                for a, b, c, d, e in rs]

    # -- feed + analysis history -------------------------------------------

    def add_stories(self, symbol: str, kind: str, items: list[dict]) -> int:
        rows = []
        for it in items:
            head = it.get("headline") or it.get("text") or ""
            if not head:
                continue
            sid = hashlib.sha1(f"{symbol}|{kind}|{head}".encode()).hexdigest()[:20]
            rows.append((sid, symbol, kind, head[:400],
                         it.get("source") or "—", it.get("url") or "",
                         it.get("published") or "", _now()))
        if rows:
            self.db.executemany(
                "INSERT OR IGNORE INTO stories(id,symbol,kind,headline,source,url,published,seen) "
                "VALUES (?,?,?,?,?,?,?,?)", rows)
            self.db.commit()
        return len(rows)

    def feed(self, kind: str | None = None, symbol: str | None = None,
             limit: int = 120) -> list[dict]:
        q = "SELECT symbol,kind,headline,source,url,published FROM stories WHERE 1=1"
        args: list = []
        if kind:
            q += " AND kind=?"; args.append(kind)
        if symbol:
            q += " AND symbol=?"; args.append(symbol)
        q += " ORDER BY published DESC, seen DESC LIMIT ?"; args.append(limit)
        return [dict(zip(("symbol", "kind", "headline", "source", "url", "published"), r))
                for r in self.db.execute(q, args).fetchall()]

    def add_analysis(self, symbol: str, a: dict) -> None:
        self.db.execute(
            "INSERT INTO analyses(symbol,ts,verdict,severity,confidence,catalyst,"
            "reasons,social_note,sources,mode,model,technical_note,bear_case) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, _now(), a.get("verdict"), int(a.get("severity") or 0),
             float(a.get("confidence") or 0), a.get("catalyst"),
             json.dumps(a.get("reasons") or []), a.get("social_note") or "",
             json.dumps(a.get("sources") or {}), a.get("mode"), a.get("model"),
             a.get("technical_note") or "", a.get("bear_case") or ""))
        self.db.commit()

    def analyses(self, min_severity: int = 0, limit: int = 100) -> list[dict]:
        rs = self.db.execute(
            "SELECT symbol,ts,verdict,severity,confidence,catalyst,reasons,"
            "social_note,sources,mode,model,technical_note,bear_case "
            "FROM analyses WHERE severity>=? "
            "ORDER BY severity DESC, ts DESC LIMIT ?", (min_severity, limit)).fetchall()
        out = []
        for r in rs:
            d = dict(zip(("symbol", "ts", "verdict", "severity", "confidence", "catalyst",
                          "reasons", "social_note", "sources", "mode", "model",
                          "technical_note", "bear_case"), r))
            d["reasons"] = json.loads(d["reasons"] or "[]")
            d["sources"] = json.loads(d["sources"] or "{}")
            out.append(d)
        return out

    # -- saved strategy recipes --------------------------------------------

    def save_recipe(self, spec: dict, enabled: bool = False) -> None:
        self.db.execute(
            "INSERT INTO recipes(key,label,spec,enabled,created,updated) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "label=excluded.label, spec=excluded.spec, updated=excluded.updated",
            (spec["key"], spec.get("label") or spec["key"], json.dumps(spec),
             int(enabled), _now(), _now()))
        self.db.commit()

    def recipes(self, enabled_only: bool = False) -> list[dict]:
        q = "SELECT key,label,spec,enabled,updated FROM recipes"
        if enabled_only:
            q += " WHERE enabled=1"
        return [{"key": k, "label": l, "spec": json.loads(sp), "enabled": bool(e),
                 "updated": u} for k, l, sp, e, u in self.db.execute(q).fetchall()]

    def set_recipe_enabled(self, key: str, enabled: bool) -> bool:
        cur = self.db.execute("UPDATE recipes SET enabled=?, updated=? WHERE key=?",
                              (int(enabled), _now(), key))
        self.db.commit()
        return cur.rowcount > 0

    def delete_recipe(self, key: str) -> bool:
        cur = self.db.execute("DELETE FROM recipes WHERE key=?", (key,))
        self.db.commit()
        return cur.rowcount > 0

    # -- bots ---------------------------------------------------------------

    def save_bot(self, spec: dict, enabled: bool = False) -> None:
        self.db.execute(
            "INSERT INTO bots(key,label,spec,enabled,mode,armed,state,created,updated) "
            "VALUES (?,?,?,?,?,0,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "label=excluded.label, spec=excluded.spec, mode=excluded.mode, "
            "updated=excluded.updated",
            (spec["key"], spec.get("label") or spec["key"], json.dumps(spec),
             int(enabled), spec.get("mode", "paper"), json.dumps({}), _now(), _now()))
        self.db.commit()

    def bots(self, enabled_only: bool = False) -> list[dict]:
        q = "SELECT key,label,spec,enabled,mode,armed,state,updated FROM bots"
        if enabled_only:
            q += " WHERE enabled=1"
        return [{"key": k, "label": l, "spec": json.loads(sp), "enabled": bool(e),
                 "mode": m, "armed": bool(a), "state": json.loads(st or "{}"),
                 "updated": u}
                for k, l, sp, e, m, a, st, u in self.db.execute(q).fetchall()]

    def set_bot(self, key: str, **fields) -> bool:
        allowed = {"enabled", "armed", "mode", "state"}
        sets, args = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            sets.append(f"{k}=?")
            args.append(json.dumps(v) if k == "state" else
                        (int(v) if k in ("enabled", "armed") else v))
        if not sets:
            return False
        args += [_now(), key]
        cur = self.db.execute(f"UPDATE bots SET {','.join(sets)}, updated=? WHERE key=?", args)
        self.db.commit()
        return cur.rowcount > 0

    def delete_bot(self, key: str) -> bool:
        cur = self.db.execute("DELETE FROM bots WHERE key=?", (key,))
        self.db.execute("DELETE FROM bot_events WHERE bot=?", (key,))
        self.db.commit()
        return cur.rowcount > 0

    def log_bot(self, bot: str, events: list[dict]) -> None:
        if not events:
            return
        self.db.executemany(
            "INSERT INTO bot_events(bot,ts,event,symbol,payload) VALUES (?,?,?,?,?)",
            [(bot, _now(), e.get("event", "?"), e.get("symbol", ""), json.dumps(e))
             for e in events])
        self.db.commit()

    def bot_events(self, bot: str | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT bot,ts,event,symbol,payload FROM bot_events"
        args: list = []
        if bot:
            q += " WHERE bot=?"
            args.append(bot)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [{"bot": b, "ts": t, "event": e, "symbol": s, **json.loads(p)}
                for b, t, e, s, p in self.db.execute(q, args).fetchall()]

    # -- kv ----------------------------------------------------------------

    def put(self, k: str, v) -> None:
        self.db.execute("INSERT INTO kv(k,v,ts) VALUES(?,?,?) "
                        "ON CONFLICT(k) DO UPDATE SET v=excluded.v, ts=excluded.ts",
                        (k, json.dumps(v), _now()))
        self.db.commit()

    def get(self, k: str, max_age_s: float | None = None):
        r = self.db.execute("SELECT v,ts FROM kv WHERE k=?", (k,)).fetchone()
        if not r:
            return None
        if max_age_s is not None:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(r[1])).total_seconds()
            if age > max_age_s:
                return None
        return json.loads(r[0])


def _clean(o):
    """NaN and Infinity are legal for json.dumps but illegal over HTTP.

    They survive into SQLite silently and then blow up the API response, which
    is a very confusing way to see an empty dashboard. Scrub at the boundary.
    """
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
