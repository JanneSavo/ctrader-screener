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
llm.py — the reading layer.

Scope is deliberately narrow. The model reads headlines and an earnings line
and answers one question: is there something in the text that makes this
technically-valid setup a bad idea right now?

What it is not allowed to do:
  - produce or adjust any number (entry, stop, target, size, dates)
  - promote a setup. Its verdict can only hold or downgrade.
  - run on anything that failed the technical gates

That last constraint is also the cost control: it sees the ~10 survivors,
not the 500 candidates.

Ablation modes, because a judgment layer that cannot be falsified is decoration:
  normal    real headlines for the real symbol
  shuffled  another symbol's headlines, same setup — if verdicts don't change,
            the model is reacting to the setup shape, not the news
  blind     no headlines at all, earnings line only — the floor to beat

Set mode in config, run the same universe three ways, compare.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re

SYSTEM = """You review swing-trade setups that have already passed a technical screen.

Your role is adversarial. The screen has already made the case FOR this trade —
that is what the gates are. Your job is the case AGAINST, and if there isn't a
good one, to say so plainly.

The screen looks for uptrending US large-caps that pulled back into their 50-day
moving average and closed back above it. Entry, stop and target are already fixed
by a rules engine. You do not set them.

Your only job: read the supplied headlines and earnings line, and judge whether
there is a text-visible reason not to take this trade over the next 2-4 weeks.

You may also be given retail social posts. Treat these as a risk input only.

Reasons to flag:
- pending or just-announced M&A, which caps upside and breaks technical behaviour
- regulatory action, investigation, fraud allegation, accounting issue
- guidance cut, major customer loss, executive departure under a cloud
- the bounce is clearly news-driven in a way that is likely to reverse
- litigation or index-removal risk
- promotion patterns in the social posts: coordinated or near-identical wording,
  price targets with no reasoning, urgency language, obvious ramping
- a spike in chatter volume that the headlines do not explain

Not reasons to flag:
- ordinary volatility, analyst rating changes, routine product news
- general market commentary, macro takes, listicles
- the absence of news. Quiet is normal and fine.
- retail being bullish. Enthusiasm is not evidence and is never a reason to
  raise confidence in the setup. You cannot upgrade anything.

On the tape numbers you are given:

You do NOT see a chart. You get a computed snapshot and a list of conditions the
screener already tested. Reason about those numbers and nothing else. Do not
estimate a level, do not name support or resistance, do not produce a price
target, and do not restate a figure that was not given to you.

You have no edge in predicting direction from numbers — a gradient-boosted model
beats you at that and the gates already encode it. What you can do is notice
when a setup is technically valid and structurally a bad idea anyway:

- the gates measure a pullback in an uptrend; check whether this is actually a
  bounce inside a decline, which looks identical for ten bars
- check whether the position will be tradeable: turnover, price level, ATR
  versus the universe, gap frequency. A valid signal on an untradeable name is
  not a trade.
- check whether the tape and the text disagree — quiet news against a violent
  chart, or good news against collapsing participation
- when something looks unusually cheap or unusually oversold, the burden is to
  explain why the market is pricing it that way. "The multiple is low" is a
  restatement of the price, not a reason. If the headlines do not explain it,
  say that the explanation is missing rather than assuming there isn't one.

Write the strongest bear case you can from the supplied facts. If the honest
bear case is weak, say that — an unconvincing case against is useful signal and
inventing one is not.

Rules:
- CHECK THE SUBJECT OF EVERY HEADLINE. A headline only counts if it is about the
  company you are reviewing. Feeds routinely include sector round-ups, ETF
  commentary and articles about competitors or holdings. If a headline is
  primarily about a different company, ignore it completely - do not let it
  influence the verdict, and never attribute its facts to this symbol. If the
  headlines are mostly about other companies, say so and return "clear".
- Never invent or restate a date. If the earnings line says unknown, treat it as unknown.
- Never suggest different price levels, sizes or targets.
- If the headlines contain nothing decision-relevant, say so and return "clear".
- Base everything on the supplied text. You have no other information about this company.

Respond with ONLY a JSON object, no prose, no markdown fence:
{"verdict":"clear|caution|avoid",
 "confidence":0.0-1.0,
 "reasons":["short phrase citing what in the text drove this"],
 "catalyst":"none|earnings|m&a|regulatory|guidance|litigation|promotion|other",
 "social_note":"one line on the chatter, or empty if there was none",
 "technical_note":"one line on the tape, or empty",
 "bear_case":"the strongest argument against taking this, from the supplied facts only"}"""


class Analyst:
    """Talks to whatever model server you point it at.

    Speaks the OpenAI chat-completions shape, which is what Ollama, LM Studio,
    llama.cpp's server and vLLM all expose. The name refers to the request
    format, not to a vendor - nothing leaves your machine.
    """

    def __init__(self, cfg: dict, store):
        self.cfg = cfg
        self.store = store
        self.model = cfg.get("model", "qwen2.5:7b-instruct")
        self.base_url = str(cfg.get("base_url", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.mode = cfg.get("mode", "normal")
        self.enabled = bool(cfg.get("enabled"))
        self._client = None

    def _client_or_none(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=float(self.cfg.get("timeout_s", 120)),
                headers={"Authorization": f"Bearer {self.cfg.get('api_key') or 'local'}"})
        return self._client

    async def _complete(self, brief: str, system: str | None = None) -> str:
        """One completion, returning raw text."""
        system = system or SYSTEM
        client = self._client_or_none()
        # Qwen3 and other hybrid-reasoning models emit a <think> block first.
        # On a long prompt that reasoning consumes the whole token budget and
        # the JSON never arrives - 4 of 8 replies came back truncated. The
        # "/no_think" suffix turns it off; harmless on models that ignore it.
        user = brief + ("\n\n/no_think" if self.cfg.get("no_think", True) else "")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "max_tokens": int(self.cfg.get("max_tokens", 900)),
        }
        if self.cfg.get("json_mode", True):
            body["response_format"] = {"type": "json_object"}
        r = await client.post("/chat/completions", json=body)
        if r.status_code == 400 and "response_format" in body:
            body.pop("response_format")      # older servers reject it
            r = await client.post("/chat/completions", json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        txt = msg.get("content") or ""
        if txt.strip():
            return txt
        # Ollama's OpenAI shim puts a hybrid model's chain-of-thought in a
        # separate "reasoning" field and leaves content EMPTY. Qwen3 burned
        # 900 tokens thinking and returned nothing, 4-8 times out of 8, and
        # "/no_think" in the prompt does not suppress it. The native endpoint
        # takes think=false, which does. Retry there before giving up.
        if msg.get("reasoning") or msg.get("reasoning_content"):
            native = str(client.base_url).rstrip("/")
            native = native[:-3] if native.endswith("/v1") else native
            nb = {"model": self.model, "think": False, "stream": False,
                  "format": "json" if self.cfg.get("json_mode", True) else None,
                  "options": {"temperature": float(self.cfg.get("temperature", 0.0)),
                              "num_predict": int(self.cfg.get("max_tokens", 900))},
                  "messages": body["messages"]}
            nb = {k: v for k, v in nb.items() if v is not None}
            nr = await client.post(f"{native}/api/chat", json=nb)
            nr.raise_for_status()
            return nr.json().get("message", {}).get("content") or ""
        return txt

    async def health(self) -> dict:
        """Is the endpoint up, and does it have the model we ask for?"""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self.base_url}/models")
                r.raise_for_status()
                names = [m.get("id") for m in (r.json().get("data") or [])]
            return {"base_url": self.base_url,
                    "model": self.model, "ok": True, "models": names,
                    "model_present": any(str(n).startswith(self.model.split(":")[0])
                                         for n in names)}
        except Exception as e:
            return {"base_url": self.base_url,
                    "model": self.model, "ok": False, "error": str(e)[:200]}

    # -- prompt ------------------------------------------------------------

    @staticmethod
    def _brief(row: dict, stories: list[dict], social: dict | None = None,
               tape_block: str = "") -> str:
        e = row.get("earnings") or {}
        lines = [
            f"Symbol: {row['symbol']}",
            f"Setup: closed {row['entry']} back above its 50-day average, "
            f"{row.get('depth', 0) * 100:.1f}% off its recent high, "
            f"bounce bar on {row.get('vol_ratio', 1)}x average volume",
            f"Earnings: {e.get('why', 'unknown')}",
            "",
            "Headlines from the last week:",
        ]
        if stories:
            lines += [f"- [{s['published'][:10]}] {s['headline']} ({s['source']})"
                      for s in stories]
        else:
            lines.append("- (none returned)")

        if social:
            lines += ["", f"Retail chatter: {social.get('summary', '')}"]
            if social.get("unexplained"):
                lines.append("NOTE: chatter is spiking with nothing in the headlines to explain it.")
            posts = social.get("posts") or []
            if posts:
                lines.append("Recent posts:")
                lines += [f"- [{p.get('source','')}] {str(p.get('text',''))[:180]}"
                          for p in posts[:10]]
        if tape_block:
            lines.append(tape_block)
        return "\n".join(lines)

    # -- calling -----------------------------------------------------------

    async def review(self, rows: list[dict]) -> list[dict]:
        if self.enabled and self.cfg.get("style", "holistic") == "per_headline":
            return await self._review_per_headline(rows)
        if not self.enabled:
            for r in rows:
                r["llm"] = {"verdict": "off", "reasons": [], "mode": "disabled"}
            return rows

        client = self._client_or_none()
        if client is None:
            for r in rows:
                r["llm"] = {"verdict": "error",
                            "reasons": ["model server unavailable"],
                            "mode": self.mode}
            return rows

        pool = [r.get("news") or [] for r in rows]
        sem = asyncio.Semaphore(int(self.cfg.get("concurrency", 4)))

        async def one(i: int, r: dict):
            if self.mode == "blind":
                stories = []
            elif self.mode == "shuffled" and len(rows) > 1:
                stories = pool[random.choice([j for j in range(len(rows)) if j != i])]
            else:
                stories = r.get("news") or []

            social = None if self.mode == "blind" else r.get("social")
            tape_block = "" if self.mode == "blind" else (r.get("tape_brief") or "")
            brief = self._brief(r, stories, social, tape_block)
            ck = "llm:" + hashlib.sha256(
                f"{self.model}|{self.mode}|{brief}".encode()).hexdigest()[:24]
            cached = self.store.get(ck, max_age_s=self.cfg.get("cache_ttl_s", 21600))
            if cached:
                r["llm"] = cached | {"cached": True}
                return

            async with sem:
                try:
                    txt = await self._complete(brief)
                    out = _parse(txt)
                    out["mode"] = self.mode
                    out["model"] = self.model
                    out["severity"] = severity(out, r)
                    out["sources"] = {"news": len(stories),
                                      "social": len((social or {}).get("posts") or []),
                                      "tape_flags": len(r.get("tape_flags") or [])}
                    self.store.put(ck, out)
                    r["llm"] = out
                except Exception as e:
                    r["llm"] = {"verdict": "error", "confidence": 0.0,
                                "reasons": [str(e)[:160]], "catalyst": "none",
                                "mode": self.mode}

        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))

        # apply: downgrade only, never promote
        drop = self.cfg.get("drop_on_avoid", False)
        kept = []
        for r in rows:
            v = (r.get("llm") or {}).get("verdict")
            if v == "avoid":
                r["flag"] = "avoid"
                if drop:
                    r["pass"] = False
                    r["failed"] = list(r.get("failed", [])) + ["News"]
                    continue
            elif v == "caution":
                r["flag"] = "caution"
            kept.append(r)
        return kept


    # -- per-headline path -------------------------------------------------

    async def _review_per_headline(self, rows: list[dict]) -> list[dict]:
        """Classify each headline on its own, then compute the verdict in Python.

        Misattribution becomes structurally impossible: a call only ever sees one
        company and one headline, and only headlines labelled as being about that
        company can move the verdict.
        """
        from headlines import aggregate, classify

        sem = asyncio.Semaphore(int(self.cfg.get("concurrency", 1)))
        pool = [r.get("news") or [] for r in rows]

        async def one(i: int, r: dict):
            if self.mode == "blind":
                news = []
            elif self.mode == "shuffled" and len(rows) > 1:
                news = pool[random.choice([j for j in range(len(rows)) if j != i])]
            else:
                news = r.get("news") or []

            ck = "hl:" + hashlib.sha256(
                f"{self.model}|{self.mode}|{r['symbol']}|"
                f"{json.dumps([h.get('headline') for h in news])}".encode()).hexdigest()[:24]
            cached = self.store.get(ck, max_age_s=self.cfg.get("cache_ttl_s", 21600))
            if cached:
                r["llm"] = cached | {"cached": True}
                return

            async with sem:
                labels = await classify(self, r["symbol"],
                                        r.get("company") or r["symbol"].split(".")[0],
                                        news, concurrency=1)
            out = aggregate(r["symbol"], labels, r)
            out["mode"] = self.mode
            out["model"] = self.model
            out["style"] = "per_headline"
            out["sources"] = {"news": len(news), "social": 0,
                              "tape_flags": len(r.get("tape_flags") or [])}
            self.store.put(ck, out)
            r["llm"] = out

        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))

        drop = self.cfg.get("drop_on_avoid", False)
        kept = []
        for r in rows:
            v = (r.get("llm") or {}).get("verdict")
            if v == "avoid":
                r["flag"] = "avoid"
                if drop:
                    r["pass"] = False
                    r["failed"] = list(r.get("failed", [])) + ["News"]
                    continue
            elif v == "caution":
                r["flag"] = "caution"
            kept.append(r)
        return kept


def severity(verdict: dict, row: dict) -> int:
    """0 routine · 1 note · 2 important · 3 blocking.

    The Important tab is severity >= 2. Kept as one function so the tab, the
    row marker and the badge can never disagree about what matters.
    """
    v, conf = verdict.get("verdict"), verdict.get("confidence") or 0
    if v == "avoid":
        return 3
    if v == "caution":
        return 2 if conf >= 0.6 else 1
    soc = row.get("social") or {}
    if soc.get("unexplained"):
        return 2
    # Objective tape conditions, computed not judged. Two heavy flags is a
    # structurally awkward trade even when every gate passed and the text is clean.
    heavy = sum(f.get("weight", 1) for f in (row.get("tape_flags") or []))
    if heavy >= 4:
        return 2
    if soc.get("spiking") or heavy >= 2:
        return 1
    return 0


SEVERITY_LABEL = {0: "routine", 1: "note", 2: "important", 3: "blocking"}


def _parse(txt: str) -> dict:
    """Model output is JSON or it is nothing. No regex archaeology on prose."""
    # strip any reasoning block a hybrid model emitted before the answer
    s = re.sub(r"<think>.*?</think>", "", txt, flags=re.S | re.I).strip()
    s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return {"verdict": "error", "confidence": 0.0,
                "reasons": ["non-JSON response"], "catalyst": "none"}
    v = str(d.get("verdict", "")).lower()
    if v not in ("clear", "caution", "avoid"):
        v = "error"
    try:
        conf = min(1.0, max(0.0, float(d.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    reasons = [str(x)[:200] for x in (d.get("reasons") or [])][:4]
    return {"verdict": v, "confidence": round(conf, 2), "reasons": reasons,
            "catalyst": str(d.get("catalyst", "none")).lower()[:20],
            "social_note": str(d.get("social_note", ""))[:200],
            "technical_note": str(d.get("technical_note", ""))[:250],
            "bear_case": str(d.get("bear_case", ""))[:600]}

