"""
assistant.py — talk to the screener.

A chat loop over the same local model, with tools that read this project's own
data and can draft strategies. It does not use the provider's native
function-calling: that varies between Ollama, LM Studio and vLLM, and the models
that work here are already reliable at emitting strict JSON. So the protocol is
plain JSON, one step at a time, which works on any server the Analyst can reach.

Deliberately limited. The assistant can READ everything and WRITE exactly one
kind of thing: a builder recipe, saved disabled. It cannot place orders, arm a
bot, draw on your charts, run a scan, or enable a strategy. Those are all one
click away from real money or a mutated workspace, and a model that has been
wrong about which company a headline belonged to should not hold that pen.

Recipes it saves land disabled, so a strategy it invents cannot start producing
signals until you look at it and turn it on yourself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

MAX_STEPS = 6

SYSTEM = """You are the assistant built into a self-hosted stock screener that
the user wrote. You help them understand their setups, strategies and backtests,
and you can draft new strategies.

HOW THIS SYSTEM WORKS - you must reason from this, not from generic trading advice:

The screener pulls daily bars for every US-listed symbol the user's broker
offers, from cTrader over MCP. Each enabled STRATEGY evaluates every symbol and
produces GATES - named pass/fail checks with the measured value. A setup passes
only if every gate passes. Entry, stop, target and position size are computed by
the strategy, never by a model.

After the gates, survivors go through: an EARNINGS blackout (deterministic, a
setup reporting within N days is rejected), a NEWS fetch, a TAPE digest
(computed structural facts like distance from the 52-week high and relative
volatility, with objective flags), and an LLM REVIEW that reads headlines and
argues AGAINST the setup. The review can only add caution, never promote.

Scores are PERCENTILE RANKS WITHIN ONE STRATEGY. A breakout scoring 90 and a
pullback scoring 90 are each "best of what that strategy found today" and are
NOT comparable to each other.

Strategies come in two kinds: coded ones (Python files) and RECIPES built from a
fixed vocabulary of features and operators. You can only draft recipes. Every
gate is a feature compared to a NUMBER, for example
{"feature": "dist_ma_pct", "args": {"length": 200}, "op": ">", "value": 0}
means "price is above its 200-day moving average".

The BACKTEST replays gates forward on price and volume only. It deliberately
excludes news, earnings and the review, because those are not point-in-time and
replaying them would be lookahead.

Call get_context() when you need to know how THIS installation is configured:
equity, risk per trade, universe size, enabled strategies, regime, last scan,
and a list of known limitations of the data. Do that before answering anything
that depends on the user's setup rather than on general logic.

You work by calling tools. Every reply MUST be a single JSON object, nothing
else, in one of two shapes:

  {"tool": "<name>", "args": {...}}          to look something up
  {"reply": "<your answer to the user>"}     when you are ready to answer

Available tools:

  get_context()                  how this installation is configured, plus
                                 known limitations of the data
  list_setups()                  today's passing setups with scores and levels
  get_setup(symbol)              full gate-by-gate detail, tape, earnings, review
  list_strategies()              coded strategies and saved recipes, with params
  get_recipe(key)                the full spec of one saved recipe
  vocabulary()                   features and operators available for recipes
  preview_recipe(spec)           test a draft recipe on cached bars. Returns how
                                 many symbols pass, which gate rejects the most,
                                 and how often it fired historically
  save_recipe(spec)              save a recipe. It is saved DISABLED.
  list_backtests()               past backtest runs and their headline numbers
  get_backtest(id)               full result of one backtest
  search_news(symbol)            headlines collected for a symbol
  explain_move(symbol)           why a stock is moving: measures how much of
                                 the move the index already explains, then
                                 matches the remainder against a dated
                                 headline timeline. Use this for any 'why is
                                 X falling' question - never answer that from
                                 headlines alone

Rules:

- Look things up before answering. Do not guess a number that a tool can give you.
- Before EVER calling save_recipe, you must have called preview_recipe on that
  exact spec and shown the user the hit rate, the rejection counts and the
  historical firing rate. A recipe that fires twice a year, or that passes a
  quarter of the universe, is not worth saving and you should say so.
- Recipes are saved disabled. Tell the user they must enable it themselves.
- You cannot place orders, run scans, plot charts or start bots. If asked, say so.
- The screener's own numbers are the source of truth. If you disagree with a
  gate result, say why, but do not restate the number differently.
- Be concise. The user is a developer and knows the domain.
- Never invent a symbol, a strategy key or a feature name. Call vocabulary()
  BEFORE drafting any recipe - guessing feature names wastes your step
  budget and you only get a few.
- If a tool returns an error, immediately call it again with the fix in the
  same reply. Saying you will retry is not retrying; the user sees nothing."""


@dataclass
class Deps:
    """Callables the server injects. Keeps this module free of server imports."""
    list_setups: Callable[[], Any]
    get_setup: Callable[[str], Any]
    list_strategies: Callable[[], Any]
    get_recipe: Callable[[str], Any]
    vocabulary: Callable[[], Any]
    preview_recipe: Callable[[dict], Any]
    save_recipe: Callable[[dict], Any]
    list_backtests: Callable[[], Any]
    get_backtest: Callable[[str], Any]
    search_news: Callable[[str], Any]
    get_context: Callable[[], Any]
    explain_move: Callable[[str], Any]


def _parse(txt: str) -> dict:
    s = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S | re.I).strip()
    s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    # not JSON: treat the whole thing as the answer rather than failing the turn
    return {"reply": s[:2000] or "(empty response)"}


def _truncate(obj: Any, limit: int = 6000) -> str:
    """Tool results go back into the prompt, so they must not blow the context."""
    s = json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + f'... (truncated, {len(s)} chars)'


async def _run_tool(name: str, args: dict, deps: Deps, state: dict) -> Any:
    args = args or {}
    if name == "list_setups":
        return deps.list_setups()
    if name == "get_setup":
        return deps.get_setup(str(args.get("symbol", "")))
    if name == "list_strategies":
        return deps.list_strategies()
    if name == "get_recipe":
        return deps.get_recipe(str(args.get("key", "")))
    if name == "vocabulary":
        return deps.vocabulary()
    if name == "list_backtests":
        return deps.list_backtests()
    if name == "get_backtest":
        return deps.get_backtest(str(args.get("id", "")))
    if name == "search_news":
        return deps.search_news(str(args.get("symbol", "")))
    if name == "get_context":
        return deps.get_context()
    if name == "explain_move":
        return await _maybe_await(deps.explain_move(str(args.get("symbol", ""))))

    if name == "preview_recipe":
        spec = args.get("spec") or args
        res = await _maybe_await(deps.preview_recipe(spec))
        # remember what was previewed, so save_recipe cannot skip this step
        state["previewed"] = json.dumps(spec.get("gates"), sort_keys=True, default=str)
        return res

    if name == "save_recipe":
        spec = args.get("spec") or args
        sig = json.dumps(spec.get("gates"), sort_keys=True, default=str)
        if state.get("previewed") != sig:
            return {"refused": "preview_recipe must be run on this exact spec first, "
                                "and its results shown to the user, before saving."}
        res = await _maybe_await(deps.save_recipe(spec))
        state["previewed"] = None
        return res

    return {"error": f"no such tool: {name}"}


async def _maybe_await(v):
    if hasattr(v, "__await__"):
        return await v
    return v


async def chat(analyst, messages: list[dict], deps: Deps) -> dict:
    """One user turn. Loops tool calls until the model answers or runs out.

    `messages` is the whole visible conversation ([{role, content}, ...]).
    Tool results are appended as extra user turns, so any chat model works
    without native tool-calling support.
    """
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}" for m in convo)

    state: dict = {}
    steps: list[dict] = []
    scratch = ""

    for _ in range(MAX_STEPS):
        prompt = transcript + scratch + "\n\nRespond with one JSON object."
        raw = await analyst._complete(prompt, system=SYSTEM)
        out = _parse(raw)

        if "reply" in out and "tool" not in out:
            return {"reply": str(out["reply"]), "steps": steps}

        tool = str(out.get("tool", "")).strip()
        args = out.get("args") or {}
        if not tool:
            return {"reply": str(out.get("reply") or raw)[:2000], "steps": steps}

        try:
            result = await _run_tool(tool, args, deps, state)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"[:300]}

        steps.append({"tool": tool, "args": args, "result_preview": _truncate(result, 600)})
        failed = isinstance(result, dict) and ("error" in result or "refused" in result)
        scratch += (f"\n\nYou called {tool}({json.dumps(args, default=str)[:300]}).\n"
                    f"Result: {_truncate(result)}\n")
        if failed:
            # Models narrate the retry instead of performing it: "Let's try that
            # again with the corrected name" - and the turn ends there. Say plainly
            # that describing the next step is not a step.
            scratch += ("That call FAILED. Fix it and CALL THE TOOL AGAIN NOW in "
                        "this reply. Do not describe what you are about to do - "
                        "announcing a retry is not a retry, and the user sees "
                        "nothing. Emit the corrected tool call as JSON, or "
                        '{"reply": ...} if you cannot fix it.')
        else:
            scratch += ("Now respond with one JSON object: either another tool call, "
                        'or {"reply": ...} to answer the user.')

    return {"reply": "I ran out of steps before answering. Ask something narrower.",
            "steps": steps}
