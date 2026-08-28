# cTrader Screener

A self-hosted stock screener that gets its market data from **cTrader Desktop
over MCP** no broker API registration, no market data subscription, no cloud
service. If cTrader runs on your machine, this can screen every symbol your
broker offers.

It finds technical setups, filters them through an earnings blackout and a news
review run by a **local LLM**, draws the survivors back onto your cTrader
charts, and backtests the whole thing on cached bars.

Everything runs on your own hardware. Nothing leaves the machine except optional
news and earnings lookups.

---

## What it does

**Screens** every symbol your broker offers against pluggable strategies. Three
ship: a 50DMA pullback, a volatility-squeeze breakout, and an RSI(2) oversold
snapback. Each produces named pass/fail **gates** carrying the measured value,
so you can always see *why* something did or didn't qualify.

**Filters** survivors through a deterministic earnings blackout, a computed
"tape" digest (distance from the 52-week high, volatility relative to the
universe, gap frequency), and an optional LLM review that reads headlines and
argues *against* the setup rather than for it.

**Explains moves.** Ask why a stock is dipping and it first computes how much of
the move the index already accounts for, beta-adjusted. Only the genuinely
company-specific remainder reaches a model, along with a dated headline
timeline. "Unexplained" is a supported and frequently correct answer.

**Draws** setups back onto your cTrader charts entry, stop and target as a
native risk/reward block, the pullback window as a rectangle, plus a label. It
records the object IDs it creates and never touches anything you drew yourself.

**Backtests** by walking the gates forward on price and volume only, with
next-open fills and stop-wins-ties on ambiguous bars.

**Builds strategies without code.** Compose gates from a fixed vocabulary in the
browser, test against cached bars, and see which gate is actually filtering.

**Talks.** A chat dock on every page, backed by your local model, with tools
over your own scans, strategies, backtests and headlines.

---

## Requirements

- **Windows** with **cTrader Desktop** running and "AI Agent Connect" enabled
  (Settings → AI Agent Connect → Local MCP). This is the data source.
- **Python 3.11+** (3.13 works)
- Optional: **[Ollama](https://ollama.com)** for the LLM review and chat
- Optional: a free **[Finnhub](https://finnhub.io)** key for earnings dates and
  company news

Without Ollama the screener still works you lose the review and the chat.
Without Finnhub you lose the earnings blackout and company news.

---

## Setup

```bash
git clone <this-repo> screener
cd screener

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy config.example.yaml config.yaml
```

### 1. Point it at cTrader

cTrader Desktop serves MCP as a **local HTTP server**, normally on
`127.0.0.1:9876`. Confirm the address in its AI Agent Connect screen and set
`ctrader.remote.url` in `config.yaml` to match.

### 2. Verify the tool names do not skip this

```bash
python ctrader_mcp.py --dump-tools
```

This prints every tool cTrader exposes with its argument names, plus what the
client auto-matched. **Tool names differ between cTrader versions.** The shipped
config pins about twenty of them against cTrader Desktop 2.0.0; if yours
differs, the dump tells you exactly what to change.

Auto discovery alone is not safe here. On one version it matched the object
deletion tool to `clear_chart_objects`, which erases *every* drawing on a chart.
The pinned config exists for that reason.

### 3. Run

```bash
python server.py            # dashboard on http://127.0.0.1:8790
python server.py --scan     # one headless scan, prints the table, exits
```

The first scan backfills roughly 280 daily bars per symbol and takes a few
minutes. After that it pulls ~10 bars per symbol and reads the rest from SQLite.

### 4. Optional: the local model

```bash
ollama pull qwen2.5:14b-instruct
```

Set `llm.enabled: true`, then check `GET /api/llm/health` and run one real
verdict with `POST /api/llm/test`.

A 12GB card runs a 14B at Q4 comfortably. Read **Model notes** below before
choosing something else the obvious alternatives have non-obvious failure
modes.

---

## The dashboard

| tab | what it is |
|---|---|
| Screener | today's setups, live prices, severity markers; click a row for gate detail |
| News | every headline collected, newest first |
| Analysis | every LLM review as a card |
| Important | severity ≥ 2 only — empty is the normal state |
| Builder | compose a strategy, test it on cached bars, save it |
| Backtest | walk the gates forward and see whether they paid |
| Bots | paper bots with hard caps, or export a cTrader cBot |

The **assistant dock** is reachable from every tab (bottom-right, or `Ctrl+K`).

---

## Things that will bite you

All of these are measured, not theoretical.

**cTrader's `Volume` is tick count, not share volume.** AAPL reports ~23,000
where real volume is ~50 million. Every absolute turnover figure is therefore
meaningless and only relative comparisons are valid. The code treats it that
way don't add a dollar-volume threshold without reading `tape.py` first.

**cTrader only quotes subscribed symbols.** A symbol with no open chart may
answer "unsubscribed" and show no live price. Plotting it opens a chart, which
subscribes it.

**cTrader's MCP server does not tolerate concurrent requests.** Eight parallel
calls deadlocked for 90+ seconds; the same eight sequentially took 2.4 seconds.
`ctrader.max_concurrency` must stay at 1.

**cTrader's `get_news` ignores the symbol argument.** It returns a general
FX/crypto wire, so every stock would get identical headlines. Company news needs
Finnhub, and `catalysts.prefer` is set accordingly.

**Scores are percentile ranks within one strategy.** A breakout at 90 and a
pullback at 90 are each "best of what that strategy found today". They are not
comparable to each other.

---

## Model notes

The review is narrow classification, not reasoning, and model choice matters
more than parameter count:

- **qwen2.5:14b-instruct** the default. Zero JSON errors, ~2.5s per setup.
- **qwen2.5:7b-instruct** flagged setups for the *absence* of news, which the
  prompt explicitly forbids, and misattributed headlines more often.
- **qwen3:14b / qwen3:8b** hybrid reasoning models. Ollama's
  OpenAI-compatible endpoint puts their chain of thought in a separate
  `reasoning` field and returns an **empty** `content`, so every reply looks
  like a failure. `llm.py` detects this and retries against the native
  `/api/chat` with `think: false`. Even fixed, qwen3:14b ran ~6x slower for no
  measurable gain on this task.

Two review styles are available. `holistic` sends one call per setup and is
faster. `per_headline` classifies each headline separately and computes the
verdict in Python, which makes misattribution structurally impossible at the
cost of more calls.

---

## Safety

The screener **places no orders**. Bots are paper only unless you set
`bots.allow_live`, *and* arm each bot individually by typing its key, *and* an
order tool is resolved on the session. Saving a bot directly as live is refused.

The chat assistant reads everything but writes exactly one thing: a strategy
recipe, saved **disabled**. It cannot scan, plot, trade or arm a bot. It also
cannot save a recipe without previewing it first that is enforced in code, not
requested in the prompt.

Chart plotting only ever deletes object IDs it created itself.

---

## Layout

```
server.py          FastAPI service, SSE, background clocks
ctrader_mcp.py     MCP client, tool discovery, response normalizers
store.py           SQLite: bars, scans, headlines, analyses, recipes, bots
strategies/        one file per strategy, auto-discovered
spec.py            spec-driven strategy used by the builder
builder.py         declarative spec -> Strategy, with preview diagnostics
tape.py            computed tape digest + structural flags
catalysts.py       earnings calendar + news, deterministic blackout
social.py          retail chatter, volume baseline, risk input only
llm.py             the review: adversarial, severity tiers, ablation modes
headlines.py       per-headline classification, deterministic aggregation
explain.py         why a stock is moving: market share first, then text
assistant.py       chat loop with tools over your own data
quotes.py          live price polling + forming-bar clock
plot.py            draws setups back onto cTrader charts
bots.py            bot spec, paper engine with caps, cBot codegen
backtest.py        walk-forward gate replay and trade simulation
static/index.html  the dashboard
```

`DESIGN.md` has the long form engineering record: what broke, what the
measurements showed, and why several things are deliberately not what they look
like they should be.

---

## Status

It works and is tested against a live cTrader install.
It ships with no proven edge and the backtest exists precisely so you can
find that out for your own strategies.

Trading involves risk of loss. Nothing here is financial advice.

## License

MIT
