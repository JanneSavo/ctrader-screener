"""
bots.py — declarative bots, two backends, no code written by hand.

A bot spec is JSON: where entries come from, how positions are sized, how they
are managed, and the caps that stop it. From one spec you get:

  RUNTIME   a paper engine that runs in this service against live quotes.
            Simulated fills, full event log, equity curve.

  CODEGEN   a cTrader cBot (.cs) implementing the same rules, which you compile
            in cTrader Automate. This is the one that should ever touch real
            money: it runs in-platform with proper tick handling and keeps
            working when this web service is not.

The screener finds candidates; a bot acts on them. That is a real change in
blast radius, so:

  - paper is the default and the only mode enabled out of the box
  - live requires allow_live in config AND an explicit arm step per bot
  - every bot carries hard caps: max concurrent positions, daily loss cap,
    per-position cap, session window
  - the kill switch is global, immediate, and flattens nothing automatically
    (it stops new entries; closing existing positions stays your decision)

None of this makes a bot safe. It makes it bounded. Whether the rules make
money is a separate question that only a backtest answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone

from builder import FEATURES, OPS, SpecError

# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

EXITS = {
    "stop_target": "Fixed stop and target from the entry signal",
    "break_even": "Move the stop to entry once price reaches N R",
    "trail_atr": "Trail the stop N ATR behind the high water mark",
    "time_exit": "Close after N bars regardless",
    "target_partial": "Take part of the position off at N R",
}

DEFAULT_CAPS = {
    "max_concurrent": 3,
    "max_daily_loss_pct": 2.0,
    "max_position_pct": 20.0,
    "session_start": "09:35",
    "session_end": "15:55",
    "max_new_per_day": 2,
}


def validate_bot(spec: dict, known_strategies: set[str]) -> dict:
    key = str(spec.get("key") or "").strip()
    if not key or not key.replace("_", "").isalnum():
        raise SpecError("key must be alphanumeric with underscores")
    src = spec.get("entry_strategy")
    if src not in known_strategies:
        raise SpecError(f"entry_strategy {src!r} is not a strategy or saved recipe. "
                        f"Known: {', '.join(sorted(known_strategies)) or 'none'}")
    if spec.get("mode", "paper") not in ("paper", "live"):
        raise SpecError("mode must be paper or live")
    for rule in spec.get("exits") or []:
        if rule.get("kind") not in EXITS:
            raise SpecError(f"unknown exit rule {rule.get('kind')!r}")
        v = rule.get("value")
        if not isinstance(v, (int, float)) or v <= 0:
            raise SpecError(f"exit {rule['kind']}: value must be a positive number")
    caps = {**DEFAULT_CAPS, **(spec.get("caps") or {})}
    for k in ("max_concurrent", "max_new_per_day"):
        if int(caps[k]) < 1:
            raise SpecError(f"caps.{k} must be at least 1")
    for k in ("max_daily_loss_pct", "max_position_pct"):
        if not 0 < float(caps[k]) <= 100:
            raise SpecError(f"caps.{k} must be between 0 and 100")
    for k in ("session_start", "session_end"):
        try:
            _parse_hhmm(caps[k])
        except ValueError:
            raise SpecError(f"caps.{k} must be HH:MM")
    spec["caps"] = caps
    if not 0 < float(spec.get("risk_per_trade_pct", 1.0)) <= 5:
        raise SpecError("risk_per_trade_pct must be between 0 and 5")
    return spec


def _parse_hhmm(s: str) -> time:
    h, m = str(s).split(":")
    return time(int(h), int(m))


# ---------------------------------------------------------------------------
# paper engine
# ---------------------------------------------------------------------------


@dataclass
class Position:
    bot: str
    symbol: str
    strategy: str
    opened: str
    entry: float
    stop: float
    target: float
    units: float
    high_water: float = 0.0
    bars_held: int = 0
    partial_done: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {**vars(self)}


@dataclass
class BotState:
    key: str
    equity_start: float
    realised: float = 0.0
    opened_today: int = 0
    day: str = ""
    halted: str = ""          # non-empty = why it stopped
    positions: list[Position] = field(default_factory=list)


class PaperEngine:
    """Simulated fills. Never sends an order anywhere."""

    def __init__(self, spec: dict, store, equity: float):
        self.spec = spec
        self.store = store
        self.equity = equity
        self.caps = spec["caps"]

    # -- entries -----------------------------------------------------------

    def considers(self, row: dict) -> bool:
        if row.get("strategy") != self.spec["entry_strategy"]:
            return False
        scope = self.spec.get("symbols") or []
        return not scope or row["symbol"] in scope

    def can_open(self, st: BotState, now: datetime) -> tuple[bool, str]:
        if st.halted:
            return False, st.halted
        if self.store.get("kill_switch"):
            return False, "kill switch is on"
        if len(st.positions) >= int(self.caps["max_concurrent"]):
            return False, f"at the {self.caps['max_concurrent']}-position cap"
        if st.opened_today >= int(self.caps["max_new_per_day"]):
            return False, f"opened {st.opened_today} today, cap is {self.caps['max_new_per_day']}"
        loss_cap = -abs(self.equity * float(self.caps["max_daily_loss_pct"]) / 100)
        if st.realised <= loss_cap:
            return False, f"daily loss cap hit ({st.realised:.2f})"
        t = now.time()
        if not (_parse_hhmm(self.caps["session_start"]) <= t <= _parse_hhmm(self.caps["session_end"])):
            return False, "outside the session window"
        return True, ""

    def size(self, entry: float, stop: float) -> float:
        risk_amt = self.equity * float(self.spec.get("risk_per_trade_pct", 1.0)) / 100
        units = risk_amt / max(entry - stop, 1e-9)
        cap_units = self.equity * float(self.caps["max_position_pct"]) / 100 / entry
        return round(min(units, cap_units), 2)

    def open(self, st: BotState, row: dict, now: datetime) -> dict | None:
        ok, why = self.can_open(st, now)
        if not ok:
            return {"event": "skipped", "symbol": row["symbol"], "why": why}
        if any(p.symbol == row["symbol"] for p in st.positions):
            return {"event": "skipped", "symbol": row["symbol"], "why": "already holding it"}
        p = Position(bot=st.key, symbol=row["symbol"], strategy=row["strategy"],
                     opened=now.isoformat(), entry=row["entry"], stop=row["stop"],
                     target=row["target"], units=self.size(row["entry"], row["stop"]),
                     high_water=row["entry"])
        st.positions.append(p)
        st.opened_today += 1
        return {"event": "opened", "symbol": p.symbol, "entry": p.entry,
                "stop": p.stop, "target": p.target, "units": p.units,
                "risk": round(p.units * (p.entry - p.stop), 2)}

    # -- management --------------------------------------------------------

    def manage(self, st: BotState, prices: dict[str, float], now: datetime) -> list[dict]:
        events = []
        rules = {r["kind"]: float(r["value"]) for r in (self.spec.get("exits") or [])}
        for p in list(st.positions):
            px = prices.get(p.symbol)
            if px is None:
                continue
            p.high_water = max(p.high_water, px)
            r_now = (px - p.entry) / max(p.entry - p.stop, 1e-9)

            if "break_even" in rules and r_now >= rules["break_even"] and p.stop < p.entry:
                p.stop = p.entry
                events.append({"event": "stop_to_be", "symbol": p.symbol, "stop": p.stop})

            if "trail_atr" in rules and self.spec.get("atr_hint"):
                trail = p.high_water - rules["trail_atr"] * float(self.spec["atr_hint"])
                if trail > p.stop:
                    p.stop = round(trail, 4)
                    events.append({"event": "trailed", "symbol": p.symbol, "stop": p.stop})

            if "target_partial" in rules and not p.partial_done and r_now >= rules["target_partial"]:
                half = round(p.units / 2, 2)
                pnl = half * (px - p.entry)
                p.units = round(p.units - half, 2)
                p.partial_done = True
                st.realised += pnl
                events.append({"event": "partial", "symbol": p.symbol, "units": half,
                               "price": px, "pnl": round(pnl, 2)})

            exit_why = None
            if px <= p.stop:
                exit_why = "stop"
            elif px >= p.target:
                exit_why = "target"
            elif "time_exit" in rules and p.bars_held >= rules["time_exit"]:
                exit_why = "time"

            if exit_why:
                pnl = p.units * (px - p.entry)
                st.realised += pnl
                st.positions.remove(p)
                events.append({"event": "closed", "symbol": p.symbol, "why": exit_why,
                               "price": px, "pnl": round(pnl, 2),
                               "r": round(r_now, 2), "bars": p.bars_held})

        loss_cap = -abs(self.equity * float(self.caps["max_daily_loss_pct"]) / 100)
        if st.realised <= loss_cap and not st.halted:
            st.halted = f"daily loss cap hit ({st.realised:.2f})"
            events.append({"event": "halted", "why": st.halted})
        return events

    def roll_day(self, st: BotState, now: datetime) -> None:
        today = now.date().isoformat()
        if st.day != today:
            st.day, st.opened_today, st.realised = today, 0, 0.0
            if st.halted.startswith("daily loss"):
                st.halted = ""


# ---------------------------------------------------------------------------
# cBot code generation
# ---------------------------------------------------------------------------

# Feature -> C# expression, evaluated at index `i`. Anything not here cannot be
# generated, and codegen says so rather than emitting something that lies.
CS_FEATURES: dict[str, str] = {
    "rsi": "_rsi{id}.Result[i]",
    "dist_ma_atr": "(Bars.ClosePrices[i] - _ma{id}.Result[i]) / _atr{id}.Result[i]",
    "dist_ma_pct": "100.0 * (Bars.ClosePrices[i] / _ma{id}.Result[i] - 1.0)",
    "low_dist_ma_atr": "LowestDistToMa(i, {lookback}, _ma{id}, _atr{id})",
    "pct_from_high": "100.0 * (1.0 - Bars.ClosePrices[i] / HighestHigh(i, {length}))",
    "pct_from_low": "100.0 * (Bars.ClosePrices[i] / LowestLow(i, {length}) - 1.0)",
    "frac_above_ma": "FracAboveMa(i, {window}, _ma{id})",
    "ma_slope_pct": "100.0 * (_ma{id}.Result[i] / _ma{id}.Result[i - {window}] - 1.0)",
    "breakout_atr": "(Bars.ClosePrices[i] - HighestHigh(i - 1, {length})) / _atr{id}.Result[i]",
    "squeeze": "AtrMean(i, {short}, _atr{id}) / AtrMean(i, {long}, _atr{id})",
    "vol_ratio": "Bars.TickVolumes[i] / VolMean(i, {length})",
    "close_pos": "ClosePosition(i)",
    "down_days": "DownDays(i, {window})",
    "atr_pct": "100.0 * _atr{id}.Result[i] / Bars.ClosePrices[i]",
    "gap_pct": "100.0 * (Bars.OpenPrices[i] / Bars.ClosePrices[i - 1] - 1.0)",
    "ret_pct": "100.0 * (Bars.ClosePrices[i] / Bars.ClosePrices[i - {window}] - 1.0)",
    "turnover": "Bars.ClosePrices[i] * VolMean(i, {length})",
}

CS_OPS = {">=": ">=", "<=": "<=", ">": ">", "<": "<"}


class CodegenError(SpecError):
    pass


def generate_cbot(bot: dict, strategy_spec: dict) -> str:
    """Emit a cTrader cBot implementing this bot's entry gates and exits.

    NOTE: this is generated source, not compiled source. It has never been
    through the cTrader compiler here — build it in Automate and expect to fix
    an API detail or two. Read it before you run it.
    """
    gates, decls, ids = [], [], {}
    for n, g in enumerate(strategy_spec.get("gates") or []):
        feat = g["feature"]
        if feat not in CS_FEATURES:
            raise CodegenError(f"feature {feat!r} cannot be generated as C# yet")
        if g["op"] not in CS_OPS:
            raise CodegenError(f"operator {g['op']!r} cannot be generated ('between' "
                               f"needs two gates instead)")
        args = {**FEATURES[feat]["args"], **(g.get("args") or {})}
        gid = n
        ids[gid] = args
        expr = CS_FEATURES[feat].format(id=gid, **{k: int(v) for k, v in args.items()})
        gates.append(f"            && {expr} {CS_OPS[g['op']]} {float(g['value'])}")
        if "length" in args or "window" in args:
            ln = int(args.get("length", args.get("window", 50)))
            decls.append(f"        _ma{gid} = Indicators.SimpleMovingAverage(Bars.ClosePrices, {ln});")
        decls.append(f"        _atr{gid} = Indicators.AverageTrueRange({int(args.get('atr_len', 14))}, "
                     f"MovingAverageType.Exponential);")
        if feat == "rsi":
            decls.append(f"        _rsi{gid} = Indicators.RelativeStrengthIndex("
                         f"Bars.ClosePrices, {int(args.get('length', 14))});")

    fields = []
    for gid in ids:
        fields += [f"    private SimpleMovingAverage _ma{gid};",
                   f"    private AverageTrueRange _atr{gid};",
                   f"    private RelativeStrengthIndex _rsi{gid};"]

    rules = {r["kind"]: float(r["value"]) for r in (bot.get("exits") or [])}
    caps = bot["caps"]
    stop_spec = strategy_spec.get("stop") or {"kind": "atr", "mult": 2.0}
    target_spec = strategy_spec.get("target") or {"kind": "r", "mult": 2.0}

    return _CBOT_TEMPLATE.format(
        cls=_pascal(bot["key"]),
        label=bot.get("label", bot["key"]),
        source=strategy_spec.get("label", strategy_spec.get("key", "")),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        fields="\n".join(dict.fromkeys(fields)),
        decls="\n".join(dict.fromkeys(decls)),
        gates="\n".join(gates) or "            && true",
        risk_pct=float(bot.get("risk_per_trade_pct", 1.0)),
        stop_mult=float(stop_spec.get("mult", stop_spec.get("pct", 2.0))),
        stop_is_pct="true" if stop_spec.get("kind") == "pct" else "false",
        target_r=float(target_spec.get("mult", 2.0)),
        max_concurrent=int(caps["max_concurrent"]),
        max_new=int(caps["max_new_per_day"]),
        daily_loss=float(caps["max_daily_loss_pct"]),
        be_r=rules.get("break_even", 0.0),
        trail_atr=rules.get("trail_atr", 0.0),
        time_bars=int(rules.get("time_exit", 0)),
        partial_r=rules.get("target_partial", 0.0),
        session_start=caps["session_start"], session_end=caps["session_end"],
    )


def _pascal(key: str) -> str:
    return "".join(p.capitalize() for p in key.split("_")) or "GeneratedBot"


_CBOT_TEMPLATE = '''// {label}
// Generated from the screener bot builder on {generated}
// Entry gates come from the saved strategy: {source}
//
// GENERATED SOURCE, NOT COMPILED SOURCE. Build it in cTrader Automate and
// expect to fix an API detail or two. Read it before you run it, and run it
// on a demo account first.

using System;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{{
    [Robot(AccessRights = AccessRights.None, AddIndicators = true)]
    public class {cls} : Robot
    {{
        [Parameter("Risk per trade %", DefaultValue = {risk_pct}, MinValue = 0.1, MaxValue = 5)]
        public double RiskPercent {{ get; set; }}

        [Parameter("Stop multiple", DefaultValue = {stop_mult}, MinValue = 0.1)]
        public double StopMult {{ get; set; }}

        [Parameter("Target R", DefaultValue = {target_r}, MinValue = 0.1)]
        public double TargetR {{ get; set; }}

        [Parameter("Break even at R", DefaultValue = {be_r}, MinValue = 0)]
        public double BreakEvenR {{ get; set; }}

        [Parameter("Trail ATR", DefaultValue = {trail_atr}, MinValue = 0)]
        public double TrailAtr {{ get; set; }}

        [Parameter("Time exit bars", DefaultValue = {time_bars}, MinValue = 0)]
        public int TimeExitBars {{ get; set; }}

        [Parameter("Partial at R", DefaultValue = {partial_r}, MinValue = 0)]
        public double PartialR {{ get; set; }}

        [Parameter("Max positions", DefaultValue = {max_concurrent}, MinValue = 1)]
        public int MaxPositions {{ get; set; }}

        [Parameter("Max new per day", DefaultValue = {max_new}, MinValue = 1)]
        public int MaxNewPerDay {{ get; set; }}

        [Parameter("Daily loss cap %", DefaultValue = {daily_loss}, MinValue = 0.1)]
        public double DailyLossCap {{ get; set; }}

        [Parameter("Session start", DefaultValue = "{session_start}")]
        public string SessionStart {{ get; set; }}

        [Parameter("Session end", DefaultValue = "{session_end}")]
        public string SessionEnd {{ get; set; }}

        private const string Tag = "{cls}";
{fields}

        private DateTime _day;
        private int _openedToday;
        private double _dayStartEquity;
        private bool _halted;

        protected override void OnStart()
        {{
{decls}
            _day = Server.Time.Date;
            _dayStartEquity = Account.Equity;
        }}

        protected override void OnBar()
        {{
            RollDay();
            ManageOpen();

            if (_halted || !InSession()) return;
            if (CountMine() >= MaxPositions) return;
            if (_openedToday >= MaxNewPerDay) return;

            int i = Bars.Count - 2;   // last CLOSED bar, never the forming one
            if (i < 250) return;

            bool entry = true
{gates};

            if (!entry) return;

            double atr = _atr0 != null ? _atr0.Result[i] : Symbol.PipSize * 10;
            double entryPrice = Symbol.Ask;
            double stop = entryPrice - StopMult * atr;
            double risk = entryPrice - stop;
            if (risk <= 0) return;

            double cashRisk = Account.Equity * RiskPercent / 100.0;
            double volume = Symbol.NormalizeVolumeInUnits(cashRisk / risk, RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return;

            double stopPips = risk / Symbol.PipSize;
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, Tag,
                               stopPips, stopPips * TargetR);
            _openedToday++;
        }}

        private void ManageOpen()
        {{
            foreach (var p in Positions)
            {{
                if (p.Label != Tag || p.SymbolName != SymbolName) continue;

                double risk = Math.Abs(p.EntryPrice - (p.StopLoss ?? p.EntryPrice));
                if (risk <= 0) continue;
                double rNow = (Symbol.Bid - p.EntryPrice) / risk;

                if (BreakEvenR > 0 && rNow >= BreakEvenR &&
                    (p.StopLoss == null || p.StopLoss < p.EntryPrice))
                    ModifyPosition(p, p.EntryPrice, p.TakeProfit);

                if (TrailAtr > 0 && _atr0 != null)
                {{
                    double trail = Symbol.Bid - TrailAtr * _atr0.Result[Bars.Count - 1];
                    if (p.StopLoss == null || trail > p.StopLoss)
                        ModifyPosition(p, trail, p.TakeProfit);
                }}

                if (PartialR > 0 && rNow >= PartialR && p.Comment != "partial")
                {{
                    double half = Symbol.NormalizeVolumeInUnits(p.VolumeInUnits / 2,
                                                               RoundingMode.Down);
                    if (half >= Symbol.VolumeInUnitsMin) ClosePosition(p, half);
                }}

                if (TimeExitBars > 0)
                {{
                    var bars = (int)((Server.Time - p.EntryTime).TotalMinutes /
                                     (int)Bars.TimeFrame.ToString().Length);
                    if (bars >= TimeExitBars) ClosePosition(p);
                }}
            }}
        }}

        private void RollDay()
        {{
            if (Server.Time.Date == _day) return;
            _day = Server.Time.Date;
            _openedToday = 0;
            _dayStartEquity = Account.Equity;
            _halted = false;
        }}

        private int CountMine()
        {{
            int n = 0;
            foreach (var p in Positions)
                if (p.Label == Tag && p.SymbolName == SymbolName) n++;
            return n;
        }}

        private bool InSession()
        {{
            var t = Server.Time.TimeOfDay;
            return t >= TimeSpan.Parse(SessionStart) && t <= TimeSpan.Parse(SessionEnd);
        }}

        protected override void OnTick()
        {{
            if (_dayStartEquity <= 0) return;
            double dd = 100.0 * (_dayStartEquity - Account.Equity) / _dayStartEquity;
            if (dd >= DailyLossCap && !_halted)
            {{
                _halted = true;
                Print("Daily loss cap hit at {{0:F2}}%. No new entries today.", dd);
            }}
        }}

        // ---- helpers the generated gates call ----

        private double HighestHigh(int i, int n)
        {{
            double v = double.MinValue;
            for (int k = i - n + 1; k <= i; k++) v = Math.Max(v, Bars.HighPrices[k]);
            return v;
        }}

        private double LowestLow(int i, int n)
        {{
            double v = double.MaxValue;
            for (int k = i - n + 1; k <= i; k++) v = Math.Min(v, Bars.LowPrices[k]);
            return v;
        }}

        private double LowestDistToMa(int i, int n, SimpleMovingAverage ma, AverageTrueRange atr)
        {{
            double v = double.MaxValue;
            for (int k = i - n + 1; k <= i; k++)
                v = Math.Min(v, (Bars.LowPrices[k] - ma.Result[k]) / atr.Result[k]);
            return v;
        }}

        private double FracAboveMa(int i, int n, SimpleMovingAverage ma)
        {{
            int hit = 0;
            for (int k = i - n + 1; k <= i; k++)
                if (Bars.ClosePrices[k] > ma.Result[k]) hit++;
            return (double)hit / n;
        }}

        private double AtrMean(int i, int n, AverageTrueRange atr)
        {{
            double s = 0;
            for (int k = i - n + 1; k <= i; k++) s += atr.Result[k];
            return s / n;
        }}

        private double VolMean(int i, int n)
        {{
            double s = 0;
            for (int k = i - n + 1; k <= i; k++) s += Bars.TickVolumes[k];
            return s / n;
        }}

        private double ClosePosition(int i)
        {{
            double rng = Bars.HighPrices[i] - Bars.LowPrices[i];
            return rng <= 0 ? 0.5 : (Bars.ClosePrices[i] - Bars.LowPrices[i]) / rng;
        }}

        private double DownDays(int i, int n)
        {{
            int d = 0;
            for (int k = i - n + 1; k <= i; k++)
                if (Bars.ClosePrices[k] < Bars.ClosePrices[k - 1]) d++;
            return d;
        }}
    }}
}}
'''
