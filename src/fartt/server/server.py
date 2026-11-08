import os
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

from dotenv import dotenv_values
from loguru import logger
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from fartt.data import (
    FX_SUFFIX,
    SUFFIX_CALENDARS,
    Candle,
    Dividend,
    Listing,
    ProviderUnavailable,
    Split,
    UnknownTicker,
    UnsupportedExchange,
    YFinanceProvider,
    calendar_for,
    exchange_date,
    fx_pair,
    market_status,
    missing_sessions,
    newest_closed_session,
    normalise_ticker,
    session_after,
)
from fartt.journal import (
    BENCHMARK,
    BenchmarkClose,
    Decision,
    DecisionDraft,
    Direction,
    Exit,
    ExitReason,
    Expiry,
    Fill,
    Journal,
    Positions,
    Score,
    Signal,
)
from fartt.performance import (
    MAX_TICKERS,
    MIN_CALLS,
    Group,
    SignalPerformance,
    TickerPerformance,
    calculate_performance,
)
from fartt.portfolio import (
    ClosedSummary,
    Holding,
    Market,
    Portfolio,
    Totals,
    WaitingPlan,
    calculate_portfolio,
)
from fartt.positions import (
    Blocked,
    Position,
    order_window,
    position_state,
    settle_plan,
)
from fartt.procedure import analysis_prompt
from fartt.risk_config import (
    RISK_FILE,
    Costs,
    ExchangeCosts,
    Limits,
    RiskConfig,
    RiskConfigError,
    load_risk_config,
)
from fartt.scoring import RATE_DECIMALS, Unscorable, score_call
from fartt.stock_cache import NoHistory, StockCache, UpdateResult
from fartt.technical import (
    DECIMALS,
    Atr,
    Macd,
    MovingAverage,
    Rsi,
    Sign,
    TimeframeTrend,
    Volume,
    Zone,
    calculate_indicators,
    calculate_levels,
    calculate_trend,
)
from fartt.trade_plan import (
    DEFAULT_VALID_FOR,
    MAX_VALID_FOR,
    LegCosts,
    PlanRefused,
    PlanSummary,
    TradePlan,
    calculate_plan,
)

# Claude Code caps MCP tool output at about 25k tokens by default, and
# numbers cost roughly one token per three characters. 200 compact rows stay
# well under that, and the agent needs recent context, not years of history.
MAX_CANDLES = 200
# A worst-case journal entry in get_decisions' list (2,000 characters of
# reasoning, every signal with a 200-character note, its score, a buy's plan
# in summary and its closed paper position, fill and exit, #144) is about
# 5,900 characters of JSON, so ten are about 58,700, under
# MAX_DECISIONS_CHARS. That bound, 60,000 characters (#143 v3), is about
# 20,000 tokens at one token per three characters, the pessimistic rate for
# numbers: under the limit above. Narrowing by ticker or status reaches
# older calls; get_decisions(id=...) returns one call with its full plan.
MAX_DECISIONS = 10
MAX_DECISIONS_CHARS = 60_000
# A scored call is about 200 characters and a call not scored about 150, so
# 50 of each stay well under the limit. Every score is stored either way.
MAX_SCORED = 50

COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# Paths are relative to the repository, wherever Claude Code starts the server
# from. This assumes an editable install (`uv sync`), as the project runs.
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS_DIR = ROOT / "assets"
# FARTT_ASSETS_DIR moves the cache, set in the environment (the integration
# tests do) or in the repository's .env (see .env.example).
ASSETS_DIR_ENV = "FARTT_ASSETS_DIR"
ENV_FILE = ROOT / ".env"
# FARTT_NOW pins the server's clock to one moment (ISO 8601 with an offset,
# `2026-11-05T12:00:00+00:00`), so a cache filled up to that moment counts as
# current and nothing is fetched for it. The integration tests run the server
# on a checked-in snapshot this way. Read from the environment only, never
# from .env, so a forgotten line can't freeze the operator's own server.
NOW_ENV = "FARTT_NOW"

INSTRUCTIONS = (
    "Fartt's stock analysis server. Tools take one ticker per call, in "
    "Yahoo's format: AAPL for a US listing, ASML.AS for Euronext Amsterdam "
    "(.PA Paris, .BR Brussels, .LS Lisbon, .MI Milan, .IR Dublin, .OL Oslo). "
    "A ticker the server hasn't seen is fetched on the spot. Prices are daily "
    "and split-adjusted, not dividend-adjusted: they are prices that traded, "
    "with dividends kept separately. "
    "Returns, costs and rates are fractions: 0.004 means 0.4%. If is_current "
    "is false the data is stale: read the warning before relying on it. "
    "Indicators, trend and support/resistance zones are evidence, not "
    "signals: weigh them together. get_support_resistance's signs say which "
    "way each points (holding or breaking); they aren't summed into a verdict. "
    "Record every call, buy or pass, with record_decision when it's made; "
    "entries can't be changed afterwards. Before a daily check-in or a "
    "re-analysis of a ticker, read the earlier calls with get_decisions. "
    "At the daily check-in, call score_decisions: it scores every call whose "
    "horizon has passed against IWDA.AS, in EUR. Read get_performance to see "
    "how the calls have done, overall, per signal and per ticker; they are "
    "calls, not trades: no costs or position sizes. Before analysing a "
    "ticker, call get_analysis_procedure with it, first, and follow what it "
    "returns step by step, ending in a call; skip that only when the "
    "analyze_ticker prompt already gave you the procedure. "
    "The risk limits and trading costs are the operator's: read them with "
    "get_risk_config; you can't change them. plan_trade sizes a buy against "
    "them after costs, and record_decision takes the plan with the buy. "
    "settle_positions paper trades the recorded plans on the closed daily "
    "candles, filling, closing and expiring them. "
    "At the daily check-in, call get_portfolio, which settles the plans "
    "first and then shows cash, open positions against their plans, waiting "
    "plans and the P&L after costs, and call score_decisions."
)

# What a ticker the analyze_ticker prompt accepts looks like, closing every
# refusal so the operator can retype it.
TICKER_FORMAT = (
    "Give one stock in Yahoo's format: AAPL for a US listing, ASML.AS for "
    "Euronext Amsterdam. The universe is US tickers without a suffix and "
    "Euronext suffixes " + ", ".join(s for s in SUFFIX_CALENDARS if s) + "."
)

# The ticker argument of the analyze_ticker prompt and get_analysis_procedure.
TICKER_ARGUMENT = (
    "One stock in Yahoo's format: AAPL for a US listing, ASML.AS for "
    "Euronext Amsterdam."
)

MIN_HORIZON = 5
MAX_HORIZON = 60
MAX_REASONING = 2000

Row = tuple[str, float, float, float, float, float]


class CandlesResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    name: str = Field(description="The company's name.")
    currency: str = Field(description="The currency prices are quoted in.")
    exchange: str = Field(
        description="The exchange's calendar code, e.g. XAMS or XNYS."
    )
    columns: list[str] = Field(
        default=COLUMNS, description="Names of the values in each row, in order."
    )
    rows: list[Row] = Field(
        description=(
            "Closed daily sessions, oldest first: date (the session's date on "
            "the exchange, ISO 8601), open, high, low, close, volume. Prices "
            "are split-adjusted, to 4 decimals; an exchange rate's (EURUSD=X) "
            "to 6. The last row is the newest."
        )
    )
    new_candles: int = Field(
        description="Sessions fetched from the provider and cached by this call."
    )
    rebuilt: str | None = Field(
        default=None,
        description=(
            "Why this call rewrote the ticker's whole cached history (a split "
            "or an amended price), if it did."
        ),
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Sessions of the exchange missing between the rows, ISO 8601.",
    )
    is_current: bool = Field(
        description=(
            "True when the last row is the exchange's most recently closed "
            "session. False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(
        default=None, description="What's stale or missing, if anything."
    )


AVERAGE_SESSIONS = 20
# A spread is a small fraction (0.0002 on a liquid stock), so it keeps more
# decimals than prices do.
SPREAD_DECIMALS = 6


class QuoteResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    currency: str = Field(description="The currency prices are quoted in.")
    last: float = Field(description="The last traded price.")
    bid: float | None = Field(
        description="The best bid. Null when unknown; then so are ask and spread."
    )
    ask: float | None = Field(
        description="The best ask. Null when unknown; then so are bid and spread."
    )
    spread: float | None = Field(
        description=(
            "Ask minus bid as a fraction of their midpoint: 0.002 means 0.2%. "
            "Null when unknown, never 0."
        )
    )
    volume: float | None = Field(
        description="Shares traded in the current (or last) session so far."
    )
    average_volume_20d: float | None = Field(
        description=(
            "Mean daily volume over the 20 cached sessions before the quote's session."
        )
    )
    volume_ratio: float | None = Field(
        description=(
            "volume / average_volume_20d. Not normalised for time of day: "
            "during the session it compares a partial day with full days, so "
            "it reads low early on."
        )
    )
    market_open: bool = Field(
        description="Whether the ticker's exchange is trading now."
    )
    next_open: str | None = Field(
        description=(
            "When the next session opens, ISO 8601 in UTC. Null while the "
            "exchange is open."
        )
    )
    quote_time: str = Field(description="When the quote was struck, ISO 8601 in UTC.")
    delay_minutes: int | None = Field(
        description="How many minutes the provider delays this exchange's quotes."
    )
    note: str | None = Field(
        default=None, description="What's missing or unusual, if anything."
    )


class IndicatorsResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    currency: str = Field(description="The currency prices are quoted in.")
    based_on: str = Field(
        description="The session the values are as of, ISO 8601: the last cached."
    )
    close: float = Field(description="That session's close.")
    sma_20: MovingAverage | None = Field(description="The 20-session simple average.")
    sma_50: MovingAverage | None = Field(description="The 50-session simple average.")
    sma_200: MovingAverage | None = Field(description="The 200-session simple average.")
    ema_20: MovingAverage | None = Field(
        description="The 20-session exponential average."
    )
    rsi: Rsi | None = Field(description="RSI 14.")
    macd: Macd | None = Field(description="MACD (12, 26, 9).")
    atr: Atr | None = Field(description="ATR 14.")
    volume: Volume | None = Field(description="Volume against its 20-session average.")
    note: str | None = Field(
        default=None, description="Which indicators are null and why, if any."
    )
    is_current: bool = Field(
        description=(
            "True when based_on is the exchange's most recently closed session. "
            "False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(default=None, description="What's stale, if anything.")


class TrendResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    currency: str = Field(description="The currency prices are quoted in.")
    daily: TimeframeTrend = Field(
        description="The trend on daily sessions, as of the last cached session."
    )
    weekly: TimeframeTrend | None = Field(
        description=(
            "The trend on closed weeks, as of the last closed week's last "
            "session; null when no week has closed yet."
        )
    )
    aligned: bool = Field(
        description="True only when both timeframes are up, or both are down."
    )
    note: str | None = Field(
        default=None,
        description=(
            "What's null and why, what a direction rests on, and whether the "
            "current week is left out, if any."
        ),
    )
    is_current: bool = Field(
        description=(
            "True when the daily based_on is the exchange's most recently closed "
            "session. False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(default=None, description="What's stale, if anything.")


class LevelsResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    currency: str = Field(description="The currency prices are quoted in.")
    based_on: str = Field(
        description="The session the zones are as of, ISO 8601: the last cached."
    )
    last: float = Field(description="That session's close.")
    atr: float | None = Field(
        description="ATR 14 as of that session; null for too short a history."
    )
    support: list[Zone] = Field(
        description=(
            "Zones whose low is at or below the last close, nearest first: "
            "at most 3, each turned at at least twice. A zone holding the "
            "last close comes first, at distance 0."
        )
    )
    resistance: list[Zone] = Field(
        description=(
            "Zones whose low is above the last close, nearest first: at most "
            "3, each turned at at least twice."
        )
    )
    signs: list[Sign] = Field(
        description=(
            "For the first support zone, what points to it holding or "
            "breaking. Inconclusive signs are left out; they aren't summed."
        )
    )
    note: str | None = Field(
        default=None,
        description=(
            "An empty side and why, a zone beyond 5 ATR, or the last close "
            "inside a zone, if any."
        ),
    )
    is_current: bool = Field(
        description=(
            "True when based_on is the exchange's most recently closed session. "
            "False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(default=None, description="What's stale, if anything.")


Status = Literal["open", "scored"]
StatusFilter = Literal["open", "scored", "all"]


class JournalEntry(Decision):
    """A call as record_decision stored it, read back with its state now."""

    status: Status = Field(
        description="open until score_decisions has scored the call, then scored."
    )
    horizon_passed: bool = Field(
        description=(
            "True once the session the horizon ends on has closed on the "
            "stock's exchange. An open call with it true is due for scoring."
        )
    )
    score: Score | None = Field(
        default=None,
        description="How the call turned out, in EUR against IWDA.AS; null while open.",
    )
    # Wider than Decision's: a list carries each plan in summary.
    plan: TradePlan | PlanSummary | None = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=None,
        description=(
            "A buy's trade plan: in full when one call is asked for by id, "
            "in summary in a list; null without a plan."
        ),
    )
    position: Position | None = Field(
        default=None,
        description=(
            "The plan's paper position as settle_positions last settled it: "
            "its state and its fill, exit or expiry (null until written); "
            "null without a plan. Reading doesn't settle."
        ),
    )


class DecisionsResult(BaseModel):
    decisions: list[JournalEntry] = Field(
        description="The newest matching calls, newest first, up to the limit."
    )
    total: int = Field(
        description="How many calls matched the filters, before the limit."
    )
    note: str | None = Field(
        default=None,
        description="Why the list is empty or cut short, if it is.",
    )


class ScoredCall(BaseModel):
    """A call this run scored, in brief; get_decisions has the whole score."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock the call is on.")
    direction: Direction = Field(description="buy or pass.")
    decision_session: date = Field(description="The session the call was made on.")
    horizon_ends: date = Field(description="The session its horizon ended on.")
    return_: float = Field(
        validation_alias="return",
        serialization_alias="return",
        description="The stock's return with dividends, in EUR: 0.004 means 0.4%.",
    )
    benchmark_return: float = Field(
        description="IWDA.AS's return over the same horizon, in EUR."
    )
    excess_return: float = Field(description="return minus benchmark_return.")
    right: bool = Field(
        description="buy: excess_return above 0; pass: excess_return 0 or below."
    )


class NotScoredCall(BaseModel):
    """A call whose horizon has passed but that can't be scored yet."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock the call is on.")
    horizon_ends: date = Field(description="The session its horizon ended on.")
    reason: str = Field(description="What's missing; a later run scores it.")


class ScoreResult(BaseModel):
    scored: list[ScoredCall] = Field(
        description=f"The calls this run scored, oldest first, at most {MAX_SCORED}."
    )
    not_scored: list[NotScoredCall] = Field(
        description=(
            "Calls whose horizon has passed that stay open, oldest first, "
            f"at most {MAX_SCORED}, each with the reason."
        )
    )
    running: int = Field(
        description="How many open calls haven't reached their horizon."
    )
    already_scored: int = Field(
        description="How many calls were scored before this run."
    )
    note: str | None = Field(
        default=None,
        description="Why nothing was scored, or what the lists leave out, if anything.",
    )


class FilledPosition(BaseModel):
    """A plan whose limit buy this run filled."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    session: date = Field(description="The session it filled on.")
    price: float = Field(description="The fill price, in the stock's currency.")
    at_open: bool = Field(description="True when filled at the session's open.")
    shares: int = Field(description="The plan's shares.")
    rate: float | None = Field(description="The session's EUR rate; null for EUR.")
    value: float = Field(description="shares × price, in EUR.")
    costs: LegCosts = Field(description="The buy leg's costs, itemised, in EUR.")


class ExpiredOrder(BaseModel):
    """A plan whose limit buy this run expired unfilled."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    session: date = Field(description="The window's last session.")


class ClosedPosition(BaseModel):
    """A position this run closed."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    session: date = Field(description="The session it closed on.")
    reason: ExitReason = Field(description="stop, target or horizon.")
    price: float = Field(description="The exit price, in the stock's currency.")
    at_open: bool = Field(description="True when closed at the session's open.")
    slippage: float = Field(
        description="How far below the stop a gap closed it; else 0."
    )
    rate: float | None = Field(description="The session's EUR rate; null for EUR.")
    value: float = Field(description="shares × price, in EUR.")
    costs: LegCosts = Field(description="The sell leg's costs, itemised, in EUR.")
    dividends: float = Field(description="Dividends while held, gross, in EUR.")
    pnl: float = Field(description="The realised P&L after both legs' costs, in EUR.")


class OpenPosition(BaseModel):
    """A filled position still running."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    since: date = Field(description="The session it filled on.")
    stop: float = Field(description="The plan's stop.")
    target: float = Field(description="The plan's target.")
    horizon_ends: date = Field(
        description="The session it closes on at the latest, at its close."
    )


class WaitingOrder(BaseModel):
    """A limit buy still live, not filled."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    entry: float = Field(description="The limit price.")
    until: date = Field(description="The window's last session.")


class NotSettled(BaseModel):
    """A plan that couldn't advance as far as the closed sessions go."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    reason: str = Field(
        description="What's missing; a later run settles it once it's there."
    )


class SettleResult(BaseModel):
    filled: list[FilledPosition] = Field(
        description=f"The orders this run filled, oldest call first, at most {MAX_SCORED}."
    )
    expired: list[ExpiredOrder] = Field(
        description=f"The orders this run expired, at most {MAX_SCORED}."
    )
    closed: list[ClosedPosition] = Field(
        description=f"The positions this run closed, at most {MAX_SCORED}."
    )
    open: list[OpenPosition] = Field(
        description=f"The positions still open, at most {MAX_SCORED}."
    )
    waiting: list[WaitingOrder] = Field(
        description=f"The orders still waiting for their entry, at most {MAX_SCORED}."
    )
    not_settled: list[NotSettled] = Field(
        description=(
            "Plans that couldn't be settled as far as the closed sessions go, "
            f"each with the reason, at most {MAX_SCORED}."
        )
    )
    note: str | None = Field(
        default=None,
        description="Why there was nothing to settle, or what the lists leave out.",
    )


class SettledNow(BaseModel):
    """What the settlement run before reading the portfolio just did."""

    filled: list[FilledPosition] = Field(
        description=f"The orders it filled, at most {MAX_SCORED}."
    )
    expired: list[ExpiredOrder] = Field(
        description=f"The orders it expired, at most {MAX_SCORED}."
    )
    closed: list[ClosedPosition] = Field(
        description=f"The positions it closed, at most {MAX_SCORED}."
    )
    not_settled: list[NotSettled] = Field(
        description=(
            "Plans it couldn't settle as far as the closed sessions go, with "
            f"the reason, at most {MAX_SCORED}."
        )
    )


class PortfolioResult(BaseModel):
    based_on: date | None = Field(
        description=(
            "The newest session an open position is valued at; null with none open."
        )
    )
    settled: SettledNow = Field(
        description="What settling the plans first just did, as settle_positions reports it."
    )
    totals: Totals = Field(description="The portfolio in EUR.")
    open: list[Holding] = Field(description="The open positions, oldest call first.")
    waiting: list[WaitingPlan] = Field(
        description=(
            f"The plans waiting for their entry, oldest call first, at most "
            f"{MAX_SCORED}."
        )
    )
    closed: ClosedSummary = Field(description="The closed positions, summarised.")
    note: str | None = Field(
        default=None,
        description="What to know first: no plans yet, stale data, skipped days, negative cash.",
    )


BASIS = (
    "These are calls, not trades: each return is in EUR against IWDA.AS, "
    "without costs or position sizes. Calls whose horizons overlap aren't "
    "independent, so the hit rate can look steadier than it is."
)


class PerformanceResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    basis: str = Field(description="What the figures measure, and what they don't.")
    min_calls: int = Field(
        description=(
            "How many scored calls a group (overall, buy, pass, either side "
            "of a signal) needs before it reports a rate or average."
        )
    )
    scored: int = Field(description="How many calls are scored: overall.calls.")
    open: int = Field(
        description="How many calls aren't scored yet; left out of every figure."
    )
    due: int = Field(
        description=(
            "How many of the open calls have reached their horizon: "
            "score_decisions scores them."
        )
    )
    overall: Group = Field(description="Every scored call.")
    buy: Group = Field(description="The scored buy calls.")
    pass_: Group = Field(
        validation_alias="pass",
        serialization_alias="pass",
        description="The scored pass calls.",
    )
    signals: list[SignalPerformance] = Field(
        description=(
            "Each signal record_decision takes, in order: the scored calls "
            "that used it (for or against) next to those that didn't."
        )
    )
    tickers: list[TickerPerformance] = Field(
        description=(
            "Every ticker with a scored call, most scored calls first, then "
            f"by ticker; at most {MAX_TICKERS}. Its figures hold whatever "
            "its count: weigh them by calls."
        )
    )
    note: str | None = Field(
        default=None, description="What to know first about the figures, if anything."
    )


class ProcedureResult(BaseModel):
    ticker: str = Field(description="The ticker, normalised to Yahoo's format.")
    procedure: str = Field(
        description=(
            "The analysis procedure for this ticker, the text the analyze_ticker "
            "prompt gives: follow its steps in order and report in its sections."
        )
    )


class ExchangeCostsResult(ExchangeCosts):
    suffixes: list[str] = Field(
        description="The ticker suffixes on this exchange's calendar ('' is no suffix)."
    )


class CostsResult(BaseModel):
    tariff: str
    currency_conversion: float
    slippage_per_leg: float
    exchanges: dict[str, ExchangeCostsResult]


class RiskConfigResult(BaseModel):
    source: str = Field(description="The file the server read them from.")
    limits: Limits
    costs: CostsResult


def _risk_config_result(config: RiskConfig) -> RiskConfigResult:
    """The config with each exchange's suffixes, in the universe's order."""
    costs: Costs = config.costs
    exchanges = {
        code: ExchangeCostsResult(
            **costs.exchanges[code].model_dump(),
            suffixes=[s for s, c in SUFFIX_CALENDARS.items() if c == code],
        )
        for code in dict.fromkeys(SUFFIX_CALENDARS.values())
    }
    return RiskConfigResult(
        source=str(config.source),
        limits=config.limits,
        costs=CostsResult(
            tariff=costs.tariff,
            currency_conversion=costs.currency_conversion,
            slippage_per_leg=costs.slippage_per_leg,
            exchanges=exchanges,
        ),
    )


def _performance_note(scored: int, due: int, more_tickers: int) -> str | None:
    notes: list[str] = []
    if not scored:
        notes.append("No call has been scored yet.")
    if due:
        notes.append(
            "1 call is due: call score_decisions to score it."
            if due == 1
            else f"{due} calls are due: call score_decisions to score them."
        )
    if more_tickers:
        notes.append(
            f"Listing the {MAX_TICKERS} tickers with the most scored calls; "
            f"{more_tickers} more {'has' if more_tickers == 1 else 'have'} "
            "scored calls too."
        )
    return " ".join(notes) or None


def _status(decision: Decision, scores: Mapping[int, Score]) -> Status:
    return "scored" if decision.id in scores else "open"


def _horizon_passed(decision: Decision, now: datetime) -> bool:
    """Whether the session the call's horizon ends on has closed: due for scoring."""
    return newest_closed_session(decision.exchange, now) >= decision.horizon_ends


def _no_match_note(status: StatusFilter, ticker: str | None, any_scored: bool) -> str:
    filters = [f"status={status}"] if status != "all" else []
    if ticker is not None:
        filters.append(f"ticker={ticker}")
    if not filters:
        return "No calls are recorded yet."
    note = f"No calls match {', '.join(filters)}."
    if status == "scored" and not any_scored:
        note += " No call has been scored yet."
    return note


def _position(decision: Decision, positions: Positions) -> Position | None:
    """A call's paper position as stored; `None` without a plan."""
    if decision.plan is None:
        return None
    fill = positions.fills.get(decision.id)
    expiry = positions.expiries.get(decision.id)
    exit = positions.exits.get(decision.id)
    return Position(
        state=position_state(fill, expiry, exit), fill=fill, exit=exit, expiry=expiry
    )


def _entry(
    decision: Decision,
    scores: Mapping[int, Score],
    positions: Positions,
    now: datetime,
    *,
    brief: bool,
) -> JournalEntry:
    """A call with its state now; its plan in summary when `brief`."""
    plan = decision.plan
    return JournalEntry(
        **decision.model_dump(exclude={"plan"}),
        status=_status(decision, scores),
        horizon_passed=_horizon_passed(decision, now),
        score=scores.get(decision.id),
        plan=PlanSummary.of(plan) if brief and plan is not None else plan,
        position=_position(decision, positions),
    )


def _one_decision(
    cache: StockCache, journal: Journal, decision_id: int
) -> DecisionsResult:
    """get_decisions(id=...): the call with its full plan, or none and a note."""
    try:
        decisions = journal.decisions()
        scores = journal.scores()
        positions = journal.positions()
    except CACHE_ERRORS as error:
        raise _journal_unreadable("get_decisions", journal.path, error) from error
    found = [decision for decision in decisions if decision.id == decision_id]
    logger.info(f"get_decisions id={decision_id}: {len(found)} match")
    if not found:
        known = (
            f"ids run from 1 to {len(decisions)}"
            if decisions
            else "no call is recorded yet"
        )
        return DecisionsResult(
            decisions=[], total=0, note=f"No call has id {decision_id}; {known}."
        )
    return DecisionsResult(
        decisions=[_entry(found[0], scores, positions, cache.clock(), brief=False)],
        total=1,
    )


def _journal_unreadable(tool: str, path: Path, error: Exception) -> ToolError:
    logger.error(f"{tool}: {error}")
    return ToolError(
        f"The journal at {path} can't be read: {error}. The operator must fix "
        "the file; the server never rewrites it."
    )


def _score_note(
    due: int, running: int, more_scored: int, more_not_scored: int
) -> str | None:
    notes: list[str] = []
    if not due:
        notes.append(
            f"No open call's horizon has passed; {running} "
            f"{'is' if running == 1 else 'are'} running."
        )
    if more_scored:
        notes.append(
            f"Showing the first {MAX_SCORED} calls scored; {more_scored} more "
            f"{'was' if more_scored == 1 else 'were'} scored. Read them with "
            'get_decisions(status="scored").'
        )
    if more_not_scored:
        calls = "call wasn't" if more_not_scored == 1 else "calls weren't"
        notes.append(
            f"Showing the first {MAX_SCORED} calls not scored; {more_not_scored} "
            f"more {calls} scored."
        )
    return " ".join(notes) or None


def _settle_note(
    plans: int, closed: int, expired: int, cut: Mapping[str, int]
) -> str | None:
    notes: list[str] = []
    if not plans:
        notes.append("No call has a trade plan to settle.")
    elif closed + expired == plans:
        notes.append(
            f"No plan is waiting or open: {closed} closed, {expired} expired. "
            "Nothing to settle."
        )
    notes.extend(
        f"Showing the first {MAX_SCORED} of {MAX_SCORED + more} {name}."
        for name, more in cut.items()
        if more
    )
    return " ".join(notes) or None


def _portfolio_note(portfolio: Portfolio, any_plans: bool) -> str | None:
    notes: list[str] = []
    if not any_plans:
        notes.append(
            "No plans yet: the portfolio is the starting capital. A buy "
            "recorded with a trade plan starts it."
        )
    if portfolio.stale:
        notes.append(
            "Valued on a stale close, see each position's warning: "
            + ", ".join(portfolio.stale)
            + "."
        )
    days = portfolio.skipped_days
    if days:
        notes.append(
            f"{days} valuation day{'s' if days != 1 else ''} left out of the "
            "peak: a close or exchange rate isn't cached yet."
        )
    cash = portfolio.totals.cash
    if cash < 0:
        notes.append(
            f"Cash is below 0 (€{cash:,.2f}): a fill's exchange rate was worse "
            "than its plan's."
        )
    waiting = len(portfolio.waiting)
    if waiting > MAX_SCORED:
        notes.append(f"Showing the first {MAX_SCORED} of {waiting} waiting plans.")
    return " ".join(notes) or None


def _rounded(value: float | None, decimals: int = DECIMALS) -> float | None:
    return round(value, decimals) if value is not None else None


def _row(candle: Candle, decimals: int) -> Row:
    # Rounded as get_indicators rounds: Yahoo's prices are 32-bit floats, so
    # 1666.2 would otherwise arrive as 1666.199951171875.
    return (
        candle.date.isoformat(),
        round(candle.open, decimals),
        round(candle.high, decimals),
        round(candle.low, decimals),
        round(candle.close, decimals),
        round(candle.volume, DECIMALS),
    )


def _cache_error(tool: str, ticker: str, error: Exception) -> ToolError:
    # A folder it can't write, or a file it can't parse: retrying won't help.
    logger.error(f"{tool} {ticker}: {error}")
    return ToolError(
        f"The stock cache for {ticker} can't be read or written: {error}. "
        "Retrying won't help until the operator fixes or deletes the ticker's "
        "cache folder."
    )


# What a cache folder it can't write or a file it can't parse raises.
CACHE_ERRORS = (OSError, ValueError, KeyError)


def _unknown_ticker(ticker: str) -> str:
    return (
        f"The data provider doesn't know {ticker}. Check the ticker: Yahoo's "
        "format is AAPL for a US listing, ASML.AS for Euronext Amsterdam."
    )


def _update(
    tool: str, cache: StockCache, ticker: str
) -> tuple[str, UpdateResult | None]:
    """
    The ticker normalised, and the result of bringing its cache up to date,
    or `None` when the provider is unreachable. `NoHistory` propagates, for
    the caller to word. Every other failure becomes a ToolError: the SDK
    reports any other exception to the agent as a bare "Error executing
    tool", so each one says what to do.

    """
    try:
        ticker = normalise_ticker(ticker)
    except UnknownTicker as error:
        raise ToolError(str(error)) from error
    try:
        return ticker, cache.update(ticker)
    except UnsupportedExchange as error:
        raise ToolError(str(error)) from error
    except UnknownTicker as error:
        raise ToolError(_unknown_ticker(ticker)) from error
    except ProviderUnavailable as error:
        logger.warning(f"{tool} {ticker}: {error}")
        return ticker, None
    except CACHE_ERRORS as error:
        raise _cache_error(tool, ticker, error) from error


def _update_all(
    tool: str, cache: StockCache, tickers: Iterable[str]
) -> tuple[set[str], dict[str, str]]:
    """
    Bring each ticker up to date: the ones the provider couldn't reach, and
    why each ticker it can't serve at all can't be used. A cache it can't
    read or write is a ToolError.

    """
    unreachable: set[str] = set()
    unlisted: dict[str, str] = {}
    for ticker in tickers:
        try:
            cache.update(ticker)
        except ProviderUnavailable as error:
            logger.warning(f"{tool} {ticker}: {error}")
            unreachable.add(ticker)
        except UnknownTicker:
            unlisted[ticker] = f"Yahoo no longer lists {ticker}"
        except NoHistory:
            unlisted[ticker] = f"Yahoo has no daily history for {ticker}"
        except UnsupportedExchange as error:
            unlisted[ticker] = str(error)
        except CACHE_ERRORS as error:
            raise _cache_error(tool, ticker, error) from error
    return unreachable, unlisted


class _CachedReader:
    """Each ticker's cached candles and events, read once per run."""

    def __init__(self, tool: str, cache: StockCache) -> None:
        self.tool = tool
        self.cache = cache
        self.read: dict[str, list[Candle]] = {}
        self.events: dict[str, tuple[list[Dividend], list[Split]]] = {}

    def candles(self, ticker: str) -> list[Candle]:
        if ticker not in self.read:
            try:
                self.read[ticker] = self.cache.latest(ticker, sys.maxsize)
            except CACHE_ERRORS as error:
                raise _cache_error(self.tool, ticker, error) from error
        return self.read[ticker]

    def dividends(self, ticker: str) -> list[Dividend]:
        return self._events(ticker)[0]

    def splits(self, ticker: str) -> list[Split]:
        return self._events(ticker)[1]

    def _events(self, ticker: str) -> tuple[list[Dividend], list[Split]]:
        if ticker not in self.events:
            try:
                self.events[ticker] = self.cache.events(ticker)
            except CACHE_ERRORS as error:
                raise _cache_error(self.tool, ticker, error) from error
        return self.events[ticker]


class Settlement(NamedTuple):
    result: SettleResult
    not_updated: set[str]  # Tickers whose cache couldn't be brought up to date
    reader: _CachedReader  # Each ticker's cached data, read once


def _settle(
    tool: str, cache: StockCache, journal: Journal, risk_config: RiskConfig
) -> Settlement:
    """
    Paper trade every running plan up to its exchange's newest closed
    session and store what happened: what settle_positions does, and what
    get_portfolio and plan_trade do first.

    """
    try:
        decisions = journal.decisions()
        fills, expiries, exits = journal.positions()
    except CACHE_ERRORS as error:
        raise _journal_unreadable(tool, journal.path, error) from error

    planned = [d for d in decisions if d.plan is not None]
    running = [d for d in planned if d.id not in exits and d.id not in expiries]
    needed = dict.fromkeys(
        ticker
        for decision in running
        for ticker in (decision.ticker, fx_pair(decision.currency))
        if ticker is not None
    )
    unreachable, unlisted = _update_all(tool, cache, needed)
    reader = _CachedReader(tool, cache)

    now = cache.clock()
    settled_at = now.replace(microsecond=0)
    filled: list[FilledPosition] = []
    expired: list[ExpiredOrder] = []
    closed: list[ClosedPosition] = []
    still_open: list[OpenPosition] = []
    waiting: list[WaitingOrder] = []
    not_settled: list[NotSettled] = []

    def store[L: (Fill, Expiry, Exit)](
        append: Callable[[L], L | None], line: L
    ) -> bool:
        """Write `line`; False when another run wrote it in the meantime."""
        try:
            return append(line) is not None
        except CACHE_ERRORS as error:
            logger.error(f"{tool} #{line.decision_id}: {error}")
            raise ToolError(
                f"The journal at {journal.path} can't be read or written: "
                f"{error}. Lines stored before this call stay stored. The "
                "operator must fix the file; the server never rewrites it."
            ) from error

    for decision in running:
        plan = decision.plan
        assert plan is not None
        fill = fills.get(decision.id)
        pair = fx_pair(decision.currency)
        tickers = [decision.ticker] + ([pair] if pair else [])
        reason = next((unlisted[t] for t in tickers if t in unlisted), None)
        blocked = Blocked(decision.ticker, reason) if reason else None
        if blocked is None:
            result = settle_plan(
                decision,
                plan,
                fill,
                reader.candles(decision.ticker),
                reader.dividends(decision.ticker),
                reader.splits(decision.ticker),
                reader.candles(pair) if pair else None,
                risk_config.costs,
                newest_closed_session(decision.exchange, now),
                settled_at,
            )
            blocked = result.blocked
            if result.fill is not None:
                fill = result.fill
                if store(journal.append_fill, result.fill):
                    filled.append(
                        FilledPosition(
                            id=decision.id,
                            ticker=decision.ticker,
                            **result.fill.model_dump(
                                exclude={"decision_id", "settled_at"}
                            ),
                        )
                    )
            if result.expiry is not None:
                if store(journal.append_expiry, result.expiry):
                    expired.append(
                        ExpiredOrder(
                            id=decision.id,
                            ticker=decision.ticker,
                            session=result.expiry.session,
                        )
                    )
                continue
            if result.exit is not None:
                if store(journal.append_exit, result.exit):
                    closed.append(
                        ClosedPosition(
                            id=decision.id,
                            ticker=decision.ticker,
                            **result.exit.model_dump(
                                exclude={"decision_id", "settled_at"}
                            ),
                        )
                    )
                continue
        if blocked is not None:
            reason = blocked.reason
            if blocked.ticker in unreachable:
                reason += " (the provider is unreachable)"
            logger.warning(f"{tool} #{decision.id} {decision.ticker}: {reason}")
            not_settled.append(
                NotSettled(id=decision.id, ticker=decision.ticker, reason=reason)
            )
        elif fill is not None:
            still_open.append(
                OpenPosition(
                    id=decision.id,
                    ticker=decision.ticker,
                    since=fill.session,
                    stop=plan.stop,
                    target=plan.target,
                    horizon_ends=decision.horizon_ends,
                )
            )
        else:
            waiting.append(
                WaitingOrder(
                    id=decision.id,
                    ticker=decision.ticker,
                    entry=plan.entry,
                    until=order_window(decision, plan)[1],
                )
            )

    lists = {
        "filled": filled,
        "expired": expired,
        "closed": closed,
        "open positions": still_open,
        "waiting orders": waiting,
        "plans not settled": not_settled,
    }
    logger.info(
        f"{tool}: " + ", ".join(f"{len(items)} {name}" for name, items in lists.items())
    )
    result = SettleResult(
        filled=filled[:MAX_SCORED],
        expired=expired[:MAX_SCORED],
        closed=closed[:MAX_SCORED],
        open=still_open[:MAX_SCORED],
        waiting=waiting[:MAX_SCORED],
        not_settled=not_settled[:MAX_SCORED],
        note=_settle_note(
            len(planned),
            len(exits) + len(closed),
            len(expiries) + len(expired),
            {name: max(len(items) - MAX_SCORED, 0) for name, items in lists.items()},
        ),
    )
    return Settlement(result, unreachable | set(unlisted), reader)


def _portfolio(
    tool: str, cache: StockCache, journal: Journal, risk_config: RiskConfig
) -> tuple[Settlement, Portfolio, bool]:
    """
    Settle the plans, then value the portfolio on the journal and the
    cached data: the settlement, the portfolio, and whether any call has a
    plan. A journal it can't read is a ToolError.

    """
    settlement = _settle(tool, cache, journal, risk_config)
    try:
        decisions = journal.decisions()
        positions = journal.positions()
    except CACHE_ERRORS as error:
        raise _journal_unreadable(tool, journal.path, error) from error
    by_id = {decision.id: decision for decision in decisions}
    tickers = dict.fromkeys(
        ticker
        for decision_id in positions.fills
        for ticker in (
            by_id[decision_id].ticker,
            fx_pair(by_id[decision_id].currency),
        )
        if ticker is not None
    )
    reader = settlement.reader
    market = Market(
        {ticker: reader.candles(ticker) for ticker in tickers},
        {ticker: reader.dividends(ticker) for ticker in tickers},
    )
    portfolio = calculate_portfolio(
        decisions,
        positions,
        market,
        risk_config.costs,
        risk_config.limits.starting_capital,
        cache.clock(),
        settlement.not_updated,
    )
    any_plans = any(decision.plan is not None for decision in decisions)
    return settlement, portfolio, any_plans


def _no_history_error(listing: Listing, newest: date) -> ToolError:
    """
    Why a listed ticker has no closed session, from its first trade date
    against the exchange's newest closed session: only a listing whose
    first session hasn't closed yet will have one by retrying.

    """
    ticker = listing.ticker
    if listing.first_trade is None:
        return ToolError(
            f"The data provider has no closed sessions for {ticker} yet. If it "
            "listed today, try again after its first session closes; otherwise "
            "it has no daily history and can't be analysed."
        )
    if listing.first_trade <= newest:
        return ToolError(
            f"The data provider lists {ticker} ({listing.name}) but has no "
            "daily price history for it, so it can't be analysed. Retrying "
            "won't help; pick another listing."
        )
    return ToolError(
        f"The data provider has no closed sessions for {ticker} yet, for "
        "instance because it listed today. Try again after its first session "
        "closes."
    )


class Snapshot(NamedTuple):
    listing: Listing
    candles: list[Candle]  # Oldest first
    update: UpdateResult | None  # None when the provider was unreachable
    is_current: bool
    warning: str | None  # What's stale, if anything


def _snapshot(tool: str, cache: StockCache, ticker: str, count: int) -> Snapshot:
    """
    Up to `count` of the ticker's newest cached candles after bringing its
    cache up to date, and whether they're current. An unreachable provider
    serves what's cached, with a warning saying how far behind it is; with
    nothing cached, or no closed sessions yet, it's a ToolError.

    """
    try:
        ticker, update = _update(tool, cache, ticker)
    except NoHistory as error:
        newest = cache.newest_closed_session(error.listing.ticker)
        raise _no_history_error(error.listing, newest) from error
    try:
        candles = cache.latest(ticker, count)
        listing = cache.listing(ticker)
    except CACHE_ERRORS as error:
        raise _cache_error(tool, ticker, error) from error
    if update is None and (not candles or listing is None):
        raise ToolError(
            f"Nothing is cached for {ticker} yet and the data provider is "
            "unreachable. Try again later."
        )
    if not candles or listing is None:
        raise ToolError(
            f"The data provider has no closed sessions for {ticker} yet, "
            "for instance because it listed today. Try again after its "
            "first session closes."
        )

    newest = cache.newest_closed_session(ticker)
    last = candles[-1].date
    is_current = last == newest
    warning = None
    if update is None and not is_current:
        behind = len(missing_sessions(listing.exchange, [last, newest])) + 1
        warning = (
            f"The data provider is unreachable, so these are cached sessions "
            f"through {last.isoformat()}, {behind} "
            f"session{'s' if behind != 1 else ''} behind. Try again later."
        )
    elif not is_current:
        warning = (
            f"The provider hasn't published {newest.isoformat()}'s session "
            f"yet; the last session is {last.isoformat()}. Try again later."
        )
    return Snapshot(listing, candles, update, is_current, warning)


def _unreachable(ticker: str) -> ToolError:
    return ToolError(
        f"The data provider is unreachable, so there's no quote for {ticker}. "
        "Try again later."
    )


def _current_close(
    tool: str, cache: StockCache, ticker: str, count: int = 1
) -> Snapshot:
    """
    The ticker's last `count` candles after bringing its cache up to date. A
    ToolError unless the last is the exchange's newest closed session: a call
    stamped with an older close could be scored on a move that was already
    known.

    """
    snapshot = _snapshot(tool, cache, ticker, count)
    if not snapshot.is_current:
        ticker = snapshot.listing.ticker
        newest = cache.newest_closed_session(ticker)
        raise ToolError(
            f"{ticker}'s newest close ({newest.isoformat()}) isn't available "
            "yet, so the call can't be stamped with the price of the moment. "
            "Try again later."
        )
    return snapshot


class PlanRequest(BaseModel):
    """The prices of a long plan, as the agent proposes them."""

    entry: float = Field(
        description=(
            "The limit buy price, in the stock's currency: above 0, at most 1 "
            "ATR above the last close."
        )
    )
    stop: float = Field(
        description="The stop, below the entry, in the stock's currency."
    )
    target: float = Field(
        description="The target, above the entry, in the stock's currency."
    )
    valid_for: int = Field(
        default=DEFAULT_VALID_FOR,
        description=(
            f"How many sessions of the stock's exchange the limit order stays "
            f"open, 1 to {MAX_VALID_FOR}."
        ),
    )


def _portfolio_value(risk_config: RiskConfig) -> float:
    """
    The value every share-of-the-portfolio limit is taken of. The starting
    capital until the paper portfolio (#145) values itself.

    """
    return risk_config.limits.starting_capital


def _refuse_exchange_rate(ticker: str, action: str) -> None:
    """
    A ToolError when `ticker` is an exchange rate: record_decision and
    plan_trade refuse one in the same words, `action` aside.

    """
    if ticker.strip().upper().endswith(FX_SUFFIX):
        raise ToolError(
            f"{ticker.strip().upper()} is an exchange rate, not a stock. "
            f"{action} on stocks only."
        )


def _trade_plan(
    tool: str,
    cache: StockCache,
    ticker: str,
    request: PlanRequest,
    risk_config: RiskConfig,
) -> TradePlan:
    """
    The plan for `request` on the ticker's current data, or a ToolError
    saying why there is none: an exchange rate, data that isn't current, or
    a check or limit the plan breaks. plan_trade and record_decision both
    plan with it, so they refuse in the same words.

    """
    _refuse_exchange_rate(ticker, "Plan trades")
    snapshot = _snapshot(tool, cache, ticker, sys.maxsize)
    if not snapshot.is_current:
        ticker = snapshot.listing.ticker
        newest = cache.newest_closed_session(ticker)
        raise ToolError(
            f"{ticker}'s newest close ({newest.isoformat()}) isn't available "
            "yet, so the plan can't be made on the price of the moment. Try "
            "again later."
        )
    return _plan_on(tool, cache, snapshot, request, risk_config)


def _plan_on(
    tool: str,
    cache: StockCache,
    snapshot: Snapshot,
    request: PlanRequest,
    risk_config: RiskConfig,
) -> TradePlan:
    """
    The plan for `request` on a current snapshot of the stock's whole
    history, or a ToolError saying which check or limit it breaks.
    record_decision plans on the snapshot it stamps the call with, so the
    call and its plan are one session.

    """
    listing, candles = snapshot.listing, snapshot.candles
    ticker = listing.ticker
    atr = calculate_indicators(candles).atr
    rate, rate_session, rate_is_current = None, None, True
    pair = fx_pair(listing.currency)
    if pair is not None:
        fx = _snapshot(tool, cache, pair, 1)
        rate = round(fx.candles[-1].close, RATE_DECIMALS)
        rate_session, rate_is_current = fx.candles[-1].date, fx.is_current
    try:
        return calculate_plan(
            ticker=ticker,
            currency=listing.currency,
            exchange_code=listing.exchange,
            based_on=candles[-1].date,
            last_close=round(candles[-1].close, DECIMALS),
            atr=atr.value if atr else None,
            entry=request.entry,
            stop=request.stop,
            target=request.target,
            valid_for=request.valid_for,
            rate=rate,
            rate_session=rate_session,
            rate_is_current=rate_is_current,
            portfolio_value=_portfolio_value(risk_config),
            limits=risk_config.limits,
            costs=risk_config.costs,
        )
    except PlanRefused as error:
        raise ToolError(f"{error} Nothing was recorded.") from error


class NotAnalysable(ValueError):
    """A ticker the analysis procedure refuses; the message says why and what a ticker looks like."""


def _analysable_ticker(ticker: str) -> str:
    """
    The ticker normalised, checked offline as the tools check it, or
    `NotAnalysable` saying why not and what a valid ticker looks like. The
    `analyze_ticker` prompt and `get_analysis_procedure` both check with it,
    so they refuse the same tickers in the same words. Whether the ticker is
    listed is left to the tools the procedure names.

    """
    try:
        normalised = normalise_ticker(ticker)
        calendar_for(normalised)
    except UnknownTicker:
        reason = f"'{ticker}' isn't a ticker."
    except UnsupportedExchange:
        reason = f"'{normalise_ticker(ticker)}' is on an exchange outside the universe."
    else:
        if not normalised.endswith(FX_SUFFIX):
            return normalised
        reason = f"{normalised} is an exchange rate, not a stock to make a call on."
    raise NotAnalysable(f"{reason} {TICKER_FORMAT}")


def _prompt_ticker(ticker: str) -> str:
    """
    `_analysable_ticker` for the prompt: a refusal is an `MCPError`
    (INVALID_PARAMS), since the SDK passes one through prompt rendering as
    it is and turns anything else into a bare "Error rendering prompt".

    """
    try:
        return _analysable_ticker(ticker)
    except NotAnalysable as error:
        raise MCPError(INVALID_PARAMS, str(error)) from error


def build_server(
    cache: StockCache, journal: Journal, risk_config: RiskConfig
) -> MCPServer:
    """
    Build the MCP server on `cache`, which also supplies the provider (for
    quotes) and the clock (for market hours and the time a call is
    recorded), `journal`, where calls are recorded, and `risk_config`, the
    operator's limits and costs, read once at start.

    A factory rather than a module-level server, so tests can pass a cache
    over an in-memory provider and connect with the SDK's in-memory client.

    """
    server = MCPServer("fartt", instructions=INSTRUCTIONS)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_candles(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
        count: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_CANDLES,
                description="How many of the newest closed sessions to return.",
            ),
        ] = 60,
    ) -> CandlesResult:
        """
        Return a ticker's newest closed daily sessions, oldest first, after
        bringing its local cache up to date. A ticker not cached yet is
        fetched in full first.
        """
        listing, candles, update, is_current, warning = _snapshot(
            "get_candles", cache, ticker, count
        )
        ticker = listing.ticker
        gaps = missing_sessions(listing.exchange, [c.date for c in candles])
        warnings = [warning] if warning else []
        if gaps:
            warnings.append(
                "The provider has no data for these sessions: "
                + ", ".join(day.isoformat() for day in gaps)
                + "."
            )

        # An exchange rate to the decimals a score keeps it to, so the rate a
        # score used can be read back here.
        decimals = RATE_DECIMALS if ticker.endswith(FX_SUFFIX) else DECIMALS
        new_candles = update.new_candles if update else 0
        rebuilt = update.rebuilt if update else None
        logger.info(
            f"get_candles {ticker}: {len(candles)} rows, {new_candles} new"
            + (f", rebuilt ({rebuilt})" if rebuilt else "")
            + f", current={is_current}"
        )
        return CandlesResult(
            ticker=listing.ticker,
            name=listing.name,
            currency=listing.currency,
            exchange=listing.exchange,
            rows=[_row(c, decimals) for c in candles],
            new_candles=new_candles,
            rebuilt=rebuilt,
            gaps=[day.isoformat() for day in gaps],
            is_current=is_current,
            warning=" ".join(warnings) or None,
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_quote(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
    ) -> QuoteResult:
        """
        Return a ticker's latest quote: last price, bid, ask and spread,
        today's volume against its 20-session average, whether its exchange
        is open, and how delayed the quote is. Use it to judge liquidity and
        where a limit order should sit.
        """
        # A listed ticker without daily history still has a live quote; its
        # listing comes with the error, since the cache doesn't keep it.
        uncached: Listing | None = None
        try:
            ticker, update = _update("get_quote", cache, ticker)
            if update is None:
                raise _unreachable(ticker)
        except NoHistory as error:
            uncached = error.listing
            ticker = uncached.ticker
        try:
            quote = cache.provider.quote(ticker)
        except UnknownTicker as error:
            raise ToolError(_unknown_ticker(ticker)) from error
        except ProviderUnavailable as error:
            logger.warning(f"get_quote {ticker}: {error}")
            raise _unreachable(ticker) from error
        listing, candles = uncached, list[Candle]()
        if uncached is None:
            try:
                listing = cache.listing(ticker)
                candles = cache.latest(ticker, AVERAGE_SESSIONS + 1)
            except CACHE_ERRORS as error:
                raise _cache_error("get_quote", ticker, error) from error
        if listing is None:
            raise _unreachable(ticker)

        notes: list[str] = []
        bid, ask, spread = quote.bid, quote.ask, None
        if bid is None or ask is None:
            bid = ask = None
            notes.append("The spread is unknown: the provider has no bid or ask.")
        elif ask <= bid:
            # Matching orders would have traded: a crossed or locked quote is
            # stale or delayed data, not a free spread.
            bid = ask = None
            notes.append(
                "The spread is unknown: the provider's bid and ask are crossed "
                "or locked (stale or delayed data)."
            )
        else:
            spread = (ask - bid) / ((ask + bid) / 2)

        # Sessions before the quote's own, so today is never in its average.
        today = exchange_date(listing.exchange, quote.time)
        prior = [c for c in candles if c.date < today][-AVERAGE_SESSIONS:]
        if len(prior) < AVERAGE_SESSIONS and len(candles) > AVERAGE_SESSIONS:
            # A quote older than the newest cached sessions: read further back.
            try:
                candles = cache.latest(ticker, sys.maxsize)
            except CACHE_ERRORS as error:
                raise _cache_error("get_quote", ticker, error) from error
            prior = [c for c in candles if c.date < today][-AVERAGE_SESSIONS:]
        last_closed = cache.newest_closed_session(ticker)
        if today < last_closed:
            notes.append(
                f"The quote is from {today.isoformat()}, before the exchange's "
                f"last closed session ({last_closed.isoformat()}), so it may be "
                "stale, for instance because trading is suspended."
            )
        average = sum(c.volume for c in prior) / len(prior) if prior else None
        if 0 < len(prior) < AVERAGE_SESSIONS:
            notes.append(
                f"The average volume covers only {len(prior)} sessions, all that "
                "are cached."
            )
        elif not prior:
            notes.append("No earlier sessions are cached to average volume over.")
        ratio = quote.volume / average if quote.volume is not None and average else None

        status = market_status(listing.exchange, cache.clock())
        logger.info(
            f"get_quote {ticker}: {quote.last} {listing.currency}, open={status.open}"
        )
        return QuoteResult(
            ticker=listing.ticker,
            currency=listing.currency,
            last=round(quote.last, DECIMALS),
            bid=_rounded(bid),
            ask=_rounded(ask),
            spread=_rounded(spread, SPREAD_DECIMALS),
            volume=_rounded(quote.volume),
            average_volume_20d=_rounded(average),
            volume_ratio=_rounded(ratio),
            market_open=status.open,
            next_open=status.next_open.isoformat() if status.next_open else None,
            quote_time=quote.time.isoformat(),
            delay_minutes=quote.delay_minutes,
            note=" ".join(notes) or None,
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_indicators(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
    ) -> IndicatorsResult:
        """
        Return a ticker's daily moving averages (SMA 20, 50, 200 and EMA 20),
        RSI, MACD, ATR and volume, each with its state, as of the last cached
        session (based_on). Use it to judge whether a dip is overdone.
        """
        listing, candles, _, is_current, warning = _snapshot(
            "get_indicators", cache, ticker, sys.maxsize
        )
        ticker = listing.ticker
        try:
            dividends, _ = cache.events(ticker)
        except CACHE_ERRORS as error:
            raise _cache_error("get_indicators", ticker, error) from error
        indicators = calculate_indicators(
            candles, frozenset(dividend.date for dividend in dividends)
        )
        logger.info(
            f"get_indicators {ticker}: {len(candles)} sessions, current={is_current}"
        )
        return IndicatorsResult(
            ticker=ticker,
            currency=listing.currency,
            based_on=candles[-1].date.isoformat(),
            close=indicators.close,
            sma_20=indicators.sma_20,
            sma_50=indicators.sma_50,
            sma_200=indicators.sma_200,
            ema_20=indicators.ema_20,
            rsi=indicators.rsi,
            macd=indicators.macd,
            atr=indicators.atr,
            volume=indicators.volume,
            note=indicators.note,
            is_current=is_current,
            warning=warning,
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_trend(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
    ) -> TrendResult:
        """
        Return a ticker's trend direction on the daily and weekly timeframes,
        with the evidence (close against the SMA 50 and 200, their slopes, the
        last cross) and whether the timeframes agree. Use it to judge whether a
        dip is in an uptrend or the start of a fall.
        """
        listing, candles, _, is_current, warning = _snapshot(
            "get_trend", cache, ticker, sys.maxsize
        )
        trend = calculate_trend(candles, listing.exchange)
        logger.info(
            f"get_trend {listing.ticker}: {len(candles)} sessions, current={is_current}"
        )
        return TrendResult(
            ticker=listing.ticker,
            currency=listing.currency,
            daily=trend.daily,
            weekly=trend.weekly,
            aligned=trend.aligned,
            note=trend.note,
            is_current=is_current,
            warning=warning,
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_support_resistance(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
    ) -> LevelsResult:
        """
        Return a ticker's support and resistance zones around the last close:
        where price turned before, each with its range, touches, turns, volume
        and strength, nearest first. For the nearest support, signs list what
        points to it holding or breaking. Zones are for entries, stops and
        targets, not signals on their own.
        """
        listing, candles, _, is_current, warning = _snapshot(
            "get_support_resistance", cache, ticker, sys.maxsize
        )
        ticker = listing.ticker
        try:
            dividends, _ = cache.events(ticker)
        except CACHE_ERRORS as error:
            raise _cache_error("get_support_resistance", ticker, error) from error
        levels = calculate_levels(
            candles,
            listing.exchange,
            frozenset(dividend.date for dividend in dividends),
        )
        logger.info(
            f"get_support_resistance {ticker}: {len(candles)} sessions, "
            f"current={is_current}"
        )
        return LevelsResult(
            ticker=ticker,
            currency=listing.currency,
            based_on=levels.based_on,
            last=levels.last,
            atr=levels.atr,
            support=levels.support,
            resistance=levels.resistance,
            signs=levels.signs,
            note=levels.note,
            is_current=is_current,
            warning=warning,
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def plan_trade(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
        entry: Annotated[
            float, Field(description=PlanRequest.model_fields["entry"].description)
        ],
        stop: Annotated[
            float, Field(description=PlanRequest.model_fields["stop"].description)
        ],
        target: Annotated[
            float, Field(description=PlanRequest.model_fields["target"].description)
        ],
        valid_for: Annotated[
            int, Field(description=PlanRequest.model_fields["valid_for"].description)
        ] = DEFAULT_VALID_FOR,
    ) -> TradePlan:
        """
        Size a long trade from your entry (a limit buy), stop and target
        against the operator's risk limits, after costs: the whole shares to
        buy, the risk and reward in EUR with every cost of both legs
        itemised, the reward-to-risk, and the limit that set the size. A
        plan that breaks a check or limit is refused saying which. Nothing is
        recorded: try prices as often as needed, then pass the plan to
        record_decision with the buy.
        """
        plan = _trade_plan(
            "plan_trade",
            cache,
            ticker,
            PlanRequest(entry=entry, stop=stop, target=target, valid_for=valid_for),
            risk_config,
        )
        logger.info(
            f"plan_trade {plan.ticker}: {plan.shares} shares, risk {plan.risk} EUR, "
            f"reward-to-risk {plan.reward_to_risk}, sized by {plan.sized_by}"
        )
        return plan

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        )
    )
    def record_decision(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
        direction: Annotated[
            Direction,
            Field(
                description=(
                    "buy: the stock will beat the benchmark (IWDA.AS) over the "
                    "horizon; pass: it won't."
                )
            ),
        ],
        horizon: Annotated[
            int,
            Field(
                ge=MIN_HORIZON,
                le=MAX_HORIZON,
                description="How many sessions of the stock's exchange the call runs.",
            ),
        ],
        reasoning: Annotated[
            str,
            Field(
                description=(
                    f"Why, in up to {MAX_REASONING:,} characters: the summary of "
                    "the analysis."
                )
            ),
        ],
        signals: Annotated[
            list[Signal],
            Field(
                min_length=1,
                description=(
                    "The evidence the call rests on, each name at most once, "
                    "and which way it pointed."
                ),
            ),
        ],
        plan: Annotated[
            PlanRequest | None,
            Field(
                description=(
                    "With a buy only: the trade plan's entry, stop, target and "
                    "valid_for, as given to plan_trade. The server works the "
                    "plan out again on current data and stores it with the "
                    "call; a plan that breaks a limit records nothing."
                )
            ),
        ] = None,
    ) -> Decision:
        """
        Record a call (buy or pass) in the journal at the moment it's made,
        stamped with the stock's and the benchmark's last closes, so it can
        be scored later against what the market did. Record every call.
        Entries can't be changed or removed afterwards: a correction is a
        new call. A buy can carry its trade plan (see plan_trade).
        """
        if plan is not None and direction == "pass":
            raise ToolError(
                "A pass has no trade plan: give a plan with a buy only, or "
                "record the pass without one. Nothing was recorded."
            )
        reasoning = reasoning.strip()
        if not 1 <= len(reasoning) <= MAX_REASONING:
            raise ToolError(
                f"reasoning must be 1 to {MAX_REASONING:,} characters after "
                f"stripping whitespace; it is {len(reasoning):,}."
            )
        _refuse_exchange_rate(ticker, "Record calls")
        names = [signal.name for signal in signals]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ToolError(
                f"signals name {', '.join(twice)} more than once; give each "
                "signal once, with the stance it took."
            )

        # A plan needs the whole history (for its ATR); one read serves the
        # call's close and the plan, so both are the same session.
        count = 1 if plan is None else sys.maxsize
        stock = _current_close("record_decision", cache, ticker, count)
        benchmark = _current_close("record_decision", cache, BENCHMARK)
        candle = stock.candles[-1]
        trade_plan = None
        if plan is not None:
            trade_plan = _plan_on("record_decision", cache, stock, plan, risk_config)
        try:
            horizon_ends = session_after(stock.listing.exchange, candle.date, horizon)
        except ValueError as error:
            # A cached candle on a day that isn't a session, or a calendar
            # that ends too soon: neither happens with sound data.
            logger.error(f"record_decision {stock.listing.ticker}: {error}")
            raise ToolError(
                f"The horizon's end can't be placed on {stock.listing.exchange}'s "
                f"calendar: {error}. Nothing was recorded. The operator must "
                f"check {stock.listing.ticker}'s cached candles."
            ) from error
        draft = DecisionDraft(
            recorded_at=cache.clock().replace(microsecond=0),
            ticker=stock.listing.ticker,
            currency=stock.listing.currency,
            exchange=stock.listing.exchange,
            direction=direction,
            horizon=horizon,
            reasoning=reasoning,
            signals=signals,
            decision_session=candle.date,
            close=round(candle.close, DECIMALS),
            horizon_ends=horizon_ends,
            benchmark=BenchmarkClose(
                ticker=benchmark.listing.ticker,
                currency=benchmark.listing.currency,
                session=benchmark.candles[-1].date,
                close=round(benchmark.candles[-1].close, DECIMALS),
            ),
            plan=trade_plan,
        )
        try:
            recorded = journal.append(draft)
        except CACHE_ERRORS as error:
            logger.error(f"record_decision {draft.ticker}: {error}")
            raise ToolError(
                f"The journal at {journal.path} can't be read or written: "
                f"{error}. Nothing was recorded. The operator must fix the "
                "file; the server never rewrites it."
            ) from error
        logger.info(
            f"record_decision {recorded.ticker}: #{recorded.id} {recorded.direction}, "
            f"{recorded.horizon} sessions"
        )
        return recorded

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_decisions(
        status: Annotated[
            StatusFilter | None,
            Field(
                description=(
                    "open: calls not scored yet, including those whose horizon "
                    "has passed; scored: calls with a score; all (the "
                    "default): both."
                )
            ),
        ] = None,
        ticker: Annotated[
            str | None,
            Field(
                description=(
                    "Only the calls on this ticker, in Yahoo's format: AAPL, "
                    "ASML.AS. Leave out for every ticker."
                )
            ),
        ] = None,
        limit: Annotated[
            int | None,
            Field(
                ge=1,
                le=MAX_DECISIONS,
                description=(
                    f"How many of the newest matching calls to return, 1 to "
                    f"{MAX_DECISIONS} (the default)."
                ),
            ),
        ] = None,
        id: Annotated[
            int | None,
            Field(
                ge=1,
                description=(
                    "One call by its id, with its full trade plan; without "
                    "status, ticker or limit."
                ),
            ),
        ] = None,
    ) -> DecisionsResult:
        """
        Read the journal back: the calls record_decision recorded, newest
        first, filtered by status and ticker. A call is open until
        score_decisions scores it; horizon_passed says whether it's due. A
        scored call carries its score. A buy's trade plan is listed in
        summary; ask for the call by id for the whole plan, costs and
        limits included. A call with a plan carries its paper position
        (waiting, open, closed or expired, with its fill, exit or expiry) as
        settle_positions last settled it; reading doesn't settle.
        """
        if id is not None:
            given = [
                name
                for name, value in (
                    ("status", status),
                    ("ticker", ticker),
                    ("limit", limit),
                )
                if value is not None
            ]
            if given:
                raise ToolError(
                    f"id asks for one call; leave out {', '.join(given)}, or "
                    "leave out id to list calls."
                )
            return _one_decision(cache, journal, id)
        status = status or "all"
        limit = limit or MAX_DECISIONS
        if ticker is not None:
            try:
                ticker = normalise_ticker(ticker)
            except UnknownTicker as error:
                raise ToolError(str(error)) from error
        try:
            decisions = journal.decisions()
            scores = journal.scores()
            positions = journal.positions()
        except CACHE_ERRORS as error:
            raise _journal_unreadable("get_decisions", journal.path, error) from error

        now = cache.clock()
        matching = [
            _entry(decision, scores, positions, now, brief=True)
            for decision in reversed(decisions)
            if (ticker is None or decision.ticker == ticker)
            and (status == "all" or _status(decision, scores) == status)
        ]
        logger.info(
            f"get_decisions status={status} ticker={ticker}: {len(matching)} match"
        )
        note = None
        if not matching:
            note = _no_match_note(status, ticker, bool(scores))
        elif len(matching) > limit:
            note = (
                f"Showing the newest {limit} of {len(matching)} matching calls. "
                "Narrow the list with ticker or status."
            )
        return DecisionsResult(
            decisions=matching[:limit], total=len(matching), note=note
        )

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        )
    )
    def score_decisions() -> ScoreResult:
        """
        Score every open call whose horizon has passed, at the daily
        check-in, and store the scores in the journal. A call is measured in
        EUR against IWDA.AS: the stock's return from its decision session to
        the horizon's end, with the dividends going ex in between, against
        the benchmark's over the same span. A price in another currency is
        converted at that day's EUR exchange-rate close (Yahoo's EURUSD=X),
        which closes at midnight UTC, so a US call is scored the day after
        its horizon ends. A score is stored once; calls that can't be scored
        yet stay open with the reason, for a later run.
        """
        try:
            decisions = journal.decisions()
            scores = journal.scores()
        except CACHE_ERRORS as error:
            raise _journal_unreadable("score_decisions", journal.path, error) from error

        now = cache.clock()
        still_open = [d for d in decisions if d.id not in scores]
        due = [d for d in still_open if _horizon_passed(d, now)]
        running = len(still_open) - len(due)

        # Each ticker the due calls need is brought up to date once per run.
        needed = dict.fromkeys(
            ticker
            for decision in due
            for ticker in (decision.ticker, BENCHMARK, fx_pair(decision.currency))
            if ticker is not None
        )
        unreachable, unlisted = _update_all("score_decisions", cache, needed)
        # Each ticker's candles and dividends are read once per run too.
        reader = _CachedReader("score_decisions", cache)
        candles, dividends = reader.candles, reader.dividends

        scored: list[ScoredCall] = []
        not_scored: list[NotScoredCall] = []
        raced = 0
        for decision in due:
            pair = fx_pair(decision.currency)
            tickers = [decision.ticker, BENCHMARK] + ([pair] if pair else [])
            reason = next((unlisted[t] for t in tickers if t in unlisted), None)
            if reason is None:
                try:
                    score = score_call(
                        decision,
                        candles(decision.ticker),
                        dividends(decision.ticker),
                        candles(BENCHMARK),
                        candles(pair) if pair else None,
                        cache.clock().replace(microsecond=0),
                    )
                except Unscorable as error:
                    reason = str(error)
                    if error.ticker in unreachable:
                        reason += " (the provider is unreachable)"
                else:
                    try:
                        stored = journal.append_score(score)
                    except CACHE_ERRORS as error:
                        logger.error(f"score_decisions #{decision.id}: {error}")
                        raise ToolError(
                            f"The journal at {journal.path} can't be read or "
                            f"written: {error}. Scores stored before this call "
                            "stay stored. The operator must fix the file; the "
                            "server never rewrites it."
                        ) from error
                    if stored is None:
                        raced += 1  # Another run scored it in the meantime.
                    else:
                        scored.append(
                            ScoredCall(
                                id=decision.id,
                                ticker=decision.ticker,
                                direction=decision.direction,
                                decision_session=decision.decision_session,
                                horizon_ends=decision.horizon_ends,
                                return_=stored.return_,
                                benchmark_return=stored.benchmark.return_,
                                excess_return=stored.excess_return,
                                right=stored.right,
                            )
                        )
                    continue
            logger.warning(
                f"score_decisions #{decision.id} {decision.ticker}: {reason}"
            )
            not_scored.append(
                NotScoredCall(
                    id=decision.id,
                    ticker=decision.ticker,
                    horizon_ends=decision.horizon_ends,
                    reason=reason,
                )
            )

        logger.info(
            f"score_decisions: {len(scored)} scored, {len(not_scored)} not scored, "
            f"{running} running, {len(scores) + raced} already scored"
        )
        return ScoreResult(
            scored=scored[:MAX_SCORED],
            not_scored=not_scored[:MAX_SCORED],
            running=running,
            already_scored=len(scores) + raced,
            note=_score_note(
                len(due),
                running,
                max(len(scored) - MAX_SCORED, 0),
                max(len(not_scored) - MAX_SCORED, 0),
            ),
        )

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        )
    )
    def settle_positions() -> SettleResult:
        """
        Paper trade every recorded plan on the closed daily candles, at the
        daily check-in, and store what happened in the journal. A plan is a
        limit buy at its entry, live for valid_for sessions after the plan's
        session (never past the call's horizon): the first session whose low
        reaches the entry fills it, at the open when the session opened
        below it. A position closes at its stop (at the open when it gapped
        below, with the slippage), at its target (at the open when it gapped
        above; a session reaching both counts as the stop), or at the close
        of the call's horizon_ends. Each leg pays its costs and converts at
        its own session's EUR rate, so a US leg settles the day after its
        session. A fill, expiry or exit is stored once; a plan that can't
        advance yet is listed with the reason, for a later run.
        """
        return _settle("settle_positions", cache, journal, risk_config).result

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        )
    )
    def get_portfolio() -> PortfolioResult:
        """
        The paper portfolio at the daily check-in, in EUR: settles the
        recorded plans first (as settle_positions does), then returns the
        totals (cash, cash reserved for waiting plans, free cash, the open
        positions' value after the costs to close, total value, realised
        and unrealised P&L, peak and drawdown, open risk), each open
        position against its plan with its distance to stop and target, the
        plans waiting for their entry, and the closed positions in summary.
        A stale ticker is valued on its newest cached close and says so.
        """
        settlement, portfolio, any_plans = _portfolio(
            "get_portfolio", cache, journal, risk_config
        )
        settled = settlement.result
        totals = portfolio.totals
        logger.info(
            f"get_portfolio: total {totals.total_value} EUR, cash {totals.cash}, "
            f"{totals.open_positions} open, {totals.waiting_plans} waiting, "
            f"{portfolio.closed.count} closed"
        )
        return PortfolioResult(
            based_on=portfolio.based_on,
            settled=SettledNow(
                filled=settled.filled,
                expired=settled.expired,
                closed=settled.closed,
                not_settled=settled.not_settled,
            ),
            totals=totals,
            open=portfolio.open,
            waiting=portfolio.waiting[:MAX_SCORED],
            closed=portfolio.closed,
            note=_portfolio_note(portfolio, any_plans),
        )

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_performance() -> PerformanceResult:
        """
        How the scored calls have done against IWDA.AS, in EUR: the hit rate
        and the average return, benchmark return, excess return and edge,
        overall, for buy and pass, per signal (used against not used) and
        per ticker. Edge is the excess return for a buy and its negation for
        a pass, so above 0 means right on average; read it, not the excess
        return, for a group mixing both. Calls, not trades: no costs or
        position sizes. Open calls are counted, not figured.
        """
        try:
            decisions = journal.decisions()
            scores = journal.scores()
        except CACHE_ERRORS as error:
            raise _journal_unreadable("get_performance", journal.path, error) from error

        now = cache.clock()
        still_open = [d for d in decisions if _status(d, scores) == "open"]
        due = sum(_horizon_passed(d, now) for d in still_open)
        performance = calculate_performance(decisions, scores)
        logger.info(
            f"get_performance: {performance.overall.calls} scored, "
            f"{len(still_open)} open, {due} due"
        )
        return PerformanceResult(
            basis=BASIS,
            min_calls=MIN_CALLS,
            scored=performance.overall.calls,
            open=len(still_open),
            due=due,
            overall=performance.overall,
            buy=performance.buy,
            pass_=performance.pass_,
            signals=performance.signals,
            tickers=performance.tickers,
            note=_performance_note(
                performance.overall.calls, due, performance.more_tickers
            ),
        )

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_analysis_procedure(
        ticker: Annotated[str, Field(description=TICKER_ARGUMENT)],
    ) -> ProcedureResult:
        """
        Return fartt's analysis procedure for one ticker: the steps, the
        decision rules and the report's sections. Call it before analysing
        the ticker and follow it step by step, ending in a recorded call.
        """
        try:
            normalised = _analysable_ticker(ticker)
        except NotAnalysable as error:
            raise ToolError(str(error)) from error
        logger.info(f"get_analysis_procedure {normalised}")
        return ProcedureResult(ticker=normalised, procedure=analysis_prompt(normalised))

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_risk_config() -> RiskConfigResult:
        """
        Return the operator's risk limits and trading costs, as the server
        read them from its config file at start: every trade plan is held to
        them. Fractions throughout (0.02 means 2%). Only the operator can
        change them, in the file, and a change takes effect on restart.
        """
        logger.info(f"get_risk_config from {risk_config.source}")
        return _risk_config_result(risk_config)

    @server.prompt(
        name="analyze_ticker",
        title="Analyse a ticker",
        description=(
            "Analyse one stock by fartt's eight-step procedure, in order, "
            "ending in a buy or pass call recorded in the journal."
        ),
    )
    def analyze_ticker(
        ticker: Annotated[str, Field(description=TICKER_ARGUMENT)],
    ) -> str:
        return analysis_prompt(_prompt_ticker(ticker))

    return server


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _clock() -> Callable[[], datetime]:
    """The real clock, or the moment FARTT_NOW pins it to."""
    setting = os.environ.get(NOW_ENV)
    if not setting:
        return _utc_now
    try:
        pinned = datetime.fromisoformat(setting)
    except ValueError as error:
        raise SystemExit(
            f"{NOW_ENV}={setting} isn't an ISO 8601 date and time; "
            "give one with an offset, such as 2026-11-05T12:00:00+00:00"
        ) from error
    if pinned.tzinfo is None:
        raise SystemExit(f"{NOW_ENV}={setting} has no offset; add one, such as +00:00")
    pinned = pinned.astimezone(UTC)
    logger.warning(f"{NOW_ENV} pins the clock at {pinned.isoformat()}")
    return lambda: pinned


def main() -> None:
    """
    Run the server over stdio: Claude Code starts it from `.mcp.json`
    (`uv run fartt`). Logs go to stderr, since stdout carries the protocol.

    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    setting = os.environ.get(ASSETS_DIR_ENV) or dotenv_values(ENV_FILE).get(
        ASSETS_DIR_ENV
    )
    assets_dir = ROOT / setting if setting else DEFAULT_ASSETS_DIR
    clock = _clock()
    try:
        risk_config = load_risk_config(assets_dir / RISK_FILE)
    except RiskConfigError as error:
        raise SystemExit(str(error)) from error
    logger.info(f"Serving stocks cached in {assets_dir} over stdio")
    build_server(
        StockCache(YFinanceProvider(clock=clock), assets_dir, clock=clock),
        Journal(assets_dir / "journal"),
        risk_config,
    ).run()
