from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fartt.data import (
    Candle,
    History,
    Listing,
    ProviderUnavailable,
    Quote,
    UnknownTicker,
    newest_closed_session,
    normalise_ticker,
)
from fartt.journal import (
    BenchmarkClose,
    BenchmarkScore,
    DecisionDraft,
    Exit,
    Expiry,
    Fill,
    Leg,
    Score,
    Signal,
)
from fartt.risk_config import RISK_FILE, load_risk_config
from fartt.trade_plan import LegCosts, TradePlan, calculate_plan

# The checked-in risk config, which every server under test is built on.
RISK_CONFIG = load_risk_config(
    Path(__file__).resolve().parents[1] / "assets" / RISK_FILE
)


class FakeProvider:
    """
    In-memory `DataProvider`: serves each ticker's stored `History` the way
    `YFinanceProvider` does -- closed sessions only (judged against `now`),
    from `start` onwards, and each ticker's stored `Quote`. Tests change `now` and the stored histories to
    simulate the next day, a split or an amended candle.

    """

    def __init__(
        self,
        histories: dict[str, History],
        listings: dict[str, Listing],
        now: datetime,
        quotes: dict[str, Quote] | None = None,
    ) -> None:
        self.histories = histories
        self.quotes = quotes or {}
        self.listings = listings
        self.now = now
        self.reachable = True
        self.unreachable: set[str] = set()  # Tickers it can't answer for alone
        self.fail_on_full_history = False
        self.history_calls: list[tuple[str, date | None]] = []

    def listing(self, ticker: str) -> Listing:
        ticker = normalise_ticker(ticker)
        self._check(ticker)
        return self.listings[ticker]

    def history(self, ticker: str, start: date | None) -> History:
        ticker = normalise_ticker(ticker)
        self._check(ticker)
        self.history_calls.append((ticker, start))
        if start is None and self.fail_on_full_history:
            raise ProviderUnavailable("Fake provider failed on the full history")
        newest = newest_closed_session(self.listings[ticker].exchange, self.now)

        def keep(day: date) -> bool:
            return (start is None or day >= start) and day <= newest

        stored = self.histories[ticker]
        return History(
            [c for c in stored.candles if keep(c.date)],
            [d for d in stored.dividends if keep(d.date)],
            [s for s in stored.splits if keep(s.date)],
        )

    def quote(self, ticker: str) -> Quote:
        ticker = normalise_ticker(ticker)
        self._check(ticker)
        return self.quotes[ticker]

    def _check(self, ticker: str) -> None:
        if not self.reachable or ticker in self.unreachable:
            raise ProviderUnavailable("Fake provider is unreachable")
        if ticker not in self.listings:
            raise UnknownTicker(f"Fake provider doesn't list '{ticker}'")


def stock_candle(day: date, close: float = 100.0, volume: float = 1000.0) -> Candle:
    """A daily candle for `day` around `close`."""
    return Candle(day, close - 0.5, close + 1.0, close - 1.0, close, volume)


def decision_draft(**changes: Any) -> DecisionDraft:
    """A call on ASML.AS recorded after 2026-10-29's close, with `changes`."""
    fields: dict[str, Any] = {
        "recorded_at": datetime(2026, 10, 29, 21, 0, tzinfo=UTC),
        "ticker": "ASML.AS",
        "currency": "EUR",
        "exchange": "XAMS",
        "direction": "buy",
        "horizon": 20,
        "reasoning": "Trend up, support holding.",
        "signals": [
            Signal(name="trend", stance="for", note="up, aligned"),
            Signal(name="rsi", stance="against"),
        ],
        "decision_session": date(2026, 10, 29),
        "close": 612.4,
        "horizon_ends": date(2026, 11, 26),
        "benchmark": BenchmarkClose(
            ticker="IWDA.AS", currency="EUR", session=date(2026, 10, 29), close=101.25
        ),
    }
    return DecisionDraft(**(fields | changes))


def decision_score(decision_id: int, **changes: Any) -> Score:
    """A score for call `decision_id` (ASML.AS 612.4 -> 640, IWDA 101.25 -> 104)."""
    fields: dict[str, Any] = {
        "decision_id": decision_id,
        "scored_at": datetime(2026, 11, 27, 9, 0, tzinfo=UTC),
        "start": Leg(session=date(2026, 10, 29), close=612.4),
        "end": Leg(session=date(2026, 11, 26), close=640.0),
        "dividends": 0.0,
        "return_": 0.045069,
        "benchmark": BenchmarkScore(
            ticker="IWDA.AS",
            start=Leg(session=date(2026, 10, 29), close=101.25),
            end=Leg(session=date(2026, 11, 26), close=104.0),
            return_=0.02716,
        ),
        "excess_return": 0.017909,
        "right": True,
    }
    return Score(**(fields | changes))


def buy_plan(**changes: Any) -> TradePlan:
    """A plan on ASML.AS made on 2026-10-29's close: entry 600, stop 580, target 680."""
    arguments: dict[str, Any] = {
        "ticker": "ASML.AS",
        "currency": "EUR",
        "exchange_code": "XAMS",
        "based_on": date(2026, 10, 29),
        "last_close": 612.4,
        "atr": 15.0,
        "entry": 600.0,
        "stop": 580.0,
        "target": 680.0,
        "valid_for": 5,
        "rate": None,
        "rate_session": None,
        "rate_is_current": True,
        "portfolio_value": RISK_CONFIG.limits.starting_capital,
        "limits": RISK_CONFIG.limits,
        "costs": RISK_CONFIG.costs,
    }
    return calculate_plan(**(arguments | changes))


SETTLED_AT = datetime(2026, 11, 3, 21, 0, tzinfo=UTC)
LEG_COSTS = LegCosts(
    commission=3.0, slippage=0.6, conversion=0.0, tax=0.0, fee=0.0, total=3.6
)


def fill_line(decision_id: int, **changes: Any) -> Fill:
    """A fill of call `decision_id` at 600 on 2026-10-30."""
    fields: dict[str, Any] = {
        "decision_id": decision_id,
        "settled_at": SETTLED_AT,
        "session": date(2026, 10, 30),
        "price": 600.0,
        "at_open": False,
        "shares": 2,
        "rate": None,
        "value": 1200.0,
        "costs": LEG_COSTS,
    }
    return Fill(**(fields | changes))


def expiry_line(decision_id: int, **changes: Any) -> Expiry:
    """The expiry of call `decision_id`'s order on 2026-11-05."""
    fields: dict[str, Any] = {
        "decision_id": decision_id,
        "settled_at": SETTLED_AT,
        "session": date(2026, 11, 5),
    }
    return Expiry(**(fields | changes))


def exit_line(decision_id: int, **changes: Any) -> Exit:
    """An exit of call `decision_id` at the 580 stop on 2026-11-02."""
    fields: dict[str, Any] = {
        "decision_id": decision_id,
        "settled_at": SETTLED_AT,
        "session": date(2026, 11, 2),
        "reason": "stop",
        "price": 580.0,
        "at_open": False,
        "slippage": 0.0,
        "rate": None,
        "value": 1160.0,
        "costs": LEG_COSTS,
        "dividends": 0.0,
        "pnl": -47.2,
    }
    return Exit(**(fields | changes))
