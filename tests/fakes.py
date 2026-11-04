from datetime import UTC, date, datetime
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
    Leg,
    Score,
    Signal,
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
        if not self.reachable:
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
