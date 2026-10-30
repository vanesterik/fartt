from datetime import date, datetime

from fartt.candle_cache import interval_to_ms
from fartt.data import Candle as StockCandle
from fartt.data import (
    History,
    Listing,
    ProviderUnavailable,
    Quote,
    UnknownTicker,
    newest_closed_session,
    normalise_ticker,
)
from fartt.exchange import Candle, ExchangeUnavailable

HOUR_MS = 3_600_000


class FakeExchange:
    """
    In-memory `Exchange`: serves `candles` the way `CcxtExchange` over
    Bitvavo does -- closed candles only (judged against `now_ms`), sorted,
    from the window `[since_ms, since_ms + limit * interval)`. ccxt sends
    Bitvavo exactly that window, so a gap longer than it comes back empty.

    """

    def __init__(
        self,
        candles: list[Candle],
        now_ms: int,
        markets: set[str] | None = None,
        intervals: list[str] | None = None,
        reachable: bool = True,
        fail_after_fetches: int | None = None,
    ) -> None:
        self.candles = candles
        self.now_ms = now_ms
        self.markets = markets if markets is not None else {"BTC/EUR"}
        self.intervals = intervals if intervals is not None else ["1m", "1h"]
        self.reachable = reachable
        self.fail_after_fetches = fail_after_fetches
        self.fetch_calls: list[int] = []

    def has_market(self, market: str) -> bool:
        if not self.reachable:
            raise ExchangeUnavailable("Exchange 'fake' is unavailable: offline")
        return market in self.markets

    def supported_intervals(self) -> list[str]:
        return self.intervals

    def fetch_closed_candles(
        self, market: str, interval: str, since_ms: int, limit: int
    ) -> list[Candle]:
        if not self.has_market(market):
            raise ValueError(f"Market '{market}' not found on exchange 'fake'")
        if (
            self.fail_after_fetches is not None
            and len(self.fetch_calls) >= self.fail_after_fetches
        ):
            raise ExchangeUnavailable("Exchange 'fake' is unavailable: timed out")
        self.fetch_calls.append(since_ms)
        interval_ms = interval_to_ms(interval)
        window_end_ms = since_ms + limit * interval_ms
        return sorted(
            (
                c
                for c in self.candles
                if since_ms <= c.timestamp < window_end_ms
                and c.timestamp + interval_ms <= self.now_ms
            ),
            key=lambda c: c.timestamp,
        )


def candle(hour: int, close: float = 100.0) -> Candle:
    """An hourly candle opening `hour` hours after the epoch."""
    return Candle(hour * HOUR_MS, close, close + 1, close - 1, close, 2.5)


class FakeProvider:
    """
    In-memory `DataProvider`: serves each ticker's stored `History` the way
    `YFinanceProvider` does -- closed sessions only (judged against `now`),
    from `start` onwards. Tests change `now` and the stored histories to
    simulate the next day, a split or an amended candle.

    """

    def __init__(
        self,
        histories: dict[str, History],
        listings: dict[str, Listing],
        now: datetime,
    ) -> None:
        self.histories = histories
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
        raise NotImplementedError

    def _check(self, ticker: str) -> None:
        if not self.reachable:
            raise ProviderUnavailable("Fake provider is unreachable")
        if ticker not in self.listings:
            raise UnknownTicker(f"Fake provider doesn't list '{ticker}'")


def stock_candle(
    day: date, close: float = 100.0, volume: float = 1000.0
) -> StockCandle:
    """A daily candle for `day` around `close`."""
    return StockCandle(day, close - 0.5, close + 1.0, close - 1.0, close, volume)
