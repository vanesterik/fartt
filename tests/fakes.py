from datetime import date, datetime

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


def stock_candle(day: date, close: float = 100.0, volume: float = 1000.0) -> Candle:
    """A daily candle for `day` around `close`."""
    return Candle(day, close - 0.5, close + 1.0, close - 1.0, close, volume)
