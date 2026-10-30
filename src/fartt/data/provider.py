from datetime import date
from typing import Protocol

from fartt.data.types import History, Listing, Quote


class UnknownTicker(Exception):
    """The provider doesn't list this ticker."""


class ProviderUnavailable(Exception):
    """
    The provider couldn't answer: a network failure, a rate limit or an
    unexpected response. Callers outside the data layer catch this instead
    of a library's own exceptions.

    """


class UnsupportedExchange(ValueError):
    """The ticker's suffix names an exchange outside the universe."""


class DataProvider(Protocol):
    """
    Structural interface for stock market data.

    `YFinanceProvider` (`yfinance_provider.py`) is the implementation; tests
    use a fake. Tickers use Yahoo's format: `AAPL` for a US listing,
    `ASML.AS` for Euronext Amsterdam. A future IBKR provider translates its
    own symbols to it.

    Every method raises `UnsupportedExchange` for a suffix outside the
    universe, `UnknownTicker` for a ticker the provider doesn't list, and
    `ProviderUnavailable` when it can't answer.

    """

    def listing(self, ticker: str) -> Listing: ...

    def history(self, ticker: str, start: date | None) -> History:
        """
        Return split-adjusted daily candles with the dividends and splits
        between them: the full history when `start` is `None`, otherwise
        from `start` onwards, inclusive.

        Only closed sessions are returned (see `sessions.py`); the session
        still in progress is never included.

        """
        ...

    def quote(self, ticker: str) -> Quote: ...
