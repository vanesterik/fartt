from datetime import date, datetime
from typing import NamedTuple


class Candle(NamedTuple):
    """
    One trading session's split-adjusted OHLCV candle. Prices are not
    adjusted for dividends: they are what traded, once splits are accounted
    for, so support levels, stops and limit orders can use them directly.

    """

    date: date  # The trading session, in the exchange's own calendar
    open: float
    high: float
    low: float
    close: float
    volume: float


class Dividend(NamedTuple):
    date: date  # Ex-dividend date
    amount: float  # Per share, in the ticker's currency


class Split(NamedTuple):
    date: date
    ratio: float  # New shares per old share: 4.0 for 4-for-1, 0.1 for 1-for-10


class History(NamedTuple):
    """Candles and the events between them, each list in ascending date order."""

    candles: list[Candle]
    dividends: list[Dividend]
    splits: list[Split]


class Listing(NamedTuple):
    ticker: str
    name: str
    currency: str  # ISO 4217, e.g. "EUR"
    exchange: str  # The exchange_calendars code, e.g. "XAMS"


class Quote(NamedTuple):
    """
    The latest quote. `bid`, `ask` and `volume` are `None` when the provider
    doesn't know them, never 0, so a spread is never computed from a
    missing price.

    """

    last: float
    bid: float | None
    ask: float | None
    volume: float | None  # Today's volume so far
    time: datetime  # When the quote was struck, UTC
    delay_minutes: int | None
