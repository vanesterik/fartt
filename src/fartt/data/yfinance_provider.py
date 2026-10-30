import math
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any, Protocol

import pandas as pd
import yfinance
from yfinance.exceptions import YFTickerMissingError

from fartt.data.provider import ProviderUnavailable, UnknownTicker
from fartt.data.sessions import calendar_for, newest_closed_session
from fartt.data.types import Candle, Dividend, History, Listing, Split

_PRICES = ("Open", "High", "Low", "Close")


class YahooTicker(Protocol):
    """The part of `yfinance.Ticker` this module calls."""

    @property
    def info(self) -> dict[str, Any]: ...

    def history(self, **kwargs: Any) -> pd.DataFrame: ...


def _now() -> datetime:
    return datetime.now(UTC)


class YFinanceProvider:
    """
    `DataProvider` implementation backed by yfinance -- the only module in
    the project that imports yfinance, so the provider stays swappable for
    IBKR's feed.

    Prices come from `Ticker.history(auto_adjust=False, actions=True)`:
    split-adjusted OHLC, with dividends and splits as separate columns.
    `ticker_factory` and `clock` are injectable for tests; `clock` returns
    the current time (UTC) and decides which sessions have closed.

    """

    def __init__(
        self,
        ticker_factory: Callable[[str], YahooTicker] = yfinance.Ticker,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._ticker_factory = ticker_factory
        self._clock = clock
        # yfinance logs most failures and returns an empty frame unless told
        # to raise; this module needs the exception to tell an unknown ticker
        # from an unreachable provider. (The per-call raise_errors is
        # deprecated in yfinance 1.7.)
        yfinance.config.debug.hide_exceptions = False

    def listing(self, ticker: str) -> Listing:
        ticker = ticker.upper()
        exchange = calendar_for(ticker)
        with _translate_errors(ticker):
            info = self._ticker_factory(ticker).info
        if "currency" not in info or "quoteType" not in info:
            raise UnknownTicker(f"Yahoo doesn't list '{ticker}'")
        name = info.get("longName") or info.get("shortName") or ticker
        return Listing(ticker, str(name), str(info["currency"]), exchange)

    def history(self, ticker: str, start: date | None) -> History:
        ticker = ticker.upper()
        exchange = calendar_for(ticker)
        window: dict[str, str] = (
            {"period": "max"} if start is None else {"start": start.isoformat()}
        )
        try:
            with _translate_errors(ticker):
                frame = self._ticker_factory(ticker).history(
                    **window,
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                )
        except UnknownTicker:
            if start is None:
                raise
            # The ticker exists (it was cached); nothing new since `start`.
            return History([], [], [])

        newest = newest_closed_session(exchange, self._clock())
        candles: list[Candle] = []
        dividends: list[Dividend] = []
        splits: list[Split] = []
        for day, values in _rows(frame):
            if day > newest or (start is not None and day < start):
                continue
            if values["Dividends"]:
                dividends.append(Dividend(day, values["Dividends"]))
            if values["Stock Splits"]:
                splits.append(Split(day, values["Stock Splits"]))
            if any(math.isnan(values[column]) for column in _PRICES):
                continue
            volume = values["Volume"]
            candles.append(
                Candle(
                    day,
                    open=values["Open"],
                    high=values["High"],
                    low=values["Low"],
                    close=values["Close"],
                    volume=0.0 if math.isnan(volume) else volume,
                )
            )
        return History(candles, dividends, splits)


def _rows(frame: pd.DataFrame) -> Iterator[tuple[date, dict[str, float]]]:
    # Each row's date is the index's date in its own (the exchange's) time
    # zone, which is the session date. Missing action columns count as 0.
    frame = frame.sort_index()
    index = pd.DatetimeIndex(frame.index)
    for position, timestamp in enumerate(index):
        row = {
            str(column): float(value) for column, value in frame.iloc[position].items()
        }
        row.setdefault("Dividends", 0.0)
        row.setdefault("Stock Splits", 0.0)
        yield timestamp.date(), row


@contextmanager
def _translate_errors(ticker: str) -> Generator[None]:
    # No yfinance exception leaves this module. A missing ticker is
    # UnknownTicker: yfinance's own error for it (and its subclasses,
    # missing prices or time zone), or Yahoo answering HTTP 404, which is
    # how an unknown symbol surfaces once exceptions aren't hidden. Anything
    # else -- rate limits, network errors, odd responses -- is unavailable.
    try:
        yield
    except YFTickerMissingError as error:
        raise UnknownTicker(f"Yahoo doesn't list '{ticker}'") from error
    except Exception as error:
        if _status_code(error) == 404:
            raise UnknownTicker(f"Yahoo doesn't list '{ticker}'") from error
        raise ProviderUnavailable(
            f"Yahoo Finance couldn't answer for '{ticker}': {error}"
        ) from error


def _status_code(error: Exception) -> int | None:
    # yfinance's HTTP errors come from curl_cffi and carry the response;
    # read it by attribute so this module doesn't import curl_cffi.
    response: object = getattr(error, "response", None)
    status_code: object = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None
