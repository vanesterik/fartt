from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError, YFTzMissingError

from fartt.data import (
    Candle,
    Dividend,
    Listing,
    ProviderUnavailable,
    Quote,
    Split,
    UnknownTicker,
    UnsupportedExchange,
    YFinanceProvider,
)

NY = "America/New_York"
COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]
# NYSE closed (plus margin) for 2026-10-29 at 20:30 UTC.
AFTER_CLOSE = datetime(2026, 10, 29, 21, 0, tzinfo=UTC)
DURING_SESSION = datetime(2026, 10, 29, 20, 0, tzinfo=UTC)


def frame(rows: list[tuple[str, list[float]]], tz: str = NY) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(day, tz=tz) for day, _ in rows])
    return pd.DataFrame([values for _, values in rows], index=index, columns=COLUMNS)


def row(close: float, dividend: float = 0.0, split: float = 0.0) -> list[float]:
    return [
        close - 0.5,
        close + 0.5,
        close - 1.0,
        close,
        close - 0.1,
        1000.0,
        dividend,
        split,
    ]


class FakeTicker:
    def __init__(
        self,
        history: pd.DataFrame | None = None,
        info: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._history = history if history is not None else frame([])
        self.info = info if info is not None else {}
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._history


def provider_for(
    ticker: FakeTicker, now: datetime = AFTER_CLOSE
) -> tuple[YFinanceProvider, list[str]]:
    requested: list[str] = []

    def factory(symbol: str) -> FakeTicker:
        requested.append(symbol)
        return ticker

    return YFinanceProvider(ticker_factory=factory, clock=lambda: now), requested


def test_history_requests_split_adjusted_daily_prices() -> None:
    ticker = FakeTicker()
    provider, _ = provider_for(ticker)

    provider.history("AAPL", None)
    provider.history("AAPL", date(2026, 10, 1))

    common = {
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
    }
    assert ticker.calls == [
        {"period": "max", **common},
        {"start": "2026-10-01", **common},
    ]


def test_history_maps_rows_to_candles_on_the_exchange_date() -> None:
    provider, _ = provider_for(FakeTicker(frame([("2026-10-28", row(10.5))])))

    history = provider.history("AAPL", None)

    assert history.candles == [
        Candle(date(2026, 10, 28), 10.0, 11.0, 9.5, 10.5, 1000.0)
    ]


def test_history_drops_the_forming_session() -> None:
    rows = [("2026-10-28", row(10.0)), ("2026-10-29", row(11.0))]
    provider, _ = provider_for(FakeTicker(frame(rows)), now=DURING_SESSION)

    history = provider.history("AAPL", None)

    assert [c.date for c in history.candles] == [date(2026, 10, 28)]


def test_history_drops_rows_before_start_and_rows_with_nan_prices() -> None:
    nan_close = row(12.0)
    nan_close[3] = float("nan")
    nan_volume = row(13.0)
    nan_volume[5] = float("nan")
    rows = [
        ("2026-10-26", row(10.0)),
        ("2026-10-27", nan_close),
        ("2026-10-28", nan_volume),
    ]
    provider, _ = provider_for(FakeTicker(frame(rows)))

    history = provider.history("AAPL", date(2026, 10, 27))

    assert [(c.date, c.volume) for c in history.candles] == [(date(2026, 10, 28), 0.0)]


def test_history_extracts_dividends_and_splits() -> None:
    rows = [
        ("2026-10-23", row(10.0, split=0.1)),
        ("2026-10-26", row(11.0, split=4.0)),
        ("2026-10-27", row(12.0, dividend=0.25)),
    ]
    provider, _ = provider_for(FakeTicker(frame(rows)))

    history = provider.history("AAPL", None)

    assert history.dividends == [Dividend(date(2026, 10, 27), 0.25)]
    assert history.splits == [
        Split(date(2026, 10, 23), 0.1),
        Split(date(2026, 10, 26), 4.0),
    ]


def test_history_dates_are_in_the_exchange_time_zone() -> None:
    rows = [("2026-10-28", row(10.0))]
    provider, _ = provider_for(FakeTicker(frame(rows, tz="Europe/Amsterdam")))

    history = provider.history("ASML.AS", None)

    assert [c.date for c in history.candles] == [date(2026, 10, 28)]


def test_unknown_ticker_raises_unknown_ticker() -> None:
    error = YFTzMissingError("NOSUCH")
    provider, _ = provider_for(FakeTicker(error=error))

    with pytest.raises(UnknownTicker):
        provider.history("NOSUCH", None)


def test_missing_prices_after_start_is_an_empty_history() -> None:
    error = YFTzMissingError("AAPL")
    provider, _ = provider_for(FakeTicker(error=error))

    history = provider.history("AAPL", date(2026, 10, 29))

    assert history == ([], [], [])


@pytest.mark.parametrize("error", [YFRateLimitError(), ConnectionError("reset")])
def test_other_errors_raise_provider_unavailable(error: Exception) -> None:
    provider, _ = provider_for(FakeTicker(error=error))

    with pytest.raises(ProviderUnavailable) as raised:
        provider.history("AAPL", None)
    assert raised.value.__cause__ is error


def test_listing_maps_info() -> None:
    info = {"currency": "EUR", "quoteType": "EQUITY", "longName": "ASML Holding N.V."}
    provider, requested = provider_for(FakeTicker(info=info))

    assert provider.listing("asml.as") == Listing(
        "ASML.AS", "ASML Holding N.V.", "EUR", "XAMS"
    )
    assert requested == ["ASML.AS"]


def test_listing_falls_back_to_short_name() -> None:
    info = {"currency": "USD", "quoteType": "EQUITY", "shortName": "Apple"}
    provider, _ = provider_for(FakeTicker(info=info))

    assert provider.listing("AAPL").name == "Apple"


def test_listing_of_unknown_ticker_raises_unknown_ticker() -> None:
    provider, _ = provider_for(FakeTicker(info={"trailingPegRatio": None}))

    with pytest.raises(UnknownTicker):
        provider.listing("NOSUCH")


def test_unsupported_suffix_is_rejected_before_any_call() -> None:
    provider, requested = provider_for(FakeTicker())

    with pytest.raises(UnsupportedExchange):
        provider.listing("VOD.L")
    with pytest.raises(UnsupportedExchange):
        provider.history("VOD.L", None)
    assert requested == []


def test_yfinance_raises_instead_of_logging() -> None:
    import yfinance

    YFinanceProvider()

    assert yfinance.config.debug.hide_exceptions is False


class HTTPError(OSError):
    """Stands in for curl_cffi's HTTPError: it carries the response."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP Error {status_code}")
        self.response = type("Response", (), {"status_code": status_code})()


def test_http_404_means_unknown_ticker() -> None:
    provider, _ = provider_for(FakeTicker(error=HTTPError(404)))

    with pytest.raises(UnknownTicker):
        provider.history("NOSUCH", None)


def test_other_http_errors_mean_unavailable() -> None:
    provider, _ = provider_for(FakeTicker(error=HTTPError(503)))

    with pytest.raises(ProviderUnavailable):
        provider.history("AAPL", None)


QUOTE_INFO: dict[str, Any] = {
    "currency": "USD",
    "quoteType": "EQUITY",
    "regularMarketPrice": 270.5,
    "bid": 270.4,
    "ask": 270.6,
    "regularMarketVolume": 1_000_000,
    "regularMarketTime": int(datetime(2026, 10, 29, 20, 0, tzinfo=UTC).timestamp()),
    "exchangeDataDelayedBy": 0,
}


def test_quote_maps_info() -> None:
    provider, _ = provider_for(FakeTicker(info=QUOTE_INFO))

    assert provider.quote("AAPL") == Quote(
        last=270.5,
        bid=270.4,
        ask=270.6,
        volume=1_000_000.0,
        time=datetime(2026, 10, 29, 20, 0, tzinfo=UTC),
        delay_minutes=0,
    )


def test_zero_or_missing_bid_and_ask_become_none() -> None:
    info = {k: v for k, v in QUOTE_INFO.items() if k != "ask"} | {"bid": 0.0}
    provider, _ = provider_for(FakeTicker(info=info))

    quote = provider.quote("AAPL")

    assert (quote.bid, quote.ask) == (None, None)


def test_quote_of_unknown_ticker_raises_unknown_ticker() -> None:
    provider, _ = provider_for(FakeTicker(info={"trailingPegRatio": None}))

    with pytest.raises(UnknownTicker):
        provider.quote("NOSUCH")


def test_quote_errors_raise_provider_unavailable() -> None:
    class BrokenTicker(FakeTicker):
        @property
        def info(self) -> dict[str, Any]:
            raise ConnectionError("reset")

        @info.setter
        def info(self, value: dict[str, Any]) -> None:
            pass

    provider, _ = provider_for(BrokenTicker())

    with pytest.raises(ProviderUnavailable):
        provider.quote("AAPL")


def test_quote_without_a_time_is_unavailable() -> None:
    info = {k: v for k, v in QUOTE_INFO.items() if k != "regularMarketTime"}
    provider, _ = provider_for(FakeTicker(info=info))

    with pytest.raises(ProviderUnavailable):
        provider.quote("AAPL")
