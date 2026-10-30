from datetime import UTC, datetime, timedelta

import pytest

from fartt.data import UnknownTicker, YFinanceProvider


@pytest.mark.network
@pytest.mark.parametrize("ticker", ["AAPL", "ASML.AS"])
def test_history_returns_recent_closed_sessions(ticker: str) -> None:
    history = YFinanceProvider().history(
        ticker, datetime.now(UTC).date() - timedelta(days=30)
    )

    dates = [c.date for c in history.candles]
    assert dates
    assert dates == sorted(dates)
    assert all(c.low <= c.close <= c.high for c in history.candles)


@pytest.mark.network
def test_unknown_ticker_is_reported() -> None:
    with pytest.raises(UnknownTicker):
        YFinanceProvider().history("NOSUCHTICKERXYZ", None)


@pytest.mark.network
def test_quote_returns_a_price_and_an_aware_time() -> None:
    quote = YFinanceProvider().quote("AAPL")

    assert quote.last > 0
    assert quote.time.tzinfo is not None
