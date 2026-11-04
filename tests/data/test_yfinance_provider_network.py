from datetime import UTC, date, datetime, timedelta

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


@pytest.mark.network
def test_dividends_are_split_adjusted_like_the_candles() -> None:
    # NVDA split 10-for-1 on 2024-06-10; its 2023 dividends were 0.04 a share,
    # 0.004 in today's shares, as its 2023 candles are.
    history = YFinanceProvider().history("NVDA", date(2023, 1, 1))

    in_2023 = [d.amount for d in history.dividends if d.date.year == 2023]
    assert in_2023
    assert all(amount == pytest.approx(0.004) for amount in in_2023)
