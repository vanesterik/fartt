import warnings
from datetime import UTC, date, datetime

import pytest
from pandas.errors import Pandas4Warning

from fartt.data import (
    MarketStatus,
    UnknownTicker,
    UnsupportedExchange,
    calendar_for,
    exchange_calendar,
    exchange_date,
    fx_pair,
    last_session_of_week,
    market_status,
    missing_sessions,
    newest_closed_session,
    normalise_ticker,
    session_after,
)


def utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_calendar_for_maps_suffixes() -> None:
    assert calendar_for("AAPL") == "XNYS"
    assert calendar_for("ASML.AS") == "XAMS"
    assert calendar_for("MC.PA") == "XPAR"
    assert calendar_for("asml.as") == "XAMS"


def test_calendar_for_rejects_unsupported_suffix() -> None:
    with pytest.raises(UnsupportedExchange, match=r"\.AS"):
        calendar_for("VOD.L")


def test_session_is_not_closed_within_the_margin() -> None:
    # XNYS closes 16:00 New York = 20:00 UTC on 2026-10-29 (EDT).
    assert newest_closed_session("XNYS", utc(2026, 10, 29, 20, 29)) == date(
        2026, 10, 28
    )
    assert newest_closed_session("XNYS", utc(2026, 10, 29, 20, 30)) == date(
        2026, 10, 29
    )


def test_weekend_falls_back_to_friday() -> None:
    assert newest_closed_session("XAMS", utc(2026, 10, 25, 12, 0)) == date(2026, 10, 23)


def test_holidays_are_not_sessions() -> None:
    # Good Friday 2026-04-03 on Euronext; Thanksgiving 2026-11-26 on NYSE.
    assert newest_closed_session("XAMS", utc(2026, 4, 3, 20, 0)) == date(2026, 4, 2)
    assert newest_closed_session("XNYS", utc(2026, 11, 26, 23, 0)) == date(2026, 11, 25)


def test_early_close_counts() -> None:
    # NYSE closes at 13:00 New York (18:00 UTC) on 2026-11-27, after Thanksgiving.
    assert newest_closed_session("XNYS", utc(2026, 11, 27, 18, 30)) == date(
        2026, 11, 27
    )


def test_normalise_ticker_strips_and_upper_cases() -> None:
    assert normalise_ticker(" asml.as ") == "ASML.AS"
    assert normalise_ticker("brk-b") == "BRK-B"


@pytest.mark.parametrize("ticker", ["", "../X", "A/B", "AAPL;rm"])
def test_normalise_ticker_rejects_what_cannot_be_a_ticker(ticker: str) -> None:
    with pytest.raises(UnknownTicker):
        normalise_ticker(ticker)


def test_calendar_for_ignores_surrounding_whitespace() -> None:
    assert calendar_for(" asml.as ") == "XAMS"


def test_holidays_and_weekends_are_not_gaps() -> None:
    # Euronext is closed on Good Friday and Easter Monday 2026; NYSE on Thanksgiving.
    assert missing_sessions("XAMS", [date(2026, 4, 2), date(2026, 4, 7)]) == []
    assert missing_sessions("XNYS", [date(2026, 11, 25), date(2026, 11, 27)]) == []


def test_a_missing_trading_day_is_a_gap() -> None:
    dates = [date(2026, 10, 26), date(2026, 10, 28)]

    assert missing_sessions("XNYS", dates) == [date(2026, 10, 27)]


def test_fewer_than_two_dates_have_no_gaps() -> None:
    assert missing_sessions("XNYS", []) == []
    assert missing_sessions("XNYS", [date(2026, 10, 26)]) == []


def test_dates_before_the_calendar_starts_are_not_checked() -> None:
    dates = [date(1980, 12, 12), date(2026, 10, 26), date(2026, 10, 27)]

    gaps = missing_sessions("XNYS", dates)

    assert gaps
    assert min(gaps).year >= 2000


def test_market_status_during_the_session() -> None:
    assert market_status("XNYS", utc(2026, 10, 29, 18, 0)) == MarketStatus(True, None)


def test_market_status_after_the_close() -> None:
    assert market_status("XNYS", utc(2026, 10, 29, 20, 30)) == MarketStatus(
        False, utc(2026, 10, 30, 13, 30)
    )


def test_market_status_on_a_weekend() -> None:
    # US daylight saving time ends on 2026-11-01, so Monday opens an hour later.
    assert market_status("XNYS", utc(2026, 10, 31, 12, 0)) == MarketStatus(
        False, utc(2026, 11, 2, 14, 30)
    )


def test_market_status_on_a_holiday() -> None:
    # Thanksgiving.
    assert market_status("XNYS", utc(2026, 11, 26, 15, 0)) == MarketStatus(
        False, utc(2026, 11, 27, 14, 30)
    )


def test_market_status_in_amsterdam() -> None:
    assert market_status("XAMS", utc(2026, 10, 29, 10, 0)).open is True


def test_exchange_date_is_the_date_on_the_exchange() -> None:
    # 01:00 UTC on the 30th is still the 29th in New York.
    assert exchange_date("XNYS", utc(2026, 10, 30, 1, 0)) == date(2026, 10, 29)
    assert exchange_date("XAMS", utc(2026, 10, 29, 23, 30)) == date(2026, 10, 30)


def test_last_session_of_a_normal_week() -> None:
    assert last_session_of_week("XAMS", date(2026, 10, 28)) == date(2026, 10, 30)


def test_last_session_of_a_week_ending_on_a_holiday() -> None:
    # Good Friday 2026 is 3 April: Euronext is closed.
    assert last_session_of_week("XAMS", date(2026, 3, 31)) == date(2026, 4, 2)


def test_a_weekend_day_belongs_to_its_iso_week() -> None:
    # Sunday 1 November is in the ISO week of Monday 26 October.
    assert last_session_of_week("XNYS", date(2026, 11, 1)) == date(2026, 10, 30)


def test_a_week_outside_the_calendar_is_unknown() -> None:
    assert last_session_of_week("XAMS", date(2056, 10, 28)) is None


def test_session_after_skips_the_weekend() -> None:
    assert session_after("XAMS", date(2026, 10, 30), 1) == date(2026, 11, 2)


def test_session_after_skips_thanksgiving() -> None:
    # Thanksgiving, 26 November 2026: NYSE is closed.
    assert session_after("XNYS", date(2026, 11, 20), 5) == date(2026, 11, 30)


def test_session_after_skips_easter() -> None:
    # Euronext closes on Good Friday 26 March and Easter Monday 29 March 2027.
    assert session_after("XAMS", date(2027, 3, 25), 1) == date(2027, 3, 30)


def test_session_after_reaches_60_sessions() -> None:
    expected = exchange_calendar("XNYS").session_offset("2026-11-04", 60).date()
    assert session_after("XNYS", date(2026, 11, 4), 60) == expected


def test_calendars_raise_no_pandas_deprecation_warning() -> None:
    # exchange_calendars 4.13.2 reads .values on time-zone-aware data, which
    # pandas 3.1 deprecates, building a calendar and its first minute lookup;
    # the warning is theirs, not fartt's.
    exchange_calendar.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", Pandas4Warning)
        newest_closed_session("XAMS", utc(2026, 10, 29, 21, 0))
        market_status("XAMS", utc(2026, 10, 29, 10, 0))


def test_an_fx_pair_is_a_ticker() -> None:
    assert normalise_ticker(" eurusd=x ") == "EURUSD=X"
    assert calendar_for("EURUSD=X") == "24/5"


@pytest.mark.parametrize("ticker", ["A/B", "..\\X", "../x"])
def test_paths_are_still_rejected(ticker: str) -> None:
    with pytest.raises(UnknownTicker):
        normalise_ticker(ticker)


def test_fx_pair() -> None:
    assert fx_pair("EUR") is None
    assert fx_pair("USD") == "EURUSD=X"
    assert fx_pair("NOK") == "EURNOK=X"


def test_the_24_5_calendar_closes_a_day_at_midnight_utc() -> None:
    assert newest_closed_session("24/5", utc(2026, 11, 4, 0, 29)) == date(2026, 11, 2)
    assert newest_closed_session("24/5", utc(2026, 11, 4, 0, 31)) == date(2026, 11, 3)
