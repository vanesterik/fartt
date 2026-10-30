from datetime import UTC, date, datetime

import pytest

from fartt.data import (
    UnknownTicker,
    UnsupportedExchange,
    calendar_for,
    missing_sessions,
    newest_closed_session,
    normalise_ticker,
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
