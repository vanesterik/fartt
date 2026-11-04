from datetime import date

import pytest

from fartt.data import Candle
from fartt.technical import trend
from fartt.technical.trend import weekly_candles
from tests.fakes import stock_candle


def days(*dates: tuple[int, int, int]) -> list[date]:
    return [date(*d) for d in dates]


def test_a_week_aggregates_its_sessions() -> None:
    week = [
        Candle(date(2026, 10, 26), 10.0, 12.0, 9.0, 11.0, 100.0),
        Candle(date(2026, 10, 27), 11.0, 15.0, 10.0, 14.0, 200.0),
        Candle(date(2026, 10, 28), 14.0, 14.5, 8.0, 9.5, 300.0),
        Candle(date(2026, 10, 29), 9.5, 13.0, 9.0, 12.0, 400.0),
        Candle(date(2026, 10, 30), 12.0, 13.5, 11.0, 13.0, 500.0),
    ]

    weeks, unclosed = weekly_candles(week, "XAMS")

    assert weeks == [Candle(date(2026, 10, 30), 10.0, 15.0, 8.0, 13.0, 1500.0)]
    assert unclosed is None


def test_a_holiday_week_has_fewer_sessions() -> None:
    # Good Friday, 3 April 2026: Euronext is closed.
    sessions = days((2026, 3, 30), (2026, 3, 31), (2026, 4, 1), (2026, 4, 2))

    weeks, unclosed = weekly_candles([stock_candle(d) for d in sessions], "XAMS")

    assert [w.date for w in weeks] == [date(2026, 4, 2)]
    assert weeks[0].volume == 4000.0
    assert unclosed is None


def test_an_unclosed_week_is_left_out() -> None:
    sessions = days(
        (2026, 10, 26),
        (2026, 10, 27),
        (2026, 10, 28),
        (2026, 10, 29),
        (2026, 10, 30),
        (2026, 11, 2),
        (2026, 11, 3),
        (2026, 11, 4),
    )

    weeks, unclosed = weekly_candles([stock_candle(d) for d in sessions], "XAMS")

    assert [w.date for w in weeks] == [date(2026, 10, 30)]
    assert unclosed == date(2026, 11, 4)


def test_a_cache_that_stops_mid_week_leaves_that_week_out() -> None:
    sessions = days(
        (2026, 10, 19),
        (2026, 10, 20),
        (2026, 10, 21),
        (2026, 10, 22),
        (2026, 10, 23),
        (2026, 10, 26),
        (2026, 10, 27),
    )

    weeks, unclosed = weekly_candles([stock_candle(d) for d in sessions], "XAMS")

    assert weeks[-1].date == date(2026, 10, 23)
    assert unclosed == date(2026, 10, 27)


def test_a_week_the_calendar_cannot_place_counts_as_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trend, "last_session_of_week", lambda _exchange, _day: None)
    sessions = days((2026, 11, 2), (2026, 11, 3))

    weeks, unclosed = weekly_candles([stock_candle(d) for d in sessions], "XAMS")

    assert [w.date for w in weeks] == [date(2026, 11, 3)]
    assert unclosed is None
