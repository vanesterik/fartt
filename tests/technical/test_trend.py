import random
from datetime import date

import numpy as np
import pytest
import talib

from fartt.data import Candle, exchange_calendar
from fartt.technical import trend
from fartt.technical.trend import TimeframeTrend, calculate_trend, weekly_candles
from tests.fakes import stock_candle

# A Friday: the series' last week has closed.
FRIDAY = date(2026, 10, 30)
LONG = 1100  # sessions: about 220 weeks, enough for the weekly SMA 200


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


def sessions(count: int, last: date = FRIDAY) -> list[date]:
    """The `count` XNYS sessions up to and including `last`."""
    window = exchange_calendar("XNYS").sessions_window(last.isoformat(), -count)
    return [session.date() for session in window]


def series(closes: list[float], last: date = FRIDAY) -> list[Candle]:
    return [
        stock_candle(day, close)
        for day, close in zip(sessions(len(closes), last), closes, strict=True)
    ]


def rise(count: int = LONG) -> list[float]:
    return [100.0 + i for i in range(count)]


def fall(count: int = LONG) -> list[float]:
    return [100.0 + count - i for i in range(count)]


def random_walk(count: int = LONG) -> list[float]:
    rng = random.Random(7)
    closes = [100.0]
    for _ in range(count - 1):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.015)))
    return closes


def expected_cross(closes: list[float]) -> tuple[str, int] | None:
    """The latest change of side of the rounded SMA 50 against the SMA 200."""
    array = np.array(closes)
    sma_50 = talib.SMA(array, timeperiod=50)
    sma_200 = talib.SMA(array, timeperiod=200)
    last_sign, cross = 0, None
    for i in range(199, len(closes)):
        diff = round(float(sma_50[i]), 4) - round(float(sma_200[i]), 4)
        sign = (diff > 0) - (diff < 0)
        if sign == 0:
            continue
        if last_sign and sign != last_sign:
            cross = ("golden" if sign > 0 else "death", i)
        last_sign = sign
    return cross


def rule(timeframe: TimeframeTrend) -> str | None:
    """The direction AC2 expects from the reported evidence."""
    close, sma_50, sma_200 = timeframe.close, timeframe.sma_50, timeframe.sma_200
    if sma_50 is None:
        return None
    if sma_200 is None:
        up = close > sma_50.value
        down = close < sma_50.value
    else:
        up = close > sma_50.value > sma_200.value
        down = close < sma_50.value < sma_200.value
    if up and sma_50.slope == "rising":
        return "up"
    if down and sma_50.slope == "falling":
        return "down"
    return "sideways"


def test_a_steady_rise_is_up_on_both_timeframes() -> None:
    result = calculate_trend(series(rise()), "XNYS")

    assert result.weekly is not None
    for timeframe in (result.daily, result.weekly):
        assert timeframe.direction == "up"
        assert timeframe.basis == "sma_50_and_200"
        assert timeframe.based_on == FRIDAY.isoformat()
    assert result.daily.close == 100.0 + LONG - 1
    assert result.aligned is True
    assert result.note is None


def test_a_steady_fall_is_down() -> None:
    result = calculate_trend(series(fall()), "XNYS")

    assert result.weekly is not None
    assert result.daily.direction == "down"
    assert result.weekly.direction == "down"
    assert result.aligned is True


def test_a_rise_then_a_fall_crosses_to_death() -> None:
    closes = rise(600) + [700.0 - i for i in range(400)]
    candles = series(closes)

    result = calculate_trend(candles, "XNYS")

    cross = expected_cross(closes)
    assert cross is not None
    kind, index = cross
    assert kind == "death"
    assert result.daily.last_cross is not None
    assert result.daily.last_cross.kind == "death"
    assert result.daily.last_cross.date == candles[index].date.isoformat()


def test_a_touch_without_a_change_of_side_is_not_a_cross() -> None:
    # Down, up (golden), flat until both averages meet, then up again: the
    # averages touch, and the SMA 50 leaves on the side it came from.
    closes = fall(300) + [1.0 + i for i in range(300)] + [300.0] * 260
    closes += [300.0 + i for i in range(60)]
    candles = series(closes)

    result = calculate_trend(candles, "XNYS")

    cross = expected_cross(closes)
    assert cross is not None
    assert cross[0] == "golden"
    assert cross[1] < 600  # the golden cross of the rise, not the touch
    assert result.daily.last_cross is not None
    assert result.daily.last_cross.kind == "golden"
    assert result.daily.last_cross.date == candles[cross[1]].date.isoformat()


def test_a_flat_sma_50_is_sideways() -> None:
    result = calculate_trend(series([100.0] * LONG), "XNYS")

    assert result.weekly is not None
    for timeframe in (result.daily, result.weekly):
        assert timeframe.direction == "sideways"
        assert timeframe.sma_50 is not None
        assert timeframe.sma_50.slope == "flat"
        assert timeframe.last_cross is None
    assert result.aligned is False


@pytest.mark.parametrize("closes", [rise(), fall(), random_walk()])
def test_the_direction_follows_from_its_evidence(closes: list[float]) -> None:
    result = calculate_trend(series(closes), "XNYS")

    assert result.weekly is not None
    for timeframe in (result.daily, result.weekly):
        assert timeframe.direction == rule(timeframe)
    assert result.aligned == (
        result.daily.direction == result.weekly.direction
        and result.daily.direction in ("up", "down")
    )


def test_too_few_weeks_rest_on_the_sma_50() -> None:
    result = calculate_trend(series(rise(300)), "XNYS")

    assert result.weekly is not None
    assert result.weekly.sma_200 is None
    assert result.weekly.basis == "sma_50_only"
    assert result.weekly.last_cross is None
    assert result.weekly.direction == "up"
    assert result.note is not None
    assert "weekly sma_200 needs 205 weeks, " in result.note


def test_too_few_periods_leave_the_direction_unjudged() -> None:
    result = calculate_trend(series(rise(40)), "XNYS")

    assert result.daily.direction is None
    assert result.daily.basis is None
    assert result.daily.sma_50 is None
    assert result.aligned is False
    assert result.note is not None
    assert "daily sma_50 needs 55 sessions, 40 are cached." in result.note


def test_no_closed_week_leaves_weekly_null() -> None:
    candles = series(rise(3), last=date(2026, 11, 4))  # Monday to Wednesday

    result = calculate_trend(candles, "XNYS")

    assert result.weekly is None
    assert result.aligned is False
    assert result.note is not None
    assert "No week has closed yet" in result.note


def test_an_unclosed_week_is_in_the_note() -> None:
    result = calculate_trend(series(rise(), last=date(2026, 11, 4)), "XNYS")

    assert result.weekly is not None
    assert result.weekly.based_on == "2026-10-30"
    assert result.daily.based_on == "2026-11-04"
    assert result.note is not None
    assert "week ending 2026-10-30" in result.note


def test_a_stale_cache_mid_week_is_not_called_the_current_week() -> None:
    # Cached through Tuesday 27 October: that week has closed, it just isn't
    # all in the cache, so the note mustn't call it the current week.
    result = calculate_trend(series(rise(), last=date(2026, 10, 27)), "XNYS")

    assert result.note is not None
    assert "current week" not in result.note
    assert (
        "The newest cached week (through 2026-10-27) isn't complete: the weekly "
        "figures are as of the week ending 2026-10-23."
    ) in result.note
