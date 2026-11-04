from datetime import date
from itertools import groupby
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from talib import SMA

from fartt.data import Candle, last_session_of_week
from fartt.technical.indicators import (
    SLOPE_SESSIONS,
    Array,
    MovingAverage,
    moving_average,
    round_value,
)

TrendDirection = Literal["up", "down", "sideways"]
Basis = Literal["sma_50_and_200", "sma_50_only"]


class TrendCross(BaseModel):
    kind: Literal["golden", "death"] = Field(
        description="Golden: the SMA 50 crossed above the SMA 200. Death: below."
    )
    date: str = Field(
        description="The period it crossed on (ISO 8601; a week's last session)."
    )


class TimeframeTrend(BaseModel):
    direction: TrendDirection | None = Field(
        description=(
            "Up: close > SMA 50 > SMA 200 with the SMA 50 rising. Down: the "
            "reverse, with it falling. Sideways otherwise. Null when there's "
            "too little history to judge."
        )
    )
    basis: Basis | None = Field(
        description=(
            "What the direction rests on: both averages, or the SMA 50 alone "
            "when the SMA 200 needs more history (close against the SMA 50 "
            "and its slope). Null when the direction isn't judged."
        )
    )
    based_on: str = Field(
        description="The date of the timeframe's last period (ISO 8601)."
    )
    close: float = Field(description="That period's close.")
    sma_50: MovingAverage | None
    sma_200: MovingAverage | None
    last_cross: TrendCross | None = Field(
        description=(
            "The SMA 50's latest cross of the SMA 200; null if they never "
            "crossed or the SMA 200 isn't computed."
        )
    )


class Trend(BaseModel):
    daily: TimeframeTrend
    weekly: TimeframeTrend | None = Field(
        description="Null when no week has closed yet."
    )
    aligned: bool = Field(
        description="True only when both timeframes are up, or both are down."
    )
    note: str | None = Field(
        default=None,
        description="What's missing and why, if anything.",
    )


def _week(candle: Candle) -> tuple[int, int]:
    year, week, _ = candle.date.isocalendar()
    return year, week


def _aggregate(sessions: list[Candle]) -> Candle:
    return Candle(
        sessions[-1].date,
        sessions[0].open,
        max(c.high for c in sessions),
        min(c.low for c in sessions),
        sessions[-1].close,
        sum(c.volume for c in sessions),
    )


def weekly_candles(
    candles: list[Candle], exchange: str
) -> tuple[list[Candle], date | None]:
    """
    Weekly candles (ISO weeks, oldest first) from daily `candles` (oldest
    first): the first open, highest high, lowest low, last close and summed
    volume, dated by the week's last session. A holiday week simply has fewer
    sessions. Closed weeks only: the newest week counts once its last
    scheduled session is among the candles, or when the calendar can't place
    it. Returns the weeks, and the date of the newest cached session when its
    week was left out (else `None`).

    """
    weeks = [list(group) for _, group in groupby(candles, key=_week)]
    if not weeks:
        return [], None
    newest = weeks[-1]
    last = last_session_of_week(exchange, newest[-1].date)
    unclosed = None
    if last is not None and last > newest[-1].date:
        unclosed = newest[-1].date
        weeks.pop()
    return [_aggregate(week) for week in weeks], unclosed


def _last_cross(
    candles: list[Candle], sma_50: Array, sma_200: Array
) -> TrendCross | None:
    # On rounded values, and only a change of side counts: periods where the
    # averages are equal are skipped, so a touch isn't a cross.
    last_sign = 0
    cross = None
    for candle, fast, slow in zip(candles, sma_50, sma_200, strict=True):
        if np.isnan(fast) or np.isnan(slow):
            continue
        diff = round_value(fast) - round_value(slow)
        sign = (diff > 0) - (diff < 0)
        if sign == 0:
            continue
        if last_sign and sign != last_sign:
            cross = TrendCross(
                kind="golden" if sign > 0 else "death",
                date=candle.date.isoformat(),
            )
        last_sign = sign
    return cross


def _direction(
    close: float, sma_50: MovingAverage, sma_200: MovingAverage | None
) -> TrendDirection:
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


def _timeframe(
    candles: list[Candle], name: str, unit: str, notes: list[str]
) -> TimeframeTrend:
    count = len(candles)
    closes = np.array([c.close for c in candles], dtype=np.float64)
    series: dict[int, Array] = {}
    averages: dict[int, MovingAverage | None] = {}
    for period in (50, 200):
        needs = period + SLOPE_SESSIONS
        if count >= needs:
            series[period] = SMA(closes, timeperiod=period)
            averages[period] = moving_average(closes, series[period])
        else:
            averages[period] = None
            notes.append(
                f"{name} sma_{period} needs {needs} {unit}, {count} are cached."
            )
    sma_50, sma_200 = averages[50], averages[200]
    close = round_value(closes[-1])

    direction: TrendDirection | None = None
    basis: Basis | None = None
    if sma_50 is None:
        notes.append(f"{name} direction isn't judged.")
    else:
        direction = _direction(close, sma_50, sma_200)
        if sma_200 is None:
            basis = "sma_50_only"
            notes.append(f"{name} direction rests on the sma_50 alone.")
        else:
            basis = "sma_50_and_200"

    return TimeframeTrend(
        direction=direction,
        basis=basis,
        based_on=candles[-1].date.isoformat(),
        close=close,
        sma_50=sma_50,
        sma_200=sma_200,
        last_cross=(
            _last_cross(candles, series[50], series[200])
            if sma_200 is not None
            else None
        ),
    )


def calculate_trend(candles: list[Candle], exchange: str) -> Trend:
    """
    The trend on the daily and weekly timeframes from daily `candles` (oldest
    first, at least one) of `exchange`: each timeframe's direction with its
    evidence, and whether the two agree. Weekly candles are closed weeks only.
    Evidence needing more periods than there are is `None`, and the note says
    how many it needs.

    """
    notes: list[str] = []
    daily = _timeframe(candles, "daily", "sessions", notes)
    weeks, unclosed = weekly_candles(candles, exchange)
    weekly = _timeframe(weeks, "weekly", "weeks", notes) if weeks else None
    if weekly is None:
        notes.append("No week has closed yet, so there is no weekly trend.")
    elif unclosed is not None:
        # Not "the current week": a stale cache stops in a week that closed.
        notes.append(
            f"The newest cached week (through {unclosed.isoformat()}) isn't "
            "complete: the weekly figures are as of the week ending "
            f"{weekly.based_on}."
        )
    aligned = (
        weekly is not None
        and daily.direction == weekly.direction
        and daily.direction in ("up", "down")
    )
    return Trend(
        daily=daily,
        weekly=weekly,
        aligned=aligned,
        note=" ".join(notes) or None,
    )
