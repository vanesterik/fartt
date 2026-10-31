from datetime import date
from itertools import pairwise
from statistics import mean
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from talib import ATR, EMA, MACD, RSI, SMA

from fartt.data import Candle
from fartt.features.technical_indicators_config import TechnicalIndicatorsConfig

DEFAULT_CONFIG = TechnicalIndicatorsConfig()

Direction = Literal["rising", "falling", "flat"]
Zone = Literal["oversold", "neutral", "overbought"]
Array = npt.NDArray[np.float64]

# A moving average's slope compares with its value this many sessions earlier.
SLOPE_SESSIONS = 5
# The RSI's direction compares with its value this many sessions earlier.
RSI_DIRECTION_SESSIONS = 3
ATR_PERIOD = 14
AVERAGE_SESSIONS = 20
# Down days against up days over this many of the latest sessions.
DOWN_DAY_SESSIONS = 10
# Floats are rounded to this many decimals: digits past it mean nothing to
# the agent, and they would cost the output its size budget.
DECIMALS = 4


class MovingAverage(BaseModel):
    value: float = Field(description="The average's value.")
    price_vs: Literal["above", "below"] = Field(
        description="Where the close is against the average (on it counts as above)."
    )
    distance: float = Field(
        description="(close - average) / average: -0.03 means 3% below."
    )
    slope: Direction = Field(
        description=f"The average against its value {SLOPE_SESSIONS} sessions earlier."
    )
    days: int = Field(
        description="Sessions in a row, the last included, the close has been on this side."
    )


class Rsi(BaseModel):
    value: float = Field(description="RSI, 0-100.")
    zone: Zone = Field(description="Oversold below 30, overbought above 70.")
    days: int = Field(description="Sessions in a row, the last included, in this zone.")
    direction: Direction | None = Field(
        description=(
            f"The RSI against its value {RSI_DIRECTION_SESSIONS} sessions earlier; "
            "null when that one isn't computed yet."
        )
    )


class Cross(BaseModel):
    kind: Literal["bullish", "bearish"] = Field(
        description="Bullish: the MACD line crossed above its signal line."
    )
    sessions_ago: int = Field(description="0 means on the last session.")


class Macd(BaseModel):
    line: float = Field(description="The MACD line: the fast EMA minus the slow.")
    signal: float = Field(description="The signal line: an EMA of the MACD line.")
    histogram: float = Field(description="line - signal.")
    histogram_direction: Direction | None = Field(
        description="The histogram against the previous session's."
    )
    last_cross: Cross | None = Field(
        description="The line's latest cross of the signal; null if it never crossed."
    )


class Atr(BaseModel):
    value: float = Field(description="The average true range, in price.")
    fraction: float | None = Field(
        description="ATR / close: a day's typical move. Null on a zero close."
    )
    stretch: float | None = Field(
        description=(
            "(close - SMA 20) / ATR: how many ATRs price has run from its 20-day "
            "average. -2 is a dip twice the typical daily move below it."
        )
    )


class Volume(BaseModel):
    last: float = Field(description="The last session's volume.")
    average_20d: float = Field(
        description="Mean volume of the 20 sessions before the last."
    )
    ratio: float | None = Field(description="last / average_20d.")
    down_day_ratio: float | None = Field(
        description=(
            f"Mean volume on down days over the last {DOWN_DAY_SESSIONS} sessions "
            "divided by the mean on up days, ex-dividend sessions left out. Above "
            "1 means heavier selling; null without both kinds of day."
        )
    )


class Indicators(BaseModel):
    close: float = Field(description="The last session's close.")
    sma_20: MovingAverage | None
    sma_50: MovingAverage | None
    sma_200: MovingAverage | None
    ema_20: MovingAverage | None
    rsi: Rsi | None
    macd: Macd | None
    atr: Atr | None
    volume: Volume | None
    note: str | None = Field(
        default=None,
        description="Which indicators are null and why, if any.",
    )


def _round(value: float) -> float:
    return round(float(value), DECIMALS)


def _direction(now: float, before: float) -> Direction:
    # On rounded values: TA-Lib's running sums leave noise in the last digits,
    # which would turn a flat stretch into rising or falling.
    now, before = _round(now), _round(before)
    if now > before:
        return "rising"
    if now < before:
        return "falling"
    return "flat"


def _streak(flags: list[bool]) -> int:
    """How many of the last flags, the last included, equal the last."""
    count = 0
    for flag in reversed(flags):
        if flag != flags[-1]:
            break
        count += 1
    return count


def _moving_average(closes: Array, values: Array) -> MovingAverage:
    close, value = _round(closes[-1]), _round(values[-1])
    defined = ~np.isnan(values)
    # Rounded, so a close on its average counts as above despite float noise.
    above = [
        _round(c) >= _round(v)
        for c, v in zip(closes[defined], values[defined], strict=True)
    ]
    return MovingAverage(
        value=value,
        price_vs="above" if above[-1] else "below",
        distance=_round((close - value) / value),
        slope=_direction(value, float(values[-1 - SLOPE_SESSIONS])),
        days=_streak(above),
    )


def _zone(value: float, config: TechnicalIndicatorsConfig) -> Zone:
    if value < config.rsi.oversold:
        return "oversold"
    if value > config.rsi.overbought:
        return "overbought"
    return "neutral"


def _rsi(values: Array, config: TechnicalIndicatorsConfig) -> Rsi:
    defined = [float(v) for v in values if not np.isnan(v)]
    zones: list[Zone] = [_zone(v, config) for v in defined]
    earlier = (
        defined[-1 - RSI_DIRECTION_SESSIONS]
        if len(defined) > RSI_DIRECTION_SESSIONS
        else None
    )
    return Rsi(
        value=_round(defined[-1]),
        zone=zones[-1],
        days=_streak([zone == zones[-1] for zone in zones]),
        direction=_direction(defined[-1], earlier) if earlier is not None else None,
    )


def _macd(line: Array, signal: Array, histogram: Array) -> Macd:
    defined = ~np.isnan(histogram)
    above = [bool(h > 0) for h in histogram[defined]]
    cross = None
    for i in range(len(above) - 1, 0, -1):
        if above[i] != above[i - 1]:
            cross = Cross(
                kind="bullish" if above[i] else "bearish",
                sessions_ago=len(above) - 1 - i,
            )
            break
    return Macd(
        line=_round(line[-1]),
        signal=_round(signal[-1]),
        histogram=_round(histogram[-1]),
        histogram_direction=(
            _direction(float(histogram[-1]), float(histogram[-2]))
            if len(above) > 1
            else None
        ),
        last_cross=cross,
    )


def _volume(candles: list[Candle], ex_dividend: frozenset[date]) -> Volume:
    last = candles[-1].volume
    average = mean(c.volume for c in candles[-1 - AVERAGE_SESSIONS : -1])
    down: list[float] = []
    up: list[float] = []
    recent = candles[-1 - DOWN_DAY_SESSIONS :]
    for previous, candle in pairwise(recent):
        if candle.date in ex_dividend:
            continue
        if candle.close < previous.close:
            down.append(candle.volume)
        elif candle.close > previous.close:
            up.append(candle.volume)
    up_mean = mean(up) if up else 0.0
    return Volume(
        last=_round(last),
        average_20d=_round(average),
        ratio=_round(last / average) if average > 0 else None,
        down_day_ratio=_round(mean(down) / up_mean) if down and up_mean > 0 else None,
    )


def calculate_indicators(
    candles: list[Candle],
    ex_dividend: frozenset[date] = frozenset(),
    config: TechnicalIndicatorsConfig = DEFAULT_CONFIG,
) -> Indicators:
    """
    Moving averages, RSI, MACD, ATR and volume as of the last of `candles`
    (oldest first), each with its state. An indicator needing more sessions
    than there are is `None`, and the note says how many it needs.
    `ex_dividend` holds the ex-dividend dates: prices aren't adjusted for
    dividends, so their drop isn't counted as selling.

    """
    count = len(candles)
    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    notes: list[str] = []

    def enough(name: str, needs: int) -> bool:
        if count < needs:
            notes.append(f"{name} needs {needs} sessions, {count} are cached.")
            return False
        return True

    averages: dict[str, MovingAverage | None] = {}
    for name, average, period in [
        ("sma_20", SMA, 20),
        ("sma_50", SMA, 50),
        ("sma_200", SMA, 200),
        ("ema_20", EMA, 20),
    ]:
        averages[name] = (
            _moving_average(closes, average(closes, timeperiod=period))
            if enough(name, period + SLOPE_SESSIONS)
            else None
        )

    rsi = None
    if enough("rsi", config.rsi.period + 1):
        recent = closes[-1 - config.rsi.period :]
        if np.all(recent == recent[0]):
            # No gains and no losses over the period, as on a suspended stock:
            # TA-Lib keeps the old reading or reports 0 (oversold), and either
            # would look real.
            notes.append(
                f"rsi is undefined: the close hasn't moved in {len(recent)} sessions."
            )
        else:
            rsi = _rsi(RSI(closes, timeperiod=config.rsi.period), config)

    macd = None
    warm_up = config.macd.slow_period + config.macd.signal_period - 1
    if enough("macd", warm_up):
        macd = _macd(
            *MACD(
                closes,
                fastperiod=config.macd.fast_period,
                slowperiod=config.macd.slow_period,
                signalperiod=config.macd.signal_period,
            )
        )

    atr = None
    if enough("atr", ATR_PERIOD + 1):
        value = float(ATR(highs, lows, closes, timeperiod=ATR_PERIOD)[-1])
        sma_20 = float(SMA(closes, timeperiod=20)[-1])
        close = float(closes[-1])
        atr = Atr(
            value=_round(value),
            fraction=_round(value / close) if close > 0 else None,
            stretch=(
                _round((close - sma_20) / value)
                if not np.isnan(sma_20) and value > 0
                else None
            ),
        )

    volume = (
        _volume(candles, ex_dividend)
        if enough("volume", AVERAGE_SESSIONS + 1)
        else None
    )

    return Indicators(
        close=_round(closes[-1]),
        **averages,
        rsi=rsi,
        macd=macd,
        atr=atr,
        volume=volume,
        note=" ".join(notes) or None,
    )
