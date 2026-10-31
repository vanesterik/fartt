import random
from datetime import date, timedelta
from itertools import pairwise
from statistics import mean

import numpy as np
import pytest
import talib

from fartt.data import Candle
from fartt.technical import calculate_indicators
from tests.fakes import stock_candle

START = date(2020, 1, 1)


def candles(closes: list[float], volumes: list[float] | None = None) -> list[Candle]:
    """Candles on consecutive days, one per close."""
    volumes = volumes or [1000.0] * len(closes)
    return [
        stock_candle(START + timedelta(days=i), close, volume)
        for i, (close, volume) in enumerate(zip(closes, volumes, strict=True))
    ]


def rising(count: int) -> list[float]:
    return [100.0 + i for i in range(count)]


def wilder(values: list[float], period: int) -> float:
    """Wilder's smoothing as TA-Lib seeds it: a simple mean, then (n-1)/n."""
    average = mean(values[:period])
    for value in values[period:]:
        average = (average * (period - 1) + value) / period
    return average


def test_sma_matches_a_hand_computed_mean() -> None:
    result = calculate_indicators(candles([float(i) for i in range(1, 31)]))

    assert result.sma_20 is not None
    assert result.sma_20.value == pytest.approx(mean(range(11, 31)))


def test_ema_matches_a_hand_computation() -> None:
    closes = [100.0 + (i % 7) * 1.5 - i * 0.2 for i in range(30)]

    result = calculate_indicators(candles(closes))

    ema = mean(closes[:20])
    for close in closes[20:]:
        ema += 2 / 21 * (close - ema)
    assert result.ema_20 is not None
    assert result.ema_20.value == pytest.approx(ema, abs=1e-4)


def test_rsi_matches_wilders_smoothing() -> None:
    closes = [100.0, 101.5, 100.8, 102.0, 101.0, 103.2, 102.5, 104.0, 103.1, 102.2]
    closes += [101.0, 102.8, 103.5, 102.9, 104.4, 103.0, 102.1, 103.6, 104.8, 104.0]
    changes = [b - a for a, b in pairwise(closes)]

    result = calculate_indicators(candles(closes))

    gain = wilder([max(c, 0.0) for c in changes], 14)
    loss = wilder([max(-c, 0.0) for c in changes], 14)
    assert result.rsi is not None
    assert result.rsi.value == pytest.approx(100 * gain / (gain + loss), abs=1e-4)


def test_atr_matches_wilders_smoothing() -> None:
    closes = [100.0, 102.0, 99.0, 101.5, 98.0, 103.0, 100.0, 104.0, 101.0, 99.5]
    closes += [102.5, 100.5, 105.0, 101.0, 103.5, 99.0, 104.5, 102.0, 106.0, 103.0]
    series = candles(closes)

    result = calculate_indicators(series)

    true_ranges = [
        max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        for p, c in pairwise(series)
    ]
    assert result.atr is not None
    assert result.atr.value == pytest.approx(wilder(true_ranges, 14), abs=1e-4)
    assert result.atr.fraction == pytest.approx(result.atr.value / 103.0, abs=1e-4)


def test_macd_histogram_is_line_minus_signal() -> None:
    closes = [100.0 + i + (i % 4) * 0.7 for i in range(60)]

    result = calculate_indicators(candles(closes))

    assert result.macd is not None
    assert result.macd.line > 0
    assert result.macd.histogram == pytest.approx(
        result.macd.line - result.macd.signal, abs=2e-4
    )


def test_a_steady_uptrend() -> None:
    result = calculate_indicators(candles(rising(250)))

    for average, period in [
        (result.sma_20, 20),
        (result.sma_50, 50),
        (result.sma_200, 200),
        (result.ema_20, 20),
    ]:
        assert average is not None
        assert average.price_vs == "above"
        assert average.slope == "rising"
        assert average.days == 250 - period + 1
        assert average.distance > 0
    assert result.rsi is not None
    assert result.rsi.zone == "overbought"
    assert result.close == 349.0
    assert result.note is None


def test_a_dip_reads_oversold_and_stretched() -> None:
    closes = [100.0 + (i % 2) * 0.5 for i in range(60)] + [
        98.0 - 2 * i for i in range(10)
    ]

    result = calculate_indicators(candles(closes))

    assert result.rsi is not None
    assert result.rsi.zone == "oversold"
    assert result.rsi.days >= 1
    assert result.rsi.direction == "falling"
    assert result.atr is not None
    assert result.atr.stretch is not None
    assert result.atr.stretch < 0
    assert result.sma_20 is not None
    assert result.sma_20.price_vs == "below"


def test_a_macd_cross_is_dated() -> None:
    closes = [
        100.0 + 0.05 * i * i for i in range(60)
    ]  # Accelerating: line above signal
    closes += [closes[-1] - 2 * i for i in range(1, 11)]

    result = calculate_indicators(candles(closes))

    line, signal, _ = talib.MACD(np.array(closes), 12, 26, 9)
    above = line > signal
    crossed = max(i for i in range(34, len(closes)) if above[i] != above[i - 1])
    assert result.macd is not None
    assert result.macd.last_cross is not None
    assert result.macd.last_cross.kind == "bearish"
    assert result.macd.last_cross.sessions_ago == len(closes) - 1 - crossed
    assert result.macd.histogram_direction == "falling"


def test_volume_against_the_20_before_the_last() -> None:
    result = calculate_indicators(candles(rising(21), [1000.0] * 20 + [3000.0]))

    assert result.volume is not None
    assert result.volume.last == 3000.0
    assert result.volume.average_20d == 1000.0
    assert result.volume.ratio == 3.0


def alternating(count: int) -> list[float]:
    """Closes that fall on even sessions and rise on odd ones."""
    return [100.0 + (i % 2) for i in range(count)]


def test_down_day_ratio_weighs_selling() -> None:
    closes = alternating(21)  # Even sessions fall, odd ones rise
    volumes = [2000.0 if i % 2 == 0 else 1000.0 for i in range(21)]

    result = calculate_indicators(candles(closes, volumes))

    assert result.volume is not None
    assert result.volume.down_day_ratio == 2.0


def test_an_ex_dividend_session_is_not_a_down_day() -> None:
    closes = alternating(21)
    volumes = [1000.0] * 21
    volumes[18] = 5000.0  # A down day: its close is below session 17's
    series = candles(closes, volumes)

    result = calculate_indicators(series, ex_dividend=frozenset({series[18].date}))

    assert result.volume is not None
    assert result.volume.down_day_ratio == 1.0


def test_zero_volume_leaves_the_ratios_null() -> None:
    result = calculate_indicators(candles(alternating(30), [0.0] * 30))

    assert result.volume is not None
    assert result.volume.average_20d == 0.0
    assert result.volume.ratio is None
    assert result.volume.down_day_ratio is None


def test_a_flat_series() -> None:
    result = calculate_indicators(candles([100.0] * 30))

    assert result.sma_20 is not None
    assert result.sma_20.price_vs == "above"
    assert result.sma_20.slope == "flat"
    assert result.rsi is None
    assert result.note is not None
    assert "rsi" in result.note


def test_a_short_history_is_null_with_a_note() -> None:
    result = calculate_indicators(candles(rising(30)))

    assert (result.sma_50, result.sma_200, result.macd) == (None, None, None)
    assert result.note is not None
    assert "sma_200 needs 205 sessions, 30 are cached." in result.note
    assert "macd needs 34 sessions" in result.note
    for block in (result.sma_20, result.ema_20, result.rsi, result.atr, result.volume):
        assert block is not None


@pytest.mark.parametrize(
    ("name", "needs"),
    [
        ("sma_20", 25),
        ("sma_50", 55),
        ("sma_200", 205),
        ("ema_20", 25),
        ("rsi", 15),
        ("macd", 34),
        ("atr", 15),
        ("volume", 21),
    ],
)
def test_each_threshold_is_exact(name: str, needs: int) -> None:
    assert getattr(calculate_indicators(candles(rising(needs - 1))), name) is None
    assert getattr(calculate_indicators(candles(rising(needs))), name) is not None


def walk(count: int, seed: int = 1) -> list[float]:
    """A random walk of closes at two decimals, as prices are quoted."""
    rng = random.Random(seed)
    closes = [40.0]
    for _ in range(count - 1):
        closes.append(round(closes[-1] + rng.uniform(-1, 1), 2))
    return closes


def test_a_flat_stretch_after_movement_counts_ties_as_above() -> None:
    closes = walk(300)
    closes += [closes[-1]] * 30

    result = calculate_indicators(candles(closes))

    average = result.sma_20
    assert average is not None
    assert average.price_vs == "above"
    assert average.slope == "flat"
    assert average.distance == 0.0
    assert average.days >= 30 - 20 + 1  # Every session whose window is flat


def test_a_suspended_stock_has_no_rsi() -> None:
    result = calculate_indicators(candles(rising(60) + [159.0] * 15))

    assert result.rsi is None
    assert result.note is not None
    assert "rsi is undefined" in result.note


def test_a_zero_close_leaves_the_fraction_null() -> None:
    result = calculate_indicators(candles(walk(30) + [0.0]))

    assert result.atr is not None
    assert result.atr.fraction is None
