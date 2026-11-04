"""
Support and resistance zones around the last close, and the signs that the
nearest support holds or breaks, as pure functions over daily candles (oldest
first). Every threshold is a named constant here, so epic I (Validation) can
tune them against how often zones held.

"""

import math
from datetime import date
from typing import Literal

import numpy as np

from fartt.data import Candle
from fartt.technical.indicators import round_value

Source = Literal[
    "swing_high", "swing_low", "volume", "pivot", "sma_50", "sma_200", "round"
]

# Sessions the zones are found and measured over.
WINDOW = 500
# Fewer sessions than this give no zones: the ATR needs 15, a swing 11.
MIN_SESSIONS = 30
ATR_PERIOD = 14
# A swing high is above this many sessions on each side; a swing low below.
SWING_SESSIONS = 5
# Volume profile bins are this many ATRs wide; the top share of bins by
# volume are candidates.
VOLUME_BIN_ATR = 0.5
VOLUME_TOP = 0.10
# A candidate joins a zone within this many ATRs of the previous one, while
# the zone stays at most MAX_WIDTH_ATR wide; a zone is at least MIN_WIDTH_ATR.
MERGE_ATR = 0.5
MAX_WIDTH_ATR = 1.5
MIN_WIDTH_ATR = 0.5
# A turn closes back out within TURN_SESSIONS of the touch, and at least
# BOUNCE_ATR beyond the zone within BOUNCE_SESSIONS.
TURN_SESSIONS = 3
BOUNCE_ATR = 1.0
BOUNCE_SESSIONS = 10
# A zone is listed with at least this many turns.
MIN_TURNS = 2
# Strength saturates at these turns, volume share and other sources.
TURNS_FULL = 10
VOLUME_FULL = 0.10
SOURCES_FULL = 3
# Zones are listed within this many ATRs of the last close, this many a side.
REACH_ATR = 5
PER_SIDE = 3
# Signs for the nearest support.
DOWN_DAY_HOLDING = 0.8
DOWN_DAY_BREAKING = 1.2
DIVERGENCE_SESSIONS = 20
DIVERGENCE_SWING = 3
RETEST_SESSIONS = 60
RETESTS = 3
BREAK_SESSIONS = 20
STRONG_TURNS = 3
DETAIL_MAX = 80

# Round-number steps: (below this price, step), then ROUND_STEP_ABOVE.
ROUND_STEPS = ((20.0, 1.0), (100.0, 5.0), (500.0, 10.0))
ROUND_STEP_ABOVE = 50.0

ZoneRange = tuple[float, float, frozenset[Source]]


def _swings(
    window: list[Candle], ex_dividend: frozenset[date]
) -> list[tuple[float, Source]]:
    found: list[tuple[float, Source]] = []
    for i in range(SWING_SESSIONS, len(window) - SWING_SESSIONS):
        neighbours = (
            window[i - SWING_SESSIONS : i] + window[i + 1 : i + 1 + SWING_SESSIONS]
        )
        candle = window[i]
        if all(candle.high > other.high for other in neighbours):
            found.append((candle.high, "swing_high"))
        if candle.date not in ex_dividend and all(
            candle.low < other.low for other in neighbours
        ):
            found.append((candle.low, "swing_low"))
    return found


def _volume_nodes(window: list[Candle], atr: float) -> list[tuple[float, Source]]:
    if atr <= 0 or not window:
        return []
    width = VOLUME_BIN_ATR * atr
    bottom = min(c.low for c in window)
    top = max(c.high for c in window)
    count = max(1, math.ceil((top - bottom) / width))
    bins = np.zeros(count, dtype=np.float64)
    for candle in window:
        first = min(count - 1, math.floor((candle.low - bottom) / width))
        last = min(count - 1, max(first, math.ceil((candle.high - bottom) / width) - 1))
        bins[first : last + 1] += candle.volume / (last - first + 1)
    # Strictly above the quantile: bins tied with it aren't a volume node.
    threshold = float(np.quantile(bins, 1 - VOLUME_TOP))
    return [
        (round_value(bottom + (k + 0.5) * width), "volume")
        for k in range(count)
        if bins[k] > threshold
    ]


def _pivots(week: Candle) -> list[tuple[float, Source]]:
    pivot = (week.high + week.low + week.close) / 3
    return [
        (round_value(2 * pivot - week.high), "pivot"),
        (round_value(pivot), "pivot"),
        (round_value(2 * pivot - week.low), "pivot"),
    ]


def candidate_levels(
    window: list[Candle],
    atr: float,
    ex_dividend: frozenset[date],
    pivot_week: Candle | None,
    sma_50: float | None,
    sma_200: float | None,
) -> list[tuple[float, Source]]:
    """
    The prices zones are built from: swing highs and lows (not a low on an
    ex-dividend date), the busiest volume-profile bins, the floor pivots of
    `pivot_week`, and the SMA 50 and 200 when known. Round numbers are never
    candidates; `group_zones` marks them.

    """
    found = _swings(window, ex_dividend) + _volume_nodes(window, atr)
    if pivot_week is not None:
        found += _pivots(pivot_week)
    if sma_50 is not None:
        found.append((round_value(sma_50), "sma_50"))
    if sma_200 is not None:
        found.append((round_value(sma_200), "sma_200"))
    return found


def _round_step(price: float) -> float:
    for below, step in ROUND_STEPS:
        if price < below:
            return step
    return ROUND_STEP_ABOVE


def _zone(prices: list[float], sources: set[Source], atr: float) -> ZoneRange:
    low, high = min(prices), max(prices)
    centre = (low + high) / 2
    half = MIN_WIDTH_ATR * atr / 2
    low, high = min(low, centre - half), max(high, centre + half)
    step = _round_step(centre)
    marks: set[Source] = {"round"} if math.floor(high / step) * step >= low else set()
    return round_value(low), round_value(high), frozenset(sources | marks)


def group_zones(candidates: list[tuple[float, Source]], atr: float) -> list[ZoneRange]:
    """
    Candidates merged into zones, ascending: (low, high, sources). A candidate
    joins the current zone within MERGE_ATR of the previous one, while the
    zone stays at most MAX_WIDTH_ATR wide; a zone is widened to at least
    MIN_WIDTH_ATR around its centre, and marked `round` when a round number
    lies in it.

    """
    zones: list[ZoneRange] = []
    prices: list[float] = []
    sources: set[Source] = set()
    for price, source in sorted(candidates):
        if prices and (
            price - prices[-1] > MERGE_ATR * atr
            or price - prices[0] > MAX_WIDTH_ATR * atr
        ):
            zones.append(_zone(prices, sources, atr))
            prices, sources = [], set()
        prices.append(price)
        sources.add(source)
    if prices:
        zones.append(_zone(prices, sources, atr))
    return zones
