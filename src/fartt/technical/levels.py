"""
Support and resistance zones around the last close, and the signs that the
nearest support holds or breaks, as pure functions over daily candles (oldest
first). Every threshold is a named constant here, so epic I (Validation) can
tune them against how often zones held.

"""

import math
from datetime import date
from typing import Literal, NamedTuple

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
# Strength weights: turns dominate, then recency, then volume, then sources.
TURNS_WEIGHT = 0.55
RECENCY_WEIGHT = 0.20
VOLUME_WEIGHT = 0.15
SOURCES_WEIGHT = 0.10
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


class ZoneMeasure(NamedTuple):
    touches: int
    turns: int
    last_touch: int | None  # Index into the window of the latest touch's last session
    volume_share: float


class _Touch(NamedTuple):
    start: int  # First session of the episode
    end: int  # Last session of the episode
    gap: bool  # Closes on opposite sides with no session inside: never a turn


def _enters(candle: Candle, low: float, high: float) -> bool:
    return round_value(candle.high) >= low and round_value(candle.low) <= high


def _side(close: float, low: float, high: float) -> int:
    """1 above the zone, -1 below it, 0 inside."""
    close = round_value(close)
    return 1 if close > high else -1 if close < low else 0


def _touches(low: float, high: float, window: list[Candle]) -> list[_Touch]:
    """The episodes whose range enters the zone, and the gaps across it, in order."""
    touches: list[_Touch] = []
    start: int | None = None
    for i, candle in enumerate(window):
        if _enters(candle, low, high):
            if start is None:
                start = i
            continue
        if start is not None:
            touches.append(_Touch(start, i - 1, gap=False))
            start = None
        elif i > 0:
            # Neither this session nor the one before entered the zone.
            before = _side(window[i - 1].close, low, high)
            if before and _side(candle.close, low, high) == -before:
                touches.append(_Touch(i, i, gap=True))
    if start is not None:
        touches.append(_Touch(start, len(window) - 1, gap=False))
    return touches


def _is_turn(
    touch: _Touch, low: float, high: float, window: list[Candle], atrs: list[float]
) -> bool:
    if touch.gap or touch.start == 0:
        return False
    side = _side(window[touch.start - 1].close, low, high)
    atr = atrs[touch.end]
    if side == 0 or math.isnan(atr):
        return False
    after = [c.close for c in window[touch.end : touch.end + BOUNCE_SESSIONS]]
    if not any(_side(close, low, high) == side for close in after[:TURN_SESSIONS]):
        return False
    edge = high if side > 0 else low
    for close in after:
        if _side(close, low, high) == -side:
            return False
        if side * (round_value(close) - edge) >= BOUNCE_ATR * atr:
            return True
    return False


def _overlap(candle: Candle, low: float, high: float) -> float:
    """The share of the session's range that lies in the zone."""
    if candle.high == candle.low:
        return 1.0 if low <= round_value(candle.high) <= high else 0.0
    inside = min(candle.high, high) - max(candle.low, low)
    return max(0.0, inside) / (candle.high - candle.low)


def measure_zone(
    low: float, high: float, window: list[Candle], atrs: list[float]
) -> ZoneMeasure:
    """
    How the zone `low`-`high` was traded over `window`: its touches, the
    touches that turned, the latest touch, and its share of the volume.
    `atrs` holds each session's ATR, aligned with `window`.

    """
    touches = _touches(low, high, window)
    total = sum(c.volume for c in window)
    traded = sum(c.volume * _overlap(c, low, high) for c in window)
    return ZoneMeasure(
        touches=len(touches),
        turns=sum(_is_turn(t, low, high, window, atrs) for t in touches),
        last_touch=touches[-1].end if touches else None,
        volume_share=traded / total if total > 0 else 0.0,
    )


def zone_strength(
    measure: ZoneMeasure, sources: frozenset[Source], window_length: int
) -> int:
    """
    0-100: turns weigh most, then how recent the last touch is, then the
    zone's share of the volume, then how many kinds of level (round numbers
    aside) it holds.

    """
    recency = (
        1 - (window_length - 1 - measure.last_touch) / window_length
        if measure.last_touch is not None
        else 0.0
    )
    others = len(sources - {"round"})
    return round(
        100
        * (
            TURNS_WEIGHT * min(measure.turns / TURNS_FULL, 1)
            + RECENCY_WEIGHT * recency
            + VOLUME_WEIGHT * min(measure.volume_share / VOLUME_FULL, 1)
            + SOURCES_WEIGHT * min(others / SOURCES_FULL, 1)
        )
    )
