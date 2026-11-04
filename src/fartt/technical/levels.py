"""
Support and resistance zones around the last close, and the signs that the
nearest support holds or breaks, as pure functions over daily candles (oldest
first). Every threshold is a named constant here, so epic I (Validation) can
tune them against how often zones held.

"""

import math
from collections.abc import Callable
from datetime import date
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field
from talib import ATR

from fartt.data import Candle
from fartt.technical.indicators import calculate_indicators, round_value
from fartt.technical.trend import weekly_candles

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
Side = Literal["support", "resistance"]


class Zone(BaseModel):
    low: float = Field(description="The zone's lowest price.")
    high: float = Field(description="The zone's highest price.")
    sources: list[Source] = Field(
        description=(
            "The kinds of level in it: swing highs and lows, a volume node, "
            "last week's pivots, the SMA 50 or 200, a round number."
        )
    )
    touches: int = Field(
        description=(
            "Times price entered the zone (or gapped across it) over the "
            "sessions measured."
        )
    )
    turns: int = Field(
        description=(
            f"Touches after which price closed back out within {TURN_SESSIONS} "
            f"sessions and moved {BOUNCE_ATR:g} ATR away within "
            f"{BOUNCE_SESSIONS}, on the side it came from."
        )
    )
    last_touch: str = Field(
        description="The last session of the latest touch, ISO 8601."
    )
    volume_share: float = Field(
        description="The zone's share of the volume traded over the sessions measured."
    )
    strength: int = Field(
        description=(
            "0-100: turns weigh most, then how recent the last touch is, then "
            "the volume share, then how many kinds of level it holds."
        )
    )
    distance: float = Field(
        description=(
            "(nearest edge - last) / last: -0.03 means 3% below. 0 when the "
            "last close is inside the zone."
        )
    )
    distance_atr: float = Field(description="The same distance in ATRs.")


class Sign(BaseModel):
    sign: str = Field(description="What the evidence is, e.g. daily_trend.")
    points: Literal["holding", "breaking"] = Field(
        description="Whether it points to the support holding or breaking."
    )
    detail: str = Field(description="The evidence, briefly.")


class Levels(BaseModel):
    based_on: str = Field(description="The session the zones are as of, ISO 8601.")
    last: float = Field(description="That session's close.")
    atr: float | None = Field(
        description=f"ATR {ATR_PERIOD} as of that session; null for too short a history."
    )
    support: list[Zone]
    resistance: list[Zone]
    signs: list[Sign]
    note: str | None = Field(
        default=None, description="What's missing or unusual, if anything."
    )


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


def _range(prices: list[float], sources: set[Source], atr: float) -> ZoneRange:
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
            zones.append(_range(prices, sources, atr))
            prices, sources = [], set()
        prices.append(price)
        sources.add(source)
    if prices:
        zones.append(_range(prices, sources, atr))
    return zones


class ZoneMeasure(NamedTuple):
    touches: int
    turns: int
    last_touch: int | None  # Index into the window of the latest touch's last session
    volume_share: float


class MeasuredZone(NamedTuple):
    low: float
    high: float
    sources: frozenset[Source]
    measure: ZoneMeasure
    strength: int


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


def _measured_zones(
    candles: list[Candle],
    atrs: list[float],
    atr: float,
    ex_dividend: frozenset[date],
    pivot_week: Candle | None,
    sma_50: float | None,
    sma_200: float | None,
) -> list[MeasuredZone]:
    candidates = candidate_levels(
        candles, atr, ex_dividend, pivot_week, sma_50, sma_200
    )
    zones: list[MeasuredZone] = []
    for low, high, sources in group_zones(candidates, atr):
        measure = measure_zone(low, high, candles, atrs)
        strength = zone_strength(measure, sources, len(candles))
        zones.append(MeasuredZone(low, high, sources, measure, strength))
    return zones


def _gap(zone: MeasuredZone, last: float, side: Side) -> float:
    """How far the zone's nearest edge lies from the last close; 0 inside it."""
    if zone.low <= last <= zone.high:
        return 0.0
    return last - zone.high if side == "support" else zone.low - last


def pick_zones(
    zones: list[MeasuredZone], last: float, atr: float, side: Side
) -> tuple[list[MeasuredZone], bool]:
    """
    The zones to list on one side of the last close, nearest first, and
    whether they lie beyond reach. Support is a zone whose low is at or below
    the last close, resistance one whose low is above it; either needs
    MIN_TURNS. The PER_SIDE strongest within REACH_ATR are listed, a zone
    holding the last close always among them; with none within reach, the
    nearest beyond is listed alone.

    """
    eligible = [
        z
        for z in zones
        if z.measure.turns >= MIN_TURNS
        and ((z.low <= last) if side == "support" else (z.low > last))
    ]
    within = [z for z in eligible if _gap(z, last, side) <= REACH_ATR * atr]
    if not within:
        nearest = min(eligible, key=lambda z: _gap(z, last, side), default=None)
        return ([nearest], True) if nearest is not None else ([], False)
    ranked = sorted(within, key=lambda z: (_gap(z, last, side) > 0, -z.strength))
    return sorted(ranked[:PER_SIDE], key=lambda z: _gap(z, last, side)), False


def _zone(
    zone: MeasuredZone, candles: list[Candle], last: float, atr: float, side: Side
) -> Zone:
    gap = _gap(zone, last, side)
    signed = -gap if side == "support" else gap
    last_touch = zone.measure.last_touch
    assert last_touch is not None  # a zone with turns was touched
    return Zone(
        low=zone.low,
        high=zone.high,
        sources=sorted(zone.sources),
        touches=zone.measure.touches,
        turns=zone.measure.turns,
        last_touch=candles[last_touch].date.isoformat(),
        volume_share=round_value(zone.measure.volume_share),
        strength=zone.strength,
        distance=round_value(signed / last) if last else 0.0,
        distance_atr=round_value(signed / atr),
    )


def _side_zones(
    side: Side,
    candles: list[Candle],
    window: list[Candle],
    zones: list[MeasuredZone],
    full_zones: Callable[[], list[MeasuredZone]],
    last: float,
    atr: float,
    notes: list[str],
) -> list[Zone]:
    above = side == "resistance"
    earlier = candles[:-1]
    if above and last > max(round_value(c.high) for c in earlier):
        notes.append("There's no resistance from history above the price.")
        return []
    if not above and last < min(round_value(c.low) for c in earlier):
        notes.append("There's no support from history below the price.")
        return []
    measured_over = window
    chosen, beyond = pick_zones(zones, last, atr, side)
    if not chosen and len(candles) > len(window):
        measured_over = candles
        chosen, beyond = pick_zones(full_zones(), last, atr, side)
    if not chosen:
        where = "above" if above else "below"
        notes.append(f"No {side}: price never turned {where} the last close.")
        return []
    if beyond:
        notes.append(
            f"The nearest {side} zone lies further than {REACH_ATR} ATR from "
            "the last close."
        )
    listed = [_zone(z, measured_over, last, atr, side) for z in chosen]
    if not above and listed[0].distance == 0:
        notes.append(
            f"The last close is inside the support zone {listed[0].low:g}-"
            f"{listed[0].high:g}."
        )
    return listed


def calculate_levels(
    candles: list[Candle], exchange: str, ex_dividend: frozenset[date] = frozenset()
) -> Levels:
    """
    Support and resistance zones around the last close of `candles` (oldest
    first, at least one; the whole cached history) on `exchange`: where price
    turned before, found and measured over the last WINDOW sessions, with
    older history searched for a side the window has none on. `ex_dividend`
    holds the ex-dividend dates: a swing low on one isn't a level.

    """
    count = len(candles)
    based_on = candles[-1].date.isoformat()
    last = round_value(candles[-1].close)
    if count < MIN_SESSIONS:
        return Levels(
            based_on=based_on,
            last=last,
            atr=None,
            support=[],
            resistance=[],
            signs=[],
            note=f"Zones need {MIN_SESSIONS} sessions, {count} are cached.",
        )

    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    closes = np.array([c.close for c in candles], dtype=np.float64)
    atrs = [float(v) for v in ATR(highs, lows, closes, timeperiod=ATR_PERIOD)]
    atr = round_value(atrs[-1])
    if atr == 0:
        return Levels(
            based_on=based_on,
            last=last,
            atr=atr,
            support=[],
            resistance=[],
            signs=[],
            note="Price hasn't moved: the ATR is 0, so there are no zones.",
        )

    indicators = calculate_indicators(candles, ex_dividend)
    weeks, _ = weekly_candles(candles, exchange)

    def measured(over: list[Candle], over_atrs: list[float]) -> list[MeasuredZone]:
        return _measured_zones(
            over,
            over_atrs,
            atr,
            ex_dividend,
            weeks[-1] if weeks else None,
            indicators.sma_50.value if indicators.sma_50 else None,
            indicators.sma_200.value if indicators.sma_200 else None,
        )

    window = candles[-WINDOW:]
    zones = measured(window, atrs[-WINDOW:])
    full: list[list[MeasuredZone]] = []

    def full_zones() -> list[MeasuredZone]:
        if not full:
            full.append(measured(candles, atrs))
        return full[0]

    notes: list[str] = []
    support = _side_zones(
        "support", candles, window, zones, full_zones, last, atr, notes
    )
    resistance = _side_zones(
        "resistance", candles, window, zones, full_zones, last, atr, notes
    )
    return Levels(
        based_on=based_on,
        last=last,
        atr=atr,
        support=support,
        resistance=resistance,
        signs=[],
        note=" ".join(notes) or None,
    )
