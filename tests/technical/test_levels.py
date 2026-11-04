from datetime import date, timedelta

import exchange_calendars as xcals
import numpy as np
import pytest
import talib

from fartt.data import Candle
from fartt.technical import Indicators, Trend, calculate_indicators, calculate_trend
from fartt.technical.levels import (
    DETAIL_MAX,
    Levels,
    MeasuredZone,
    Sign,
    Source,
    Zone,
    ZoneMeasure,
    calculate_levels,
    candidate_levels,
    group_zones,
    measure_zone,
    pick_zones,
    support_signs,
    zone_strength,
)

START = date(2024, 1, 1)


def flat(count: int, high: float = 101.0, low: float = 99.0) -> list[Candle]:
    """`count` daily candles with the same range, one day apart."""
    return [
        Candle(START + timedelta(days=i), 100.0, high, low, 100.0, 1000.0)
        for i in range(count)
    ]


def replace(
    candles: list[Candle],
    index: int,
    high: float | None = None,
    low: float | None = None,
) -> None:
    candle = candles[index]
    candles[index] = candle._replace(
        high=candle.high if high is None else high,
        low=candle.low if low is None else low,
    )


def of_kind(candidates: list[tuple[float, Source]], kind: Source) -> list[float]:
    return sorted(price for price, source in candidates if source == kind)


def candidates(
    window: list[Candle],
    atr: float = 2.0,
    ex_dividend: frozenset[date] = frozenset(),
    pivot_week: Candle | None = None,
    sma_50: float | None = None,
    sma_200: float | None = None,
) -> list[tuple[float, Source]]:
    return candidate_levels(window, atr, ex_dividend, pivot_week, sma_50, sma_200)


def test_a_swing_high_and_low_are_candidates() -> None:
    window = flat(15)
    replace(window, 5, high=105.0)
    replace(window, 8, low=95.0)

    found = candidates(window)

    assert of_kind(found, "swing_high") == [105.0]
    assert of_kind(found, "swing_low") == [95.0]


def test_a_high_equal_to_a_neighbour_is_not_a_swing() -> None:
    window = flat(15)
    replace(window, 6, high=105.0)
    replace(window, 7, high=105.0)

    assert of_kind(candidates(window), "swing_high") == []


def test_a_swing_needs_five_sessions_on_each_side() -> None:
    window = flat(15)
    replace(window, 4, high=105.0)  # only four sessions before it
    replace(window, 10, low=95.0)  # only four after it

    found = candidates(window)

    assert of_kind(found, "swing_high") == []
    assert of_kind(found, "swing_low") == []


def test_a_swing_low_on_an_ex_dividend_date_is_not_a_candidate() -> None:
    window = flat(15)
    replace(window, 7, low=95.0)

    found = candidates(window, ex_dividend=frozenset({window[7].date}))

    assert of_kind(found, "swing_low") == []


def test_the_busiest_volume_bins_are_candidates() -> None:
    # Bins of 2 (half the ATR of 4) from 100: the busy band 150.5-151.5 lies
    # in the bin 150-152, whose centre is 151.
    quiet = [
        Candle(START + timedelta(days=i), 150.0, 200.0, 100.0, 150.0, 10.0)
        for i in range(20)
    ]
    busy = [
        Candle(START + timedelta(days=20 + i), 151.0, 151.5, 150.5, 151.0, 10_000.0)
        for i in range(5)
    ]

    found = candidates(quiet + busy, atr=4.0)

    assert of_kind(found, "volume") == [151.0]


def test_a_session_without_a_range_puts_its_volume_in_its_bin() -> None:
    quiet = [
        Candle(START + timedelta(days=i), 150.0, 200.0, 100.0, 150.0, 10.0)
        for i in range(20)
    ]
    halted = Candle(START + timedelta(days=20), 171.0, 171.0, 171.0, 171.0, 5000.0)

    found = candidates([*quiet, halted], atr=4.0)

    assert of_kind(found, "volume") == [171.0]


def test_pivots_come_from_the_given_week() -> None:
    week = Candle(START, 95.0, 110.0, 90.0, 100.0, 0.0)

    with_week = candidates(flat(15), pivot_week=week)
    without = candidates(flat(15))

    assert of_kind(with_week, "pivot") == [90.0, 100.0, 110.0]
    assert of_kind(without, "pivot") == []


def test_the_averages_are_candidates_when_known() -> None:
    known = candidates(flat(15), sma_50=98.0, sma_200=96.0)
    unknown = candidates(flat(15), sma_50=None, sma_200=None)

    assert of_kind(known, "sma_50") == [98.0]
    assert of_kind(known, "sma_200") == [96.0]
    assert of_kind(unknown, "sma_50") == []
    assert of_kind(unknown, "sma_200") == []


def test_round_numbers_are_never_candidates() -> None:
    assert of_kind(candidates(flat(15)), "round") == []


def test_candidates_within_half_an_atr_merge() -> None:
    merged = group_zones(
        [(100.0, "swing_low"), (104.0, "volume"), (108.0, "pivot")], atr=10.0
    )
    apart = group_zones([(100.0, "swing_low"), (106.0, "swing_low")], atr=10.0)

    assert [(low, high) for low, high, _ in merged] == [(100.0, 108.0)]
    assert merged[0][2] >= {"swing_low", "volume", "pivot"}
    assert len(apart) == 2


def test_candidates_are_merged_in_price_order() -> None:
    zones = group_zones(
        [(108.0, "pivot"), (100.0, "swing_low"), (104.0, "volume")], 10.0
    )

    assert [(low, high) for low, high, _ in zones] == [(100.0, 108.0)]


def test_a_zone_stops_growing_at_one_and_a_half_atr() -> None:
    prices = [100.0, 104.0, 108.0, 112.0, 116.0]

    zones = group_zones([(p, "swing_high") for p in prices], atr=10.0)

    assert [(low, high) for low, high, _ in zones] == [
        (100.0, 112.0),
        (113.5, 118.5),
    ]


def test_a_narrow_zone_is_widened_to_half_an_atr() -> None:
    zones = group_zones([(101.0, "swing_low")], atr=10.0)

    assert [(low, high) for low, high, _ in zones] == [(98.5, 103.5)]


def test_a_round_number_marks_a_zone_but_never_makes_one() -> None:
    zones = group_zones([(298.0, "swing_low"), (301.0, "swing_low")], atr=10.0)
    off_round = group_zones([(312.0, "swing_low")], atr=4.0)

    assert zones[0][2] == {"swing_low", "round"}
    assert off_round[0][2] == {"swing_low"}
    assert len(off_round) == 1


@pytest.mark.parametrize(
    ("price", "has_round"),
    [
        (13.0, True),  # a multiple of 1 below 20
        (52.4, False),  # 50 is beyond the zone, 52 isn't a multiple of 5
        (55.0, True),  # a multiple of 5 below 100
        (243.0, False),  # 240 is beyond the zone at ATR 2
        (250.2, True),  # a multiple of 10 below 500
        (549.8, True),  # 550, a multiple of 50 above 500
        (540.0, False),
    ],
)
def test_the_round_step_grows_with_the_price(price: float, has_round: bool) -> None:
    zones = group_zones([(price, "swing_low")], atr=2.0)

    assert ("round" in zones[0][2]) is has_round


# Touches and turns: a zone of 95-100, an ATR of 2 unless stated.
LOW, HIGH = 95.0, 100.0
ABOVE = (109.0, 111.0, 110.0)  # a session well above the zone
BELOW = (84.0, 86.0, 85.0)  # well below it


def bars(*specs: tuple[float, float, float], volume: float = 100.0) -> list[Candle]:
    """Candles from (low, high, close) one day apart."""
    return [
        Candle(START + timedelta(days=i), close, high, low, close, volume)
        for i, (low, high, close) in enumerate(specs)
    ]


def measure(window: list[Candle], atrs: list[float] | None = None) -> ZoneMeasure:
    return measure_zone(LOW, HIGH, window, atrs or [2.0] * len(window))


def test_consecutive_sessions_in_the_zone_are_one_touch() -> None:
    window = bars(ABOVE, (97.0, 104.0, 99.0), (96.0, 100.5, 98.0), (98.0, 103.0, 102.5))

    assert measure(window).touches == 1


def test_a_dip_that_bounces_is_a_turn() -> None:
    window = bars(ABOVE, (97.0, 108.0, 101.0), (101.5, 104.0, 103.0), ABOVE)

    result = measure(window)

    assert (result.touches, result.turns, result.last_touch) == (1, 1, 1)


def test_a_touch_without_a_bounce_is_not_a_turn() -> None:
    window = bars(ABOVE, (97.0, 108.0, 101.0), *[(100.5, 101.8, 101.0)] * 12)

    result = measure(window)

    assert (result.touches, result.turns) == (1, 0)


def test_a_bounce_after_ten_sessions_is_not_a_turn() -> None:
    window = bars(ABOVE, (97.0, 108.0, 101.0), *[(100.5, 101.8, 101.0)] * 9, ABOVE)

    assert measure(window).turns == 0


def test_a_close_through_before_the_bounce_is_not_a_turn() -> None:
    window = bars(
        ABOVE, (97.0, 108.0, 101.0), (93.5, 94.5, 94.0), (102.5, 103.5, 103.0)
    )

    assert measure(window).turns == 0


def test_a_gap_across_is_a_touch_but_not_a_turn() -> None:
    window = bars(ABOVE, BELOW, BELOW)

    result = measure(window)

    assert (result.touches, result.turns, result.last_touch) == (1, 0, 1)


def test_a_touch_from_inside_is_not_a_turn() -> None:
    # The first session lies in the zone: there's no close before it to say
    # which side price came from.
    window = bars((96.0, 99.0, 98.0), (101.5, 104.0, 103.0), ABOVE)

    result = measure(window)

    assert (result.touches, result.turns) == (1, 0)


def test_the_bounce_uses_the_atr_of_the_touch() -> None:
    window = bars(ABOVE, (97.0, 108.0, 101.0), (101.0, 103.0, 102.5))

    result = measure(window, [2.0, 2.0, 10.0])

    assert result.turns == 1


def test_resistance_turns_mirror_support() -> None:
    window = bars(BELOW, (88.0, 98.0, 94.0), (92.0, 94.0, 93.0), BELOW)

    result = measure(window)

    assert (result.touches, result.turns) == (1, 1)


def test_no_turn_before_the_atr_is_known() -> None:
    window = bars(ABOVE, (97.0, 108.0, 101.0), (101.5, 104.0, 103.0), ABOVE)

    assert measure(window, [float("nan")] * 4).turns == 0


def test_a_window_without_touches_has_no_last_touch() -> None:
    result = measure(bars(ABOVE, ABOVE))

    assert (result.touches, result.turns, result.last_touch) == (0, 0, None)


def test_volume_share_weights_by_range_overlap() -> None:
    half_in = bars(ABOVE, ABOVE, (97.5, 102.5, 100.0), ABOVE)
    halted = bars(ABOVE, ABOVE, (97.0, 97.0, 97.0), ABOVE)

    assert measure(half_in).volume_share == pytest.approx(0.125)
    assert measure(halted).volume_share == pytest.approx(0.25)


def test_strength_follows_the_formula() -> None:
    full = zone_strength(
        ZoneMeasure(touches=12, turns=10, last_touch=9, volume_share=0.10),
        frozenset({"swing_low", "volume", "pivot", "round"}),
        window_length=10,
    )
    half = zone_strength(
        ZoneMeasure(touches=6, turns=5, last_touch=4, volume_share=0.05),
        frozenset({"swing_low", "round"}),
        window_length=10,
    )

    assert full == 100
    assert half == round(100 * (0.55 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5 + 0.10 / 3))


def test_a_zone_never_touched_has_no_recency() -> None:
    strength = zone_strength(
        ZoneMeasure(touches=0, turns=0, last_touch=None, volume_share=0.0),
        frozenset({"swing_low"}),
        window_length=10,
    )

    assert strength == round(100 * 0.10 / 3)


# Choosing the zones to list. Series run on XAMS sessions; each session's
# range is its close +/- 0.5, so with steps of 2 the ATR is 2.5.
FRIDAY = date(2026, 10, 30)
STEP = 2.0


def sessions(count: int, last: date = FRIDAY) -> list[date]:
    window = xcals.get_calendar("XAMS").sessions_window(last.isoformat(), -count)
    return [session.date() for session in window]


def path(closes: list[float]) -> list[Candle]:
    return [
        Candle(day, close, close + 0.5, close - 0.5, close, 1000.0)
        for day, close in zip(sessions(len(closes)), closes, strict=True)
    ]


def walk(start: float, *targets: float) -> list[float]:
    """Closes from `start` to each target in turn, STEP at a time."""
    closes = [start]
    for target in targets:
        while closes[-1] != target:
            move = min(STEP, abs(target - closes[-1]))
            closes.append(closes[-1] + (move if target > closes[-1] else -move))
    return closes


def zigzag(
    low: float, high: float, cycles: int, start: float | None = None
) -> list[float]:
    """From `start` (the midpoint by default) down to `low` and up to `high`, `cycles` times."""
    begin = (low + high) / 2 if start is None else start
    return walk(begin, *[low, high] * cycles)


def levels(closes: list[float], ex_dividend: frozenset[date] = frozenset()) -> Levels:
    return calculate_levels(path(closes), "XAMS", ex_dividend)


def test_zones_are_split_by_the_last_close_nearest_first() -> None:
    result = levels(walk(*zigzag(90, 110, 15), 100))

    assert result.support and result.resistance
    assert all(zone.low <= result.last for zone in result.support)
    assert all(zone.low > result.last for zone in result.resistance)
    for side in (result.support, result.resistance):
        assert len(side) <= 3
        assert [abs(z.distance) for z in side] == sorted(abs(z.distance) for z in side)
        assert all(zone.turns >= 2 for zone in side)
    assert result.support[0].low == pytest.approx(88.875)
    assert result.resistance[0].high == pytest.approx(111.125)


def zone(low: float, high: float, strength: int, turns: int = 4) -> MeasuredZone:
    return MeasuredZone(
        low,
        high,
        frozenset({"swing_low"}),
        ZoneMeasure(turns + 1, turns, 0, 0.0),
        strength,
    )


def test_the_strongest_within_five_atr_are_listed() -> None:
    zones = [zone(80, 81, 90), zone(84, 85, 20), zone(88, 89, 70), zone(92, 93, 50)]

    chosen, beyond = pick_zones(zones, last=100.0, atr=4.0, side="support")

    assert [z.low for z in chosen] == [92, 88, 80]
    assert beyond is False


def test_zones_beyond_reach_are_left_out() -> None:
    zones = [zone(70, 71, 99), zone(92, 93, 10)]

    chosen, _ = pick_zones(zones, last=100.0, atr=4.0, side="support")

    assert [z.low for z in chosen] == [92]


def test_a_zone_needs_two_turns() -> None:
    zones = [zone(92, 93, 90, turns=1), zone(88, 89, 10, turns=2)]

    chosen, _ = pick_zones(zones, last=100.0, atr=4.0, side="support")

    assert [z.low for z in chosen] == [88]


def test_the_zone_holding_the_last_close_is_always_listed() -> None:
    zones = [zone(99, 101, 5), zone(96, 97, 50), zone(92, 93, 60), zone(88, 89, 70)]

    chosen, _ = pick_zones(zones, last=100.0, atr=4.0, side="support")

    assert [z.low for z in chosen] == [99, 92, 88]


def test_resistance_is_picked_above_the_last_close() -> None:
    zones = [zone(99, 101, 90), zone(104, 105, 50), zone(140, 141, 70)]

    chosen, beyond = pick_zones(zones, last=100.0, atr=4.0, side="resistance")

    assert [z.low for z in chosen] == [104]
    assert beyond is False


def test_a_side_with_nothing_within_reach_lists_the_nearest_beyond() -> None:
    # A rally from the 90-110 range to 128: the top zone (~111) is 6.8 ATR away.
    result = levels(walk(*zigzag(90, 110, 15), 128))

    assert len(result.support) == 1
    assert result.support[0].high == pytest.approx(111.125)
    assert result.support[0].distance_atr < -5
    assert result.note is not None and "further than 5 ATR" in result.note


def test_a_close_inside_a_zone_is_support_at_distance_zero() -> None:
    result = levels(walk(*zigzag(90, 110, 15), 90))

    first = result.support[0]
    assert first.low <= result.last <= first.high
    assert (first.distance, first.distance_atr) == (0, 0)
    # Not -0.0, which JSON would carry to the agent as "-0.0".
    assert '"distance":0.0' in first.model_dump_json()
    assert '"distance_atr":0.0' in first.model_dump_json()
    assert result.note is not None and "inside" in result.note


def test_a_new_high_has_no_resistance() -> None:
    result = levels(walk(*zigzag(90, 110, 15), 120))

    assert result.resistance == []
    assert result.support
    assert result.note is not None
    assert "no resistance from history above the price" in result.note


def test_a_new_low_has_no_support() -> None:
    result = levels(walk(*zigzag(90, 110, 15), 80))

    assert result.support == []
    assert result.resistance
    assert result.note is not None
    assert "no support from history below the price" in result.note


def test_older_history_is_searched_when_the_window_has_no_zone() -> None:
    old = zigzag(90, 110, 15, start=100)  # about 300 sessions turning at 90 and 110
    recent = walk(old[-1], 70) + zigzag(60, 80, 25, start=70)[1:]
    closes = walk(*(old + recent[1:]), 84)
    assert len(closes) > 700

    result = levels(closes)

    assert result.resistance
    assert result.resistance[0].low > result.last
    window_start = path(closes)[-500].date.isoformat()
    assert result.resistance[0].last_touch < window_start


def test_a_side_price_never_turned_on_is_empty_with_a_note() -> None:
    result = levels(walk(150, 80) + zigzag(70, 80, 10, start=80)[1:] + walk(80, 95)[1:])

    assert result.resistance == []
    assert result.note is not None and "never turned above" in result.note


def test_too_short_a_history_has_no_zones() -> None:
    result = levels(zigzag(90, 110, 1)[:29])

    assert (result.support, result.resistance, result.signs) == ([], [], [])
    assert result.atr is None
    assert result.note is not None and "30 sessions" in result.note


def test_a_short_history_uses_all_of_it() -> None:
    closes = walk(*zigzag(90, 110, 6), 100)

    result = levels(closes)

    bottom = next(z for z in result.support if z.low == pytest.approx(88.875))
    assert bottom.touches == 6
    # The first dip comes before the ATR is known, so it can't turn.
    assert bottom.turns == 5


def test_a_flat_price_has_no_zones() -> None:
    flat_price = [
        Candle(day, 100.0, 100.0, 100.0, 100.0, 1000.0) for day in sessions(60)
    ]

    result = calculate_levels(flat_price, "XAMS")

    assert (result.support, result.resistance) == ([], [])
    assert result.atr == 0
    assert result.note is not None and "hasn't moved" in result.note


def test_distance_is_measured_to_the_nearest_edge() -> None:
    result = levels(walk(*zigzag(90, 110, 15), 100))
    last, atr = result.last, result.atr
    assert atr is not None

    for z in result.support:
        assert z.distance == round((z.high - last) / last, 4)
        assert z.distance_atr == round((z.high - last) / atr, 4)
    for z in result.resistance:
        assert z.distance == round((z.low - last) / last, 4)
        assert z.distance_atr == round((z.low - last) / atr, 4)


def test_a_zone_reports_its_evidence() -> None:
    closes = walk(*zigzag(90, 110, 15), 100)
    result = levels(closes)
    bottom = result.support[0]

    assert result.based_on == FRIDAY.isoformat()
    assert result.atr == pytest.approx(2.5)
    assert "swing_low" in bottom.sources
    assert bottom.touches >= bottom.turns >= 14
    assert 0 < bottom.volume_share < 1
    assert 0 <= bottom.strength <= 100
    assert bottom.last_touch in {c.date.isoformat() for c in path(closes)}


# Signs for the nearest support. A neutral case decides no sign; each test
# changes only the evidence under test.
BASE = path(walk(*zigzag(90, 110, 15), 100))
SUPPORT = Zone(
    low=95.0,
    high=100.0,
    sources=["swing_low"],
    touches=4,
    turns=2,
    last_touch="2026-10-01",
    volume_share=0.05,
    strength=50,
    distance=0.0,
    distance_atr=0.0,
)


def neutral_trend(daily: str = "sideways", weekly: str = "sideways") -> Trend:
    trend = calculate_trend(BASE, "XAMS")
    assert trend.weekly is not None
    return trend.model_copy(
        update={
            "daily": trend.daily.model_copy(update={"direction": daily}),
            "weekly": trend.weekly.model_copy(update={"direction": weekly}),
        }
    )


def neutral_indicators(
    price_vs: str | None = None,
    down_day_ratio: float = 1.0,
    rsi: str = "flat",
    histogram: str = "flat",
) -> Indicators:
    indicators = calculate_indicators(BASE)
    assert indicators.sma_200 and indicators.volume and indicators.rsi
    assert indicators.macd
    return indicators.model_copy(
        update={
            "sma_200": (
                indicators.sma_200.model_copy(
                    update={"price_vs": price_vs, "distance": 0.031}
                )
                if price_vs
                else None
            ),
            "volume": indicators.volume.model_copy(
                update={"down_day_ratio": down_day_ratio}
            ),
            "rsi": indicators.rsi.model_copy(update={"direction": rsi}),
            "macd": indicators.macd.model_copy(
                update={"histogram_direction": histogram}
            ),
        }
    )


def signs(
    candles: list[Candle] | None = None,
    zone: Zone = SUPPORT,
    trend: Trend | None = None,
    indicators: Indicators | None = None,
    ex_dividend: frozenset[date] = frozenset(),
) -> dict[str, Sign]:
    found = support_signs(
        zone,
        candles or BASE,
        ex_dividend,
        trend or neutral_trend(),
        indicators or neutral_indicators(),
    )
    return {sign.sign: sign for sign in found}


def points(found: dict[str, Sign], name: str) -> str | None:
    return found[name].points if name in found else None


def test_a_neutral_case_has_no_signs() -> None:
    assert signs(zone=SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})) == {}


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("up", "holding"), ("down", "breaking"), ("sideways", None)],
)
def test_trend_signs_follow_get_trend(direction: str, expected: str | None) -> None:
    far = SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})
    found = signs(zone=far, trend=neutral_trend(direction, direction))

    assert points(found, "daily_trend") == expected
    assert points(found, "weekly_trend") == expected
    if expected:
        assert found["daily_trend"].detail == f"daily trend {direction}"


@pytest.mark.parametrize(
    ("price_vs", "expected"),
    [("above", "holding"), ("below", "breaking"), (None, None)],
)
def test_the_sma_200_sign(price_vs: str | None, expected: str | None) -> None:
    far = SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})

    found = signs(zone=far, indicators=neutral_indicators(price_vs=price_vs))

    assert points(found, "sma_200") == expected


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.7, "holding"), (1.3, "breaking"), (1.0, None), (0.8, None), (1.2, None)],
)
def test_volume_on_drop_uses_the_thresholds(ratio: float, expected: str | None) -> None:
    far = SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})

    found = signs(zone=far, indicators=neutral_indicators(down_day_ratio=ratio))

    assert points(found, "volume_on_drop") == expected
    if expected:
        assert f"{ratio:.2f}x" in found["volume_on_drop"].detail


def divergence(second_drop: list[float]) -> list[Candle]:
    """A flat base, a sharp drop to 80, a bounce, then `second_drop` and a rise."""
    closes = [100.0, 101.0] * 30 + [96, 92, 88, 84, 80, 82, 84, 86, 88]
    return path([*closes, *second_drop, 81, 83, 85])


def test_a_bullish_rsi_divergence_points_to_holding() -> None:
    candles = divergence([87, 86, 85, 84, 83, 82, 81, 80, 79])
    rsi = talib.RSI(np.array([c.close for c in candles]), timeperiod=14)
    assert rsi[-4] > rsi[-17]  # the later, lower low has the higher RSI
    far = SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})

    found = signs(candles, zone=far)

    assert points(found, "rsi_divergence") == "holding"


def test_a_lower_low_on_a_lower_rsi_is_no_divergence() -> None:
    candles = divergence([84, 80, 76, 72, 68, 64, 60, 56, 52])
    far = SUPPORT.model_copy(update={"low": 30.0, "high": 31.0})

    assert "rsi_divergence" not in signs(candles, zone=far)


@pytest.mark.parametrize(
    ("rsi", "histogram", "expected"),
    [
        ("rising", "rising", "holding"),
        ("falling", "falling", "breaking"),
        ("rising", "falling", None),
        ("flat", "rising", None),
    ],
)
def test_momentum_needs_both_rsi_and_macd(
    rsi: str, histogram: str, expected: str | None
) -> None:
    far = SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})

    found = signs(zone=far, indicators=neutral_indicators(rsi=rsi, histogram=histogram))

    assert points(found, "momentum") == expected


def dips(count: int, bottom: float = 96.0) -> list[Candle]:
    closes = [110.0] * 30
    for _ in range(count):
        closes = walk(*closes, bottom, 110)
    return path(closes)


def test_three_recent_retests_point_to_breaking() -> None:
    assert points(signs(dips(3)), "retests") == "breaking"
    assert "retests" not in signs(dips(2))


def test_a_recent_close_below_points_to_breaking() -> None:
    found = signs(dips(1, bottom=94.0))

    assert points(found, "recent_break") == "breaking"
    assert "recent_break" not in signs(dips(1, bottom=96.0))


def test_an_ex_dividend_close_below_does_not_count() -> None:
    candles = dips(1, bottom=94.0)
    below = frozenset(c.date for c in candles if c.close < 95)

    assert "recent_break" not in signs(candles, ex_dividend=below)


def test_three_turns_point_to_holding() -> None:
    far = SUPPORT.model_copy(update={"low": 50.0, "high": 51.0})

    assert points(signs(zone=far.model_copy(update={"turns": 3})), "turns") == "holding"
    assert "turns" not in signs(zone=far)


def test_no_support_means_no_signs() -> None:
    result = levels(walk(*zigzag(90, 110, 15), 80))

    assert result.support == []
    assert result.signs == []


def test_signs_are_listed_for_the_nearest_support() -> None:
    closes = walk(*zigzag(90, 110, 15), 140, 126)
    candles = path(closes)

    result = calculate_levels(candles, "XAMS")

    assert result.support
    assert calculate_trend(candles, "XAMS").daily.direction == "up"
    assert points({s.sign: s for s in result.signs}, "daily_trend") == "holding"


def test_every_detail_is_short_and_nothing_is_neutral() -> None:
    mixed = neutral_indicators(
        price_vs="below", down_day_ratio=0.5, rsi="rising", histogram="rising"
    )
    candles = dips(3, bottom=94.0)
    zone = SUPPORT.model_copy(update={"turns": 12})

    found = signs(
        candles, zone=zone, trend=neutral_trend("up", "down"), indicators=mixed
    )

    assert len(found) >= 7
    for sign in found.values():
        assert sign.points in ("holding", "breaking")
        assert len(sign.detail) < DETAIL_MAX
