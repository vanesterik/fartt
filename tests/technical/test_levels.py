from datetime import date, timedelta

import pytest

from fartt.data import Candle
from fartt.technical.levels import (
    Source,
    ZoneMeasure,
    candidate_levels,
    group_zones,
    measure_zone,
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
