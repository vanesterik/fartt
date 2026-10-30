import math

import pytest

from fartt.features.calculate_trade_returns import calculate_trade_returns


def test_calculate_trade_returns_opens_and_closes_a_trade() -> None:
    candle_returns = [float("nan"), 0.01, 0.02, -0.01]

    returns, profits = calculate_trade_returns(candle_returns, initial_capital=500)

    gross_return = (1 + 0.02) * (1 - 0.01)
    cost_factor = (1 - 0.0005) ** 2
    expected_return = gross_return * cost_factor - 1
    assert returns == pytest.approx([expected_return])
    assert profits == pytest.approx([500 * expected_return])


def test_calculate_trade_returns_no_trade_within_threshold() -> None:
    candle_returns = [0.0001, -0.0002, 0.0003]

    returns, profits = calculate_trade_returns(candle_returns)

    assert returns == []
    assert profits == []


def test_calculate_trade_returns_default_threshold_derives_from_costs() -> None:
    cost_pct = 0.01
    slippage_pct = 0.01

    below_threshold_returns, _ = calculate_trade_returns(
        [0.035, -0.035], cost_pct=cost_pct, slippage_pct=slippage_pct
    )
    above_threshold_returns, _ = calculate_trade_returns(
        [0.05, -0.05], cost_pct=cost_pct, slippage_pct=slippage_pct
    )

    assert below_threshold_returns == []
    assert len(above_threshold_returns) == 1


def test_calculate_trade_returns_force_closes_open_position_at_end() -> None:
    candle_returns = [0.01, 0.02]

    returns, _ = calculate_trade_returns(candle_returns)

    gross_return = 1 + 0.02
    cost_factor = (1 - 0.0005) ** 2
    expected_return = gross_return * cost_factor - 1
    assert returns == pytest.approx([expected_return])


def test_calculate_trade_returns_slippage_reduces_return() -> None:
    candle_returns = [0.02, -0.02]

    returns_no_slippage, _ = calculate_trade_returns(
        candle_returns, cost_pct=0.0025, slippage_pct=0.0, threshold=0.005
    )
    returns_with_slippage, _ = calculate_trade_returns(
        candle_returns, cost_pct=0.0025, slippage_pct=0.001, threshold=0.005
    )

    assert returns_with_slippage[0] < returns_no_slippage[0]


def test_calculate_trade_returns_profits_compound_but_returns_do_not() -> None:
    candle_returns = [0.01, -0.01, 0.01, -0.01]

    returns, profits = calculate_trade_returns(candle_returns, initial_capital=500)

    assert len(returns) == 2
    assert returns[0] == pytest.approx(returns[1])
    assert profits[0] != pytest.approx(profits[1])
    assert abs(profits[1]) < abs(profits[0])


def test_calculate_trade_returns_ignores_leading_nan() -> None:
    candle_returns = [float("nan"), 0.01, -0.01]

    returns, profits = calculate_trade_returns(candle_returns)

    assert not math.isnan(returns[0])
    assert len(returns) == 1
    assert len(profits) == 1


def test_calculate_trade_returns_max_holding_period_force_closes_early() -> None:
    candle_returns = [0.01, 0.001, 0.001, 0.001, -0.01]

    unlimited_returns, _ = calculate_trade_returns(candle_returns)
    capped_returns, _ = calculate_trade_returns(candle_returns, max_holding_period=2)

    gross_return = (1 + 0.001) * (1 + 0.001)
    cost_factor = (1 - 0.0005) ** 2
    expected_capped_return = gross_return * cost_factor - 1
    assert len(unlimited_returns) == 1
    assert len(capped_returns) == 1
    assert capped_returns == pytest.approx([expected_capped_return])
    assert capped_returns[0] != pytest.approx(unlimited_returns[0])


def test_calculate_trade_returns_max_holding_period_does_not_affect_earlier_close() -> (
    None
):
    candle_returns = [0.01, 0.02, -0.01]

    unlimited_returns, _ = calculate_trade_returns(candle_returns)
    capped_returns, _ = calculate_trade_returns(candle_returns, max_holding_period=10)

    assert capped_returns == pytest.approx(unlimited_returns)


def test_calculate_trade_returns_stop_loss_force_closes_before_signal() -> None:
    # Entry, then six -0.4% candles: cumulative loss crosses -2% on the 6th
    # (0.996**6 - 1 ~= -2.37%), well before any exit/threshold signal would
    # fire (-0.004 never drops below -threshold=-0.005). Two big positive
    # candles follow, which re-trigger a fresh entry once the stopped-out
    # trade has freed up the position.
    candle_returns = [0.01] + [-0.004] * 6 + [0.05, 0.05]

    unstopped_returns, _ = calculate_trade_returns(candle_returns, threshold=0.005)
    stopped_returns, _ = calculate_trade_returns(
        candle_returns, threshold=0.005, stop_loss_pct=0.02
    )

    gross_return = (1 - 0.004) ** 6
    cost_factor = (1 - 0.0005) ** 2
    expected_stopped_return = gross_return * cost_factor - 1
    assert len(unstopped_returns) == 1
    assert len(stopped_returns) == 2
    assert stopped_returns[0] == pytest.approx(expected_stopped_return)
    assert stopped_returns[0] < unstopped_returns[0]


def test_calculate_trade_returns_stop_loss_does_not_affect_earlier_close() -> None:
    candle_returns = [0.01, -0.004, -0.004, -0.01]

    unstopped_returns, _ = calculate_trade_returns(candle_returns)
    stopped_returns, _ = calculate_trade_returns(candle_returns, stop_loss_pct=0.02)

    assert stopped_returns == pytest.approx(unstopped_returns)


def test_calculate_trade_returns_predicted_candle_returns_defaults_to_candle_returns() -> (
    None
):
    candle_returns = [0.01, 0.02, -0.01]

    default_returns, default_profits = calculate_trade_returns(candle_returns)
    explicit_returns, explicit_profits = calculate_trade_returns(
        candle_returns, candle_returns
    )

    assert explicit_returns == pytest.approx(default_returns)
    assert explicit_profits == pytest.approx(default_profits)


def test_calculate_trade_returns_predicted_candle_returns_can_act_before_the_move() -> (
    None
):
    # predicted_candle_returns[i] == candle_returns[i + 1] simulates a perfect
    # one-tick-ahead forecast: entry can trigger one tick earlier than the
    # reactive rule, capturing the spike at index 2 instead of missing it.
    candle_returns = [0.001, 0.001, 0.05, 0.001, -0.01, 0.001]
    predicted_candle_returns = [0.001, 0.05, 0.001, -0.01, 0.001, float("nan")]

    reactive_returns, _ = calculate_trade_returns(candle_returns)
    oracle_returns, _ = calculate_trade_returns(
        candle_returns, predicted_candle_returns
    )

    cost_factor = (1 - 0.0005) ** 2
    expected_reactive_return = (1 + 0.001) * (1 - 0.01) * cost_factor - 1
    expected_oracle_return = (1 + 0.05) * (1 + 0.001) * cost_factor - 1
    assert reactive_returns == pytest.approx([expected_reactive_return])
    assert oracle_returns == pytest.approx([expected_oracle_return])
    assert oracle_returns[0] > reactive_returns[0]


def test_calculate_trade_returns_mismatched_predicted_length_raises() -> None:
    with pytest.raises(ValueError):
        calculate_trade_returns([0.01, -0.01], [0.01, -0.01, 0.02])


def test_calculate_trade_returns_predicted_candle_returns_trailing_nan_is_no_signal() -> (
    None
):
    # A NaN signal (e.g. the trailing value of a real .shift(-1) series)
    # doesn't crash and isn't treated as a close trigger -- the position
    # stays open and is force-closed at the end of the data as usual.
    candle_returns = [0.01, -0.01]
    predicted_candle_returns = [0.01, float("nan")]

    returns, _ = calculate_trade_returns(candle_returns, predicted_candle_returns)

    cost_factor = (1 - 0.0005) ** 2
    expected_return = (1 - 0.01) * cost_factor - 1
    assert returns == pytest.approx([expected_return])
