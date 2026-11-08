import math
from datetime import date
from typing import Any

import pytest

from fartt.trade_plan import PlanRefused, TradePlan, calculate_plan, leg_costs
from tests.fakes import RISK_CONFIG

COSTS = RISK_CONFIG.costs
LIMITS = RISK_CONFIG.limits
EXCHANGES = COSTS.exchanges


def costs(leg: Any) -> tuple[float, float, float, float, float, float]:
    return (
        leg.commission,
        leg.slippage,
        leg.conversion,
        leg.tax,
        leg.fee,
        leg.total,
    )


def test_a_euronext_leg_pays_the_minimum_commission_and_slippage() -> None:
    leg = leg_costs(1283.60, None, "EUR", EXCHANGES["XAMS"], COSTS, "buy")

    assert costs(leg) == (2.0, 1.28, 0.0, 0.0, 0.0, 3.28)


def test_a_us_leg_is_converted_at_the_rate() -> None:
    buy = leg_costs(2000.0, 1.16, "USD", EXCHANGES["XNYS"], COSTS, "buy")
    sell = leg_costs(2000.0, 1.16, "USD", EXCHANGES["XNYS"], COSTS, "sell")

    # 1.60 USD commission, 0.1% slippage and 0.25% conversion of 1724.14 EUR.
    assert costs(buy) == (1.38, 1.72, 4.31, 0.0, 0.0, 7.41)
    # The SEC fee on the sale, 0.00229% of 1724.14 EUR.
    assert costs(sell) == (1.38, 1.72, 4.31, 0.04, 0.0, 7.45)


def test_the_french_ftt_is_on_the_buy_leg_only() -> None:
    buy = leg_costs(1000.0, None, "EUR", EXCHANGES["XPAR"], COSTS, "buy")
    sell = leg_costs(1000.0, None, "EUR", EXCHANGES["XPAR"], COSTS, "sell")

    assert costs(buy) == (2.0, 1.0, 0.0, 4.0, 0.0, 7.0)
    assert costs(sell) == (2.0, 1.0, 0.0, 0.0, 0.0, 3.0)


def test_dublin_charges_stamp_duty_and_a_fixed_fee_on_each_leg() -> None:
    buy = leg_costs(1000.0, None, "EUR", EXCHANGES["XDUB"], COSTS, "buy")
    sell = leg_costs(1000.0, None, "EUR", EXCHANGES["XDUB"], COSTS, "sell")

    assert costs(buy) == (5.0, 1.0, 0.0, 10.0, 1.25, 17.25)
    assert costs(sell) == (5.0, 1.0, 0.0, 0.0, 1.25, 7.25)


def test_oslo_converts_its_nok_minimum_commission_at_the_nok_rate() -> None:
    leg = leg_costs(5000.0, 11.5, "NOK", EXCHANGES["XOSL"], COSTS, "buy")

    # 10 NOK minimum / 11.5; 0.1% and 0.25% of 434.78 EUR.
    assert costs(leg) == (0.87, 0.43, 1.09, 0.0, 0.0, 2.39)


def test_a_commission_in_a_third_currency_is_refused() -> None:
    odd = EXCHANGES["XOSL"].model_copy(update={"commission_currency": "USD"})

    with pytest.raises(PlanRefused, match="USD"):
        leg_costs(5000.0, 11.5, "NOK", odd, COSTS, "buy")


def plan(**changes: Any) -> TradePlan:
    arguments: dict[str, Any] = {
        "ticker": "TEST.AS",
        "currency": "EUR",
        "exchange_code": "XAMS",
        "based_on": date(2026, 11, 4),
        "last_close": 50.5,
        "atr": 1.5,
        "entry": 50.0,
        "stop": 44.0,
        "target": 70.0,
        "valid_for": 5,
        "rate": None,
        "portfolio_value": LIMITS.starting_capital,
        "limits": LIMITS,
        "costs": COSTS,
    }
    return calculate_plan(**(arguments | changes))


def test_a_wide_stop_is_sized_by_the_risk_limit() -> None:
    result = plan()

    # 32 shares risk 192 + 3.60 buying + 3.41 selling at 44 = 199.01; 33 risk more.
    assert result.shares == 32
    assert result.sized_by == "max_risk_per_position"
    assert result.position_value == 1600.0
    assert result.risk == 199.01
    assert result.risk_share == 0.0199
    assert result.position_share == 0.16
    assert costs(result.costs.buy) == (2.0, 1.6, 0.0, 0.0, 0.0, 3.6)
    assert costs(result.costs.sell_at_stop) == (2.0, 1.41, 0.0, 0.0, 0.0, 3.41)
    assert costs(result.costs.sell_at_target) == (2.0, 2.24, 0.0, 0.0, 0.0, 4.24)
    # 32 x 20 - 3.60 - 4.24.
    assert result.reward == 632.16
    assert result.reward_to_risk == round(632.16 / 199.01, 4)
    assert result.reward_to_risk_before_costs == round(20 / 6, 4)
    assert result.portfolio_value == 10000.0
    assert result.rate is None
    assert result.limits == LIMITS
    assert result.tariff.name == COSTS.tariff
    assert result.tariff.exchange == EXCHANGES["XAMS"]
    assert (result.tariff.currency_conversion, result.tariff.slippage_per_leg) == (
        COSTS.currency_conversion,
        COSTS.slippage_per_leg,
    )
    assert (result.entry, result.stop, result.target, result.valid_for) == (
        50.0,
        44.0,
        70.0,
        5,
    )
    assert (result.last_close, result.atr, result.based_on) == (
        50.5,
        1.5,
        date(2026, 11, 4),
    )


def test_a_tight_stop_is_sized_at_the_position_limit() -> None:
    result = plan(stop=48.0)

    # 2000 / 50 is 40 shares exactly: the share at the limit is allowed.
    assert result.shares == 40
    assert result.position_value == 2000.0
    assert result.sized_by == "max_position_size"
    assert result.risk <= 200


@pytest.mark.parametrize(
    ("changes", "exchange"),
    [
        ({}, "XAMS"),
        (
            {
                "ticker": "TEST",
                "currency": "USD",
                "exchange_code": "XNYS",
                "rate": 1.16,
                "entry": 100.0,
                "stop": 95.0,
                "target": 115.0,
                "last_close": 101.0,
                "atr": 3.0,
            },
            "XNYS",
        ),
    ],
)
def test_the_risk_is_the_loss_at_the_stop_plus_every_cost_of_both_legs(
    changes: dict[str, Any], exchange: str
) -> None:
    result = plan(**changes)

    rate = result.rate or 1.0
    distance = result.shares * (result.entry - result.stop) / rate
    buy = leg_costs(
        result.shares * result.entry,
        result.rate,
        result.currency,
        EXCHANGES[exchange],
        COSTS,
        "buy",
    )
    sell = leg_costs(
        result.shares * result.stop,
        result.rate,
        result.currency,
        EXCHANGES[exchange],
        COSTS,
        "sell",
    )
    assert result.costs.buy == buy
    assert result.costs.sell_at_stop == sell
    assert result.risk == round(distance + buy.total + sell.total, 2)
    assert result.risk <= LIMITS.max_risk_per_position * LIMITS.starting_capital


def test_a_us_plan_converts_at_the_rate() -> None:
    result = plan(
        currency="USD",
        exchange_code="XNYS",
        rate=1.16,
        entry=100.0,
        stop=95.0,
        target=115.0,
        last_close=101.0,
        atr=3.0,
    )

    # 2000 EUR at 1.16 is 2320 USD: 23 shares of 100.
    assert result.shares == 23
    assert result.rate == 1.16
    assert result.position_value == round(2300 / 1.16, 2)
    assert result.costs.buy.conversion > 0


def test_a_reward_to_risk_below_the_minimum_is_refused() -> None:
    with pytest.raises(PlanRefused) as refused:
        plan(stop=48.0, target=53.0)

    message = str(refused.value)
    assert "min_reward_to_risk" in message
    assert "2.0" in message
    assert "falls short by" in message


def test_the_reward_to_risk_is_compared_before_it_is_rounded() -> None:
    # Target 66: reward 504.29 / risk 199.01 is 2.533993, shown as 2.534.
    limits = LIMITS.model_copy(update={"min_reward_to_risk": 2.533999})

    with pytest.raises(PlanRefused) as refused:
        plan(target=66.0, limits=limits)

    assert "is 2.533993" in str(refused.value)


@pytest.mark.parametrize(
    ("changes", "named", "amount"),
    [
        (
            {"entry": 1500.0, "stop": 1290.0, "target": 2500.0},
            "max_risk_per_position",
            "would risk",
        ),
        (
            {"entry": 2500.0, "stop": 2490.0, "target": 2600.0},
            "max_position_size",
            "costs",
        ),
    ],
)
def test_one_share_beyond_a_limit_is_refused(
    changes: dict[str, Any], named: str, amount: str
) -> None:
    with pytest.raises(PlanRefused) as refused:
        plan(last_close=changes["entry"], atr=50.0, **changes)

    message = str(refused.value)
    assert "One share" in message
    assert named in message
    assert amount in message


def test_one_share_beyond_both_limits_names_both() -> None:
    with pytest.raises(PlanRefused) as refused:
        plan(entry=3000.0, stop=2700.0, target=4000.0, last_close=3000.0, atr=50.0)

    message = str(refused.value)
    assert "max_risk_per_position" in message
    assert "max_position_size" in message


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"stop": 50.0}, "stop"),
        ({"stop": 51.0}, "stop"),
        ({"target": 50.0}, "target"),
        ({"target": 49.0}, "target"),
        ({"entry": 0.0}, "entry"),
        ({"stop": -1.0}, "stop"),
        ({"target": 0.0}, "target"),
        ({"valid_for": 0}, "valid_for"),
        ({"valid_for": 11}, "valid_for"),
        ({"entry": 52.01}, "1 ATR"),
    ],
)
def test_prices_that_do_not_make_a_long_plan_are_refused(
    changes: dict[str, Any], named: str
) -> None:
    with pytest.raises(PlanRefused, match=named):
        plan(**changes)


@pytest.mark.parametrize("name", ["entry", "stop", "target"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_price_that_is_not_a_number_is_refused(name: str, value: float) -> None:
    with pytest.raises(PlanRefused, match=f"{name} must be a finite number"):
        plan(**{name: value})


def test_an_entry_exactly_1_atr_above_the_close_is_allowed() -> None:
    assert plan(entry=52.0).entry == 52.0


def test_an_entry_far_below_the_close_is_allowed() -> None:
    assert plan(entry=30.0, stop=28.0, target=60.0).entry == 30.0


def test_without_an_atr_the_plan_is_refused() -> None:
    with pytest.raises(PlanRefused, match="ATR"):
        plan(atr=None)


def test_a_plan_round_trips_through_json() -> None:
    result = plan()

    assert TradePlan.model_validate_json(result.model_dump_json()) == result
