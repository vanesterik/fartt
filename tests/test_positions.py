from datetime import UTC, date, datetime
from typing import Any

import pytest

from fartt.data import Candle, Dividend, Split, session_after
from fartt.journal import Decision, Exit, Expiry, Fill
from fartt.positions import (
    Blocked,
    Settled,
    order_window,
    position_state,
    settle_plan,
)
from fartt.trade_plan import TradePlan, leg_costs
from tests.fakes import RISK_CONFIG, buy_plan, decision_draft

COSTS = RISK_CONFIG.costs
BASED_ON = date(2026, 10, 1)
SETTLED_AT = datetime(2026, 11, 5, 21, 0, tzinfo=UTC)
ENTRY, STOP, TARGET = 100.0, 95.0, 115.0


def day(n: int, exchange: str = "XAMS") -> date:
    """The `n`-th session after the plan's session."""
    return session_after(exchange, BASED_ON, n)


HORIZON_ENDS = day(20)


def plan(**changes: Any) -> TradePlan:
    fields: dict[str, Any] = {
        "ticker": "TEST.AS",
        "based_on": BASED_ON,
        "last_close": 101.0,
        "atr": 3.0,
        "entry": ENTRY,
        "stop": STOP,
        "target": TARGET,
        "valid_for": 5,
    }
    return buy_plan(**(fields | changes))


def decision(trade_plan: TradePlan | None = None, **changes: Any) -> Decision:
    trade_plan = trade_plan or plan()
    fields: dict[str, Any] = {
        "ticker": trade_plan.ticker,
        "currency": trade_plan.currency,
        "exchange": trade_plan.exchange,
        "decision_session": BASED_ON,
        "close": trade_plan.last_close,
        "horizon_ends": HORIZON_ENDS,
        "plan": trade_plan,
    }
    return Decision(id=1, **decision_draft(**(fields | changes)).model_dump())


QUIET = (102.0, 103.0, 101.0, 102.0)  # Above the entry, between stop and target


def candles(
    sessions: dict[int, tuple[float, float, float, float]] | None = None,
    count: int = 25,
    exchange: str = "XAMS",
) -> list[Candle]:
    """Quiet sessions 1..`count` after the plan, with `sessions` (open, high, low, close)."""
    sessions = sessions or {}
    return [
        Candle(day(n, exchange), *sessions.get(n, QUIET), 1000.0)
        for n in range(1, count + 1)
    ]


def settle(
    trade_plan: TradePlan | None = None,
    sessions: dict[int, tuple[float, float, float, float]] | None = None,
    *,
    fill: Fill | None = None,
    newest: date | None = None,
    dividends: list[Dividend] | None = None,
    splits: list[Split] | None = None,
    rates: list[Candle] | None = None,
    count: int = 25,
    **changes: Any,
) -> Settled:
    call = decision(trade_plan, **changes)
    assert call.plan is not None
    return settle_plan(
        call,
        call.plan,
        fill,
        candles(sessions, count, call.exchange),
        dividends or [],
        splits or [],
        rates,
        COSTS,
        newest or day(25, call.exchange),
        SETTLED_AT,
    )


def buy_costs(price: float, shares: int, rate: float | None = None) -> Any:
    return leg_costs(
        shares * price, rate, "EUR" if rate is None else "USD",
        COSTS.exchanges["XAMS" if rate is None else "XNYS"], COSTS, "buy",
    )  # fmt: skip


def sell_costs(price: float, shares: int, rate: float | None = None) -> Any:
    return leg_costs(
        shares * price, rate, "EUR" if rate is None else "USD",
        COSTS.exchanges["XAMS" if rate is None else "XNYS"], COSTS, "sell",
    )  # fmt: skip


def test_a_low_at_the_entry_fills_at_the_entry() -> None:
    # @AC1
    shares = plan().shares

    settled = settle(sessions={2: (101.0, 102.0, 99.5, 101.0)})

    assert settled.fill == Fill(
        decision_id=1,
        settled_at=SETTLED_AT,
        session=day(2),
        price=ENTRY,
        at_open=False,
        shares=shares,
        rate=None,
        value=round(shares * ENTRY, 2),
        costs=buy_costs(ENTRY, shares),
    )
    assert settled.blocked is None


def test_an_open_below_the_entry_fills_at_the_open() -> None:
    # @AC2
    settled = settle(sessions={3: (98.7, 99.0, 97.0, 98.0)})

    assert settled.fill is not None
    assert settled.fill.session == day(3)
    assert settled.fill.price == 98.7
    assert settled.fill.at_open
    assert settled.fill.costs == buy_costs(98.7, plan().shares)


def test_an_open_at_the_entry_fills_at_the_open() -> None:
    # @AC2
    settled = settle(sessions={1: (100.0, 101.0, 100.0, 100.5)})

    assert settled.fill is not None
    assert settled.fill.at_open


def test_a_plan_whose_entry_is_not_reached_waits_then_expires() -> None:
    # @AC3
    waiting = settle(newest=day(4))
    assert waiting == Settled(None, None, None, None)

    expired = settle(newest=day(5))
    assert expired.fill is None
    assert expired.expiry == Expiry(
        decision_id=1, settled_at=SETTLED_AT, session=day(5)
    )


def test_a_low_after_the_window_does_not_fill() -> None:
    # @AC3
    settled = settle(sessions={6: (99.0, 100.0, 98.0, 99.0)})

    assert settled.fill is None
    assert settled.expiry is not None


def test_the_window_never_runs_past_the_horizon() -> None:
    # @AC3
    call = decision(horizon_ends=day(2))
    assert call.plan is not None

    assert order_window(call, call.plan) == (day(1), day(2))
    assert order_window(decision(), plan()) == (day(1), day(5))


def filled(sessions: dict[int, tuple[float, float, float, float]]) -> Settled:
    """Settled after a fill at the entry on session 1."""
    return settle(sessions={1: (101.0, 101.5, 99.8, 100.5)} | sessions)


def test_a_low_at_the_stop_closes_at_the_stop() -> None:
    # @AC4
    shares = plan().shares

    settled = filled({4: (98.0, 99.0, 94.9, 96.0)})

    assert settled.fill is not None
    sell = sell_costs(STOP, shares)
    value = round(shares * STOP, 2)
    assert settled.exit == Exit(
        decision_id=1,
        settled_at=SETTLED_AT,
        session=day(4),
        reason="stop",
        price=STOP,
        at_open=False,
        slippage=0.0,
        rate=None,
        value=value,
        costs=sell,
        dividends=0.0,
        pnl=round(
            value - settled.fill.value - settled.fill.costs.total - sell.total, 2
        ),
    )


def test_an_open_below_the_stop_closes_at_the_open_with_slippage() -> None:
    # @AC5
    settled = filled({4: (93.2, 96.0, 92.0, 95.5)})

    assert settled.exit is not None
    assert settled.exit.reason == "stop"
    assert settled.exit.price == 93.2
    assert settled.exit.at_open
    assert settled.exit.slippage == 1.8


def test_a_high_at_the_target_closes_at_the_target() -> None:
    # @AC6
    settled = filled({7: (110.0, 115.5, 109.0, 114.0)})

    assert settled.exit is not None
    assert (settled.exit.reason, settled.exit.price, settled.exit.at_open) == (
        "target",
        TARGET,
        False,
    )
    assert settled.exit.slippage == 0.0


def test_an_open_above_the_target_closes_at_the_open() -> None:
    # @AC6
    settled = filled({7: (117.25, 118.0, 116.0, 117.0)})

    assert settled.exit is not None
    assert (settled.exit.reason, settled.exit.price, settled.exit.at_open) == (
        "target",
        117.25,
        True,
    )


def test_a_session_reaching_both_closes_at_the_stop() -> None:
    # @AC7
    settled = filled({5: (100.0, 116.0, 94.0, 100.0)})

    assert settled.exit is not None
    assert (settled.exit.reason, settled.exit.price) == ("stop", STOP)


def test_on_the_fill_session_the_target_does_not_close_but_the_stop_does() -> None:
    # @AC7
    target_only = settle(sessions={2: (101.0, 116.0, 99.0, 112.0)})
    assert target_only.fill is not None
    assert target_only.exit is not None
    assert target_only.exit.reason == "horizon"

    stopped = settle(sessions={2: (101.0, 116.0, 94.0, 96.0)})
    assert stopped.exit is not None
    assert (stopped.exit.session, stopped.exit.reason, stopped.exit.price) == (
        day(2),
        "stop",
        STOP,
    )


def test_a_fill_below_the_stop_closes_at_that_open() -> None:
    settled = settle(sessions={2: (94.0, 97.0, 93.0, 96.0)})

    assert settled.fill is not None
    assert settled.fill.price == 94.0
    assert settled.exit is not None
    assert (settled.exit.session, settled.exit.reason, settled.exit.price) == (
        day(2),
        "stop",
        94.0,
    )
    assert settled.exit.at_open
    assert settled.exit.slippage == 1.0


def test_neither_closes_at_the_horizon_close() -> None:
    # @AC8
    settled = filled({20: (102.0, 103.0, 101.0, 102.37)})

    assert settled.exit is not None
    assert (settled.exit.session, settled.exit.reason, settled.exit.price) == (
        HORIZON_ENDS,
        "horizon",
        102.37,
    )
    assert not settled.exit.at_open


def test_an_open_position_waits_for_the_horizon() -> None:
    # @AC8
    settled = settle(sessions={1: (101.0, 101.5, 99.8, 100.5)}, newest=day(19))

    assert settled.fill is not None
    assert settled.exit is None
    assert settled.blocked is None


def us_plan() -> TradePlan:
    return plan(
        ticker="TEST",
        currency="USD",
        exchange_code="XNYS",
        rate=1.16,
        rate_session=BASED_ON,
    )


def eurusd(closes: dict[date, float], last: date) -> list[Candle]:
    """EUR/USD closes: 1.16 on every weekday up to `last`, but `closes`."""
    days = [
        session_after("24/5", date(2026, 9, 30), n)
        for n in range(1, 60)
        if session_after("24/5", date(2026, 9, 30), n) <= last
    ]
    return [Candle(d, 1.16, 1.16, 1.16, closes.get(d, 1.16), 0.0) for d in days]


def test_a_us_position_converts_each_leg_at_its_own_rate() -> None:
    # @AC9
    trade_plan = us_plan()
    shares = trade_plan.shares
    rates = eurusd({day(2, "XNYS"): 1.1712356, day(6, "XNYS"): 1.15}, day(25, "XNYS"))

    settled = settle(
        trade_plan,
        {2: (101.0, 102.0, 99.0, 101.0), 6: (99.0, 99.5, 94.0, 95.0)},
        rates=rates,
    )

    assert settled.fill is not None
    assert settled.fill.rate == 1.171236
    assert settled.fill.value == round(shares * ENTRY / 1.171236, 2)
    assert settled.fill.costs == buy_costs(ENTRY, shares, 1.171236)
    assert settled.exit is not None
    assert settled.exit.rate == 1.15
    assert settled.exit.value == round(shares * STOP / 1.15, 2)
    assert settled.exit.costs == sell_costs(STOP, shares, 1.15)


def test_a_dividend_while_held_is_added_to_the_pnl() -> None:
    # @AC9
    shares = plan().shares
    dividends = [
        Dividend(day(1), 5.0),  # Ex on the fill session: not the holder's
        Dividend(day(3), 1.25),
        Dividend(day(10), 0.5),  # After the exit
    ]

    settled = settle(
        sessions={1: (101.0, 101.5, 99.8, 100.5), 4: (98.0, 99.0, 94.9, 96.0)},
        dividends=dividends,
    )

    assert settled.fill is not None
    assert settled.exit is not None
    assert settled.exit.dividends == round(shares * 1.25, 2)
    assert settled.exit.pnl == round(
        settled.exit.value
        - settled.fill.value
        + settled.exit.dividends
        - settled.fill.costs.total
        - settled.exit.costs.total,
        2,
    )


def test_a_session_not_cached_blocks_and_writes_nothing() -> None:
    # @AC12
    settled = settle(count=2, newest=day(4))

    assert settled == Settled(
        None,
        None,
        None,
        Blocked(
            "TEST.AS", f"TEST.AS's candle for {day(3).isoformat()} isn't available yet"
        ),
    )


def test_a_leg_without_its_rate_writes_nothing_until_it_comes() -> None:
    # @AC12
    sessions = {2: (101.0, 102.0, 99.0, 101.0)}

    early = settle(us_plan(), sessions, rates=eurusd({}, day(1, "XNYS")))
    assert early.fill is None
    assert early.blocked is not None
    assert early.blocked.ticker == "EURUSD=X"
    assert day(2, "XNYS").isoformat() in early.blocked.reason

    later = settle(us_plan(), sessions, rates=eurusd({}, day(25, "XNYS")))
    assert later.fill is not None
    assert later.blocked is None


def test_an_exit_without_its_rate_still_gives_the_fill() -> None:
    # @AC12
    rates = eurusd({}, day(5, "XNYS"))

    settled = settle(
        us_plan(),
        {2: (101.0, 102.0, 99.0, 101.0), 6: (99.0, 99.5, 94.0, 95.0)},
        rates=rates,
    )

    assert settled.fill is not None
    assert settled.exit is None
    assert settled.blocked is not None
    assert settled.blocked.ticker == "EURUSD=X"


def test_a_split_after_the_fill_is_not_settled() -> None:
    # @AC12
    first = settle(sessions={1: (101.0, 101.5, 99.8, 100.5)}, newest=day(2))
    assert first.fill is not None

    settled = settle(fill=first.fill, splits=[Split(day(3), 4.0)], newest=day(5))

    assert settled.exit is None
    assert settled.blocked is not None
    assert settled.blocked.ticker == "TEST.AS"
    assert "split" in settled.blocked.reason
    assert day(3).isoformat() in settled.blocked.reason


def test_a_split_while_waiting_is_not_settled() -> None:
    settled = settle(
        sessions={3: (25.0, 25.5, 24.0, 25.0)}, splits=[Split(day(2), 4.0)]
    )

    assert settled.fill is None
    assert settled.expiry is None
    assert settled.blocked is not None


def test_a_split_before_the_plan_is_no_concern() -> None:
    settled = settle(splits=[Split(BASED_ON, 2.0)])

    assert settled.blocked is None
    assert settled.expiry is not None


def test_resuming_an_open_position_gives_the_same_exit() -> None:
    # @AC10
    sessions = {1: (101.0, 101.5, 99.8, 100.5)}
    whole = settle(sessions=sessions | {6: (99.0, 99.5, 94.0, 95.0)})

    first = settle(sessions=sessions, newest=day(3))
    assert first.exit is None
    resumed = settle(sessions=sessions | {6: (99.0, 99.5, 94.0, 95.0)}, fill=first.fill)

    assert resumed.fill is None
    assert resumed.exit == whole.exit


@pytest.mark.parametrize(
    ("fill", "expiry", "exit", "state"),
    [
        (False, False, False, "waiting"),
        (True, False, False, "open"),
        (True, False, True, "closed"),
        (False, True, False, "expired"),
    ],
)
def test_position_state(fill: bool, expiry: bool, exit: bool, state: str) -> None:
    settled = filled({4: (98.0, 99.0, 94.9, 96.0)})
    assert settled.fill is not None
    assert settled.exit is not None

    assert (
        position_state(
            settled.fill if fill else None,
            Expiry(decision_id=1, settled_at=SETTLED_AT, session=day(5))
            if expiry
            else None,
            settled.exit if exit else None,
        )
        == state
    )
