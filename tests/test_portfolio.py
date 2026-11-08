from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from fartt.data import Candle, Dividend, sessions_in_range
from fartt.journal import Decision, Exit, Expiry, Fill, Positions
from fartt.portfolio import MAX_RECENT, Market, Portfolio, calculate_portfolio
from fartt.positions import order_window
from fartt.technical import calculate_indicators
from fartt.trade_plan import leg_costs
from tests.fakes import (
    RISK_CONFIG,
    buy_plan,
    decision_draft,
    exit_line,
    fill_line,
    stock_candle,
)

# Euronext Amsterdam and NYSE have both closed (plus margin) for 2026-11-05.
NOW = datetime(2026, 11, 5, 23, 0, tzinfo=UTC)
CAPITAL = RISK_CONFIG.limits.starting_capital
COSTS = RISK_CONFIG.costs
XAMS = COSTS.exchanges["XAMS"]
XNYS = COSTS.exchanges["XNYS"]
DAYS = sessions_in_range("XAMS", date(2026, 9, 1), date(2026, 11, 5))
US_DAYS = sessions_in_range("XNYS", date(2026, 9, 1), date(2026, 11, 5))
FX_DAYS = sessions_in_range("24/5", date(2026, 9, 1), date(2026, 11, 5))
LAST = 610.0  # ASML.AS's close on every session but where a test says otherwise


def history(
    days: Sequence[date] = DAYS, closes: Mapping[date, float] | None = None
) -> list[Candle]:
    closes = closes or {}
    return [stock_candle(day, closes.get(day, LAST)) for day in days]


def call(id_: int, **changes: Any) -> Decision:
    """A buy on ASML.AS on 2026-10-29 with its plan (entry 600, stop 580, target 680)."""
    fields: dict[str, Any] = {"plan": buy_plan()}
    return Decision(id=id_, **decision_draft(**(fields | changes)).model_dump())


def us_call(id_: int) -> Decision:
    plan = buy_plan(
        ticker="AAPL", currency="USD", exchange_code="XNYS", rate=1.16,
        rate_session=date(2026, 10, 29),
    )  # fmt: skip
    return call(id_, ticker="AAPL", currency="USD", exchange="XNYS", plan=plan)


def portfolio(
    decisions: Iterable[Decision],
    fills: Iterable[Fill] = (),
    exits: Iterable[Exit] = (),
    expiries: Iterable[Expiry] = (),
    candles: Mapping[str, Sequence[Candle]] | None = None,
    dividends: Mapping[str, Sequence[Dividend]] | None = None,
    not_updated: frozenset[str] = frozenset(),
) -> Portfolio:
    positions = Positions(
        {f.decision_id: f for f in fills},
        {e.decision_id: e for e in expiries},
        {e.decision_id: e for e in exits},
    )
    market = Market(
        candles if candles is not None else {"ASML.AS": history()},
        dividends or {},
    )
    return calculate_portfolio(
        list(decisions), positions, market, COSTS, CAPITAL, NOW, not_updated
    )


def sale(shares: int, price: float, rate: float | None = None) -> float:
    """What selling `shares` at `price` brings in EUR after the sell leg's costs."""
    currency = "EUR" if rate is None else "USD"
    exchange = XAMS if rate is None else XNYS
    costs = leg_costs(shares * price, rate, currency, exchange, COSTS, "sell")
    return round(round(shares * price / (rate or 1.0), 2) - costs.total, 2)


def test_no_plans_is_the_starting_capital() -> None:
    # @AC1
    result = portfolio([])

    totals = result.totals
    assert (totals.cash, totals.free_cash, totals.total_value) == (CAPITAL,) * 3
    assert (totals.open_value, totals.reserved, totals.open_risk) == (0, 0, 0)
    assert (totals.open_positions, totals.waiting_plans) == (0, 0)
    assert (result.open, result.waiting) == ([], [])
    assert (result.closed.count, result.closed.recent) == (0, [])
    assert result.based_on is None


def test_an_open_position_against_its_plan() -> None:
    # @AC2
    result = portfolio([call(1)], fills=[fill_line(1)])

    (held,) = result.open
    atr = calculate_indicators(history()).atr
    assert atr is not None
    value = sale(2, LAST)
    assert (held.id, held.ticker, held.currency, held.shares) == (
        1,
        "ASML.AS",
        "EUR",
        2,
    )
    assert (held.entry, held.stop, held.target) == (600.0, 580.0, 680.0)
    assert (held.last_close, held.based_on, held.rate) == (
        LAST,
        date(2026, 11, 5),
        None,
    )
    assert held.value == value
    assert held.dividends == 0
    assert held.unrealised_pnl == round(value - 1200.0 - 3.6, 2)
    assert held.to_stop == round((LAST - 580.0) / LAST, 4)
    assert held.to_target == round((680.0 - LAST) / LAST, 4)
    assert held.to_stop_atr == round((LAST - 580.0) / atr.value, 4)
    assert held.to_target_atr == round((680.0 - LAST) / atr.value, 4)
    # Filled on Friday 2026-10-30; held through 11-02, 11-03, 11-04 and 11-05.
    assert held.sessions_held == 4
    assert held.horizon_ends == date(2026, 11, 26)
    assert (held.is_current, held.warning) == (True, None)
    assert result.based_on == date(2026, 11, 5)
    assert result.totals.open_positions == 1


def test_the_totals_add_up_to_the_cent() -> None:
    # @AC3
    at_target = exit_line(
        3, reason="target", price=680.0, value=1360.0, dividends=4.0, pnl=156.8
    )
    result = portfolio(
        [call(1), call(2), call(3)],
        fills=[fill_line(1), fill_line(2), fill_line(3)],
        exits=[exit_line(2), at_target],
        dividends={"ASML.AS": [Dividend(date(2026, 11, 3), 1.37)]},
    )

    totals = result.totals
    assert totals.realised_pnl == round(-47.2 + 156.8, 2)
    assert totals.cash == round(
        CAPITAL - 3 * 1203.6 + (1160.0 - 3.6) + (1360.0 - 3.6 + 4.0) + 2.74, 2
    )
    assert round(totals.cash + totals.open_value, 2) == totals.total_value
    assert (
        round(CAPITAL + totals.realised_pnl + totals.unrealised_pnl, 2)
        == totals.total_value
    )


def test_a_waiting_plan_reserves_its_cash() -> None:
    # @AC5
    waiting = call(4)
    plan = waiting.plan
    assert plan is not None

    result = portfolio([waiting])

    reserved = round(plan.position_value + plan.costs.buy.total, 2)
    (listed,) = result.waiting
    assert listed.model_dump() == {
        "id": 4,
        "ticker": "ASML.AS",
        "entry": 600.0,
        "stop": 580.0,
        "target": 680.0,
        "shares": plan.shares,
        "reserved": reserved,
        "valid_until": order_window(waiting, plan)[1],
    }
    totals = result.totals
    assert (totals.cash, totals.reserved) == (CAPITAL, reserved)
    assert totals.free_cash == round(CAPITAL - reserved, 2)
    assert totals.total_value == CAPITAL
    assert totals.waiting_risk == plan.risk
    assert totals.waiting_plans == 1


def test_open_risk_is_the_further_loss_at_the_stops() -> None:
    # @AC9
    result = portfolio(
        [call(1), call(2)],
        fills=[fill_line(1), fill_line(2, shares=3, value=1800.0)],
    )

    expected = (sale(2, LAST) - sale(2, 580.0)) + (sale(3, LAST) - sale(3, 580.0))
    assert result.totals.open_risk == round(expected, 2)


def test_closed_positions_by_reason_and_recent() -> None:
    # @AC10
    exits = [
        exit_line(1, session=date(2026, 10, 30), reason="stop", pnl=-40.0),
        exit_line(2, session=date(2026, 11, 2), reason="target", pnl=150.0),
        exit_line(3, session=date(2026, 11, 3), reason="horizon", pnl=12.5),
        exit_line(4, session=date(2026, 11, 3), reason="stop", pnl=-35.1),
        exit_line(5, session=date(2026, 11, 4), reason="target", pnl=140.0),
        exit_line(6, session=date(2026, 11, 5), reason="horizon", pnl=-3.3),
        exit_line(7, session=date(2026, 11, 2), reason="stop", pnl=-50.0),
    ]
    ids = range(1, 8)

    result = portfolio(
        [call(i) for i in ids], fills=[fill_line(i) for i in ids], exits=exits
    )

    closed = result.closed
    assert closed.count == 7
    assert closed.realised_pnl == round(-40 + 150 + 12.5 - 35.1 + 140 - 3.3 - 50, 2)
    assert (closed.by_reason.stop.count, closed.by_reason.stop.pnl) == (3, -125.1)
    assert (closed.by_reason.target.count, closed.by_reason.target.pnl) == (2, 290.0)
    assert (closed.by_reason.horizon.count, closed.by_reason.horizon.pnl) == (2, 9.2)
    assert len(closed.recent) == MAX_RECENT
    assert [r.id for r in closed.recent] == [6, 5, 4, 3, 7]
    assert closed.recent[0].model_dump() == {
        "id": 6,
        "ticker": "ASML.AS",
        "session": date(2026, 11, 5),
        "reason": "horizon",
        "pnl": -3.3,
    }
    assert result.open == []


def test_a_dividend_while_open_is_in_the_unrealised_pnl() -> None:
    # @AC12
    dividends = [
        Dividend(date(2026, 10, 30), 5.0),  # Ex on the fill session: not held
        Dividend(date(2026, 11, 3), 3.0),
    ]

    result = portfolio(
        [call(1)], fills=[fill_line(1)], dividends={"ASML.AS": dividends}
    )

    (held,) = result.open
    assert held.dividends == 6.0
    assert held.unrealised_pnl == round(sale(2, LAST) + 6.0 - 1200.0 - 3.6, 2)


def test_a_same_session_round_trip_is_closed_only() -> None:
    same_day = exit_line(1, session=date(2026, 10, 30), price=575.0, value=1150.0)

    result = portfolio([call(1)], fills=[fill_line(1)], exits=[same_day])

    assert result.open == []
    assert result.closed.count == 1
    assert result.totals.cash == round(CAPITAL - 1203.6 + 1150.0 - 3.6, 2)


def rates(last: float, last_day: date = date(2026, 11, 5)) -> list[Candle]:
    days = [d for d in FX_DAYS if d < last_day]
    return [Candle(d, 1.16, 1.16, 1.16, 1.16, 0.0) for d in days] + [
        Candle(last_day, last, last, last, last, 0.0)
    ]


def test_a_usd_position_is_valued_at_its_closes_rate() -> None:
    fill = fill_line(1, rate=1.16, value=round(1200 / 1.16, 2))
    candles = {"AAPL": history(US_DAYS), "EURUSD=X": rates(1.15)}

    result = portfolio([us_call(1)], fills=[fill], candles=candles)

    (held,) = result.open
    assert (held.currency, held.rate) == ("USD", 1.15)
    assert held.value == sale(2, LAST, 1.15)
    assert held.warning is None


def test_a_usd_position_without_its_closes_rate_uses_the_one_before() -> None:
    fill = fill_line(1, rate=1.16, value=round(1200 / 1.16, 2))
    candles = {"AAPL": history(US_DAYS), "EURUSD=X": rates(1.14, date(2026, 11, 4))}

    result = portfolio([us_call(1)], fills=[fill], candles=candles)

    (held,) = result.open
    assert held.rate == 1.14
    assert held.value == sale(2, LAST, 1.14)
    assert held.is_current is True
    assert held.warning is not None
    assert "EUR/USD" in held.warning and "2026-11-04" in held.warning


def test_a_ticker_that_couldnt_be_updated_is_valued_on_its_cached_close() -> None:
    # @AC11
    candles = {"ASML.AS": history(DAYS[:-1], {DAYS[-2]: 605.0})}

    result = portfolio(
        [call(1)], fills=[fill_line(1)], candles=candles,
        not_updated=frozenset({"ASML.AS"}),
    )  # fmt: skip

    (held,) = result.open
    assert (held.last_close, held.based_on) == (605.0, date(2026, 11, 4))
    assert held.value == sale(2, 605.0)
    assert held.is_current is False
    assert held.warning is not None
    assert "couldn't bring ASML.AS up to date" in held.warning
    assert "1 session behind" in held.warning
    assert result.stale == ["ASML.AS"]


def test_a_session_not_published_yet_is_not_current() -> None:
    candles = {"ASML.AS": history(DAYS[:-1])}

    result = portfolio([call(1)], fills=[fill_line(1)], candles=candles)

    (held,) = result.open
    assert held.is_current is False
    assert held.warning is not None
    assert "hasn't published 2026-11-05" in held.warning
    assert result.stale == ["ASML.AS"]
