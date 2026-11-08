from datetime import UTC, date, datetime
from typing import Any

import pytest

from fartt.data import Candle, Dividend
from fartt.journal import BenchmarkClose, Decision
from fartt.scoring import Unscorable, score_call, score_plan
from tests.fakes import buy_plan, decision_draft, exit_line, fill_line, stock_candle

START = date(2026, 9, 1)
END = date(2026, 9, 29)
SCORED_AT = datetime(2026, 9, 30, 9, 0, tzinfo=UTC)


def decision(**changes: Any) -> Decision:
    fields: dict[str, Any] = {
        "ticker": "ASML.AS",
        "currency": "EUR",
        "exchange": "XAMS",
        "decision_session": START,
        "close": 100.0,
        "horizon_ends": END,
        "benchmark": BenchmarkClose(
            ticker="IWDA.AS", currency="EUR", session=START, close=50.0
        ),
    }
    draft = decision_draft(**(fields | changes))
    return Decision(id=1, **draft.model_dump())


def apple(**changes: Any) -> Decision:
    return decision(ticker="AAPL", currency="USD", exchange="XNYS", **changes)


def candles(closes: dict[date, float]) -> list[Candle]:
    return [stock_candle(day, close) for day, close in sorted(closes.items())]


STOCK = candles({START: 100.0, date(2026, 9, 15): 104.0, END: 110.0})
IWDA = candles({START: 50.0, date(2026, 9, 15): 51.0, END: 52.0})
EURUSD = candles({START: 1.10, END: 1.13})


def test_a_eur_stock() -> None:
    score = score_call(decision(), STOCK, [], IWDA, None, SCORED_AT)

    assert score.decision_id == 1
    assert score.scored_at == SCORED_AT
    assert score.return_ == 0.1
    assert score.benchmark.return_ == 0.04
    assert score.benchmark.ticker == "IWDA.AS"
    assert score.excess_return == 0.06
    assert score.right
    assert score.start.session == START
    assert score.end.session == END
    assert score.start.rate is None
    assert score.end.rate is None


def test_a_usd_stock_is_measured_in_euros() -> None:
    stock = candles({START: 100.0, END: 105.0})

    score = score_call(apple(), stock, [], IWDA, EURUSD, SCORED_AT)

    assert score.return_ == round(105 / 1.13 / (100 / 1.10) - 1, 6)
    assert score.start.rate == 1.1
    assert score.end.rate == 1.13
    assert score.benchmark.start.rate is None


def test_dividends_inside_the_horizon_count() -> None:
    dividends = [
        Dividend(START, 1.0),
        Dividend(date(2026, 9, 2), 1.0),
        Dividend(END, 1.0),
        Dividend(date(2026, 9, 30), 1.0),
    ]

    score = score_call(decision(), STOCK, dividends, IWDA, None, SCORED_AT)

    assert score.dividends == 2.0
    assert score.return_ == round((110 + 2) / 100 - 1, 6)


def test_a_usd_dividend_is_converted_at_the_end_rate() -> None:
    stock = candles({START: 100.0, END: 105.0})
    dividends = [Dividend(date(2026, 9, 10), 1.0)]

    score = score_call(apple(), stock, dividends, IWDA, EURUSD, SCORED_AT)

    assert score.return_ == round((105 + 1) / 1.13 / (100 / 1.10) - 1, 6)


def test_closes_come_from_the_candles_not_the_decision() -> None:
    score = score_call(decision(close=400.0), STOCK, [], IWDA, None, SCORED_AT)

    assert score.start.close == 100.0
    assert score.return_ == 0.1


def test_later_candles_change_nothing() -> None:
    later = date(2026, 10, 6)

    score = score_call(decision(), STOCK, [], IWDA, None, SCORED_AT)
    with_later = score_call(
        decision(),
        [*STOCK, stock_candle(later, 200.0)],
        [Dividend(later, 5.0)],
        [*IWDA, stock_candle(later, 10.0)],
        None,
        SCORED_AT,
    )

    assert with_later == score


@pytest.mark.parametrize(
    ("direction", "benchmark_end", "excess", "right"),
    [
        ("buy", 52.0, 0.01, True),
        ("buy", 52.5, 0.0, False),
        ("buy", 53.0, -0.01, False),
        ("pass", 52.0, 0.01, False),
        ("pass", 52.5, 0.0, True),
        ("pass", 53.0, -0.01, True),
    ],
)
def test_right_and_wrong(
    direction: str, benchmark_end: float, excess: float, right: bool
) -> None:
    stock = candles({START: 100.0, END: 105.0})
    benchmark = candles({START: 50.0, END: benchmark_end})

    score = score_call(
        decision(direction=direction), stock, [], benchmark, None, SCORED_AT
    )

    assert score.excess_return == excess
    assert score.right is right


def test_the_benchmark_ends_before_an_amsterdam_holiday() -> None:
    # 2027-03-26 is Good Friday: NYSE trades, Euronext doesn't.
    start, good_friday = date(2027, 3, 1), date(2027, 3, 26)
    call = apple(
        decision_session=start,
        horizon_ends=good_friday,
        benchmark=BenchmarkClose(
            ticker="IWDA.AS", currency="EUR", session=start, close=50.0
        ),
    )
    stock = candles({start: 100.0, good_friday: 105.0})
    benchmark = candles({start: 50.0, date(2027, 3, 25): 51.0})
    rates = candles({start: 1.1, good_friday: 1.1})

    score = score_call(call, stock, [], benchmark, rates, SCORED_AT)

    assert score.benchmark.end.session == date(2027, 3, 25)
    assert score.benchmark.return_ == 0.02


def test_a_missing_rate_falls_back_up_to_5_days() -> None:
    stock = candles({START: 100.0, END: 105.0})
    two_days = candles({START: 1.10, date(2026, 9, 27): 1.12, date(2026, 9, 30): 1.2})

    score = score_call(apple(), stock, [], IWDA, two_days, SCORED_AT)

    assert score.end.rate == 1.12

    six_days = candles({START: 1.10, date(2026, 9, 23): 1.12, date(2026, 9, 30): 1.2})
    with pytest.raises(
        Unscorable, match="there's no EUR/USD rate within 5 days before 2026-09-29"
    ):
        score_call(apple(), stock, [], IWDA, six_days, SCORED_AT)


@pytest.mark.parametrize(
    ("call", "stock", "benchmark", "rates", "reason"),
    [
        (
            decision(),
            candles({START: 100.0}),
            IWDA,
            None,
            "ASML.AS's close for 2026-09-29 isn't available yet",
        ),
        (
            decision(),
            candles({END: 100.0}),
            IWDA,
            None,
            "ASML.AS's close for 2026-09-01 isn't available yet",
        ),
        (
            decision(),
            STOCK,
            candles({START: 50.0}),
            None,
            "IWDA.AS's close for 2026-09-29 isn't available yet",
        ),
        (
            apple(),
            STOCK,
            IWDA,
            candles({START: 1.1, date(2026, 9, 28): 1.12}),
            "the EUR/USD rate for 2026-09-29 isn't available yet",
        ),
        (
            apple(),
            STOCK,
            IWDA,
            [],
            "the EUR/USD rate for 2026-09-01 isn't available yet",
        ),
    ],
)
def test_unscorable_reasons(
    call: Decision,
    stock: list[Candle],
    benchmark: list[Candle],
    rates: list[Candle] | None,
    reason: str,
) -> None:
    with pytest.raises(Unscorable) as raised:
        score_call(call, stock, [], benchmark, rates, SCORED_AT)

    assert str(raised.value) == reason


# A plan made on 2026-10-29 that filled at 600 on 2026-10-30 (Friday).
FILLED = date(2026, 10, 30)


def planned(**changes: Any) -> Decision:
    draft = decision_draft(plan=buy_plan(), **changes)
    return Decision(id=1, **draft.model_dump())


def test_a_plan_closed_at_its_target_is_a_hit() -> None:
    # @AC1
    call = planned()
    exit = exit_line(
        1,
        session=date(2026, 11, 4),
        reason="target",
        price=680.0,
        value=1360.0,
        pnl=152.8,
    )
    benchmark = candles(
        {date(2026, 10, 29): 101.25, FILLED: 101.5, date(2026, 11, 4): 104.0}
    )

    score = score_plan(call, fill_line(1), exit, benchmark, SCORED_AT)

    assert call.plan is not None
    stock = round(152.8 / (1200.0 + 3.6), 6)
    assert score.decision_id == 1
    assert score.scored_at == SCORED_AT
    assert score.return_ == stock
    assert score.benchmark.ticker == "IWDA.AS"
    assert score.benchmark.start.session == date(2026, 10, 29)
    assert score.benchmark.start.close == 101.25
    assert score.benchmark.end.session == date(2026, 11, 4)
    assert score.benchmark.end.close == 104.0
    assert score.benchmark.return_ == round(104.0 / 101.25 - 1, 6)
    assert score.excess_return == round(stock - round(104.0 / 101.25 - 1, 6), 6)
    assert score.hit is True
    assert score.realised_r == round(152.8 / call.plan.risk, 6)
    assert score.reason == "target"
    assert score.holding_sessions == 4


@pytest.mark.parametrize(("reason", "pnl"), [("stop", -47.2), ("horizon", 12.5)])
def test_a_stop_and_a_horizon_exit_are_misses(reason: str, pnl: float) -> None:
    # @AC2 @AC7
    call = planned()
    benchmark = candles({date(2026, 10, 29): 101.25, date(2026, 11, 2): 102.0})

    score = score_plan(
        call, fill_line(1), exit_line(1, reason=reason, pnl=pnl), benchmark, SCORED_AT
    )

    assert call.plan is not None
    assert score.hit is False
    assert score.reason == reason
    assert score.return_ == round(pnl / 1203.6, 6)
    assert score.realised_r == round(pnl / call.plan.risk, 6)
    assert score.holding_sessions == 2


def test_the_benchmark_starts_on_the_last_amsterdam_session_before_the_fill() -> None:
    # 2027-03-26 (Good Friday) and 2027-03-29 (Easter Monday): Euronext is
    # closed, so a US fill on Tuesday 2027-03-30 starts from Thursday's close.
    call = planned(ticker="AAPL", currency="USD", exchange="XNYS")
    fill = fill_line(1, session=date(2027, 3, 30))
    exit = exit_line(1, session=date(2027, 4, 1))
    benchmark = candles({date(2027, 3, 25): 50.0, date(2027, 4, 1): 51.0})

    score = score_plan(call, fill, exit, benchmark, SCORED_AT)

    assert score.benchmark.start.session == date(2027, 3, 25)
    assert score.benchmark.end.session == date(2027, 4, 1)
    assert score.benchmark.return_ == 0.02
    assert score.holding_sessions == 3


def test_benchmark_closes_are_rounded_before_the_return() -> None:
    benchmark = candles({date(2026, 10, 29): 100.00004, date(2026, 11, 2): 101.00004})

    score = score_plan(planned(), fill_line(1), exit_line(1), benchmark, SCORED_AT)

    assert score.benchmark.start.close == 100.0
    assert score.benchmark.return_ == 0.01


def test_a_missing_benchmark_close_is_unscorable() -> None:
    # @AC3
    benchmark = candles({date(2026, 10, 29): 101.25})

    with pytest.raises(Unscorable) as raised:
        score_plan(planned(), fill_line(1), exit_line(1), benchmark, SCORED_AT)

    assert raised.value.ticker == "IWDA.AS"
    assert str(raised.value) == "IWDA.AS's close for 2026-11-02 isn't available yet"
