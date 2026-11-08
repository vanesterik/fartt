from datetime import UTC, date, datetime
from typing import Any

from fartt.journal import (
    SIGNALS,
    BenchmarkScore,
    Decision,
    ExitReason,
    Leg,
    PlanScore,
    Positions,
    Score,
    Signal,
)
from fartt.performance import (
    MAX_TICKERS,
    MIN_CALLS,
    Gate,
    GateCondition,
    Period,
    PlanPerformance,
    PlanPeriod,
    ReasonGroup,
    calculate_gate,
    calculate_performance,
    calculate_plans,
)
from tests.fakes import (
    buy_plan,
    decision_draft,
    decision_score,
    exit_line,
    fill_line,
    plan_score_line,
)


class Journal:
    """Calls and their scores, as `Journal.decisions()` and `scores()` return them."""

    def __init__(self) -> None:
        self.decisions: list[Decision] = []
        self.scores: dict[int, Score] = {}

    def open(self, **changes: Any) -> Decision:
        decision = Decision(
            id=len(self.decisions) + 1, **decision_draft(**changes).model_dump()
        )
        self.decisions.append(decision)
        return decision

    def scored(
        self,
        excess: float,
        stock: float = 0.03,
        benchmark: float = 0.02,
        **changes: Any,
    ) -> Decision:
        decision = self.open(**changes)
        right = excess > 0 if decision.direction == "buy" else excess <= 0
        self.scores[decision.id] = decision_score(
            decision.id,
            return_=stock,
            benchmark=BenchmarkScore(
                ticker="IWDA.AS",
                start=Leg(session=decision.decision_session, close=100.0),
                end=Leg(session=decision.horizon_ends, close=102.0),
                return_=benchmark,
            ),
            excess_return=excess,
            right=right,
        )
        return decision


def test_groups_report_the_means_of_the_stored_scores() -> None:
    # @AC1 @AC2
    journal = Journal()
    buys = [0.02, -0.01, 0.03, 0.01, 0.0, 0.04, -0.02, 0.05, 0.01, 0.02]
    for day, excess in enumerate(buys, start=1):
        journal.scored(
            excess,
            stock=excess + 0.01,
            benchmark=0.01,
            decision_session=date(2026, 9, day),
            horizon_ends=date(2026, 10, day),
        )
    passes = [-0.02, 0.01, -0.03, 0.0, -0.01, 0.02, -0.04, -0.01, 0.03, -0.02]
    for day, excess in enumerate(passes, start=11):
        journal.scored(
            excess,
            stock=excess + 0.02,
            benchmark=0.02,
            direction="pass",
            decision_session=date(2026, 9, day),
            horizon_ends=date(2026, 10, day),
        )

    performance = calculate_performance(journal.decisions, journal.scores)

    buy = performance.buy
    assert buy.calls == 10
    assert buy.hit_rate == 0.7  # 0.0 isn't above 0
    assert buy.average_excess_return == round(sum(buys) / 10, 6) == 0.015
    assert buy.average_edge == 0.015
    assert buy.average_return == 0.025
    assert buy.average_benchmark_return == 0.01
    assert buy.period == Period(start=date(2026, 9, 1), end=date(2026, 10, 10))
    assert buy.note is None

    pass_ = performance.pass_
    assert pass_.calls == 10
    assert pass_.hit_rate == 0.7  # 0.0 is right for a pass
    assert pass_.average_excess_return == round(sum(passes) / 10, 6) == -0.007
    assert pass_.average_edge == 0.007
    assert pass_.average_return == 0.013
    assert pass_.average_benchmark_return == 0.02
    assert pass_.period == Period(start=date(2026, 9, 11), end=date(2026, 10, 20))

    overall = performance.overall
    assert overall.calls == 20
    assert overall.hit_rate == 0.7
    assert overall.average_excess_return == 0.004
    assert overall.average_edge == 0.011
    assert overall.average_return == 0.019
    assert overall.average_benchmark_return == 0.015
    assert overall.period == Period(start=date(2026, 9, 1), end=date(2026, 10, 20))
    assert overall.note is None


def test_hit_rates_are_rounded_to_4_and_averages_to_6_decimals() -> None:
    # @AC2
    journal = Journal()
    for excess in [0.01] * 3 + [-0.0000003] * 10:
        journal.scored(excess)

    overall = calculate_performance(journal.decisions, journal.scores).overall

    assert overall.hit_rate == round(3 / 13, 4) == 0.2308
    assert overall.average_excess_return == 0.002307


def test_below_the_minimum_figures_are_null_with_a_note() -> None:
    # @AC4
    journal = Journal()
    for _ in range(7):
        journal.scored(0.01)

    performance = calculate_performance(journal.decisions, journal.scores)

    for group in (performance.overall, performance.buy):
        assert group.calls == 7
        assert group.hit_rate is None
        assert group.average_return is None
        assert group.average_benchmark_return is None
        assert group.average_excess_return is None
        assert group.average_edge is None
        assert group.note == "7 scored calls; needs 3 more (minimum 10)."
    assert performance.pass_.calls == 0
    assert performance.pass_.note == "0 scored calls; needs 10 more (minimum 10)."


def test_one_call_short_is_singular() -> None:
    # @AC4
    journal = Journal()
    journal.scored(0.01)
    for _ in range(9):
        journal.scored(-0.01, direction="pass")

    performance = calculate_performance(journal.decisions, journal.scores)

    assert performance.buy.note == "1 scored call; needs 9 more (minimum 10)."
    assert performance.pass_.note == "9 scored calls; needs 1 more (minimum 10)."
    assert performance.overall.note is None


def test_the_minimum_is_inclusive() -> None:
    # @AC4
    journal = Journal()
    for _ in range(MIN_CALLS):
        journal.scored(0.01)

    buy = calculate_performance(journal.decisions, journal.scores).buy

    assert MIN_CALLS == 10
    assert buy.hit_rate == 1.0
    assert buy.note is None


def test_no_scored_calls_is_zero_counts() -> None:
    # @AC7
    performance = calculate_performance([], {})

    for group in (performance.overall, performance.buy, performance.pass_):
        assert group.calls == 0
        assert group.hit_rate is None
        assert group.average_edge is None
        assert group.period is None


def test_open_calls_are_left_out() -> None:
    # @AC6
    journal = Journal()
    for _ in range(10):
        journal.scored(0.01, decision_session=date(2026, 9, 1))
    journal.open(decision_session=date(2026, 8, 1), horizon_ends=date(2026, 12, 1))

    overall = calculate_performance(journal.decisions, journal.scores).overall

    assert overall.calls == 10
    assert overall.average_excess_return == 0.01
    assert overall.period is not None
    assert overall.period.start == date(2026, 9, 1)


def test_the_period_is_kept_below_the_minimum() -> None:
    # @AC4: the period says what the count covers, not how the calls did.
    journal = Journal()
    journal.scored(0.01, decision_session=date(2026, 9, 1))

    assert calculate_performance(
        journal.decisions, journal.scores
    ).buy.period == Period(start=date(2026, 9, 1), end=date(2026, 11, 26))


def test_a_zero_edge_is_not_negative_zero() -> None:
    journal = Journal()
    for _ in range(10):
        journal.scored(0.0, direction="pass")

    pass_ = calculate_performance(journal.decisions, journal.scores).pass_

    assert pass_.average_edge == 0.0
    assert str(pass_.average_edge) == "0.0"


def test_every_signal_is_listed_in_order() -> None:
    # @AC3
    journal = Journal()
    journal.scored(0.01)

    signals = calculate_performance(journal.decisions, journal.scores).signals

    assert [s.signal for s in signals] == list(SIGNALS)
    assert len(signals) == 8
    by_name = {s.signal: s for s in signals}
    assert by_name["trend"].used.calls == 1  # decision_draft uses trend and rsi
    assert by_name["rsi"].used.calls == 1
    assert by_name["macd"].used.calls == 0
    assert by_name["macd"].not_used.calls == 1


def test_a_signal_is_used_with_either_stance() -> None:
    # @AC3
    journal = Journal()
    for _ in range(10):
        journal.scored(0.02, signals=[Signal(name="volume", stance="for")])
    for _ in range(10):
        journal.scored(
            0.01, direction="pass", signals=[Signal(name="volume", stance="against")]
        )
    for _ in range(10):
        journal.scored(-0.01, signals=[Signal(name="macd", stance="for")])

    by_name = {
        s.signal: s
        for s in calculate_performance(journal.decisions, journal.scores).signals
    }

    volume = by_name["volume"]
    assert volume.used.calls == 20
    assert volume.used.hit_rate == 0.5
    assert volume.used.average_excess_return == 0.015
    assert volume.used.average_edge == 0.005
    assert volume.used.note is None
    assert volume.not_used.calls == 10
    assert volume.not_used.hit_rate == 0.0
    assert volume.not_used.average_excess_return == -0.01
    assert volume.not_used.average_edge == -0.01


def test_each_side_of_a_signal_has_its_own_minimum() -> None:
    # @AC4
    journal = Journal()
    for _ in range(3):
        journal.scored(0.02, signals=[Signal(name="atr", stance="for")])
    for _ in range(12):
        journal.scored(0.01)

    atr = next(
        s
        for s in calculate_performance(journal.decisions, journal.scores).signals
        if s.signal == "atr"
    )

    assert atr.used.calls == 3
    assert atr.used.hit_rate is None
    assert atr.used.average_excess_return is None
    assert atr.used.average_edge is None
    assert atr.used.note == "3 scored calls; needs 7 more (minimum 10)."
    assert atr.not_used.calls == 12
    assert atr.not_used.average_edge == 0.01
    assert atr.not_used.note is None


def test_tickers_are_filled_whatever_the_count() -> None:
    # @AC5
    journal = Journal()
    journal.scored(-0.03, ticker="AAPL", currency="USD", exchange="XNYS")
    journal.scored(
        0.02, direction="pass", ticker="AAPL", currency="USD", exchange="XNYS"
    )
    journal.open(ticker="AAPL", currency="USD", exchange="XNYS")
    journal.open(ticker="AAPL", currency="USD", exchange="XNYS")

    tickers = calculate_performance(journal.decisions, journal.scores).tickers

    assert len(tickers) == 1
    aapl = tickers[0]
    assert aapl.ticker == "AAPL"
    assert aapl.calls == 2
    assert aapl.open == 2
    assert aapl.hit_rate == 0.0
    assert aapl.average_excess_return == -0.005
    assert aapl.average_edge == -0.025


def test_tickers_are_ordered_by_calls_then_ticker() -> None:
    # @AC5
    journal = Journal()
    journal.scored(0.01, ticker="SAP.AS")
    for _ in range(2):
        journal.scored(0.01, ticker="AAPL", currency="USD", exchange="XNYS")
    journal.scored(0.01, ticker="ASML.AS")

    tickers = calculate_performance(journal.decisions, journal.scores).tickers

    assert [t.ticker for t in tickers] == ["AAPL", "ASML.AS", "SAP.AS"]


def test_a_ticker_with_only_open_calls_is_not_listed() -> None:
    # @AC5
    journal = Journal()
    journal.scored(0.01)
    journal.open(ticker="AAPL", currency="USD", exchange="XNYS")

    tickers = calculate_performance(journal.decisions, journal.scores).tickers

    assert [t.ticker for t in tickers] == ["ASML.AS"]


def test_tickers_are_capped() -> None:
    # @AC5
    journal = Journal()
    for i in range(MAX_TICKERS + 5):
        journal.scored(0.01, ticker=f"T{i:03}.AS")
    journal.scored(0.01, ticker="T054.AS")

    performance = calculate_performance(journal.decisions, journal.scores)

    assert MAX_TICKERS == 50
    assert len(performance.tickers) == 50
    assert performance.tickers[0].ticker == "T054.AS"
    assert performance.more_tickers == 5


def test_no_scored_calls_has_no_tickers_and_empty_signals() -> None:
    # @AC7
    journal = Journal()
    journal.open()

    performance = calculate_performance(journal.decisions, journal.scores)

    assert performance.tickers == []
    assert performance.more_tickers == 0
    assert all(s.used.calls == s.not_used.calls == 0 for s in performance.signals)


# Plans: closed paper positions and their scores.


class Book:
    """Planned calls with their fills, exits and plan scores."""

    def __init__(self) -> None:
        self.decisions: list[Decision] = []
        self.positions = Positions({}, {}, {})
        self.plan_scores: dict[int, PlanScore] = {}

    def closed(
        self,
        realised_r: float,
        reason: ExitReason = "stop",
        excess: float = 0.01,
        trade: float = 0.03,
        benchmark: float = 0.02,
        slippage: float = 0.0,
        fill: date = date(2026, 10, 30),
        exit: date = date(2026, 11, 2),
        scored: bool = True,
        **changes: Any,
    ) -> Decision:
        draft = decision_draft(plan=buy_plan(), **changes)
        decision = Decision(id=len(self.decisions) + 1, **draft.model_dump())
        self.decisions.append(decision)
        self.positions.fills[decision.id] = fill_line(decision.id, session=fill)
        self.positions.exits[decision.id] = exit_line(
            decision.id, session=exit, reason=reason, slippage=slippage
        )
        if scored:
            self.plan_scores[decision.id] = plan_score_line(
                decision.id,
                return_=trade,
                benchmark=BenchmarkScore(
                    ticker="IWDA.AS",
                    start=Leg(session=fill, close=100.0),
                    end=Leg(session=exit, close=102.0),
                    return_=benchmark,
                ),
                excess_return=excess,
                hit=reason == "target",
                realised_r=realised_r,
                reason=reason,
                holding_sessions=3,
            )
        return decision

    def plans(self) -> PlanPerformance:
        return calculate_plans(self.decisions, self.positions, self.plan_scores)

    def gate(self, today: date = date(2027, 2, 1)) -> Gate:
        return calculate_gate(
            self.decisions, self.positions, self.plan_scores, today, 2.0
        )


def ten_plans() -> Book:
    book = Book()
    for _ in range(4):
        book.closed(3.0, "target", excess=0.05, trade=0.06, benchmark=0.01)
    for _ in range(4):
        book.closed(-1.0, "stop", excess=-0.03, trade=-0.02, benchmark=0.01)
    book.closed(-1.5, "stop", excess=-0.04, trade=-0.03, benchmark=0.01, slippage=5.8)
    book.closed(0.5, "horizon", excess=0.0, trade=0.01, benchmark=0.01)
    return book


def test_plans_report_the_means_of_the_stored_scores() -> None:
    # @AC5 @AC7
    plans = ten_plans().plans()

    plan = buy_plan()
    assert plans.plans == 10
    assert plans.open == 0
    assert plans.hits == 4
    assert plans.hit_rate == 0.4
    assert plans.average_return == round((0.24 - 0.08 - 0.03 + 0.01) / 10, 6)
    assert plans.average_benchmark_return == 0.01
    assert plans.average_excess_return == round((0.2 - 0.12 - 0.04) / 10, 6)
    assert plans.average_reward_to_risk == plan.reward_to_risk
    assert plans.average_realised_r == 0.7
    assert plans.expectancy == 0.7
    assert plans.average_stop_slippage == round(5.8 / 580.0, 6)
    assert plans.average_holding_sessions == 3.0
    assert plans.by_reason.target == ReasonGroup(plans=4, average_realised_r=3.0)
    assert plans.by_reason.stop == ReasonGroup(plans=5, average_realised_r=-1.1)
    assert plans.by_reason.horizon == ReasonGroup(plans=1, average_realised_r=0.5)
    assert plans.period == PlanPeriod(start=date(2026, 10, 30), end=date(2026, 11, 2))
    assert plans.note is None


def test_expectancy_is_the_mean_realised_r_with_gaps_and_horizon_exits() -> None:
    # @AC7
    book = Book()
    rs = [2.5, -1.0, -1.3, 0.2, -0.4, 2.5, -1.0, -1.0, 0.7, -1.21]
    for r in rs:
        book.closed(r, "horizon" if abs(r) < 1 else "stop")

    plans = book.plans()

    assert plans.expectancy == round(sum(rs) / len(rs), 6)
    assert plans.expectancy == plans.average_realised_r


def test_stop_slippage_is_null_without_a_gapped_stop() -> None:
    book = Book()
    for _ in range(10):
        book.closed(-1.0, "stop")

    assert book.plans().average_stop_slippage is None


def test_below_the_minimum_plan_figures_are_null_with_a_note() -> None:
    # @AC5
    book = Book()
    for _ in range(3):
        book.closed(3.0, "target")
    book.closed(-1.0, scored=False)

    plans = book.plans()

    assert plans.plans == 3
    assert plans.open == 1
    assert plans.hits == 3
    assert plans.by_reason.target == ReasonGroup(plans=3, average_realised_r=None)
    for figure in (
        plans.hit_rate,
        plans.average_return,
        plans.average_benchmark_return,
        plans.average_excess_return,
        plans.average_reward_to_risk,
        plans.average_realised_r,
        plans.expectancy,
        plans.average_stop_slippage,
        plans.average_holding_sessions,
    ):
        assert figure is None
    assert plans.period is not None
    assert plans.note == (
        f"3 scored plans; needs {MIN_CALLS - 3} more (minimum {MIN_CALLS}). "
        "1 closed plan isn't scored yet: call score_decisions."
    )


def test_no_plans_is_zero_counts_and_a_note() -> None:
    # @AC8
    plans = Book().plans()

    assert plans.plans == plans.open == plans.hits == 0
    assert plans.hit_rate is None
    assert plans.by_reason.stop == ReasonGroup(plans=0, average_realised_r=None)
    assert plans.period is None
    assert plans.note == "No plan has been scored yet."


def test_the_gate_reports_each_condition() -> None:
    # @AC6
    book = ten_plans()

    gate = book.gate()

    plan = buy_plan()
    assert gate.months == GateCondition(met=True, value=3, required=3)
    assert gate.closed_plans == GateCondition(met=False, value=10, required=30)
    assert gate.average_excess_return == GateCondition(
        met=True, value=0.004, required=0
    )
    assert gate.average_reward_to_risk == GateCondition(
        met=True, value=plan.reward_to_risk, required=2.0
    )
    assert gate.expectancy == GateCondition(met=True, value=0.7, required=0)
    assert gate.met is False


def test_months_are_whole_calendar_months_from_the_first_plan() -> None:
    # @AC6: the first plan was recorded on 2026-10-29.
    book = Book()
    book.closed(1.0)
    book.closed(1.0, recorded_at=datetime(2026, 12, 1, 9, 0, tzinfo=UTC))
    # A call without a plan, recorded earlier, doesn't start the count.
    early = decision_draft(recorded_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC))
    book.decisions.insert(0, Decision(id=99, **early.model_dump()))

    assert book.gate(date(2027, 1, 28)).months.value == 2
    assert book.gate(date(2027, 1, 29)).months.value == 3


def test_the_gate_is_met_when_every_condition_is() -> None:
    # @AC6
    book = Book()
    for _ in range(30):
        book.closed(0.5, "target", excess=0.01)

    gate = book.gate()

    assert gate.closed_plans == GateCondition(met=True, value=30, required=30)
    assert gate.met is True


def test_the_gate_shows_its_values_below_the_figures_minimum() -> None:
    # @AC6
    book = Book()
    book.closed(-1.0, excess=-0.02)

    gate = book.gate()

    assert gate.closed_plans.value == 1
    assert gate.average_excess_return == GateCondition(
        met=False, value=-0.02, required=0
    )
    assert gate.expectancy.value == -1.0


def test_without_plans_the_gate_is_not_met() -> None:
    # @AC8
    gate = Book().gate()

    assert gate.months == GateCondition(met=False, value=None, required=3)
    assert gate.closed_plans == GateCondition(met=False, value=0, required=30)
    for condition in (
        gate.average_excess_return,
        gate.average_reward_to_risk,
        gate.expectancy,
    ):
        assert condition.value is None
        assert condition.met is False
    assert gate.met is False
