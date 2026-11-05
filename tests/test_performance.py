from datetime import date
from typing import Any

from fartt.journal import SIGNALS, BenchmarkScore, Decision, Leg, Score, Signal
from fartt.performance import (
    MAX_TICKERS,
    MIN_CALLS,
    Period,
    calculate_performance,
)
from tests.fakes import decision_draft, decision_score


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
