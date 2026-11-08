"""
How the scored calls did: hit rate and average returns against the benchmark.

Every figure comes from the journal's stored scores as they are; nothing is
re-scored. A call's edge is its excess return for a `buy` and the negated
excess return for a `pass`, so a positive edge means right by that much in
any group, also one that mixes directions, where right calls of opposite
directions cancel in the average excess return.

These are calls, not trades: no costs, no position sizes. The closed paper
positions are reported apart, as trades: from their stored plan scores
(EUR, after costs, against IWDA.AS over each holding period), with the
stage 1 gate read from them. Pure functions, no I/O.

"""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from statistics import fmean
from typing import NamedTuple

from pydantic import BaseModel, Field

from fartt.journal import SIGNALS, Decision, Exit, Fill, PlanScore, Positions, Score

# How many scored calls a group needs before it reports a rate or average.
MIN_CALLS = 10
# A ticker entry is about 150 characters of JSON, so 50 stay well under the
# MCP output limit; the note says how many more there are.
MAX_TICKERS = 50

HIT_RATE_DECIMALS = 4
RETURN_DECIMALS = 6

# The stage 1 -> 2 gate (PRD section 6): paper trading for at least this
# many whole months and this many scored plans.
GATE_MONTHS = 3
GATE_CLOSED_PLANS = 30


class Period(BaseModel):
    """The span a group's calls cover."""

    start: date = Field(description="The earliest decision_session.")
    end: date = Field(description="The latest horizon_ends.")


class Group(BaseModel):
    """A group of scored calls and how they did."""

    calls: int = Field(description="How many scored calls are in the group.")
    hit_rate: float | None = Field(
        description=(
            f"The share scored right, 0-1; null below {MIN_CALLS} scored calls."
        )
    )
    average_return: float | None = Field(
        description="The mean of the stocks' returns with dividends, in EUR."
    )
    average_benchmark_return: float | None = Field(
        description="The mean of IWDA.AS's returns over the same horizons, in EUR."
    )
    average_excess_return: float | None = Field(
        description=(
            "The mean excess return (return minus the benchmark's). In a group "
            "mixing buy and pass, right calls of opposite directions cancel "
            "here: read average_edge instead."
        )
    )
    average_edge: float | None = Field(
        description=(
            "The mean edge: the excess return for a buy, its negation for a "
            "pass. Above 0 means the calls were right on average, by that much."
        )
    )
    period: Period | None = Field(
        description="The span the calls cover; null when there are none."
    )
    note: str | None = Field(
        default=None,
        description="How many more scored calls the figures need, if they're null.",
    )


class SignalSide(BaseModel):
    """The scored calls on one side of a signal: used, or not."""

    calls: int = Field(description="How many scored calls are on this side.")
    hit_rate: float | None = Field(
        description=f"The share scored right, 0-1; null below {MIN_CALLS} calls."
    )
    average_excess_return: float | None = Field(
        description=(
            "The mean excess return; buy and pass cancel here, so read average_edge."
        )
    )
    average_edge: float | None = Field(
        description="The mean edge; above 0 means right on average, by that much."
    )
    note: str | None = Field(
        default=None,
        description="How many more scored calls the figures need, if they're null.",
    )


class SignalPerformance(BaseModel):
    """A signal's step value: the calls that used it against those that didn't."""

    signal: str = Field(description="The signal's name, as record_decision takes it.")
    used: SignalSide = Field(
        description="The scored calls that named it, for or against."
    )
    not_used: SignalSide = Field(description="The scored calls that didn't.")


class TickerPerformance(BaseModel):
    """How the scored calls on one ticker did; weigh the figures by calls."""

    ticker: str = Field(description="The stock the calls are on.")
    calls: int = Field(
        description=(
            "How many of its calls are scored. The figures hold from a single "
            "call: weigh them by this count."
        )
    )
    open: int = Field(description="How many of its calls aren't scored yet.")
    hit_rate: float = Field(description="The share scored right, 0-1.")
    average_excess_return: float = Field(
        description="The mean excess return; buy and pass cancel here."
    )
    average_edge: float = Field(
        description="The mean edge; above 0 means right on average, by that much."
    )


class Performance(NamedTuple):
    overall: Group
    buy: Group
    pass_: Group
    signals: list[SignalPerformance]
    tickers: list[TickerPerformance]
    more_tickers: int  # How many tickers the cap left out.


def edge(decision: Decision, score: Score) -> float:
    """The excess return for a buy, its negation for a pass: above 0 is right."""
    return score.excess_return if decision.direction == "buy" else -score.excess_return


def calculate_performance(
    decisions: Sequence[Decision], scores: Mapping[int, Score]
) -> Performance:
    """The figures over the scored calls among `decisions`; open calls are left out."""
    scored = [(d, scores[d.id]) for d in decisions if d.id in scores]
    tickers = _tickers(decisions, scored)
    return Performance(
        overall=_group(scored),
        buy=_group([(d, s) for d, s in scored if d.direction == "buy"]),
        pass_=_group([(d, s) for d, s in scored if d.direction == "pass"]),
        signals=[_signal(name, scored) for name in SIGNALS],
        tickers=tickers[:MAX_TICKERS],
        more_tickers=max(len(tickers) - MAX_TICKERS, 0),
    )


def _signal(name: str, calls: Sequence[tuple[Decision, Score]]) -> SignalPerformance:
    used = [(d, s) for d, s in calls if any(sig.name == name for sig in d.signals)]
    not_used = [(d, s) for d, s in calls if all(sig.name != name for sig in d.signals)]
    return SignalPerformance(
        signal=name, used=_signal_side(used), not_used=_signal_side(not_used)
    )


def _signal_side(calls: Sequence[tuple[Decision, Score]]) -> SignalSide:
    if len(calls) < MIN_CALLS:
        return SignalSide(
            calls=len(calls),
            hit_rate=None,
            average_excess_return=None,
            average_edge=None,
            note=_short_note(len(calls)),
        )
    return SignalSide(
        calls=len(calls),
        hit_rate=_hit_rate(calls),
        average_excess_return=_mean([s.excess_return for _, s in calls]),
        average_edge=_mean([edge(d, s) for d, s in calls]),
    )


def _tickers(
    decisions: Sequence[Decision], scored: Sequence[tuple[Decision, Score]]
) -> list[TickerPerformance]:
    """Every ticker with a scored call, most scored calls first, then by ticker."""
    open_calls = Counter(d.ticker for d in decisions) - Counter(
        d.ticker for d, _ in scored
    )
    by_ticker: dict[str, list[tuple[Decision, Score]]] = {}
    for decision, score in scored:
        by_ticker.setdefault(decision.ticker, []).append((decision, score))
    return sorted(
        (
            TickerPerformance(
                ticker=ticker,
                calls=len(calls),
                open=open_calls[ticker],
                hit_rate=_hit_rate(calls),
                average_excess_return=_mean([s.excess_return for _, s in calls]),
                average_edge=_mean([edge(d, s) for d, s in calls]),
            )
            for ticker, calls in by_ticker.items()
        ),
        key=lambda t: (-t.calls, t.ticker),
    )


def _group(calls: Sequence[tuple[Decision, Score]]) -> Group:
    period = (
        Period(
            start=min(d.decision_session for d, _ in calls),
            end=max(d.horizon_ends for d, _ in calls),
        )
        if calls
        else None
    )
    if len(calls) < MIN_CALLS:
        return Group(
            calls=len(calls),
            hit_rate=None,
            average_return=None,
            average_benchmark_return=None,
            average_excess_return=None,
            average_edge=None,
            period=period,
            note=_short_note(len(calls)),
        )
    return Group(
        calls=len(calls),
        hit_rate=_hit_rate(calls),
        average_return=_mean([s.return_ for _, s in calls]),
        average_benchmark_return=_mean([s.benchmark.return_ for _, s in calls]),
        average_excess_return=_mean([s.excess_return for _, s in calls]),
        average_edge=_mean([edge(d, s) for d, s in calls]),
        period=period,
    )


def _hit_rate(calls: Sequence[tuple[Decision, Score]]) -> float:
    return round(sum(s.right for _, s in calls) / len(calls), HIT_RATE_DECIMALS)


def _mean(values: Sequence[float]) -> float:
    # + 0.0 turns -0.0 (a mean of negated zeros) into 0.0.
    return round(fmean(values), RETURN_DECIMALS) + 0.0


def _short_note(calls: int) -> str:
    return (
        f"{calls} scored {'call' if calls == 1 else 'calls'}; needs "
        f"{MIN_CALLS - calls} more (minimum {MIN_CALLS})."
    )


class ReasonGroup(BaseModel):
    """The scored plans that closed for one reason."""

    plans: int = Field(description="How many scored plans closed this way.")
    average_realised_r: float | None = Field(
        description=f"Their mean realised R; null below {MIN_CALLS} scored plans."
    )


class ByReason(BaseModel):
    stop: ReasonGroup = Field(description="Closed at the stop, gapped or not.")
    target: ReasonGroup = Field(description="Closed at the target: the hits.")
    horizon: ReasonGroup = Field(description="Closed at the call's horizon_ends.")


class PlanPeriod(BaseModel):
    """The span the scored plans cover."""

    start: date = Field(description="The earliest fill session.")
    end: date = Field(description="The latest exit session.")


class PlanPerformance(BaseModel):
    """How the closed and scored paper positions did, as trades after costs."""

    plans: int = Field(description="How many closed plans are scored.")
    open: int = Field(
        description="How many closed plans aren't scored yet: score_decisions scores them."
    )
    hits: int = Field(description="How many scored plans closed at their target.")
    hit_rate: float | None = Field(
        description=(
            "hits / plans, 0-1: the share that reached the target before the "
            f"stop; null below {MIN_CALLS} scored plans, as every figure here."
        )
    )
    average_return: float | None = Field(
        description="The mean return, in EUR after all costs, dividends included."
    )
    average_benchmark_return: float | None = Field(
        description="The mean of IWDA.AS's returns over the same holding periods."
    )
    average_excess_return: float | None = Field(
        description="The mean excess return after costs: the gate's primary metric."
    )
    average_reward_to_risk: float | None = Field(
        description="The mean planned reward-to-risk, after costs."
    )
    average_realised_r: float | None = Field(
        description="The mean realised R: P&L over the plan's risk."
    )
    expectancy: float | None = Field(
        description=(
            "In R: the mean realised R, measured, so gaps through stops and "
            "horizon exits count as they happened."
        )
    )
    average_stop_slippage: float | None = Field(
        description=(
            "The mean of slippage / stop over the stop exits that gapped "
            "through, as a fraction; null without one."
        )
    )
    average_holding_sessions: float | None = Field(
        description="The mean sessions held, fill and exit sessions included."
    )
    by_reason: ByReason = Field(description="The scored plans per exit reason.")
    period: PlanPeriod | None = Field(
        description="The span the scored plans cover; null when there are none."
    )
    note: str | None = Field(
        default=None,
        description="Why the figures are null, or plans wait for scoring, if so.",
    )


class GateCondition(BaseModel):
    """One condition of the stage 1 gate."""

    met: bool = Field(description="Whether the value meets what's required.")
    value: int | float | None = Field(
        description="The current value; null when there's nothing to measure yet."
    )
    required: int | float = Field(description="The threshold it's measured against.")


class Gate(BaseModel):
    """The gate from stage 1 (paper trading) to stage 2, PRD section 6."""

    months: GateCondition = Field(
        description=(
            "Whole calendar months from the first plan's recording to today; "
            f"met at {GATE_MONTHS} or more."
        )
    )
    closed_plans: GateCondition = Field(
        description=f"Closed and scored plans; met at {GATE_CLOSED_PLANS} or more."
    )
    average_excess_return: GateCondition = Field(
        description="The plans' mean excess return after costs; met above 0."
    )
    average_reward_to_risk: GateCondition = Field(
        description=(
            "The plans' mean planned reward-to-risk after costs; met at "
            "min_reward_to_risk (risk.toml) or more."
        )
    )
    expectancy: GateCondition = Field(
        description="The plans' mean realised R; met above 0."
    )
    met: bool = Field(description="Whether every condition is met.")


class _Scored(NamedTuple):
    decision: Decision
    fill: Fill
    exit: Exit
    score: PlanScore


def _scored_plans(
    decisions: Sequence[Decision],
    positions: Positions,
    plan_scores: Mapping[int, PlanScore],
) -> list[_Scored]:
    # The journal stores a plan score only after its call's fill and exit.
    return [
        _Scored(d, positions.fills[d.id], positions.exits[d.id], plan_scores[d.id])
        for d in decisions
        if d.id in plan_scores
    ]


def _planned_reward_to_risk(plans: Sequence[_Scored]) -> list[float]:
    return [p.decision.plan.reward_to_risk for p in plans if p.decision.plan]


def _plans_note(scored: int, waiting: int) -> str | None:
    notes: list[str] = []
    if not scored:
        notes.append("No plan has been scored yet.")
    elif scored < MIN_CALLS:
        notes.append(
            f"{scored} scored {'plan' if scored == 1 else 'plans'}; needs "
            f"{MIN_CALLS - scored} more (minimum {MIN_CALLS})."
        )
    if waiting:
        notes.append(
            "1 closed plan isn't scored yet: call score_decisions."
            if waiting == 1
            else f"{waiting} closed plans aren't scored yet: call score_decisions."
        )
    return " ".join(notes) or None


def calculate_plans(
    decisions: Sequence[Decision],
    positions: Positions,
    plan_scores: Mapping[int, PlanScore],
) -> PlanPerformance:
    """The figures over the scored plans; closed plans not scored are counted."""
    plans = _scored_plans(decisions, positions, plan_scores)
    count = len(plans)
    waiting = sum(1 for d in positions.exits if d not in plan_scores)
    full = count >= MIN_CALLS

    def mean(values: Sequence[float]) -> float | None:
        return _mean(values) if full and values else None

    def reason(name: str) -> ReasonGroup:
        rs = [p.score.realised_r for p in plans if p.exit.reason == name]
        return ReasonGroup(plans=len(rs), average_realised_r=mean(rs))

    realised = mean([p.score.realised_r for p in plans])
    hits = sum(p.score.hit for p in plans)
    gapped = [
        p.exit.slippage / p.decision.plan.stop
        for p in plans
        if p.decision.plan and p.exit.reason == "stop" and p.exit.slippage > 0
    ]
    return PlanPerformance(
        plans=count,
        open=waiting,
        hits=hits,
        hit_rate=round(hits / count, HIT_RATE_DECIMALS) if full else None,
        average_return=mean([p.score.return_ for p in plans]),
        average_benchmark_return=mean([p.score.benchmark.return_ for p in plans]),
        average_excess_return=mean([p.score.excess_return for p in plans]),
        average_reward_to_risk=mean(_planned_reward_to_risk(plans)),
        average_realised_r=realised,
        expectancy=realised,
        average_stop_slippage=mean(gapped),
        average_holding_sessions=mean([p.score.holding_sessions for p in plans]),
        by_reason=ByReason(
            stop=reason("stop"), target=reason("target"), horizon=reason("horizon")
        ),
        period=PlanPeriod(
            start=min(p.fill.session for p in plans),
            end=max(p.exit.session for p in plans),
        )
        if plans
        else None,
        note=_plans_note(count, waiting),
    )


def calculate_gate(
    decisions: Sequence[Decision],
    positions: Positions,
    plan_scores: Mapping[int, PlanScore],
    today: date,
    min_reward_to_risk: float,
) -> Gate:
    """
    The stage 1 gate on `today` (UTC). Its values show from the first
    scored plan, whatever `MIN_CALLS` says; months from the first plan
    recorded.

    """
    plans = _scored_plans(decisions, positions, plan_scores)
    recorded = [d.recorded_at.date() for d in decisions if d.plan is not None]
    months = _whole_months(min(recorded), today) if recorded else None

    def average(values: Sequence[float]) -> float | None:
        return _mean(values) if values else None

    excess = average([p.score.excess_return for p in plans])
    reward_to_risk = average(_planned_reward_to_risk(plans))
    expectancy = average([p.score.realised_r for p in plans])
    conditions = {
        "months": GateCondition(
            met=months is not None and months >= GATE_MONTHS,
            value=months,
            required=GATE_MONTHS,
        ),
        "closed_plans": GateCondition(
            met=len(plans) >= GATE_CLOSED_PLANS,
            value=len(plans),
            required=GATE_CLOSED_PLANS,
        ),
        "average_excess_return": GateCondition(
            met=excess is not None and excess > 0, value=excess, required=0
        ),
        "average_reward_to_risk": GateCondition(
            met=reward_to_risk is not None and reward_to_risk >= min_reward_to_risk,
            value=reward_to_risk,
            required=min_reward_to_risk,
        ),
        "expectancy": GateCondition(
            met=expectancy is not None and expectancy > 0, value=expectancy, required=0
        ),
    }
    return Gate(**conditions, met=all(c.met for c in conditions.values()))


def _whole_months(start: date, end: date) -> int:
    """Whole calendar months from `start` to `end`: 29 Oct to 28 Jan is 2."""
    months = (end.year - start.year) * 12 + end.month - start.month
    return max(months - (end.day < start.day), 0)
