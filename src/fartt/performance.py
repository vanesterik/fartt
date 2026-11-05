"""
How the scored calls did: hit rate and average returns against the benchmark.

Every figure comes from the journal's stored scores as they are; nothing is
re-scored. A call's edge is its excess return for a `buy` and the negated
excess return for a `pass`, so a positive edge means right by that much in
any group, also one that mixes directions, where right calls of opposite
directions cancel in the average excess return.

These are calls, not trades: no costs, no position sizes. Pure functions,
no I/O.

"""

from collections.abc import Mapping, Sequence
from datetime import date
from statistics import fmean
from typing import NamedTuple

from pydantic import BaseModel, Field

from fartt.journal import Decision, Score

# How many scored calls a group needs before it reports a rate or average.
MIN_CALLS = 10

HIT_RATE_DECIMALS = 4
RETURN_DECIMALS = 6


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


class Performance(NamedTuple):
    overall: Group
    buy: Group
    pass_: Group


def edge(decision: Decision, score: Score) -> float:
    """The excess return for a buy, its negation for a pass: above 0 is right."""
    return score.excess_return if decision.direction == "buy" else -score.excess_return


def calculate_performance(
    decisions: Sequence[Decision], scores: Mapping[int, Score]
) -> Performance:
    """The figures over the scored calls among `decisions`; open calls are left out."""
    scored = [(d, scores[d.id]) for d in decisions if d.id in scores]
    return Performance(
        overall=_group(scored),
        buy=_group([(d, s) for d, s in scored if d.direction == "buy"]),
        pass_=_group([(d, s) for d, s in scored if d.direction == "pass"]),
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
