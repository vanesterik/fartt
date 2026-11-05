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

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from statistics import fmean
from typing import NamedTuple

from pydantic import BaseModel, Field

from fartt.journal import SIGNALS, Decision, Score

# How many scored calls a group needs before it reports a rate or average.
MIN_CALLS = 10
# A ticker entry is about 150 characters of JSON, so 50 stay well under the
# MCP output limit; the note says how many more there are.
MAX_TICKERS = 50

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
