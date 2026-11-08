"""
Integration tests for scoring closed plans (`score_decisions`'s plan lists)
and the `plans` and `gate` blocks of `get_performance` (#147), driven the way
Claude Code drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder holding the snapshot, its clock
pinned to the snapshot's moment and Yahoo out of reach (`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-147` and the acceptance
criteria it verifies. As the design says, plans are recorded for real and
backdated (the helpers of `test_settle_positions_stdio.py`), `settle_positions`
closes them on the snapshot's candles, and `score_decisions` scores them.
Expectations are worked out here from the fill and exit lines `settle_positions`
stored and IWDA.AS's cached closes, by the definitions in the design: return =
pnl / (fill value + buy costs), the benchmark from IWDA.AS's close on the last
XAMS session before the fill session to its close on the last XAMS session on
or before the exit session, realised R = pnl / planned risk, hit = target.
"""

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fartt.data import exchange_calendar
from tests.integration.test_settle_positions_stdio import (
    BENCHMARK,
    BOTH_IN_ONE_SESSION,
    FILL_AT_ENTRY_THEN_STOP,
    FILL_THEN_GAP_TARGET,
    FILL_THEN_TARGET,
    GAP_FILL_THEN_GAP_STOP,
    HELD_TO_HORIZON,
    STILL_OPEN,
    STOP_ON_FILL_SESSION,
    STOPPED_ON_NOV_3,
    USD_HELD_TO_HORIZON,
    Plan,
    _call,
    _candles,
    _journal,
    _seed,
    _server,
    _text,
)

TOLERANCE = 2e-6  # returns and R are stored to 6 decimals
MIN_PLANS = 10  # the design's minimum before the plans block shows figures
GATE_PLANS = 30
GATE_MONTHS = 3
MIN_REWARD_TO_RISK = 2.0  # risk.toml's min_reward_to_risk

# Eleven plans that close on the snapshot: three at the target (one at an open
# above it), six at the stop (two gapping through it), two at the horizon (one
# in USD).
ELEVEN = [
    FILL_THEN_TARGET,
    FILL_AT_ENTRY_THEN_STOP,
    GAP_FILL_THEN_GAP_STOP,
    HELD_TO_HORIZON,
    USD_HELD_TO_HORIZON,
    FILL_THEN_GAP_TARGET,
    BOTH_IN_ONE_SESSION,
    STOP_ON_FILL_SESSION,
    STOPPED_ON_NOV_3,
    GAP_FILL_THEN_GAP_STOP,
    FILL_THEN_TARGET,
]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _lines(assets_dir: Path, kind: str) -> dict[int, dict[str, Any]]:
    """The journal's lines of one kind, by decision id."""
    key = "id" if kind == "decision" else "decision_id"
    lines = [json.loads(line) for line in _journal(assets_dir).read_text().splitlines()]
    return {line[key]: line for line in lines if line["kind"] == kind}


def _xams_session(day: str, *, before: bool) -> str:
    """The last XAMS session before `day`, or on or before it."""
    calendar = exchange_calendar("XAMS")
    stamp = pd.Timestamp(day)
    if before:
        session = calendar.previous_session(
            calendar.date_to_session(stamp, direction="next")
        )
    else:
        session = calendar.date_to_session(stamp, direction="previous")
    return session.date().isoformat()


def _expected(assets_dir: Path) -> dict[int, dict[str, Any]]:
    """Each closed plan's score, worked out from its fill, exit and IWDA.AS's closes."""
    decisions = _lines(assets_dir, "decision")
    fills = _lines(assets_dir, "fill")
    exits = _lines(assets_dir, "exit")
    iwda = _candles(assets_dir, BENCHMARK)
    expected = {}
    for decision_id, exit_ in exits.items():
        fill = fills[decision_id]
        plan = decisions[decision_id]["plan"]
        start = _xams_session(fill["session"], before=True)
        end = _xams_session(exit_["session"], before=False)
        benchmark = round(iwda[end]["Close"] / iwda[start]["Close"] - 1, 6)
        return_ = exit_["pnl"] / (fill["value"] + fill["costs"]["total"])
        expected[decision_id] = {
            "ticker": decisions[decision_id]["ticker"],
            "fill_session": fill["session"],
            "exit_session": exit_["session"],
            "reason": exit_["reason"],
            "return": return_,
            "benchmark_start": start,
            "benchmark_end": end,
            "benchmark_return": benchmark,
            "excess_return": return_ - benchmark,
            "hit": exit_["reason"] == "target",
            "realised_r": exit_["pnl"] / plan["risk"],
            "reward_to_risk": plan["reward_to_risk"],
            "stop_slippage": (
                exit_["slippage"] / plan["stop"]
                if exit_["reason"] == "stop" and exit_["slippage"] > 0
                else None
            ),
        }
    return expected


def _check_scored(scored: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("ticker", "fill_session", "exit_session", "reason", "hit"):
        assert scored[key] == expected[key], key
    for key in ("return", "benchmark_return", "excess_return", "realised_r"):
        assert scored[key] == pytest.approx(expected[key], abs=TOLERANCE), key


def _check_stored(stored: dict[str, Any], expected: dict[str, Any]) -> None:
    assert stored["reason"] == expected["reason"]
    assert stored["hit"] is expected["hit"]
    assert stored["benchmark"]["ticker"] == BENCHMARK
    assert stored["benchmark"]["start"]["session"] == expected["benchmark_start"]
    assert stored["benchmark"]["end"]["session"] == expected["benchmark_end"]
    for key in ("return", "excess_return", "realised_r"):
        assert stored[key] == pytest.approx(expected[key], abs=TOLERANCE), key
    assert stored["benchmark"]["return"] == pytest.approx(
        expected["benchmark_return"], abs=TOLERANCE
    )


async def _settle_and_score(
    assets_dir: Path, plans: list[Plan]
) -> tuple[dict[str, Any], dict[str, Any]]:
    await _seed(assets_dir, plans)
    async with _server(assets_dir) as client:
        await _call(client, "settle_positions")
        run = await _call(client, "score_decisions")
        performance = await _call(client, "get_performance")
    return run, performance


async def test_should_score_a_plan_closed_at_its_target_as_a_hit_and_store_it(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC1 Should list a plan closed at its target under plans_scored
    with its return after costs, IWDA.AS's return from the close before the
    fill session to the exit session, the excess return, hit true and its
    realised R, and store the same in the journal.
    """
    run, _ = await _settle_and_score(seeded_assets, [FILL_THEN_TARGET])
    expected = _expected(seeded_assets)[1]

    [scored] = run["plans_scored"]
    assert scored["id"] == 1
    assert expected["reason"] == "target"
    # Filled on Monday 09-21: the benchmark starts at Friday 09-18's close.
    assert (expected["fill_session"], expected["benchmark_start"]) == (
        "2026-09-21",
        "2026-09-18",
    )
    _check_scored(scored, expected)
    assert scored["hit"] is True
    assert scored["return"] > 0 and scored["realised_r"] > 0
    assert run["plans_not_scored"] == []

    stored = _lines(seeded_assets, "plan_score")[1]
    _check_stored(stored, expected)
    # Fill and exit sessions both counted: 09-21 and 09-22.
    assert stored["holding_sessions"] == 2


async def test_should_score_stop_and_horizon_exits_as_misses_after_costs(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC2 Should score a plan closed at its stop (also through a
    gap) or at its horizon (also in USD) as a miss, with its realised loss or
    gain after costs and its realised R.
    """
    plans = [
        FILL_AT_ENTRY_THEN_STOP,
        GAP_FILL_THEN_GAP_STOP,
        HELD_TO_HORIZON,
        USD_HELD_TO_HORIZON,
    ]
    run, _ = await _settle_and_score(seeded_assets, plans)
    expected = _expected(seeded_assets)
    stored = _lines(seeded_assets, "plan_score")

    scored = {item["id"]: item for item in run["plans_scored"]}
    assert sorted(scored) == [1, 2, 3, 4]
    assert [expected[i]["reason"] for i in (1, 2, 3, 4)] == [
        "stop",
        "stop",
        "horizon",
        "horizon",
    ]
    for decision_id, item in scored.items():
        _check_scored(item, expected[decision_id])
        _check_stored(stored[decision_id], expected[decision_id])
        assert item["hit"] is False
    # Stopped out: a loss after costs, more than 1 R when the stop gapped.
    assert scored[1]["return"] < 0 and scored[1]["realised_r"] < 0
    assert scored[2]["realised_r"] < scored[1]["realised_r"]
    # Held to the horizon: a loss for ASML.AS, a gain for AAPL; still misses.
    assert scored[3]["realised_r"] < 0
    assert scored[4]["realised_r"] > 0
    # Oldest exit first.
    exits = [item["exit_session"] for item in run["plans_scored"]]
    assert exits == sorted(exits)


async def test_should_hold_a_plan_until_the_benchmark_close_is_cached_then_score_it(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC3 Should list a closed plan whose IWDA.AS close isn't
    available yet under plans_not_scored with the reason, write nothing for
    it, count it as open in get_performance, and score it in a later run.
    """
    await _seed(seeded_assets, [STOPPED_ON_NOV_3])
    async with _server(seeded_assets) as client:
        settled = await _call(client, "settle_positions")
    assert [c["session"] for c in settled["closed"]] == ["2026-11-03"]

    candles = seeded_assets / "stocks" / BENCHMARK / "candles.csv"
    full = candles.read_text()
    candles.write_text("".join(full.splitlines(keepends=True)[:-2]))  # to 11-02
    before = _journal(seeded_assets).read_bytes()
    async with _server(seeded_assets) as client:
        cut = await _call(client, "score_decisions")
        performance = await _call(client, "get_performance")
    assert _journal(seeded_assets).read_bytes() == before

    [blocked] = cut["plans_not_scored"]
    assert blocked["id"] == 1
    assert blocked["ticker"] == "ASML.AS"
    assert blocked["exit_session"] == "2026-11-03"
    assert BENCHMARK in blocked["reason"] and "2026-11-03" in blocked["reason"]
    assert "(the provider is unreachable)" in blocked["reason"]
    assert cut["plans_scored"] == []
    assert performance["plans"]["plans"] == 0
    assert performance["plans"]["open"] == 1

    candles.write_text(full)
    async with _server(seeded_assets) as client:
        later = await _call(client, "score_decisions")
    [scored] = later["plans_scored"]
    assert later["plans_not_scored"] == []
    _check_scored(scored, _expected(seeded_assets)[1])
    assert list(_lines(seeded_assets, "plan_score")) == [1]


async def test_should_write_nothing_new_when_scoring_runs_again(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC4 Should leave the journal file unchanged when
    score_decisions runs again, also after a restart, and count the plans as
    already scored.
    """
    plans = [FILL_THEN_TARGET, FILL_AT_ENTRY_THEN_STOP, HELD_TO_HORIZON]
    first, _ = await _settle_and_score(seeded_assets, plans)
    assert len(first["plans_scored"]) == 3
    after_first = _journal(seeded_assets).read_bytes()

    async with _server(seeded_assets) as client:
        second = await _call(client, "score_decisions")
        third = await _call(client, "score_decisions")

    assert _journal(seeded_assets).read_bytes() == after_first
    for run in (second, third):
        assert run["plans_scored"] == []
        assert run["plans_not_scored"] == []
        assert run["plans_already_scored"] == 3
    assert len(_lines(seeded_assets, "plan_score")) == 3


async def test_should_report_the_plans_figures_from_ten_scored_plans(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC5 @AC7 Should report, from ten scored plans on, the plans'
    hit rate, average return, benchmark return and excess return after costs,
    average planned reward-to-risk, average realised R, an expectancy equal
    to it (stop gaps and horizon exits included), the average stop slippage
    and the counts per exit reason, next to the calls' figures.
    """
    run, performance = await _settle_and_score(seeded_assets, ELEVEN)
    assert len(run["plans_scored"]) == len(ELEVEN)
    expected = list(_expected(seeded_assets).values())
    plans = performance["plans"]

    def mean(key: str) -> float:
        return statistics.mean(e[key] for e in expected)

    hits = sum(e["hit"] for e in expected)
    assert plans["plans"] == len(ELEVEN)
    assert plans["open"] == 0
    assert plans["hits"] == hits == 3
    assert plans["hit_rate"] == pytest.approx(hits / len(ELEVEN), abs=1e-4)
    assert plans["average_return"] == pytest.approx(mean("return"), abs=TOLERANCE)
    assert plans["average_benchmark_return"] == pytest.approx(
        mean("benchmark_return"), abs=TOLERANCE
    )
    assert plans["average_excess_return"] == pytest.approx(
        mean("excess_return"), abs=TOLERANCE
    )
    assert plans["average_reward_to_risk"] == pytest.approx(
        mean("reward_to_risk"), abs=TOLERANCE
    )
    assert plans["average_realised_r"] == pytest.approx(
        mean("realised_r"), abs=TOLERANCE
    )
    # AC7: measured, not derived: the mean realised R over every scored plan.
    stored_r = [s["realised_r"] for s in _lines(seeded_assets, "plan_score").values()]
    assert plans["expectancy"] == pytest.approx(
        statistics.mean(stored_r), abs=TOLERANCE
    )
    assert plans["expectancy"] == plans["average_realised_r"]
    slippages = [e["stop_slippage"] for e in expected if e["stop_slippage"]]
    assert len(slippages) == 2
    assert plans["average_stop_slippage"] == pytest.approx(
        statistics.mean(slippages), abs=TOLERANCE
    )
    for reason in ("stop", "target", "horizon"):
        group = [e["realised_r"] for e in expected if e["reason"] == reason]
        assert plans["by_reason"][reason]["plans"] == len(group)
        assert plans["by_reason"][reason]["average_realised_r"] == pytest.approx(
            statistics.mean(group), abs=TOLERANCE
        )
    assert plans["period"] == {"start": "2026-08-06", "end": "2026-11-03"}
    # Next to the calls' figures: the same decisions scored as calls.
    assert performance["scored"] == len(run["scored"])
    assert performance["overall"]["calls"] == len(run["scored"])


async def test_should_show_counts_only_below_ten_scored_plans(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC5 Should keep the plans' figures null below ten scored plans,
    with the counts and a note saying how many more are needed.
    """
    _, performance = await _settle_and_score(seeded_assets, ELEVEN[:9])
    plans = performance["plans"]

    assert plans["plans"] == 9
    assert plans["hits"] == 2
    for key in (
        "hit_rate",
        "average_return",
        "average_benchmark_return",
        "average_excess_return",
        "average_reward_to_risk",
        "average_realised_r",
        "expectancy",
        "average_stop_slippage",
    ):
        assert plans[key] is None, key
    assert sum(g["plans"] for g in plans["by_reason"].values()) == 9
    assert all(g["average_realised_r"] is None for g in plans["by_reason"].values())
    assert "1 more" in plans["note"]
    assert str(MIN_PLANS) in plans["note"]


async def test_should_report_each_gate_condition_with_its_value_and_requirement(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC6 @AC7 Should return the stage 1 gate with months since the
    first plan, scored plans, average excess return, average planned
    reward-to-risk and expectancy, each with its value, requirement and
    whether it's met, and the gate not met while any condition isn't.
    """
    plans = [USD_HELD_TO_HORIZON, FILL_THEN_TARGET, GAP_FILL_THEN_GAP_STOP]
    _, performance = await _settle_and_score(seeded_assets, plans)
    expected = list(_expected(seeded_assets).values())
    gate = performance["gate"]

    # The first plan was recorded 2026-08-05; the clock is 2026-11-05.
    assert gate["months"] == {"met": True, "value": 3, "required": GATE_MONTHS}
    assert gate["closed_plans"] == {"met": False, "value": 3, "required": GATE_PLANS}
    excess = statistics.mean(e["excess_return"] for e in expected)
    reward = statistics.mean(e["reward_to_risk"] for e in expected)
    expectancy = statistics.mean(e["realised_r"] for e in expected)
    for name, value, required in (
        ("average_excess_return", excess, 0),
        ("average_reward_to_risk", reward, MIN_REWARD_TO_RISK),
        ("expectancy", expectancy, 0),
    ):
        condition = gate[name]
        assert condition["value"] == pytest.approx(value, abs=TOLERANCE), name
        assert condition["required"] == required, name
        assert condition["met"] is (
            value >= required if name == "average_reward_to_risk" else value > 0
        ), name
    assert gate["met"] is False


async def test_should_meet_the_gate_with_thirty_winning_plans_over_three_months(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC6 Should report the gate as met when every condition is,
    and not met when the first plan is a day short of three whole months.
    """
    _, performance = await _settle_and_score(
        seeded_assets, [USD_HELD_TO_HORIZON] + [FILL_THEN_TARGET] * 29
    )
    gate = performance["gate"]
    for name in (
        "months",
        "closed_plans",
        "average_excess_return",
        "average_reward_to_risk",
        "expectancy",
    ):
        assert gate[name]["met"] is True, name
    assert gate["closed_plans"]["value"] == GATE_PLANS
    assert gate["met"] is True

    journal = _journal(seeded_assets)
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    lines[0]["recorded_at"] = "2026-08-06T09:00:00Z"
    journal.write_text("".join(json.dumps(line) + "\n" for line in lines))
    async with _server(seeded_assets) as client:
        short = (await _call(client, "get_performance"))["gate"]
    assert short["months"] == {"met": False, "value": 2, "required": GATE_MONTHS}
    assert short["met"] is False


async def test_should_score_and_report_calls_as_before_alongside_plans(
    seeded_assets: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """
    @issue-147 @AC8 Should score and report the calls the same with or without
    their plans, and return an empty plans block with a note and an unmet gate
    for a journal without plans.
    """
    plans = [FILL_THEN_TARGET, FILL_AT_ENTRY_THEN_STOP, HELD_TO_HORIZON, STILL_OPEN]
    seeded = await _seed(seeded_assets, plans)
    async with _server(seeded_assets) as client:
        await _call(client, "settle_positions")
        with_plans = await _call(client, "score_decisions")
        performance_with = await _call(client, "get_performance")

    calls_only = tmp_path_factory.mktemp("calls_only")
    for item in seeded_assets.iterdir():
        if item.name != "journal":
            (calls_only / item.name).symlink_to(item)
    (calls_only / "journal").mkdir()
    for entry in seeded:
        entry["plan"] = None
    _journal(calls_only).write_text("".join(json.dumps(e) + "\n" for e in seeded))
    async with _server(calls_only) as client:
        without_plans = await _call(client, "score_decisions")
        performance_without = await _call(client, "get_performance")

    assert len(with_plans["scored"]) == 3
    assert with_plans["scored"] == without_plans["scored"]
    assert with_plans["running"] == without_plans["running"] == 1
    for block in (
        "scored",
        "open",
        "due",
        "overall",
        "buy",
        "pass",
        "signals",
        "tickers",
    ):
        assert performance_with[block] == performance_without[block], block

    plans_block = performance_without["plans"]
    assert plans_block["plans"] == 0 and plans_block["open"] == 0
    assert plans_block["hits"] == 0
    assert plans_block["hit_rate"] is None and plans_block["expectancy"] is None
    assert plans_block["period"] is None
    assert plans_block["note"]
    assert without_plans["plans_scored"] == []
    assert without_plans["plans_already_scored"] == 0
    gate = performance_without["gate"]
    assert gate["met"] is False
    assert gate["closed_plans"] == {"met": False, "value": 0, "required": GATE_PLANS}
    assert gate["months"]["met"] is False and gate["months"]["value"] is None


async def test_should_refuse_a_score_for_a_plan_that_has_not_closed(
    seeded_assets: Path,
) -> None:
    """
    @issue-147 @AC4 Should refuse a journal holding a plan score for a position
    that hasn't closed, naming the file, and write nothing.
    """
    await _seed(seeded_assets, [STILL_OPEN])
    journal = _journal(seeded_assets)
    stray = {
        "kind": "plan_score",
        "decision_id": 1,
        "scored_at": "2026-11-05T12:00:00Z",
        "return": 0.01,
        "benchmark": {
            "ticker": BENCHMARK,
            "start": {"session": "2026-10-30", "close": 129.7, "rate": None},
            "end": {"session": "2026-11-03", "close": 129.0, "rate": None},
            "return": -0.005,
        },
        "excess_return": 0.015,
        "hit": True,
        "realised_r": 1.0,
        "reason": "target",
        "holding_sessions": 2,
    }
    journal.write_text(journal.read_text() + json.dumps(stray) + "\n")
    before = journal.read_bytes()

    async with _server(seeded_assets) as client:
        for tool in ("score_decisions", "get_performance"):
            result = await client.call_tool(tool, {})
            assert result.is_error, tool
            assert str(journal) in _text(result), tool

    assert journal.read_bytes() == before
