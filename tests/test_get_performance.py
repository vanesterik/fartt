import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from fartt.journal import SIGNALS, Signal
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from tests.fakes import (
    buy_plan,
    decision_draft,
    decision_score,
    exit_line,
    fill_line,
    plan_score_line,
)
from tests.test_server import (
    cache_for,
    call,
    journal_of,
    provider,
    seed,
    sessions,
)

# decision_draft's horizon ends on 2026-11-26, after AFTER_CLOSE (2026-10-29):
# running. These calls' horizon ended on 2026-09-29: due.
DUE = {"decision_session": date(2026, 9, 1), "horizon_ends": date(2026, 9, 29)}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def a_cache(tmp_path: Path) -> StockCache:
    return cache_for(provider(sessions("XAMS", 3)), tmp_path)


def scored(cache: StockCache, excess: float, **changes: Any) -> None:
    (decision,) = seed(cache, decision_draft(**(DUE | changes)))
    right = excess > 0 if decision.direction == "buy" else excess <= 0
    journal_of(cache).append_score(
        decision_score(decision.id, excess_return=excess, right=right)
    )


async def get_performance(cache: StockCache) -> Any:
    return await call(cache, {}, "get_performance")


@pytest.mark.anyio
async def test_get_performance_reports_the_scored_calls(tmp_path: Path) -> None:
    # @AC1 @AC2 @AC3 @AC5
    cache = a_cache(tmp_path)
    for _ in range(6):
        scored(cache, 0.02)
    for _ in range(4):
        scored(cache, 0.01, direction="pass")

    result = await get_performance(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["min_calls"] == 10
    assert content["scored"] == 10
    assert content["open"] == 0
    assert content["due"] == 0
    assert content["note"] is None
    assert content["overall"] == {
        "calls": 10,
        "hit_rate": 0.6,
        "average_return": 0.045069,
        "average_benchmark_return": 0.02716,
        "average_excess_return": 0.016,
        "average_edge": 0.008,
        "period": {"start": "2026-09-01", "end": "2026-09-29"},
        "note": None,
    }
    assert content["buy"]["calls"] == 6
    assert content["buy"]["note"] == "6 scored calls; needs 4 more (minimum 10)."
    assert content["pass"]["calls"] == 4
    assert content["pass"]["hit_rate"] is None
    assert [s["signal"] for s in content["signals"]] == list(SIGNALS)
    trend = content["signals"][0]
    assert trend["used"]["calls"] == 10
    assert trend["used"]["average_edge"] == 0.008
    assert trend["not_used"]["calls"] == 0
    assert content["tickers"] == [
        {
            "ticker": "ASML.AS",
            "calls": 10,
            "open": 0,
            "hit_rate": 0.6,
            "average_excess_return": 0.016,
            "average_edge": 0.008,
        }
    ]


@pytest.mark.anyio
async def test_open_and_due_calls_are_counted_not_figured(tmp_path: Path) -> None:
    # @AC6
    cache = a_cache(tmp_path)
    for _ in range(10):
        scored(cache, 0.02)
    seed(cache, decision_draft(**DUE), decision_draft(**DUE), decision_draft())

    content = (await get_performance(cache)).structured_content

    assert content["scored"] == 10
    assert content["open"] == 3
    assert content["due"] == 2
    assert content["overall"]["calls"] == 10
    assert content["overall"]["average_excess_return"] == 0.02
    assert content["tickers"][0]["open"] == 3
    assert content["note"] == "2 calls are due: call score_decisions to score them."


@pytest.mark.anyio
async def test_one_due_call_is_singular(tmp_path: Path) -> None:
    # @AC6
    cache = a_cache(tmp_path)
    scored(cache, 0.02)
    seed(cache, decision_draft(**DUE))

    content = (await get_performance(cache)).structured_content

    assert content["note"] == "1 call is due: call score_decisions to score it."


@pytest.mark.anyio
async def test_an_empty_journal_is_zero_counts_and_a_note(tmp_path: Path) -> None:
    # @AC7: no journal folder at all.
    cache = a_cache(tmp_path)

    result = await get_performance(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["scored"] == content["open"] == content["due"] == 0
    for group in ("overall", "buy", "pass"):
        assert content[group]["calls"] == 0
        assert content[group]["hit_rate"] is None
        assert content[group]["average_edge"] is None
        assert content[group]["period"] is None
    assert content["tickers"] == []
    assert content["note"] == "No call has been scored yet."
    assert not journal_of(cache).directory.exists()


@pytest.mark.anyio
async def test_only_open_calls_count_as_open(tmp_path: Path) -> None:
    # @AC6 @AC7
    cache = a_cache(tmp_path)
    seed(cache, decision_draft(**DUE), decision_draft())

    content = (await get_performance(cache)).structured_content

    assert content["scored"] == 0
    assert content["open"] == 2
    assert content["due"] == 1
    assert content["overall"]["calls"] == 0
    assert content["tickers"] == []
    assert content["note"] == (
        "No call has been scored yet. 1 call is due: call score_decisions to score it."
    )


@pytest.mark.anyio
async def test_the_basis_says_calls_not_trades(tmp_path: Path) -> None:
    # @AC8
    content = (await get_performance(a_cache(tmp_path))).structured_content

    basis = content["basis"]
    assert "calls, not trades" in basis
    assert "EUR" in basis
    assert "IWDA.AS" in basis
    assert "costs" in basis
    assert "position sizes" in basis


@pytest.mark.anyio
async def test_get_performance_never_calls_the_provider_or_writes(
    tmp_path: Path,
) -> None:
    # @AC9
    fake = provider(sessions("XAMS", 3))
    fake.reachable = False
    cache = cache_for(fake, tmp_path)
    scored(cache, 0.02)
    before = journal_of(cache).path.read_bytes()

    result = await get_performance(cache)

    assert not result.is_error, result.content
    assert result.structured_content["scored"] == 1
    assert fake.history_calls == []
    assert journal_of(cache).path.read_bytes() == before
    assert not (tmp_path / "stocks").exists()


@pytest.mark.anyio
async def test_a_damaged_journal_is_a_tool_error_for_get_performance(
    tmp_path: Path,
) -> None:
    # @AC9
    cache = a_cache(tmp_path)
    scored(cache, 0.02)
    with journal_of(cache).path.open("ab") as file:
        file.write(b"not json\n")
    before = journal_of(cache).path.read_bytes()

    result = await get_performance(cache)

    assert result.is_error
    text = result.content[0].text
    assert str(journal_of(cache).path) in text
    assert "line 3" in text
    assert "operator" in text
    assert journal_of(cache).path.read_bytes() == before


@pytest.mark.anyio
async def test_the_ticker_cap_is_in_the_note(tmp_path: Path) -> None:
    # @AC5
    cache = a_cache(tmp_path)
    for i in range(52):
        scored(cache, 0.01, ticker=f"T{i:03}.AS")

    content = (await get_performance(cache)).structured_content

    assert len(content["tickers"]) == 50
    assert content["note"] == (
        "Listing the 50 tickers with the most scored calls; 2 more have scored "
        "calls too."
    )


@pytest.mark.anyio
async def test_the_largest_result_fits(tmp_path: Path) -> None:
    # A full ticker list and every signal on both sides stay well under the
    # MCP output limit (about 25k tokens).
    cache = a_cache(tmp_path)
    every_signal = [Signal(name=name, stance="for") for name in SIGNALS]
    for i in range(60):
        scored(
            cache,
            0.0123456,
            ticker=f"LONGTICKER{i:02}.AS",
            signals=every_signal if i % 2 else every_signal[:1],
        )

    content = (await get_performance(cache)).structured_content

    assert content["overall"]["hit_rate"] is not None
    assert len(json.dumps(content)) < 20_000


def test_the_instructions_mention_get_performance() -> None:
    assert "get_performance" in server_module.INSTRUCTIONS
    assert "calls, not trades" in server_module.INSTRUCTIONS


def closed_plan(cache: StockCache, scored: bool = True, **score: Any) -> None:
    journal = journal_of(cache)
    (decision,) = seed(cache, decision_draft(plan=buy_plan()))
    journal.append_fill(fill_line(decision.id))
    journal.append_exit(exit_line(decision.id))
    if scored:
        journal.append_plan_score(plan_score_line(decision.id, **score))


@pytest.mark.anyio
async def test_get_performance_reports_the_plans_and_the_gate(tmp_path: Path) -> None:
    # @AC5 @AC6 @AC7: ten plans stopped out (realised R -0.988), next to calls.
    cache = a_cache(tmp_path)
    for _ in range(10):
        scored(cache, 0.02)
    for _ in range(10):
        closed_plan(cache)

    content = (await get_performance(cache)).structured_content

    assert content["overall"]["calls"] == 10
    plans = content["plans"]
    assert plans["plans"] == 10
    assert plans["open"] == 0
    assert plans["hits"] == 0
    assert plans["hit_rate"] == 0.0
    assert plans["average_return"] == -0.03924
    assert plans["average_benchmark_return"] == 0.007407
    assert plans["average_excess_return"] == -0.046647
    assert plans["average_reward_to_risk"] == buy_plan().reward_to_risk
    assert plans["average_realised_r"] == plans["expectancy"] == -0.988
    assert plans["average_stop_slippage"] is None
    assert plans["average_holding_sessions"] == 2.0
    assert plans["by_reason"]["stop"] == {"plans": 10, "average_realised_r": -0.988}
    assert plans["by_reason"]["target"] == {"plans": 0, "average_realised_r": None}
    assert plans["period"] == {"start": "2026-10-30", "end": "2026-11-02"}
    assert plans["note"] is None
    gate = content["gate"]
    # The plans were recorded on 2026-10-29; the clock says 2026-10-29.
    assert gate["months"] == {"met": False, "value": 0, "required": 3}
    assert gate["closed_plans"] == {"met": False, "value": 10, "required": 30}
    assert gate["average_excess_return"] == {
        "met": False,
        "value": -0.046647,
        "required": 0,
    }
    assert gate["average_reward_to_risk"] == {
        "met": True,
        "value": buy_plan().reward_to_risk,
        "required": 2.0,
    }
    assert gate["expectancy"] == {"met": False, "value": -0.988, "required": 0}
    assert gate["met"] is False


@pytest.mark.anyio
async def test_below_ten_plans_the_figures_wait(tmp_path: Path) -> None:
    # @AC5
    cache = a_cache(tmp_path)
    for _ in range(3):
        closed_plan(cache)
    closed_plan(cache, scored=False)

    content = (await get_performance(cache)).structured_content

    plans = content["plans"]
    assert plans["plans"] == 3
    assert plans["open"] == 1
    assert plans["hit_rate"] is None
    assert plans["expectancy"] is None
    assert plans["note"] == (
        "3 scored plans; needs 7 more (minimum 10). "
        "1 closed plan isn't scored yet: call score_decisions."
    )
    assert content["gate"]["expectancy"]["value"] == -0.988


@pytest.mark.anyio
async def test_without_plans_the_block_is_empty_and_the_gate_not_met(
    tmp_path: Path,
) -> None:
    # @AC8
    cache = a_cache(tmp_path)
    scored(cache, 0.02)

    content = (await get_performance(cache)).structured_content

    assert content["overall"]["calls"] == 1
    plans = content["plans"]
    assert plans["plans"] == plans["open"] == plans["hits"] == 0
    assert plans["average_return"] is None
    assert plans["period"] is None
    assert plans["note"] == "No plan has been scored yet."
    assert content["gate"]["met"] is False
    assert content["gate"]["months"]["value"] is None
    assert content["gate"]["closed_plans"]["value"] == 0


@pytest.mark.anyio
async def test_the_basis_says_plans_are_trades_after_costs(tmp_path: Path) -> None:
    content = (await get_performance(a_cache(tmp_path))).structured_content

    basis = content["basis"]
    assert "calls, not trades" in basis
    assert "plans" in basis
    assert "after costs" in basis
    assert "holding period" in basis
