import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pytest

from fartt.data import Candle, History, Split, session_after
from fartt.journal import BenchmarkClose, DecisionDraft
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from fartt.trade_plan import leg_costs
from tests.fakes import RISK_CONFIG, FakeProvider, buy_plan, decision_draft
from tests.test_server import (
    AAPL,
    AFTER_CLOSE,
    ASML,
    IWDA,
    cache_for,
    call,
    journal_of,
    seed,
    sessions,
)

DAYS = sessions("XAMS", 70)  # Up to 2026-10-29, AFTER_CLOSE's session
QUIET = (102.0, 103.0, 101.0, 102.0)  # Above the entry, between stop and target
ENTRY, STOP, TARGET = 100.0, 95.0, 115.0


def asml(sessions: dict[date, tuple[float, float, float, float]]) -> History:
    """ASML.AS on quiet sessions, but `sessions` (open, high, low, close)."""
    return History(
        [Candle(day, *sessions.get(day, QUIET), 1000.0) for day in DAYS], [], []
    )


def fake(
    sessions: dict[date, tuple[float, float, float, float]] | None = None,
) -> FakeProvider:
    return FakeProvider(
        histories={
            "ASML.AS": asml(sessions or {}),
            "IWDA.AS": History([Candle(d, *QUIET, 0.0) for d in DAYS], [], []),
        },
        listings={"ASML.AS": ASML, "AAPL": AAPL, "IWDA.AS": IWDA},
        now=AFTER_CLOSE,
    )


def after(based_on: date, n: int) -> date:
    return session_after("XAMS", based_on, n)


def planned_call(based_on: date, **changes: Any) -> DecisionDraft:
    """A buy on ASML.AS made on `based_on`'s close: entry 100, stop 95, target 115."""
    plan = buy_plan(
        based_on=based_on, last_close=102.0, atr=3.0, entry=ENTRY, stop=STOP,
        target=TARGET, valid_for=5,
    )  # fmt: skip
    fields: dict[str, Any] = {
        "decision_session": based_on,
        "close": 102.0,
        "horizon_ends": after(based_on, 20),
        "benchmark": BenchmarkClose(
            ticker="IWDA.AS", currency="EUR", session=based_on, close=102.0
        ),
        "plan": plan,
    }
    return decision_draft(**(fields | changes))


async def settle_positions(cache: StockCache) -> Any:
    return await call(cache, {}, "settle_positions")


def lines(cache: StockCache, kind: str) -> list[dict[str, Any]]:
    path = journal_of(cache).path
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    return [entry for entry in entries if entry["kind"] == kind]


B = DAYS[-15]  # The plan's session; its horizon ends after AFTER_CLOSE


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_a_reached_entry_is_filled_and_reported(tmp_path: Path) -> None:
    # @AC1
    cache = cache_for(fake({after(B, 2): (101.0, 102.0, 99.5, 101.0)}), tmp_path)
    (call_,) = seed(cache, planned_call(B))
    assert call_.plan is not None
    shares = call_.plan.shares

    result = await settle_positions(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    costs = leg_costs(
        shares * ENTRY, None, "EUR", RISK_CONFIG.costs.exchanges["XAMS"],
        RISK_CONFIG.costs, "buy",
    )  # fmt: skip
    assert content["filled"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "session": after(B, 2).isoformat(),
            "price": ENTRY,
            "at_open": False,
            "shares": shares,
            "rate": None,
            "value": round(shares * ENTRY, 2),
            "costs": costs.model_dump(),
        }
    ]
    assert content["open"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "since": after(B, 2).isoformat(),
            "stop": STOP,
            "target": TARGET,
            "horizon_ends": after(B, 20).isoformat(),
        }
    ]
    assert (content["expired"], content["closed"], content["waiting"]) == ([], [], [])
    assert content["not_settled"] == []
    assert content["note"] is None
    (fill,) = lines(cache, "fill")
    assert (fill["decision_id"], fill["price"]) == (1, ENTRY)


@pytest.mark.anyio
async def test_an_unreached_plan_waits_then_expires(tmp_path: Path) -> None:
    # @AC3
    recent = DAYS[-3]
    cache = cache_for(fake(), tmp_path)
    seed(cache, planned_call(B), planned_call(recent))

    content = (await settle_positions(cache)).structured_content

    assert content["expired"] == [
        {"id": 1, "ticker": "ASML.AS", "session": after(B, 5).isoformat()}
    ]
    assert content["waiting"] == [
        {
            "id": 2,
            "ticker": "ASML.AS",
            "entry": ENTRY,
            "until": after(recent, 5).isoformat(),
        }
    ]
    assert [line["decision_id"] for line in lines(cache, "expiry")] == [1]


@pytest.mark.anyio
async def test_a_closed_position_is_reported(tmp_path: Path) -> None:
    # @AC4
    cache = cache_for(
        fake(
            {
                after(B, 2): (101.0, 102.0, 99.5, 101.0),
                after(B, 4): (98.0, 99.0, 94.0, 96.0),
            }
        ),
        tmp_path,
    )
    seed(cache, planned_call(B))

    content = (await settle_positions(cache)).structured_content

    (closed,) = content["closed"]
    (exit_,) = lines(cache, "exit")
    assert closed == {
        "id": 1,
        "ticker": "ASML.AS",
        **{
            k: v
            for k, v in exit_.items()
            if k not in ("kind", "decision_id", "settled_at")
        },
    }
    assert (closed["reason"], closed["price"]) == ("stop", STOP)
    assert content["open"] == []


@pytest.mark.anyio
async def test_settling_again_writes_nothing(tmp_path: Path) -> None:
    # @AC10
    cache = cache_for(fake({after(B, 2): (101.0, 102.0, 99.5, 101.0)}), tmp_path)
    seed(cache, planned_call(B), planned_call(DAYS[-30]))
    await settle_positions(cache)
    before = journal_of(cache).path.read_bytes()

    content = (await settle_positions(cache)).structured_content

    assert journal_of(cache).path.read_bytes() == before
    assert (content["filled"], content["expired"], content["closed"]) == ([], [], [])
    assert [position["id"] for position in content["open"]] == [1]


@pytest.mark.anyio
async def test_a_session_not_cached_is_not_settled_until_it_is(
    tmp_path: Path,
) -> None:
    # @AC12
    provider = fake({after(B, 2): (101.0, 102.0, 99.5, 101.0)})
    provider.reachable = False
    cache = cache_for(provider, tmp_path)
    seed(cache, planned_call(B))

    content = (await settle_positions(cache)).structured_content

    assert content["not_settled"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "reason": f"ASML.AS's candle for {after(B, 1).isoformat()} isn't "
            "available yet (the provider is unreachable)",
        }
    ]
    assert content["waiting"] == []
    assert lines(cache, "fill") == []

    provider.reachable = True
    content = (await settle_positions(cache)).structured_content

    assert [fill["id"] for fill in content["filled"]] == [1]
    assert content["not_settled"] == []


@pytest.mark.anyio
async def test_a_fill_whose_exit_waits_is_filled_and_not_settled(
    tmp_path: Path,
) -> None:
    # @AC12
    provider = fake({after(B, 2): (101.0, 102.0, 99.5, 101.0)})
    stored = provider.histories["ASML.AS"]
    gap = after(B, 4)
    provider.histories["ASML.AS"] = stored._replace(
        candles=[c for c in stored.candles if c.date != gap]
    )
    cache = cache_for(provider, tmp_path)
    seed(cache, planned_call(B))

    content = (await settle_positions(cache)).structured_content

    assert [fill["id"] for fill in content["filled"]] == [1]
    assert content["not_settled"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "reason": f"ASML.AS's candle for {gap.isoformat()} isn't available yet",
        }
    ]
    assert content["open"] == []


@pytest.mark.anyio
async def test_a_split_after_the_fill_is_not_settled(tmp_path: Path) -> None:
    # @AC12
    provider = fake({after(B, 2): (101.0, 102.0, 99.5, 101.0)})
    provider.now = datetime.combine(after(B, 3), time(21, 0), tzinfo=UTC)
    cache = cache_for(provider, tmp_path)
    seed(cache, planned_call(B))
    first = (await settle_positions(cache)).structured_content
    assert [fill["id"] for fill in first["filled"]] == [1]
    provider.now = AFTER_CLOSE
    provider.histories["ASML.AS"] = provider.histories["ASML.AS"]._replace(
        splits=[Split(after(B, 6), 2.0)]
    )
    before = journal_of(cache).path.read_bytes()

    content = (await settle_positions(cache)).structured_content

    (not_settled,) = content["not_settled"]
    assert not_settled["id"] == 1
    assert "split" in not_settled["reason"]
    assert after(B, 6).isoformat() in not_settled["reason"]
    assert content["open"] == []
    assert journal_of(cache).path.read_bytes() == before


@pytest.mark.anyio
async def test_without_plans_the_lists_are_empty_with_a_note(tmp_path: Path) -> None:
    # @AC13
    cache = cache_for(fake(), tmp_path)
    seed(cache, decision_draft(direction="pass"))

    content = (await settle_positions(cache)).structured_content

    for name in ("filled", "expired", "closed", "open", "waiting", "not_settled"):
        assert content[name] == []
    assert content["note"] == "No call has a trade plan to settle."
    assert len(journal_of(cache).path.read_text().splitlines()) == 1


@pytest.mark.anyio
async def test_with_every_plan_settled_the_note_says_so(tmp_path: Path) -> None:
    # @AC13
    cache = cache_for(fake(), tmp_path)
    seed(cache, planned_call(B))
    await settle_positions(cache)

    content = (await settle_positions(cache)).structured_content

    assert content["expired"] == []
    assert content["note"] == (
        "No plan is waiting or open: 0 closed, 1 expired. Nothing to settle."
    )


@pytest.mark.anyio
async def test_a_damaged_journal_is_a_tool_error(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)
    seed(cache, planned_call(B))
    with journal_of(cache).path.open("ab") as file:
        file.write(b"not json\n")

    result = await settle_positions(cache)

    assert result.is_error
    assert str(journal_of(cache).path) in result.content[0].text


def test_the_instructions_name_settle_positions() -> None:
    assert "settle_positions" in server_module.INSTRUCTIONS
