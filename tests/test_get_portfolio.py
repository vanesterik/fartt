from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from fartt.server import build_server
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from fartt.trade_plan import leg_costs
from tests.fakes import RISK_CONFIG, exit_line, fill_line, reset_line
from tests.test_server import cache_for, call, journal_of, seed
from tests.test_settle_positions import (
    DAYS,
    ENTRY,
    STOP,
    TARGET,
    B,
    after,
    fake,
    planned_call,
)

CAPITAL = RISK_CONFIG.limits.starting_capital
FILLED = {after(B, 2): (101.0, 102.0, 99.5, 101.0)}  # Fills the plan at 100
QUIET_CLOSE = 102.0
HOLDING_FIELDS = {
    "id", "ticker", "currency", "shares", "entry", "stop", "target",
    "last_close", "based_on", "rate", "value", "dividends", "unrealised_pnl",
    "to_stop", "to_stop_atr", "to_target", "to_target_atr", "sessions_held",
    "horizon_ends", "is_current", "warning",
}  # fmt: skip


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def get_portfolio(cache: StockCache) -> Any:
    result = await call(cache, {}, "get_portfolio")
    assert not result.is_error, result.content
    return result.structured_content


def sale(shares: int, price: float) -> float:
    costs = leg_costs(
        shares * price, None, "EUR", RISK_CONFIG.costs.exchanges["XAMS"],
        RISK_CONFIG.costs, "sell",
    )  # fmt: skip
    return round(round(shares * price, 2) - costs.total, 2)


@pytest.mark.anyio
async def test_an_empty_portfolio_says_so(tmp_path: Path) -> None:
    # @AC1
    content = await get_portfolio(cache_for(fake(), tmp_path))

    totals = content["totals"]
    assert (totals["cash"], totals["free_cash"], totals["total_value"]) == (
        CAPITAL,
    ) * 3
    assert totals["drawdown"] == 0
    assert (content["open"], content["waiting"]) == ([], [])
    assert content["closed"]["count"] == 0
    assert content["settled"] == {
        "filled": [], "expired": [], "closed": [], "not_settled": [],
    }  # fmt: skip
    assert content["note"].startswith("No plans yet")


@pytest.mark.anyio
async def test_an_open_position_is_shown_against_its_plan(tmp_path: Path) -> None:
    # @AC2
    cache = cache_for(fake(FILLED), tmp_path)
    (call_,) = seed(cache, planned_call(B))
    assert call_.plan is not None
    shares = call_.plan.shares

    content = await get_portfolio(cache)

    (held,) = content["open"]
    assert set(held) == HOLDING_FIELDS
    assert (held["id"], held["ticker"], held["shares"]) == (1, "ASML.AS", shares)
    assert (held["entry"], held["stop"], held["target"]) == (ENTRY, STOP, TARGET)
    assert held["last_close"] == QUIET_CLOSE
    assert held["based_on"] == DAYS[-1].isoformat()
    assert held["value"] == sale(shares, QUIET_CLOSE)
    assert held["to_stop"] == round((QUIET_CLOSE - STOP) / QUIET_CLOSE, 4)
    assert held["to_stop_atr"] is not None
    assert held["sessions_held"] == len([d for d in DAYS if d > after(B, 2)])
    assert held["is_current"] is True
    assert [f["id"] for f in content["settled"]["filled"]] == [1]
    assert content["based_on"] == DAYS[-1].isoformat()
    assert content["note"] is None


@pytest.mark.anyio
async def test_a_stop_hit_in_the_newest_session_shows_as_closed(
    tmp_path: Path,
) -> None:
    # @AC4
    stopped = {**FILLED, DAYS[-1]: (98.0, 99.0, 94.0, 96.0)}
    cache = cache_for(fake(stopped), tmp_path)
    seed(cache, planned_call(B))

    content = await get_portfolio(cache)

    assert content["open"] == []
    (closed,) = content["settled"]["closed"]
    assert (closed["id"], closed["reason"]) == (1, "stop")
    assert content["closed"]["count"] == 1
    assert content["closed"]["recent"][0]["session"] == DAYS[-1].isoformat()
    totals = content["totals"]
    assert totals["total_value"] == round(CAPITAL + closed["pnl"], 2)


@pytest.mark.anyio
async def test_a_waiting_plan_is_listed_and_reserves_cash(tmp_path: Path) -> None:
    # @AC5
    cache = cache_for(fake(), tmp_path)
    (call_,) = seed(cache, planned_call(DAYS[-3]))
    plan = call_.plan
    assert plan is not None

    content = await get_portfolio(cache)

    reserved = round(plan.position_value + plan.costs.buy.total, 2)
    assert content["waiting"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "entry": ENTRY,
            "stop": STOP,
            "target": TARGET,
            "shares": plan.shares,
            "reserved": reserved,
            "valid_until": after(DAYS[-3], 5).isoformat(),
        }
    ]
    assert content["totals"]["free_cash"] == round(CAPITAL - reserved, 2)


@pytest.mark.anyio
async def test_a_stale_ticker_is_marked_and_named_in_the_note(
    tmp_path: Path,
) -> None:
    # @AC11
    provider = fake(FILLED)
    cache = cache_for(provider, tmp_path)
    seed(cache, planned_call(B))
    await get_portfolio(cache)
    provider.now = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)
    provider.unreachable = {"ASML.AS"}

    content = await get_portfolio(cache)

    (held,) = content["open"]
    assert held["is_current"] is False
    assert held["based_on"] == DAYS[-1].isoformat()
    assert "couldn't bring ASML.AS up to date" in held["warning"]
    assert "ASML.AS" in content["note"]


@pytest.mark.anyio
async def test_a_blocked_waiting_plan_still_reserves_cash(tmp_path: Path) -> None:
    provider = fake()
    provider.reachable = False
    cache = cache_for(provider, tmp_path)
    seed(cache, planned_call(DAYS[-3]))

    content = await get_portfolio(cache)

    assert [w["id"] for w in content["waiting"]] == [1]
    assert content["totals"]["reserved"] > 0
    assert [n["id"] for n in content["settled"]["not_settled"]] == [1]


@pytest.mark.anyio
async def test_negative_cash_is_noted(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)
    seed(cache, planned_call(B))
    journal_of(cache).append_fill(
        fill_line(1, session=after(B, 2), price=ENTRY, value=CAPITAL + 100.0)
    )

    content = await get_portfolio(cache)

    assert content["totals"]["cash"] < 0
    assert "Cash is below 0" in content["note"]


@pytest.mark.anyio
async def test_waiting_is_capped_with_a_note(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)
    seed(cache, *(planned_call(DAYS[-3]) for _ in range(51)))

    content = await get_portfolio(cache)

    assert len(content["waiting"]) == server_module.MAX_SCORED
    assert content["totals"]["waiting_plans"] == 51
    assert "first 50 of 51 waiting plans" in content["note"]


@pytest.mark.anyio
async def test_get_portfolio_settles_so_it_is_idempotent_not_read_only(
    tmp_path: Path,
) -> None:
    cache = cache_for(fake(), tmp_path)
    async with Client(build_server(cache, journal_of(cache), RISK_CONFIG)) as client:
        tools = (await client.list_tools()).tools
    (tool,) = [t for t in tools if t.name == "get_portfolio"]

    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is False
    assert tool.annotations.destructive_hint is False
    assert tool.annotations.idempotent_hint is True
    assert tool.input_schema.get("properties", {}) == {}


def test_the_instructions_say_what_get_portfolio_shows() -> None:
    (sentence,) = [
        s for s in server_module.INSTRUCTIONS.split(". ") if "get_portfolio" in s
    ]
    assert "settles the plans first" in sentence
    assert "open positions against their plans" in sentence


@pytest.mark.anyio
async def test_get_portfolio_shows_the_limit_and_the_kill_switch(
    tmp_path: Path,
) -> None:
    # @AC8
    cache = cache_for(fake(FILLED), tmp_path)
    seed(cache, planned_call(B), planned_call(DAYS[-3]), planned_call(DAYS[-3]))

    content = await get_portfolio(cache)

    assert content["limits"] == {
        "max_open_positions": 8,
        "open_positions": 1,
        "waiting_plans": 2,
        "room": 5,
    }
    assert content["kill_switch"] == {
        "tripped": False,
        "tripped_on": None,
        "drawdown_at_trip": None,
        "threshold": 0.16,
        "drawdown": content["totals"]["drawdown"],
        "last_reset": None,
    }


# Call 1 fills 50 at 102 and exits through a gap at 60 two sessions later:
# the portfolio closes that session 21% below its peak of 10,000.
LOSS_FILLED, LOSS_EXIT = DAYS[-27], DAYS[-26]
LOSS_VALUE = round(CAPITAL - 5103.6 + 3000.0 - 3.6, 2)


def lose(cache: StockCache) -> None:
    journal = journal_of(cache)
    journal.append_fill(
        fill_line(1, session=LOSS_FILLED, price=102.0, shares=50, value=5100.0)
    )
    journal.append_exit(
        exit_line(1, session=LOSS_EXIT, price=60.0, value=3000.0, pnl=-2107.2)
    )


@pytest.mark.anyio
async def test_a_tripped_switch_shows_and_open_positions_still_close(
    tmp_path: Path,
) -> None:
    # @AC2 @AC4
    stopped = {**FILLED, DAYS[-1]: (98.0, 99.0, 94.0, 96.0)}
    cache = cache_for(fake(stopped), tmp_path)
    seed(cache, planned_call(DAYS[-30]), planned_call(B))
    lose(cache)

    content = await get_portfolio(cache)

    switch = content["kill_switch"]
    assert (switch["tripped"], switch["tripped_on"]) == (True, LOSS_EXIT.isoformat())
    assert switch["drawdown_at_trip"] == round((CAPITAL - LOSS_VALUE) / CAPITAL, 4)
    # Call 2 filled and hit its stop after the trip, settled as ever.
    assert [f["id"] for f in content["settled"]["filled"]] == [2]
    assert [(c["id"], c["reason"]) for c in content["settled"]["closed"]] == [
        (2, "stop")
    ]


@pytest.mark.anyio
async def test_get_portfolio_shows_the_reset_and_measures_from_it(
    tmp_path: Path,
) -> None:
    # @AC5
    cache = cache_for(fake(), tmp_path)
    seed(cache, planned_call(DAYS[-30]))
    lose(cache)
    reset = reset_line(session=LOSS_EXIT, total_value=LOSS_VALUE, drawdown=0.2107)
    journal_of(cache).append_reset(reset)

    content = await get_portfolio(cache)

    switch, totals = content["kill_switch"], content["totals"]
    assert switch["tripped"] is False
    assert switch["last_reset"] == {
        "reset_at": reset.reset_at.isoformat().replace("+00:00", "Z"),
        "session": LOSS_EXIT.isoformat(),
        "drawdown": 0.2107,
        "note": reset.note,
    }
    assert (totals["peak"], totals["peak_session"]) == (
        LOSS_VALUE,
        LOSS_EXIT.isoformat(),
    )
    assert totals["drawdown"] == switch["drawdown"] == 0
