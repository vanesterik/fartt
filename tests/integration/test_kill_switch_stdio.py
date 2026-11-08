"""
Integration tests for the open-position limit and the kill switch (#146),
driven the way Claude Code and the operator drive them: the server started
from `.mcp.json` over stdio, with `FARTT_ASSETS_DIR` set to a temporary folder
holding the snapshot, its clock pinned to the snapshot's moment and Yahoo out
of reach (`snapshot.OFFLINE`), and the reset run as the operator runs it,
`task reset-kill-switch`, with the same folder and clock.

Each test's docstring starts with its tags: `@issue-146` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and the
snapshot's candles, worked out by hand below with `risk.toml`'s costs.

As the design says, a portfolio 16% below its peak is built by backdating a
recorded plan (`_seed`, as #144's tests do): `TRIPS` buys 5 ASML.AS shares at
the open of 06-30 (1666.6) and holds them to the horizon (09-21, close
1481.4), through the fall to 1388.4 on 07-28.

- Cash after the fill: 10,000 - 8,333.00 - 15.00 buy costs = 1,652.00.
- Peak, 06-30's close 1721.4: 1,652.00 + 8,607.00 - 15.50 sell costs =
  10,243.50.
- 07-27 (close 1429.8, a 1.88 dividend ex): 8,797.53, a drawdown of 0.1412.
- 07-28 (close 1388.4): 1,661.40 + 6,942.00 - 12.49 = 8,590.91, a drawdown of
  0.1613: the first close at or beyond the threshold of 0.16.
- Closed at its horizon on 09-21: a total of 9,055.06, a drawdown of 0.116,
  back under the threshold.
"""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from tests.integration.snapshot import SNAPSHOT_NOW
from tests.integration.test_settle_positions_stdio import (
    CENT,
    FILL_AT_ENTRY_THEN_STOP,
    ROOT,
    Plan,
    _call,
    _journal,
    _seed,
    _server,
    _text,
)

pytestmark = pytest.mark.anyio

TRIPS = Plan("ASML.AS", "2026-06-29", 60, 5, 1720, 1000, 3000, 5)
TRIPPED_ON = "2026-07-28"
DRAWDOWN_AT_TRIP = 0.1613
PEAK = 10_243.5
TOTAL_AFTER_TRIP = 9_055.06
THRESHOLD = 0.16
LIMIT = 8

# Open: filled at its entry on 11-02; no later low reaches 1200 or high 1700.
OPEN = Plan("ASML.AS", "2026-10-30", 20, 5, 1405, 1200, 1700, 1)
# Waiting: recorded on the newest session, entry far below the last close.
WAITING = Plan("AAPL", "2026-11-04", 20, 5, 300, 280, 400, 1)

# Recorded on 10-26 while the switch is tripped; the snapshot is cut there so
# they wait, then the sessions after it arrive.
# 10-27 opens at 1424 (fill at the open); 10-28's high 1503.6 reaches 1500.
TO_TARGET = Plan("ASML.AS", "2026-10-26", 20, 5, 1430, 1300, 1500, 2)
# 10-30's low 1404.2 reaches 1410; 11-03's low 1358 reaches 1360.
TO_STOP = Plan("ASML.AS", "2026-10-26", 20, 5, 1410, 1360, 1700, 2)
# Filled at 1424 on 10-27, held to horizon_ends 11-02 (close 1408).
TO_HORIZON = Plan("ASML.AS", "2026-10-26", 5, 5, 1430, 1200, 1700, 2)
# Opened 11-02 at 1405 and held: valued at 11-02's close, then 11-04's.
OPEN_THROUGH_RESET = Plan("ASML.AS", "2026-10-30", 20, 5, 1405, 1200, 1700, 2)

NEW_PLAN = {"entry": 1330, "stop": 1280, "target": 1500, "valid_for": 5}
CALL = {
    "ticker": "ASML.AS",
    "horizon": 20,
    "reasoning": "A call made while the limits are tested.",
    "signals": [{"name": "trend", "stance": "for"}],
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _seed_sized(assets_dir: Path, plans: list[Plan]) -> None:
    """
    `_seed`, with each plan's position value worked out again from its
    shares and entry, so the cash it reserves is what the plan would cost.
    """
    entries = await _seed(assets_dir, plans)
    for entry in entries:
        plan = entry["plan"]
        plan["position_value"] = round(
            plan["shares"] * plan["entry"] / (plan["rate"] or 1), 2
        )
    _journal(assets_dir).write_text("".join(json.dumps(e) + "\n" for e in entries))


def _cut_asml_after(assets_dir: Path, day: str) -> Callable[[], None]:
    """Leave out ASML.AS's sessions after `day`; return what puts them back."""
    candles = assets_dir / "stocks" / "ASML.AS" / "candles.csv"
    full = candles.read_text()
    lines = full.splitlines(keepends=True)
    candles.write_text(lines[0] + "".join(x for x in lines[1:] if x[:10] <= day))
    return lambda: candles.write_text(full)


def _reset(assets_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run `task reset-kill-switch` as the operator does, on this folder and clock."""
    env = {
        **os.environ,
        "FARTT_ASSETS_DIR": str(assets_dir),
        "FARTT_NOW": SNAPSHOT_NOW.isoformat(),
    }
    command = ["task", "reset-kill-switch"]
    if arguments:
        command += ["--", *arguments]
    return subprocess.run(
        command,
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _journal_lines(assets_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _journal(assets_dir).read_text().splitlines()]


async def _refused(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error, result.structured_content
    return _text(result)


async def _plan_both_ways(client: Client) -> tuple[str, str]:
    """The refusals of plan_trade and of record_decision with the same plan."""
    planned = await _refused(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)
    recorded = await _refused(
        client, "record_decision", direction="buy", plan=NEW_PLAN, **CALL
    )
    return planned, recorded


def _check_kill_switch_message(message: str) -> None:
    assert "kill switch tripped" in message
    assert TRIPPED_ON in message
    assert str(DRAWDOWN_AT_TRIP) in message
    assert "Only the operator can reset it" in message
    assert "task reset-kill-switch" in message
    assert "Nothing was recorded" in message


class TestOpenPositionLimit:
    async def test_should_accept_the_eighth_plan_and_refuse_the_ninth(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC1 @AC8 Should accept a plan with 3 open positions and 4
        waiting plans (7 together), then refuse the next in plan_trade naming
        the limit and both counts; get_portfolio shows the room going from 1
        to 0.
        """
        await _seed_sized(seeded_assets, [OPEN] * 3 + [WAITING] * 4)
        async with _server(seeded_assets) as client:
            before = (await _call(client, "get_portfolio"))["limits"]
            planned = await _call(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)
            recorded = await _call(
                client, "record_decision", direction="buy", plan=NEW_PLAN, **CALL
            )
            after = (await _call(client, "get_portfolio"))["limits"]
            refused = await _refused(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)

        assert before == {
            "max_open_positions": LIMIT,
            "open_positions": 3,
            "waiting_plans": 4,
            "room": 1,
        }
        assert planned["shares"] >= 1
        assert recorded["id"] == 8 and recorded["plan"] is not None
        assert after == {
            "max_open_positions": LIMIT,
            "open_positions": 3,
            "waiting_plans": 5,
            "room": 0,
        }
        assert "open-position limit of 8" in refused
        assert "3 positions are open" in refused
        assert "5 plans wait" in refused

    @pytest.mark.parametrize(("open_count", "waiting_count"), [(3, 5), (0, 8), (8, 0)])
    async def test_should_refuse_a_plan_at_eight_and_record_nothing(
        self, seeded_assets: Path, open_count: int, waiting_count: int
    ) -> None:
        """
        @issue-146 @AC1 @AC8 Should refuse plan_trade and record_decision with
        a plan when open positions and waiting plans together reach 8, naming
        the limit and both counts, and leave the journal unchanged; the
        portfolio shows no room.
        """
        await _seed_sized(
            seeded_assets, [OPEN] * open_count + [WAITING] * waiting_count
        )
        async with _server(seeded_assets) as client:
            limits = (await _call(client, "get_portfolio"))["limits"]
            before = _journal(seeded_assets).read_bytes()
            messages = await _plan_both_ways(client)

        assert limits == {
            "max_open_positions": LIMIT,
            "open_positions": open_count,
            "waiting_plans": waiting_count,
            "room": 0,
        }
        for message in messages:
            assert "open-position limit of 8" in message
            assert f"{open_count} positions are open" in message
            assert f"{waiting_count} plans wait" in message
            assert "Nothing was recorded" in message
        assert _journal(seeded_assets).read_bytes() == before

    async def test_should_show_room_for_eight_plans_on_an_empty_portfolio(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC8 Should show the limit of 8 with no open positions, no
        waiting plans and room for 8 when nothing has been recorded.
        """
        async with _server(seeded_assets) as client:
            limits = (await _call(client, "get_portfolio"))["limits"]

        assert limits == {
            "max_open_positions": LIMIT,
            "open_positions": 0,
            "waiting_plans": 0,
            "room": LIMIT,
        }


class TestKillSwitch:
    async def test_should_show_the_trip_and_keep_it_after_the_value_recovers(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC2 Should show the kill switch as tripped on the first
        close 16% or more below the peak (07-28, at 0.1613), and still
        tripped once the drawdown is back under the threshold.
        """
        await _seed(seeded_assets, [TRIPS])
        async with _server(seeded_assets) as client:
            portfolio = await _call(client, "get_portfolio")

        switch = portfolio["kill_switch"]
        assert switch["tripped"] is True
        assert switch["tripped_on"] == TRIPPED_ON
        assert switch["drawdown_at_trip"] == pytest.approx(DRAWDOWN_AT_TRIP, abs=1e-4)
        assert switch["threshold"] == THRESHOLD
        assert switch["last_reset"] is None
        totals = portfolio["totals"]
        assert totals["peak"] == pytest.approx(PEAK, abs=CENT)
        assert totals["peak_session"] == "2026-06-30"
        assert totals["total_value"] == pytest.approx(TOTAL_AFTER_TRIP, abs=CENT)
        assert switch["drawdown"] == pytest.approx(0.116, abs=1e-4)
        assert switch["drawdown"] < THRESHOLD

    async def test_should_refuse_plans_while_tripped_and_still_record_a_pass(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC3 Should refuse plan_trade and record_decision with a
        plan while the switch is tripped, saying it tripped, on which session,
        at what drawdown and that only the operator can reset it, recording
        nothing; a pass and a buy without a plan are still recorded.
        """
        await _seed(seeded_assets, [TRIPS])
        async with _server(seeded_assets) as client:
            await _call(client, "get_portfolio")
            before = _journal(seeded_assets).read_bytes()
            messages = await _plan_both_ways(client)
            unchanged = _journal(seeded_assets).read_bytes() == before
            passed = await _call(client, "record_decision", direction="pass", **CALL)
            bought = await _call(client, "record_decision", direction="buy", **CALL)

        for message in messages:
            _check_kill_switch_message(message)
        assert unchanged
        assert passed["direction"] == "pass" and passed["plan"] is None
        assert bought["direction"] == "buy" and bought["plan"] is None

    async def test_should_give_the_kill_switch_before_the_limit(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC3 Should refuse a plan with the kill switch's message
        when the switch has tripped and 8 plans are waiting as well.
        """
        await _seed_sized(seeded_assets, [TRIPS] + [WAITING] * 8)
        async with _server(seeded_assets) as client:
            message = await _refused(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)

        _check_kill_switch_message(message)

    async def test_should_fill_and_close_positions_while_tripped(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC4 Should, while the switch is tripped, fill waiting
        plans and close the positions at their target, stop and horizon when
        the sessions that reach them arrive.
        """
        await _seed(seeded_assets, [TRIPS, TO_TARGET, TO_STOP, TO_HORIZON])
        restore = _cut_asml_after(seeded_assets, "2026-10-26")
        async with _server(seeded_assets) as client:
            waiting = await _call(client, "get_portfolio")
        restore()
        async with _server(seeded_assets) as client:
            settled = await _call(client, "settle_positions")
            portfolio = await _call(client, "get_portfolio")

        assert waiting["kill_switch"]["tripped"] is True
        assert [w["id"] for w in waiting["waiting"]] == [2, 3, 4]
        filled = {f["id"]: (f["session"], f["price"]) for f in settled["filled"]}
        assert filled == {
            2: ("2026-10-27", 1424),
            3: ("2026-10-30", 1410),
            4: ("2026-10-27", 1424),
        }
        closed = {
            c["id"]: (c["session"], c["reason"], c["price"]) for c in settled["closed"]
        }
        assert closed == {
            2: ("2026-10-28", "target", 1500),
            3: ("2026-11-03", "stop", 1360),
            4: ("2026-11-02", "horizon", 1408),
        }
        assert portfolio["open"] == [] and portfolio["waiting"] == []
        assert portfolio["kill_switch"]["tripped"] is True
        assert portfolio["kill_switch"]["tripped_on"] == TRIPPED_ON


class TestReset:
    async def test_should_write_a_reset_line_and_accept_plans_again(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC5 Should, on a tripped switch, write a reset line with
        the time, the session, the total, the peak, the drawdown and the
        note, exit 0, after which plan_trade accepts a plan and get_portfolio
        shows the reset and a drawdown of 0 from the value at the reset.
        """
        await _seed(seeded_assets, [TRIPS])
        async with _server(seeded_assets) as client:
            await _call(client, "get_portfolio")
        done = _reset(seeded_assets, "--note", "gap on ASML")
        async with _server(seeded_assets) as client:
            portfolio = await _call(client, "get_portfolio")
            planned = await _call(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)

        assert done.returncode == 0, done.stderr
        line = _journal_lines(seeded_assets)[-1]
        assert line["kind"] == "reset"
        assert line["reset_at"] == "2026-11-05T12:00:00Z"
        assert line["session"] == "2026-09-21"
        assert line["total_value"] == pytest.approx(TOTAL_AFTER_TRIP, abs=CENT)
        assert line["peak"] == pytest.approx(PEAK, abs=CENT)
        assert line["drawdown"] == pytest.approx(0.116, abs=1e-4)
        assert line["tripped_on"] == TRIPPED_ON
        assert line["note"] == "gap on ASML"
        switch = portfolio["kill_switch"]
        assert switch["tripped"] is False
        assert switch["drawdown"] == 0
        assert switch["last_reset"] == {
            "reset_at": "2026-11-05T12:00:00Z",
            "session": "2026-09-21",
            "drawdown": line["drawdown"],
            "note": "gap on ASML",
        }
        assert portfolio["totals"]["peak"] == pytest.approx(TOTAL_AFTER_TRIP, abs=CENT)
        assert planned["shares"] >= 1

    async def test_should_measure_a_later_drawdown_from_the_value_at_the_reset(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC5 Should measure the drawdown after a reset from the
        total value at the reset: a reset on 11-02 at 9,050.93, then 11-03
        and 11-04 arrive with the open position lower, a drawdown of 0.0155.
        """
        await _seed(seeded_assets, [TRIPS, OPEN_THROUGH_RESET])
        restore = _cut_asml_after(seeded_assets, "2026-11-02")
        async with _server(seeded_assets) as client:
            at_reset = await _call(client, "get_portfolio")
        done = _reset(seeded_assets, "--note", "looked at the July gap")
        restore()
        async with _server(seeded_assets) as client:
            portfolio = await _call(client, "get_portfolio")

        assert done.returncode == 0, done.stderr
        line = _journal_lines(seeded_assets)[-1]
        value = at_reset["totals"]["total_value"]
        assert value == pytest.approx(9_050.93, abs=CENT)
        assert (line["kind"], line["session"]) == ("reset", "2026-11-02")
        assert line["total_value"] == pytest.approx(value, abs=CENT)
        totals = portfolio["totals"]
        assert totals["peak"] == pytest.approx(value, abs=CENT)
        assert totals["total_value"] < value
        drawdown = (value - totals["total_value"]) / value
        assert portfolio["kill_switch"]["drawdown"] == pytest.approx(drawdown, abs=1e-4)
        assert portfolio["kill_switch"]["drawdown"] == pytest.approx(0.0155, abs=1e-4)
        assert portfolio["kill_switch"]["tripped"] is False
        assert (
            portfolio["kill_switch"]["last_reset"]["note"] == "looked at the July gap"
        )

    async def test_should_write_nothing_and_fail_when_the_switch_has_not_tripped(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC6 Should, on a switch that hasn't tripped, write
        nothing, say so with the current drawdown and exit non-zero: with no
        journal at all, and after a small loss (a drawdown of 0.0095).
        """
        empty = _reset(seeded_assets)
        assert empty.returncode != 0
        assert "isn't tripped" in empty.stdout
        assert "drawdown of 0.0 " in empty.stdout
        assert not (seeded_assets / "journal" / "decisions.jsonl").exists()

        await _seed(seeded_assets, [FILL_AT_ENTRY_THEN_STOP])
        async with _server(seeded_assets) as client:
            drawdown = (await _call(client, "get_portfolio"))["kill_switch"]["drawdown"]
        before = _journal(seeded_assets).read_bytes()
        small = _reset(seeded_assets, "--note", "not needed")

        assert drawdown == pytest.approx(0.0095, abs=1e-4)
        assert small.returncode != 0
        assert "isn't tripped" in small.stdout
        assert f"drawdown of {drawdown}" in small.stdout
        assert _journal(seeded_assets).read_bytes() == before

    async def test_should_write_nothing_on_a_damaged_journal(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC6 Should fail with exit status 2 and leave a damaged journal as it is,
        naming the file.
        """
        await _seed(seeded_assets, [TRIPS])
        journal = _journal(seeded_assets)
        journal.write_text(journal.read_text() + "{broken\n")
        before = journal.read_bytes()

        done = _reset(seeded_assets, "--note", "anything")

        # task exits 201 for a failed command and names the command's status.
        assert done.returncode != 0
        assert "exit status 2" in done.stderr
        assert str(journal) in done.stdout + done.stderr
        assert journal.read_bytes() == before


class TestAgentCannotReset:
    async def test_should_list_no_tool_that_resets_the_switch_or_changes_a_limit(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-146 @AC7 Should list no tool whose name or arguments reset the
        kill switch or set a limit, and leave a tripped switch tripped and the
        limits unchanged after every tool that writes has been called.
        """
        await _seed(seeded_assets, [TRIPS])
        async with _server(seeded_assets) as client:
            limits = (await _call(client, "get_risk_config"))["limits"]
            tools = (await client.list_tools()).tools
            await _call(client, "settle_positions")
            await _call(client, "score_decisions")
            await _call(client, "record_decision", direction="pass", **CALL)
            switch = (await _call(client, "get_portfolio"))["kill_switch"]
            limits_after = (await _call(client, "get_risk_config"))["limits"]

        # get_decisions' `limit` caps the list it reads; get_risk_config only
        # reads (checked below).
        allowed = {("get_decisions", "limit"), ("get_risk_config", "get_risk_config")}
        forbidden = (
            "reset",
            "kill",
            "switch",
            "limit",
            "threshold",
            "drawdown",
            "risk",
        )
        for tool in tools:
            words = [tool.name, *(tool.input_schema or {}).get("properties", {})]
            for word in words:
                if (tool.name, word) in allowed:
                    continue
                assert not any(f in word.lower() for f in forbidden), (tool.name, word)
        get_risk_config = next(t for t in tools if t.name == "get_risk_config")
        assert not (get_risk_config.input_schema or {}).get("properties")
        assert get_risk_config.annotations is not None
        assert get_risk_config.annotations.read_only_hint is True
        assert switch["tripped"] is True and switch["tripped_on"] == TRIPPED_ON
        assert limits_after == limits
        assert not any(
            line.get("kind") == "reset" for line in _journal_lines(seeded_assets)
        )
