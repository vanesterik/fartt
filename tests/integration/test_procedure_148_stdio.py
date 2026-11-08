"""
Integration tests for the trade plan and the daily check-in in the procedure
(#148), driven the way Claude Code drives the server: started from
`.mcp.json` over stdio, with `FARTT_ASSETS_DIR` set to a temporary folder
holding the snapshot (or nothing), its clock pinned to the snapshot's moment
and Yahoo out of reach (`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-148` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and its
design, so the procedures' texts are checked for what they must say and in
which order, not for their exact wording. How the agent follows them in a
Claude session is out of the story's scope.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from tests.integration.risk import written
from tests.integration.test_settle_positions_stdio import (
    FILL_AT_ENTRY_THEN_STOP,
    STILL_OPEN,
    Plan,
    _call,
    _journal,
    _seed,
    _server,
    _text,
)

pytestmark = pytest.mark.anyio

ANALYSIS_PROMPT = "analyze_ticker"
ANALYSIS_TOOL = "get_analysis_procedure"
CHECKIN_PROMPT = "daily_checkin"
CHECKIN_TOOL = "get_checkin_procedure"
TICKERS = ["ASML.AS", "AAPL", "MC.PA"]

STEPS = [
    "market context",
    "sector selection",
    "screening",
    "fundamental analysis",
    "technical analysis",
    "trade plan and risk",
    "execution",
    "monitoring and review",
]
# The design's check-in report sections, in order.
CHECKIN_SECTIONS = [
    "Settled",
    "Portfolio",
    "Open positions",
    "Waiting plans",
    "New scores",
    "Not scored yet",
    "The gate",
]
CHECKIN_TOOLS = ["get_portfolio", "score_decisions", "get_performance"]
# A tool name as the procedures write one: in backticks, maybe with arguments.
TOOL_NAME = re.compile(r"`((?:get|plan|record|score|settle)_[a-z_]+)[(`]")

CALL: dict[str, Any] = {
    "ticker": "ASML.AS",
    "horizon": 20,
    "reasoning": "A call made while the procedure is tested.",
    "signals": [{"name": "trend", "stance": "for"}],
}
# The kill switch trips on 07-28 (see test_kill_switch_stdio.py).
TRIPS = Plan("ASML.AS", "2026-06-29", 60, 5, 1720, 1000, 3000, 5)
NEW_PLAN = {"entry": 1330, "stop": 1280, "target": 1500, "valid_for": 5}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _prompt_text(client: Client, name: str, arguments: dict[str, str]) -> str:
    result = await client.get_prompt(name, arguments)
    assert len(result.messages) == 1
    message = result.messages[0]
    assert message.role == "user"
    assert message.content.type == "text"
    return message.content.text


def _positions(text: str, phrases: list[str], start: int = 0) -> list[int]:
    """Where each phrase first appears after the previous one, case-blind."""
    lowered = text.lower()
    positions = []
    for phrase in phrases:
        found = lowered.find(phrase.lower(), start)
        assert found >= 0, f"{phrase!r} not found in order"
        positions.append(found)
        start = found + len(phrase)
    return positions


def _steps(text: str) -> dict[str, str]:
    """Each step's text, from its heading to the next step's (or the end)."""
    begins = _positions(text, [f"step {n}: {s}" for n, s in enumerate(STEPS, 1)])
    ends = [*begins[1:], len(text)]
    return {s: text[b:e] for s, b, e in zip(STEPS, begins, ends, strict=True)}


def _section(text: str, heading: str) -> str:
    """The text under a `## heading`, up to the next `## `."""
    start = text.find(f"## {heading}")
    assert start >= 0, heading
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else None]


def _numbered_bold(section: str) -> list[str]:
    """The bold names of a numbered list: `1. **Name**: ...`."""
    return re.findall(r"^\d+\. \*\*(.+?)\*\*", section, re.MULTILINE)


def _sentence_with(text: str, phrase: str) -> str:
    at = text.find(phrase)
    assert at >= 0, phrase
    begin = text.rfind(". ", 0, at) + 2
    end = text.find(". ", at)
    return text[begin : end if end >= 0 else None]


class TestAnalysisProcedure:
    async def test_should_name_the_tools_of_steps_6_and_7(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC1 Should name get_portfolio and plan_trade in step 6 and get_quote and get_risk_config in step 7, in the prompt and the tool alike, and never skip them for epic H."""
        async with _server(empty_assets) as client:
            for ticker in TICKERS:
                text = await _prompt_text(client, ANALYSIS_PROMPT, {"ticker": ticker})
                tool = await _call(client, ANALYSIS_TOOL, ticker=ticker)
                assert tool["procedure"] == text, ticker

                steps = _steps(text)
                plan, execution = steps["trade plan and risk"], steps["execution"]
                _positions(plan, ["`get_portfolio(", "`plan_trade("])
                assert f'plan_trade(ticker="{ticker}"' in plan
                _positions(execution, ["`get_quote(", "`get_risk_config("])
                for section in (plan, execution):
                    assert "skipped" not in section.lower(), ticker
                    assert "no tool yet" not in section.lower(), ticker
                assert not re.search(r"no tool yet \(epic H\)", text, re.IGNORECASE)
                # The other skipped steps are still reported as skipped.
                assert "no tool yet (epic F)" in steps["market context"]
                assert "no tool yet (epic G)" in steps["fundamental analysis"]
        # Text only: nothing fetched, nothing written.
        assert written(empty_assets) == []

    async def test_should_say_where_the_plan_comes_from_and_what_to_do_without_room(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC2 Should say in step 6 where entry, stop and target come from, allow one adjustment on a refused reward-to-risk, and say what to do with no room."""
        async with _server(empty_assets) as client:
            text = await _prompt_text(client, ANALYSIS_PROMPT, {"ticker": "ASML.AS"})
        plan = _steps(text)["trade plan and risk"].lower()

        # From the evidence of step 5 (the support and resistance zones).
        assert "from the evidence of step 5" in plan
        assert re.search(r"entry[^.]*support zone", plan)
        assert re.search(r"stop[^.]*below[^.]*zone", plan)
        assert re.search(r"target[^.]*resistance zone", plan)
        # One adjustment within the evidence on a refused reward-to-risk.
        refused = plan.find("refuses on reward-to-risk")
        assert refused >= 0
        after = plan[refused : refused + 400]
        assert "adjust once" in after
        assert "within the evidence" in after
        assert "evidence against a buy" in after
        # No room: the kill switch, the open-position limit, the free cash.
        room = plan.find("room")
        assert room >= 0
        for limit in ("kill_switch", "limits.room", "free_cash"):
            assert limit in plan, limit
        assert "records nothing for a buy" in plan
        # Reports the plan or the refusal.
        assert "reward-to-risk after costs" in plan
        assert "refusal" in plan

    async def test_should_name_three_kinds_of_pass_and_the_plan_section(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC3 Should name the three kinds of pass, record nothing for a buy without room, and report The plan after Evidence against."""
        async with _server(empty_assets) as client:
            text = await _prompt_text(client, ANALYSIS_PROMPT, {"ticker": "ASML.AS"})
        rules = _section(text, "Decision rules")
        lowered = rules.lower()

        kinds = lowered.find("three kinds")
        assert kinds >= 0
        kinds_rule = lowered[kinds : lowered.find("\n", kinds)]
        for kind in ("pointing down", "mixed evidence", '"no plan fits"'):
            assert kind in kinds_rule, kind
        no_room = _sentence_with(lowered, "no room in the portfolio")
        assert "record nothing" in no_room
        assert "buy" in no_room

        sections = _numbered_bold(_section(text, "Report"))
        assert "The plan" in sections
        assert sections.index("The plan") == sections.index("Evidence against") + 1
        # The analysis prompt stays well under the MCP output limit.
        assert len(text) < 20_000


class TestBuyNeedsAPlan:
    async def test_should_refuse_a_buy_without_a_plan_and_record_nothing(
        self, seeded_assets: Path
    ) -> None:
        """@issue-148 @AC4 Should refuse a buy without a plan, naming plan_trade, and record nothing, on an empty and on a filled journal."""
        async with _server(seeded_assets) as client:
            first = await client.call_tool(
                "record_decision", {**CALL, "direction": "buy"}
            )
            journal = _journal(seeded_assets)
            empty_after_first = not journal.exists() or journal.read_bytes() == b""
            passed = await _call(client, "record_decision", **CALL, direction="pass")
            before = _journal(seeded_assets).read_bytes()
            second = await client.call_tool(
                "record_decision", {**CALL, "ticker": "AAPL", "direction": "buy"}
            )
            after = _journal(seeded_assets).read_bytes()
            read_back = await _call(client, "get_decisions")

        for result in (first, second):
            assert result.is_error
            message = _text(result)
            assert "buy needs a plan" in message
            assert "plan_trade" in message
            assert "Nothing was recorded" in message
        assert empty_after_first
        assert before.count(b"\n") == 1
        assert after == before
        assert read_back["total"] == 1
        assert [d["id"] for d in read_back["decisions"]] == [passed["id"]]

    async def test_should_record_a_pass_without_a_plan_as_before(
        self, seeded_assets: Path
    ) -> None:
        """@issue-148 @AC4 Should record a pass without a plan, with plan null, as before."""
        async with _server(seeded_assets) as client:
            entry = await _call(client, "record_decision", **CALL, direction="pass")
            read_back = await _call(client, "get_decisions", id=entry["id"])

        assert entry["id"] == 1
        assert entry["direction"] == "pass"
        assert entry["plan"] is None
        assert read_back["decisions"][0]["plan"] is None
        stored = json.loads(_journal(seeded_assets).read_text())
        assert stored["direction"] == "pass" and stored["plan"] is None

    async def test_should_record_a_buy_with_the_plan_plan_trade_accepted(
        self, seeded_assets: Path
    ) -> None:
        """@issue-148 @AC4 Should record a buy with the plan plan_trade accepted, as step 8 says."""
        async with _server(seeded_assets) as client:
            accepted = await _call(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)
            entry = await _call(
                client, "record_decision", **CALL, direction="buy", plan=NEW_PLAN
            )

        assert entry["direction"] == "buy"
        assert entry["plan"] == accepted


class TestCheckinProcedure:
    async def test_should_return_the_same_checkin_from_prompt_and_tool(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC5 Should list daily_checkin without arguments and get_checkin_procedure read-only without arguments, both returning the same text."""
        async with _server(empty_assets) as client:
            prompts = {p.name: p for p in (await client.list_prompts()).prompts}
            tools = {t.name: t for t in (await client.list_tools()).tools}
            text = await _prompt_text(client, CHECKIN_PROMPT, {})
            tool = await _call(client, CHECKIN_TOOL)

        assert CHECKIN_PROMPT in prompts
        assert not prompts[CHECKIN_PROMPT].arguments
        assert CHECKIN_TOOL in tools
        assert tools[CHECKIN_TOOL].input_schema.get("properties", {}) == {}
        annotations = tools[CHECKIN_TOOL].annotations
        assert annotations is not None and annotations.read_only_hint is True
        assert tool == {"procedure": text}
        assert written(empty_assets) == []

    async def test_should_call_three_tools_and_report_seven_sections_in_order(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC5 Should name get_portfolio, score_decisions and get_performance in that order, and the seven report sections in order."""
        async with _server(empty_assets) as client:
            text = await _prompt_text(client, CHECKIN_PROMPT, {})

        steps = _section(text, "The steps")
        named = TOOL_NAME.findall(steps)
        assert named == CHECKIN_TOOLS
        sections = _numbered_bold(_section(text, "Report"))
        assert sections == CHECKIN_SECTIONS
        # No analysis and no new calls or plans in a check-in.
        assert "no new calls or plans" in text.lower()
        assert "record_decision" not in text
        assert "plan_trade" not in text

    async def test_should_run_the_checkin_tools_on_a_portfolio(
        self, seeded_assets: Path
    ) -> None:
        """@issue-148 @AC5 Should run the check-in's tools in its order on open positions and a call whose horizon passed."""
        await _seed(seeded_assets, [FILL_AT_ENTRY_THEN_STOP, STILL_OPEN])
        async with _server(seeded_assets) as client:
            text = await _prompt_text(client, CHECKIN_PROMPT, {})
            outputs = {
                tool: await _call(client, tool) for tool in TOOL_NAME.findall(text)
            }

        assert list(outputs) == CHECKIN_TOOLS
        portfolio = outputs["get_portfolio"]
        assert len(portfolio["open"]) == 1
        assert portfolio["settled"]
        scored = outputs["score_decisions"]["scored"]
        assert [call["id"] for call in scored] == [1]
        assert outputs["get_performance"]["scored"] == 1


class TestInstructions:
    async def test_should_send_a_plain_checkin_request_to_the_procedure(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC6 Should tell the agent to call get_checkin_procedure before a daily check-in, unless the prompt gave it."""
        async with _server(empty_assets) as client:
            instructions = client.instructions

        assert instructions is not None
        sentence = _sentence_with(instructions, CHECKIN_TOOL).lower()
        assert "before a daily check-in" in sentence
        assert "follow" in sentence
        assert CHECKIN_PROMPT in sentence
        # The analysis keeps its own pointer.
        assert ANALYSIS_TOOL in instructions


class TestProcedureTools:
    async def test_should_list_every_tool_either_procedure_names(
        self, empty_assets: Path
    ) -> None:
        """@issue-148 @AC7 Should list every tool the analysis and check-in procedures name."""
        async with _server(empty_assets) as client:
            listed = {t.name for t in (await client.list_tools()).tools}
            named: set[str] = set()
            for ticker in TICKERS:
                text = await _prompt_text(client, ANALYSIS_PROMPT, {"ticker": ticker})
                named |= set(TOOL_NAME.findall(text))
            named |= set(
                TOOL_NAME.findall(await _prompt_text(client, CHECKIN_PROMPT, {}))
            )

        # Not vacuous: the tools steps 5 to 8 and the check-in call.
        assert named >= {
            "get_decisions",
            "get_trend",
            "get_indicators",
            "get_support_resistance",
            "get_candles",
            "get_portfolio",
            "plan_trade",
            "get_quote",
            "get_risk_config",
            "record_decision",
            "score_decisions",
            "get_performance",
        }
        assert named <= listed, named - listed


class TestNoRoomAdvice:
    async def test_should_not_advise_recording_a_buy_while_the_kill_switch_is_tripped(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-148 @AC3 @AC4 Should not, when the kill switch refuses a plan,
        tell the agent to record the call as a pass or a buy without a plan:
        the procedure records nothing for a buy without room (AC3) and the
        server refuses a buy without a plan (AC4).
        """
        await _seed(seeded_assets, [TRIPS])
        async with _server(seeded_assets) as client:
            await _call(client, "get_portfolio")
            refused = await client.call_tool(
                "plan_trade", {"ticker": "ASML.AS", **NEW_PLAN}
            )
            without_plan = await client.call_tool(
                "record_decision", {**CALL, "direction": "buy"}
            )

        assert refused.is_error
        message = _text(refused)
        assert "kill switch tripped" in message
        # A buy without a plan can't be recorded any more ...
        assert without_plan.is_error
        assert "buy needs a plan" in _text(without_plan)
        # ... so the refusal must not advise it, nor a pass for a no-room buy.
        assert "without a plan" not in message, message
        assert not re.search(r"record the call as a pass", message), message
