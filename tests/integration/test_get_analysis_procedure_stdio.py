"""
Integration tests for the `get_analysis_procedure` tool (#141), driven the way
Claude Code drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to an empty temporary folder, its clock pinned and
Yahoo out of reach (`snapshot.OFFLINE`): the tool calls no provider.

Each test's docstring starts with its tags: `@issue-141` and the acceptance
criteria it verifies. The tool's text is checked against the `analyze_ticker`
prompt's text for the same argument, which the criteria name as the source of
truth; the prompt's own content is covered by `test_analyze_ticker_stdio.py`
(#131), which also guards AC6.
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult

from tests.integration.risk import written
from tests.integration.snapshot import OFFLINE

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 120
TOOL = "get_analysis_procedure"
PROMPT = "analyze_ticker"
# The SDK frames every tool error's message this way.
TOOL_ERROR_PREFIX = f"Error executing tool {TOOL}: "

# Argument given -> the normalised ticker.
ACCEPTED = {
    "ASML.AS": "ASML.AS",
    "AAPL": "AAPL",
    "asml.as": "ASML.AS",
    " asml.as ": "ASML.AS",
    "aapl": "AAPL",
}
NOT_A_TICKER = ["", "   ", "AS ML!", "ASML/AS"]
REFUSED = [*NOT_A_TICKER, "VOD.L", "EURUSD=X", "eurusd=x"]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(assets_dir: Path) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir), **OFFLINE}
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _prompt_text(client: Client, ticker: str) -> str:
    result = await client.get_prompt(PROMPT, {"ticker": ticker})
    assert len(result.messages) == 1
    content = result.messages[0].content
    assert content.type == "text"
    return content.text


def _error_text(result: CallToolResult) -> str:
    texts = [item.text for item in result.content if item.type == "text"]
    assert len(texts) == 1, texts
    return texts[0]


class TestListing:
    async def test_tool_is_listed_with_one_required_ticker(
        self, empty_assets: Path
    ) -> None:
        """@issue-141 @AC1 Should list get_analysis_procedure, read-only, with one required ticker."""
        async with _server(empty_assets) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        assert TOOL in tools
        tool = tools[TOOL]
        assert tool.description
        schema = tool.input_schema
        assert list(schema["properties"]) == ["ticker"]
        assert schema.get("required") == ["ticker"]
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True

    async def test_missing_ticker_is_an_error(self, empty_assets: Path) -> None:
        """@issue-141 @AC1 Should refuse a call without a ticker."""
        async with _server(empty_assets) as client:
            result = await client.call_tool(TOOL, {})

        assert result.is_error
        assert "ticker" in _error_text(result)
        assert written(empty_assets) == []


class TestProcedure:
    async def test_procedure_is_the_prompt_text(self, empty_assets: Path) -> None:
        """@issue-141 @AC2 @AC4 Should return the prompt's text for the normalised ticker, offline, writing nothing."""
        async with _server(empty_assets) as client:
            for given, normalised in ACCEPTED.items():
                result = await client.call_tool(TOOL, {"ticker": given})
                assert not result.is_error, (given, result.content)
                content = result.structured_content
                assert content is not None
                assert content["ticker"] == normalised, given
                assert content["procedure"] == await _prompt_text(client, given), given
                assert normalised in content["procedure"], given

        # Yahoo was out of reach and nothing was written: no journal, no stock.
        assert written(empty_assets) == []

    @pytest.mark.parametrize("ticker", REFUSED)
    async def test_refusal_matches_the_prompt(
        self, empty_assets: Path, ticker: str
    ) -> None:
        """@issue-141 @AC3 @AC4 Should refuse a ticker that can't be analysed with the prompt's message."""
        async with _server(empty_assets) as client:
            result = await client.call_tool(TOOL, {"ticker": ticker})
            with pytest.raises(MCPError) as refused:
                await client.get_prompt(PROMPT, {"ticker": ticker})

        assert result.is_error
        prompt_message = refused.value.message
        assert _error_text(result) == TOOL_ERROR_PREFIX + prompt_message
        # Ends in what a valid ticker looks like.
        assert "Give one stock in Yahoo's format" in prompt_message
        assert "ASML.AS" in prompt_message and "AAPL" in prompt_message
        assert written(empty_assets) == []


class TestInstructions:
    async def test_instructions_point_to_the_tool(self, empty_assets: Path) -> None:
        """@issue-141 @AC5 Should tell the agent to call get_analysis_procedure before analysing, unless the prompt gave it."""
        async with _server(empty_assets) as client:
            instructions = client.instructions

        assert instructions is not None
        lowered = instructions.lower()
        tool_at = lowered.find(TOOL)
        assert tool_at >= 0
        sentence_start = lowered.rfind(". ", 0, tool_at) + 2
        sentence_end = lowered.find(".", tool_at + len(TOOL))
        sentence = lowered[sentence_start : sentence_end if sentence_end >= 0 else None]
        # Before an analysis of a ticker, and follow what it returns.
        assert "before analys" in sentence or "before analyz" in sentence
        assert "follow" in sentence
        # Unless the analyze_ticker prompt already gave the procedure.
        assert PROMPT in sentence
        # No longer says an analysis follows the prompt alone.
        assert (
            "an analysis of a ticker follows the analyze_ticker prompt" not in lowered
        )
