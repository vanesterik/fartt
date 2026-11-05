"""
Integration tests for the `analyze_ticker` prompt (#131), driven the way
Claude Code drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder. The prompt itself calls no
provider, but the tests start the real server and sit with the other stdio
tests, so they're marked `network` too.

The check that a refused ticker never reaches the provider builds the same
server in this process on the real `YFinanceProvider`, with a ticker factory
that records every symbol it's asked for.

Each test's docstring starts with its tags: `@issue-131` and the acceptance
criteria it verifies. Expectations come from the issue's criteria, so the
prompt's text is checked for what it must say and in which order, not for
its exact wording.

AC8 and AC9 need a live Claude Code session and are checked by hand by the
test role, not here.
"""

import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import yfinance
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

from fartt.data import YFinanceProvider
from fartt.journal import Journal
from fartt.server import build_server
from fartt.stock_cache import StockCache

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 120
PROMPT = "analyze_ticker"

# The PRD's eight steps, in order.
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
# The steps without tools yet and the epic that adds each.
SKIPPED = {
    "market context": "F",
    "sector selection": "F",
    "screening": "G",
    "fundamental analysis": "G",
    "trade plan and risk": "H",
}
SIGNALS = [
    "trend",
    "support_resistance",
    "moving_averages",
    "rsi",
    "macd",
    "volume",
    "atr",
    "spread",
]
REPORT_SECTIONS = [
    "the call",
    "evidence for",
    "evidence against",
    "skipped steps",
    "earlier calls",
    "what was recorded",
]
NOT_A_TICKER = ["", "   ", "AS ML!", "ASML/AS"]
REFUSED = [*NOT_A_TICKER, "VOD.L", "EURUSD=X"]

pytestmark = [pytest.mark.network, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(assets_dir: Path) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir)}
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _prompt_text(client: Client, ticker: str) -> str:
    result = await client.get_prompt(PROMPT, {"ticker": ticker})
    assert len(result.messages) == 1
    message = result.messages[0]
    assert message.role == "user"
    assert message.content.type == "text"
    return message.content.text


def _in_order(text: str, phrases: list[str], start: int = 0) -> list[int]:
    """Where each phrase first appears after the one before it."""
    lowered = text.lower()
    positions: list[int] = []
    for phrase in phrases:
        found = lowered.find(phrase, start)
        assert found >= 0, f"{phrase!r} not found after position {start}"
        positions.append(found)
        start = found + len(phrase)
    return positions


def _step_sections(text: str) -> dict[str, str]:
    """Each step's text, from its name to the next step's name (or the end)."""
    positions = _in_order(text, STEPS)
    ends = [*positions[1:], len(text)]
    return {
        step: text[begin:end]
        for step, begin, end in zip(STEPS, positions, ends, strict=True)
    }


async def test_prompt_is_listed_with_one_required_ticker(tmp_path: Path) -> None:
    """@issue-131 @AC1 Should list analyze_ticker with one required argument."""
    async with _server(tmp_path) as client:
        prompts = {p.name: p for p in (await client.list_prompts()).prompts}

    assert PROMPT in prompts
    prompt = prompts[PROMPT]
    assert prompt.description
    assert prompt.arguments is not None
    assert [(a.name, a.required) for a in prompt.arguments] == [("ticker", True)]


async def test_prompt_names_the_eight_steps_in_order(tmp_path: Path) -> None:
    """@issue-131 @AC2 Should name the PRD's eight steps in order for ASML.AS."""
    async with _server(tmp_path) as client:
        text = await _prompt_text(client, "asml.as")

    assert "ASML.AS" in text
    assert "asml.as" not in text
    _in_order(text, STEPS)


async def test_prompt_names_the_tools_and_skipped_steps(tmp_path: Path) -> None:
    """@issue-131 @AC3 Should name each step's tools and mark the rest skipped."""
    async with _server(tmp_path) as client:
        text = await _prompt_text(client, "ASML.AS")
    sections = _step_sections(text)

    for step, epic in SKIPPED.items():
        section = sections[step].lower()
        assert "skipped" in section, step
        assert re.search(rf"\bepic {epic}\b", sections[step], re.IGNORECASE), step
    for tool in (
        "get_trend",
        "get_indicators",
        "get_support_resistance",
        "get_candles",
    ):
        assert tool in sections["technical analysis"], tool
    assert "get_quote" in sections["execution"]
    # The analysis starts with the ticker's earlier calls, before step 1.
    before_step_1 = text[: text.lower().find(STEPS[0])]
    assert "get_decisions" in before_step_1
    assert "ASML.AS" in before_step_1
    assert "record_decision" in sections["monitoring and review"]


async def test_prompt_states_the_decision_rules(tmp_path: Path) -> None:
    """@issue-131 @AC4 Should state the decision rules the agent follows."""
    async with _server(tmp_path) as client:
        text = await _prompt_text(client, "ASML.AS")
    lowered = text.lower()

    # Evidence before a view.
    assert "evidence" in lowered and "view" in lowered
    # No buy while the weekly trend is down.
    assert re.search(r"no `?buy`? while the weekly trend", lowered)
    assert "`down`" in lowered or " down" in lowered
    # Weighed together, not summed.
    assert "weighed together" in lowered
    assert "summed" in lowered
    # A buy names the evidence against; otherwise pass.
    assert "against" in lowered
    assert "pass" in lowered
    # Which kind of pass: evidence pointing down or mixed evidence.
    assert "pointing down" in lowered
    assert "mixed evidence" in lowered
    # A horizon of 20 sessions.
    assert re.search(r"20\s*\(?sessions", lowered)
    # Signal names from the journal's eight, each for or against.
    for signal in SIGNALS:
        assert f"`{signal}`" in text, signal
    assert "`for`" in text and "`against`" in text


async def test_prompt_says_when_no_call_is_recorded(tmp_path: Path) -> None:
    """@issue-131 @AC5 Should say an open call or stale data means no new call."""
    async with _server(tmp_path) as client:
        text = await _prompt_text(client, "ASML.AS")
    lowered = text.lower()
    no_record = r"(don't|do not|never) record"

    open_call = lowered.find("still open")
    assert open_call >= 0
    assert re.search(no_record, lowered[open_call : open_call + 400])
    stale = lowered.find("is_current: false")
    assert stale >= 0
    assert re.search(no_record, lowered[stale : stale + 300])


async def test_prompt_names_the_report_sections_in_order(tmp_path: Path) -> None:
    """@issue-131 @AC6 Should name the report's six sections in order."""
    async with _server(tmp_path) as client:
        text = await _prompt_text(client, "ASML.AS")

    # The report comes after the procedure's last step.
    after_steps = _in_order(text, STEPS)[-1]
    _in_order(text, REPORT_SECTIONS, start=after_steps)


@pytest.mark.parametrize("ticker", REFUSED)
async def test_refused_ticker_is_an_error_and_no_prompt(
    tmp_path: Path, ticker: str
) -> None:
    """@issue-131 @AC7 Should refuse a ticker that can't be analysed."""
    async with _server(tmp_path) as client:
        with pytest.raises(MCPError) as raised:
            await client.get_prompt(PROMPT, {"ticker": ticker})

    message = raised.value.message
    # Says what a valid ticker looks like.
    assert "ASML.AS" in message
    assert "AAPL" in message
    assert not any(tmp_path.iterdir())


async def test_refused_ticker_never_reaches_the_provider(tmp_path: Path) -> None:
    """@issue-131 @AC7 Should refuse every bad ticker without calling Yahoo."""
    asked: list[str] = []

    def ticker_factory(symbol: str) -> Any:
        asked.append(symbol)
        return yfinance.Ticker(symbol)

    cache = StockCache(YFinanceProvider(ticker_factory=ticker_factory), tmp_path)
    server = build_server(cache, Journal(tmp_path / "journal"))
    async with Client(server) as client:
        for ticker in REFUSED:
            with pytest.raises(MCPError):
                await client.get_prompt(PROMPT, {"ticker": ticker})
        # A valid ticker gets its prompt without a fetch either.
        await _prompt_text(client, "ASML.AS")

    assert asked == []
    assert not any(tmp_path.iterdir())
