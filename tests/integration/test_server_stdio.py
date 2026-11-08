"""
Integration check of the whole MCP server over stdio, the way Claude Code runs it:
the command and arguments come from the project's `.mcp.json`, unchanged, and
`FARTT_ASSETS_DIR` points the cache at a temporary folder.
`test_tools_over_stdio` fills an empty folder from the live data provider, the
one run of every tool against Yahoo, so it's marked `network` (`task
test:network`); the stdout check runs offline on the snapshot.

Each tool story adds its tool to `EXPECTED_TOOLS` and its call to
`test_tools_over_stdio`; each prompt story its prompt to `EXPECTED_PROMPTS`.
"""

import json
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import LATEST_PROTOCOL_VERSION

from tests.integration.risk import copy_risk_config
from tests.integration.snapshot import OFFLINE

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TOOLS = [
    "get_candles",
    "get_quote",
    "get_indicators",
    "get_trend",
    "get_support_resistance",
    "record_decision",
    "get_decisions",
    "score_decisions",
    "get_performance",
    "get_analysis_procedure",
    "get_risk_config",
    "plan_trade",
]
EXPECTED_PROMPTS = ["analyze_ticker"]
# Per response; also the MCP client's read timeout, which defaults to none.
RESPONSE_TIMEOUT_S = 120


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _server_command() -> list[str]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    return [server["command"], *server["args"]]


def _environment(
    assets_dir: Path, settings: dict[str, str] | None = None
) -> dict[str, str]:
    return {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir), **(settings or {})}


@pytest.mark.network
@pytest.mark.anyio
async def test_tools_over_stdio(tmp_path: Path) -> None:
    risk_file = copy_risk_config(tmp_path)
    command, *args = _server_command()
    params = StdioServerParameters(
        command=command, args=args, cwd=str(ROOT), env=_environment(tmp_path)
    )

    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        tools = sorted(tool.name for tool in (await client.list_tools()).tools)
        assert tools == sorted(EXPECTED_TOOLS)

        # @issue-131 @AC10
        prompts = sorted(p.name for p in (await client.list_prompts()).prompts)
        assert prompts == sorted(EXPECTED_PROMPTS)
        analysis = await client.get_prompt("analyze_ticker", {"ticker": "ASML.AS"})
        assert len(analysis.messages) == 1
        assert analysis.messages[0].content.type == "text"
        assert "ASML.AS" in analysis.messages[0].content.text

        # @issue-141 @AC2
        procedure = await client.call_tool(
            "get_analysis_procedure", {"ticker": "ASML.AS"}
        )
        assert not procedure.is_error, procedure.content
        assert procedure.structured_content == {
            "ticker": "ASML.AS",
            "procedure": analysis.messages[0].content.text,
        }

        candles = await client.call_tool(
            "get_candles", {"ticker": "ASML.AS", "count": 5}
        )
        assert not candles.is_error, candles.content
        content = candles.structured_content
        assert content["ticker"] == "ASML.AS"
        assert content["exchange"] == "XAMS"
        assert content["columns"] == ["date", "open", "high", "low", "close", "volume"]
        assert len(content["rows"]) == 5

        quote = await client.call_tool("get_quote", {"ticker": "ASML.AS"})
        assert not quote.is_error, quote.content
        prices = quote.structured_content
        assert prices["ticker"] == "ASML.AS"
        assert prices["last"] > 0
        assert isinstance(prices["market_open"], bool)

        indicators = await client.call_tool("get_indicators", {"ticker": "ASML.AS"})
        assert not indicators.is_error, indicators.content
        values = indicators.structured_content
        assert values["ticker"] == "ASML.AS"
        assert values["sma_200"] is not None
        assert len(json.dumps(values)) < 2000

        # @issue-108 @AC9
        trend = await client.call_tool("get_trend", {"ticker": "ASML.AS"})
        assert not trend.is_error, trend.content
        timeframes = trend.structured_content
        assert timeframes["ticker"] == "ASML.AS"
        assert timeframes["weekly"]["sma_200"] is not None
        assert len(json.dumps(timeframes)) < 2000

        # @issue-109 @AC11
        levels = await client.call_tool("get_support_resistance", {"ticker": "ASML.AS"})
        assert not levels.is_error, levels.content
        zones = levels.structured_content
        assert zones["ticker"] == "ASML.AS"
        assert zones["support"] or zones["resistance"]
        assert len(json.dumps(zones)) < 3500

        # @issue-119 @AC9
        decision = await client.call_tool(
            "record_decision",
            {
                "ticker": "ASML.AS",
                "direction": "buy",
                "horizon": 20,
                "reasoning": "Integration check over stdio.",
                "signals": [{"name": "trend", "stance": "for"}],
            },
        )
        assert not decision.is_error, decision.content
        entry = decision.structured_content
        assert entry["id"] == 1
        assert entry["ticker"] == "ASML.AS"
        assert entry["benchmark"]["ticker"] == "IWDA.AS"

        # @issue-120 @AC10
        journal = await client.call_tool("get_decisions", {})
        assert not journal.is_error, journal.content
        calls = journal.structured_content
        assert calls["total"] == 1
        read_back = calls["decisions"][0]
        assert read_back["status"] == "open"
        assert read_back["score"] is None
        assert {
            k: v
            for k, v in read_back.items()
            if k not in ("status", "horizon_passed", "score")
        } == entry

        # @issue-121 @AC12
        scoring = await client.call_tool("score_decisions", {})
        assert not scoring.is_error, scoring.content
        run = scoring.structured_content
        assert run["scored"] == []
        assert run["not_scored"] == []
        assert run["running"] == 1
        assert run["already_scored"] == 0

        # @issue-122 @AC10
        performance = await client.call_tool("get_performance", {})
        assert not performance.is_error, performance.content
        figures = performance.structured_content
        assert figures["scored"] == 0
        assert figures["open"] == 1
        assert figures["overall"]["calls"] == 0
        assert figures["tickers"] == []

        # @issue-149 @AC1
        risk = await client.call_tool("get_risk_config", {})
        assert not risk.is_error, risk.content
        config = risk.structured_content
        assert Path(config["source"]).resolve() == risk_file.resolve()
        assert config["limits"]["max_risk_per_position"] == 0.02
        assert sorted(config["costs"]["exchanges"]) == sorted(
            ["XNYS", "XAMS", "XPAR", "XBRU", "XLIS", "XMIL", "XDUB", "XOSL"]
        )

        # @issue-143 @AC1
        last = zones["last"]
        plan = await client.call_tool(
            "plan_trade",
            {
                "ticker": "ASML.AS",
                "entry": round(last * 0.98, 2),
                "stop": round(last * 0.98 * 0.95, 2),
                "target": round(last * 0.98 * 1.3, 2),
            },
        )
        assert not plan.is_error, plan.content
        sized = plan.structured_content
        assert sized["ticker"] == "ASML.AS"
        assert sized["shares"] >= 1
        assert sized["risk"] <= 0.02 * config["limits"]["starting_capital"]

    assert (tmp_path / "stocks" / "ASML.AS" / "candles.csv").exists()
    assert (tmp_path / "journal" / "decisions.jsonl").exists()


def test_stdout_carries_only_the_protocol(seeded_assets: Path) -> None:
    messages: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "fartt-integration", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_candles",
                "arguments": {"ticker": "ASML.AS", "count": 2},
            },
        },
        # @issue-131 @AC10
        {"jsonrpc": "2.0", "id": 4, "method": "prompts/list"},
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "prompts/get",
            "params": {"name": "analyze_ticker", "arguments": {"ticker": "ASML.AS"}},
        },
    ]
    process = subprocess.Popen(
        _server_command(),
        cwd=ROOT,
        env=_environment(seeded_assets, OFFLINE),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        # Its own process group, so a failed test can kill `uv` and the
        # server it started together; killing `uv` alone leaves the
        # server running.
        start_new_session=True,
    )
    try:
        _check_stdout(process, messages)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _check_stdout(
    process: subprocess.Popen[str], messages: list[dict[str, Any]]
) -> None:
    assert process.stdin is not None and process.stdout is not None

    # Read stdout on a thread: readline() blocks, and a deadline per line
    # keeps a hung server from hanging the test run. None marks the end.
    lines: queue.Queue[str | None] = queue.Queue()
    stdout = process.stdout

    def read_stdout() -> None:
        for line in iter(stdout.readline, ""):
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()

    received: list[dict[str, Any]] = []

    def next_line() -> str | None:
        try:
            return lines.get(timeout=RESPONSE_TIMEOUT_S)
        except queue.Empty:
            pytest.fail(f"no output within {RESPONSE_TIMEOUT_S} s; got {received}")

    def check(line: str) -> dict[str, Any]:
        try:
            parsed: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            pytest.fail(f"stdout carried a non-JSON line: {line!r}")
        assert parsed.get("jsonrpc") == "2.0", f"not a JSON-RPC message: {line!r}"
        return parsed

    try:
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

        # Keep stdin open until the last response: closing it ends the
        # session before a tool call (run on a worker thread) can answer.
        last_id = max(message["id"] for message in messages if "id" in message)
        while not all(
            any(message.get("id") == i for message in received)
            for i in range(1, last_id + 1)
        ):
            line = next_line()
            if line is None:
                pytest.fail(f"server closed stdout early; got {received}")
            received.append(check(line))
    finally:
        process.stdin.close()

    # Then read to the end: output buffered before serving started only
    # reaches stdout when the process exits.
    while (line := next_line()) is not None:
        received.append(check(line))
    process.wait(timeout=30)

    responses = {message["id"]: message for message in received if "id" in message}
    assert set(responses) == {1, 2, 3, 4, 5}
    assert "error" not in responses[3]
    assert responses[3]["result"].get("isError") is not True
    # @issue-131 @AC10
    assert [p["name"] for p in responses[4]["result"]["prompts"]] == EXPECTED_PROMPTS
    assert "error" not in responses[5]
    assert len(responses[5]["result"]["messages"]) == 1
