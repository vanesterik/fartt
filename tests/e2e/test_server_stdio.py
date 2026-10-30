"""
End-to-end check of the MCP server over stdio, the way Claude Code runs it:
the command and arguments come from the project's `.mcp.json`, unchanged, and
`FARTT_ASSETS_DIR` points the cache at a temporary folder. Calls the live data
provider, so it's marked `network` (run with `task test:network`).

Each tool story adds its tool to `EXPECTED_TOOLS` and its call to
`test_tools_over_stdio`.
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

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TOOLS = ["get_candles", "get_quote"]
# Per response; also the MCP client's read timeout, which defaults to none.
RESPONSE_TIMEOUT_S = 120


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _server_command() -> list[str]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    return [server["command"], *server["args"]]


def _environment(assets_dir: Path) -> dict[str, str]:
    return {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir)}


@pytest.mark.network
@pytest.mark.anyio
async def test_tools_over_stdio(tmp_path: Path) -> None:
    command, *args = _server_command()
    params = StdioServerParameters(
        command=command, args=args, cwd=str(ROOT), env=_environment(tmp_path)
    )

    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        tools = sorted(tool.name for tool in (await client.list_tools()).tools)
        assert tools == sorted(EXPECTED_TOOLS)

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

    assert (tmp_path / "stocks" / "ASML.AS" / "candles.csv").exists()


@pytest.mark.network
def test_stdout_carries_only_the_protocol(tmp_path: Path) -> None:
    messages: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "fartt-e2e", "version": "0"},
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
    ]
    process = subprocess.Popen(
        _server_command(),
        cwd=ROOT,
        env=_environment(tmp_path),
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
        while not any(message.get("id") == 3 for message in received):
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
    assert set(responses) == {1, 2, 3}
    assert "error" not in responses[3]
    assert responses[3]["result"].get("isError") is not True
