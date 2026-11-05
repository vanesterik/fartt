"""
Integration tests for `get_decisions` (#120), driven the way Claude Code
drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder holding the snapshot, its clock
pinned to the snapshot's moment and Yahoo out of reach (`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-120` and the acceptance
criteria it verifies. Expectations come from the issue's criteria. Calls are
recorded through `record_decision`, so they are the entries the server itself
stores. The `horizon_passed` test rewrites a recorded entry's `horizon_ends`
to sessions either side of the exchange's newest closed session at the
pinned clock.
"""

import datetime as dt
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest
from mcp import Client, StdioServerParameters

from tests.integration.snapshot import OFFLINE, SNAPSHOT_NOW

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
# The project's margin after an exchange's close before a session counts as
# closed (CLAUDE.md: `newest_closed_session`, a 30-minute `CLOSE_MARGIN`).
CLOSE_MARGIN = dt.timedelta(minutes=30)
# The signal names `record_decision` accepts (`fartt.journal.SIGNALS`); a
# name added there belongs here too, so AC9's worst case stays the worst.
SIGNAL_NAMES = [
    "trend",
    "support_resistance",
    "moving_averages",
    "rsi",
    "macd",
    "volume",
    "atr",
    "spread",
]
ENTRY_FIELDS = {
    "id",
    "recorded_at",
    "ticker",
    "currency",
    "exchange",
    "direction",
    "horizon",
    "reasoning",
    "signals",
    "decision_session",
    "close",
    "horizon_ends",
    "benchmark",
}

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(assets_dir: Path, *, live: bool = False) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {
        **os.environ,
        "FARTT_ASSETS_DIR": str(assets_dir),
        **({} if live else OFFLINE),
    }
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


def _text(result: Any) -> str:
    return " ".join(getattr(part, "text", "") for part in result.content)


async def _decisions(client: Client, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool("get_decisions", arguments)
    assert not result.is_error, result.content
    content = result.structured_content
    assert content is not None
    return content


async def _error(client: Client, **arguments: Any) -> str:
    result = await client.call_tool("get_decisions", arguments)
    assert result.is_error, result.structured_content
    return _text(result)


async def _record(
    client: Client,
    ticker: str = "ASML.AS",
    *,
    reasoning: str = "Integration test call.",
    signals: list[dict[str, Any]] | None = None,
    horizon: int = 20,
    direction: str = "buy",
) -> dict[str, Any]:
    result = await client.call_tool(
        "record_decision",
        {
            "ticker": ticker,
            "direction": direction,
            "horizon": horizon,
            "reasoning": reasoning,
            "signals": signals or [{"name": "trend", "stance": "for"}],
        },
    )
    assert not result.is_error, result.content
    entry = result.structured_content
    assert entry is not None
    return entry


def _ids(content: dict[str, Any]) -> list[int]:
    return [entry["id"] for entry in content["decisions"]]


def _stored(entry: dict[str, Any]) -> dict[str, Any]:
    """An entry as get_decisions returns it, without the fields it adds."""
    return {
        k: v for k, v in entry.items() if k not in ("status", "horizon_passed", "score")
    }


def _journal(assets_dir: Path) -> Path:
    return assets_dir / "journal" / "decisions.jsonl"


def _newest_closed_session(exchange: str) -> dt.date:
    """The exchange's latest session whose close lies CLOSE_MARGIN in the past."""
    calendar = xcals.get_calendar(exchange)
    now = pd.Timestamp(SNAPSHOT_NOW - CLOSE_MARGIN)
    start = now.normalize().tz_localize(None) - pd.Timedelta(days=14)
    sessions = calendar.sessions_in_range(start, now.normalize().tz_localize(None))
    closed = [s for s in sessions if calendar.session_close(s) <= now]
    return closed[-1].date()


def _next_session(exchange: str, day: dt.date) -> dt.date:
    return xcals.get_calendar(exchange).next_session(pd.Timestamp(day)).date()


async def test_should_return_open_calls_newest_first_as_recorded(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC1 @AC7 @AC10 Should return open calls newest first, as recorded."""
    async with _server(seeded_assets) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        assert "get_decisions" in tools
        recorded = [
            await _record(client, "ASML.AS"),
            await _record(client, "AAPL", direction="pass"),
            await _record(client, "ASML.AS", horizon=5),
        ]
        open_calls = await _decisions(client, status="open")
        every_call = await _decisions(client)
        scored = await _decisions(client, status="scored")

    by_id = {entry["id"]: entry for entry in recorded}
    assert _ids(open_calls) == [3, 2, 1]
    assert open_calls["total"] == 3
    for entry in open_calls["decisions"]:
        assert set(entry) >= ENTRY_FIELDS
        assert _stored(entry) == by_id[entry["id"]]
        assert entry["status"] == "open"
        assert isinstance(entry["horizon_passed"], bool)

    assert every_call["decisions"] == open_calls["decisions"]
    assert every_call["total"] == 3

    assert scored["decisions"] == []
    assert scored["total"] == 0
    assert scored["note"]


async def test_should_filter_by_ticker_case_and_space_insensitively(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC2 Should return only that ticker's calls, newest first."""
    async with _server(seeded_assets) as client:
        await _record(client, "ASML.AS")
        await _record(client, "AAPL")
        await _record(client, "ASML.AS")
        asml = await _decisions(client, ticker=" asml.as ")
        aapl = await _decisions(client, ticker="aapl")
        asml_open = await _decisions(client, ticker="Asml.As", status="open")
        asml_scored = await _decisions(client, ticker="ASML.AS", status="scored")

    assert _ids(asml) == [3, 1]
    assert {entry["ticker"] for entry in asml["decisions"]} == {"ASML.AS"}
    assert asml["total"] == 2
    assert _ids(aapl) == [2]
    assert _ids(asml_open) == [3, 1]
    assert asml_scored["decisions"] == []


@pytest.mark.parametrize("ticker", ["AS ML", "../x", ""])
async def test_should_reject_a_string_that_cant_be_a_ticker(
    tmp_path: Path, ticker: str
) -> None:
    """@issue-120 @AC2 Should be a tool error for a string that can't be a ticker."""
    async with _server(tmp_path) as client:
        text = await _error(client, ticker=ticker)

    assert "isn't a ticker" in text


async def test_should_return_calls_unchanged_after_a_restart(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC3 Should return calls recorded before a restart unchanged."""
    async with _server(seeded_assets) as client:
        recorded = [await _record(client, "ASML.AS"), await _record(client, "AAPL")]
    journal_before = _journal(seeded_assets).read_bytes()

    async with _server(seeded_assets) as client:
        after = await _decisions(client)

    assert [_stored(entry) for entry in after["decisions"]] == recorded[::-1]
    assert _journal(seeded_assets).read_bytes() == journal_before


async def test_should_cap_at_the_limit_with_the_total_and_a_note(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC4 Should return the newest up to the limit, the total and a note."""
    async with _server(seeded_assets) as client:
        for number in range(1, 13):
            await _record(client, reasoning=f"Call {number}.")
        default = await _decisions(client)
        three = await _decisions(client, limit=3)

    assert _ids(default) == list(range(12, 2, -1))
    assert default["total"] == 12
    assert all(word in default["note"] for word in ("10", "12", "ticker", "status"))
    assert _ids(three) == [12, 11, 10]
    assert three["total"] == 12
    assert "3" in three["note"] and "12" in three["note"]


@pytest.mark.parametrize("limit", [0, 11, -1])
async def test_should_reject_a_limit_outside_1_to_10(
    tmp_path: Path, limit: int
) -> None:
    """@issue-120 @AC4 Should reject a limit below 1 or above 10."""
    async with _server(tmp_path) as client:
        text = await _error(client, limit=limit)

    assert "limit" in text


async def test_should_return_an_empty_list_with_a_note_when_nothing_matches(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC5 Should return an empty list, total 0 and a note, not an error."""
    async with _server(seeded_assets) as client:
        empty_journal = await _decisions(client)
        await _record(client, "ASML.AS")
        no_match = await _decisions(client, ticker="MSFT")

    assert empty_journal["decisions"] == []
    assert empty_journal["total"] == 0
    assert empty_journal["note"]
    assert no_match["decisions"] == []
    assert no_match["total"] == 0
    assert "MSFT" in no_match["note"]


async def test_should_pass_the_horizon_from_the_close_of_its_last_session(
    seeded_assets: Path,
) -> None:
    """
    @issue-120 @AC6 Should set horizon_passed from the close of horizon_ends.

    The newest closed session (close plus margin in the past) has passed; the
    session after it hasn't closed yet, so it hasn't.
    """
    async with _server(seeded_assets) as client:
        await _record(client, "ASML.AS")
        await _record(client, "AAPL")
    stored = [
        json.loads(line) for line in _journal(seeded_assets).read_text().splitlines()
    ]

    lines: list[str] = []
    expected: dict[int, bool] = {}
    for entry in stored:
        closed = _newest_closed_session(entry["exchange"])
        after = _next_session(entry["exchange"], closed)
        for ends, passed in ((closed, True), (after, False)):
            line_id = len(lines) + 1
            changed = {**entry, "id": line_id, "horizon_ends": ends.isoformat()}
            lines.append(json.dumps(changed))
            expected[line_id] = passed
    _journal(seeded_assets).write_text("\n".join(lines) + "\n")

    async with _server(seeded_assets) as client:
        content = await _decisions(client)

    got = {entry["id"]: entry["horizon_passed"] for entry in content["decisions"]}
    assert got == expected


async def test_should_be_a_tool_error_on_a_damaged_line_and_leave_the_file(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC8 Should name the journal file and leave it byte-for-byte."""
    async with _server(seeded_assets) as client:
        await _record(client, "ASML.AS")
    journal = _journal(seeded_assets)
    with journal.open("ab") as file:
        file.write(b"not json\n")
    before = journal.read_bytes()

    async with _server(seeded_assets) as client:
        text = await _error(client)
        filtered = await _error(client, ticker="ASML.AS")

    assert str(journal) in text
    assert str(journal) in filtered
    assert journal.read_bytes() == before


async def test_should_fit_ten_of_the_largest_calls_under_60000_characters(
    seeded_assets: Path,
) -> None:
    """@issue-120 @AC9 Should return ten of the largest calls in under 60,000 characters."""
    signals = [
        {"name": name, "stance": "for", "note": "n" * 200} for name in SIGNAL_NAMES
    ]
    async with _server(seeded_assets) as client:
        for _ in range(10):
            await _record(client, reasoning="x" * 2000, signals=signals)
        content = await _decisions(client)

    assert len(content["decisions"]) == 10
    assert all(len(entry["reasoning"]) == 2000 for entry in content["decisions"])
    assert all(len(entry["signals"]) == 8 for entry in content["decisions"])
    assert len(json.dumps(content)) < 60_000
