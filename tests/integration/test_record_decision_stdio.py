"""
Integration tests for `record_decision` (#119), driven the way Claude Code
drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder, against the live data provider.
Marked `network`.

Each test's docstring starts with its tags: `@issue-119` and the acceptance
criteria it verifies. Expectations come from the issue's criteria. Closes
change with the market, so the checks compare the entry with what
`get_candles` returns in the same session and with the exchange calendar,
never with fixed prices. A stale cache is made by cutting a cached history's
last session and keeping the provider unreachable.
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

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
# A proxy nobody listens on: every request to the data provider fails fast.
UNREACHABLE = "http://127.0.0.1:9"
PROVIDER_DOWN = {name: UNREACHABLE for name in ("HTTPS_PROXY", "https_proxy")}
BENCHMARK = "IWDA.AS"
# The project's margin after an exchange's close before a session counts as
# closed (CLAUDE.md: `newest_closed_session`, a 30-minute `CLOSE_MARGIN`).
CLOSE_MARGIN = dt.timedelta(minutes=30)
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
SIGNALS = [
    {"name": "trend", "stance": "for", "note": "daily and weekly up"},
    {"name": "rsi", "stance": "against"},
]
CALL: dict[str, Any] = {
    "ticker": "ASML.AS",
    "direction": "buy",
    "horizon": 20,
    "reasoning": "Trend up on both timeframes; RSI stretched.",
    "signals": SIGNALS,
}
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
THANKSGIVING = {dt.date(2026, 11, 26), dt.date(2027, 11, 25)}

pytestmark = [pytest.mark.network, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(
    assets_dir: Path, extra_env: dict[str, str] | None = None
) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir), **(extra_env or {})}
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    content = result.structured_content
    assert content is not None
    return content


async def _error(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error, result.structured_content
    return " ".join(getattr(part, "text", "") for part in result.content)


def _journal(assets_dir: Path) -> Path:
    return assets_dir / "journal" / "decisions.jsonl"


def _newest_closed_session(exchange: str) -> dt.date:
    """The exchange's latest session whose close lies CLOSE_MARGIN in the past."""
    calendar = xcals.get_calendar(exchange)
    now = pd.Timestamp(dt.datetime.now(dt.UTC) - CLOSE_MARGIN)
    start = now.normalize().tz_localize(None) - pd.Timedelta(days=14)
    sessions = calendar.sessions_in_range(start, now.normalize().tz_localize(None))
    closed = [s for s in sessions if calendar.session_close(s) <= now]
    return closed[-1].date()


def _sessions_after(exchange: str, day: dt.date, count: int) -> list[dt.date]:
    calendar = xcals.get_calendar(exchange)
    sessions = calendar.sessions_in_range(
        pd.Timestamp(day + dt.timedelta(days=1)),
        pd.Timestamp(day + dt.timedelta(days=count * 2 + 30)),
    )
    return [s.date() for s in sessions[:count]]


async def test_should_return_the_stored_entry(live_assets: Path) -> None:
    """@issue-119 @AC1 Should return the stored entry with every field."""
    async with _server(live_assets) as client:
        entry = await _call(client, "record_decision", **CALL)

    assert set(entry) >= ENTRY_FIELDS
    assert entry["id"] == 1
    recorded_at = dt.datetime.fromisoformat(entry["recorded_at"])
    assert recorded_at.utcoffset() == dt.timedelta(0)
    assert abs(dt.datetime.now(dt.UTC) - recorded_at) < dt.timedelta(minutes=5)
    assert (entry["ticker"], entry["currency"], entry["exchange"]) == (
        "ASML.AS",
        "EUR",
        "XAMS",
    )
    assert (entry["direction"], entry["horizon"]) == ("buy", 20)
    assert entry["reasoning"] == CALL["reasoning"]
    assert [(s["name"], s["stance"]) for s in entry["signals"]] == [
        ("trend", "for"),
        ("rsi", "against"),
    ]
    assert entry["signals"][0]["note"] == "daily and weekly up"
    dt.date.fromisoformat(entry["decision_session"])
    dt.date.fromisoformat(entry["horizon_ends"])
    assert entry["close"] > 0
    benchmark = entry["benchmark"]
    assert set(benchmark) >= {"ticker", "currency", "session", "close"}
    assert (benchmark["ticker"], benchmark["currency"]) == (BENCHMARK, "EUR")
    dt.date.fromisoformat(benchmark["session"])
    assert benchmark["close"] > 0


@pytest.mark.parametrize(
    ("ticker", "exchange"), [("ASML.AS", "XAMS"), ("AAPL", "XNYS")]
)
async def test_should_stamp_the_newest_closes_get_candles_returns(
    live_assets: Path, ticker: str, exchange: str
) -> None:
    """
    @issue-119 @AC2 Should stamp the stock's and the benchmark's newest closes.

    Between an exchange's close (plus the margin) and Yahoo publishing that
    session, record_decision refuses by design; the test skips then rather
    than reading a late Yahoo as a regression.
    """
    async with _server(live_assets) as client:
        result = await client.call_tool("record_decision", {**CALL, "ticker": ticker})
        if result.is_error:
            text = " ".join(getattr(part, "text", "") for part in result.content)
            if "isn't available" in text:
                pytest.skip(f"Yahoo hasn't published the newest session: {text}")
        assert not result.is_error, result.content
        entry = result.structured_content
        assert entry is not None
        stock = await _call(client, "get_candles", ticker=ticker, count=1)
        benchmark = await _call(client, "get_candles", ticker=BENCHMARK, count=1)

    date, *_, close, _volume = stock["rows"][-1]
    assert (entry["decision_session"], entry["close"]) == (date, close)
    assert entry["decision_session"] == _newest_closed_session(exchange).isoformat()
    bench_date, *_, bench_close, _volume = benchmark["rows"][-1]
    assert (entry["benchmark"]["session"], entry["benchmark"]["close"]) == (
        bench_date,
        bench_close,
    )
    assert entry["benchmark"]["session"] == _newest_closed_session("XAMS").isoformat()


@pytest.mark.parametrize(
    ("ticker", "exchange", "horizon"),
    [("ASML.AS", "XAMS", 20), ("AAPL", "XNYS", 20), ("AAPL", "XNYS", 60)],
)
async def test_should_end_the_horizon_on_the_exchange_s_own_sessions(
    live_assets: Path, ticker: str, exchange: str, horizon: int
) -> None:
    """@issue-119 @AC3 Should count the horizon in the exchange's sessions."""
    async with _server(live_assets) as client:
        entry = await _call(
            client, "record_decision", **{**CALL, "ticker": ticker, "horizon": horizon}
        )

    decided = dt.date.fromisoformat(entry["decision_session"])
    sessions = _sessions_after(exchange, decided, horizon)
    assert entry["horizon_ends"] == sessions[-1].isoformat()
    ends = dt.date.fromisoformat(entry["horizon_ends"])
    if exchange == "XNYS":
        skipped = [d for d in THANKSGIVING if decided < d <= ends]
        if skipped:
            # Counting weekdays alone would count Thanksgiving as a session,
            # so the server's answer must lie beyond that naive end.
            naive, counted = decided, 0
            while counted < horizon:
                naive += dt.timedelta(days=1)
                counted += naive.weekday() < 5
            assert ends > naive


async def test_should_append_one_line_per_call_and_never_rewrite(
    live_assets: Path,
) -> None:
    """@issue-119 @AC4 Should append one line per call and keep earlier lines."""
    journal = _journal(live_assets)
    async with _server(live_assets) as client:
        first = await _call(client, "record_decision", **CALL)
        after_first = journal.read_bytes()
        second = await _call(client, "record_decision", **CALL)
        after_second = journal.read_bytes()
        third = await _call(
            client, "record_decision", **{**CALL, "direction": "pass", "horizon": 5}
        )

    assert [first["id"], second["id"], third["id"]] == [1, 2, 3]
    content = journal.read_bytes()
    assert content.startswith(after_second)
    assert after_second.startswith(after_first)
    lines = content.decode().splitlines(keepends=True)
    assert len(lines) == 3
    assert all(line.endswith("\n") for line in lines)
    for line, entry in zip(lines, (first, second, third), strict=True):
        stored = json.loads(line)
        assert {key: stored[key] for key in entry} == entry


async def test_should_continue_the_ids_after_a_restart(live_assets: Path) -> None:
    """@issue-119 @AC5 Should keep the journal and continue its ids after a restart."""
    journal = _journal(live_assets)
    async with _server(live_assets) as client:
        await _call(client, "record_decision", **CALL)
        await _call(client, "record_decision", **CALL)
    before = journal.read_bytes()

    async with _server(live_assets) as client:
        entry = await _call(client, "record_decision", **CALL)

    assert entry["id"] == 3
    assert journal.read_bytes().startswith(before)
    assert len(journal.read_bytes().splitlines()) == 3


INVALID_CALLS = [
    pytest.param({"ticker": "NOPEZZZ"}, ["NOPEZZZ"], id="unknown-ticker"),
    pytest.param({"ticker": "BMW.DE"}, ["BMW.DE", "universe"], id="exchange"),
    pytest.param({"direction": "sell"}, ["direction", "buy", "pass"], id="direction"),
    pytest.param({"horizon": 4}, ["horizon", "5"], id="horizon-4"),
    pytest.param({"horizon": 61}, ["horizon", "60"], id="horizon-61"),
    pytest.param({"signals": []}, ["signals", "1"], id="no-signals"),
    pytest.param(
        {"signals": [{"name": "vibes", "stance": "for"}]},
        ["name", *SIGNAL_NAMES],
        id="signal-name",
    ),
    pytest.param(
        {"signals": [{"name": "rsi", "stance": "neutral"}]},
        ["stance", "for", "against"],
        id="stance",
    ),
    pytest.param(
        {
            "signals": [
                {"name": "trend", "stance": "for"},
                {"name": "trend", "stance": "against"},
            ]
        },
        ["signals", "trend"],
        id="signal-twice",
    ),
    pytest.param({"reasoning": "   "}, ["reasoning", "2,000"], id="empty-reasoning"),
    pytest.param(
        {"reasoning": "x" * 2001}, ["reasoning", "2,000"], id="reasoning-2001"
    ),
]


@pytest.mark.parametrize(("change", "words"), INVALID_CALLS)
async def test_should_reject_an_invalid_call_and_record_nothing(
    live_assets: Path, change: dict[str, Any], words: list[str]
) -> None:
    """@issue-119 @AC6 Should reject an invalid call, saying what's allowed."""
    journal = _journal(live_assets)
    async with _server(live_assets) as client:
        await _call(client, "record_decision", **CALL)
        before = journal.read_bytes()
        message = await _error(client, "record_decision", **{**CALL, **change})

    for word in words:
        assert word in message, (word, message)
    assert journal.read_bytes() == before


async def test_should_record_nothing_on_a_first_invalid_call(
    live_assets: Path,
) -> None:
    """@issue-119 @AC6 Should leave no journal behind when the first call is invalid."""
    async with _server(live_assets) as client:
        await _error(client, "record_decision", **{**CALL, "direction": "sell"})
        await _error(client, "record_decision", **{**CALL, "ticker": "NOPEZZZ"})

    assert not _journal(live_assets).exists()


async def test_should_list_the_valid_values_in_the_input_schema(
    live_assets: Path,
) -> None:
    """@issue-119 @AC6 Should list the valid names and stances in the schema."""
    async with _server(live_assets) as client:
        tools = (await client.list_tools()).tools
    schema = json.dumps(
        next(t for t in tools if t.name == "record_decision").input_schema
    )

    for value in [*SIGNAL_NAMES, "for", "against", "buy", "pass"]:
        assert f'"{value}"' in schema, value


@pytest.mark.parametrize("stale", ["ASML.AS", BENCHMARK])
async def test_should_refuse_when_a_newest_close_is_not_cached(
    live_assets: Path, stale: str
) -> None:
    """@issue-119 @AC7 Should refuse a call without the newest close, provider down."""
    async with _server(live_assets) as client:
        await _call(client, "record_decision", **CALL)
    journal = _journal(live_assets)
    before = journal.read_bytes()
    candles = live_assets / "stocks" / stale / "candles.csv"
    rows = candles.read_text().splitlines(keepends=True)
    candles.write_text("".join(rows[:-1]))

    async with _server(live_assets, PROVIDER_DOWN) as client:
        message = await _error(client, "record_decision", **CALL)

    assert stale in message
    assert "newest close" in message
    assert "try again later" in message.lower()
    assert journal.read_bytes() == before


async def test_should_record_with_the_provider_down_when_both_closes_are_cached(
    live_assets: Path,
) -> None:
    """@issue-119 @AC7 Should record as usual when both newest closes are cached."""
    async with _server(live_assets) as client:
        first = await _call(client, "record_decision", **CALL)

    async with _server(live_assets, PROVIDER_DOWN) as client:
        entry = await _call(client, "record_decision", **CALL)

    assert entry["id"] == first["id"] + 1
    assert entry["decision_session"] == first["decision_session"]
    assert entry["benchmark"] == first["benchmark"]


async def test_should_offer_no_tool_that_edits_or_removes_an_entry(
    live_assets: Path,
) -> None:
    """@issue-119 @AC8 Should offer the five read tools plus record_decision only."""
    async with _server(live_assets) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == sorted(
        [
            "get_candles",
            "get_quote",
            "get_indicators",
            "get_trend",
            "get_support_resistance",
            "record_decision",
        ]
    )
    record = next(tool for tool in tools if tool.name == "record_decision")
    assert record.annotations is not None
    assert record.annotations.destructive_hint is False


async def test_should_normalise_the_ticker_and_keep_the_reasoning_as_sent(
    live_assets: Path,
) -> None:
    """@issue-119 @AC1 @AC4 Should normalise the ticker and store multi-line text intact."""
    reasoning = "Trend up.\nTarget €55."
    async with _server(live_assets) as client:
        entry = await _call(
            client,
            "record_decision",
            **{**CALL, "ticker": " asml.as ", "reasoning": f"  {reasoning}  "},
        )

    assert entry["ticker"] == "ASML.AS"
    assert entry["reasoning"] == reasoning
    lines = _journal(live_assets).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["reasoning"] == reasoning


async def test_should_report_a_damaged_journal_without_rewriting_it(
    live_assets: Path,
) -> None:
    """@issue-119 @AC4 Should refuse to append to a damaged journal and leave it."""
    journal = _journal(live_assets)
    async with _server(live_assets) as client:
        await _call(client, "record_decision", **CALL)
        with journal.open("ab") as file:
            file.write(b"not json\n")
        damaged = journal.read_bytes()
        message = await _error(client, "record_decision", **CALL)

    assert "line 2" in message
    assert journal.read_bytes() == damaged
