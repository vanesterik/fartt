"""
Integration tests for a listed ticker without daily history (#116), driven
the way Claude Code drives the server: started from `.mcp.json` over stdio,
with `FARTT_ASSETS_DIR` set to a temporary folder. Marked `network`: they're
about what Yahoo answers for a listing without history, except the check
that a current cached ticker is served as it is, which runs offline on the
snapshot (`snapshot.OFFLINE`). The cases that refill a cache from Yahoo keep
the clock pinned to the snapshot's moment (`snapshot.PINNED`), so Yahoo fills
it up to that moment and the snapshot's last session stays the newest.

GOOG.AS is the live example: Yahoo lists it (first trade May 2021) but its
history holds no closed session. The cases that need another first trade
date (a real new listing, none at all) build the same server in this process
on the real `YFinanceProvider`, with only GOOG.AS's
`firstTradeDateMilliseconds` altered in Yahoo's answer.

Each test's docstring starts with its tags: `@issue-116` and the acceptance
criteria it verifies. Expectations come from the issue's criteria.
"""

import csv
import json
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yfinance
from mcp import Client, StdioServerParameters

from fartt.data import (
    YFinanceProvider,
    exchange_calendar,
    newest_closed_session,
    session_after,
)
from fartt.journal import Journal
from fartt.server import build_server
from fartt.stock_cache import StockCache
from tests.integration.snapshot import OFFLINE, PINNED

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
NO_HISTORY = "GOOG.AS"
UNKNOWN = "NOSUCHTICKERXYZ.AS"
CACHED = "ASML.AS"
HISTORY_TOOLS: list[tuple[str, dict[str, Any]]] = [
    ("get_candles", {}),
    ("get_indicators", {}),
    ("get_trend", {}),
    ("get_support_resistance", {}),
    (
        "record_decision",
        {
            "direction": "buy",
            "horizon": 20,
            "reasoning": "Integration check of a ticker without history.",
            "signals": [{"name": "trend", "stance": "for"}],
        },
    ),
]
READ_TOOLS = ["get_candles", "get_indicators", "get_trend", "get_support_resistance"]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(
    assets_dir: Path, settings: dict[str, str] | None = None
) -> AsyncIterator[Client]:
    """The server on Yahoo and the real clock, unless `settings` say otherwise."""
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir), **(settings or {})}
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


class _AlteredFirstTrade:
    """A yfinance Ticker whose `info` carries another first trade date."""

    def __init__(self, real: Any, first_trade_ms: int | None) -> None:
        self._real = real
        self._first_trade_ms = first_trade_ms

    @property
    def info(self) -> dict[str, Any]:
        info = dict(self._real.info)
        info.pop("firstTradeDateMilliseconds", None)
        if self._first_trade_ms is not None:
            info["firstTradeDateMilliseconds"] = self._first_trade_ms
        return info

    def history(self, **kwargs: Any) -> pd.DataFrame:
        return self._real.history(**kwargs)


@asynccontextmanager
async def _server_with_first_trade(
    assets_dir: Path, first_trade_ms: int | None
) -> AsyncIterator[Client]:
    def ticker(symbol: str) -> Any:
        real = yfinance.Ticker(symbol)
        if symbol != NO_HISTORY:
            return real
        return _AlteredFirstTrade(real, first_trade_ms)

    cache = StockCache(YFinanceProvider(ticker_factory=ticker), assets_dir)
    async with Client(build_server(cache, Journal(assets_dir / "journal"))) as client:
        yield client


def _session_open_ms(day: Any) -> int:
    opens = exchange_calendar("XAMS").session_open(pd.Timestamp(day))
    return int(opens.timestamp() * 1000)


def _text(result: Any) -> str:
    return " ".join(getattr(part, "text", "") for part in result.content)


async def _errors(client: Client, ticker: str) -> dict[str, str]:
    """Each history tool's error on `ticker`; a tool that succeeds fails the test."""
    errors: dict[str, str] = {}
    for tool, arguments in HISTORY_TOOLS:
        result = await client.call_tool(tool, {"ticker": ticker, **arguments})
        assert result.is_error, f"{tool} succeeded: {result.structured_content}"
        errors[tool] = _text(result)
    return errors


def _assert_no_history_message(errors: dict[str, str]) -> None:
    for tool, message in errors.items():
        assert f"lists {NO_HISTORY}" in message, (tool, message)
        assert "no daily price history" in message, (tool, message)
        assert "can't be analysed" in message, (tool, message)
        assert "listed today" not in message, (tool, message)
        assert "try again" not in message.lower(), (tool, message)


def _assert_no_folder(assets_dir: Path, ticker: str) -> None:
    assert not (assets_dir / "stocks" / ticker).exists()


def _candle_lines(assets_dir: Path, ticker: str) -> list[list[str]]:
    with (assets_dir / "stocks" / ticker / "candles.csv").open() as file:
        return list(csv.reader(file))


async def _skip_if_goog_as_has_history(client: Client) -> None:
    result = await client.call_tool("get_candles", {"ticker": NO_HISTORY})
    if not result.is_error:
        pytest.skip(
            f"{NO_HISTORY} has daily history now; #116's tests need another "
            "listing without history"
        )


@pytest.mark.network
async def test_should_say_a_listed_ticker_has_no_history(tmp_path: Path) -> None:
    """@issue-116 @AC1 @AC4 Should say a listed ticker has no history, and cache nothing."""
    async with _server(tmp_path) as client:
        await _skip_if_goog_as_has_history(client)
        errors = await _errors(client, NO_HISTORY)

    _assert_no_history_message(errors)
    _assert_no_folder(tmp_path, NO_HISTORY)


@pytest.mark.network
@pytest.mark.parametrize(
    ("first_trade", "expected"),
    [
        pytest.param(
            "next_session",
            "no closed sessions for GOOG.AS yet, for instance because it listed "
            "today. Try again after its first session closes.",
            id="AC2-first-session-not-closed",
        ),
        pytest.param(
            None,
            "no closed sessions for GOOG.AS yet. If it listed today, try again "
            "after its first session closes; otherwise it has no daily history "
            "and can't be analysed.",
            id="AC3-no-first-trade-date",
        ),
    ],
)
async def test_should_word_the_error_from_the_first_trade_date(
    tmp_path: Path, first_trade: str | None, expected: str
) -> None:
    """@issue-116 @AC2 @AC3 @AC4 Should word the error from the first trade date."""
    newest = newest_closed_session("XAMS", datetime.now(UTC))
    first_trade_ms = (
        _session_open_ms(session_after("XAMS", newest, 1))
        if first_trade == "next_session"
        else None
    )

    async with _server_with_first_trade(tmp_path, first_trade_ms) as client:
        await _skip_if_goog_as_has_history(client)
        errors = await _errors(client, NO_HISTORY)

    for tool, message in errors.items():
        assert expected in message, (tool, message)
        if first_trade == "next_session":
            assert "can't be analysed" not in message, (tool, message)
    _assert_no_folder(tmp_path, NO_HISTORY)


@pytest.mark.network
async def test_should_count_a_first_trade_on_the_newest_closed_session_as_history(
    tmp_path: Path,
) -> None:
    """@issue-116 @AC1 Should treat a first trade on the newest closed session as traded."""
    newest = newest_closed_session("XAMS", datetime.now(UTC))

    async with _server_with_first_trade(tmp_path, _session_open_ms(newest)) as client:
        await _skip_if_goog_as_has_history(client)
        errors = await _errors(client, NO_HISTORY)

    _assert_no_history_message(errors)


@pytest.mark.network
async def test_should_leave_no_folder_for_an_unknown_ticker(tmp_path: Path) -> None:
    """@issue-116 @AC4 Should leave no folder for a ticker the provider doesn't know."""
    async with _server(tmp_path) as client:
        for tool in ("get_candles", "get_quote"):
            result = await client.call_tool(tool, {"ticker": UNKNOWN})
            assert result.is_error, result.structured_content
            assert f"doesn't know {UNKNOWN}" in _text(result)

    _assert_no_folder(tmp_path, UNKNOWN)


@pytest.mark.parametrize(
    "behind",
    [
        pytest.param(0, id="current"),
        pytest.param(3, id="three-sessions-behind", marks=pytest.mark.network),
    ],
)
async def test_should_keep_a_cache_with_a_listing_from_before_the_change(
    seeded_assets: Path, behind: int
) -> None:
    """@issue-116 @AC5 Should keep serving a cache whose listing.json has no first trade date."""
    directory = seeded_assets / "stocks" / CACHED
    listing = json.loads((directory / "listing.json").read_text())
    listing.pop("first_trade", None)
    (directory / "listing.json").write_text(json.dumps(listing))
    full = _candle_lines(seeded_assets, CACHED)
    if behind:
        with (directory / "candles.csv").open("w", newline="") as file:
            csv.writer(file).writerows(full[:-behind])

    # Three sessions behind, it refills from Yahoo up to the pinned clock.
    async with _server(seeded_assets, PINNED if behind else OFFLINE) as client:
        for tool in READ_TOOLS:
            result = await client.call_tool(tool, {"ticker": CACHED})
            assert not result.is_error, (tool, _text(result))
            assert result.structured_content["ticker"] == CACHED
        candles = await client.call_tool("get_candles", {"ticker": CACHED, "count": 3})

    content = candles.structured_content
    assert content["is_current"] is True
    assert content["rebuilt"] is None
    assert content["rows"][-1][0] == full[-1][0]
    after = _candle_lines(seeded_assets, CACHED)
    assert len(after) >= len(full)
    assert after[:2] == full[:2]


@pytest.mark.network
async def test_should_not_serve_a_header_only_leftover_as_history(
    seeded_assets: Path,
) -> None:
    """@issue-116 @AC5 Should report a no-history leftover and refill a real one."""
    stocks = seeded_assets / "stocks"
    for ticker, name in ((NO_HISTORY, "LS 1x Alphabet Tracker ETP"), (CACHED, "ASML")):
        shutil.rmtree(stocks / ticker, ignore_errors=True)
        leftover = stocks / ticker
        leftover.mkdir()
        (leftover / ".lock").touch()
        (leftover / "candles.csv").write_text("Date,Open,High,Low,Close,Volume\n")
        (leftover / "events.csv").write_text("Date,Type,Value\n")
        (leftover / "listing.json").write_text(
            json.dumps(
                {"ticker": ticker, "name": name, "currency": "EUR", "exchange": "XAMS"}
            )
        )

    async with _server(seeded_assets, PINNED) as client:
        await _skip_if_goog_as_has_history(client)
        errors = await _errors(client, NO_HISTORY)
        refilled = await client.call_tool("get_candles", {"ticker": CACHED, "count": 2})

    _assert_no_history_message(errors)
    assert not refilled.is_error, _text(refilled)
    assert refilled.structured_content["new_candles"] > 1000
    assert len(_candle_lines(seeded_assets, CACHED)) > 1000


@pytest.mark.network
async def test_should_quote_a_listed_ticker_without_history(tmp_path: Path) -> None:
    """@issue-116 @AC6 Should still quote a listed ticker without daily history."""
    async with _server(tmp_path) as client:
        await _skip_if_goog_as_has_history(client)
        result = await client.call_tool("get_quote", {"ticker": NO_HISTORY})

    assert not result.is_error, _text(result)
    quote = result.structured_content
    assert quote["ticker"] == NO_HISTORY
    assert quote["currency"] == "EUR"
    assert quote["last"] > 0
    assert quote["average_volume_20d"] is None
    assert quote["volume_ratio"] is None
    _assert_no_folder(tmp_path, NO_HISTORY)
