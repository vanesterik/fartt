"""
Integration tests for `get_support_resistance` (#109), driven the way Claude
Code drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder holding the snapshot, its clock
pinned to the snapshot's moment and Yahoo out of reach (`snapshot.OFFLINE`).
Only the check that asks Yahoo for a ticker it doesn't list is live, and
marked `network`.

Each test's docstring starts with its tags: `@issue-109` and the acceptance
criteria it verifies. Expectations come from the issue's criteria. Zones
change with the market, so the live checks compare the output with itself,
with the cached sessions and with the other tools, never with fixed prices.
Scenarios that need a particular last session (a close at a new high, a close
inside a zone) cut a cached history short at a past date and keep the
provider unreachable, so the server serves that history as it stands. The
sweep over ten past sessions per ticker builds the same server in memory
instead of starting one over stdio for each.
"""

import csv
import json
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from fartt.journal import Journal
from fartt.server import build_server
from fartt.stock_cache import StockCache
from tests.fakes import FakeProvider
from tests.integration.risk import checked_in_risk_config, copy_risk_config
from tests.integration.snapshot import OFFLINE, SNAPSHOT_NOW

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
TEST_TICKERS = ["ASML.AS", "AAPL", "GOOG"]
# A proxy nobody listens on: every request to the data provider fails fast.
UNREACHABLE = "http://127.0.0.1:9"
PROVIDER_DOWN = {name: UNREACHABLE for name in ("HTTPS_PROXY", "https_proxy")}
ZONE_FIELDS = {
    "low",
    "high",
    "sources",
    "touches",
    "turns",
    "last_touch",
    "volume_share",
    "strength",
    "distance",
    "distance_atr",
}
SIGN_POINTS = {"holding", "breaking"}
SIZE_BUDGET = 3500
REACH_ATR = 5
# Past sessions whose close lies inside a zone the tool lists (found while
# exploring on 2026-11-04). The history up to them doesn't change, but the
# zones do when epic I retunes levels.py's thresholds: a date that no longer
# has its close in a listed zone skips its case, and wants a new date.
INSIDE_A_ZONE = [
    ("ASML.AS", "2026-05-19"),
    ("ASML.AS", "2026-09-14"),
    ("AAPL", "2026-10-22"),
]
# Past sessions that closed above every earlier high.
AT_A_NEW_HIGH = [
    ("ASML.AS", "2026-06-30"),
    ("AAPL", "2026-10-22"),
    ("GOOG", "2026-05-13"),
]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(
    assets_dir: Path,
    extra_env: dict[str, str] | None = None,
    *,
    live: bool = False,
) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {
        **os.environ,
        "FARTT_ASSETS_DIR": str(assets_dir),
        **({} if live else OFFLINE),
        **(extra_env or {}),
    }
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _levels_in_memory(assets_dir: Path, ticker: str) -> dict[str, Any]:
    """
    `get_support_resistance` from the same server, built in this process on
    `assets_dir` with the data provider unreachable: the tool and its output
    are the stdio server's, without starting a process.
    """
    unreachable = FakeProvider({}, {}, now=SNAPSHOT_NOW)
    unreachable.reachable = False
    cache = StockCache(unreachable, assets_dir, clock=lambda: SNAPSHOT_NOW)
    async with Client(
        build_server(cache, Journal(assets_dir / "journal"), checked_in_risk_config())
    ) as client:
        return await _call(client, "get_support_resistance", ticker=ticker)


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


def _cached(assets_dir: Path, ticker: str) -> list[dict[str, Any]]:
    path = assets_dir / "stocks" / ticker / "candles.csv"
    with path.open() as file:
        return [
            {
                "date": row["Date"],
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
            }
            for row in csv.DictReader(file)
        ]


def _cut_history(source: Path, target: Path, ticker: str, through: str) -> None:
    """
    Copy a ticker's cache, keeping the sessions up to `through`, and the
    checked-in risk config the server needs to start.
    """
    shutil.copytree(source / "stocks" / ticker, target / "stocks" / ticker)
    copy_risk_config(target)
    path = target / "stocks" / ticker / "candles.csv"
    header, *rows = path.read_text().splitlines(keepends=True)
    kept = [row for row in rows if row.split(",")[0] <= through]
    assert kept and kept[-1].split(",")[0] == through, f"{through} isn't cached"
    path.write_text(header + "".join(kept))


def _edge_atr(zone: dict[str, Any], last: float, atr: float, side: str) -> float:
    """How far the zone's nearest edge lies from the last close, in ATRs."""
    if side == "support":
        return max(0.0, last - zone["high"]) / atr
    return (zone["low"] - last) / atr


def _assert_zones_hold_the_criteria(
    levels: dict[str, Any], cached: list[dict[str, Any]]
) -> None:
    """AC2, AC3, AC4, AC5, AC7's shape and AC10, against the cached sessions."""
    last, atr, note = levels["last"], levels["atr"], levels["note"] or ""
    assert levels["based_on"] == cached[-1]["date"]
    assert last == pytest.approx(cached[-1]["close"])
    assert len(json.dumps(levels)) < SIZE_BUDGET  # AC10
    sessions = {candle["date"]: index for index, candle in enumerate(cached)}

    for side in ("support", "resistance"):
        zones = levels[side]
        assert len(zones) <= 3, side  # AC2
        edges = [_edge_atr(zone, last, atr, side) for zone in zones]
        assert edges == sorted(edges), (side, "not nearest first", edges)  # AC2
        for zone, edge in zip(zones, edges, strict=True):
            label = (side, zone["low"], zone["high"])
            assert set(zone) >= ZONE_FIELDS, label  # AC1
            # AC2
            if side == "support":
                assert zone["low"] <= last, label
            else:
                assert zone["low"] > last, label
            assert zone["low"] < zone["high"], label
            assert 2 <= zone["turns"] <= zone["touches"], label
            assert 0 <= zone["strength"] <= 100, label
            # AC3
            if edge > REACH_ATR:
                assert len(zones) == 1, (label, edge)
                assert "further than 5 ATR" in note, (label, note)
            # AC4
            assert zone["last_touch"] in sessions, label
            index = sessions[zone["last_touch"]]
            touch = cached[index]
            overlaps = touch["low"] <= zone["high"] and touch["high"] >= zone["low"]
            crossed = False
            if index > 0:
                before, after = cached[index - 1]["close"], touch["close"]
                crossed = (before > zone["high"] and after < zone["low"]) or (
                    before < zone["low"] and after > zone["high"]
                )
            assert overlaps or crossed, (label, touch)
            # AC5
            if zone["low"] <= last <= zone["high"]:
                assert side == "support" and zone is zones[0], label
                assert zone["distance"] == 0, label
                assert zone["distance_atr"] == 0, label
                assert "inside" in note, (label, note)

    # AC7
    assert bool(levels["signs"]) == bool(levels["support"]), levels["signs"]
    for sign in levels["signs"]:
        assert sign["points"] in SIGN_POINTS, sign
        assert sign["sign"]
        assert len(sign["detail"]) < 80, sign
    assert not [key for key in levels if "score" in key or "verdict" in key]


async def test_should_return_every_field(seeded_assets: Path) -> None:
    """@issue-109 @AC1 @AC10 Should return every field, within its size budget."""
    async with _server(seeded_assets) as client:
        levels = await _call(client, "get_support_resistance", ticker="ASML.AS")

    assert set(levels) >= {
        "ticker",
        "currency",
        "based_on",
        "last",
        "atr",
        "support",
        "resistance",
        "signs",
        "note",
        "is_current",
        "warning",
    }
    assert (levels["ticker"], levels["currency"]) == ("ASML.AS", "EUR")
    assert levels["is_current"] is True
    assert levels["warning"] is None
    assert levels["atr"] > 0
    zones = levels["support"] + levels["resistance"]
    assert zones
    for zone in zones:
        assert set(zone) >= ZONE_FIELDS, zone
    assert len(json.dumps(levels)) < SIZE_BUDGET


@pytest.mark.parametrize("ticker", TEST_TICKERS)
async def test_should_list_zones_that_hold_the_criteria(
    seeded_assets: Path, ticker: str
) -> None:
    """
    @issue-109 @AC2 @AC3 @AC4 @AC5 @AC7 @AC10 Should list a handful of zones
    per side, nearest first, each touched on a cached session.
    """
    async with _server(seeded_assets) as client:
        levels = await _call(client, "get_support_resistance", ticker=ticker)

    _assert_zones_hold_the_criteria(levels, _cached(seeded_assets, ticker))


@pytest.mark.parametrize("ticker", TEST_TICKERS)
async def test_should_agree_with_get_trend_and_get_indicators(
    seeded_assets: Path, ticker: str
) -> None:
    """@issue-109 @AC7 Should give trend and volume signs that the other tools agree with."""
    async with _server(seeded_assets) as client:
        levels = await _call(client, "get_support_resistance", ticker=ticker)
        trend = await _call(client, "get_trend", ticker=ticker)
        indicators = await _call(client, "get_indicators", ticker=ticker)

    if not levels["support"]:
        # A close at a new low has no support, so no signs: not a failure.
        pytest.skip(f"{ticker} has no support zone today, so no signs to check")
    signs = {sign["sign"]: sign["points"] for sign in levels["signs"]}
    for timeframe in ("daily", "weekly"):
        direction = trend[timeframe]["direction"]
        expected = {"up": "holding", "down": "breaking"}.get(direction)
        assert signs.get(f"{timeframe}_trend") == expected, (timeframe, direction)
    ratio = indicators["volume"]["down_day_ratio"]
    expected = None
    if ratio is not None and ratio < 0.8:
        expected = "holding"
    elif ratio is not None and ratio > 1.2:
        expected = "breaking"
    assert signs.get("volume_on_drop") == expected, ratio


@pytest.mark.parametrize(("ticker", "through"), INSIDE_A_ZONE)
async def test_should_list_a_zone_holding_the_last_close_first(
    seeded_assets: Path, ticker: str, through: str
) -> None:
    """@issue-109 @AC5 @AC2 @AC3 @AC4 Should list a zone holding the close first, at 0."""
    _cut_history(seeded_assets, seeded_assets / "cut", ticker, through)

    async with _server(seeded_assets / "cut", PROVIDER_DOWN) as client:
        levels = await _call(client, "get_support_resistance", ticker=ticker)

    assert levels["based_on"] == through
    holding = [
        zone
        for zone in levels["support"]
        if zone["low"] <= levels["last"] <= zone["high"]
    ]
    if not holding:
        pytest.skip(
            f"No listed zone holds {ticker}'s close on {through} any more "
            "(retuned thresholds?): pick a new date for INSIDE_A_ZONE"
        )
    first = levels["support"][0]
    assert first["low"] <= levels["last"] <= first["high"], first
    assert first["distance"] == 0
    assert "inside" in (levels["note"] or "")
    _assert_zones_hold_the_criteria(levels, _cached(seeded_assets / "cut", ticker))


@pytest.mark.parametrize(("ticker", "through"), AT_A_NEW_HIGH)
async def test_should_have_no_resistance_at_a_new_high(
    seeded_assets: Path, ticker: str, through: str
) -> None:
    """@issue-109 @AC6 @AC3 Should list no resistance when the close is a new high."""
    _cut_history(seeded_assets, seeded_assets / "cut", ticker, through)
    cached = _cached(seeded_assets / "cut", ticker)
    # Given the history ends with a close above every earlier high.
    assert cached[-1]["close"] > max(candle["high"] for candle in cached[:-1])

    async with _server(seeded_assets / "cut", PROVIDER_DOWN) as client:
        levels = await _call(client, "get_support_resistance", ticker=ticker)

    assert levels["resistance"] == []
    assert levels["support"]
    assert "no resistance from history above the price" in (levels["note"] or "")
    _assert_zones_hold_the_criteria(levels, cached)


SWEEP_SESSIONS = 120
SWEEP_STEP = 12


@pytest.mark.parametrize("ticker", TEST_TICKERS)
async def test_should_hold_the_criteria_on_past_sessions(
    seeded_assets: Path, ticker: str
) -> None:
    """
    @issue-109 @AC2 @AC3 @AC4 @AC5 @AC7 @AC10 Should hold the zone criteria
    with the history cut at each of ten past sessions, not only today's.
    """
    # Given a full history, cut at past sessions spread over the last 120.
    dates = [
        candle["date"]
        for candle in _cached(seeded_assets, ticker)[-SWEEP_SESSIONS::SWEEP_STEP]
    ]

    for date in dates:
        cut = seeded_assets / f"cut-{date}"
        _cut_history(seeded_assets, cut, ticker, date)
        # When the agent asks with only that history cached. In memory: ten
        # stdio servers would cost seconds, and stdio is covered above.
        levels = await _levels_in_memory(cut, ticker)
        # Then every listed zone holds the criteria against those sessions.
        _assert_zones_hold_the_criteria(levels, _cached(cut, ticker))


async def test_should_serve_a_stale_cache_when_the_provider_is_unreachable(
    seeded_assets: Path,
) -> None:
    """@issue-109 @AC8 Should serve zones from a stale cache when the provider is down."""
    async with _server(seeded_assets) as client:
        fresh = await _call(client, "get_support_resistance", ticker="ASML.AS")
    assert fresh["is_current"] is True

    # Given the cache is two sessions behind, and the provider unreachable.
    path = seeded_assets / "stocks" / "ASML.AS" / "candles.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-2]))
    behind = lines[-3].split(",")[0]

    async with _server(seeded_assets, PROVIDER_DOWN) as client:
        stale = await _call(client, "get_support_resistance", ticker="ASML.AS")

    assert stale["is_current"] is False
    assert stale["based_on"] == behind
    assert "2 sessions behind" in (stale["warning"] or "")
    assert stale["support"] or stale["resistance"]
    _assert_zones_hold_the_criteria(stale, _cached(seeded_assets, "ASML.AS"))


@pytest.mark.network
async def test_should_say_what_to_do_for_a_ticker_it_cannot_serve(
    empty_assets: Path,
) -> None:
    """@issue-109 @AC9 Should say what to do for a ticker it can't serve."""
    async with _server(empty_assets, live=True) as client:
        for ticker in ("ZZZQXQ", "VOD.L"):
            levels_error = await _error(client, "get_support_resistance", ticker=ticker)
            candles_error = await _error(client, "get_candles", ticker=ticker)
            assert "Yahoo" in levels_error or "Supported" in levels_error
            assert (
                levels_error.replace("get_support_resistance", "get_candles")
                == candles_error
            )


async def test_should_explain_a_history_too_short_for_zones(
    seeded_assets: Path,
) -> None:
    """@issue-109 @X Should explain, not fail, when fewer than 30 sessions are cached."""
    _cut_history(seeded_assets, seeded_assets / "cut", "GOOG", "2004-09-29")

    async with _server(seeded_assets / "cut", PROVIDER_DOWN) as client:
        levels = await _call(client, "get_support_resistance", ticker="GOOG")

    assert levels["support"] == levels["resistance"] == levels["signs"] == []
    assert "30 sessions" in (levels["note"] or "")
