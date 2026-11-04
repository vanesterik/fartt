"""
Integration tests for `get_trend` (#108), driven the way Claude Code drives
the server: started from `.mcp.json` over stdio, with `FARTT_ASSETS_DIR` set to
a temporary folder, against the live data provider. Marked `network`.

Each test's docstring starts with its tags: `@issue-108` and the acceptance
criteria it verifies. Expectations come from the issue's criteria, and the
live tickers' directions change with the market, so the checks compare the
direction with the evidence reported next to it, never with a fixed value.
"""

import csv
import datetime as dt
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
TEST_TICKERS = ["ASML.AS", "AAPL", "GOOG"]
# A proxy nobody listens on: every request to the data provider fails fast.
UNREACHABLE = "http://127.0.0.1:9"

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


def _expected_direction(timeframe: dict[str, Any]) -> str:
    """The AC2 rule, applied to the evidence the timeframe reports."""
    close = timeframe["close"]
    sma_50 = timeframe["sma_50"]["value"]
    sma_200 = timeframe["sma_200"]["value"]
    slope = timeframe["sma_50"]["slope"]
    if close > sma_50 > sma_200 and slope == "rising":
        return "up"
    if close < sma_50 < sma_200 and slope == "falling":
        return "down"
    return "sideways"


def _cached_closes(assets_dir: Path, ticker: str) -> list[tuple[dt.date, float]]:
    path = assets_dir / "stocks" / ticker / "candles.csv"
    with path.open() as file:
        return [
            (dt.date.fromisoformat(row["Date"]), float(row["Close"]))
            for row in csv.DictReader(file)
        ]


def _weekly_closes(
    daily: list[tuple[dt.date, float]], through: dt.date
) -> list[tuple[dt.date, float]]:
    """The last close of each ISO week, for the weeks up to `through`."""
    weeks: dict[tuple[int, int], tuple[dt.date, float]] = {}
    for day, close in daily:
        if day <= through:
            iso = day.isocalendar()
            weeks[(iso.year, iso.week)] = (day, close)
    return [weeks[key] for key in sorted(weeks)]


def _sma(closes: list[float], period: int) -> float:
    return round(sum(closes[-period:]) / period, 4)


async def test_should_return_both_timeframes_with_their_evidence(
    tmp_path: Path,
) -> None:
    """@issue-108 @AC1 @AC8 Should return both timeframes with their evidence."""
    async with _server(tmp_path) as client:
        trend = await _call(client, "get_trend", ticker="ASML.AS")

    assert set(trend) >= {
        "ticker",
        "currency",
        "daily",
        "weekly",
        "aligned",
        "note",
        "is_current",
        "warning",
    }
    assert (trend["ticker"], trend["currency"]) == ("ASML.AS", "EUR")
    assert isinstance(trend["aligned"], bool)
    assert trend["is_current"] is True
    for name in ("daily", "weekly"):
        timeframe = trend[name]
        assert timeframe["direction"] in {"up", "down", "sideways"}, name
        assert timeframe["basis"] == "sma_50_and_200", name
        dt.date.fromisoformat(timeframe["based_on"])
        assert timeframe["close"] > 0
        for average in ("sma_50", "sma_200"):
            block = timeframe[average]
            assert set(block) >= {"value", "price_vs", "distance", "slope", "days"}
            assert block["price_vs"] in {"above", "below"}
            assert block["slope"] in {"rising", "falling", "flat"}
        cross = timeframe["last_cross"]
        assert cross is None or (
            cross["kind"] in {"golden", "death"}
            and dt.date.fromisoformat(cross["date"])
            <= dt.date.fromisoformat(timeframe["based_on"])
        )
    assert len(json.dumps(trend)) < 2000


@pytest.mark.parametrize("ticker", TEST_TICKERS)
async def test_should_judge_the_direction_from_its_evidence(
    tmp_path: Path, ticker: str
) -> None:
    """@issue-108 @AC2 Should judge each timeframe's direction from its evidence."""
    async with _server(tmp_path) as client:
        trend = await _call(client, "get_trend", ticker=ticker)

    directions = []
    for name in ("daily", "weekly"):
        timeframe = trend[name]
        assert timeframe["direction"] == _expected_direction(timeframe), (
            name,
            timeframe,
        )
        directions.append(timeframe["direction"])
    assert trend["aligned"] is (directions in (["up", "up"], ["down", "down"]))
    assert len(json.dumps(trend)) < 2000


async def test_should_report_averages_that_agree_with_the_cached_closes(
    tmp_path: Path,
) -> None:
    """@issue-108 @AC2 Should report averages that agree with the cached closes."""
    async with _server(tmp_path) as client:
        trend = await _call(client, "get_trend", ticker="ASML.AS")

    daily = _cached_closes(tmp_path, "ASML.AS")
    closes = [close for _, close in daily]
    assert trend["daily"]["based_on"] == daily[-1][0].isoformat()
    assert trend["daily"]["sma_50"]["value"] == pytest.approx(_sma(closes, 50))
    assert trend["daily"]["sma_200"]["value"] == pytest.approx(_sma(closes, 200))

    weekly_based_on = dt.date.fromisoformat(trend["weekly"]["based_on"])
    weeks = _weekly_closes(daily, weekly_based_on)
    weekly = [close for _, close in weeks]
    assert weeks[-1][0] == weekly_based_on
    assert trend["weekly"]["close"] == pytest.approx(round(weekly[-1], 4))
    assert trend["weekly"]["sma_50"]["value"] == pytest.approx(_sma(weekly, 50))
    assert trend["weekly"]["sma_200"]["value"] == pytest.approx(_sma(weekly, 200))


@pytest.mark.parametrize("ticker", TEST_TICKERS)
async def test_should_match_get_indicators_daily_averages(
    tmp_path: Path, ticker: str
) -> None:
    """@issue-108 @AC3 Should match get_indicators' daily SMA 50 and SMA 200."""
    async with _server(tmp_path) as client:
        trend = await _call(client, "get_trend", ticker=ticker)
        indicators = await _call(client, "get_indicators", ticker=ticker)

    assert trend["daily"]["based_on"] == indicators["based_on"]
    assert trend["daily"]["sma_50"] == indicators["sma_50"]
    assert trend["daily"]["sma_200"] == indicators["sma_200"]


async def test_should_base_the_weekly_timeframe_on_closed_weeks_only(
    tmp_path: Path,
) -> None:
    """
    @issue-108 @AC4 Should base the weekly timeframe on closed weeks only.

    Against live data the note is checked only when this runs before the
    week's last session has closed; after Friday's close or over a weekend
    the week is complete and there's no note to check. The unit tests
    `test_an_unclosed_week_is_in_the_note` and
    `test_get_trend_returns_both_timeframes` check the note either way.
    """
    import exchange_calendars

    async with _server(tmp_path) as client:
        trend = await _call(client, "get_trend", ticker="ASML.AS")
        candles = await _call(client, "get_candles", ticker="ASML.AS", count=15)

    rows = [(dt.date.fromisoformat(row[0]), row[4]) for row in candles["rows"]]
    newest = dt.date.fromisoformat(trend["daily"]["based_on"])
    assert rows[-1][0] == newest
    monday = newest - dt.timedelta(days=newest.weekday())
    sessions = exchange_calendars.get_calendar("XAMS").sessions_in_range(
        monday, monday + dt.timedelta(days=6)
    )
    week_closed = newest >= sessions[-1].date()

    weekly = trend["weekly"]
    if week_closed:
        expected_day = newest
    else:
        expected_day = max(day for day, _ in rows if day < monday)
        assert "isn't complete" in (trend["note"] or "")
        assert expected_day.isoformat() in trend["note"]
    assert weekly["based_on"] == expected_day.isoformat()
    assert weekly["close"] == dict(rows)[expected_day]


async def test_should_rest_a_short_listing_on_the_weekly_sma_50(
    tmp_path: Path,
) -> None:
    """@issue-108 @AC5 Should rest a short listing's weekly trend on the SMA 50."""
    async with _server(tmp_path) as client:
        trend = await _call(client, "get_trend", ticker="ARM")

    weekly = trend["weekly"]
    assert weekly["sma_200"] is None
    assert weekly["last_cross"] is None
    assert weekly["basis"] == "sma_50_only"
    assert weekly["sma_50"] is not None

    closed_weeks = _weekly_closes(
        _cached_closes(tmp_path, "ARM"), dt.date.fromisoformat(weekly["based_on"])
    )
    assert len(closed_weeks) < 205
    note = trend["note"] or ""
    assert "sma_200" in note
    assert "205 weeks" in note
    assert f"{len(closed_weeks)} are cached" in note


async def test_should_serve_a_stale_cache_when_the_provider_is_unreachable(
    tmp_path: Path,
) -> None:
    """@issue-108 @AC6 Should serve a stale cache when the provider is down."""
    async with _server(tmp_path) as client:
        fresh = await _call(client, "get_trend", ticker="ASML.AS")
    assert fresh["is_current"] is True

    # Given the cache is two sessions behind, and the provider unreachable.
    path = tmp_path / "stocks" / "ASML.AS" / "candles.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-2]))
    behind = dt.date.fromisoformat(lines[-3].split(",")[0])
    proxy = {name: UNREACHABLE for name in ("HTTPS_PROXY", "https_proxy")}

    async with _server(tmp_path, proxy) as client:
        stale = await _call(client, "get_trend", ticker="ASML.AS")

    assert stale["is_current"] is False
    assert stale["daily"]["based_on"] == behind.isoformat()
    assert stale["daily"]["direction"] in {"up", "down", "sideways"}
    assert "2 sessions behind" in (stale["warning"] or "")


async def test_should_say_what_to_do_for_a_ticker_it_cannot_serve(
    tmp_path: Path,
) -> None:
    """@issue-108 @AC7 Should say what to do for a ticker it can't serve."""
    async with _server(tmp_path) as client:
        for ticker in ("ZZZQXQ", "VOD.L"):
            trend_error = await _error(client, "get_trend", ticker=ticker)
            candles_error = await _error(client, "get_candles", ticker=ticker)
            assert "Yahoo" in trend_error or "Supported" in trend_error
            assert trend_error.replace("get_trend", "get_candles") == candles_error
