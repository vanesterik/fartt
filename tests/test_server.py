import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from fartt.data import (
    Candle,
    Dividend,
    History,
    Listing,
    Quote,
    exchange_calendar,
    session_after,
)
from fartt.journal import SIGNALS, Decision, DecisionDraft, Journal, Signal
from fartt.server import build_server
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from tests.fakes import FakeProvider, decision_draft, stock_candle

ROOT = Path(__file__).resolve().parents[1]
ASML = Listing("ASML.AS", "ASML Holding N.V.", "EUR", "XAMS")
AAPL = Listing("AAPL", "Apple Inc.", "USD", "XNYS")
IWDA = Listing("IWDA.AS", "iShares Core MSCI World UCITS ETF USD (Acc)", "EUR", "XAMS")
# Euronext Amsterdam and NYSE have both closed (plus margin) for 2026-10-29.
AFTER_CLOSE = datetime(2026, 10, 29, 21, 0, tzinfo=UTC)
AFTER_CLOSE_27 = datetime(2026, 10, 27, 21, 0, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def sessions(exchange: str, count: int, last: date = date(2026, 10, 29)) -> list[date]:
    """The `count` sessions of `exchange` up to and including `last`."""
    window = exchange_calendar(exchange).sessions_window(last.isoformat(), -count)
    return [session.date() for session in window]


def history(days: list[date], first: float = 100.0) -> History:
    return History([stock_candle(day, first + i) for i, day in enumerate(days)], [], [])


def provider(asml_days: list[date], now: datetime = AFTER_CLOSE) -> FakeProvider:
    return FakeProvider(
        histories={
            "ASML.AS": history(asml_days),
            "AAPL": history(sessions("XNYS", 3)),
            "IWDA.AS": history(sessions("XAMS", 3), first=50.0),
        },
        listings={"ASML.AS": ASML, "AAPL": AAPL, "IWDA.AS": IWDA},
        now=now,
    )


def cache_for(fake: FakeProvider, assets_dir: Path) -> StockCache:
    return StockCache(fake, assets_dir, clock=lambda: fake.now)


def journal_of(cache: StockCache) -> Journal:
    """The journal beside the cache's stocks/ folder, where main() keeps it."""
    return Journal(cache.directory("AAPL").parents[1] / "journal")


async def call(
    cache: StockCache, arguments: dict[str, Any], tool: str = "get_candles"
) -> Any:
    async with Client(build_server(cache, journal_of(cache))) as client:
        return await client.call_tool(tool, arguments)


@pytest.mark.anyio
async def test_get_candles_returns_the_newest_closed_sessions(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await call(cache, {"ticker": " asml.as ", "count": 2})

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["ticker"] == "ASML.AS"
    assert (content["name"], content["currency"], content["exchange"]) == (
        "ASML Holding N.V.",
        "EUR",
        "XAMS",
    )
    assert content["columns"] == ["date", "open", "high", "low", "close", "volume"]
    assert content["rows"] == [
        ["2026-10-28", 100.5, 102.0, 100.0, 101.0, 1000.0],
        ["2026-10-29", 101.5, 103.0, 101.0, 102.0, 1000.0],
    ]
    assert content["is_current"] is True
    assert content["warning"] is None
    assert content["gaps"] == []
    assert content["rebuilt"] is None


@pytest.mark.anyio
async def test_count_defaults_to_60_and_is_capped_at_200(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 61)), tmp_path)

    default = await call(cache, {"ticker": "ASML.AS"})
    too_many = await call(cache, {"ticker": "ASML.AS", "count": 201})

    assert len(default.structured_content["rows"]) == 60
    assert too_many.is_error
    assert "count" in too_many.content[0].text


@pytest.mark.anyio
async def test_count_beyond_the_history_returns_what_is_cached(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await call(cache, {"ticker": "ASML.AS", "count": 50})

    assert not result.is_error, result.content
    assert len(result.structured_content["rows"]) == 3


@pytest.mark.anyio
async def test_first_call_reports_the_fill_and_later_calls_do_not(
    tmp_path: Path,
) -> None:
    cache = cache_for(provider(sessions("XAMS", 5)), tmp_path)

    first = await call(cache, {"ticker": "ASML.AS"})
    second = await call(cache, {"ticker": "ASML.AS"})

    assert first.structured_content["new_candles"] == 5
    assert second.structured_content["new_candles"] == 0


@pytest.mark.anyio
async def test_a_rebuild_is_reported(tmp_path: Path) -> None:
    days = sessions("XAMS", 3)
    fake = provider(days, now=AFTER_CLOSE_27)
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    fake.now = AFTER_CLOSE
    amended = fake.histories["ASML.AS"].candles
    amended[0] = amended[0]._replace(close=90.0)

    content = (await call(cache, {"ticker": "ASML.AS"})).structured_content

    assert content["rebuilt"] == "last candle amended"


@pytest.mark.anyio
async def test_gaps_are_reported_in_a_warning(tmp_path: Path) -> None:
    days = [d for d in sessions("XAMS", 4) if d != date(2026, 10, 27)]
    cache = cache_for(provider(days), tmp_path)

    content = (await call(cache, {"ticker": "ASML.AS"})).structured_content

    assert content["gaps"] == ["2026-10-27"]
    assert "2026-10-27" in content["warning"]
    assert content["is_current"] is True


@pytest.mark.anyio
async def test_a_session_the_provider_has_not_published_is_not_current(
    tmp_path: Path,
) -> None:
    days = sessions("XAMS", 3, last=date(2026, 10, 28))
    cache = cache_for(provider(days), tmp_path)

    content = (await call(cache, {"ticker": "ASML.AS"})).structured_content

    assert content["is_current"] is False
    assert "2026-10-29" in content["warning"]


@pytest.mark.anyio
async def test_an_unreachable_provider_serves_the_cache_with_a_warning(
    tmp_path: Path,
) -> None:
    fake = provider(sessions("XAMS", 5), now=AFTER_CLOSE_27)
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    fake.now = AFTER_CLOSE
    fake.reachable = False

    result = await call(cache, {"ticker": "ASML.AS", "count": 2})

    assert not result.is_error, result.content
    content = result.structured_content
    assert [row[0] for row in content["rows"]] == ["2026-10-26", "2026-10-27"]
    assert content["is_current"] is False
    assert "2 sessions behind" in content["warning"]
    assert content["new_candles"] == 0


@pytest.mark.anyio
async def test_an_unreachable_provider_and_no_cache_is_a_tool_error(
    tmp_path: Path,
) -> None:
    fake = provider(sessions("XAMS", 3))
    fake.reachable = False

    result = await call(cache_for(fake, tmp_path), {"ticker": "ASML.AS"})

    assert result.is_error
    assert "unreachable" in result.content[0].text


@pytest.mark.anyio
async def test_a_ticker_without_closed_sessions_is_a_tool_error(
    tmp_path: Path,
) -> None:
    # Listed today: the provider answers, but has no closed session yet.
    result = await call(cache_for(provider([]), tmp_path), {"ticker": "ASML.AS"})

    assert result.is_error
    text = result.content[0].text
    assert "no closed sessions" in text
    assert "unreachable" not in text


@pytest.mark.anyio
async def test_a_corrupt_cache_is_a_tool_error_for_the_operator(
    tmp_path: Path,
) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    cache.update("ASML.AS")
    candles = cache.directory("ASML.AS") / "candles.csv"
    candles.write_text(candles.read_text().replace("102.0", "x"))

    result = await call(cache, {"ticker": "ASML.AS"})

    assert result.is_error
    text = result.content[0].text
    assert "operator" in text
    assert "ASML.AS" in text


@pytest.mark.anyio
async def test_an_unknown_ticker_is_a_tool_error_with_the_format(
    tmp_path: Path,
) -> None:
    result = await call(cache_for(provider([]), tmp_path), {"ticker": "NOSUCH"})

    assert result.is_error
    text = result.content[0].text
    assert "NOSUCH" in text
    assert "ASML.AS" in text


@pytest.mark.anyio
async def test_an_unsupported_exchange_is_a_tool_error(tmp_path: Path) -> None:
    result = await call(cache_for(provider([]), tmp_path), {"ticker": "VOD.L"})

    assert result.is_error
    assert ".AS" in result.content[0].text


@pytest.mark.anyio
async def test_a_ticker_that_names_no_exchange_is_a_tool_error(
    tmp_path: Path,
) -> None:
    result = await call(cache_for(provider([]), tmp_path), {"ticker": ".."})

    assert result.is_error
    assert not (tmp_path / "stocks").exists()


@pytest.mark.anyio
async def test_an_unwritable_cache_is_a_tool_error_for_the_operator(
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    cache = cache_for(provider(sessions("XAMS", 3)), blocked)

    result = await call(cache, {"ticker": "ASML.AS"})

    assert result.is_error
    text = result.content[0].text
    assert "operator" in text
    assert "not-a-directory" in text


@pytest.mark.anyio
async def test_the_tools_are_listed(tmp_path: Path) -> None:
    cache = cache_for(provider([]), tmp_path)
    async with Client(build_server(cache, journal_of(cache))) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == [
        "get_candles",
        "get_decisions",
        "get_indicators",
        "get_quote",
        "get_support_resistance",
        "get_trend",
        "record_decision",
    ]


# Euronext Amsterdam trades 08:00-16:30 UTC in winter time.
DURING_ASML_SESSION = datetime(2026, 10, 29, 10, 0, tzinfo=UTC)


def quote(
    bid: float | None = 101.9,
    ask: float | None = 102.1,
    volume: float | None = 500.0,
    time: datetime = datetime(2026, 10, 29, 9, 45, tzinfo=UTC),
) -> Quote:
    return Quote(102.0, bid, ask, volume, time, 15)


def quoting(
    days: list[date],
    volumes: list[float],
    now: datetime = DURING_ASML_SESSION,
    asml_quote: Quote | None = None,
) -> FakeProvider:
    fake = provider([], now=now)
    candles = [stock_candle(d, volume=v) for d, v in zip(days, volumes, strict=True)]
    fake.histories["ASML.AS"] = History(candles, [], [])
    fake.quotes["ASML.AS"] = asml_quote or quote()
    return fake


async def get_quote(fake: FakeProvider, tmp_path: Path, ticker: str = "ASML.AS") -> Any:
    return await call(cache_for(fake, tmp_path), {"ticker": ticker}, "get_quote")


@pytest.mark.anyio
async def test_get_quote_during_the_session(tmp_path: Path) -> None:
    days = sessions("XAMS", 25, last=date(2026, 10, 28))
    volumes = [1000.0 + 10 * i for i in range(25)]

    result = await get_quote(quoting(days, volumes), tmp_path)

    assert not result.is_error, result.content
    content = result.structured_content
    average = sum(volumes[-20:]) / 20
    assert (content["ticker"], content["currency"]) == ("ASML.AS", "EUR")
    assert (content["last"], content["bid"], content["ask"]) == (102.0, 101.9, 102.1)
    assert content["spread"] == 0.001961  # 0.2 / 102, to 6 decimals
    assert content["volume"] == 500.0
    assert content["average_volume_20d"] == average
    assert content["volume_ratio"] == 0.4367  # 500 / 1145, to 4 decimals
    assert content["market_open"] is True
    assert content["next_open"] is None
    assert content["quote_time"] == "2026-10-29T09:45:00+00:00"
    assert content["delay_minutes"] == 15
    assert content["note"] is None


@pytest.mark.anyio
async def test_get_quote_rounds_its_prices(tmp_path: Path) -> None:
    # Yahoo's prices are 32-bit floats: 1412.8 arrives as 1412.800048828125.
    raw = Quote(
        1412.800048828125,
        1412.199951171875,
        1413.4000244140625,
        500.0,
        datetime(2026, 10, 29, 9, 45, tzinfo=UTC),
        15,
    )
    days = sessions("XAMS", 3, last=date(2026, 10, 28))

    result = await get_quote(quoting(days, [1000.0] * 3, asml_quote=raw), tmp_path)

    content = result.structured_content
    assert (content["last"], content["bid"], content["ask"]) == (1412.8, 1412.2, 1413.4)
    assert content["spread"] == 0.000849  # 1.2 / 1412.8, to 6 decimals


@pytest.mark.anyio
async def test_get_quote_outside_the_session(tmp_path: Path) -> None:
    days = sessions("XAMS", 25)
    fake = quoting(days, [1000.0] * 25, now=AFTER_CLOSE)

    content = (await get_quote(fake, tmp_path)).structured_content

    assert content["market_open"] is False
    assert content["next_open"] == "2026-10-30T08:00:00+00:00"


@pytest.mark.anyio
async def test_today_is_not_in_its_own_average(tmp_path: Path) -> None:
    # After the close the cache holds today's session too.
    days = sessions("XAMS", 21)
    after_close = quote(time=datetime(2026, 10, 29, 16, 30, tzinfo=UTC))
    fake = quoting(days, [1000.0] * 20 + [10_000.0], AFTER_CLOSE, after_close)

    content = (await get_quote(fake, tmp_path)).structured_content

    assert content["average_volume_20d"] == pytest.approx(1000.0)


@pytest.mark.anyio
async def test_a_short_history_averages_what_is_cached(tmp_path: Path) -> None:
    days = sessions("XAMS", 5, last=date(2026, 10, 28))
    volumes = [100.0, 200.0, 300.0, 400.0, 500.0]

    content = (await get_quote(quoting(days, volumes), tmp_path)).structured_content

    assert content["average_volume_20d"] == pytest.approx(300.0)
    assert "5 sessions" in content["note"]


@pytest.mark.anyio
async def test_a_missing_bid_or_ask_makes_the_spread_unknown(tmp_path: Path) -> None:
    days = sessions("XAMS", 20, last=date(2026, 10, 28))
    fake = quoting(days, [1000.0] * 20, asml_quote=quote(bid=None))

    content = (await get_quote(fake, tmp_path)).structured_content

    assert (content["bid"], content["ask"], content["spread"]) == (None, None, None)
    assert "spread is unknown" in content["note"]


@pytest.mark.anyio
async def test_a_crossed_quote_makes_the_spread_unknown(tmp_path: Path) -> None:
    days = sessions("XAMS", 20, last=date(2026, 10, 28))
    fake = quoting(days, [1000.0] * 20, asml_quote=quote(bid=102.2, ask=102.1))

    content = (await get_quote(fake, tmp_path)).structured_content

    assert (content["bid"], content["ask"], content["spread"]) == (None, None, None)
    assert "crossed" in content["note"]


@pytest.mark.anyio
async def test_a_locked_quote_makes_the_spread_unknown(tmp_path: Path) -> None:
    days = sessions("XAMS", 20, last=date(2026, 10, 28))
    fake = quoting(days, [1000.0] * 20, asml_quote=quote(bid=102.0, ask=102.0))

    content = (await get_quote(fake, tmp_path)).structured_content

    assert (content["bid"], content["ask"], content["spread"]) == (None, None, None)
    assert "locked" in content["note"]


@pytest.mark.anyio
async def test_a_stale_quote_says_so_and_averages_before_it(tmp_path: Path) -> None:
    # A suspended stock: the quote is weeks older than the cached sessions.
    days = sessions("XAMS", 45, last=date(2026, 10, 28))
    volumes = [1000.0] * 20 + [5000.0] * 25
    struck = days[20]
    old = quote(time=datetime(struck.year, struck.month, struck.day, 9, 45, tzinfo=UTC))
    fake = quoting(days, volumes, asml_quote=old)

    content = (await get_quote(fake, tmp_path)).structured_content

    assert content["average_volume_20d"] == pytest.approx(1000.0)
    assert days[20].isoformat() in content["note"]
    assert "stale" in content["note"]


@pytest.mark.anyio
async def test_get_quote_with_the_provider_down_is_a_tool_error(
    tmp_path: Path,
) -> None:
    fake = quoting(sessions("XAMS", 3), [1000.0] * 3)
    fake.reachable = False

    result = await get_quote(fake, tmp_path)

    assert result.is_error
    assert "unreachable" in result.content[0].text


@pytest.mark.anyio
async def test_get_quote_of_an_unknown_ticker_is_a_tool_error(tmp_path: Path) -> None:
    result = await get_quote(quoting([], []), tmp_path, ticker="NOSUCH")

    assert result.is_error
    text = result.content[0].text
    assert "NOSUCH" in text
    assert "ASML.AS" in text


@pytest.mark.anyio
async def test_the_volume_ratio_says_it_is_not_normalised(tmp_path: Path) -> None:
    cache = cache_for(provider([]), tmp_path)
    async with Client(build_server(cache, journal_of(cache))) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    schema = tools["get_quote"].output_schema
    assert schema is not None
    assert "time of day" in schema["properties"]["volume_ratio"]["description"]


async def get_indicators(
    fake: FakeProvider, tmp_path: Path, ticker: str = "ASML.AS"
) -> Any:
    return await call(cache_for(fake, tmp_path), {"ticker": ticker}, "get_indicators")


@pytest.mark.anyio
async def test_get_indicators_returns_every_block(tmp_path: Path) -> None:
    result = await get_indicators(provider(sessions("XAMS", 250)), tmp_path)

    assert not result.is_error, result.content
    content = result.structured_content
    assert (content["ticker"], content["currency"]) == ("ASML.AS", "EUR")
    assert content["based_on"] == "2026-10-29"
    assert content["close"] == 349.0
    for block in ["sma_20", "sma_50", "sma_200", "ema_20", "rsi", "macd", "atr"]:
        assert content[block] is not None, block
    assert content["volume"]["ratio"] == 1.0
    assert content["note"] is None
    assert content["is_current"] is True
    assert content["warning"] is None


@pytest.mark.anyio
async def test_get_indicators_stays_within_its_size_budget(tmp_path: Path) -> None:
    result = await get_indicators(provider(sessions("XAMS", 250)), tmp_path)

    assert not result.is_error, result.content
    assert len(json.dumps(result.structured_content)) < 2000


@pytest.mark.anyio
async def test_get_indicators_on_a_short_history_says_what_is_missing(
    tmp_path: Path,
) -> None:
    result = await get_indicators(provider(sessions("XAMS", 30)), tmp_path)

    content = result.structured_content
    assert content["sma_200"] is None
    assert "sma_200 needs 205 sessions" in content["note"]


@pytest.mark.anyio
async def test_get_indicators_leaves_ex_dividend_sessions_out(tmp_path: Path) -> None:
    days = sessions("XAMS", 30)
    # Closes alternate 101, 100, ...; the heavy down day is an ex-dividend date.
    volumes = [5000.0 if i == 27 else 1000.0 for i in range(30)]
    candles = [
        stock_candle(day, 101.0 - i % 2, volumes[i]) for i, day in enumerate(days)
    ]
    fake = provider([])
    fake.histories["ASML.AS"] = History(candles, [Dividend(days[27], 1.0)], [])

    result = await get_indicators(fake, tmp_path)

    assert result.structured_content["volume"]["down_day_ratio"] == 1.0


@pytest.mark.anyio
async def test_get_indicators_with_the_provider_down_serves_the_cache(
    tmp_path: Path,
) -> None:
    fake = provider(sessions("XAMS", 30), now=AFTER_CLOSE_27)
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    fake.now = AFTER_CLOSE
    fake.reachable = False

    result = await call(cache, {"ticker": "ASML.AS"}, "get_indicators")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["based_on"] == "2026-10-27"
    assert content["is_current"] is False
    assert "2 sessions behind" in content["warning"]


@pytest.mark.anyio
async def test_get_indicators_of_an_unknown_ticker_is_a_tool_error(
    tmp_path: Path,
) -> None:
    result = await get_indicators(provider([]), tmp_path, ticker="NOSUCH")

    assert result.is_error
    assert "NOSUCH" in result.content[0].text


async def get_trend(fake: FakeProvider, tmp_path: Path, ticker: str = "ASML.AS") -> Any:
    return await call(cache_for(fake, tmp_path), {"ticker": ticker}, "get_trend")


TIMEFRAME_FIELDS = {
    "direction",
    "basis",
    "based_on",
    "close",
    "sma_50",
    "sma_200",
    "last_cross",
}


@pytest.mark.anyio
async def test_get_trend_returns_both_timeframes(tmp_path: Path) -> None:
    result = await get_trend(provider(sessions("XAMS", 1100)), tmp_path)

    assert not result.is_error, result.content
    content = result.structured_content
    assert set(content) == {
        "ticker",
        "currency",
        "daily",
        "weekly",
        "aligned",
        "note",
        "is_current",
        "warning",
    }
    assert (content["ticker"], content["currency"]) == ("ASML.AS", "EUR")
    assert set(content["daily"]) == TIMEFRAME_FIELDS
    assert set(content["weekly"]) == TIMEFRAME_FIELDS
    assert content["daily"]["based_on"] == "2026-10-29"
    # The week of Thursday the 29th hasn't closed: Friday the 30th is to come.
    assert content["weekly"]["based_on"] == "2026-10-23"
    for timeframe in ("daily", "weekly"):
        assert content[timeframe]["direction"] == "up"
        assert content[timeframe]["basis"] == "sma_50_and_200"
        assert content[timeframe]["sma_200"] is not None
    assert content["aligned"] is True
    assert "week ending 2026-10-23" in content["note"]
    assert content["is_current"] is True
    assert content["warning"] is None


@pytest.mark.anyio
async def test_get_trend_daily_averages_match_get_indicators(tmp_path: Path) -> None:
    fake = provider(sessions("XAMS", 1100))

    trend = (await get_trend(fake, tmp_path)).structured_content
    indicators = (await get_indicators(fake, tmp_path)).structured_content

    assert trend["daily"]["sma_50"] == indicators["sma_50"]
    assert trend["daily"]["sma_200"] == indicators["sma_200"]


@pytest.mark.anyio
@pytest.mark.parametrize("count", [1100, 300, 40])
async def test_get_trend_stays_within_its_size_budget(
    tmp_path: Path, count: int
) -> None:
    result = await get_trend(provider(sessions("XAMS", count)), tmp_path)

    assert not result.is_error, result.content
    assert len(json.dumps(result.structured_content)) < 2000


@pytest.mark.anyio
async def test_get_trend_on_a_short_history_says_what_is_missing(
    tmp_path: Path,
) -> None:
    result = await get_trend(provider(sessions("XAMS", 300)), tmp_path)

    weekly = result.structured_content["weekly"]
    assert weekly["sma_200"] is None
    assert weekly["last_cross"] is None
    assert weekly["basis"] == "sma_50_only"
    assert "weekly sma_200 needs 205 weeks" in result.structured_content["note"]


@pytest.mark.anyio
async def test_get_trend_with_the_provider_down_serves_the_cache(
    tmp_path: Path,
) -> None:
    fake = provider(sessions("XAMS", 300), now=AFTER_CLOSE_27)
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    fake.now = AFTER_CLOSE
    fake.reachable = False

    result = await call(cache, {"ticker": "ASML.AS"}, "get_trend")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["daily"]["based_on"] == "2026-10-27"
    assert content["is_current"] is False
    assert "2 sessions behind" in content["warning"]


@pytest.mark.anyio
async def test_get_trend_of_an_unknown_ticker_is_a_tool_error(tmp_path: Path) -> None:
    result = await get_trend(provider([]), tmp_path, ticker="NOSUCH")

    assert result.is_error
    assert "NOSUCH" in result.content[0].text


@pytest.mark.anyio
async def test_get_trend_of_an_unsupported_exchange_is_a_tool_error(
    tmp_path: Path,
) -> None:
    result = await get_trend(provider([]), tmp_path, ticker="VOD.L")

    assert result.is_error
    assert ".AS" in result.content[0].text


def test_the_instructions_say_indicators_are_evidence() -> None:
    assert "evidence, not signals" in server_module.INSTRUCTIONS
    assert "aren't summed" in server_module.INSTRUCTIONS


async def get_support_resistance(
    fake: FakeProvider, tmp_path: Path, ticker: str = "ASML.AS"
) -> Any:
    return await call(
        cache_for(fake, tmp_path), {"ticker": ticker}, "get_support_resistance"
    )


def oscillating(count: int = 600) -> list[float]:
    """Closes turning at 90 and 110 in steps of 2, ending on 100 on the way up."""
    cycle = [90.0 + 2 * i for i in range(10)] + [110.0 - 2 * i for i in range(10)]
    closes = cycle * (count // len(cycle) + 2)
    end = len(closes) - 14  # the last close is cycle[5]: 100 on the way up
    return closes[end - count : end]


def oscillating_provider(
    count: int = 600,
    now: datetime = AFTER_CLOSE,
    dividends: list[Dividend] | None = None,
) -> FakeProvider:
    fake = provider([], now=now)
    days = sessions("XAMS", count)
    candles = [
        stock_candle(day, close)
        for day, close in zip(days, oscillating(count), strict=True)
    ]
    fake.histories["ASML.AS"] = History(candles, dividends or [], [])
    return fake


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


@pytest.mark.anyio
async def test_get_support_resistance_returns_every_field(tmp_path: Path) -> None:
    result = await get_support_resistance(oscillating_provider(), tmp_path)

    assert not result.is_error, result.content
    content = result.structured_content
    assert set(content) == {
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
    assert (content["ticker"], content["currency"]) == ("ASML.AS", "EUR")
    assert content["based_on"] == "2026-10-29"
    assert content["last"] == 100.0
    assert content["support"] and content["resistance"]
    for zone in content["support"] + content["resistance"]:
        assert set(zone) == ZONE_FIELDS
    assert content["is_current"] is True
    assert content["warning"] is None


@pytest.mark.anyio
async def test_get_support_resistance_signs_agree_with_get_trend(
    tmp_path: Path,
) -> None:
    fake = oscillating_provider()

    levels = (await get_support_resistance(fake, tmp_path)).structured_content
    trend = (await get_trend(fake, tmp_path)).structured_content

    signs = {sign["sign"]: sign["points"] for sign in levels["signs"]}
    expected = {"up": "holding", "down": "breaking", "sideways": None}
    assert signs.get("daily_trend") == expected[trend["daily"]["direction"]]
    assert signs.get("weekly_trend") == expected[trend["weekly"]["direction"]]


@pytest.mark.anyio
@pytest.mark.parametrize("count", [600, 120, 29])
async def test_get_support_resistance_stays_within_its_size_budget(
    tmp_path: Path, count: int
) -> None:
    result = await get_support_resistance(oscillating_provider(count), tmp_path)

    assert not result.is_error, result.content
    assert len(json.dumps(result.structured_content)) < 3500


@pytest.mark.anyio
async def test_get_support_resistance_leaves_ex_dividend_sessions_out(
    tmp_path: Path,
) -> None:
    # The last dip, 5 sessions before the end, drops to 87 on its ex-dividend
    # date: below the zone at the bottom of the range (about 88.25-89.75).
    days = sessions("XAMS", 600)
    fake = oscillating_provider(dividends=[Dividend(days[-6], 3.0)])
    candles = fake.histories["ASML.AS"].candles
    candles[-6] = stock_candle(days[-6], 87.0)

    content = (await get_support_resistance(fake, tmp_path)).structured_content

    assert content["support"][0]["low"] > 87.0
    assert "recent_break" not in {sign["sign"] for sign in content["signs"]}


@pytest.mark.anyio
async def test_get_support_resistance_with_the_provider_down_serves_the_cache(
    tmp_path: Path,
) -> None:
    fake = oscillating_provider(now=AFTER_CLOSE_27)
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    fake.now = AFTER_CLOSE
    fake.reachable = False

    result = await call(cache, {"ticker": "ASML.AS"}, "get_support_resistance")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["based_on"] == "2026-10-27"
    assert content["support"] or content["resistance"]
    assert content["is_current"] is False
    assert "2 sessions behind" in content["warning"]


@pytest.mark.anyio
async def test_get_support_resistance_of_an_unknown_ticker_is_a_tool_error(
    tmp_path: Path,
) -> None:
    result = await get_support_resistance(provider([]), tmp_path, ticker="NOSUCH")

    assert result.is_error
    assert "NOSUCH" in result.content[0].text


@pytest.mark.anyio
async def test_get_support_resistance_of_an_unsupported_exchange_is_a_tool_error(
    tmp_path: Path,
) -> None:
    result = await get_support_resistance(provider([]), tmp_path, ticker="VOD.L")

    assert result.is_error
    assert ".AS" in result.content[0].text


class FakeServer:
    def __init__(self, cache: StockCache, journal: Journal) -> None:
        self.cache = cache
        self.journal = journal
        self.ran = False

    def run(self) -> None:
        self.ran = True


def run_main(
    monkeypatch: pytest.MonkeyPatch, env_file: Path | None = None
) -> FakeServer:
    built: list[FakeServer] = []

    def fake_build_server(cache: StockCache, journal: Journal) -> FakeServer:
        built.append(FakeServer(cache, journal))
        return built[-1]

    monkeypatch.setattr(server_module, "build_server", fake_build_server)
    monkeypatch.setattr(server_module, "YFinanceProvider", lambda: provider([]))
    # A developer's own .env must not decide what these tests see.
    monkeypatch.setattr(
        server_module, "ENV_FILE", env_file or ROOT / "no-such-dir" / ".env"
    )
    server_module.main()
    assert built[0].ran
    return built[0]


def test_main_uses_the_repository_assets_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(server_module.ASSETS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    server = run_main(monkeypatch)

    assert server_module.DEFAULT_ASSETS_DIR == ROOT / "assets"
    assert server.cache.directory("AAPL") == ROOT / "assets" / "stocks" / "AAPL"


def test_main_reads_the_assets_dir_from_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(server_module.ASSETS_DIR_ENV, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"FARTT_ASSETS_DIR={tmp_path / 'cache'}\n")

    server = run_main(monkeypatch, env_file)

    assert server.cache.directory("AAPL") == tmp_path / "cache" / "stocks" / "AAPL"


def test_main_keeps_the_journal_in_the_assets_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(server_module.ASSETS_DIR_ENV, str(tmp_path / "assets"))

    server = run_main(monkeypatch)

    assert server.journal.path == tmp_path / "assets" / "journal" / "decisions.jsonl"


def test_a_relative_assets_dir_is_relative_to_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(server_module.ASSETS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("FARTT_ASSETS_DIR=data\n")

    server = run_main(monkeypatch, env_file)

    assert server.cache.directory("AAPL") == ROOT / "data" / "stocks" / "AAPL"


def test_the_environment_wins_over_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(server_module.ASSETS_DIR_ENV, str(tmp_path / "from-env"))
    env_file = tmp_path / ".env"
    env_file.write_text(f"FARTT_ASSETS_DIR={tmp_path / 'from-file'}\n")

    server = run_main(monkeypatch, env_file)

    assert server.cache.directory("AAPL") == tmp_path / "from-env" / "stocks" / "AAPL"


def test_the_example_env_file_names_the_assets_dir() -> None:
    assert server_module.ASSETS_DIR_ENV in (ROOT / ".env.example").read_text()


@pytest.mark.anyio
async def test_get_candles_rounds_prices_to_4_decimals(tmp_path: Path) -> None:
    # Yahoo's prices are 32-bit floats: 1666.2 arrives as 1666.199951171875.
    day = sessions("XAMS", 1)[0]
    candle = Candle(day, 1660.0, 1666.199951171875, 1639.0, 1656.199951171875, 1.0)
    fake = provider([])
    fake.histories["ASML.AS"] = History([candle], [], [])

    result = await call(cache_for(fake, tmp_path), {"ticker": "ASML.AS"})

    assert result.structured_content["rows"] == [
        [day.isoformat(), 1660.0, 1666.2, 1639.0, 1656.2, 1.0]
    ]


SIGNALS_SENT = [
    {"name": "trend", "stance": "for", "note": "up, aligned"},
    {"name": "rsi", "stance": "against", "note": None},
]


def decision(**changes: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "ticker": "ASML.AS",
        "direction": "buy",
        "horizon": 20,
        "reasoning": "Trend up, support holding.",
        "signals": SIGNALS_SENT,
    }
    return arguments | changes


async def record(cache: StockCache, **changes: Any) -> Any:
    return await call(cache, decision(**changes), "record_decision")


def journal_path(cache: StockCache) -> Path:
    return journal_of(cache).path


@pytest.mark.anyio
async def test_record_decision_returns_the_stored_entry(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await record(cache)

    assert not result.is_error, result.content
    assert result.structured_content == {
        "id": 1,
        "recorded_at": "2026-10-29T21:00:00Z",
        "ticker": "ASML.AS",
        "currency": "EUR",
        "exchange": "XAMS",
        "direction": "buy",
        "horizon": 20,
        "reasoning": "Trend up, support holding.",
        "signals": SIGNALS_SENT,
        "decision_session": "2026-10-29",
        "close": 102.0,
        "horizon_ends": session_after("XAMS", date(2026, 10, 29), 20).isoformat(),
        "benchmark": {
            "ticker": "IWDA.AS",
            "currency": "EUR",
            "session": "2026-10-29",
            "close": 52.0,
        },
    }


@pytest.mark.anyio
async def test_the_entry_is_appended_to_the_journal(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    first = await record(cache)
    second = await record(cache)

    assert [first.structured_content["id"], second.structured_content["id"]] == [1, 2]
    stored = [d.model_dump(mode="json") for d in journal_of(cache).decisions()]
    assert stored == [first.structured_content, second.structured_content]


@pytest.mark.anyio
async def test_the_reasoning_is_stored_stripped(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await record(cache, reasoning="  Trend up.\n")

    assert result.structured_content["reasoning"] == "Trend up."


@pytest.mark.anyio
async def test_the_benchmark_session_follows_its_own_exchange(tmp_path: Path) -> None:
    # Boxing Day 2025: NYSE trades, Euronext is closed (and was on the 25th).
    boxing_day = datetime(2025, 12, 26, 23, 0, tzinfo=UTC)
    fake = provider([], now=boxing_day)
    fake.histories["AAPL"] = history(sessions("XNYS", 3, last=date(2025, 12, 26)))
    fake.histories["IWDA.AS"] = history(sessions("XAMS", 3, last=date(2025, 12, 24)))

    result = await record(cache_for(fake, tmp_path), ticker="AAPL")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["decision_session"] == "2025-12-26"
    assert content["benchmark"]["session"] == "2025-12-24"


@pytest.mark.anyio
async def test_a_ticker_is_normalised(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await record(cache, ticker=" asml.as ")

    assert result.structured_content["ticker"] == "ASML.AS"
    assert journal_of(cache).decisions()[0].ticker == "ASML.AS"


@pytest.mark.anyio
async def test_a_call_on_the_benchmark_itself(tmp_path: Path) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await record(cache, ticker="IWDA.AS")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["close"] == content["benchmark"]["close"] == 52.0


@pytest.mark.anyio
async def test_a_call_during_the_session_uses_the_previous_close(
    tmp_path: Path,
) -> None:
    during = datetime(2026, 10, 30, 10, 0, tzinfo=UTC)
    fake = provider(sessions("XAMS", 3, last=date(2026, 10, 30)), now=during)
    fake.histories["IWDA.AS"] = history(sessions("XAMS", 3, last=date(2026, 10, 30)))

    result = await record(cache_for(fake, tmp_path))

    assert not result.is_error, result.content
    assert result.structured_content["decision_session"] == "2026-10-29"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"direction": "sell"}, "direction"),
        ({"horizon": 4}, "horizon"),
        ({"horizon": 61}, "horizon"),
        ({"signals": []}, "signals"),
        ({"signals": [{"name": "vibes", "stance": "for"}]}, "name"),
        ({"signals": [{"name": "trend", "stance": "neutral"}]}, "stance"),
        (
            {"signals": [{"name": "trend", "stance": "for"}] * 2},
            "trend",
        ),
        ({"signals": [{"name": "trend", "stance": "for", "note": "x" * 201}]}, "note"),
        ({"reasoning": "   "}, "reasoning"),
        ({"reasoning": "x" * 2001}, "reasoning"),
        ({"ticker": "NOPE"}, "NOPE"),
        ({"ticker": "BMW.DE"}, ".AS"),
    ],
)
async def test_invalid_arguments_are_tool_errors_and_record_nothing(
    tmp_path: Path, changes: dict[str, Any], named: str
) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await record(cache, **changes)

    assert result.is_error
    assert named in result.content[0].text
    assert not journal_path(cache).exists()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("changes", "allowed"),
    [
        ({"direction": "sell"}, ["buy", "pass"]),
        ({"signals": [{"name": "vibes", "stance": "for"}]}, ["trend", "spread"]),
        ({"signals": [{"name": "trend", "stance": "neutral"}]}, ["for", "against"]),
        ({"horizon": 61}, ["60"]),
        ({"reasoning": "x" * 2001}, ["2,000"]),
    ],
)
async def test_an_invalid_value_error_says_what_is_allowed(
    tmp_path: Path, changes: dict[str, Any], allowed: list[str]
) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    text = (await record(cache, **changes)).content[0].text

    for value in allowed:
        assert value in text


@pytest.mark.anyio
async def test_the_valid_names_are_in_the_input_schema(tmp_path: Path) -> None:
    cache = cache_for(provider([]), tmp_path)
    async with Client(build_server(cache, journal_of(cache))) as client:
        tools = (await client.list_tools()).tools

    schema = json.dumps(
        next(t for t in tools if t.name == "record_decision").input_schema
    )
    for value in [
        "trend",
        "support_resistance",
        "moving_averages",
        "rsi",
        "macd",
        "volume",
        "atr",
        "spread",
        "for",
        "against",
        "buy",
        "pass",
    ]:
        assert f'"{value}"' in schema


@pytest.mark.anyio
@pytest.mark.parametrize("behind", ["ASML.AS", "IWDA.AS"])
async def test_a_newest_close_that_is_not_cached_records_nothing(
    tmp_path: Path, behind: str
) -> None:
    fake = provider(sessions("XAMS", 3))
    cache = cache_for(fake, tmp_path)
    fake.now = AFTER_CLOSE_27
    cache.update(behind)
    fake.now = AFTER_CLOSE
    cache.update("IWDA.AS" if behind == "ASML.AS" else "ASML.AS")
    fake.reachable = False

    result = await record(cache)

    assert result.is_error
    text = result.content[0].text
    assert f"{behind}'s newest close (2026-10-29)" in text
    assert "Try again later" in text
    assert not journal_path(cache).exists()


@pytest.mark.anyio
async def test_an_unpublished_newest_close_records_nothing(tmp_path: Path) -> None:
    fake = provider(sessions("XAMS", 3))
    fake.histories["IWDA.AS"] = history(sessions("XAMS", 3, last=date(2026, 10, 28)))

    result = await record(cache_for(fake, tmp_path))

    assert result.is_error
    assert "IWDA.AS's newest close (2026-10-29)" in result.content[0].text
    assert not (tmp_path / "journal" / "decisions.jsonl").exists()


@pytest.mark.anyio
async def test_cached_newest_closes_record_with_the_provider_down(
    tmp_path: Path,
) -> None:
    fake = provider(sessions("XAMS", 3))
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    cache.update("IWDA.AS")
    fake.reachable = False

    result = await record(cache)

    assert not result.is_error, result.content
    assert len(journal_of(cache).decisions()) == 1


@pytest.mark.anyio
async def test_a_damaged_journal_is_a_tool_error_for_the_operator(
    tmp_path: Path,
) -> None:
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    await record(cache)
    with journal_path(cache).open("ab") as file:
        file.write(b"not json\n")
    before = journal_path(cache).read_bytes()

    result = await record(cache)

    assert result.is_error
    text = result.content[0].text
    assert "journal" in text
    assert "line 2" in text
    assert "operator" in text
    assert journal_path(cache).read_bytes() == before


@pytest.mark.anyio
async def test_a_horizon_the_calendar_cannot_place_is_a_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # exchange_calendars raises ValueError for a day that isn't a session or
    # an offset past the calendar's end; neither happens with cached candles.
    def unplaceable(exchange: str, day: date, count: int) -> date:
        raise ValueError(f"{day} is not a session")

    monkeypatch.setattr(server_module, "session_after", unplaceable)
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await record(cache)

    assert result.is_error
    text = result.content[0].text
    assert "horizon" in text
    assert "is not a session" in text
    assert "operator" in text
    assert not journal_path(cache).exists()


def test_the_instructions_say_to_record_every_call() -> None:
    assert "record_decision" in server_module.INSTRUCTIONS


def seed(cache: StockCache, *drafts: DecisionDraft) -> list[Decision]:
    journal = journal_of(cache)
    return [journal.append(draft) for draft in drafts]


async def get_decisions(cache: StockCache, **arguments: Any) -> Any:
    return await call(cache, arguments, "get_decisions")


def without_read_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """An entry as get_decisions returns it, less what it adds to the stored call."""
    return {k: v for k, v in entry.items() if k not in ("status", "horizon_passed")}


def three_calls(cache: StockCache) -> list[Decision]:
    return seed(
        cache,
        decision_draft(),
        decision_draft(ticker="AAPL", currency="USD", exchange="XNYS"),
        decision_draft(direction="pass"),
    )


@pytest.mark.anyio
async def test_get_decisions_returns_open_calls_newest_first(tmp_path: Path) -> None:
    # @AC1
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    seeded = three_calls(cache)

    result = await get_decisions(cache, status="open")

    assert not result.is_error, result.content
    content = result.structured_content
    assert [entry["id"] for entry in content["decisions"]] == [3, 2, 1]
    assert content["total"] == 3
    assert content["note"] is None
    assert [without_read_fields(entry) for entry in content["decisions"]] == [
        decision.model_dump(mode="json") for decision in reversed(seeded)
    ]
    assert all(entry["status"] == "open" for entry in content["decisions"])


@pytest.mark.anyio
async def test_get_decisions_filters_by_ticker(tmp_path: Path) -> None:
    # @AC2
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    three_calls(cache)

    by_ticker = await get_decisions(cache, ticker=" asml.as ")
    with_status = await get_decisions(cache, ticker="ASML.AS", status="open")

    for result in (by_ticker, with_status):
        content = result.structured_content
        assert [entry["id"] for entry in content["decisions"]] == [3, 1]
        assert content["total"] == 2


@pytest.mark.anyio
@pytest.mark.parametrize("ticker", ["AS ML", "../x"])
async def test_get_decisions_rejects_what_cant_be_a_ticker(
    tmp_path: Path, ticker: str
) -> None:
    # @AC2
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    three_calls(cache)

    result = await get_decisions(cache, ticker=ticker)

    assert result.is_error
    assert "isn't a ticker" in result.content[0].text


@pytest.mark.anyio
async def test_get_decisions_reads_calls_recorded_before_a_restart(
    tmp_path: Path,
) -> None:
    # @AC3
    recorded = await record(cache_for(provider(sessions("XAMS", 3)), tmp_path))
    restarted = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await get_decisions(restarted)

    entries = result.structured_content["decisions"]
    assert [without_read_fields(entry) for entry in entries] == [
        recorded.structured_content
    ]


@pytest.mark.anyio
async def test_get_decisions_on_an_empty_journal_is_an_empty_list(
    tmp_path: Path,
) -> None:
    # @AC5
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await get_decisions(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["decisions"] == []
    assert content["total"] == 0
    assert "No calls" in content["note"]
    assert not journal_path(cache).parent.exists()


@pytest.mark.anyio
async def test_get_decisions_with_no_match_says_which_filters(
    tmp_path: Path,
) -> None:
    # @AC5
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    seed(cache, decision_draft())

    result = await get_decisions(cache, ticker="aapl", status="open")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["decisions"] == []
    assert content["total"] == 0
    assert "AAPL" in content["note"]
    assert "open" in content["note"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("now", "passed"),
    [
        # Euronext Amsterdam closes at 16:30 UTC in winter time; the session
        # counts as closed 30 minutes later.
        (datetime(2026, 10, 29, 16, 59, tzinfo=UTC), False),
        (datetime(2026, 10, 29, 17, 0, tzinfo=UTC), True),
    ],
)
async def test_horizon_passed_turns_true_at_the_close_of_the_last_session(
    tmp_path: Path, now: datetime, passed: bool
) -> None:
    # @AC6
    cache = cache_for(provider(sessions("XAMS", 3), now=now), tmp_path)
    seed(cache, decision_draft(horizon_ends=date(2026, 10, 29)))

    result = await get_decisions(cache)

    assert result.structured_content["decisions"][0]["horizon_passed"] is passed


@pytest.mark.anyio
async def test_horizon_passed_follows_the_stocks_own_exchange(
    tmp_path: Path,
) -> None:
    # @AC6: at 17:00 UTC Amsterdam has closed 2026-10-29, New York hasn't.
    now = datetime(2026, 10, 29, 17, 0, tzinfo=UTC)
    cache = cache_for(provider(sessions("XAMS", 3), now=now), tmp_path)
    seed(
        cache,
        decision_draft(horizon_ends=date(2026, 10, 29)),
        decision_draft(
            ticker="AAPL",
            currency="USD",
            exchange="XNYS",
            horizon_ends=date(2026, 10, 29),
        ),
    )

    result = await get_decisions(cache)

    passed = {
        entry["ticker"]: entry["horizon_passed"]
        for entry in result.structured_content["decisions"]
    }
    assert passed == {"ASML.AS": True, "AAPL": False}


@pytest.mark.anyio
async def test_get_decisions_defaults_to_all_and_nothing_is_scored_yet(
    tmp_path: Path,
) -> None:
    # @AC7
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    seed(cache, decision_draft(), decision_draft())

    every = await get_decisions(cache)
    scored = await get_decisions(cache, status="scored")

    assert [entry["id"] for entry in every.structured_content["decisions"]] == [2, 1]
    assert every.structured_content["total"] == 2
    assert scored.structured_content["decisions"] == []
    assert scored.structured_content["total"] == 0
    assert "scored" in scored.structured_content["note"]


@pytest.mark.anyio
async def test_a_damaged_journal_is_a_tool_error_for_get_decisions(
    tmp_path: Path,
) -> None:
    # @AC8
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    seed(cache, decision_draft())
    with journal_path(cache).open("ab") as file:
        file.write(b"not json\n")
    before = journal_path(cache).read_bytes()

    result = await get_decisions(cache)

    assert result.is_error
    text = result.content[0].text
    assert str(journal_path(cache)) in text
    assert "line 2" in text
    assert "operator" in text
    assert journal_path(cache).read_bytes() == before


@pytest.mark.anyio
async def test_an_unknown_exchange_in_the_journal_is_a_tool_error_for_get_decisions(
    tmp_path: Path,
) -> None:
    # @AC8: a hand-edited line names a calendar that doesn't exist.
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    seed(cache, decision_draft())
    line = journal_path(cache).read_text().replace('"XAMS"', '"XXXX"')
    with journal_path(cache).open("a") as file:
        file.write(line)
    before = journal_path(cache).read_bytes()

    result = await get_decisions(cache)

    assert result.is_error
    text = result.content[0].text
    assert str(journal_path(cache)) in text
    assert "line 2" in text
    assert journal_path(cache).read_bytes() == before


@pytest.mark.anyio
async def test_get_decisions_never_calls_the_provider(tmp_path: Path) -> None:
    fake = provider(sessions("XAMS", 3))
    fake.reachable = False
    cache = cache_for(fake, tmp_path)
    seed(cache, decision_draft())

    result = await get_decisions(cache)

    assert not result.is_error, result.content
    assert len(result.structured_content["decisions"]) == 1
    assert fake.history_calls == []


@pytest.mark.anyio
async def test_get_decisions_returns_the_newest_up_to_the_limit(
    tmp_path: Path,
) -> None:
    # @AC4
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    seed(cache, *(decision_draft() for _ in range(12)))

    default = (await get_decisions(cache)).structured_content
    three = (await get_decisions(cache, limit=3)).structured_content

    assert [entry["id"] for entry in default["decisions"]] == list(range(12, 2, -1))
    assert default["total"] == 12
    for word in ("10", "12", "ticker", "status"):
        assert word in default["note"]
    assert [entry["id"] for entry in three["decisions"]] == [12, 11, 10]
    assert three["total"] == 12
    assert "3" in three["note"]
    assert "12" in three["note"]


@pytest.mark.anyio
@pytest.mark.parametrize("limit", [0, 11])
async def test_get_decisions_rejects_a_limit_outside_1_to_10(
    tmp_path: Path, limit: int
) -> None:
    # @AC4
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)

    result = await get_decisions(cache, limit=limit)

    assert result.is_error
    assert "limit" in result.content[0].text


@pytest.mark.anyio
async def test_ten_of_the_largest_calls_fit_under_60000_characters(
    tmp_path: Path,
) -> None:
    # @AC9
    cache = cache_for(provider(sessions("XAMS", 3)), tmp_path)
    largest = decision_draft(
        reasoning="x" * 2000,
        signals=[Signal(name=name, stance="for", note="n" * 200) for name in SIGNALS],
    )
    seed(cache, *(largest for _ in range(12)))

    result = await get_decisions(cache)

    assert len(result.structured_content["decisions"]) == 10
    assert len(json.dumps(result.structured_content)) < 60_000
