import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pytest
from mcp import Client

from fartt.data import Candle, Dividend, History, Listing, Quote
from fartt.server import build_server
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from tests.fakes import FakeProvider, stock_candle

ROOT = Path(__file__).resolve().parents[1]
ASML = Listing("ASML.AS", "ASML Holding N.V.", "EUR", "XAMS")
AAPL = Listing("AAPL", "Apple Inc.", "USD", "XNYS")
# Euronext Amsterdam and NYSE have both closed (plus margin) for 2026-10-29.
AFTER_CLOSE = datetime(2026, 10, 29, 21, 0, tzinfo=UTC)
AFTER_CLOSE_27 = datetime(2026, 10, 27, 21, 0, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def sessions(exchange: str, count: int, last: date = date(2026, 10, 29)) -> list[date]:
    """The `count` sessions of `exchange` up to and including `last`."""
    window = xcals.get_calendar(exchange).sessions_window(last.isoformat(), -count)
    return [session.date() for session in window]


def history(days: list[date]) -> History:
    return History([stock_candle(day, 100.0 + i) for i, day in enumerate(days)], [], [])


def provider(asml_days: list[date], now: datetime = AFTER_CLOSE) -> FakeProvider:
    return FakeProvider(
        histories={
            "ASML.AS": history(asml_days),
            "AAPL": history(sessions("XNYS", 3)),
        },
        listings={"ASML.AS": ASML, "AAPL": AAPL},
        now=now,
    )


def cache_for(fake: FakeProvider, assets_dir: Path) -> StockCache:
    return StockCache(fake, assets_dir, clock=lambda: fake.now)


async def call(
    cache: StockCache, arguments: dict[str, Any], tool: str = "get_candles"
) -> Any:
    async with Client(build_server(cache)) as client:
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
    async with Client(build_server(cache_for(provider([]), tmp_path))) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == [
        "get_candles",
        "get_indicators",
        "get_quote",
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
    async with Client(build_server(cache_for(provider([]), tmp_path))) as client:
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


def test_the_instructions_say_indicators_are_evidence() -> None:
    assert "evidence, not signals" in server_module.INSTRUCTIONS


class FakeServer:
    def __init__(self, cache: StockCache) -> None:
        self.cache = cache
        self.ran = False

    def run(self) -> None:
        self.ran = True


def run_main(
    monkeypatch: pytest.MonkeyPatch, env_file: Path | None = None
) -> FakeServer:
    built: list[FakeServer] = []

    def fake_build_server(cache: StockCache) -> FakeServer:
        built.append(FakeServer(cache))
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
