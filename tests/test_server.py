from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pytest
from mcp import Client

from fartt.data import History, Listing
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


async def call(cache: StockCache, arguments: dict[str, Any]) -> Any:
    async with Client(build_server(cache)) as client:
        return await client.call_tool("get_candles", arguments)


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
async def test_the_only_tool_is_get_candles(tmp_path: Path) -> None:
    async with Client(build_server(cache_for(provider([]), tmp_path))) as client:
        tools = (await client.list_tools()).tools

    assert [tool.name for tool in tools] == ["get_candles"]


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
