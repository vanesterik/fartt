import threading
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from fartt.data import Dividend, History, Listing, ProviderUnavailable
from fartt.stock_cache import StockCache, UpdateResult
from tests.fakes import FakeProvider, stock_candle

AAPL = Listing("AAPL", "Apple Inc.", "USD", "XNYS")
# NYSE is open at 20:00 UTC on 2026-10-29 and closed (plus margin) at 21:00.
DURING_SESSION = datetime(2026, 10, 29, 20, 0, tzinfo=UTC)
AFTER_CLOSE = datetime(2026, 10, 29, 21, 0, tzinfo=UTC)
DAYS = [date(2026, 10, 26), date(2026, 10, 27), date(2026, 10, 28), date(2026, 10, 29)]


def aapl_history() -> History:
    return History(
        [stock_candle(day, 100.0 + i) for i, day in enumerate(DAYS)],
        [Dividend(date(2026, 10, 27), 0.26)],
        [],
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider({"AAPL": aapl_history()}, {"AAPL": AAPL}, now=DURING_SESSION)


@pytest.fixture
def cache(provider: FakeProvider, tmp_path: Path) -> StockCache:
    return StockCache(provider, tmp_path, clock=lambda: provider.now)


def test_first_update_writes_the_full_history(
    cache: StockCache, tmp_path: Path
) -> None:
    result = cache.update("aapl")

    assert result == UpdateResult(new_candles=3, rebuilt=None, fetched=True)
    lines = (tmp_path / "stocks" / "AAPL" / "candles.csv").read_text().splitlines()
    assert lines[0] == "Date,Open,High,Low,Close,Volume"
    assert lines[1] == "2026-10-26,99.5,101.0,99.0,100.0,1000.0"


def test_latest_returns_the_newest_candles_oldest_first(cache: StockCache) -> None:
    cache.update("AAPL")

    assert [c.date for c in cache.latest("AAPL", 2)] == DAYS[1:3]
    assert cache.latest("AAPL", 2)[-1] == stock_candle(DAYS[2], 102.0)


def test_latest_of_an_uncached_ticker_is_empty(cache: StockCache) -> None:
    assert cache.latest("AAPL", 5) == []
    assert cache.events("AAPL") == ([], [])
    assert cache.listing("AAPL") is None


def test_events_and_listing_are_cached(cache: StockCache) -> None:
    cache.update("AAPL")

    assert cache.events("AAPL") == ([Dividend(date(2026, 10, 27), 0.26)], [])
    assert cache.listing("AAPL") == AAPL


def test_first_update_never_writes_the_forming_session(cache: StockCache) -> None:
    cache.update("AAPL")

    assert cache.latest("AAPL", 10)[-1].date == date(2026, 10, 28)


def test_an_unreachable_provider_writes_nothing(
    cache: StockCache, provider: FakeProvider, tmp_path: Path
) -> None:
    provider.reachable = False

    with pytest.raises(ProviderUnavailable):
        cache.update("AAPL")
    assert not list((tmp_path / "stocks").glob("*/*.csv"))


def test_a_reader_waits_for_a_running_update(
    provider: FakeProvider, tmp_path: Path
) -> None:
    started = threading.Event()
    release = threading.Event()
    serve_history = provider.history

    def blocking_history(ticker: str, start: date | None) -> History:
        started.set()
        release.wait(timeout=5)
        return serve_history(ticker, start)

    provider.history = blocking_history  # type: ignore[method-assign]
    cache = StockCache(provider, tmp_path, clock=lambda: provider.now)
    cache.directory("AAPL").mkdir(parents=True)
    writer = threading.Thread(target=cache.update, args=("AAPL",))
    writer.start()
    assert started.wait(timeout=5)

    read: list[int] = []
    reader = threading.Thread(target=lambda: read.append(len(cache.latest("AAPL", 10))))
    reader.start()
    reader.join(timeout=0.2)
    assert reader.is_alive()  # blocked by the update's lock

    release.set()
    writer.join(timeout=5)
    reader.join(timeout=5)
    assert read == [3]
