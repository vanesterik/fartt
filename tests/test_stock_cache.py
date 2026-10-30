import threading
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from fartt import stock_cache
from fartt.data import (
    Candle,
    Dividend,
    History,
    Listing,
    ProviderUnavailable,
    Split,
    UnsupportedExchange,
)
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


def filled(cache: StockCache, provider: FakeProvider) -> list[Candle]:
    """Fill the cache during the 10-29 session, then move past its close."""
    cache.update("AAPL")
    provider.now = AFTER_CLOSE
    return cache.latest("AAPL", 10)


def test_next_day_appends_only_the_new_session(
    cache: StockCache, provider: FakeProvider
) -> None:
    filled(cache, provider)

    assert cache.update("AAPL") == UpdateResult(1, None, True)
    assert provider.history_calls[-1] == ("AAPL", date(2026, 10, 28))
    assert [c.date for c in cache.latest("AAPL", 10)] == DAYS


def test_an_up_to_date_cache_does_not_call_the_provider(
    cache: StockCache, provider: FakeProvider
) -> None:
    cache.update("AAPL")

    assert cache.update("AAPL") == UpdateResult(0, None, False)
    assert len(provider.history_calls) == 1


def test_a_new_split_rebuilds_the_history(
    cache: StockCache, provider: FakeProvider
) -> None:
    before = filled(cache, provider)
    split = History(
        [stock_candle(c.date, c.close / 4) for c in aapl_history().candles[:3]]
        + [stock_candle(DAYS[3], 26.0)],
        [Dividend(date(2026, 10, 27), 0.065)],
        [Split(DAYS[3], 4.0)],
    )
    provider.histories["AAPL"] = split

    result = cache.update("AAPL")

    assert result == UpdateResult(
        4, "split on 2026-10-29 (4.0 new shares per old)", True
    )
    assert cache.latest("AAPL", 10)[0].close == pytest.approx(before[0].close / 4)
    assert cache.events("AAPL") == (
        [Dividend(date(2026, 10, 27), 0.065)],
        [Split(DAYS[3], 4.0)],
    )


def test_an_amended_last_candle_rebuilds_the_history(
    cache: StockCache, provider: FakeProvider
) -> None:
    filled(cache, provider)
    provider.histories["AAPL"].candles[2] = stock_candle(DAYS[2], 102.5)

    result = cache.update("AAPL")

    assert result.rebuilt == "last candle amended"
    assert cache.latest("AAPL", 2)[0].close == 102.5


def test_a_volume_only_change_is_not_an_amendment(
    cache: StockCache, provider: FakeProvider
) -> None:
    filled(cache, provider)
    provider.histories["AAPL"].candles[2] = stock_candle(DAYS[2], 102.0, volume=5.0)

    assert cache.update("AAPL") == UpdateResult(1, None, True)


def test_a_new_dividend_is_appended_and_no_price_changes(
    cache: StockCache, provider: FakeProvider
) -> None:
    before = filled(cache, provider)
    provider.histories["AAPL"].dividends.append(Dividend(DAYS[3], 0.26))

    cache.update("AAPL")

    assert cache.latest("AAPL", 10)[:-1] == before
    assert cache.events("AAPL")[0][-1] == Dividend(DAYS[3], 0.26)


def test_a_dividend_on_the_last_cached_date_is_not_recorded_twice(
    provider: FakeProvider, tmp_path: Path
) -> None:
    provider.histories["AAPL"].dividends.append(Dividend(DAYS[2], 0.1))
    cache = StockCache(provider, tmp_path, clock=lambda: provider.now)
    filled(cache, provider)

    cache.update("AAPL")  # re-fetches from 10-28, which carries the dividend

    assert cache.events("AAPL")[0].count(Dividend(DAYS[2], 0.1)) == 1


def test_a_torn_last_line_is_healed_before_updating(
    cache: StockCache, provider: FakeProvider, tmp_path: Path
) -> None:
    filled(cache, provider)
    with (tmp_path / "stocks" / "AAPL" / "candles.csv").open("a") as file:
        file.write("2026-10-29,10")  # a killed update's half line

    assert cache.update("AAPL") == UpdateResult(1, None, True)
    assert [c.date for c in cache.latest("AAPL", 2)] == DAYS[2:]


def test_a_last_line_missing_its_newline_is_refetched(
    cache: StockCache, provider: FakeProvider, tmp_path: Path
) -> None:
    filled(cache, provider)
    path = tmp_path / "stocks" / "AAPL" / "candles.csv"
    path.write_text(path.read_text().rstrip("\n"))

    assert cache.update("AAPL") == UpdateResult(2, None, True)
    assert [c.date for c in cache.latest("AAPL", 10)] == DAYS


def test_a_failed_rebuild_leaves_the_old_files(
    cache: StockCache, provider: FakeProvider
) -> None:
    before = filled(cache, provider)
    provider.histories["AAPL"].splits.append(Split(DAYS[3], 4.0))
    provider.fail_on_full_history = True

    with pytest.raises(ProviderUnavailable):
        cache.update("AAPL")
    assert cache.latest("AAPL", 10) == before


def test_an_empty_refetch_appends_nothing(
    cache: StockCache, provider: FakeProvider
) -> None:
    filled(cache, provider)
    provider.histories["AAPL"] = History([], [], [])

    assert cache.update("AAPL") == UpdateResult(0, None, True)


def test_an_interrupted_split_rebuild_is_redone(
    cache: StockCache, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    filled(cache, provider)
    provider.histories["AAPL"].splits.append(Split(DAYS[3], 4.0))
    real_replace = stock_cache._replace

    def crash_on_events(path: Path, header: list[str], rows: Any) -> None:
        if path.name == "events.csv":
            raise OSError("disk full")
        real_replace(path, header, rows)

    monkeypatch.setattr(stock_cache, "_replace", crash_on_events)
    with pytest.raises(OSError):
        cache.update("AAPL")
    monkeypatch.setattr(stock_cache, "_replace", real_replace)

    assert (
        cache.update("AAPL").rebuilt == "split on 2026-10-29 (4.0 new shares per old)"
    )
    assert cache.events("AAPL")[1] == [Split(DAYS[3], 4.0)]


def test_a_missing_listing_is_restored(
    cache: StockCache, provider: FakeProvider, tmp_path: Path
) -> None:
    filled(cache, provider)
    cache.update("AAPL")
    (tmp_path / "stocks" / "AAPL" / "listing.json").unlink()

    cache.update("AAPL")

    assert cache.listing("AAPL") == AAPL
    assert cache.update("AAPL").fetched is False


def test_a_dividend_ahead_of_its_candle_is_recorded_once(
    cache: StockCache, provider: FakeProvider
) -> None:
    filled(cache, provider)
    # Yahoo has the 10-29 dividend but not yet the 10-29 candle.
    provider.histories["AAPL"].candles.pop()
    provider.histories["AAPL"].dividends.append(Dividend(DAYS[3], 0.26))

    for _ in range(3):
        cache.update("AAPL")

    assert cache.events("AAPL")[0].count(Dividend(DAYS[3], 0.26)) == 1


def test_a_late_dividend_on_the_last_cached_date_is_recorded(
    cache: StockCache, provider: FakeProvider
) -> None:
    filled(cache, provider)
    provider.histories["AAPL"].dividends.append(Dividend(DAYS[2], 0.1))

    cache.update("AAPL")

    assert Dividend(DAYS[2], 0.1) in cache.events("AAPL")[0]


def test_an_unterminated_last_line_is_refetched(
    cache: StockCache, provider: FakeProvider, tmp_path: Path
) -> None:
    filled(cache, provider)
    cache.update("AAPL")
    path = tmp_path / "stocks" / "AAPL" / "candles.csv"
    # A kill cut the volume short: the line still parses, but is wrong.
    path.write_text(path.read_text().rstrip("\n").rsplit(",", 1)[0] + ",12")

    provider.now = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)
    cache.update("AAPL")

    assert cache.latest("AAPL", 1)[0].volume == 1000.0


def test_a_ticker_that_names_no_exchange_is_rejected_by_reads(
    cache: StockCache,
) -> None:
    with pytest.raises(UnsupportedExchange):
        cache.latest("..", 1)
