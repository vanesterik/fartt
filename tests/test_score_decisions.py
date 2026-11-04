import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from fartt.data import Candle, Dividend, History, Listing
from fartt.journal import BenchmarkClose, DecisionDraft, Journal, Score
from fartt.server import build_server
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from tests.fakes import FakeProvider, decision_draft
from tests.test_server import (
    AAPL,
    AFTER_CLOSE,
    ASML,
    IWDA,
    cache_for,
    call,
    get_decisions,
    history,
    journal_of,
    seed,
    sessions,
)

# A call made after 2026-09-01's close whose 20-session horizon ended on
# 2026-09-29; AFTER_CLOSE (2026-10-29) is well past it.
START = date(2026, 9, 1)
END = date(2026, 9, 29)
EURUSD = Listing("EURUSD=X", "EUR/USD", "USD", "24/5")


def fx_history(days: list[date]) -> History:
    return History(
        [
            Candle(day, 1.1, 1.1, 1.1, round(1.1 + 0.001 * i, 4), 0.0)
            for i, day in enumerate(days)
        ],
        [],
        [],
    )


def fake(**histories: History) -> FakeProvider:
    stored = {
        "ASML.AS": history(sessions("XAMS", 70)),
        "AAPL": history(sessions("XNYS", 70)),
        "IWDA.AS": history(sessions("XAMS", 70), first=50.0),
        "EURUSD=X": fx_history(sessions("24/5", 70, last=date(2026, 10, 28))),
    }
    return FakeProvider(
        histories=stored | histories,
        listings={"ASML.AS": ASML, "AAPL": AAPL, "IWDA.AS": IWDA, "EURUSD=X": EURUSD},
        now=AFTER_CLOSE,
    )


def past_call(**changes: Any) -> DecisionDraft:
    fields: dict[str, Any] = {
        "decision_session": START,
        "horizon_ends": END,
        "benchmark": BenchmarkClose(
            ticker="IWDA.AS", currency="EUR", session=START, close=50.0
        ),
    }
    return decision_draft(**(fields | changes))


def apple_call(**changes: Any) -> DecisionDraft:
    return past_call(ticker="AAPL", currency="USD", exchange="XNYS", **changes)


def close_on(provider: FakeProvider, ticker: str, day: date) -> float:
    return next(c.close for c in provider.histories[ticker].candles if c.date == day)


async def score_decisions(cache: StockCache) -> Any:
    return await call(cache, {}, "score_decisions")


def score_lines(cache: StockCache) -> list[dict[str, Any]]:
    path = journal_of(cache).path
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return [line for line in lines if line["kind"] == "score"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_a_due_call_is_scored_and_stored(tmp_path: Path) -> None:
    # @AC1 @AC3 @AC4
    provider = fake()
    cache = cache_for(provider, tmp_path)
    seed(cache, past_call())

    result = await score_decisions(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    stock = round(
        close_on(provider, "ASML.AS", END) / close_on(provider, "ASML.AS", START) - 1, 6
    )
    benchmark = round(
        close_on(provider, "IWDA.AS", END) / close_on(provider, "IWDA.AS", START) - 1, 6
    )
    assert content["scored"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "direction": "buy",
            "decision_session": "2026-09-01",
            "horizon_ends": "2026-09-29",
            "return": stock,
            "benchmark_return": benchmark,
            "excess_return": round(stock - benchmark, 6),
            "right": stock - benchmark > 0,
        }
    ]
    assert content["not_scored"] == []
    assert content["running"] == 0
    assert content["already_scored"] == 0
    assert content["note"] is None
    lines = score_lines(cache)
    assert len(lines) == 1
    assert lines[0]["decision_id"] == 1
    assert lines[0]["return"] == stock


@pytest.mark.anyio
async def test_dividends_after_the_decision_count(tmp_path: Path) -> None:
    # @AC1 @AC5
    stored = history(sessions("XAMS", 70))
    provider = fake(
        **{
            "ASML.AS": History(
                stored.candles,
                [Dividend(START, 1.0), Dividend(date(2026, 9, 2), 2.0)],
                [],
            )
        }
    )
    cache = cache_for(provider, tmp_path)
    seed(cache, past_call())

    result = await score_decisions(cache)

    start = close_on(provider, "ASML.AS", START)
    end = close_on(provider, "ASML.AS", END)
    assert result.structured_content["scored"][0]["return"] == round(
        (end + 2.0) / start - 1, 6
    )


@pytest.mark.anyio
async def test_a_usd_call_uses_the_cached_rates(tmp_path: Path) -> None:
    # @AC2
    provider = fake()
    cache = cache_for(provider, tmp_path)
    seed(cache, apple_call())

    result = await score_decisions(cache)

    assert not result.is_error, result.content
    rate_start = close_on(provider, "EURUSD=X", START)
    rate_end = close_on(provider, "EURUSD=X", END)
    expected = round(
        close_on(provider, "AAPL", END)
        / rate_end
        / (close_on(provider, "AAPL", START) / rate_start)
        - 1,
        6,
    )
    assert result.structured_content["scored"][0]["return"] == expected
    stored = journal_of(cache).scores()[1]
    assert (stored.start.rate, stored.end.rate) == (rate_start, rate_end)


@pytest.mark.anyio
async def test_scoring_twice_scores_nothing(tmp_path: Path) -> None:
    # @AC6
    cache = cache_for(fake(), tmp_path)
    seed(cache, past_call())
    await score_decisions(cache)
    before = journal_of(cache).path.read_bytes()

    result = await score_decisions(cache)

    assert result.structured_content["scored"] == []
    assert result.structured_content["already_scored"] == 1
    assert journal_of(cache).path.read_bytes() == before


@pytest.mark.anyio
async def test_a_running_call_is_counted_not_scored(tmp_path: Path) -> None:
    # @AC8
    cache = cache_for(fake(), tmp_path)
    seed(cache, decision_draft())  # horizon ends 2026-11-26

    result = await score_decisions(cache)

    content = result.structured_content
    assert content["running"] == 1
    assert content["scored"] == content["not_scored"] == []
    assert content["note"] == "No open call's horizon has passed; 1 is running."
    assert journal_of(cache).scores() == {}
    listed = await get_decisions(cache)
    assert listed.structured_content["decisions"][0]["status"] == "open"


@pytest.mark.anyio
async def test_unreachable_without_the_data_stays_open(tmp_path: Path) -> None:
    # @AC7
    provider = fake()
    provider.reachable = False
    cache = cache_for(provider, tmp_path)
    seed(cache, past_call())

    result = await score_decisions(cache)

    assert not result.is_error, result.content
    assert result.structured_content["scored"] == []
    assert result.structured_content["not_scored"] == [
        {
            "id": 1,
            "ticker": "ASML.AS",
            "horizon_ends": "2026-09-29",
            "reason": "ASML.AS's close for 2026-09-01 isn't available yet "
            "(the provider is unreachable)",
        }
    ]
    listed = await get_decisions(cache)
    assert listed.structured_content["decisions"][0]["status"] == "open"

    provider.reachable = True
    later = await score_decisions(cache)

    assert [c["id"] for c in later.structured_content["scored"]] == [1]


@pytest.mark.anyio
async def test_unreachable_with_the_data_cached_scores(tmp_path: Path) -> None:
    # @AC7
    provider = fake()
    cache = cache_for(provider, tmp_path)
    for ticker in ("AAPL", "IWDA.AS", "EURUSD=X"):
        cache.update(ticker)
    seed(cache, apple_call())
    provider.reachable = False

    result = await score_decisions(cache)

    assert [c["id"] for c in result.structured_content["scored"]] == [1]


@pytest.mark.anyio
async def test_an_unpublished_candle_stays_open(tmp_path: Path) -> None:
    # @AC7
    days = [day for day in sessions("XAMS", 70) if day != END]
    cache = cache_for(fake(**{"ASML.AS": history(days)}), tmp_path)
    seed(cache, past_call())

    result = await score_decisions(cache)

    assert result.structured_content["not_scored"][0]["reason"] == (
        "ASML.AS's close for 2026-09-29 isn't available yet"
    )
    assert journal_of(cache).scores() == {}


@pytest.mark.anyio
async def test_a_rate_not_closed_yet_stays_open(tmp_path: Path) -> None:
    # @AC7
    rates = fx_history(sessions("24/5", 70, last=date(2026, 9, 28)))
    cache = cache_for(fake(**{"EURUSD=X": rates}), tmp_path)
    seed(cache, apple_call())

    result = await score_decisions(cache)

    assert result.structured_content["not_scored"][0]["reason"] == (
        "the EUR/USD rate for 2026-09-29 isn't available yet"
    )


@pytest.mark.anyio
async def test_a_delisted_ticker_stays_open(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)
    seed(cache, past_call(ticker="XYZ.AS"), past_call())

    result = await score_decisions(cache)

    content = result.structured_content
    assert content["not_scored"] == [
        {
            "id": 1,
            "ticker": "XYZ.AS",
            "horizon_ends": "2026-09-29",
            "reason": "Yahoo no longer lists XYZ.AS",
        }
    ]
    assert [c["id"] for c in content["scored"]] == [2]


@pytest.mark.anyio
async def test_each_ticker_is_updated_once_per_run(tmp_path: Path) -> None:
    provider = fake()
    cache = cache_for(provider, tmp_path)
    seed(cache, past_call(), past_call(), past_call(direction="pass"))

    result = await score_decisions(cache)

    assert len(result.structured_content["scored"]) == 3
    updated = [ticker for ticker, _ in provider.history_calls]
    assert updated.count("ASML.AS") == 1
    assert updated.count("IWDA.AS") == 1
    assert "EURUSD=X" not in updated


@pytest.mark.anyio
async def test_a_score_written_by_another_run_counts_as_already_scored(
    tmp_path: Path,
) -> None:
    class RacedJournal(Journal):
        """Another run scores every call between the read and the append."""

        def append_score(self, score: Score) -> Score | None:
            super().append_score(score)
            return None

    cache = cache_for(fake(), tmp_path)
    seed(cache, past_call())
    journal = RacedJournal(journal_of(cache).directory)

    async with Client(build_server(cache, journal)) as client:
        result = await client.call_tool("score_decisions", {})

    assert result.structured_content["scored"] == []
    assert result.structured_content["already_scored"] == 1
    assert len(score_lines(cache)) == 1


@pytest.mark.anyio
async def test_a_call_on_the_benchmark_itself(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)
    seed(cache, past_call(ticker="IWDA.AS"))

    result = await score_decisions(cache)

    scored = result.structured_content["scored"][0]
    assert scored["return"] == scored["benchmark_return"]
    assert scored["excess_return"] == 0
    assert scored["right"] is False


@pytest.mark.anyio
async def test_the_lists_are_capped(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)
    seed(cache, *(past_call() for _ in range(51)))

    result = await score_decisions(cache)

    content = result.structured_content
    assert [c["id"] for c in content["scored"]] == list(range(1, 51))
    assert content["note"] == (
        "Showing the first 50 calls scored; 1 more was scored. Read them with "
        'get_decisions(status="scored").'
    )
    assert len(score_lines(cache)) == 51


@pytest.mark.anyio
async def test_the_not_scored_list_is_capped(tmp_path: Path) -> None:
    provider = fake()
    provider.reachable = False
    cache = cache_for(provider, tmp_path)
    seed(cache, *(past_call() for _ in range(52)))

    result = await score_decisions(cache)

    content = result.structured_content
    assert [c["id"] for c in content["not_scored"]] == list(range(1, 51))
    assert "2 more calls weren't scored" in content["note"]


@pytest.mark.anyio
async def test_a_damaged_journal_is_a_tool_error(tmp_path: Path) -> None:
    # @AC11
    cache = cache_for(fake(), tmp_path)
    seed(cache, past_call())
    with journal_of(cache).path.open("ab") as file:
        file.write(b"not json\n")
    before = journal_of(cache).path.read_bytes()

    result = await score_decisions(cache)

    assert result.is_error
    text = result.content[0].text
    assert str(journal_of(cache).path) in text
    assert "line 2" in text
    assert "operator" in text
    assert journal_of(cache).path.read_bytes() == before


@pytest.mark.anyio
async def test_an_empty_journal_scores_nothing(tmp_path: Path) -> None:
    cache = cache_for(fake(), tmp_path)

    result = await score_decisions(cache)

    assert not result.is_error, result.content
    assert result.structured_content["note"] == (
        "No open call's horizon has passed; 0 are running."
    )


def test_the_instructions_say_to_score_at_the_check_in() -> None:
    assert "score_decisions" in server_module.INSTRUCTIONS
    assert "check-in" in server_module.INSTRUCTIONS


@pytest.mark.anyio
@pytest.mark.parametrize(
    "tool", ["get_candles", "get_indicators", "get_trend", "get_support_resistance"]
)
async def test_the_read_tools_take_an_fx_pair(tmp_path: Path, tool: str) -> None:
    # @AC10
    provider = fake()
    cache = cache_for(provider, tmp_path)

    result = await call(cache, {"ticker": "eurusd=x"}, tool)

    assert not result.is_error, result.content
    if tool == "get_candles":
        rows = result.structured_content["rows"]
        last = provider.histories["EURUSD=X"].candles[-1]
        assert rows[-1][0] == last.date.isoformat() == "2026-10-28"
        assert rows[-1][4] == last.close
        assert result.structured_content["exchange"] == "24/5"
