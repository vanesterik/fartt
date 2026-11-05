"""
Integration tests for `get_performance` (#122), driven the way Claude Code
drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder, its clock pinned to the
snapshot's moment and Yahoo out of reach (`snapshot.OFFLINE`):
`get_performance` reads only the journal.

Each test's docstring starts with its tags: `@issue-122` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and are
computed here from the journal the test writes, never from the server's code.

Scoring a call needs its horizon to have passed, which takes weeks in real
time, so each test writes its journal as JSON Lines in the format the journal
stores (`"kind": "decision"` and `"kind": "score"` lines) before starting the
server. The provider is pointed at a proxy nobody listens on, so a call to it
would fail rather than pass unnoticed.
"""

import hashlib
import json
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from tests.integration.snapshot import OFFLINE, UNREACHABLE

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 120
MIN_CALLS = 10  # the design's minimum before a group reports figures
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
PAST = date(2026, 1, 2)  # a horizon end long passed: an open call on it is due
FUTURE = date(2027, 6, 1)  # a horizon end still ahead: not due

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(assets_dir: Path) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {
        **os.environ,
        "FARTT_ASSETS_DIR": str(assets_dir),
        **OFFLINE,
        **{name: UNREACHABLE for name in ("HTTP_PROXY", "http_proxy")},
    }
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _performance(assets_dir: Path) -> dict[str, Any]:
    async with _server(assets_dir) as client:
        result = await client.call_tool("get_performance", {})
    assert not result.is_error, result.content
    content = result.structured_content
    assert content is not None
    return content


# --- the journal ------------------------------------------------------------

EXCHANGES = {"ASML.AS": ("XAMS", "EUR"), "AAPL": ("XNYS", "USD")}


def _call(
    id_: int,
    ticker: str,
    direction: str,
    signals: list[str],
    decision_session: date,
    horizon_ends: date,
) -> dict[str, Any]:
    exchange, currency = EXCHANGES.get(ticker, ("XNYS", "USD"))
    return {
        "kind": "decision",
        "id": id_,
        "recorded_at": f"{decision_session.isoformat()}T18:00:00Z",
        "ticker": ticker,
        "currency": currency,
        "exchange": exchange,
        "direction": direction,
        "horizon": 20,
        "reasoning": f"Integration test call {id_}.",
        "signals": [{"name": name, "stance": "for"} for name in signals],
        "decision_session": decision_session.isoformat(),
        "close": 100.0,
        "horizon_ends": horizon_ends.isoformat(),
        "benchmark": {
            "ticker": "IWDA.AS",
            "currency": "EUR",
            "session": decision_session.isoformat(),
            "close": 100.0,
        },
    }


def _score(call: dict[str, Any], stock: float, benchmark: float) -> dict[str, Any]:
    excess = round(stock - benchmark, 6)
    right = excess > 0 if call["direction"] == "buy" else excess <= 0
    start, end = call["decision_session"], call["horizon_ends"]
    return {
        "kind": "score",
        "decision_id": call["id"],
        "scored_at": f"{end}T23:00:00Z",
        "start": {"session": start, "close": 100.0, "rate": None},
        "end": {"session": end, "close": round(100 * (1 + stock), 4), "rate": None},
        "dividends": 0.0,
        "return": stock,
        "benchmark": {
            "ticker": "IWDA.AS",
            "start": {"session": start, "close": 100.0, "rate": None},
            "end": {"session": end, "close": round(100 * (1 + benchmark), 4)},
            "return": benchmark,
        },
        "excess_return": excess,
        "right": right,
    }


class Book:
    """The journal a test writes, and what it holds, for the expectations."""

    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.scored: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def scored_call(
        self,
        ticker: str,
        direction: str,
        signals: list[str],
        stock: float,
        benchmark: float,
        decision_session: date,
    ) -> None:
        call = _call(
            len(self.calls) + 1,
            ticker,
            direction,
            signals,
            decision_session,
            decision_session + timedelta(days=28),
        )
        score = _score(call, stock, benchmark)
        self.lines += [call, score]
        self.calls.append(call)
        self.scored.append((call, score))

    def open_call(self, ticker: str, horizon_ends: date) -> None:
        call = _call(
            len(self.calls) + 1,
            ticker,
            "buy",
            ["trend"],
            horizon_ends - timedelta(days=28),
            horizon_ends,
        )
        self.lines.append(call)
        self.calls.append(call)

    def write(self, assets_dir: Path) -> Path:
        path = assets_dir / "journal" / "decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(line) + "\n" for line in self.lines))
        return path


def _full_book() -> Book:
    """
    22 scored calls (11 buy, 11 pass) on three tickers, with `trend` on every
    call, `rsi` on 12 and `spread` on none, plus 3 open calls: 2 due (one on
    ARM, which has no scored call) and 1 running.
    """
    book = Book()
    rng = random.Random(122)
    tickers = ["ASML.AS"] * 10 + ["AAPL"] * 6 + ["GOOG"] * 6
    for i, ticker in enumerate(tickers):
        direction = "buy" if i % 2 == 0 else "pass"
        signals = (
            ["trend"] + (["rsi"] if i < 12 else []) + (["macd"] if i % 3 == 0 else [])
        )
        stock = round(rng.uniform(-0.08, 0.1), 6)
        benchmark = round(rng.uniform(-0.03, 0.04), 6)
        if i == 5:  # a pass whose excess is exactly 0 is right
            benchmark = stock
        book.scored_call(
            ticker,
            direction,
            signals,
            stock,
            benchmark,
            date(2026, 2, 2) + timedelta(days=7 * i),
        )
    book.open_call("ASML.AS", PAST)
    book.open_call("ARM", PAST)
    book.open_call("AAPL", FUTURE)
    return book


# --- expectations, from the criteria ----------------------------------------


def _edge(call: dict[str, Any], score: dict[str, Any]) -> float:
    return (
        score["excess_return"]
        if call["direction"] == "buy"
        else -score["excess_return"]
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _assert_close(actual: float | None, expected: float) -> None:
    # Averages are rounded to 6 decimals; allow for the last digit's rounding.
    assert actual is not None
    assert actual == pytest.approx(expected, abs=1.5e-6)


def _assert_group(
    group: dict[str, Any], members: list[tuple[dict[str, Any], dict[str, Any]]]
) -> None:
    assert group["calls"] == len(members)
    rights = [score["right"] for _, score in members]
    assert group["hit_rate"] == pytest.approx(sum(rights) / len(rights), abs=5e-5)
    _assert_close(group["average_return"], _mean([s["return"] for _, s in members]))
    _assert_close(
        group["average_benchmark_return"],
        _mean([s["benchmark"]["return"] for _, s in members]),
    )
    _assert_close(
        group["average_excess_return"], _mean([s["excess_return"] for _, s in members])
    )
    _assert_close(group["average_edge"], _mean([_edge(c, s) for c, s in members]))
    assert group["period"] == {
        "start": min(c["decision_session"] for c, _ in members),
        "end": max(c["horizon_ends"] for c, _ in members),
    }


def _assert_too_few(group: dict[str, Any], count: int, figures: list[str]) -> None:
    assert group["calls"] == count
    for figure in figures:
        assert group[figure] is None, figure
    note = group["note"] or ""
    assert str(MIN_CALLS - count) in note and "more" in note, note


GROUP_FIGURES = [
    "hit_rate",
    "average_return",
    "average_benchmark_return",
    "average_excess_return",
    "average_edge",
]
SIDE_FIGURES = ["hit_rate", "average_excess_return", "average_edge"]


# --- the tests ----------------------------------------------------------------


async def test_overall_and_per_direction_match_the_stored_scores(
    tmp_path: Path,
) -> None:
    """@issue-122 @AC1 @AC2"""
    book = _full_book()
    book.write(tmp_path)

    result = await _performance(tmp_path)

    _assert_group(result["overall"], book.scored)
    _assert_group(result["buy"], [m for m in book.scored if m[0]["direction"] == "buy"])
    _assert_group(
        result["pass"], [m for m in book.scored if m[0]["direction"] == "pass"]
    )
    assert result["scored"] == 22


async def test_signals_compare_used_with_not_used(tmp_path: Path) -> None:
    """@issue-122 @AC3 @AC4"""
    book = _full_book()
    book.write(tmp_path)

    result = await _performance(tmp_path)

    assert [entry["signal"] for entry in result["signals"]] == SIGNAL_NAMES
    by_name = {entry["signal"]: entry for entry in result["signals"]}
    for name in SIGNAL_NAMES:
        used = [m for m in book.scored if name in {s["name"] for s in m[0]["signals"]}]
        not_used = [m for m in book.scored if m not in used]
        for side, members in (("used", used), ("not_used", not_used)):
            reported = by_name[name][side]
            assert reported["calls"] == len(members), (name, side)
            if len(members) >= MIN_CALLS:
                rights = [s["right"] for _, s in members]
                assert reported["hit_rate"] == pytest.approx(
                    sum(rights) / len(rights), abs=5e-5
                )
                _assert_close(
                    reported["average_excess_return"],
                    _mean([s["excess_return"] for _, s in members]),
                )
                _assert_close(
                    reported["average_edge"], _mean([_edge(c, s) for c, s in members])
                )
            else:
                _assert_too_few(reported, len(members), SIDE_FIGURES)
    # The fixture covers both sides reported, and both kinds of a side too small.
    assert (
        by_name["rsi"]["used"]["calls"] == 12
        and by_name["rsi"]["not_used"]["calls"] == 10
    )
    assert by_name["trend"]["not_used"]["calls"] == 0
    assert by_name["spread"]["used"]["calls"] == 0


async def test_a_group_below_the_minimum_shows_its_count_only(tmp_path: Path) -> None:
    """@issue-122 @AC4 @AC5"""
    book = Book()
    values = [(0.05, 0.01), (-0.02, 0.01), (0.03, 0.02), (0.01, 0.04), (0.06, 0.0)]
    for i, (stock, benchmark) in enumerate(values):
        book.scored_call(
            "ASML.AS",
            "buy",
            ["trend"],
            stock,
            benchmark,
            date(2026, 3, 2) + timedelta(days=i),
        )
    book.scored_call("AAPL", "pass", ["rsi"], -0.03, 0.01, date(2026, 3, 9))
    book.scored_call("AAPL", "pass", ["rsi"], 0.04, 0.01, date(2026, 3, 10))
    book.write(tmp_path)

    result = await _performance(tmp_path)

    _assert_too_few(result["overall"], 7, GROUP_FIGURES)
    _assert_too_few(result["buy"], 5, GROUP_FIGURES)
    _assert_too_few(result["pass"], 2, GROUP_FIGURES)
    # Tickers are filled whatever their count.
    by_ticker = {entry["ticker"]: entry for entry in result["tickers"]}
    asml = [m for m in book.scored if m[0]["ticker"] == "ASML.AS"]
    assert by_ticker["ASML.AS"]["calls"] == 5
    assert by_ticker["ASML.AS"]["hit_rate"] == pytest.approx(
        sum(s["right"] for _, s in asml) / 5, abs=5e-5
    )
    _assert_close(by_ticker["AAPL"]["average_edge"], _mean([0.04, -0.03]))
    _assert_close(by_ticker["AAPL"]["average_excess_return"], _mean([-0.04, 0.03]))


async def test_tickers_are_listed_with_their_open_calls_most_called_first(
    tmp_path: Path,
) -> None:
    """@issue-122 @AC5 @AC6"""
    book = _full_book()
    book.write(tmp_path)

    result = await _performance(tmp_path)

    # ARM has only an open call, so it isn't listed; AAPL and GOOG tie on 6.
    assert [t["ticker"] for t in result["tickers"]] == ["ASML.AS", "AAPL", "GOOG"]
    open_by_ticker = {"ASML.AS": 1, "AAPL": 1, "GOOG": 0}
    for entry in result["tickers"]:
        members = [m for m in book.scored if m[0]["ticker"] == entry["ticker"]]
        assert entry["calls"] == len(members)
        assert entry["open"] == open_by_ticker[entry["ticker"]]
        assert entry["hit_rate"] == pytest.approx(
            sum(s["right"] for _, s in members) / len(members), abs=5e-5
        )
        _assert_close(
            entry["average_excess_return"],
            _mean([s["excess_return"] for _, s in members]),
        )
        _assert_close(entry["average_edge"], _mean([_edge(c, s) for c, s in members]))


async def test_open_calls_are_counted_not_figured(tmp_path: Path) -> None:
    """@issue-122 @AC6"""
    book = _full_book()
    book.write(tmp_path)

    result = await _performance(tmp_path)

    assert result["scored"] == 22
    assert result["overall"]["calls"] == 22
    assert result["open"] == 3
    assert result["due"] == 2
    assert "score_decisions" in (result["note"] or "")


async def test_no_call_due_says_nothing_about_scoring(tmp_path: Path) -> None:
    """@issue-122 @AC6"""
    book = _full_book()
    book.lines = [
        line
        for line in book.lines
        if line["kind"] == "score" or line["horizon_ends"] != PAST.isoformat()
    ]
    book.write(tmp_path)

    result = await _performance(tmp_path)

    assert result["open"] == 1
    assert result["due"] == 0
    assert "score_decisions" not in (result["note"] or "")


def _assert_empty(result: dict[str, Any]) -> None:
    assert result["scored"] == 0
    for name in ("overall", "buy", "pass"):
        group = result[name]
        assert group["calls"] == 0
        for figure in GROUP_FIGURES:
            assert group[figure] is None, (name, figure)
        assert group["period"] is None
    for entry in result["signals"]:
        assert entry["used"]["calls"] == 0 and entry["not_used"]["calls"] == 0
    assert result["tickers"] == []
    assert result["note"]


async def test_a_missing_journal_is_zero_counts_and_a_note(tmp_path: Path) -> None:
    """@issue-122 @AC7"""
    result = await _performance(tmp_path)

    _assert_empty(result)
    assert result["open"] == 0
    assert not (tmp_path / "journal").exists()


async def test_an_empty_journal_is_zero_counts_and_a_note(tmp_path: Path) -> None:
    """@issue-122 @AC7"""
    Book().write(tmp_path)

    result = await _performance(tmp_path)

    _assert_empty(result)


async def test_only_open_calls_is_zero_counts_and_a_note(tmp_path: Path) -> None:
    """@issue-122 @AC7 @AC6"""
    book = Book()
    book.open_call("ASML.AS", PAST)
    book.open_call("AAPL", FUTURE)
    book.write(tmp_path)

    result = await _performance(tmp_path)

    _assert_empty(result)
    assert result["open"] == 2
    assert result["due"] == 1


async def test_the_result_says_it_reports_calls_not_trades(tmp_path: Path) -> None:
    """@issue-122 @AC8"""
    _full_book().write(tmp_path)

    result = await _performance(tmp_path)

    basis = result["basis"].lower()
    for words in ("calls", "trades", "eur", "iwda.as", "costs", "position size"):
        assert words in basis, words
    assert result["min_calls"] == MIN_CALLS


async def test_it_reads_offline_and_leaves_the_journal_as_it_was(
    tmp_path: Path,
) -> None:
    """@issue-122 @AC9"""
    path = _full_book().write(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    result = await _performance(tmp_path)

    assert result["scored"] == 22
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # Any cache update or provider fetch would create the stocks folder.
    assert not (tmp_path / "stocks").exists()


async def test_a_damaged_journal_is_an_error_naming_the_file(tmp_path: Path) -> None:
    """@issue-122 @AC9"""
    book = _full_book()
    path = book.write(tmp_path)
    with path.open("a") as file:
        file.write("{not json\n")
    before = path.read_bytes()

    async with _server(tmp_path) as client:
        result = await client.call_tool("get_performance", {})

    assert result.is_error
    text = " ".join(getattr(part, "text", "") for part in result.content)
    assert str(path) in text, text
    assert path.read_bytes() == before
