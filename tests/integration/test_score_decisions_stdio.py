"""
Integration tests for `score_decisions` (#121), driven the way Claude Code
drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder, against the live data provider.
Marked `network`.

Each test's docstring starts with its tags: `@issue-121` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and are
computed here from what the server itself returns: closes from `get_candles`
(also for `EURUSD=X`), and the dividends Yahoo published, from the cache's
`events.csv`.

`record_decision` only records calls whose horizon starts today, so, as the
design says, past calls are written straight into the temporary
`journal/decisions.jsonl` before the server starts: real tickers, real
sessions and closes from the cache. A provider that's down is simulated by
pointing the server's HTTP(S) proxy at a closed local port.
"""

import csv
import datetime as dt
import json
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
BENCHMARK = "IWDA.AS"
# Returns are rounded to 6 decimals (AC: "fractions, rounded to 6 decimals").
TOLERANCE = 2e-6
# A closed local port: every request through it fails at once.
UNREACHABLE = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"}

# (ticker, currency, exchange, direction, decision_session, horizon[, benchmark session])
# ASML.AS dividends went ex on 2026-04-24, 2026-07-27 and 2026-10-26; AAPL's
# on 2026-05-11. 2026-04-06 is Easter Monday: NYSE open, Euronext closed.
EUR_CALLS: list[tuple[Any, ...]] = [
    ("ASML.AS", "EUR", "XAMS", "buy", "2026-07-20", 20),  # dividend inside
    ("ASML.AS", "EUR", "XAMS", "pass", "2026-04-24", 5),  # dividend on decision_session
    ("ASML.AS", "EUR", "XAMS", "buy", "2026-10-19", 5),  # dividend on horizon_ends
    ("IWDA.AS", "EUR", "XAMS", "buy", "2026-06-01", 10),  # the benchmark: excess 0
    ("IWDA.AS", "EUR", "XAMS", "pass", "2026-06-01", 10),
]
USD_CALLS: list[tuple[Any, ...]] = [
    ("AAPL", "USD", "XNYS", "buy", "2026-03-20", 10),  # ends on Easter Monday
    ("AAPL", "USD", "XNYS", "pass", "2026-05-04", 10, "2026-04-30"),  # dividend inside
]

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


def _text(result: Any) -> str:
    return " ".join(getattr(part, "text", "") for part in result.content)


async def _call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, _text(result)
    content = result.structured_content
    assert content is not None
    return content


def _journal(assets_dir: Path) -> Path:
    return assets_dir / "journal" / "decisions.jsonl"


def _cached_closes(assets_dir: Path, ticker: str) -> dict[str, float]:
    with (assets_dir / "stocks" / ticker / "candles.csv").open() as file:
        return {
            row["Date"]: round(float(row["Close"]), 4) for row in csv.DictReader(file)
        }


def _dividends(assets_dir: Path, ticker: str) -> dict[str, float]:
    with (assets_dir / "stocks" / ticker / "events.csv").open() as file:
        return {
            row["Date"]: float(row["Value"])
            for row in csv.DictReader(file)
            if row["Type"] == "dividend"
        }


def _session_after(exchange: str, day: str, count: int) -> str:
    calendar = xcals.get_calendar(exchange)
    session = pd.Timestamp(day)
    for _ in range(count):
        session = calendar.next_session(session)
    return session.date().isoformat()


def _session_on_or_before(exchange: str, day: str) -> str:
    calendar = xcals.get_calendar(exchange)
    return (
        calendar.date_to_session(pd.Timestamp(day), direction="previous")
        .date()
        .isoformat()
    )


def _seed(assets_dir: Path, calls: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Writes past calls into the journal, as the operator could; returns them."""
    benchmark = _cached_closes(assets_dir, BENCHMARK)
    path = _journal(assets_dir)
    path.parent.mkdir(exist_ok=True)
    existing = path.read_text().count('"kind": "decision"') if path.exists() else 0
    entries = []
    with path.open("a") as file:
        for number, spec in enumerate(calls, start=existing + 1):
            ticker, currency, exchange, direction, session, horizon, *rest = spec
            benchmark_session = rest[0] if rest else session
            entry = {
                "kind": "decision",
                "recorded_at": f"{session}T21:00:00Z",
                "ticker": ticker,
                "currency": currency,
                "exchange": exchange,
                "direction": direction,
                "horizon": horizon,
                "reasoning": "Seeded past call for the score_decisions test.",
                "signals": [{"name": "trend", "stance": "for", "note": None}],
                "decision_session": session,
                "close": _cached_closes(assets_dir, ticker)[session],
                "horizon_ends": _session_after(exchange, session, horizon),
                "benchmark": {
                    "ticker": BENCHMARK,
                    "currency": "EUR",
                    "session": benchmark_session,
                    "close": benchmark[benchmark_session],
                },
                "id": number,
            }
            file.write(json.dumps(entry) + "\n")
            entries.append(entry)
    return entries


async def _closes(client: Client, ticker: str) -> dict[str, float]:
    """A ticker's closes by date, as get_candles returns them."""
    candles = await _call(client, "get_candles", ticker=ticker, count=200)
    return {row[0]: row[4] for row in candles["rows"]}


def _expected(
    entry: dict[str, Any],
    closes: dict[str, dict[str, float]],
    dividends: dict[str, float],
    rates: dict[str, float] | None,
) -> dict[str, Any]:
    """The score the criteria define (AC1-AC5), computed independently."""
    start, end = entry["decision_session"], entry["horizon_ends"]
    stock = closes[entry["ticker"]]
    paid = sum(v for d, v in dividends.items() if start < d <= end)
    rate_start = rates[start] if rates else 1.0
    rate_end = rates[end] if rates else 1.0
    stock_return = (stock[end] + paid) / rate_end / (stock[start] / rate_start) - 1
    benchmark = closes[BENCHMARK]
    benchmark_end = _session_on_or_before("XAMS", end)
    benchmark_return = (
        benchmark[benchmark_end] / benchmark[entry["benchmark"]["session"]] - 1
    )
    return {
        "return": stock_return,
        "benchmark_return": benchmark_return,
        "benchmark_end": benchmark_end,
        "dividends": paid,
    }


def _check_score(scored: dict[str, Any], expected: dict[str, Any]) -> None:
    assert scored["return"] == pytest.approx(expected["return"], abs=TOLERANCE)
    assert scored["benchmark_return"] == pytest.approx(
        expected["benchmark_return"], abs=TOLERANCE
    )
    # AC3: excess is return minus benchmark_return.
    assert scored["excess_return"] == pytest.approx(
        scored["return"] - scored["benchmark_return"], abs=TOLERANCE
    )
    # AC4: buy right iff excess > 0; pass right iff excess <= 0.
    if scored["direction"] == "buy":
        assert scored["right"] is (scored["excess_return"] > 0)
    else:
        assert scored["right"] is (scored["excess_return"] <= 0)


async def test_should_score_eur_calls_from_closes_and_dividends_after_the_decision(
    live_assets: Path,
) -> None:
    """@issue-121 @AC1 @AC3 @AC4 @AC5 @AC8 Should score due EUR calls as the criteria define and leave a running one open."""
    seeded = _seed(live_assets, EUR_CALLS)
    running = _seed(live_assets, [("ASML.AS", "EUR", "XAMS", "buy", "2026-11-03", 20)])[
        0
    ]
    dividends = _dividends(live_assets, "ASML.AS")
    assert (
        "2026-07-27" in dividends
        and "2026-04-24" in dividends
        and "2026-10-26" in dividends
    )

    async with _server(live_assets) as client:
        result = await _call(client, "score_decisions")
        closes = {t: await _closes(client, t) for t in ("ASML.AS", BENCHMARK)}
        journal = await _call(client, "get_decisions", limit=10)

    assert [s["id"] for s in result["scored"]] == [1, 2, 3, 4, 5]
    assert result["not_scored"] == []
    assert result["running"] == 1  # AC8
    assert result["already_scored"] == 0
    for entry, scored in zip(seeded, result["scored"], strict=True):
        assert scored["ticker"] == entry["ticker"]
        assert scored["direction"] == entry["direction"]
        assert scored["decision_session"] == entry["decision_session"]
        assert scored["horizon_ends"] == entry["horizon_ends"]
        stock_dividends = dividends if entry["ticker"] == "ASML.AS" else {}
        _check_score(scored, _expected(entry, closes, stock_dividends, None))

    by_id = {e["id"]: e for e in journal["decisions"]}
    # AC5: a dividend going ex on decision_session (call 2) isn't added;
    # one inside (call 1) and one on horizon_ends (call 3) are.
    assert by_id[2]["score"]["dividends"] == 0
    assert by_id[1]["score"]["dividends"] == pytest.approx(dividends["2026-07-27"])
    assert by_id[3]["score"]["dividends"] == pytest.approx(dividends["2026-10-26"])
    # AC4 at the boundary: a call on the benchmark has excess 0.
    assert (
        result["scored"][3]["excess_return"] == 0
        and result["scored"][3]["right"] is False
    )
    assert (
        result["scored"][4]["excess_return"] == 0
        and result["scored"][4]["right"] is True
    )
    # AC8: the running call isn't scored and stays open.
    assert by_id[running["id"]]["status"] == "open"
    assert by_id[running["id"]]["score"] is None
    assert '"decision_id": 6' not in _journal(live_assets).read_text()


async def _score_usd_calls(
    assets_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, float],
    dict[int, dict[str, Any]],
]:
    seeded = _seed(assets_dir, USD_CALLS)
    async with _server(assets_dir) as client:
        result = await _call(client, "score_decisions")
        closes = {t: await _closes(client, t) for t in ("AAPL", BENCHMARK)}
        rates = await _closes(client, "EURUSD=X")
        journal = await _call(client, "get_decisions", status="scored")
    assert [s["id"] for s in result["scored"]] == [1, 2]
    scores = {e["id"]: e["score"] for e in journal["decisions"]}
    return seeded, result, closes, rates, scores


async def test_should_end_a_usd_calls_benchmark_on_its_last_amsterdam_session(
    live_assets: Path,
) -> None:
    """@issue-121 @AC3 @AC4 Should measure IWDA.AS from the call's benchmark.session to its last session on or before horizon_ends."""
    seeded, result, closes, _, scores = await _score_usd_calls(live_assets)

    benchmark = closes[BENCHMARK]
    for entry, scored in zip(seeded, result["scored"], strict=True):
        start = entry["benchmark"]["session"]
        end = _session_on_or_before("XAMS", entry["horizon_ends"])
        assert scored["benchmark_return"] == pytest.approx(
            benchmark[end] / benchmark[start] - 1, abs=TOLERANCE
        )
        assert scores[entry["id"]]["benchmark"]["start"]["session"] == start
        assert scores[entry["id"]]["benchmark"]["end"]["session"] == end
        assert scored["excess_return"] == pytest.approx(
            scored["return"] - scored["benchmark_return"], abs=TOLERANCE
        )
        if scored["direction"] == "buy":
            assert scored["right"] is (scored["excess_return"] > 0)
        else:
            assert scored["right"] is (scored["excess_return"] <= 0)
    # Easter Monday: Euronext closed, so IWDA.AS ends on the Thursday before;
    # call 2's benchmark.session (2026-04-30) lags its decision_session.
    assert scores[1]["benchmark"]["end"]["session"] == "2026-04-02"
    assert scores[2]["benchmark"]["start"]["session"] == "2026-04-30"


async def test_should_measure_a_usd_call_in_euros_at_the_eurusd_closes(
    live_assets: Path,
) -> None:
    """@issue-121 @AC2 Should divide each end's close (and the dividends at the end) by EURUSD=X's close of that date, as get_candles returns it."""
    dividends = _dividends(live_assets, "AAPL")
    assert "2026-05-11" in dividends
    seeded, result, closes, rates, scores = await _score_usd_calls(live_assets)

    for entry, scored in zip(seeded, result["scored"], strict=True):
        score = scores[entry["id"]]
        assert score["start"]["rate"] == rates[entry["decision_session"]]
        assert score["end"]["rate"] == rates[entry["horizon_ends"]]
        expected = _expected(entry, closes, dividends, rates)
        assert scored["return"] == pytest.approx(expected["return"], abs=TOLERANCE)


async def test_should_not_change_a_score_when_later_candles_are_cached(
    live_assets: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """@issue-121 @AC5 Should score the same with or without candles after horizon_ends."""
    truncated = tmp_path_factory.mktemp("truncated")
    shutil.copytree(live_assets / "stocks", truncated / "stocks")
    calls = [EUR_CALLS[0], USD_CALLS[1]]
    _seed(live_assets, calls)
    entries = _seed(truncated, calls)
    cutoff = max(e["horizon_ends"] for e in entries)
    for ticker in ("ASML.AS", "AAPL", BENCHMARK, "EURUSD=X"):
        path = truncated / "stocks" / ticker / "candles.csv"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(
            "".join([lines[0], *(l for l in lines[1:] if l[:10] <= cutoff)])
        )

    # The truncated cache can't be brought up to date: it scores on what it has.
    async with _server(truncated, UNREACHABLE) as client:
        without_later = await _call(client, "score_decisions")
    async with _server(live_assets) as client:
        with_later = await _call(client, "score_decisions")

    assert len(without_later["scored"]) == 2, without_later
    assert without_later["scored"] == with_later["scored"]


async def test_should_score_nothing_twice_and_leave_the_journal_unchanged(
    live_assets: Path,
) -> None:
    """@issue-121 @AC6 Should score nothing on a second run and keep every byte of the journal."""
    _seed(live_assets, [*EUR_CALLS, *USD_CALLS])
    async with _server(live_assets) as client:
        first = await _call(client, "score_decisions")
        before = _journal(live_assets).read_bytes()
        second = await _call(client, "score_decisions")
    async with _server(live_assets) as client:  # and after a restart
        third = await _call(client, "score_decisions")

    assert len(first["scored"]) == 7
    for again in (second, third):
        assert again["scored"] == []
        assert again["not_scored"] == []
        assert again["already_scored"] == 7
    assert _journal(live_assets).read_bytes() == before


async def test_should_keep_a_call_open_with_a_reason_until_its_data_is_there(
    live_assets: Path,
) -> None:
    """@issue-121 @AC7 Should list unscorable calls with the ticker or rate and session, keep them open, and score them later."""
    seeded = _seed(live_assets, [EUR_CALLS[0], USD_CALLS[0]])
    # Then the stock's and the rate's caches are gone, and Yahoo is down.
    shutil.rmtree(live_assets / "stocks" / "ASML.AS")
    shutil.rmtree(live_assets / "stocks" / "EURUSD=X", ignore_errors=True)

    async with _server(live_assets, UNREACHABLE) as client:
        down = await _call(client, "score_decisions")
        journal = await _call(client, "get_decisions")
    lines_while_down = _journal(live_assets).read_text().splitlines()
    async with _server(live_assets) as client:
        up = await _call(client, "score_decisions")

    reasons = {c["id"]: c["reason"] for c in down["not_scored"]}
    assert down["scored"] == []
    assert sorted(reasons) == [1, 2]
    asml, aapl = seeded
    assert "ASML.AS" in reasons[1]
    assert asml["decision_session"] in reasons[1] or asml["horizon_ends"] in reasons[1]
    assert "EUR/USD" in reasons[2] or "EURUSD" in reasons[2]
    assert aapl["decision_session"] in reasons[2] or aapl["horizon_ends"] in reasons[2]
    assert [c["horizon_ends"] for c in down["not_scored"]] == [
        asml["horizon_ends"],
        aapl["horizon_ends"],
    ]
    assert {e["status"] for e in journal["decisions"]} == {"open"}
    assert all(e["score"] is None for e in journal["decisions"])
    assert not any('"kind": "score"' in line for line in lines_while_down)

    assert [s["id"] for s in up["scored"]] == [1, 2]
    assert up["not_scored"] == []


async def test_should_show_scores_after_a_restart(live_assets: Path) -> None:
    """@issue-121 @AC9 Should return scored calls with their score after a restart and filter on status."""
    _seed(live_assets, [EUR_CALLS[0], USD_CALLS[0]])
    _seed(live_assets, [("ASML.AS", "EUR", "XAMS", "buy", "2026-11-03", 20)])
    async with _server(live_assets) as client:
        result = await _call(client, "score_decisions")
    async with _server(live_assets) as client:
        every = await _call(client, "get_decisions")
        scored = await _call(client, "get_decisions", status="scored")
        still_open = await _call(client, "get_decisions", status="open")

    assert [s["id"] for s in result["scored"]] == [1, 2]
    by_id = {e["id"]: e for e in every["decisions"]}
    for compact in result["scored"]:
        entry = by_id[compact["id"]]
        assert entry["status"] == "scored"
        assert entry["score"]["decision_id"] == compact["id"]
        assert entry["score"]["return"] == compact["return"]
        assert entry["score"]["excess_return"] == compact["excess_return"]
        assert entry["score"]["right"] == compact["right"]
    assert by_id[3]["status"] == "open" and by_id[3]["score"] is None
    assert sorted(e["id"] for e in scored["decisions"]) == [1, 2]
    assert scored["total"] == 2
    assert [e["id"] for e in still_open["decisions"]] == [3]


async def test_should_read_fx_candles_and_refuse_a_call_on_an_fx_pair(
    tmp_path: Path,
) -> None:
    """@issue-121 @AC10 Should return EURUSD=X's daily closes and refuse record_decision on it."""
    async with _server(tmp_path) as client:
        candles = await _call(client, "get_candles", ticker="eurusd=x", count=10)
        refused = await client.call_tool(
            "record_decision",
            {
                "ticker": "EURUSD=X",
                "direction": "buy",
                "horizon": 20,
                "reasoning": "An FX pair isn't a stock.",
                "signals": [{"name": "trend", "stance": "for"}],
            },
        )

    assert candles["ticker"] == "EURUSD=X"
    assert candles["currency"] == "USD"
    assert len(candles["rows"]) == 10
    dates = [row[0] for row in candles["rows"]]
    assert dates == sorted(dates)
    assert all(dt.date.fromisoformat(d).weekday() < 5 for d in dates)
    assert all(0.5 < row[4] < 2 for row in candles["rows"])  # a EUR/USD rate
    assert refused.is_error
    assert "exchange rate, not a stock" in _text(refused)
    assert not _journal(tmp_path).exists()


async def test_should_be_a_tool_error_on_a_damaged_journal_and_write_nothing(
    live_assets: Path,
) -> None:
    """@issue-121 @AC11 Should name the file and line of a damaged journal and leave it unchanged."""
    _seed(live_assets, [EUR_CALLS[0]])
    with _journal(live_assets).open("a") as file:
        file.write("this is not json\n")
    _seed(live_assets, [EUR_CALLS[2]])  # a due call after the damage
    before = _journal(live_assets).read_bytes()

    async with _server(live_assets) as client:
        result = await client.call_tool("score_decisions", {})

    assert result.is_error, result.structured_content
    message = _text(result)
    assert str(_journal(live_assets)) in message or "decisions.jsonl" in message
    assert "line 2" in message
    assert _journal(live_assets).read_bytes() == before


async def test_should_refuse_a_call_on_any_fx_pair_and_reject_path_like_ones(
    tmp_path: Path,
) -> None:
    """@issue-121 @AC10 Should refuse record_decision on any FX pair however it's written, and reject an FX-looking ticker that could escape the cache folder."""
    # Given an empty assets folder and the server started from .mcp.json
    async with _server(tmp_path) as client:
        # When the agent records calls on FX pairs written in other ways
        refusals = {
            ticker: await client.call_tool(
                "record_decision",
                {
                    "ticker": ticker,
                    "direction": "pass",
                    "horizon": 10,
                    "reasoning": "An FX pair isn't a stock.",
                    "signals": [{"name": "trend", "stance": "for"}],
                },
            )
            for ticker in (" eurusd=x ", "EURNOK=X")
        }
        # And asks for candles on FX-looking tickers holding a path
        paths = {
            ticker: await client.call_tool("get_candles", {"ticker": ticker})
            for ticker in ("../X=X", "A/B=X")
        }

    # Then every call is refused as an exchange rate, and nothing is recorded
    for ticker, result in refusals.items():
        assert result.is_error, ticker
        assert "exchange rate, not a stock" in _text(result)
        assert ticker.strip().upper() in _text(result)
    assert not _journal(tmp_path).exists()
    # And the path-like tickers aren't tickers, with no folder written for them
    for ticker, result in paths.items():
        assert result.is_error, ticker
        assert "isn't a ticker" in _text(result)
    stocks = tmp_path / "stocks"
    assert not stocks.exists() or sorted(p.name for p in stocks.iterdir()) == []
