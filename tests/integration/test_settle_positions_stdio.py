"""
Integration tests for `settle_positions` and the `position` that
`get_decisions` shows (#144), driven the way Claude Code drives the server:
started from `.mcp.json` over stdio, with `FARTT_ASSETS_DIR` set to a
temporary folder holding the snapshot, its clock pinned to the snapshot's
moment and Yahoo out of reach (`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-144` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and the
snapshot's candles, worked out by hand below; the costs are checked against
`risk.toml`'s rates, not against the server's own cost calculation.

`record_decision` only records plans based on the newest closed session, so,
as the design says, each test records a real plan per ticker and then
backdates it in the journal before the server starts again: the plan's
session, entry, stop, target, window and shares are rewritten to a past
session of the snapshot, and the window and horizon then run over sessions
the snapshot already holds. Data that isn't there yet is simulated by cutting
a ticker's candles in the test's copy of the snapshot, and arrives by copying
them back.
"""

import csv
import json
import os
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from mcp import Client, StdioServerParameters

from fartt.data import exchange_calendar
from tests.integration.snapshot import OFFLINE

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
BENCHMARK = "IWDA.AS"
CENT = 0.0101
RISK = tomllib.loads((ROOT / "assets" / "risk.toml").read_text())

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(frozen=True)
class Plan:
    """A plan backdated to `session`: what the operator recorded then."""

    ticker: str
    session: str
    horizon: int
    valid_for: int
    entry: float
    stop: float
    target: float
    shares: int


# Prices of the snapshot (rounded to 4 decimals) each plan is built on.
# ASML.AS: 09-09 O 1520 L 1472.2; 09-10 O 1498.6 L 1447.4 H 1498.6.
FILL_AT_ENTRY_THEN_STOP = Plan("ASML.AS", "2026-09-04", 20, 5, 1475, 1450, 1600, 2)
# 10-13 O 1533 H 1561.6 L 1509.6; 10-14 O 1510 H 1564 L 1494; 10-15 O 1451.8.
GAP_FILL_THEN_GAP_STOP = Plan("ASML.AS", "2026-10-12", 10, 5, 1540, 1480, 1700, 2)
# 09-21 O 1482.6 H 1496.6 L 1462.8; 09-22 O 1489 H 1520.4 L 1468.8.
FILL_THEN_TARGET = Plan("ASML.AS", "2026-09-18", 20, 5, 1470, 1400, 1520, 2)
# 09-28 O 1515 H 1553.8 L 1498.6; 09-29 O 1569.
FILL_THEN_GAP_TARGET = Plan("ASML.AS", "2026-09-25", 20, 5, 1510, 1450, 1560, 2)
# Fill on 10-13 at the open, whose high (1561.6) reaches the target; 10-14
# reaches both the stop (low 1494) and the target (high 1564).
BOTH_IN_ONE_SESSION = Plan("ASML.AS", "2026-10-12", 10, 5, 1540, 1500, 1560, 2)
# 09-14 O 1429 L 1369.6: filled at the open, stopped the same session.
STOP_ON_FILL_SESSION = Plan("ASML.AS", "2026-09-11", 20, 5, 1440, 1380, 1500, 2)
# 10-20 O 1452.6; horizon_ends 10-26 (close 1430), ex-dividend 1.88 that day;
# lows down to 1384.4 and highs up to 1500.2 in between.
HELD_TO_HORIZON = Plan("ASML.AS", "2026-10-19", 5, 5, 1460, 1300, 1700, 2)
# No session after 10-19 has a low at or below 1300 (the lowest is 1316.8).
EXPIRES_AT_HORIZON = Plan("ASML.AS", "2026-10-19", 5, 10, 1300, 1200, 1700, 2)
EXPIRES_AT_WINDOW_END = Plan("ASML.AS", "2026-10-19", 20, 3, 1300, 1200, 1700, 2)
STILL_WAITING = Plan("ASML.AS", "2026-10-30", 20, 5, 1300, 1200, 1700, 2)
# 11-02 O 1415 L 1401.2: filled at the entry; nothing after reaches 1200/1700.
STILL_OPEN = Plan("ASML.AS", "2026-10-30", 20, 5, 1405, 1200, 1700, 2)
# 11-02 fill at 1405; 11-03 O 1367 L 1358: stopped at 1360.
STOPPED_ON_NOV_3 = Plan("ASML.AS", "2026-10-30", 20, 5, 1405, 1360, 1700, 2)
# AAPL: 08-06 O 314.34 L 309.23: filled at the entry; lows down to 300.57 and
# highs up to 319.28 until horizon_ends 08-19 (close 316.83); ex-dividend 0.27
# on 08-10.
USD_HELD_TO_HORIZON = Plan("AAPL", "2026-08-05", 10, 5, 310, 290, 330, 6)


@asynccontextmanager
async def _server(assets_dir: Path) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir), **OFFLINE}
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


def _candles(
    assets_dir: Path, ticker: str, decimals: int = 4
) -> dict[str, dict[str, float]]:
    """A ticker's cached candles by date; prices rounded as the server rounds them."""
    with (assets_dir / "stocks" / ticker / "candles.csv").open() as file:
        return {
            row["Date"]: {
                k: round(float(v), decimals) for k, v in row.items() if k != "Date"
            }
            for row in csv.DictReader(file)
        }


def _session_after(exchange: str, day: str, count: int) -> str:
    calendar = exchange_calendar(exchange)
    session = pd.Timestamp(day)
    for _ in range(count):
        session = calendar.next_session(session)
    return session.date().isoformat()


def _rate(assets_dir: Path, session: str) -> float:
    return _candles(assets_dir, "EURUSD=X", 6)[session]["Close"]


async def _seed(assets_dir: Path, plans: list[Plan]) -> list[dict[str, Any]]:
    """
    Records a real plan per ticker, then rewrites the journal so it holds
    `plans`, in order (ids 1, 2, ...), each backdated to its session.
    """
    templates: dict[str, dict[str, Any]] = {}
    async with _server(assets_dir) as client:
        for ticker, entry, stop, target in (
            ("ASML.AS", 1330, 1280, 1500),
            ("AAPL", 345, 335, 400),
        ):
            if ticker in {plan.ticker for plan in plans}:
                templates[ticker] = await _call(
                    client,
                    "record_decision",
                    ticker=ticker,
                    direction="buy",
                    horizon=20,
                    reasoning="Template for a backdated plan.",
                    signals=[{"name": "trend", "stance": "for"}],
                    plan={
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "valid_for": 5,
                    },
                )
    benchmark = _candles(assets_dir, BENCHMARK)
    entries = []
    for number, plan in enumerate(plans, start=1):
        entry = json.loads(json.dumps(templates[plan.ticker]))
        close = _candles(assets_dir, plan.ticker)[plan.session]["Close"]
        entry.update(
            kind="decision",
            id=number,
            recorded_at=f"{plan.session}T21:00:00Z",
            decision_session=plan.session,
            close=close,
            horizon=plan.horizon,
            horizon_ends=_session_after(entry["exchange"], plan.session, plan.horizon),
        )
        entry["benchmark"].update(
            session=plan.session, close=benchmark[plan.session]["Close"]
        )
        entry["plan"].update(
            based_on=plan.session,
            last_close=close,
            entry=plan.entry,
            stop=plan.stop,
            target=plan.target,
            valid_for=plan.valid_for,
            shares=plan.shares,
        )
        entries.append(entry)
    _journal(assets_dir).write_text("".join(json.dumps(e) + "\n" for e in entries))
    return entries


def _expected_costs(
    value: float, rate: float | None, exchange: str, side: str
) -> dict[str, float]:
    """A leg's costs in EUR from risk.toml's rates; `value` in the stock's currency."""
    table = RISK["costs"]["exchanges"][exchange]
    to_eur = rate or 1.0
    commission = min(
        max(value * table["commission"], table["commission_min"]),
        table.get("commission_max", float("inf")),
    )
    commission_eur = commission / to_eur
    value_eur = value / to_eur
    costs = {
        "commission": commission_eur,
        "slippage": value_eur * RISK["costs"]["slippage_per_leg"],
        "conversion": value_eur * RISK["costs"]["currency_conversion"] if rate else 0,
        "tax": value_eur * table["tax_on_buy" if side == "buy" else "fee_on_sell"],
        "fee": table["fee_per_trade"] / to_eur,
    }
    return {**costs, "total": sum(costs.values())}


def _check_costs(
    actual: dict[str, float], value: float, rate: float | None, exchange: str, side: str
) -> None:
    expected = _expected_costs(value, rate, exchange, side)
    for item in ("commission", "slippage", "conversion", "tax", "fee"):
        assert actual[item] == pytest.approx(expected[item], abs=CENT), item
    parts = sum(
        actual[i] for i in ("commission", "slippage", "conversion", "tax", "fee")
    )
    assert actual["total"] == pytest.approx(parts, abs=CENT)


def _check_pnl(fill: dict[str, Any], exit_: dict[str, Any]) -> None:
    expected = (
        exit_["value"]
        - fill["value"]
        + exit_["dividends"]
        - fill["costs"]["total"]
        - exit_["costs"]["total"]
    )
    assert exit_["pnl"] == pytest.approx(expected, abs=CENT)


def _by_id(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {item["id"]: item for item in items}


async def _position(client: Client, decision_id: int) -> dict[str, Any] | None:
    content = await _call(client, "get_decisions", id=decision_id)
    return content["decisions"][0]["position"]


async def test_should_fill_at_the_entry_and_show_the_position_open(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC1 Should fill a plan at its entry on the first window
    session whose low reaches it, with the shares, EUR value and itemised buy
    costs, and show the position open in get_decisions.
    """
    await _seed(seeded_assets, [STILL_OPEN])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")
        position = await _position(client, 1)

    [filled] = run["filled"]
    assert filled["id"] == 1
    assert filled["session"] == "2026-11-02"
    assert filled["price"] == 1405
    assert filled["at_open"] is False
    assert filled["shares"] == 2
    assert filled["rate"] is None
    assert filled["value"] == pytest.approx(2 * 1405, abs=CENT)
    _check_costs(filled["costs"], 2 * 1405, None, "XAMS", "buy")
    assert run["closed"] == [] and run["expired"] == []
    assert [o["id"] for o in run["open"]] == [1]

    assert position is not None
    assert position["state"] == "open"
    assert position["fill"]["session"] == "2026-11-02"
    assert position["fill"]["price"] == 1405
    assert position["exit"] is None and position["expiry"] is None


async def test_should_fill_at_the_open_when_the_session_opens_at_or_below_the_entry(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC2 Should fill at the window session's open when it opens at
    or below the entry.
    """
    await _seed(seeded_assets, [GAP_FILL_THEN_GAP_STOP, HELD_TO_HORIZON])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")

    filled = _by_id(run["filled"])
    assert (filled[1]["session"], filled[1]["price"], filled[1]["at_open"]) == (
        "2026-10-13",
        1533,
        True,
    )
    assert (filled[2]["session"], filled[2]["price"], filled[2]["at_open"]) == (
        "2026-10-20",
        1452.6,
        True,
    )
    _check_costs(filled[1]["costs"], 2 * 1533, None, "XAMS", "buy")


async def test_should_wait_then_expire_an_unreached_order_within_the_horizon(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC3 Should keep an unreached order waiting while window
    sessions are to come, and expire it with no position on the window's last
    session, never past the call's horizon_ends.
    """
    await _seed(
        seeded_assets, [EXPIRES_AT_HORIZON, EXPIRES_AT_WINDOW_END, STILL_WAITING]
    )
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")
        positions = [await _position(client, i) for i in (1, 2, 3)]

    # Window of 10 sessions after 10-19 would end 11-02; horizon_ends is 10-26.
    assert _by_id(run["expired"]) == {
        1: {"id": 1, "ticker": "ASML.AS", "session": "2026-10-26"},
        2: {"id": 2, "ticker": "ASML.AS", "session": "2026-10-22"},
    }
    assert run["filled"] == []
    assert run["waiting"] == [
        {"id": 3, "ticker": "ASML.AS", "entry": 1300, "until": "2026-11-06"}
    ]
    expired, also_expired, waiting = positions
    assert expired is not None and also_expired is not None and waiting is not None
    assert expired["state"] == "expired"
    assert expired["fill"] is None and expired["exit"] is None
    assert expired["expiry"]["session"] == "2026-10-26"
    assert also_expired["expiry"]["session"] == "2026-10-22"
    assert waiting == {"state": "waiting", "fill": None, "exit": None, "expiry": None}


async def test_should_close_at_the_stop_with_the_pnl_after_both_legs(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC4 Should close a position whose session low reaches the stop
    at the stop, reason stop, with the P&L in EUR as exit value minus fill
    value, minus both legs' costs, plus dividends.
    """
    await _seed(seeded_assets, [FILL_AT_ENTRY_THEN_STOP])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")
        position = await _position(client, 1)

    [filled] = run["filled"]
    [closed] = run["closed"]
    assert (filled["session"], filled["price"], filled["at_open"]) == (
        "2026-09-09",
        1475,
        False,
    )
    assert closed["session"] == "2026-09-10"
    assert closed["reason"] == "stop"
    assert closed["price"] == 1450
    assert closed["at_open"] is False
    assert closed["slippage"] == 0
    assert closed["value"] == pytest.approx(2 * 1450, abs=CENT)
    assert closed["dividends"] == 0
    _check_costs(closed["costs"], 2 * 1450, None, "XAMS", "sell")
    _check_pnl(filled, closed)
    assert closed["pnl"] < 0
    assert position is not None and position["state"] == "closed"
    assert position["exit"]["pnl"] == closed["pnl"]


async def test_should_close_at_the_open_and_record_the_slippage_on_a_gap_below_the_stop(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC5 Should close a position whose session opens below the stop
    at that open, with the slippage beyond the stop on the exit.
    """
    await _seed(seeded_assets, [GAP_FILL_THEN_GAP_STOP])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")
        position = await _position(client, 1)

    [closed] = run["closed"]
    assert closed["session"] == "2026-10-15"
    assert closed["reason"] == "stop"
    assert closed["price"] == 1451.8
    assert closed["at_open"] is True
    assert closed["slippage"] == pytest.approx(1480 - 1451.8, abs=1e-4)
    _check_pnl(run["filled"][0], closed)
    assert position is not None
    assert position["exit"]["slippage"] == closed["slippage"]


async def test_should_close_at_the_target_or_at_an_open_above_it(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC6 Should close a position whose session high reaches the
    target at the target, reason target, or at the open when it opens at or
    above it.
    """
    await _seed(seeded_assets, [FILL_THEN_TARGET, FILL_THEN_GAP_TARGET])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")

    closed = _by_id(run["closed"])
    filled = _by_id(run["filled"])
    assert (
        closed[1]["session"],
        closed[1]["reason"],
        closed[1]["price"],
        closed[1]["at_open"],
        closed[1]["slippage"],
    ) == ("2026-09-22", "target", 1520, False, 0)
    assert (
        closed[2]["session"],
        closed[2]["reason"],
        closed[2]["price"],
        closed[2]["at_open"],
        closed[2]["slippage"],
    ) == ("2026-09-29", "target", 1569, True, 0)
    for i in (1, 2):
        _check_pnl(filled[i], closed[i])
        assert closed[i]["pnl"] > 0


async def test_should_take_the_stop_when_a_session_reaches_both_and_ignore_the_target_on_the_fill_session(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC7 Should close at the stop when a session's range reaches
    both stop and target; on the fill session a high at the target doesn't
    close the position, a low at the stop does.
    """
    await _seed(seeded_assets, [BOTH_IN_ONE_SESSION, STOP_ON_FILL_SESSION])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")

    filled = _by_id(run["filled"])
    closed = _by_id(run["closed"])
    # Filled 10-13, whose high reached the target: not closed that session.
    assert filled[1]["session"] == "2026-10-13"
    assert (closed[1]["session"], closed[1]["reason"], closed[1]["price"]) == (
        "2026-10-14",
        "stop",
        1500,
    )
    # Filled at the open on 09-14, whose low then reached the stop.
    assert (filled[2]["session"], filled[2]["price"]) == ("2026-09-14", 1429)
    assert (
        closed[2]["session"],
        closed[2]["reason"],
        closed[2]["price"],
        closed[2]["at_open"],
    ) == ("2026-09-14", "stop", 1380, False)


async def test_should_close_at_the_horizon_close_when_neither_stop_nor_target_is_reached(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC8 @AC9 Should close a position that reaches neither stop nor
    target at the close of horizon_ends, reason horizon, adding the dividend
    that went ex while it was held.
    """
    await _seed(seeded_assets, [HELD_TO_HORIZON])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")

    [closed] = run["closed"]
    assert closed["session"] == "2026-10-26"
    assert closed["reason"] == "horizon"
    assert closed["price"] == 1430
    assert closed["at_open"] is False
    assert closed["dividends"] == pytest.approx(2 * 1.88, abs=CENT)
    _check_pnl(run["filled"][0], closed)


async def test_should_convert_each_leg_of_a_usd_stock_at_its_own_session_rate(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC9 Should convert a USD stock's fill and exit each at its own
    session's EUR rate, given on the line, and add a dividend that went ex
    while held to the P&L.
    """
    await _seed(seeded_assets, [USD_HELD_TO_HORIZON])
    async with _server(seeded_assets) as client:
        run = await _call(client, "settle_positions")

    [filled] = run["filled"]
    [closed] = run["closed"]
    fill_rate = _rate(seeded_assets, "2026-08-06")
    exit_rate = _rate(seeded_assets, "2026-08-19")
    assert fill_rate != exit_rate
    assert (filled["session"], filled["price"]) == ("2026-08-06", 310)
    assert filled["rate"] == pytest.approx(fill_rate, abs=1e-6)
    assert filled["value"] == pytest.approx(6 * 310 / fill_rate, abs=CENT)
    _check_costs(filled["costs"], 6 * 310, fill_rate, "XNYS", "buy")
    assert (closed["session"], closed["reason"], closed["price"]) == (
        "2026-08-19",
        "horizon",
        316.83,
    )
    assert closed["rate"] == pytest.approx(exit_rate, abs=1e-6)
    assert closed["value"] == pytest.approx(6 * 316.83 / exit_rate, abs=CENT)
    _check_costs(closed["costs"], 6 * 316.83, exit_rate, "XNYS", "sell")
    assert closed["dividends"] == pytest.approx(6 * 0.27 / exit_rate, abs=CENT)
    _check_pnl(filled, closed)


async def test_should_write_nothing_when_settled_again_and_keep_everything_across_a_restart(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC10 @AC11 Should leave the journal byte-for-byte unchanged
    and list nothing new when settled again, and return every position,
    fill, expiry and exit as before after a restart.
    """
    plans = [
        FILL_AT_ENTRY_THEN_STOP,
        EXPIRES_AT_HORIZON,
        STILL_WAITING,
        STILL_OPEN,
        USD_HELD_TO_HORIZON,
    ]
    await _seed(seeded_assets, plans)
    ids = range(1, len(plans) + 1)
    async with _server(seeded_assets) as client:
        first = await _call(client, "settle_positions")
        written = _journal(seeded_assets).read_bytes()
        second = await _call(client, "settle_positions")
        before = [await _call(client, "get_decisions", id=i) for i in ids]
        listed_before = await _call(client, "get_decisions")
    assert _journal(seeded_assets).read_bytes() == written
    async with _server(seeded_assets) as client:
        after = [await _call(client, "get_decisions", id=i) for i in ids]
        listed_after = await _call(client, "get_decisions")

    assert len(first["filled"]) == 3
    assert len(first["closed"]) == 2
    assert len(first["expired"]) == 1
    assert (second["filled"], second["expired"], second["closed"]) == ([], [], [])
    assert [p["id"] for p in second["open"]] == [4]
    assert [w["id"] for w in second["waiting"]] == [3]
    states = [entry["decisions"][0]["position"]["state"] for entry in before]
    assert states == ["closed", "expired", "waiting", "open", "closed"]
    assert after == before
    assert listed_after == listed_before


async def test_should_hold_a_plan_whose_session_is_not_cached_and_settle_it_later(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC12 Should list a plan whose next session isn't cached under
    not_settled with the reason, write nothing for it, and settle it once the
    session is cached.
    """
    await _seed(seeded_assets, [STOPPED_ON_NOV_3])
    candles = seeded_assets / "stocks" / "ASML.AS" / "candles.csv"
    full = candles.read_text()
    candles.write_text("".join(full.splitlines(keepends=True)[:-2]))  # to 11-02

    async with _server(seeded_assets) as client:
        cut = await _call(client, "settle_positions")
    after_cut = _journal(seeded_assets).read_bytes()
    async with _server(seeded_assets) as client:
        again = await _call(client, "settle_positions")
    assert _journal(seeded_assets).read_bytes() == after_cut

    candles.write_text(full)
    async with _server(seeded_assets) as client:
        later = await _call(client, "settle_positions")
        position = await _position(client, 1)

    # The fill on 11-02 is cached and stored; the stop on 11-03 isn't yet.
    assert [f["session"] for f in cut["filled"]] == ["2026-11-02"]
    assert cut["closed"] == []
    [blocked] = cut["not_settled"]
    assert blocked["id"] == 1
    assert "2026-11-03" in blocked["reason"]
    assert "(the provider is unreachable)" in blocked["reason"]
    assert again["filled"] == [] and again["closed"] == []
    assert [b["id"] for b in again["not_settled"]] == [1]

    assert later["not_settled"] == []
    [closed] = later["closed"]
    assert (closed["session"], closed["reason"], closed["price"]) == (
        "2026-11-03",
        "stop",
        1360,
    )
    assert position is not None and position["state"] == "closed"


async def test_should_hold_a_usd_plan_whose_rate_is_not_cached_and_settle_it_later(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC12 Should list a USD plan whose exit session's EUR rate isn't
    cached under not_settled naming the rate, write nothing for it, and close
    it once the rate is cached.
    """
    await _seed(seeded_assets, [USD_HELD_TO_HORIZON])
    rates = seeded_assets / "stocks" / "EURUSD=X" / "candles.csv"
    full = rates.read_text()
    rates.write_text(
        "".join(
            line
            for line in full.splitlines(keepends=True)
            if line.startswith("Date") or line[:10] <= "2026-08-18"
        )
    )

    async with _server(seeded_assets) as client:
        cut = await _call(client, "settle_positions")
    rates.write_text(full)
    async with _server(seeded_assets) as client:
        later = await _call(client, "settle_positions")

    assert [f["session"] for f in cut["filled"]] == ["2026-08-06"]
    assert cut["closed"] == []
    [blocked] = cut["not_settled"]
    assert blocked["id"] == 1
    assert "EUR/USD rate for 2026-08-19" in blocked["reason"]
    assert later["not_settled"] == []
    [closed] = later["closed"]
    assert (closed["session"], closed["reason"]) == ("2026-08-19", "horizon")
    assert closed["rate"] == pytest.approx(_rate(seeded_assets, "2026-08-19"), abs=1e-6)


async def test_should_not_settle_a_position_across_a_split_after_its_fill(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC12 Should list an open position whose stock split after the
    fill under not_settled naming the split, write nothing for it, and keep
    it there on a later run.
    """
    await _seed(seeded_assets, [STOPPED_ON_NOV_3])
    candles = seeded_assets / "stocks" / "ASML.AS" / "candles.csv"
    full = candles.read_text()
    candles.write_text("".join(full.splitlines(keepends=True)[:-2]))  # to 11-02
    async with _server(seeded_assets) as client:
        cut = await _call(client, "settle_positions")
    assert [f["session"] for f in cut["filled"]] == ["2026-11-02"]

    candles.write_text(full)
    events = seeded_assets / "stocks" / "ASML.AS" / "events.csv"
    events.write_text(events.read_text() + "2026-11-03,split,2.0\n")
    filled = _journal(seeded_assets).read_bytes()
    async with _server(seeded_assets) as client:
        split = await _call(client, "settle_positions")
        again = await _call(client, "settle_positions")
        position = await _position(client, 1)

    assert _journal(seeded_assets).read_bytes() == filled
    for run in (split, again):
        assert run["closed"] == []
        [blocked] = run["not_settled"]
        assert blocked["id"] == 1
        assert "split" in blocked["reason"]
        assert "2026-11-03" in blocked["reason"]
    assert position is not None and position["state"] == "open"


async def test_should_show_no_position_without_a_plan_and_a_note_with_nothing_to_settle(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC13 Should give a call without a plan position null, and
    return empty lists and a note when no plan is left to settle.
    """
    async with _server(seeded_assets) as client:
        empty = await _call(client, "settle_positions")
        await _call(
            client,
            "record_decision",
            ticker="ASML.AS",
            direction="pass",
            horizon=20,
            reasoning="No plan.",
            signals=[{"name": "trend", "stance": "against"}],
        )
        without_plan = await _call(client, "settle_positions")
        decisions = await _call(client, "get_decisions")

    lists = ("filled", "expired", "closed", "open", "waiting", "not_settled")
    for run in (empty, without_plan):
        assert all(run[name] == [] for name in lists)
        assert run["note"]
    assert decisions["decisions"][0]["position"] is None


async def test_should_note_when_every_plan_is_closed_or_expired(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @AC13 Should return empty lists and a note once every plan is
    closed or expired.
    """
    await _seed(seeded_assets, [FILL_AT_ENTRY_THEN_STOP, EXPIRES_AT_HORIZON])
    async with _server(seeded_assets) as client:
        await _call(client, "settle_positions")
        done = await _call(client, "settle_positions")

    lists = ("filled", "expired", "closed", "open", "waiting", "not_settled")
    assert all(done[name] == [] for name in lists)
    assert done["note"]


async def test_should_refuse_a_damaged_journal_naming_the_file(
    seeded_assets: Path,
) -> None:
    """
    @issue-144 @X Should refuse with a ToolError naming the journal when a
    fill line has no call before it, and write nothing.
    """
    await _seed(seeded_assets, [STILL_OPEN])
    journal = _journal(seeded_assets)
    orphan = {
        "kind": "fill",
        "decision_id": 99,
        "settled_at": "2026-11-05T12:00:00Z",
        "session": "2026-11-02",
        "price": 1405.0,
        "at_open": False,
        "shares": 2,
        "rate": None,
        "value": 2810.0,
        "costs": {
            "commission": 2.25,
            "slippage": 2.81,
            "conversion": 0.0,
            "tax": 0.0,
            "fee": 0.0,
            "total": 5.06,
        },
    }
    journal.write_text(journal.read_text() + json.dumps(orphan) + "\n")
    before = journal.read_bytes()
    async with _server(seeded_assets) as client:
        result = await client.call_tool("settle_positions", {})

    assert result.is_error
    assert str(journal) in _text(result)
    assert journal.read_bytes() == before
