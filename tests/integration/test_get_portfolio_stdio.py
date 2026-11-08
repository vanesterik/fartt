"""
Integration tests for `get_portfolio` and the portfolio `plan_trade` and
`record_decision` are held to (#145), driven the way Claude Code drives the
server: started from `.mcp.json` over stdio, with `FARTT_ASSETS_DIR` set to a
temporary folder holding the snapshot, its clock pinned to the snapshot's
moment and Yahoo out of reach (`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-145` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and the
snapshot's candles, worked out by hand; costs are checked against `risk.toml`'s
rates, not against the server's own cost calculation.

As the design says, the plans are backdated in the journal the way #144's
tests do it (`_seed` in `test_settle_positions_stdio`): a real plan is
recorded, then rewritten to a past session of the snapshot, so its window and
horizon run over sessions the snapshot already holds. The snapshot's newest
closed session is 2026-11-04 (ASML.AS close 1337.6, low 1316.8).
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from tests.integration.risk import copy_risk_config
from tests.integration.test_settle_positions_stdio import (
    CENT,
    FILL_AT_ENTRY_THEN_STOP,
    FILL_THEN_GAP_TARGET,
    FILL_THEN_TARGET,
    GAP_FILL_THEN_GAP_STOP,
    HELD_TO_HORIZON,
    STILL_OPEN,
    STILL_WAITING,
    STOP_ON_FILL_SESSION,
    USD_HELD_TO_HORIZON,
    Plan,
    _call,
    _candles,
    _expected_costs,
    _journal,
    _seed,
    _server,
    _text,
)

pytestmark = pytest.mark.anyio

STARTING_CAPITAL = 10_000
LAST_SESSION = "2026-11-04"
LAST_CLOSE = 1337.6

# Filled at the open on 10-20 (1452.6); the 1.88 dividend goes ex on 10-26;
# no low after 10-19 reaches 1300 and no high 1700, horizon_ends 11-16: open.
OPEN_OVER_DIVIDEND = Plan("ASML.AS", "2026-10-19", 20, 5, 1460, 1300, 1700, 2)
# Filled at 1405 on 11-02 (low 1401.2); 11-04's low (1316.8) reaches the stop.
STOPPED_ON_NEWEST_SESSION = Plan("ASML.AS", "2026-10-30", 20, 5, 1405, 1330, 1700, 2)
# Recorded on the newest session, entry far below: waiting.
WAITING_ON_NEWEST_SESSION = Plan("ASML.AS", LAST_SESSION, 20, 5, 1300, 1200, 1700, 2)
# A plan that fits the starting capital: one ASML.AS share at 1330.
NEW_PLAN = {"entry": 1330, "stop": 1280, "target": 1500, "valid_for": 5}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _copy_assets(source: Path, target: Path) -> Path:
    """A second assets folder with `source`'s snapshot and risk config, no journal."""
    shutil.copytree(source / "stocks", target / "stocks")
    copy_risk_config(target)
    return target


def _sell_value(shares: int, price: float) -> float:
    """What selling `shares` of ASML.AS at `price` brings after the sell costs."""
    value = shares * price
    return value - _expected_costs(value, None, "XAMS", "sell")["total"]


async def _atr(client: Client) -> float:
    content = await _call(client, "get_indicators", ticker="ASML.AS")
    return content["atr"]["value"]


async def _refused(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error, result.structured_content
    return _text(result)


def _check_totals_add_up(portfolio: dict[str, Any]) -> None:
    totals = portfolio["totals"]
    assert totals["cash"] + totals["open_value"] == pytest.approx(
        totals["total_value"], abs=0.005
    )
    assert totals["starting_capital"] + totals["realised_pnl"] + totals[
        "unrealised_pnl"
    ] == pytest.approx(totals["total_value"], abs=0.005)


async def test_should_return_the_starting_capital_and_a_note_with_no_plans(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC1 Should return the starting capital as cash, free cash and
    total value, no open, waiting or closed positions, a drawdown of 0 and a
    note when no plan has been recorded.
    """
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    totals = portfolio["totals"]
    assert totals["starting_capital"] == STARTING_CAPITAL
    assert totals["cash"] == STARTING_CAPITAL
    assert totals["free_cash"] == STARTING_CAPITAL
    assert totals["total_value"] == STARTING_CAPITAL
    assert totals["drawdown"] == 0
    assert totals["open_positions"] == totals["waiting_plans"] == 0
    assert portfolio["open"] == [] and portfolio["waiting"] == []
    assert portfolio["closed"]["count"] == 0
    assert portfolio["closed"]["recent"] == []
    assert portfolio["note"]


async def test_should_show_an_open_position_against_its_plan(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC2 Should show an open position's shares, fill price as
    entry, stop, target, last close, value after the costs to close in EUR,
    unrealised P&L, distance to stop and target as a fraction and in ATR, and
    the sessions held.
    """
    await _seed(seeded_assets, [STILL_OPEN])
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")
        atr = await _atr(client)

    [held] = portfolio["open"]
    [filled] = portfolio["settled"]["filled"]
    value = _sell_value(2, LAST_CLOSE)
    assert held["id"] == 1 and held["ticker"] == "ASML.AS"
    assert held["shares"] == 2
    assert held["entry"] == 1405
    assert (held["stop"], held["target"]) == (1200, 1700)
    assert held["last_close"] == LAST_CLOSE
    assert held["based_on"] == LAST_SESSION
    assert held["value"] == pytest.approx(value, abs=CENT)
    assert held["unrealised_pnl"] == pytest.approx(
        value - 2 * 1405 - filled["costs"]["total"], abs=CENT
    )
    assert held["to_stop"] == pytest.approx((LAST_CLOSE - 1200) / LAST_CLOSE, abs=1e-4)
    assert held["to_target"] == pytest.approx(
        (1700 - LAST_CLOSE) / LAST_CLOSE, abs=1e-4
    )
    assert held["to_stop_atr"] == pytest.approx((LAST_CLOSE - 1200) / atr, abs=1e-3)
    assert held["to_target_atr"] == pytest.approx((1700 - LAST_CLOSE) / atr, abs=1e-3)
    # Filled 11-02; held over 11-03 and 11-04.
    assert held["sessions_held"] == 2
    assert held["is_current"] is True and held["warning"] is None


async def test_should_add_up_cash_open_value_and_pnl_to_the_cent(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC3 @AC9 Should make cash plus the open positions' value the
    total value, and the starting capital plus realised and unrealised P&L
    the same total, to the cent, with open and closed positions; the open
    risk is the further loss if every open stop is hit after costs, and the
    waiting plans' planned risk is returned on its own.
    """
    entries = await _seed(
        seeded_assets,
        [
            FILL_AT_ENTRY_THEN_STOP,
            FILL_THEN_TARGET,
            USD_HELD_TO_HORIZON,
            STILL_OPEN,
            OPEN_OVER_DIVIDEND,
            STILL_WAITING,
        ],
    )
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    totals = portfolio["totals"]
    _check_totals_add_up(portfolio)
    closed = portfolio["settled"]["closed"]
    assert len(closed) == 3
    assert totals["realised_pnl"] == pytest.approx(
        sum(c["pnl"] for c in closed), abs=CENT
    )
    assert totals["unrealised_pnl"] == pytest.approx(
        sum(h["unrealised_pnl"] for h in portfolio["open"]), abs=CENT
    )
    assert totals["open_value"] == pytest.approx(
        sum(h["value"] for h in portfolio["open"]), abs=CENT
    )
    assert totals["open_positions"] == 2 and totals["waiting_plans"] == 1

    # Both open positions are 2 ASML.AS shares valued at 11-04's close; one
    # stops at 1200, the other at 1300.
    value = _sell_value(2, LAST_CLOSE)
    at_stops = _sell_value(2, 1200) + _sell_value(2, 1300)
    assert totals["open_risk"] == pytest.approx(2 * value - at_stops, abs=2 * CENT)
    assert totals["waiting_risk"] == pytest.approx(entries[5]["plan"]["risk"], abs=CENT)


async def test_should_show_a_position_stopped_in_the_newest_session_as_closed(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC4 Should show a position whose stop was hit in the newest
    closed session as closed, not open, and list it in settled.closed.
    """
    await _seed(seeded_assets, [STOPPED_ON_NEWEST_SESSION])
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    assert portfolio["open"] == []
    assert portfolio["totals"]["open_positions"] == 0
    [closed] = portfolio["settled"]["closed"]
    assert (closed["id"], closed["session"], closed["reason"], closed["price"]) == (
        1,
        LAST_SESSION,
        "stop",
        1330,
    )
    assert portfolio["closed"]["count"] == 1
    assert portfolio["closed"]["recent"][0]["id"] == 1
    _check_totals_add_up(portfolio)


async def test_should_list_a_waiting_plan_and_take_its_reservation_off_the_free_cash(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC5 Should list a plan whose entry hasn't been reached as
    waiting, with its entry, stop, target, shares, reserved cash and the
    session it's valid until, and take the reservation off the free cash.
    """
    [entry] = await _seed(seeded_assets, [STILL_WAITING])
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    reservation = (
        entry["plan"]["position_value"] + entry["plan"]["costs"]["buy"]["total"]
    )
    [waiting] = portfolio["waiting"]
    assert waiting["id"] == 1 and waiting["ticker"] == "ASML.AS"
    assert (waiting["entry"], waiting["stop"], waiting["target"]) == (1300, 1200, 1700)
    assert waiting["shares"] == 2
    assert waiting["reserved"] == pytest.approx(reservation, abs=CENT)
    # Window of 5 sessions after 10-30.
    assert waiting["valid_until"] == "2026-11-06"
    totals = portfolio["totals"]
    assert totals["cash"] == STARTING_CAPITAL
    assert totals["reserved"] == pytest.approx(reservation, abs=CENT)
    assert totals["free_cash"] == pytest.approx(
        STARTING_CAPITAL - reservation, abs=CENT
    )
    assert totals["waiting_plans"] == 1


async def test_should_refuse_a_plan_beyond_the_free_cash_and_record_nothing(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC6 Should refuse in plan_trade and record_decision a plan
    whose position value plus buy costs is more than the free cash, naming
    the free cash and by how much it misses, and record nothing.
    """
    await _seed(seeded_assets, [WAITING_ON_NEWEST_SESSION])
    journal = _journal(seeded_assets)
    waiting = json.loads(journal.read_text())
    waiting["plan"]["position_value"] = 9000  # reserves most of the cash
    journal.write_text(json.dumps(waiting) + "\n")
    reserved = 9000 + waiting["plan"]["costs"]["buy"]["total"]
    free_cash = STARTING_CAPITAL - reserved
    need = 1330 + _expected_costs(1330, None, "XAMS", "buy")["total"]
    before = journal.read_bytes()

    async with _server(seeded_assets) as client:
        totals = (await _call(client, "get_portfolio"))["totals"]
        planned = await _refused(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)
        recorded = await _refused(
            client,
            "record_decision",
            ticker="ASML.AS",
            direction="buy",
            horizon=20,
            reasoning="A plan beyond the free cash.",
            signals=[{"name": "trend", "stance": "for"}],
            plan=NEW_PLAN,
        )

    assert totals["free_cash"] == pytest.approx(free_cash, abs=CENT)
    for message in (planned, recorded):
        assert f"€{free_cash:,.2f}" in message
        assert f"€{need - free_cash:,.2f}" in message
        assert "Nothing was recorded" in message
    assert journal.read_bytes() == before


async def test_should_size_a_plan_on_the_total_value_after_a_loss_or_gain(
    tmp_path: Path, seeded_assets: Path
) -> None:
    """
    @issue-145 @AC7 Should size plan_trade on the portfolio's total value:
    after a realised loss or gain the same plan's portfolio_value is the new
    total, and its shares of the portfolio are of that total.
    """
    results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, plans in (
        ("loss", [FILL_AT_ENTRY_THEN_STOP]),
        ("gain", [FILL_THEN_TARGET]),
    ):
        assets = _copy_assets(seeded_assets, tmp_path / name)
        await _seed(assets, plans)
        async with _server(assets) as client:
            portfolio = await _call(client, "get_portfolio")
            planned = await _call(client, "plan_trade", ticker="ASML.AS", **NEW_PLAN)
        results[name] = (portfolio, planned)

    for name, sign in (("loss", -1), ("gain", 1)):
        portfolio, planned = results[name]
        total = portfolio["totals"]["total_value"]
        realised = portfolio["totals"]["realised_pnl"]
        assert sign * realised > 0, name
        assert total == pytest.approx(STARTING_CAPITAL + realised, abs=CENT)
        assert planned["portfolio_value"] == pytest.approx(total, abs=CENT)
        assert planned["position_share"] == pytest.approx(
            planned["position_value"] / total, abs=1e-4
        )
        assert planned["risk_share"] == pytest.approx(planned["risk"] / total, abs=1e-4)


async def test_should_report_the_peak_and_the_drawdown_below_it(
    tmp_path: Path, seeded_assets: Path
) -> None:
    """
    @issue-145 @AC8 Should return the highest value at a session close since
    the first fill as the peak and the total's fall below it as the drawdown;
    after a close above the earlier peak the drawdown is 0.
    """
    loss_assets = _copy_assets(seeded_assets, tmp_path / "loss")
    await _seed(loss_assets, [FILL_AT_ENTRY_THEN_STOP])
    await _seed(seeded_assets, [FILL_THEN_TARGET])
    async with _server(loss_assets) as client:
        loss = await _call(client, "get_portfolio")
    async with _server(seeded_assets) as client:
        gain = await _call(client, "get_portfolio")

    # Filled 09-09 at 1475 (2 shares), stopped 09-10 at 1450. At 09-09's close
    # the portfolio was the cash after the fill plus 2 shares sold at that
    # close, above the starting capital: that's the peak.
    [filled] = loss["settled"]["filled"]
    close_0909 = _candles(loss_assets, "ASML.AS")["2026-09-09"]["Close"]
    peak = STARTING_CAPITAL - filled["value"] - filled["costs"]["total"]
    peak += _sell_value(2, close_0909)
    assert peak > STARTING_CAPITAL
    totals = loss["totals"]
    assert totals["peak"] == pytest.approx(peak, abs=CENT)
    assert totals["peak_session"] == "2026-09-09"
    assert totals["drawdown"] == pytest.approx(
        (peak - totals["total_value"]) / peak, abs=1e-4
    )
    assert totals["drawdown"] > 0

    # Closed at its target on 09-22: no earlier close was worth more.
    totals = gain["totals"]
    assert totals["total_value"] > STARTING_CAPITAL
    assert totals["peak"] == pytest.approx(totals["total_value"], abs=CENT)
    assert totals["peak_session"] == "2026-09-22"
    assert totals["drawdown"] == 0


async def test_should_summarise_closed_positions_by_reason_with_the_five_most_recent(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC10 Should return the closed positions as a count and
    realised P&L, split by stop, target and horizon, with the 5 most recent
    exits newest first.
    """
    await _seed(
        seeded_assets,
        [
            USD_HELD_TO_HORIZON,
            FILL_AT_ENTRY_THEN_STOP,
            STOP_ON_FILL_SESSION,
            FILL_THEN_TARGET,
            FILL_THEN_GAP_TARGET,
            GAP_FILL_THEN_GAP_STOP,
            HELD_TO_HORIZON,
        ],
    )
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    exits = portfolio["settled"]["closed"]
    summary = portfolio["closed"]
    assert summary["count"] == 7
    assert summary["realised_pnl"] == pytest.approx(
        sum(e["pnl"] for e in exits), abs=CENT
    )
    for reason, count in (("stop", 3), ("target", 2), ("horizon", 2)):
        group = summary["by_reason"][reason]
        assert group["count"] == count, reason
        assert group["pnl"] == pytest.approx(
            sum(e["pnl"] for e in exits if e["reason"] == reason), abs=CENT
        ), reason
    newest = sorted(exits, key=lambda e: e["session"], reverse=True)[:5]
    assert [(r["id"], r["session"], r["reason"]) for r in summary["recent"]] == [
        (e["id"], e["session"], e["reason"]) for e in newest
    ]
    assert [r["session"] for r in summary["recent"]] == [
        "2026-10-26",
        "2026-10-15",
        "2026-09-29",
        "2026-09-22",
        "2026-09-14",
    ]


async def test_should_value_a_stale_ticker_at_its_newest_cached_close(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC11 Should value a position whose ticker can't be brought up
    to date at its newest cached close, marked is_current false with a
    warning, and name the ticker in the note.
    """
    await _seed(seeded_assets, [STILL_OPEN])
    candles = seeded_assets / "stocks" / "ASML.AS" / "candles.csv"
    candles.write_text("".join(candles.read_text().splitlines(keepends=True)[:-1]))
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    [held] = portfolio["open"]
    assert held["last_close"] == 1385  # 11-03's close; 11-04 isn't cached
    assert held["based_on"] == "2026-11-03"
    assert held["value"] == pytest.approx(_sell_value(2, 1385), abs=CENT)
    assert held["is_current"] is False
    assert held["warning"] and "ASML.AS" in held["warning"]
    assert portfolio["note"] and "ASML.AS" in portfolio["note"]
    assert portfolio["totals"]["open_value"] == held["value"]
    _check_totals_add_up(portfolio)


async def test_should_include_a_dividend_gone_ex_while_open_in_the_unrealised_pnl(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @AC12 Should include a dividend that went ex while the
    position was open in its unrealised P&L.
    """
    await _seed(seeded_assets, [OPEN_OVER_DIVIDEND])
    async with _server(seeded_assets) as client:
        portfolio = await _call(client, "get_portfolio")

    [held] = portfolio["open"]
    [filled] = portfolio["settled"]["filled"]
    assert (filled["session"], filled["price"]) == ("2026-10-20", 1452.6)
    assert held["dividends"] == pytest.approx(2 * 1.88, abs=CENT)
    assert held["unrealised_pnl"] == pytest.approx(
        _sell_value(2, LAST_CLOSE) + 2 * 1.88 - 2 * 1452.6 - filled["costs"]["total"],
        abs=CENT,
    )
    _check_totals_add_up(portfolio)


async def test_should_write_nothing_and_answer_the_same_when_called_again(
    seeded_assets: Path,
) -> None:
    """
    @issue-145 @X Should leave the journal unchanged and return the same
    portfolio, with nothing newly settled, when called a second time.
    """
    await _seed(seeded_assets, [STILL_OPEN, STILL_WAITING, FILL_AT_ENTRY_THEN_STOP])
    async with _server(seeded_assets) as client:
        first = await _call(client, "get_portfolio")
        settled = _journal(seeded_assets).read_bytes()
        second = await _call(client, "get_portfolio")

    assert _journal(seeded_assets).read_bytes() == settled
    assert second["settled"] == {
        "filled": [],
        "expired": [],
        "closed": [],
        "not_settled": [],
    }
    assert {k: v for k, v in second.items() if k != "settled"} == {
        k: v for k, v in first.items() if k != "settled"
    }
