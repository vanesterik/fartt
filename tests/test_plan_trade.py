from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from fartt.data import Candle, History, Listing
from fartt.journal import DecisionDraft
from fartt.server import build_server
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from tests.fakes import (
    RISK_CONFIG,
    FakeProvider,
    buy_plan,
    decision_draft,
    exit_line,
    fill_line,
)
from tests.test_server import (
    AAPL,
    AFTER_CLOSE,
    ASML,
    IWDA,
    cache_for,
    call,
    history,
    journal_of,
    seed,
    sessions,
)

# Euronext Amsterdam has closed (plus margin) for 2026-10-30.
AFTER_CLOSE_30 = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)
EURUSD = Listing("EURUSD=X", "EUR/USD", "USD", "24/5")
RATE = 1.16
# 30 sessions closing 100, 101, ... 129, each ranging 2 around its close: ATR 2.
LAST_CLOSE = 129.0
ATR = 2.0


def fx_history(days: list[date]) -> History:
    return History([Candle(day, RATE, RATE, RATE, RATE, 0.0) for day in days], [], [])


def provider(
    asml_last: date = date(2026, 10, 29),
    count: int = 30,
    rate_last: date = date(2026, 10, 28),
) -> FakeProvider:
    return FakeProvider(
        histories={
            "ASML.AS": history(sessions("XAMS", count, last=asml_last)),
            "AAPL": history(sessions("XNYS", 30)),
            "IWDA.AS": history(sessions("XAMS", 30), first=50.0),
            "EURUSD=X": fx_history(sessions("24/5", 30, last=rate_last)),
        },
        listings={"ASML.AS": ASML, "AAPL": AAPL, "IWDA.AS": IWDA, "EURUSD=X": EURUSD},
        now=AFTER_CLOSE,
    )


def arguments(**changes: Any) -> dict[str, Any]:
    return {
        "ticker": "ASML.AS",
        "entry": 128.0,
        "stop": 120.0,
        "target": 150.0,
    } | changes


async def plan_trade(cache: StockCache, **changes: Any) -> Any:
    return await call(cache, arguments(**changes), "plan_trade")


@pytest.mark.anyio
async def test_plan_trade_returns_the_sized_plan_and_records_nothing(
    tmp_path: Path,
) -> None:
    # @AC1
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    assert (content["ticker"], content["currency"], content["exchange"]) == (
        "ASML.AS",
        "EUR",
        "XAMS",
    )
    assert content["based_on"] == "2026-10-29"
    assert (content["last_close"], content["atr"]) == (LAST_CLOSE, ATR)
    assert (content["entry"], content["stop"], content["target"]) == (
        128.0,
        120.0,
        150.0,
    )
    assert content["valid_for"] == 5
    assert content["rate"] is None
    # 2,000 EUR / 128 is 15 shares; their risk is far below 200.
    assert content["shares"] == 15
    assert content["sized_by"] == "max_position_size"
    assert content["position_value"] == 1920.0
    assert content["portfolio_value"] == 10000.0
    buy = content["costs"]["buy"]
    assert buy == {
        "commission": 2.0,
        "slippage": 1.92,
        "conversion": 0.0,
        "tax": 0.0,
        "fee": 0.0,
        "total": 3.92,
    }
    assert set(content["costs"]) == {"buy", "sell_at_stop", "sell_at_target"}
    assert content["risk"] == round(
        15 * 8 + buy["total"] + content["costs"]["sell_at_stop"]["total"], 2
    )
    assert content["reward_to_risk_before_costs"] == 2.75
    assert content["reward_to_risk"] == round(content["reward"] / content["risk"], 4)
    assert content["limits"] == RISK_CONFIG.limits.model_dump()
    assert content["tariff"]["name"] == RISK_CONFIG.costs.tariff
    assert content["tariff"]["exchange"] == RISK_CONFIG.costs.exchanges[
        "XAMS"
    ].model_dump(mode="json")
    assert not journal_of(cache).path.exists()


@pytest.mark.anyio
async def test_a_us_plan_is_converted_at_the_cached_rate(tmp_path: Path) -> None:
    # @AC10
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, ticker="AAPL")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["rate"] == RATE
    # At 21:00 UTC on 29 October the rate's 29 October session hasn't closed.
    assert content["rate_session"] == "2026-10-28"
    assert "midnight UTC" in content["warning"]
    # 2,000 EUR is 2,320 USD: 18 shares of 128.
    assert content["shares"] == 18
    assert content["position_value"] == round(18 * 128 / RATE, 2)
    assert content["costs"]["buy"]["conversion"] == round(18 * 128 / RATE * 0.0025, 2)
    assert content["risk"] == round(
        18 * 8 / RATE
        + content["costs"]["buy"]["total"]
        + content["costs"]["sell_at_stop"]["total"],
        2,
    )


@pytest.mark.anyio
async def test_a_rate_behind_its_newest_session_comes_with_a_warning(
    tmp_path: Path,
) -> None:
    # @AC10
    cache = cache_for(provider(rate_last=date(2026, 10, 26)), tmp_path)

    result = await plan_trade(cache, ticker="AAPL")

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["rate_session"] == "2026-10-26"
    assert "couldn't bring it up to date" in content["warning"]


@pytest.mark.anyio
async def test_a_euro_plan_has_no_rate_session_or_warning(tmp_path: Path) -> None:
    cache = cache_for(provider(), tmp_path)

    content = (await plan_trade(cache)).structured_content

    assert (content["rate"], content["rate_session"], content["warning"]) == (
        None,
        None,
        None,
    )


@pytest.mark.anyio
async def test_valid_for_is_returned(tmp_path: Path) -> None:
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, valid_for=10)

    assert result.structured_content["valid_for"] == 10


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"stop": 128.0}, "stop"),
        ({"target": 127.0}, "target"),
        ({"entry": 131.5}, "1 ATR"),
        ({"entry": -1.0}, "entry"),
        ({"valid_for": 11}, "valid_for"),
        ({"valid_for": 0}, "valid_for"),
        ({"target": 130.0}, "min_reward_to_risk"),
    ],
)
async def test_a_plan_breaking_a_check_is_refused(
    tmp_path: Path, changes: dict[str, Any], named: str
) -> None:
    # @AC4 @AC6
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, **changes)

    assert result.is_error
    assert named in result.content[0].text


@pytest.mark.anyio
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-inf"])
async def test_a_target_that_is_not_a_number_is_refused(
    tmp_path: Path, value: str
) -> None:
    # @AC6 @AC12
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, target=value)

    assert result.is_error
    assert "target must be a finite number" in result.content[0].text


@pytest.mark.anyio
async def test_a_target_too_large_to_work_out_is_refused(tmp_path: Path) -> None:
    # @AC6 @AC12
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, target=1e308)

    assert result.is_error
    assert "target (1e+308) is too large" in result.content[0].text


@pytest.mark.anyio
@pytest.mark.parametrize("entry", [1e-306, 1e-320])
async def test_an_entry_too_small_to_work_out_is_refused(
    tmp_path: Path, entry: float
) -> None:
    # @AC12
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, entry=entry, stop=entry / 10, target=1.0)

    assert result.is_error
    assert f"entry ({entry}) is too small" in result.content[0].text


@pytest.mark.anyio
async def test_a_ticker_without_its_newest_close_is_refused(tmp_path: Path) -> None:
    # @AC7
    cache = cache_for(provider(asml_last=date(2026, 10, 28)), tmp_path)

    result = await plan_trade(cache)

    assert result.is_error
    text = result.content[0].text
    assert "ASML.AS's newest close (2026-10-29)" in text
    assert "Try again later" in text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("ticker", "named"),
    [
        ("eurusd=x", "exchange rate, not a stock"),
        ("NOPE", "NOPE"),
        ("BMW.DE", ".AS"),
    ],
)
async def test_a_ticker_record_decision_refuses_is_refused(
    tmp_path: Path, ticker: str, named: str
) -> None:
    # @AC7
    cache = cache_for(provider(), tmp_path)

    result = await plan_trade(cache, ticker=ticker)

    assert result.is_error
    assert named in result.content[0].text


@pytest.mark.anyio
async def test_a_history_too_short_for_an_atr_is_refused(tmp_path: Path) -> None:
    cache = cache_for(provider(count=5), tmp_path)

    result = await plan_trade(cache, entry=103.0, stop=100.0, target=120.0)

    assert result.is_error
    assert "ATR" in result.content[0].text


def test_the_instructions_name_plan_trade() -> None:
    assert "plan_trade" in server_module.INSTRUCTIONS


async def record(cache: StockCache, **changes: Any) -> Any:
    arguments: dict[str, Any] = {
        "ticker": "ASML.AS",
        "direction": "buy",
        "horizon": 20,
        "reasoning": "Trend up, support holding.",
        "signals": [{"name": "trend", "stance": "for"}],
    }
    return await call(cache, arguments | changes, "record_decision")


PLAN = {"entry": 128.0, "stop": 120.0, "target": 150.0, "valid_for": 3}


@pytest.mark.anyio
async def test_a_buy_is_recorded_with_its_plan(tmp_path: Path) -> None:
    # @AC8
    cache = cache_for(provider(), tmp_path)
    planned = (await plan_trade(cache, valid_for=3)).structured_content

    result = await record(cache, plan=PLAN)

    assert not result.is_error, result.content
    assert result.structured_content["plan"] == planned
    stored = journal_of(cache).decisions()[0]
    assert stored.plan is not None
    assert stored.plan.model_dump(mode="json") == planned
    read = await call(cache, {"id": 1}, "get_decisions")
    assert read.structured_content["decisions"][0]["plan"] == planned


@pytest.mark.anyio
async def test_a_buy_whose_plan_breaks_a_limit_records_nothing(
    tmp_path: Path,
) -> None:
    # @AC8
    cache = cache_for(provider(), tmp_path)
    refused = await plan_trade(cache, target=130.0)

    result = await record(cache, plan=PLAN | {"target": 130.0})

    assert result.is_error
    # The same words, after the SDK's "Error executing tool <name>: ".
    assert (
        result.content[0].text.split(": ", 1)[1]
        == (refused.content[0].text.split(": ", 1)[1])
    )
    assert not journal_of(cache).path.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("value", ["NaN", "Infinity"])
async def test_a_buy_whose_target_is_not_a_number_records_nothing(
    tmp_path: Path, value: str
) -> None:
    # @AC8 @AC12
    cache = cache_for(provider(), tmp_path)

    result = await record(cache, plan=PLAN | {"target": value})

    assert result.is_error
    assert "target must be a finite number" in result.content[0].text
    assert not journal_of(cache).path.exists()


@pytest.mark.anyio
async def test_a_buy_whose_target_is_too_large_records_nothing(
    tmp_path: Path,
) -> None:
    # @AC8 @AC12
    cache = cache_for(provider(), tmp_path)

    result = await record(cache, plan=PLAN | {"target": 1e308})

    assert result.is_error
    assert "target (1e+308) is too large" in result.content[0].text
    assert not journal_of(cache).path.exists()


@pytest.mark.anyio
async def test_a_buy_whose_entry_is_too_small_records_nothing(
    tmp_path: Path,
) -> None:
    # @AC12
    cache = cache_for(provider(), tmp_path)

    result = await record(
        cache, plan=PLAN | {"entry": 1e-306, "stop": 1e-307, "target": 1.0}
    )

    assert result.is_error
    assert "entry (1e-306) is too small" in result.content[0].text
    assert not journal_of(cache).path.exists()


@pytest.mark.anyio
async def test_a_pass_with_a_plan_is_refused(tmp_path: Path) -> None:
    # @AC9
    cache = cache_for(provider(), tmp_path)

    result = await record(cache, direction="pass", plan=PLAN)

    assert result.is_error
    text = result.content[0].text
    assert "pass" in text
    assert "plan" in text
    assert not journal_of(cache).path.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("direction", ["buy", "pass"])
async def test_a_call_without_a_plan_is_recorded_with_plan_null(
    tmp_path: Path, direction: str
) -> None:
    # @AC9
    cache = cache_for(provider(), tmp_path)

    result = await record(cache, direction=direction)

    assert not result.is_error, result.content
    assert result.structured_content["plan"] is None
    assert journal_of(cache).decisions()[0].plan is None


@pytest.mark.anyio
async def test_a_session_closing_during_the_call_does_not_split_call_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # @AC8: the stored call and its plan describe the same session.
    fake = provider()
    cache = cache_for(fake, tmp_path)
    snapshot = server_module._snapshot  # pyright: ignore[reportPrivateUsage]

    def snapshot_then_close_a_session(
        tool: str, cache: StockCache, ticker: str, count: int
    ) -> Any:
        result = snapshot(tool, cache, ticker, count)
        if ticker == "IWDA.AS":
            # ASML.AS's next session closes once the call's closes are read.
            fake.now = AFTER_CLOSE_30
            fake.histories["ASML.AS"].candles.append(
                Candle(date(2026, 10, 30), 129.5, 131.0, 128.0, 130.0, 1000.0)
            )
        return result

    monkeypatch.setattr(server_module, "_snapshot", snapshot_then_close_a_session)

    result = await record(cache, plan=PLAN)

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["decision_session"] == "2026-10-29"
    assert content["plan"]["based_on"] == "2026-10-29"
    assert content["plan"]["last_close"] == content["close"]


SUMMARY = [
    "entry",
    "stop",
    "target",
    "valid_for",
    "shares",
    "position_value",
    "risk",
    "reward_to_risk",
    "sized_by",
    "based_on",
]


@pytest.mark.anyio
async def test_the_list_of_calls_shows_each_plan_in_summary(tmp_path: Path) -> None:
    # @AC8 @AC11
    cache = cache_for(provider(), tmp_path)
    planned = (await record(cache, plan=PLAN)).structured_content["plan"]
    await record(cache, direction="pass")

    read = await call(cache, {}, "get_decisions")

    pass_call, buy = read.structured_content["decisions"]
    assert buy["plan"] == {name: planned[name] for name in SUMMARY}
    assert pass_call["plan"] is None


@pytest.mark.anyio
async def test_one_call_by_id_comes_with_its_full_plan(tmp_path: Path) -> None:
    # @AC11
    cache = cache_for(provider(), tmp_path)
    await record(cache, direction="pass")
    recorded = (await record(cache, plan=PLAN)).structured_content
    await record(cache, direction="pass")

    read = await call(cache, {"id": 2}, "get_decisions")

    assert not read.is_error, read.content
    content = read.structured_content
    assert content["total"] == 1
    assert content["note"] is None
    [entry] = content["decisions"]
    assert {name: entry[name] for name in recorded} == recorded
    assert entry["status"] == "open"


@pytest.mark.anyio
async def test_an_unknown_id_returns_no_call_with_a_note(tmp_path: Path) -> None:
    # @AC11
    cache = cache_for(provider(), tmp_path)
    await record(cache, plan=PLAN)

    read = await call(cache, {"id": 7}, "get_decisions")

    assert not read.is_error, read.content
    content = read.structured_content
    assert (content["decisions"], content["total"]) == ([], 0)
    assert "7" in content["note"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "other",
    [{"status": "all"}, {"status": "open"}, {"ticker": "ASML.AS"}, {"limit": 10}],
)
async def test_an_id_with_another_filter_is_refused(
    tmp_path: Path, other: dict[str, Any]
) -> None:
    # @AC11
    cache = cache_for(provider(), tmp_path)
    await record(cache, plan=PLAN)

    read = await call(cache, {"id": 1} | other, "get_decisions")

    assert read.is_error
    text = read.content[0].text
    assert "id" in text
    assert next(iter(other)) in text


@pytest.mark.anyio
async def test_plan_trade_returns_the_free_cash(tmp_path: Path) -> None:
    # @AC6
    content = (await plan_trade(cache_for(provider(), tmp_path))).structured_content

    assert content["free_cash"] == 10000.0


@pytest.mark.anyio
async def test_plan_trade_sizes_on_the_total_value_after_a_realised_loss(
    tmp_path: Path,
) -> None:
    # @AC7
    cache = cache_for(provider(), tmp_path)
    seed(cache, decision_draft(plan=buy_plan()))
    journal = journal_of(cache)
    journal.append_fill(fill_line(1, session=date(2026, 10, 27)))
    journal.append_exit(exit_line(1, session=date(2026, 10, 28)))  # P&L -47.20

    content = (await plan_trade(cache)).structured_content

    assert content["portfolio_value"] == 9952.8
    assert content["position_share"] == round(1920.0 / 9952.8, 4)
    assert content["free_cash"] == 9952.8


def waiting_calls(count: int) -> list[DecisionDraft]:
    """Buys on ASML.AS whose plans (entry 600) wait from 2026-10-30 on."""
    return [decision_draft(plan=buy_plan()) for _ in range(count)]


def reserved_each() -> float:
    plan = buy_plan()
    return round(plan.position_value + plan.costs.buy.total, 2)


@pytest.mark.anyio
async def test_plan_trade_refuses_a_plan_beyond_the_free_cash(tmp_path: Path) -> None:
    # @AC6
    cache = cache_for(provider(), tmp_path)
    seed(cache, *waiting_calls(5))
    free = round(10000.0 - 5 * reserved_each(), 2)

    result = await plan_trade(cache)

    assert result.is_error
    text = result.content[0].text
    assert f"free cash of €{free:,.2f}" in text
    assert f"misses by €{round(1920.0 + 3.92 - free, 2):,.2f}" in text
    assert text.endswith("Nothing was recorded.")


@pytest.mark.anyio
async def test_record_decision_refuses_a_plan_beyond_the_free_cash(
    tmp_path: Path,
) -> None:
    # @AC6
    cache = cache_for(provider(), tmp_path)
    seed(cache, *waiting_calls(5))
    refused = await plan_trade(cache)

    result = await record(cache, plan=PLAN)

    assert result.is_error
    # The same words, after the SDK's "Error executing tool <name>: ".
    assert (
        result.content[0].text.split(": ", 1)[1]
        == refused.content[0].text.split(": ", 1)[1]
    )
    assert len(journal_of(cache).decisions()) == 5


@pytest.mark.anyio
async def test_plan_trade_settles_first_so_it_is_not_read_only(
    tmp_path: Path,
) -> None:
    cache = cache_for(provider(), tmp_path)
    async with Client(build_server(cache, journal_of(cache), RISK_CONFIG)) as client:
        tools = (await client.list_tools()).tools
    (tool,) = [t for t in tools if t.name == "plan_trade"]

    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is False
    assert tool.annotations.idempotent_hint is True
