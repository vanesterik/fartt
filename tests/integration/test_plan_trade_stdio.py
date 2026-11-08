"""
Integration tests for `plan_trade` and the plan on `record_decision` (#143),
driven the way Claude Code drives the server: started from `.mcp.json` over
stdio, with `FARTT_ASSETS_DIR` set to a temporary folder holding the snapshot
and a copy of the checked-in risk config, its clock pinned to the snapshot's
moment and Yahoo out of reach (`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-143` and the acceptance
criteria it verifies. Expectations come from the issue's criteria and the
design's cost formulas, applied here to the limits and tariff read from the
risk config itself. Prices are set relative to the last close and ATR the
server reports (`get_indicators`), never fixed, so a rebuilt snapshot keeps
the tests meaningful.
"""

import json
import math
import os
import re
import sys
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from tests.integration.risk import CHECKED_IN, written
from tests.integration.snapshot import OFFLINE

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 180
RISK = tomllib.loads(CHECKED_IN.read_text())
LIMITS = RISK["limits"]
CAPITAL = LIMITS["starting_capital"]
MAX_RISK = LIMITS["max_risk_per_position"] * CAPITAL
MAX_POSITION = LIMITS["max_position_size"] * CAPITAL
MIN_RR = LIMITS["min_reward_to_risk"]
COST_FIELDS = {"commission", "slippage", "conversion", "tax", "fee", "total"}
PLAN_FIELDS = {
    "ticker",
    "currency",
    "exchange",
    "based_on",
    "last_close",
    "atr",
    "entry",
    "stop",
    "target",
    "valid_for",
    "rate",
    "shares",
    "position_value",
    "portfolio_value",
    "position_share",
    "risk",
    "risk_share",
    "reward",
    "reward_to_risk",
    "reward_to_risk_before_costs",
    "sized_by",
    "costs",
    "limits",
    "tariff",
}
SIGNALS = [{"name": "trend", "stance": "for"}]
CALL: dict[str, Any] = {
    "horizon": 20,
    "reasoning": "Integration check of a call with its trade plan.",
    "signals": SIGNALS,
}

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _server(assets_dir: Path, *, live: bool = False) -> AsyncIterator[Client]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    env = {
        **os.environ,
        "FARTT_ASSETS_DIR": str(assets_dir),
        **({} if live else OFFLINE),
    }
    params = StdioServerParameters(
        command=server["command"], args=server["args"], cwd=str(ROOT), env=env
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    content = result.structured_content
    assert content is not None
    return content


async def _error(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error, result.structured_content
    return " ".join(getattr(part, "text", "") for part in result.content)


def _journal(assets_dir: Path) -> Path:
    return assets_dir / "journal" / "decisions.jsonl"


def _all_finite(value: Any) -> bool:
    """Whether every number anywhere in `value` is finite (no NaN or infinity)."""
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_all_finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


async def _market(client: Client, ticker: str) -> tuple[float, float]:
    """The last close and ATR the server reports for `ticker`."""
    indicators = await _call(client, "get_indicators", ticker=ticker)
    return indicators["close"], indicators["atr"]["value"]


async def _rate(client: Client) -> float:
    """EURUSD=X's newest cached close, as the server returns it."""
    candles = await _call(client, "get_candles", ticker="EURUSD=X", count=1)
    return candles["rows"][-1][4]


def _leg_costs(
    value: float, rate: float, exchange: str, *, buy: bool, foreign: bool
) -> float:
    """
    One leg's costs in EUR, by the design's formulas, from the risk config:
    `value` is in the stock's currency, `rate` units of it per euro.
    """
    tariff = RISK["costs"]["exchanges"][exchange]
    commission = value * tariff["commission"]
    commission = max(commission, tariff["commission_min"])
    commission = min(commission, tariff.get("commission_max", math.inf))
    in_eur = value / rate
    total = commission / rate if tariff["commission_currency"] != "EUR" else commission
    total += in_eur * RISK["costs"]["slippage_per_leg"]
    if foreign:
        total += in_eur * RISK["costs"]["currency_conversion"]
    total += in_eur * (tariff["tax_on_buy"] if buy else tariff["fee_on_sell"])
    total += tariff["fee_per_trade"]
    return total


def _risk(shares: int, entry: float, stop: float, rate: float, exchange: str) -> float:
    foreign = rate != 1
    return (
        shares * (entry - stop) / rate
        + _leg_costs(shares * entry, rate, exchange, buy=True, foreign=foreign)
        + _leg_costs(shares * stop, rate, exchange, buy=False, foreign=foreign)
    )


def _within_limits(
    shares: int, entry: float, stop: float, rate: float, exchange: str
) -> bool:
    return (
        _risk(shares, entry, stop, rate, exchange) <= MAX_RISK
        and shares * entry / rate <= MAX_POSITION
    )


class TestAPlanWithinTheLimits:
    @pytest.mark.parametrize(
        ("ticker", "exchange"), [("AAPL", "XNYS"), ("ASML.AS", "XAMS")]
    )
    async def test_should_return_the_sized_plan_and_record_nothing(
        self, seeded_assets: Path, ticker: str, exchange: str
    ) -> None:
        """
        @issue-143 @AC1 @AC2 Should return the full plan after costs, within
        the risk limit, and record nothing.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, ticker)
            rate = await _rate(client) if exchange == "XNYS" else 1.0
            entry = round(close * 0.98, 2)
            stop = round(entry * 0.97, 2)
            target = round(entry * 1.12, 2)
            before = await _call(client, "get_decisions")
            plan = await _call(
                client,
                "plan_trade",
                ticker=ticker,
                entry=entry,
                stop=stop,
                target=target,
            )
            after = await _call(client, "get_decisions")

        # AC1: the fields, whole shares, itemised legs, limits and tariff.
        assert set(plan) >= PLAN_FIELDS
        assert (plan["ticker"], plan["exchange"]) == (ticker, exchange)
        assert (plan["entry"], plan["stop"], plan["target"]) == (entry, stop, target)
        assert isinstance(plan["shares"], int) and plan["shares"] >= 1
        assert set(plan["costs"]) == {"buy", "sell_at_stop", "sell_at_target"}
        for leg in plan["costs"].values():
            assert set(leg) >= COST_FIELDS
            parts = sum(leg[k] for k in COST_FIELDS - {"total"})
            assert leg["total"] == pytest.approx(parts, abs=0.011)
        assert plan["sized_by"] in {"max_risk_per_position", "max_position_size"}
        assert (
            plan["limits"]["max_risk_per_position"] == LIMITS["max_risk_per_position"]
        )
        assert plan["limits"]["min_reward_to_risk"] == MIN_RR
        assert plan["tariff"]["slippage_per_leg"] == RISK["costs"]["slippage_per_leg"]
        assert plan["reward_to_risk_before_costs"] == pytest.approx(
            (target - entry) / (entry - stop), abs=1e-4
        )
        assert plan["reward_to_risk"] < plan["reward_to_risk_before_costs"]
        assert plan["reward_to_risk"] == pytest.approx(
            plan["reward"] / plan["risk"], abs=1e-3
        )
        assert plan["reward_to_risk"] >= MIN_RR
        assert before == after
        assert written(seeded_assets) == ["stocks"]

        # AC2: risk = distance in EUR plus both legs' costs, within the limit.
        shares = plan["shares"]
        assert plan["risk"] == pytest.approx(
            _risk(shares, entry, stop, rate, exchange), abs=0.02
        )
        legs = plan["costs"]["buy"]["total"] + plan["costs"]["sell_at_stop"]["total"]
        assert plan["risk"] == pytest.approx(
            shares * (entry - stop) / rate + legs, abs=0.02
        )
        assert plan["risk"] <= MAX_RISK
        assert plan["risk_share"] <= LIMITS["max_risk_per_position"]
        # The size is the largest whole number of shares within both limits.
        assert _within_limits(shares, entry, stop, rate, exchange)
        assert not _within_limits(shares + 1, entry, stop, rate, exchange)

    async def test_should_size_a_wide_stop_by_the_risk_limit(
        self, seeded_assets: Path
    ) -> None:
        """@issue-143 @AC2 Should size a wide stop by max_risk_per_position."""
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "ARM")
            rate = await _rate(client)
            entry = round(close * 0.99, 2)
            stop = round(entry * 0.84, 2)
            target = round(entry * 1.40, 2)
            plan = await _call(
                client,
                "plan_trade",
                ticker="ARM",
                entry=entry,
                stop=stop,
                target=target,
            )

        assert plan["sized_by"] == "max_risk_per_position"
        assert plan["risk"] <= MAX_RISK
        assert plan["risk"] == pytest.approx(
            _risk(plan["shares"], entry, stop, rate, "XNYS"), abs=0.02
        )
        assert _risk(plan["shares"] + 1, entry, stop, rate, "XNYS") > MAX_RISK


class TestSizeCappedByThePositionLimit:
    async def test_should_cap_a_tight_stop_at_the_position_limit(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC3 Should cap a tight stop at max_position_size and say so.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            rate = await _rate(client)
            entry = round(close * 0.98, 2)
            stop = round(entry * 0.97, 2)
            target = round(entry * 1.12, 2)
            plan = await _call(
                client,
                "plan_trade",
                ticker="AAPL",
                entry=entry,
                stop=stop,
                target=target,
            )

        shares = plan["shares"]
        # The risk limit alone would allow more than the position limit.
        by_risk = shares
        while _risk(by_risk + 1, entry, stop, rate, "XNYS") <= MAX_RISK:
            by_risk += 1
        assert by_risk * entry / rate > MAX_POSITION
        assert plan["sized_by"] == "max_position_size"
        assert plan["position_value"] == pytest.approx(shares * entry / rate, abs=0.01)
        assert plan["position_value"] <= MAX_POSITION
        assert (shares + 1) * entry / rate > MAX_POSITION
        assert plan["position_share"] <= LIMITS["max_position_size"]


class TestRewardToRiskTooLow:
    @pytest.mark.parametrize("before_costs", [1.5, 2.0])
    async def test_should_refuse_naming_the_limit_value_and_shortfall(
        self, seeded_assets: Path, before_costs: float
    ) -> None:
        """
        @issue-143 @AC4 Should refuse a reward-to-risk below the minimum after
        costs, naming the limit, the value and the shortfall.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            stop = round(entry - 10, 2)
            target = round(entry + before_costs * 10, 2)
            message = await _error(
                client,
                "plan_trade",
                ticker="AAPL",
                entry=entry,
                stop=stop,
                target=target,
            )

        assert "min_reward_to_risk" in message
        found = re.search(r"reward-to-risk after costs is (\d+\.\d+)", message)
        assert found, message
        value = float(found.group(1))
        assert value < MIN_RR
        assert str(MIN_RR) in message
        shortfall = re.search(r"short by (\d+\.\d+)", message)
        assert shortfall, message
        assert float(shortfall.group(1)) == pytest.approx(MIN_RR - value, abs=1e-4)
        assert written(seeded_assets) == ["stocks"]


class TestTooSmallToTrade:
    async def test_should_refuse_when_one_share_breaks_the_risk_limit(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC5 Should refuse when one share already breaks
        max_risk_per_position, saying what one share would risk.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "ASML.AS")
            entry = round(close * 0.98, 2)
            stop = round(entry - MAX_RISK - 5, 2)
            target = round(entry + 3 * (entry - stop), 2)
            message = await _error(
                client,
                "plan_trade",
                ticker="ASML.AS",
                entry=entry,
                stop=stop,
                target=target,
            )

        assert "max_risk_per_position" in message
        one_share = _risk(1, entry, stop, 1.0, "XAMS")
        found = re.search(r"risk €([\d,]+\.\d\d)", message)
        assert found, message
        assert float(found.group(1).replace(",", "")) == pytest.approx(
            one_share, abs=0.02
        )
        assert "0 shares" in message
        assert written(seeded_assets) == ["stocks"]

    async def test_should_refuse_when_one_share_breaks_the_position_limit(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC5 Should refuse when one share already breaks
        max_position_size, saying what one share would cost.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "ASML.AS")
        entry = round(close * 0.98, 2)
        # A portfolio whose position limit is below one share's price.
        capital = math.floor(entry * 4)
        risk_file = seeded_assets / "risk.toml"
        risk_file.write_text(
            re.sub(
                r"(?m)^starting_capital = \d+",
                f"starting_capital = {capital}",
                risk_file.read_text(),
            )
        )
        async with _server(seeded_assets) as client:
            message = await _error(
                client,
                "plan_trade",
                ticker="ASML.AS",
                entry=entry,
                stop=round(entry * 0.99, 2),
                target=round(entry * 1.1, 2),
            )

        assert "max_position_size" in message
        assert f"€{entry:,.2f}" in message
        assert "0 shares" in message
        assert written(seeded_assets) == ["stocks"]


class TestPricesThatDontMakeALongPlan:
    @pytest.mark.parametrize(
        ("stop_off", "target_off", "named"),
        [
            (0.0, 30.0, "stop"),
            (5.0, 30.0, "stop"),
            (-10.0, 0.0, "target"),
            (-10.0, -20.0, "target"),
        ],
        ids=["stop-at-entry", "stop-above", "target-at-entry", "target-below"],
    )
    async def test_should_refuse_a_stop_or_target_on_the_wrong_side(
        self, seeded_assets: Path, stop_off: float, target_off: float, named: str
    ) -> None:
        """@issue-143 @AC6 Should refuse a stop or target on the wrong side."""
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            message = await _error(
                client,
                "plan_trade",
                ticker="AAPL",
                entry=entry,
                stop=entry + stop_off,
                target=entry + target_off,
            )

        assert f"The {named}" in message
        assert "entry" in message
        assert written(seeded_assets) == ["stocks"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("stop", 0.0), ("stop", -5.0), ("entry", 0.0), ("target", -1.0)],
    )
    async def test_should_refuse_a_non_positive_price(
        self, seeded_assets: Path, field: str, value: float
    ) -> None:
        """@issue-143 @AC6 Should refuse a non-positive price, naming it."""
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            prices = {"entry": entry, "stop": entry - 10, "target": entry + 40}
            prices[field] = value
            message = await _error(client, "plan_trade", ticker="AAPL", **prices)

        assert field in message
        assert "above 0" in message
        assert written(seeded_assets) == ["stocks"]

    @pytest.mark.parametrize("field", ["entry", "stop", "target"])
    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
    async def test_should_refuse_a_price_that_is_not_a_finite_number(
        self, seeded_assets: Path, field: str, value: str
    ) -> None:
        """
        @issue-143 @AC6 @AC8 Should refuse a price that isn't a finite number,
        naming it, in plan_trade and in record_decision's plan, recording
        nothing.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            prices: dict[str, Any] = {
                "entry": entry,
                "stop": round(entry * 0.97, 2),
                "target": round(entry * 1.12, 2),
            }
            prices[field] = value
            planned = await _error(client, "plan_trade", ticker="AAPL", **prices)
            recorded = await _error(
                client,
                "record_decision",
                ticker="AAPL",
                direction="buy",
                **CALL,
                plan={**prices, "valid_for": 5},
            )

        for message in (planned, recorded):
            assert field in message
            assert "finite" in message
        assert written(seeded_assets) == ["stocks"]

    @pytest.mark.parametrize("ticker", ["AAPL", "ASML.AS"])
    async def test_should_never_store_a_plan_whose_amounts_overflow(
        self, seeded_assets: Path, ticker: str
    ) -> None:
        """
        @issue-143 @AC6 @AC8 Should, for a finite target at the float maximum,
        either refuse the plan naming the target or return and store only
        finite amounts, so get_decisions still reads the journal back.
        """
        target = sys.float_info.max
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, ticker)
            entry = round(close, 2)
            prices = {"entry": entry, "stop": round(entry * 0.97, 2), "target": target}
            planned = await client.call_tool("plan_trade", {"ticker": ticker, **prices})
            recorded = await client.call_tool(
                "record_decision",
                {
                    "ticker": ticker,
                    "direction": "buy",
                    **CALL,
                    "plan": {**prices, "valid_for": 5},
                },
            )
            decisions = await _call(client, "get_decisions")

        for result in (planned, recorded):
            if result.is_error:
                message = " ".join(getattr(c, "text", "") for c in result.content)
                assert "target" in message
                assert "Nothing was recorded" in message
            else:
                assert _all_finite(result.structured_content)
        assert decisions["total"] == (0 if recorded.is_error else 1)
        assert _all_finite(decisions)
        if recorded.is_error:
            assert not _journal(seeded_assets).exists()

    async def test_should_refuse_an_entry_more_than_one_atr_above_the_close(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC6 Should refuse an entry above the close plus 1 ATR and
        accept one just within it.
        """
        async with _server(seeded_assets) as client:
            close, atr = await _market(client, "AAPL")
            bound = close + atr
            too_high = round(bound + 0.05, 2)
            message = await _error(
                client,
                "plan_trade",
                ticker="AAPL",
                entry=too_high,
                stop=round(too_high - 15, 2),
                target=round(too_high + 60, 2),
            )
            within = math.floor((bound - 0.01) * 100) / 100
            plan = await _call(
                client,
                "plan_trade",
                ticker="AAPL",
                entry=within,
                stop=round(within - 15, 2),
                target=round(within + 60, 2),
            )

        assert "ATR" in message
        assert "entry" in message
        assert str(close) in message
        assert plan["entry"] == within

    @pytest.mark.parametrize(
        ("valid_for", "accepted"),
        [
            (0, False),
            (11, False),
            (1, True),
            (10, True),
        ],
    )
    async def test_should_hold_valid_for_to_1_through_10(
        self, seeded_assets: Path, valid_for: int, accepted: bool
    ) -> None:
        """@issue-143 @AC6 Should accept valid_for 1 to 10 and refuse others."""
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            arguments = {
                "ticker": "AAPL",
                "entry": entry,
                "stop": round(entry * 0.97, 2),
                "target": round(entry * 1.12, 2),
                "valid_for": valid_for,
            }
            if accepted:
                plan = await _call(client, "plan_trade", **arguments)
                assert plan["valid_for"] == valid_for
            else:
                message = await _error(client, "plan_trade", **arguments)
                assert "valid_for" in message
                assert str(valid_for) in message


class TestRefusedAsRecordDecisionRefuses:
    async def test_should_refuse_a_ticker_without_its_newest_close(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC7 Should refuse a stale ticker as record_decision does.
        """
        candles = seeded_assets / "stocks" / "ASML.AS" / "candles.csv"
        rows = candles.read_text().splitlines(keepends=True)
        newest = rows[-1].split(",")[0]
        candles.write_text("".join(rows[:-1]))
        async with _server(seeded_assets) as client:
            planned = await _error(
                client,
                "plan_trade",
                ticker="ASML.AS",
                entry=1000.0,
                stop=900.0,
                target=1400.0,
            )
            recorded = await _error(
                client, "record_decision", ticker="ASML.AS", direction="buy", **CALL
            )

        for message in (planned, recorded):
            assert "ASML.AS" in message
            assert f"newest close ({newest})" in message
            assert "try again later" in message.lower()
        assert written(seeded_assets) == ["stocks"]

    @pytest.mark.parametrize("ticker", ["EURUSD=X", "7203.T", "../etc"])
    async def test_should_refuse_what_record_decision_refuses(
        self, seeded_assets: Path, ticker: str
    ) -> None:
        """
        @issue-143 @AC7 Should refuse an exchange rate, an exchange outside the
        universe and a malformed ticker as record_decision does.
        """
        async with _server(seeded_assets) as client:
            planned = await _error(
                client, "plan_trade", ticker=ticker, entry=1.0, stop=0.9, target=1.5
            )
            recorded = await _error(
                client, "record_decision", ticker=ticker, direction="buy", **CALL
            )

        def reason(message: str) -> str:
            # The first sentence after the tool's prefix: the refusal itself.
            text = message.split(": ", 1)[1]
            return re.split(r"\. (Plan|Record) ", text)[0]

        assert reason(planned) == reason(recorded)
        assert written(seeded_assets) == ["stocks"]

    @pytest.mark.network
    async def test_should_refuse_a_ticker_yahoo_does_not_list(
        self, empty_assets: Path
    ) -> None:
        """
        @issue-143 @AC7 Should refuse an unknown ticker as record_decision does.
        """
        async with _server(empty_assets, live=True) as client:
            planned = await _error(
                client,
                "plan_trade",
                ticker="ZZZQXQ",
                entry=100.0,
                stop=90.0,
                target=130.0,
            )
            recorded = await _error(
                client, "record_decision", ticker="ZZZQXQ", direction="buy", **CALL
            )

        assert planned == recorded.replace("record_decision", "plan_trade")
        assert "ZZZQXQ" in planned
        assert written(empty_assets) == []


class TestRecordDecisionWithAPlan:
    async def test_should_store_the_plan_with_the_buy(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC8 Should store a passing plan's full calculation with the
        buy, returned by record_decision and get_decisions.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            plan_input = {
                "entry": entry,
                "stop": round(entry * 0.97, 2),
                "target": round(entry * 1.12, 2),
                "valid_for": 7,
            }
            planned = await _call(client, "plan_trade", ticker="AAPL", **plan_input)
            entry_ = await _call(
                client,
                "record_decision",
                ticker="AAPL",
                direction="buy",
                plan=plan_input,
                **CALL,
            )
            read_back = await _call(client, "get_decisions")

        assert entry_["plan"] == planned
        assert set(entry_["plan"]) >= PLAN_FIELDS
        assert read_back["total"] == 1
        assert read_back["decisions"][0]["plan"] == planned
        stored = json.loads(_journal(seeded_assets).read_text().splitlines()[0])
        assert stored["plan"] == planned

    @pytest.mark.parametrize(
        ("stop_factor", "target_factor"),
        [(0.97, 1.03), (1.01, 1.12), (0.5, 3.0)],
        ids=["reward-to-risk", "stop-above-entry", "one-share-too-risky"],
    )
    async def test_should_record_nothing_for_a_plan_that_breaks_a_limit(
        self, seeded_assets: Path, stop_factor: float, target_factor: float
    ) -> None:
        """
        @issue-143 @AC8 Should record nothing for a failing plan and give
        plan_trade's error.
        """
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "ASML.AS")
            entry = round(close * 0.98, 2)
            plan_input = {
                "entry": entry,
                "stop": round(entry * stop_factor, 2),
                "target": round(entry * target_factor, 2),
            }
            planned = await _error(client, "plan_trade", ticker="ASML.AS", **plan_input)
            recorded = await _error(
                client,
                "record_decision",
                ticker="ASML.AS",
                direction="buy",
                plan=plan_input,
                **CALL,
            )
            read_back = await _call(client, "get_decisions")

        assert recorded.split(": ", 1)[1] == planned.split(": ", 1)[1]
        assert read_back["total"] == 0
        assert not _journal(seeded_assets).exists()


class TestCallsWithoutAPlan:
    async def test_should_refuse_a_pass_with_a_plan(self, seeded_assets: Path) -> None:
        """@issue-143 @AC9 Should refuse a pass with a plan and record nothing."""
        async with _server(seeded_assets) as client:
            close, _atr = await _market(client, "ASML.AS")
            entry = round(close * 0.98, 2)
            plan_input = {
                "entry": entry,
                "stop": round(entry * 0.97, 2),
                "target": round(entry * 1.12, 2),
            }
            await _call(client, "plan_trade", ticker="ASML.AS", **plan_input)
            message = await _error(
                client,
                "record_decision",
                ticker="ASML.AS",
                direction="pass",
                plan=plan_input,
                **CALL,
            )
            read_back = await _call(client, "get_decisions")

        assert "pass" in message.lower()
        assert "plan" in message
        assert read_back["total"] == 0
        assert not _journal(seeded_assets).exists()

    @pytest.mark.parametrize("direction", ["buy", "pass"])
    async def test_should_record_a_call_without_a_plan_as_before(
        self, seeded_assets: Path, direction: str
    ) -> None:
        """@issue-143 @AC9 Should record a call without a plan with plan null."""
        async with _server(seeded_assets) as client:
            entry = await _call(
                client,
                "record_decision",
                ticker="ASML.AS",
                direction=direction,
                **CALL,
            )
            read_back = await _call(client, "get_decisions")

        assert entry["id"] == 1
        assert entry["direction"] == direction
        assert entry["plan"] is None
        assert read_back["decisions"][0]["plan"] is None

    async def test_should_read_a_call_recorded_before_plans_with_plan_null(
        self, seeded_assets: Path
    ) -> None:
        """@issue-143 @AC9 Should read an older journal line as plan null."""
        async with _server(seeded_assets) as client:
            await _call(
                client, "record_decision", ticker="ASML.AS", direction="buy", **CALL
            )
        journal = _journal(seeded_assets)
        line = json.loads(journal.read_text())
        line.pop("plan", None)
        journal.write_text(json.dumps(line) + "\n")

        async with _server(seeded_assets) as client:
            read_back = await _call(client, "get_decisions")

        assert read_back["total"] == 1
        assert read_back["decisions"][0]["plan"] is None


class TestAStockNotInEur:
    async def test_should_convert_at_the_rate_it_gives(
        self, seeded_assets: Path
    ) -> None:
        """
        @issue-143 @AC10 Should give the EUR/USD rate used and convert the
        position, risk and costs at it.
        """
        async with _server(seeded_assets) as client:
            rate = await _rate(client)
            close, _atr = await _market(client, "AAPL")
            entry = round(close * 0.98, 2)
            stop = round(entry * 0.97, 2)
            plan = await _call(
                client,
                "plan_trade",
                ticker="AAPL",
                entry=entry,
                stop=stop,
                target=round(entry * 1.12, 2),
            )

        assert plan["currency"] == "USD"
        assert plan["rate"] == pytest.approx(rate, abs=1e-6)
        shares = plan["shares"]
        assert plan["position_value"] == pytest.approx(shares * entry / rate, abs=0.01)
        buy = plan["costs"]["buy"]
        in_eur = shares * entry / rate
        assert buy["slippage"] == pytest.approx(
            in_eur * RISK["costs"]["slippage_per_leg"], abs=0.01
        )
        assert buy["conversion"] == pytest.approx(
            in_eur * RISK["costs"]["currency_conversion"], abs=0.01
        )
        assert plan["risk"] == pytest.approx(
            _risk(shares, entry, stop, rate, "XNYS"), abs=0.02
        )
