"""
Integration tests for the risk config and `get_risk_config` (#149), driven the
way Claude Code drives the server: started from `.mcp.json` over stdio, with
`FARTT_ASSETS_DIR` set to a temporary folder holding a copy of the checked-in
`assets/risk.toml` (or a changed one), its clock pinned and Yahoo out of reach
(`snapshot.OFFLINE`).

Each test's docstring starts with its tags: `@issue-149` and the acceptance
criteria it verifies. The expected values are the design's tables (AC2) and
the checked-in file as TOML reads it (AC1), never the server's own models.
"""

import json
import os
import subprocess
import tomllib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from tests.integration.risk import CHECKED_IN, written
from tests.integration.snapshot import OFFLINE

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_TIMEOUT_S = 120
START_TIMEOUT_S = 60
TOOL = "get_risk_config"

# The design's tables: the limits of PRD section 6 and, per calendar code,
# commission, min, max, currency, tax_on_buy, fee_on_sell, fee_per_trade.
LIMITS = {
    "starting_capital": 10000,
    "max_risk_per_position": 0.02,
    "max_position_size": 0.20,
    "max_open_positions": 8,
    "min_reward_to_risk": 2.0,
    "kill_switch_drawdown": 0.16,
}
EXCHANGES: dict[str, tuple[float, float, float | None, str, float, float, float]] = {
    "XNYS": (0.0008, 1, 150, "USD", 0, 0.0000229, 0),
    "XAMS": (0.0008, 2, 150, "EUR", 0, 0, 0),
    "XPAR": (0.0008, 2, 150, "EUR", 0.004, 0, 0),
    "XBRU": (0.0008, 2, 150, "EUR", 0, 0, 0),
    "XLIS": (0.0008, 2, 150, "EUR", 0, 0, 0),
    "XMIL": (0.0008, 3, None, "EUR", 0.002, 0, 0),
    "XDUB": (0.0012, 5, None, "EUR", 0.01, 0, 1.25),
    "XOSL": (0.0008, 10, None, "NOK", 0, 0, 0),
}
EXCHANGE_FIELDS = (
    "commission",
    "commission_min",
    "commission_max",
    "commission_currency",
    "tax_on_buy",
    "fee_on_sell",
    "fee_per_trade",
)
# The ticker suffixes on each calendar (CLAUDE.md, `calendar_for`).
SUFFIXES = {
    "XNYS": [""],
    "XAMS": [".AS"],
    "XPAR": [".PA"],
    "XBRU": [".BR"],
    "XLIS": [".LS"],
    "XMIL": [".MI"],
    "XDUB": [".IR"],
    "XOSL": [".OL"],
}
# Every argument the other tools take: none is a limit or a cost.
NOT_RISK_ARGUMENTS = {
    "ticker",
    "count",
    "direction",
    "horizon",
    "reasoning",
    "signals",
    "status",
    "limit",  # how many decisions get_decisions returns, not a risk limit
    # plan_trade's (#143) and record_decision's trade plan: prices and an
    # order's validity, checked against the limits, never setting one.
    "entry",
    "stop",
    "target",
    "valid_for",
    "plan",
}
# One call of every other tool, on the snapshot, offline.
OTHER_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_candles", {"ticker": "ASML.AS", "count": 5}),
    ("get_quote", {"ticker": "ASML.AS"}),  # a ToolError offline: still a call
    ("get_indicators", {"ticker": "ASML.AS"}),
    ("get_trend", {"ticker": "ASML.AS"}),
    ("get_support_resistance", {"ticker": "ASML.AS"}),
    (
        "record_decision",
        {
            "ticker": "ASML.AS",
            "direction": "pass",
            "horizon": 20,
            "reasoning": "Risk config check over stdio.",
            "signals": [{"name": "trend", "stance": "against"}],
        },
    ),
    ("get_decisions", {}),
    ("score_decisions", {}),
    ("get_performance", {}),
    ("get_analysis_procedure", {"ticker": "ASML.AS"}),
    (
        "plan_trade",
        {"ticker": "ASML.AS", "entry": 1300.0, "stop": 1250.0, "target": 1450.0},
    ),
]

INITIALIZE = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "fartt-integration", "version": "0"},
            },
        }
    )
    + "\n"
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _command() -> list[str]:
    server = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["fartt"]
    return [server["command"], *server["args"]]


def _env(assets_dir: Path) -> dict[str, str]:
    return {**os.environ, "FARTT_ASSETS_DIR": str(assets_dir), **OFFLINE}


@asynccontextmanager
async def _server(assets_dir: Path) -> AsyncIterator[Client]:
    command, *args = _command()
    params = StdioServerParameters(
        command=command, args=args, cwd=str(ROOT), env=_env(assets_dir)
    )
    async with Client(params, read_timeout_seconds=RESPONSE_TIMEOUT_S) as client:
        yield client


async def _risk_config(client: Client) -> dict[str, Any]:
    result = await client.call_tool(TOOL, {})
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def _start(assets_dir: Path) -> subprocess.CompletedProcess[str]:
    """Start the server, send it `initialize`, and wait for it to stop."""
    return subprocess.run(
        _command(),
        cwd=ROOT,
        check=False,
        env=_env(assets_dir),
        input=INITIALIZE,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT_S,
    )


def _with(edit: Callable[[str], str], assets_dir: Path) -> Path:
    """Write the checked-in risk config, changed by `edit`, into `assets_dir`."""
    text = CHECKED_IN.read_text()
    changed = edit(text)
    assert changed != text, "the edit changed nothing"
    path = assets_dir / "risk.toml"
    path.write_text(changed)
    return path


def _replace(old: str, new: str) -> Callable[[str], str]:
    def edit(text: str) -> str:
        assert old in text, old
        return text.replace(old, new, 1)

    return edit


def _assert_refused(
    run: subprocess.CompletedProcess[str], path: Path, *problems: tuple[str, ...]
) -> None:
    """
    The server stopped without serving: a non-zero exit, nothing on stdout
    (not even an answer to `initialize`), and stderr naming the file and, per
    problem, a line holding each of its fragments.
    """
    assert run.returncode != 0, run.stderr
    assert run.stdout == ""
    assert str(path) in run.stderr, run.stderr
    lines = run.stderr.splitlines()
    for fragments in problems:
        assert any(all(f in line for f in fragments) for line in lines), (
            fragments,
            run.stderr,
        )


class TestReadingTheConfig:
    async def test_should_list_the_tool_read_only_without_arguments(
        self, empty_assets: Path
    ) -> None:
        """@issue-149 @AC1 @AC3 Should list get_risk_config, read-only, with no arguments."""
        async with _server(empty_assets) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        tool = tools[TOOL]
        assert tool.input_schema.get("properties", {}) == {}
        assert not tool.input_schema.get("required")
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True

    async def test_should_return_the_checked_in_values(
        self, empty_assets: Path
    ) -> None:
        """@issue-149 @AC1 Should return the file's path, its six limits and its costs per exchange."""
        file = tomllib.loads(CHECKED_IN.read_text())

        async with _server(empty_assets) as client:
            config = await _risk_config(client)

        assert Path(config["source"]).is_absolute()
        assert (
            Path(config["source"]).resolve() == (empty_assets / "risk.toml").resolve()
        )
        assert config["limits"] == file["limits"]
        costs = config["costs"]
        assert costs["tariff"] == file["costs"]["tariff"]
        assert costs["currency_conversion"] == file["costs"]["currency_conversion"]
        assert costs["slippage_per_leg"] == file["costs"]["slippage_per_leg"]
        assert sorted(costs["exchanges"]) == sorted(SUFFIXES)
        for code, table in file["costs"]["exchanges"].items():
            returned = costs["exchanges"][code]
            assert returned == {
                **{field: table.get(field) for field in EXCHANGE_FIELDS},
                "suffixes": SUFFIXES[code],
            }, code

    async def test_should_answer_offline_and_write_nothing(
        self, empty_assets: Path
    ) -> None:
        """@issue-149 @AC7 Should answer with Yahoo out of reach and leave the assets folder as it was."""
        before = (empty_assets / "risk.toml").read_bytes()

        async with _server(empty_assets) as client:
            config = await _risk_config(client)

        assert (
            config["limits"]["max_risk_per_position"] == LIMITS["max_risk_per_position"]
        )
        assert written(empty_assets) == []
        assert (empty_assets / "risk.toml").read_bytes() == before

    async def test_should_read_the_risk_config_of_its_assets_folder(
        self, empty_assets: Path
    ) -> None:
        """@issue-149 @AC4 Should report a limit the operator changed, and that file's path."""
        path = _with(
            _replace("max_risk_per_position = 0.02", "max_risk_per_position = 0.01"),
            empty_assets,
        )

        async with _server(empty_assets) as client:
            config = await _risk_config(client)

        assert config["limits"]["max_risk_per_position"] == 0.01
        assert Path(config["source"]).resolve() == path.resolve()


def test_should_track_the_file_with_the_designs_values() -> None:
    """@issue-149 @AC2 Should track assets/risk.toml with the design's limits and costs, cache and journal ignored."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "assets/risk.toml"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, tracked.stderr
    for ignored in (
        "assets/stocks/ASML.AS/candles.csv",
        "assets/journal/decisions.jsonl",
    ):
        check = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", ignored], cwd=ROOT, check=False
        )
        assert check.returncode == 0, f"{ignored} isn't ignored"
    not_ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "assets/risk.toml"],
        cwd=ROOT,
        check=False,
    )
    assert not_ignored.returncode == 1, "assets/risk.toml is ignored"

    file = tomllib.loads(CHECKED_IN.read_text())
    assert file["limits"] == LIMITS
    costs = file["costs"]
    assert costs["currency_conversion"] == 0.0025
    assert costs["slippage_per_leg"] == 0.001
    assert "Bronze" in costs["tariff"] and "13 July 2026" in costs["tariff"]
    assert sorted(costs["exchanges"]) == sorted(EXCHANGES)
    for code, values in EXCHANGES.items():
        table = costs["exchanges"][code]
        assert tuple(table.get(field) for field in EXCHANGE_FIELDS) == values, code


async def test_should_offer_no_tool_that_changes_the_risk_config(
    seeded_assets: Path,
) -> None:
    """@issue-149 @AC3 Should keep the same values after every other tool, also after a restart."""
    async with _server(seeded_assets) as client:
        tools = (await client.list_tools()).tools
        before = await _risk_config(client)
        for name, arguments in OTHER_CALLS:
            await client.call_tool(name, arguments)
        after = await _risk_config(client)
    async with _server(seeded_assets) as client:
        after_restart = await _risk_config(client)

    called = {name for name, _ in OTHER_CALLS}
    assert called == {tool.name for tool in tools} - {TOOL}, "a tool isn't called"
    for tool in tools:
        arguments = set(tool.input_schema.get("properties", {}))
        assert arguments <= NOT_RISK_ARGUMENTS, (tool.name, arguments)
    assert after == before
    assert after_restart == before


class TestRefusingAConfig:
    def test_should_not_start_without_a_risk_config(self, tmp_path: Path) -> None:
        """@issue-149 @AC5 Should stop, naming the missing file and saying to copy the repository's."""
        run = _start(tmp_path)

        _assert_refused(
            run, tmp_path / "risk.toml", ("doesn't exist", "assets/risk.toml")
        )
        assert list(tmp_path.iterdir()) == []

    def test_should_not_start_on_an_unreadable_risk_config(
        self, empty_assets: Path
    ) -> None:
        """@issue-149 @AC5 Should stop, naming the file it can't read."""
        path = empty_assets / "risk.toml"
        path.chmod(0)
        try:
            if os.access(path, os.R_OK):
                pytest.skip("this user reads a file without permissions (root?)")
            run = _start(empty_assets)
        finally:
            path.chmod(0o644)

        _assert_refused(run, path, ("can't be read",))

    def test_should_not_start_on_invalid_toml(self, empty_assets: Path) -> None:
        """@issue-149 @AC5 Should stop, naming the file and where its TOML is broken."""
        path = _with(
            _replace("max_open_positions = 8", "max_open_positions = = 8"),
            empty_assets,
        )

        run = _start(empty_assets)

        _assert_refused(run, path, ("TOML", "line 14"))

    @pytest.mark.parametrize(
        ("edit", "problem"),
        [
            pytest.param(
                _replace("kill_switch_drawdown = 0.16", "# kill_switch_drawdown"),
                ("limits.kill_switch_drawdown", "missing"),
                id="missing key",
            ),
            pytest.param(
                _replace("max_risk_per_position", "max_risk_per_postion"),
                ("limits.max_risk_per_postion", "0.02", "not a known key"),
                id="unknown key",
            ),
            pytest.param(
                _replace("max_open_positions = 8", 'max_open_positions = "8"'),
                ("limits.max_open_positions", "'8'", "integer"),
                id="string for a number",
            ),
            pytest.param(
                _replace("starting_capital = 10000", "starting_capital = true"),
                ("limits.starting_capital", "True", "number"),
                id="bool for a number",
            ),
            pytest.param(
                _replace(
                    "max_risk_per_position = 0.02", "max_risk_per_position = -0.02"
                ),
                ("limits.max_risk_per_position", "-0.02", "greater than 0"),
                id="negative risk",
            ),
            pytest.param(
                _replace("min_reward_to_risk = 2.0", "min_reward_to_risk = 0.5"),
                ("limits.min_reward_to_risk", "0.5", "1"),
                id="reward-to-risk below 1",
            ),
            pytest.param(
                _replace("currency_conversion = 0.0025", "currency_conversion = 0.25"),
                ("costs.currency_conversion", "0.25", "0.05"),
                id="currency conversion of 0.25",
            ),
            pytest.param(
                lambda text: text[: text.index("[costs.exchanges.XOSL]")],
                ("costs.exchanges.XOSL", "missing"),
                id="missing exchange",
            ),
            pytest.param(
                lambda text: (
                    text + "\n[costs.exchanges.XLON]\ncommission = 0.0008\n"
                    'commission_min = 2\ncommission_currency = "GBP"\n'
                    "tax_on_buy = 0.005\nfee_on_sell = 0\nfee_per_trade = 0\n"
                ),
                ("costs.exchanges.XLON", "not an exchange"),
                id="extra exchange",
            ),
            pytest.param(
                _replace("max_risk_per_position = 0.02", "max_risk_per_position = 0.5"),
                ("limits.max_risk_per_position", "0.5", "max_position_size"),
                id="risk above position size",
            ),
            pytest.param(
                _replace("max_open_positions = 8", "max_open_positions = 0"),
                ("limits.max_open_positions", "0", "1"),
                id="no open positions",
            ),
            pytest.param(
                _replace(
                    "commission_min = 1\ncommission_max = 150",
                    "commission_min = 1\ncommission_max = 1",
                ),
                ("costs.exchanges.XNYS.commission_max", "1", "commission_min"),
                id="maximum not above minimum",
            ),
            pytest.param(
                _replace('commission_currency = "NOK"', 'commission_currency = "nok"'),
                ("costs.exchanges.XOSL.commission_currency", "nok"),
                id="lower-case currency",
            ),
        ],
    )
    def test_should_not_start_on_a_value_it_cannot_accept(
        self,
        empty_assets: Path,
        edit: Callable[[str], str],
        problem: tuple[str, ...],
    ) -> None:
        """@issue-149 @AC6 Should stop, naming the file, the key, the value and what's wrong."""
        path = _with(edit, empty_assets)

        run = _start(empty_assets)

        _assert_refused(run, path, problem)

    def test_should_name_every_problem_at_once(self, empty_assets: Path) -> None:
        """@issue-149 @AC6 Should list every problem of a config, not only the first."""

        def edit(text: str) -> str:
            for old, new in [
                ("max_risk_per_position = 0.02", "max_risk_per_position = -0.02"),
                ("min_reward_to_risk = 2.0", "min_reward_to_risk = 0.5"),
                ("currency_conversion = 0.0025", "currency_conversion = 0.25"),
            ]:
                text = _replace(old, new)(text)
            return text

        path = _with(edit, empty_assets)

        run = _start(empty_assets)

        _assert_refused(
            run,
            path,
            ("limits.max_risk_per_position", "-0.02"),
            ("limits.min_reward_to_risk", "0.5"),
            ("costs.currency_conversion", "0.25"),
        )

    async def test_should_start_on_a_reward_to_risk_of_exactly_1(
        self, empty_assets: Path
    ) -> None:
        """@issue-149 @AC6 Should accept the boundary: a reward-to-risk of 1 isn't below 1."""
        _with(
            _replace("min_reward_to_risk = 2.0", "min_reward_to_risk = 1.0"),
            empty_assets,
        )

        async with _server(empty_assets) as client:
            config = await _risk_config(client)

        assert config["limits"]["min_reward_to_risk"] == 1.0
