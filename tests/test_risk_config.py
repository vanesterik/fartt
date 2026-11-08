import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from fartt.data import SUFFIX_CALENDARS
from fartt.risk_config import (
    RISK_FILE,
    RiskConfigError,
    load_risk_config,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN = ROOT / "assets" / RISK_FILE

# The design's table: commission, min, max, currency, tax_on_buy,
# fee_on_sell, fee_per_trade.
EXCHANGES = {
    "XNYS": (0.0008, 1, 150, "USD", 0, 0.0000229, 0),
    "XAMS": (0.0008, 2, 150, "EUR", 0, 0, 0),
    "XPAR": (0.0008, 2, 150, "EUR", 0.004, 0, 0),
    "XBRU": (0.0008, 2, 150, "EUR", 0, 0, 0),
    "XLIS": (0.0008, 2, 150, "EUR", 0, 0, 0),
    "XMIL": (0.0008, 3, None, "EUR", 0.002, 0, 0),
    "XDUB": (0.0012, 5, None, "EUR", 0.01, 0, 1.25),
    "XOSL": (0.0008, 10, None, "NOK", 0, 0, 0),
}


def test_the_checked_in_file_holds_the_designs_values() -> None:
    config = load_risk_config(CHECKED_IN)

    assert config.source == CHECKED_IN.resolve()
    limits = config.limits
    assert limits.starting_capital == 10000
    assert limits.max_risk_per_position == 0.02
    assert limits.max_position_size == 0.20
    assert limits.max_open_positions == 8
    assert limits.min_reward_to_risk == 2.0
    assert limits.kill_switch_drawdown == 0.16
    costs = config.costs
    assert "Saxo NL, Bronze" in costs.tariff
    assert "13 July 2026" in costs.tariff
    assert costs.currency_conversion == 0.0025
    assert costs.slippage_per_leg == 0.001
    assert set(costs.exchanges) == set(EXCHANGES)
    for code, row in EXCHANGES.items():
        exchange = costs.exchanges[code]
        assert (
            exchange.commission,
            exchange.commission_min,
            exchange.commission_max,
            exchange.commission_currency,
            exchange.tax_on_buy,
            exchange.fee_on_sell,
            exchange.fee_per_trade,
        ) == row, code


def test_the_checked_in_file_is_tracked_and_the_cache_is_ignored() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "assets/risk.toml"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "assets/stocks/AAPL", "assets/journal/x"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert tracked.returncode == 0, tracked.stderr
    assert ignored.stdout.split() == ["assets/stocks/AAPL", "assets/journal/x"]


def raw() -> dict[str, Any]:
    return tomllib.loads(CHECKED_IN.read_text())


def write(path: Path, config: dict[str, Any]) -> Path:
    """`config` as TOML, enough of it for these tests."""
    lines: list[str] = []

    def value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return f'"{v}"'
        return repr(v)

    def table(name: str, entries: dict[str, Any]) -> None:
        scalars = {k: v for k, v in entries.items() if not isinstance(v, dict)}
        lines.append(f"[{name}]")
        lines.extend(f"{k} = {value(v)}" for k, v in scalars.items())
        for k, v in entries.items():
            if isinstance(v, dict):
                table(f"{name}.{k}", v)  # type: ignore[arg-type]

    for name, entries in config.items():
        table(name, entries)
    file = path / RISK_FILE
    file.write_text("\n".join(lines) + "\n")
    return file


def problems(file: Path) -> str:
    with pytest.raises(RiskConfigError) as raised:
        load_risk_config(file)
    message = str(raised.value)
    assert str(file) in message
    assert raised.value.path == file
    return message


def test_the_writer_round_trips_the_checked_in_file(tmp_path: Path) -> None:
    file = write(tmp_path, raw())

    assert load_risk_config(file).costs == load_risk_config(CHECKED_IN).costs


def test_a_missing_file_says_to_copy_the_repositorys(tmp_path: Path) -> None:
    message = problems(tmp_path / RISK_FILE)

    assert "doesn't exist" in message
    assert "copy the repository's assets/risk.toml" in message


def test_a_directory_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / RISK_FILE).mkdir()

    assert "can't be read" in problems(tmp_path / RISK_FILE)


def test_a_file_that_isnt_utf8_is_unreadable(tmp_path: Path) -> None:
    file = tmp_path / RISK_FILE
    file.write_bytes(b'[limits]\nname = "\xff"\n')

    assert "can't be read" in problems(file)


def test_invalid_toml_names_line_and_column(tmp_path: Path) -> None:
    file = tmp_path / RISK_FILE
    file.write_text("[limits]\nstarting_capital = = 1\n")

    message = problems(file)

    assert "isn't valid TOML" in message
    assert re.search(r"line 2, column \d+", message)


def changed(path: list[str], new: Any) -> dict[str, Any]:
    config = raw()
    node = config
    for key in path[:-1]:
        node = node[key]
    if new is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = new
    return config


DELETE = object()


@pytest.mark.parametrize(
    ("path", "new", "expected"),
    [
        (
            ["limits", "max_open_positions"],
            DELETE,
            "limits.max_open_positions: missing",
        ),
        (
            ["limits", "max_risk_per_postion"],
            0.02,
            "limits.max_risk_per_postion = 0.02: not a known key",
        ),
        (["limits", "starting_capital"], "10000", "limits.starting_capital = '10000':"),
        (["limits", "max_open_positions"], True, "limits.max_open_positions = True:"),
        (["limits", "max_open_positions"], 8.0, "limits.max_open_positions = 8.0:"),
        (
            ["limits", "max_risk_per_position"],
            -0.02,
            "limits.max_risk_per_position = -0.02: Input should be greater than 0",
        ),
        (
            ["limits", "min_reward_to_risk"],
            0.5,
            "limits.min_reward_to_risk = 0.5: Input should be greater than or equal to 1",
        ),
        (
            ["costs", "currency_conversion"],
            0.25,
            "costs.currency_conversion = 0.25: Input should be less than 0.05",
        ),
        (
            ["limits", "max_risk_per_position"],
            0.3,
            "limits.max_risk_per_position = 0.3: above limits.max_position_size (0.2)",
        ),
        (
            ["costs", "exchanges", "XAMS", "commission_max"],
            2,
            "costs.exchanges.XAMS.commission_max = 2: not above commission_min (2)",
        ),
        (
            ["costs", "exchanges", "XOSL", "commission_currency"],
            "nok",
            "costs.exchanges.XOSL.commission_currency = 'nok':",
        ),
        (
            ["costs", "exchanges", "XPAR", "tax_on_buy"],
            -0.004,
            "costs.exchanges.XPAR.tax_on_buy = -0.004:",
        ),
        (
            ["costs", "exchanges", "XMIL"],
            DELETE,
            "costs.exchanges.XMIL: missing; every exchange in the universe needs one",
        ),
        (
            ["costs", "exchanges", "XLON"],
            raw()["costs"]["exchanges"]["XAMS"],
            "costs.exchanges.XLON: not an exchange in the universe",
        ),
    ],
)
def test_a_value_it_cant_accept_names_key_value_and_problem(
    tmp_path: Path, path: list[str], new: Any, expected: str
) -> None:
    message = problems(write(tmp_path, changed(path, new)))

    assert expected in message


def test_exchanges_that_arent_a_table(tmp_path: Path) -> None:
    file = tmp_path / RISK_FILE
    text = CHECKED_IN.read_text()
    text = text[: text.index("[costs.exchanges.")] + 'exchanges = "all"\n'
    file.write_text(text)

    message = problems(file)

    assert "costs.exchanges = 'all':" in message
    assert message.count("costs.exchanges") == 1


def test_every_problem_is_listed(tmp_path: Path) -> None:
    config = changed(["limits", "max_risk_per_position"], -1)
    config["limits"]["max_position_size"] = 2
    del config["costs"]["exchanges"]["XDUB"]
    config["costs"]["exchanges"]["XPAR"]["commission_min"] = 200

    message = problems(write(tmp_path, config))

    assert "limits.max_risk_per_position = -1" in message
    assert "limits.max_position_size = 2" in message
    assert "costs.exchanges.XDUB: missing" in message
    assert "costs.exchanges.XPAR.commission_max = 150: not above" in message
    # One problem per line, after the line naming the file.
    assert len(message.splitlines()) == 5


def test_the_universe_is_the_suffix_calendars() -> None:
    assert set(load_risk_config(CHECKED_IN).costs.exchanges) == set(
        SUFFIX_CALENDARS.values()
    )


def test_a_copy_elsewhere_reports_its_own_path(tmp_path: Path) -> None:
    copy = tmp_path / RISK_FILE
    shutil.copy(CHECKED_IN, copy)

    assert load_risk_config(copy).source == copy.resolve()
