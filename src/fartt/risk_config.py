"""
The operator's risk limits and trading costs, read from `risk.toml` in the
assets folder when the server starts.

The file holds every value; the code holds no defaults. A file the server
can't read or a value it can't accept stops the server: `load_risk_config`
raises a `RiskConfigError` listing every problem, each with its key as a
dotted path, its value and what's wrong.

"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fartt.data import SUFFIX_CALENDARS

RISK_FILE = "risk.toml"

# A typo guard, not a market fact: a rate typed as a percentage (0.25 for
# 0.0025) lands far above it. A real tax above 5% would need it raised.
MAX_RATE = 0.05

Rate = Annotated[float, Field(ge=0, lt=MAX_RATE)]
Amount = Annotated[float, Field(ge=0)]

STRICT = ConfigDict(strict=True, extra="forbid", frozen=True)


class Limits(BaseModel):
    model_config = STRICT

    starting_capital: Annotated[float, Field(gt=0)]
    max_risk_per_position: Annotated[float, Field(gt=0, lt=1)]
    max_position_size: Annotated[float, Field(gt=0, le=1)]
    max_open_positions: Annotated[int, Field(ge=1)]
    min_reward_to_risk: Annotated[float, Field(ge=1)]
    kill_switch_drawdown: Annotated[float, Field(gt=0, lt=1)]


class ExchangeCosts(BaseModel):
    model_config = STRICT

    commission: Rate
    commission_min: Amount
    # Absent means no maximum.
    commission_max: float | None = None
    commission_currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    tax_on_buy: Rate
    fee_on_sell: Rate
    fee_per_trade: Amount


class Costs(BaseModel):
    model_config = STRICT

    tariff: str
    currency_conversion: Rate
    slippage_per_leg: Rate
    exchanges: dict[str, ExchangeCosts]


class RiskFile(BaseModel):
    """What `risk.toml` holds, and all it may hold."""

    model_config = STRICT

    limits: Limits
    costs: Costs


class RiskConfig(BaseModel):
    """The file's values and the file they were read from."""

    model_config = ConfigDict(frozen=True)

    source: Path
    limits: Limits
    costs: Costs


class RiskConfigError(ValueError):
    def __init__(self, path: Path, problems: list[str]) -> None:
        self.path = path
        self.problems = problems
        super().__init__(
            f"The risk config {path} can't be used:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def load_risk_config(path: Path) -> RiskConfig:
    """
    Read and check `path`. Raises `RiskConfigError` naming every problem: a
    missing or unreadable file, invalid TOML, a missing or unknown key, a
    value of the wrong type or out of range, or `costs.exchanges` without
    exactly one table per exchange in the universe.

    """
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError as error:
        raise RiskConfigError(
            path,
            [
                (
                    "it doesn't exist; copy the repository's assets/risk.toml "
                    "there and adjust it"
                )
            ],
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise RiskConfigError(path, [f"it can't be read: {error}"]) from error
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise RiskConfigError(path, [f"it isn't valid TOML: {error}"]) from error

    problems: list[str] = []
    parsed: RiskFile | None = None
    try:
        parsed = RiskFile.model_validate(raw)
    except ValidationError as error:
        problems.extend(_pydantic_problem(e) for e in error.errors())
    problems.extend(_cross_problems(raw))
    if problems or parsed is None:
        raise RiskConfigError(path, problems)
    return RiskConfig(source=path.resolve(), limits=parsed.limits, costs=parsed.costs)


def _show(value: object) -> str:
    return repr(value)


def _pydantic_problem(error: Any) -> str:
    key = ".".join(str(part) for part in error["loc"])
    if error["type"] == "missing":
        return f"{key}: missing"
    if error["type"] == "extra_forbidden":
        return f"{key} = {_show(error['input'])}: not a known key"
    return f"{key} = {_show(error['input'])}: {error['msg']}"


def _table(value: object) -> dict[str, object]:
    """`value` when it is a TOML table, else an empty one."""
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _number(table: object, key: str) -> float | None:
    """`table[key]` when it is a number (not a bool), else None."""
    value = _table(table).get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def _cross_problems(raw: Mapping[str, Any]) -> list[str]:
    """
    The checks across keys, on the raw TOML, so they're reported alongside
    the per-value problems instead of after they're fixed.

    """
    problems: list[str] = []
    limits = raw.get("limits")
    risk = _number(limits, "max_risk_per_position")
    size = _number(limits, "max_position_size")
    if risk is not None and size is not None and risk > size:
        problems.append(
            f"limits.max_risk_per_position = {_show(risk)}: above "
            f"limits.max_position_size ({_show(size)})"
        )

    exchanges = _table(raw.get("costs")).get("exchanges")
    if not isinstance(exchanges, dict):
        return problems
    universe = list(dict.fromkeys(SUFFIX_CALENDARS.values()))
    tables = _table(cast(object, exchanges))
    for code, table in tables.items():
        low = _number(table, "commission_min")
        high = _number(table, "commission_max")
        if low is not None and high is not None and high <= low:
            problems.append(
                f"costs.exchanges.{code}.commission_max = {_show(high)}: "
                f"not above commission_min ({_show(low)})"
            )
    for code in universe:
        if code not in tables:
            problems.append(
                f"costs.exchanges.{code}: missing; every exchange in the "
                "universe needs one"
            )
    for code in tables:
        if code not in universe:
            problems.append(
                f"costs.exchanges.{code}: not an exchange in the universe "
                f"({', '.join(universe)})"
            )
    return problems
