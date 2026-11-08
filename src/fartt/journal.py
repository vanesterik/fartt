"""
The journal: every call the agent makes, as it was made.

The file is `decisions.jsonl` in the journal's directory, UTF-8 JSON Lines:
one JSON object per line, each line ending in a newline. A call is a line
with `"kind": "decision"` and the entry's fields. Lines are only ever
appended: nothing here rewrites, truncates or deletes a complete line. The
one exception is a torn last line (no newline: a write a kill cut short,
never returned to the agent), which the next append cuts first.

A call's score is a line with `"kind": "score"` and its `decision_id`,
written once its horizon has passed. A decision has at most one, and it
comes after the decision's line. A buy may carry its trade plan in the
decision line's `plan`.

A buy's plan is paper traded: a `"kind": "fill"` line when its limit order
fills, then a `"kind": "exit"` line when the position closes, or a
`"kind": "expiry"` line when the order runs out unfilled. Each carries its
`decision_id`, comes after its decision's line, whose call has a plan, and
is written at most once; an expiry never goes with a fill, and an exit only
follows a fill. A line breaking these rules is a damaged line. `kind` lets
later stories add their own lines to the same file without rewriting one;
readers skip the kinds they don't know.

"""

import fcntl
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fartt.data import SUFFIX_CALENDARS
from fartt.trade_plan import LegCosts, TradePlan

# The index every call is measured against: iShares Core MSCI World
# (accumulating, EUR) on Euronext Amsterdam.
BENCHMARK = "IWDA.AS"

# The evidence the server measures today; later epics add theirs here.
SignalName = Literal[
    "trend",
    "support_resistance",
    "moving_averages",
    "rsi",
    "macd",
    "volume",
    "atr",
    "spread",
]
# Derived, so the names the SDK validates against are the only list.
SIGNALS: tuple[str, ...] = get_args(SignalName)
Direction = Literal["buy", "pass"]
Stance = Literal["for", "against"]

DECISION = "decision"
SCORE = "score"
FILL = "fill"
EXPIRY = "expiry"
EXIT = "exit"

ExitReason = Literal["stop", "target", "horizon"]


class Signal(BaseModel):
    """One piece of evidence a call rests on."""

    name: SignalName = Field(description="The evidence, one of a fixed list.")
    stance: Stance = Field(description="Whether it pointed for or against the call.")
    note: str | None = Field(
        default=None,
        max_length=200,
        description="The reading in a few words, e.g. 'RSI 74, above 70 for 3 days'.",
    )


class BenchmarkClose(BaseModel):
    """The benchmark's last closed session when the call was recorded."""

    ticker: str = Field(description="The benchmark's ticker.")
    currency: str = Field(description="The currency its close is in.")
    session: date = Field(description="Its last closed session, on its own exchange.")
    close: float = Field(description="That session's close.")


class DecisionDraft(BaseModel):
    """A call before the journal gives it an id."""

    recorded_at: datetime = Field(description="When the call was recorded (UTC).")
    ticker: str = Field(description="The stock the call is on.")
    currency: str = Field(description="The currency its close is in.")
    exchange: str = Field(description="Its exchange's calendar code, e.g. XAMS.")
    direction: Direction = Field(
        description="buy: it will beat the benchmark over the horizon; pass: it won't."
    )
    horizon: int = Field(
        description="The horizon, in sessions of the stock's exchange."
    )
    reasoning: str = Field(description="The agent's summary of why.")
    signals: list[Signal] = Field(description="The evidence the call rests on.")
    decision_session: date = Field(
        description="The stock's last closed session when the call was recorded."
    )
    close: float = Field(description="That session's close.")
    horizon_ends: date = Field(description="The session the horizon ends on.")
    benchmark: BenchmarkClose = Field(
        description="The benchmark's last close when the call was recorded."
    )
    # Absent from lines written before plans existed, which read as null.
    plan: TradePlan | None = Field(
        default=None,
        description=(
            "The trade plan recorded with a buy, as plan_trade worked it out "
            "when the call was recorded; null without one."
        ),
    )

    @field_validator("exchange")
    @classmethod
    def _in_the_universe(cls, exchange: str) -> str:
        # A hand-edited code would otherwise fail later, in the calendar.
        if exchange not in SUFFIX_CALENDARS.values():
            raise ValueError(f"{exchange} isn't an exchange in the universe")
        return exchange


class Decision(DecisionDraft):
    """A call as the journal stores it."""

    id: int = Field(description="1, 2, 3, ... in recording order.")


class Leg(BaseModel):
    """One end of a scored horizon."""

    session: date = Field(description="The session, on its own exchange.")
    close: float = Field(description="Its close, split-adjusted, as cached.")
    rate: float | None = Field(
        default=None,
        description=(
            "The exchange rate used for that session, in units of the stock's "
            "currency per euro; null for a euro price."
        ),
    )


class BenchmarkScore(BaseModel):
    """What the benchmark did over the call's horizon."""

    model_config = ConfigDict(populate_by_name=True)

    ticker: str = Field(description="The benchmark's ticker.")
    start: Leg = Field(description="Its session and close the call was stamped with.")
    end: Leg = Field(
        description="Its last session on or before the call's horizon_ends."
    )
    return_: float = Field(
        validation_alias="return",
        serialization_alias="return",
        description="Its return over the horizon, in EUR: 0.004 means 0.4%.",
    )


class Score(BaseModel):
    """How a call turned out, measured in EUR against the benchmark."""

    model_config = ConfigDict(populate_by_name=True)

    decision_id: int = Field(description="The id of the call scored.")
    scored_at: datetime = Field(description="When the call was scored (UTC).")
    start: Leg = Field(description="The stock at decision_session.")
    end: Leg = Field(description="The stock at horizon_ends.")
    dividends: float = Field(
        description=(
            "Dividends per share going ex after decision_session up to and "
            "including horizon_ends, in the stock's currency."
        )
    )
    return_: float = Field(
        validation_alias="return",
        serialization_alias="return",
        description="The stock's return with dividends, in EUR: 0.004 means 0.4%.",
    )
    benchmark: BenchmarkScore = Field(description="The benchmark over the horizon.")
    excess_return: float = Field(description="return minus the benchmark's return.")
    right: bool = Field(
        description=("buy: excess_return above 0; pass: excess_return 0 or below.")
    )


class Fill(BaseModel):
    """A plan's limit buy, filled on a closed session."""

    model_config = ConfigDict(allow_inf_nan=False)

    decision_id: int = Field(description="The id of the call whose plan filled.")
    settled_at: datetime = Field(description="When the fill was settled (UTC).")
    session: date = Field(description="The session it filled on.")
    price: float = Field(description="The fill price, in the stock's currency.")
    at_open: bool = Field(
        description="True when the session opened at or below the entry: filled at the open."
    )
    shares: int = Field(description="The plan's shares.")
    rate: float | None = Field(
        description=(
            "The session's EUR rate (units of the stock's currency per euro); "
            "null for EUR."
        )
    )
    value: float = Field(description="shares × price, in EUR.")
    costs: LegCosts = Field(description="The buy leg's costs, itemised, in EUR.")


class Expiry(BaseModel):
    """A plan's limit buy that wasn't reached within its window."""

    decision_id: int = Field(description="The id of the call whose plan expired.")
    settled_at: datetime = Field(description="When the expiry was settled (UTC).")
    session: date = Field(description="The window's last session.")


class Exit(BaseModel):
    """A paper position, closed on a closed session."""

    model_config = ConfigDict(allow_inf_nan=False)

    decision_id: int = Field(description="The id of the call whose position closed.")
    settled_at: datetime = Field(description="When the exit was settled (UTC).")
    session: date = Field(description="The session it closed on.")
    reason: ExitReason = Field(
        description="stop, target, or horizon: the close of the call's horizon_ends."
    )
    price: float = Field(description="The exit price, in the stock's currency.")
    at_open: bool = Field(description="True when it closed at the session's open.")
    slippage: float = Field(
        description=(
            "stop − price when the session opened below the stop, in the "
            "stock's currency; else 0."
        )
    )
    rate: float | None = Field(description="The session's EUR rate; null for EUR.")
    value: float = Field(description="shares × price, in EUR.")
    costs: LegCosts = Field(description="The sell leg's costs, itemised, in EUR.")
    dividends: float = Field(
        description=(
            "Dividends going ex after the fill session up to and including "
            "this one, gross, times the shares, in EUR at this rate."
        )
    )
    pnl: float = Field(
        description=("value − the fill's value + dividends − both legs' costs, in EUR.")
    )


PositionLine = Fill | Expiry | Exit


class Positions(NamedTuple):
    """The paper trading lines, each keyed by the id of its call."""

    fills: dict[int, Fill]
    expiries: dict[int, Expiry]
    exits: dict[int, Exit]


class _Contents(NamedTuple):
    decisions: list[Decision]
    scores: dict[int, Score]
    positions: Positions


def _empty() -> _Contents:
    return _Contents([], {}, Positions({}, {}, {}))


def _breach(line: PositionLine, contents: _Contents) -> str | None:
    """Why `line` can't follow `contents`, or `None` when it can."""
    decision_id = line.decision_id
    decision = next((d for d in contents.decisions if d.id == decision_id), None)
    if decision is None:
        return f"no earlier call #{decision_id}"
    if decision.plan is None:
        return f"call #{decision_id} has no plan"
    fills, expiries, _ = contents.positions
    if isinstance(line, Fill) and decision_id in expiries:
        return f"call #{decision_id}'s order expired"
    if isinstance(line, Expiry) and decision_id in fills:
        return f"call #{decision_id}'s order filled"
    if isinstance(line, Exit) and decision_id not in fills:
        return f"call #{decision_id} has no fill to exit"
    return None


def _already(line: PositionLine, positions: Positions) -> bool:
    """Whether a line of `line`'s kind exists for its call."""
    stored: dict[int, Any]
    if isinstance(line, Fill):
        stored = positions.fills
    elif isinstance(line, Expiry):
        stored = positions.expiries
    else:
        stored = positions.exits
    return line.decision_id in stored


def _add(line: PositionLine, positions: Positions) -> None:
    if isinstance(line, Fill):
        positions.fills[line.decision_id] = line
    elif isinstance(line, Expiry):
        positions.expiries[line.decision_id] = line
    else:
        positions.exits[line.decision_id] = line


KINDS: dict[str, type[Fill | Expiry | Exit]] = {
    FILL: Fill,
    EXPIRY: Expiry,
    EXIT: Exit,
}


def _kind(line: PositionLine) -> str:
    return next(kind for kind, model in KINDS.items() if isinstance(line, model))


class Journal:
    """The append-only journal in `directory`, created on the first write."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "decisions.jsonl"

    def append(self, draft: DecisionDraft) -> Decision:
        """Store `draft` with the next id, and return it as stored."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock(fcntl.LOCK_EX):
            _cut_torn_tail(self.path)
            # The id is assigned under the lock, so two servers writing at
            # once still get distinct ids.
            decisions = self._read().decisions
            last = max((d.id for d in decisions), default=0)
            decision = Decision(id=last + 1, **draft.model_dump())
            self._write_line({"kind": DECISION, **decision.model_dump(mode="json")})
        return decision

    def append_score(self, score: Score) -> Score | None:
        """
        Store `score`, and return it. `None` when its call already has a
        score, which stays as it is: a score is stored once. A `ValueError`
        when there's no such call.

        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock(fcntl.LOCK_EX):
            _cut_torn_tail(self.path)
            # Checked under the lock, so two runs at once can't both score it.
            decisions, scores, _ = self._read()
            if score.decision_id not in {d.id for d in decisions}:
                raise ValueError(f"no call #{score.decision_id} in {self.path}")
            if score.decision_id in scores:
                return None
            self._write_line(
                {"kind": SCORE, **score.model_dump(mode="json", by_alias=True)}
            )
        return score

    def append_fill(self, fill: Fill) -> Fill | None:
        """Store `fill`; `None` when its call already has one. See `_append`."""
        return self._append(fill)

    def append_expiry(self, expiry: Expiry) -> Expiry | None:
        """Store `expiry`; `None` when its call already has one. See `_append`."""
        return self._append(expiry)

    def append_exit(self, exit: Exit) -> Exit | None:
        """Store `exit`; `None` when its call already has one. See `_append`."""
        return self._append(exit)

    def _append[L: (Fill, Expiry, Exit)](self, line: L) -> L | None:
        """
        Store a paper trading line, and return it. `None` (nothing written)
        when its call already has a line of its kind, which stays as it is.
        A `ValueError` when the line breaks a rule: no such call, a call
        without a plan, an expiry with a fill, a fill after an expiry, an
        exit without a fill.

        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock(fcntl.LOCK_EX):
            _cut_torn_tail(self.path)
            # Checked under the lock, so two runs at once write it once.
            contents = self._read()
            if _already(line, contents.positions):
                return None
            breach = _breach(line, contents)
            if breach is not None:
                raise ValueError(f"{breach} in {self.path}")
            self._write_line({"kind": _kind(line), **line.model_dump(mode="json")})
        return line

    def decisions(self) -> list[Decision]:
        """Every decision, oldest first."""
        if not self.directory.exists():
            return []
        with self._lock(fcntl.LOCK_SH):
            return self._read().decisions

    def scores(self) -> dict[int, Score]:
        """Every score, keyed by the id of the call it scores."""
        if not self.directory.exists():
            return {}
        with self._lock(fcntl.LOCK_SH):
            return self._read().scores

    def positions(self) -> Positions:
        """Every fill, expiry and exit, each keyed by the id of its call."""
        if not self.directory.exists():
            return _empty().positions
        with self._lock(fcntl.LOCK_SH):
            return self._read().positions

    def _write_line(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

    @contextmanager
    def _lock(self, operation: int) -> Generator[None]:
        with open(self.directory / ".lock", "a") as lock:
            fcntl.flock(lock, operation)
            yield

    def _read(self) -> _Contents:
        # A torn last line isn't read; only an append cuts it.
        contents = _empty()
        decisions, scores, positions = contents
        if not self.path.exists():
            return contents
        ids: set[int] = set()
        content = self.path.read_bytes()
        complete = content[: content.rfind(b"\n") + 1]
        for number, raw in enumerate(complete.splitlines(), start=1):
            entry = _parse(self.path, number, raw)
            try:
                if entry.get("kind") == DECISION:
                    decision = Decision.model_validate(entry)
                    decisions.append(decision)
                    ids.add(decision.id)
                elif entry.get("kind") == SCORE:
                    score = Score.model_validate(entry)
                    if score.decision_id not in ids:
                        raise ValueError(f"no earlier call #{score.decision_id}")
                    if score.decision_id in scores:
                        raise ValueError(f"call #{score.decision_id} is already scored")
                    scores[score.decision_id] = score
                elif (model := KINDS.get(entry.get("kind", ""))) is not None:
                    line = model.model_validate(entry)
                    if _already(line, positions):
                        raise ValueError(
                            f"call #{line.decision_id} already has a {_kind(line)} line"
                        )
                    breach = _breach(line, contents)
                    if breach is not None:
                        raise ValueError(breach)
                    _add(line, positions)
            except ValueError as error:  # ValidationError is a ValueError
                raise ValueError(
                    f"{self.path}, line {number}: not a journal entry ({error})"
                ) from None
        return contents


def _parse(path: Path, number: int, raw: bytes) -> dict[str, Any]:
    try:
        entry: object = json.loads(raw.decode("utf-8"))
    except ValueError:  # JSONDecodeError and UnicodeDecodeError
        entry = None
    if isinstance(entry, dict):
        return cast(dict[str, Any], entry)
    raise ValueError(f"{path}, line {number}: not a journal entry")


def _cut_torn_tail(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        with path.open("rb+") as file:
            file.truncate(content.rfind(b"\n") + 1)
