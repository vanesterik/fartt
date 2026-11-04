"""
The journal: every call the agent makes, as it was made.

The file is `decisions.jsonl` in the journal's directory, UTF-8 JSON Lines:
one JSON object per line, each line ending in a newline. A call is a line
with `"kind": "decision"` and the entry's fields. Lines are only ever
appended: nothing here rewrites, truncates or deletes a complete line. The
one exception is a torn last line (no newline: a write a kill cut short,
never returned to the agent), which the next append cuts first.

`kind` lets later stories (scores, trade plans) add their own lines to the
same file without rewriting one; readers skip the kinds they don't know.

"""

import fcntl
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

from pydantic import BaseModel, Field, ValidationError, field_validator

from fartt.data import SUFFIX_CALENDARS

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
            last = max((d.id for d in self._read()), default=0)
            decision = Decision(id=last + 1, **draft.model_dump())
            line = json.dumps(
                {"kind": DECISION, **decision.model_dump(mode="json")},
                ensure_ascii=False,
            )
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()
                os.fsync(file.fileno())
        return decision

    def decisions(self) -> list[Decision]:
        """Every decision, oldest first."""
        if not self.directory.exists():
            return []
        with self._lock(fcntl.LOCK_SH):
            return self._read()

    @contextmanager
    def _lock(self, operation: int) -> Generator[None]:
        with open(self.directory / ".lock", "a") as lock:
            fcntl.flock(lock, operation)
            yield

    def _read(self) -> list[Decision]:
        # A torn last line isn't read; only an append cuts it.
        if not self.path.exists():
            return []
        decisions: list[Decision] = []
        content = self.path.read_bytes()
        complete = content[: content.rfind(b"\n") + 1]
        for number, raw in enumerate(complete.splitlines(), start=1):
            entry = _parse(self.path, number, raw)
            if entry.get("kind") == DECISION:
                try:
                    decisions.append(Decision.model_validate(entry))
                except ValidationError as error:
                    raise ValueError(
                        f"{self.path}, line {number}: not a journal entry ({error})"
                    ) from None
        return decisions


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
