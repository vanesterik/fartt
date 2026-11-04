import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from fartt.journal import BenchmarkClose, DecisionDraft, Journal, Signal


def draft(**changes: Any) -> DecisionDraft:
    fields: dict[str, Any] = {
        "recorded_at": datetime(2026, 10, 29, 21, 0, tzinfo=UTC),
        "ticker": "ASML.AS",
        "currency": "EUR",
        "exchange": "XAMS",
        "direction": "buy",
        "horizon": 20,
        "reasoning": "Trend up, support holding.",
        "signals": [
            Signal(name="trend", stance="for", note="up, aligned"),
            Signal(name="rsi", stance="against"),
        ],
        "decision_session": date(2026, 10, 29),
        "close": 612.4,
        "horizon_ends": date(2026, 11, 26),
        "benchmark": BenchmarkClose(
            ticker="IWDA.AS", currency="EUR", session=date(2026, 10, 29), close=101.25
        ),
    }
    return DecisionDraft(**(fields | changes))


def journal_file(directory: Path) -> Path:
    return directory / "decisions.jsonl"


def test_ids_count_up_from_1(tmp_path: Path) -> None:
    journal = Journal(tmp_path)

    appended = [journal.append(draft()) for _ in range(3)]

    assert [d.id for d in appended] == [1, 2, 3]
    assert journal.decisions() == appended


def test_a_new_journal_on_the_same_directory_continues(tmp_path: Path) -> None:
    first = Journal(tmp_path)
    appended = [first.append(draft()), first.append(draft(ticker="AAPL"))]

    restarted = Journal(tmp_path)

    assert restarted.decisions() == appended
    assert restarted.append(draft()).id == 3


def test_each_entry_is_one_line_marked_as_a_decision(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    appended = [journal.append(draft()), journal.append(draft(direction="pass"))]

    content = journal_file(tmp_path).read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    assert len(lines) == 2
    for line, decision in zip(lines, appended, strict=True):
        assert line.endswith("\n")
        assert json.loads(line) == {
            "kind": "decision",
            **decision.model_dump(mode="json"),
        }


def test_earlier_lines_are_never_rewritten(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(draft())
    before = journal_file(tmp_path).read_bytes()

    journal.append(draft())

    assert journal_file(tmp_path).read_bytes().startswith(before)


def test_a_torn_last_line_is_cut_before_appending(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(draft())
    first_line = journal_file(tmp_path).read_bytes()
    with journal_file(tmp_path).open("ab") as file:
        file.write(b'{"kind": "decision", "id": 2, "tick')

    appended = journal.append(draft())

    assert appended.id == 2
    lines = journal_file(tmp_path).read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    assert lines[0] == first_line
    assert lines[1].endswith(b"\n")


def test_a_damaged_line_is_reported_not_rewritten(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(draft())
    with journal_file(tmp_path).open("ab") as file:
        file.write(b"not json\n")
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="line 2"):
        journal.append(draft())
    with pytest.raises(ValueError, match="line 2"):
        journal.decisions()

    assert journal_file(tmp_path).read_bytes() == before


def test_reasoning_with_newlines_and_euros_round_trips(tmp_path: Path) -> None:
    journal = Journal(tmp_path)

    journal.append(draft(reasoning="Trend up.\nTarget €55."))

    assert journal.decisions()[0].reasoning == "Trend up.\nTarget €55."
    assert len(journal_file(tmp_path).read_bytes().splitlines()) == 1


def test_the_directory_is_created_on_the_first_write(tmp_path: Path) -> None:
    directory = tmp_path / "journal"
    journal = Journal(directory)

    assert journal.decisions() == []
    assert not directory.exists()

    journal.append(draft())

    assert journal_file(directory).exists()
