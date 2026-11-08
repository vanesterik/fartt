import json
from datetime import date
from pathlib import Path

import pytest

from fartt.journal import Journal
from fartt.trade_plan import calculate_plan
from tests.fakes import (
    RISK_CONFIG,
    buy_plan,
    decision_draft,
    exit_line,
    expiry_line,
    fill_line,
    plan_score_line,
    reset_line,
)
from tests.fakes import decision_score as score


def journal_file(directory: Path) -> Path:
    return directory / "decisions.jsonl"


def test_ids_count_up_from_1(tmp_path: Path) -> None:
    journal = Journal(tmp_path)

    appended = [journal.append(decision_draft()) for _ in range(3)]

    assert [d.id for d in appended] == [1, 2, 3]
    assert journal.decisions() == appended


def test_a_new_journal_on_the_same_directory_continues(tmp_path: Path) -> None:
    first = Journal(tmp_path)
    appended = [
        first.append(decision_draft()),
        first.append(decision_draft(ticker="AAPL")),
    ]

    restarted = Journal(tmp_path)

    assert restarted.decisions() == appended
    assert restarted.append(decision_draft()).id == 3


def test_each_entry_is_one_line_marked_as_a_decision(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    appended = [
        journal.append(decision_draft()),
        journal.append(decision_draft(direction="pass")),
    ]

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
    journal.append(decision_draft())
    before = journal_file(tmp_path).read_bytes()

    journal.append(decision_draft())

    assert journal_file(tmp_path).read_bytes().startswith(before)


def test_a_torn_last_line_is_cut_before_appending(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    first_line = journal_file(tmp_path).read_bytes()
    with journal_file(tmp_path).open("ab") as file:
        file.write(b'{"kind": "decision", "id": 2, "tick')

    appended = journal.append(decision_draft())

    assert appended.id == 2
    lines = journal_file(tmp_path).read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    assert lines[0] == first_line
    assert lines[1].endswith(b"\n")


def test_a_damaged_line_is_reported_not_rewritten(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    with journal_file(tmp_path).open("ab") as file:
        file.write(b"not json\n")
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="line 2"):
        journal.append(decision_draft())
    with pytest.raises(ValueError, match="line 2"):
        journal.decisions()

    assert journal_file(tmp_path).read_bytes() == before


def test_an_exchange_outside_the_universe_is_a_damaged_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    line = journal_file(tmp_path).read_text().replace('"XAMS"', '"XXXX"')
    with journal_file(tmp_path).open("a") as file:
        file.write(line)
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="line 2"):
        journal.decisions()

    assert journal_file(tmp_path).read_bytes() == before


def test_reasoning_with_newlines_and_euros_round_trips(tmp_path: Path) -> None:
    journal = Journal(tmp_path)

    journal.append(decision_draft(reasoning="Trend up.\nTarget €55."))

    assert journal.decisions()[0].reasoning == "Trend up.\nTarget €55."
    assert len(journal_file(tmp_path).read_bytes().splitlines()) == 1


def test_the_directory_is_created_on_the_first_write(tmp_path: Path) -> None:
    directory = tmp_path / "journal"
    journal = Journal(directory)

    assert journal.decisions() == []
    assert not directory.exists()

    journal.append(decision_draft())

    assert journal_file(directory).exists()


def test_a_score_round_trips(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    decision = journal.append(decision_draft())

    stored = journal.append_score(score(1))

    assert stored == score(1)
    assert journal.scores() == {1: score(1)}
    lines = journal_file(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    line = json.loads(lines[1])
    assert line["kind"] == "score"
    assert line["return"] == 0.045069
    assert line["benchmark"]["return"] == 0.02716
    assert journal.decisions() == [decision]


def test_a_call_is_scored_once(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    journal.append_score(score(1))
    before = journal_file(tmp_path).read_bytes()

    assert journal.append_score(score(1, right=False)) is None

    assert journal_file(tmp_path).read_bytes() == before
    assert journal.scores()[1].right


def test_a_score_for_an_unknown_call_is_refused(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="7"):
        journal.append_score(score(7))

    assert journal_file(tmp_path).read_bytes() == before


def test_a_second_score_line_is_a_damaged_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    journal.append_score(score(1))
    second = journal_file(tmp_path).read_text().splitlines(keepends=True)[1]
    with journal_file(tmp_path).open("a") as file:
        file.write(second)

    with pytest.raises(ValueError, match="line 3"):
        journal.scores()
    with pytest.raises(ValueError, match="line 3"):
        journal.decisions()


def test_a_score_without_its_call_is_a_damaged_line(tmp_path: Path) -> None:
    other = Journal(tmp_path / "other")
    other.append(decision_draft())
    other.append_score(score(1))
    line = journal_file(tmp_path / "other").read_text().splitlines(keepends=True)[1]
    journal_file(tmp_path).write_text(line)

    with pytest.raises(ValueError, match="line 1"):
        Journal(tmp_path).scores()


def test_a_torn_line_is_cut_before_a_score(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    first_line = journal_file(tmp_path).read_bytes()
    with journal_file(tmp_path).open("ab") as file:
        file.write(b'{"kind": "score", "decision_id": 1, "sta')

    assert journal.append_score(score(1)) == score(1)

    lines = journal_file(tmp_path).read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    assert lines[0] == first_line
    assert json.loads(lines[1])["kind"] == "score"


def test_a_line_from_before_plans_reads_without_one(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    entry = json.loads(journal_file(tmp_path).read_text(encoding="utf-8"))
    del entry["plan"]
    journal_file(tmp_path).write_text(json.dumps(entry) + "\n", encoding="utf-8")

    assert journal.decisions()[0].plan is None


def test_a_plan_is_stored_with_its_call(tmp_path: Path) -> None:
    plan = calculate_plan(
        ticker="ASML.AS",
        currency="EUR",
        exchange_code="XAMS",
        based_on=date(2026, 10, 29),
        last_close=612.4,
        atr=15.0,
        entry=600.0,
        stop=580.0,
        target=680.0,
        valid_for=5,
        rate=None,
        rate_session=None,
        rate_is_current=True,
        portfolio_value=RISK_CONFIG.limits.starting_capital,
        limits=RISK_CONFIG.limits,
        costs=RISK_CONFIG.costs,
    )
    journal = Journal(tmp_path)

    stored = journal.append(decision_draft(plan=plan))

    assert stored.plan == plan
    assert Journal(tmp_path).decisions()[0].plan == plan


def with_plan(journal: Journal) -> None:
    journal.append(decision_draft(plan=buy_plan()))


def test_a_fill_and_an_exit_round_trip(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)

    assert journal.append_fill(fill_line(1)) == fill_line(1)
    assert journal.append_exit(exit_line(1)) == exit_line(1)

    for reader in (journal, Journal(tmp_path)):
        positions = reader.positions()
        assert positions.fills == {1: fill_line(1)}
        assert positions.exits == {1: exit_line(1)}
        assert positions.expiries == {}
    lines = journal_file(tmp_path).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["decision", "fill", "exit"]
    assert json.loads(lines[2]) == {
        "kind": "exit",
        **exit_line(1).model_dump(mode="json"),
    }


def test_an_expiry_round_trips(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)

    assert journal.append_expiry(expiry_line(1)) == expiry_line(1)

    assert Journal(tmp_path).positions().expiries == {1: expiry_line(1)}
    last = journal_file(tmp_path).read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(last)["kind"] == "expiry"


def test_each_position_line_is_written_once(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)
    journal.append(decision_draft(plan=buy_plan()))
    journal.append_fill(fill_line(1))
    journal.append_exit(exit_line(1))
    journal.append_expiry(expiry_line(2))
    before = journal_file(tmp_path).read_bytes()

    assert journal.append_fill(fill_line(1, price=590.0)) is None
    assert journal.append_exit(exit_line(1, price=570.0)) is None
    assert journal.append_expiry(expiry_line(2)) is None

    assert journal_file(tmp_path).read_bytes() == before
    assert journal.positions().fills[1].price == 600.0


@pytest.mark.parametrize("decision_id", [1, 7])
def test_a_position_line_for_a_call_without_a_plan_or_unknown_is_refused(
    tmp_path: Path, decision_id: int
) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match=f"#{decision_id}"):
        journal.append_fill(fill_line(decision_id))
    with pytest.raises(ValueError, match=f"#{decision_id}"):
        journal.append_expiry(expiry_line(decision_id))

    assert journal_file(tmp_path).read_bytes() == before


def test_an_exit_without_a_fill_is_refused(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="fill"):
        journal.append_exit(exit_line(1))

    assert journal_file(tmp_path).read_bytes() == before


def test_an_expiry_after_a_fill_is_refused(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)
    journal.append_fill(fill_line(1))
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="filled"):
        journal.append_expiry(expiry_line(1))

    assert journal_file(tmp_path).read_bytes() == before


def test_a_fill_after_an_expiry_is_refused(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)
    journal.append_expiry(expiry_line(1))
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="expired"):
        journal.append_fill(fill_line(1))

    assert journal_file(tmp_path).read_bytes() == before


def test_a_duplicated_fill_line_is_a_damaged_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)
    journal.append_fill(fill_line(1))
    second = journal_file(tmp_path).read_text().splitlines(keepends=True)[1]
    with journal_file(tmp_path).open("a") as file:
        file.write(second)

    with pytest.raises(ValueError, match="line 3"):
        journal.positions()
    with pytest.raises(ValueError, match="line 3"):
        journal.decisions()


def test_an_exit_line_without_its_fill_is_a_damaged_line(tmp_path: Path) -> None:
    other = Journal(tmp_path / "other")
    with_plan(other)
    other.append_fill(fill_line(1))
    other.append_exit(exit_line(1))
    lines = journal_file(tmp_path / "other").read_text().splitlines(keepends=True)
    journal_file(tmp_path).write_text(lines[0] + lines[2])

    with pytest.raises(ValueError, match="line 2"):
        Journal(tmp_path).positions()


def test_a_reset_is_appended_and_read_back(tmp_path: Path) -> None:
    # @AC5
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    reset = reset_line()

    assert journal.append_reset(reset) == reset

    assert journal.resets() == [reset]
    last = journal_file(tmp_path).read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(last) == {"kind": "reset", **reset.model_dump(mode="json")}
    assert len(journal.decisions()) == 1


def test_resets_come_back_oldest_first(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    first = reset_line(note=None)
    second = reset_line(session=date(2026, 11, 20), note="second")

    journal.append_reset(first)
    journal.append_reset(second)

    assert journal.resets() == [first, second]


def test_without_a_journal_there_are_no_resets(tmp_path: Path) -> None:
    assert Journal(tmp_path / "journal").resets() == []


def test_a_damaged_reset_line_is_a_damaged_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    line = {"kind": "reset", **reset_line().model_dump(mode="json")}
    del line["peak"]
    with journal_file(tmp_path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(line) + "\n")

    with pytest.raises(ValueError, match="line 2"):
        journal.resets()


def test_a_reset_on_a_damaged_journal_writes_nothing(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(decision_draft())
    with journal_file(tmp_path).open("ab") as file:
        file.write(b"not json\n")
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="line 2"):
        journal.append_reset(reset_line())

    assert journal_file(tmp_path).read_bytes() == before


def test_a_reset_note_is_at_most_200_characters() -> None:
    with pytest.raises(ValueError):
        reset_line(note="x" * 201)


def closed(journal: Journal) -> None:
    with_plan(journal)
    journal.append_fill(fill_line(1))
    journal.append_exit(exit_line(1))


def test_a_plan_score_round_trips(tmp_path: Path) -> None:
    # @AC1
    journal = Journal(tmp_path)
    closed(journal)

    assert journal.append_plan_score(plan_score_line(1)) == plan_score_line(1)

    assert Journal(tmp_path).plan_scores() == {1: plan_score_line(1)}
    last = json.loads(journal_file(tmp_path).read_text().splitlines()[-1])
    assert last["kind"] == "plan_score"
    assert last["return"] == -0.03924
    assert last["benchmark"]["return"] == 0.007407
    assert last["realised_r"] == -0.988
    assert last["hit"] is False


def test_a_plan_is_scored_once(tmp_path: Path) -> None:
    # @AC4
    journal = Journal(tmp_path)
    closed(journal)
    journal.append_plan_score(plan_score_line(1))
    before = journal_file(tmp_path).read_bytes()

    assert journal.append_plan_score(plan_score_line(1, hit=True)) is None

    assert journal_file(tmp_path).read_bytes() == before
    assert not journal.plan_scores()[1].hit


def test_a_plan_score_without_an_exit_is_refused(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with_plan(journal)
    journal.append_fill(fill_line(1))
    before = journal_file(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="#1"):
        journal.append_plan_score(plan_score_line(1))

    assert journal_file(tmp_path).read_bytes() == before


def test_a_plan_score_line_before_its_exit_is_a_damaged_line(tmp_path: Path) -> None:
    other = Journal(tmp_path / "other")
    closed(other)
    other.append_plan_score(plan_score_line(1))
    lines = journal_file(tmp_path / "other").read_text().splitlines(keepends=True)
    journal_file(tmp_path).write_text(lines[0] + lines[1] + lines[3] + lines[2])

    with pytest.raises(ValueError, match="line 3"):
        Journal(tmp_path).plan_scores()


def test_a_second_plan_score_line_is_a_damaged_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    closed(journal)
    journal.append_plan_score(plan_score_line(1))
    last = journal_file(tmp_path).read_text().splitlines(keepends=True)[-1]
    with journal_file(tmp_path).open("a") as file:
        file.write(last)

    with pytest.raises(ValueError, match="line 5"):
        journal.decisions()


def test_without_a_journal_there_are_no_plan_scores(tmp_path: Path) -> None:
    assert Journal(tmp_path / "none").plan_scores() == {}
