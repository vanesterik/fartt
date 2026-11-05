from fartt.journal import SIGNALS
from fartt.procedure import (
    HORIZON,
    REPORT_SECTIONS,
    STEPS,
    TOOL_SIGNALS,
    analysis_prompt,
    procedure_tools,
)

PROMPT = analysis_prompt("ASML.AS")

STEP_NAMES = [
    "Market context",
    "Sector selection",
    "Screening",
    "Fundamental analysis",
    "Technical analysis",
    "Trade plan and risk",
    "Execution",
    "Monitoring and review",
]


def step_text(number: int) -> str:
    """The prompt's text for step `number`, up to the next heading."""
    start = PROMPT.index(f"Step {number}:")
    end = PROMPT.index("\n#", start)
    return PROMPT[start:end]


def test_the_steps_are_the_prds_eight_in_order() -> None:
    assert [step.number for step in STEPS] == list(range(1, 9))
    assert [step.name for step in STEPS][4] == "Technical analysis (timing)"
    positions = [
        PROMPT.index(f"Step {i}: {name}") for i, name in enumerate(STEP_NAMES, 1)
    ]
    assert positions == sorted(positions)


def test_the_prompt_is_about_the_ticker() -> None:
    assert "ASML.AS" in PROMPT


def test_steps_without_tools_are_reported_as_skipped_with_their_epic() -> None:
    for number, epic in [(1, "F"), (2, "F"), (3, "G"), (4, "G"), (6, "H")]:
        assert f"skipped: no tool yet (epic {epic})" in step_text(number)
    for number in (5, 7, 8):
        assert "skipped: no tool yet" not in step_text(number)


def test_step_5_names_the_technical_tools_trend_first() -> None:
    text = step_text(5)
    tools = ["get_trend", "get_indicators", "get_support_resistance", "get_candles"]
    positions = [text.index(tool) for tool in tools]
    assert positions == sorted(positions)


def test_step_7_is_the_quote_and_places_no_order() -> None:
    text = step_text(7)
    assert "get_quote" in text
    assert "No order is placed" in text


def test_the_analysis_starts_with_the_tickers_earlier_calls() -> None:
    assert PROMPT.index('get_decisions(ticker="ASML.AS")') < PROMPT.index("Step 1:")


def test_step_8_ends_with_record_decision() -> None:
    assert "record_decision" in step_text(8)


def test_the_decision_rules_are_stated() -> None:
    assert "Evidence first, view last" in PROMPT
    assert "No `buy` while the weekly trend" in PROMPT
    assert "`weekly.direction`) is `down`" in PROMPT
    assert "never scored or summed" in PROMPT
    assert "When in doubt, `pass`" in PROMPT
    assert "name what speaks against it" in PROMPT
    assert "evidence pointing down" in PROMPT
    assert "mixed evidence" in PROMPT
    assert f"horizon={HORIZON}" in PROMPT
    assert HORIZON == 20
    assert "`for` or `against`" in PROMPT


def test_every_journal_signal_is_named_and_mapped_to_a_tool() -> None:
    mapped = {name for names in TOOL_SIGNALS.values() for name in names}
    assert mapped == set(SIGNALS)
    for name in SIGNALS:
        assert f"`{name}`" in PROMPT


def test_an_unjudged_weekly_trend_is_said_in_the_reasoning() -> None:
    assert "`weekly` is null" in PROMPT


def test_an_open_call_or_stale_data_means_no_new_call() -> None:
    assert "still open" in PROMPT
    assert "`is_current: false`" in PROMPT
    assert PROMPT.count("don't record a call") >= 2


def test_skipped_steps_are_not_filled_in_from_elsewhere() -> None:
    assert "own knowledge" in PROMPT


def test_the_report_sections_are_named_in_order() -> None:
    assert REPORT_SECTIONS == (
        "The call",
        "Evidence for",
        "Evidence against",
        "Skipped steps",
        "Earlier calls",
        "What was recorded",
    )
    report = PROMPT[PROMPT.index("## Report") :]
    positions = [report.index(f"**{section}**") for section in REPORT_SECTIONS]
    assert positions == sorted(positions)


def test_procedure_tools_names_every_tool_the_prompt_uses() -> None:
    assert procedure_tools() == {
        "get_decisions",
        "get_trend",
        "get_indicators",
        "get_support_resistance",
        "get_candles",
        "get_quote",
        "record_decision",
    }
    for tool in procedure_tools():
        assert tool in PROMPT


def test_the_prompt_stays_readable_in_one_go() -> None:
    assert len(PROMPT.split()) < 1500
