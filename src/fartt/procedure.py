"""
The PRD's eight-step analysis procedure (section 5) as data, and the text of
the `analyze_ticker` prompt generated from it.

A later epic fills in its step by adding its tools to the step's entry (and
the signal names they feed to `TOOL_SIGNALS`) and clearing its `epic`.

"""

from typing import NamedTuple


class Step(NamedTuple):
    number: int
    name: str
    purpose: str
    # The tools that carry the step today, as the server registers them, in
    # the order the agent calls them.
    tools: tuple[str, ...]
    # The epic that adds the step's tools, for a step without any yet.
    epic: str | None = None
    # What the agent does with the tools, beyond calling them.
    detail: str = ""


EARLIER_CALLS_TOOL = "get_decisions"
RECORD_TOOL = "record_decision"
# Fixed until the trade plan (epic H) sets it, so returns in get_performance
# are comparable across calls.
HORIZON = 20

STEPS: tuple[Step, ...] = (
    Step(1, "Market context", "Is the market as a whole risk-on or risk-off?", (), "F"),
    Step(2, "Sector selection", "Which sectors lead and which lag?", (), "F"),
    Step(3, "Screening", "Which stocks are worth a closer look?", (), "G"),
    Step(
        4,
        "Fundamental analysis",
        "Is the business sound and fairly valued?",
        (),
        "G",
    ),
    Step(
        5,
        "Technical analysis (timing)",
        "Is now a good moment, and where is the risk?",
        ("get_trend", "get_indicators", "get_support_resistance", "get_candles"),
        detail=(
            "Start with the trend: the daily and weekly direction and whether "
            "they align. Then the indicators, the support and resistance zones "
            "with their signs, and the recent candles for what the others "
            "don't show."
        ),
    ),
    Step(
        6,
        "Trade plan and risk",
        "Entry, stop, target and position size.",
        (),
        "H",
        detail=f"The horizon is fixed at {HORIZON} sessions until then.",
    ),
    Step(
        7,
        "Execution",
        "Can the stock be traded at a fair price right now?",
        ("get_quote",),
        detail=("Read the spread and whether the market is open. No order is placed."),
    ),
    Step(
        8,
        "Monitoring and review",
        "Make the call and put it in the journal, so it is scored later.",
        (RECORD_TOOL,),
        detail=(
            "Only now form a view: weigh the evidence of steps 5 and 7 by the "
            "decision rules below, make the call, and record it."
        ),
    ),
)

# The journal's signal names (journal.SIGNALS) each tool's output feeds, so
# the agent names a signal by what it read rather than inventing one.
TOOL_SIGNALS: dict[str, tuple[str, ...]] = {
    "get_trend": ("trend",),
    "get_indicators": ("moving_averages", "rsi", "macd", "volume", "atr"),
    "get_support_resistance": ("support_resistance",),
    "get_candles": (),
    "get_quote": ("spread",),
}

REPORT_SECTIONS: tuple[str, ...] = (
    "The call",
    "Evidence for",
    "Evidence against",
    "Skipped steps",
    "Earlier calls",
    "What was recorded",
)

_REPORT_DETAILS = (
    "`buy`, `pass` (and which kind), or not recorded, with why.",
    "each piece, with the tool it came from.",
    (
        "each piece, with the tool it came from; never left empty without saying "
        "why there is none."
    ),
    'each as "skipped: no tool yet (epic X)".',
    (
        f"what `{EARLIER_CALLS_TOOL}` returned, and whether the evidence has "
        "changed since an open call."
    ),
    "the journal `id` and `horizon_ends`, or that nothing was recorded.",
)


def procedure_tools() -> set[str]:
    """Every tool the procedure names, as the server registers them."""
    return {EARLIER_CALLS_TOOL, *(tool for step in STEPS for tool in step.tools)}


def _step_text(step: Step, ticker: str) -> str:
    heading = f"### Step {step.number}: {step.name}\n{step.purpose}"
    if not step.tools:
        lines = [
            heading,
            f'Report it as "skipped: no tool yet (epic {step.epic})".',
        ]
        if step.detail:
            lines.append(step.detail)
        return "\n".join(lines)
    calls = ", then ".join(f'`{tool}(ticker="{ticker}")`' for tool in step.tools)
    if step.tools == (RECORD_TOOL,):
        calls = f"`{RECORD_TOOL}`, as below"
    return f"{heading}\nCall {calls}. {step.detail}"


def _signals_text() -> str:
    lines = [
        f"- `{tool}`: " + ", ".join(f"`{name}`" for name in names)
        for tool, names in TOOL_SIGNALS.items()
        if names
    ]
    return "\n".join(lines)


def analysis_prompt(ticker: str) -> str:
    """The prompt text for analysing `ticker` (already normalised)."""
    steps = "\n\n".join(_step_text(step, ticker) for step in STEPS)
    report = "\n".join(
        f"{i}. **{section}**: {detail}"
        for i, (section, detail) in enumerate(
            zip(REPORT_SECTIONS, _REPORT_DETAILS, strict=True), 1
        )
    )
    return f"""\
Analyse {ticker} by fartt's eight-step procedure. Work through the steps in \
order, with the server's tools only, and end with a call in the journal.

## Evidence first, view last
Collect the evidence of every step in order, including the evidence against, \
before you form a view. Don't form or state a view before step 8, and don't go \
looking for evidence that confirms a pattern you think you see.

## Before step 1: earlier calls
Call `{EARLIER_CALLS_TOOL}(ticker="{ticker}")` and mention its calls in the \
report. If one is still open (`status: open`, whatever `horizon_passed` says), \
run the full analysis and say whether the evidence has changed since that \
call, but don't record a call: two overlapping calls on one stock are one bet \
counted twice in `get_performance`.

## The steps
A step without a tool yet is reported as skipped, never silently left out. \
Don't fill it in from your own knowledge or from tools outside this server.

{steps}

## Stale data
If any tool returns `is_current: false`, report it with the tool's warning and \
don't record a call (`{RECORD_TOOL}` would refuse it anyway).

## Decision rules
- No `buy` while the weekly trend (`get_trend`'s `weekly.direction`) is \
`down`. This is the one hard rule.
- Everything else is weighed together, never scored or summed into points.
- When in doubt, `pass`. A `buy` needs the evidence to clearly favour it, and \
the reasoning must name what speaks against it.
- Say what kind of `pass` it is: from evidence pointing down, or from mixed \
evidence. A `pass` is scored as "won't beat IWDA.AS", so the two must be told \
apart later.
- If `weekly` is null (a recent listing), a `buy` is still possible, but the \
reasoning says the weekly trend couldn't be judged.

## Recording the call
Call `{RECORD_TOOL}` with ticker="{ticker}", the direction, horizon={HORIZON} \
(sessions), the reasoning, and `signals`: each piece of evidence the call \
used, by the journal's names below, each `for` or `against` the call. The \
evidence against goes in as `against`, not left out.

{_signals_text()}

## Report
Report to the operator in these sections, in this order:
{report}
"""
