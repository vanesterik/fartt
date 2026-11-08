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
PORTFOLIO_TOOL = "get_portfolio"
PLAN_TOOL = "plan_trade"
RISK_CONFIG_TOOL = "get_risk_config"
# Fixed, so returns in get_performance are comparable across calls; a plan's
# position closes at its call's horizon, so call and trade cover one span.
HORIZON = 20
# How many sessions a plan's entry waits to be filled.
VALID_FOR = 5

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
        (PORTFOLIO_TOOL, PLAN_TOOL),
        detail=(
            "A plan stores nothing: whether the trade pays at these levels is "
            "evidence, not yet a view.\n"
            "- Read the room first: `kill_switch.tripped`, `limits.room` and "
            "`totals.free_cash`. If the kill switch is tripped, `limits.room` "
            "is 0 or the free cash can't pay for a position, carry on with the analysis "
            "and report it; step 8 records nothing for a buy.\n"
            "- Then try a plan from the evidence of step 5, never from your "
            "own idea of prices: the entry at the top of the nearest support "
            "zone (or the last close when it's inside that zone), the stop "
            "below that zone's low, the target at the low of the next "
            f"resistance zone above the entry; `valid_for` {VALID_FOR}.\n"
            f"- If `{PLAN_TOOL}` refuses on reward-to-risk, adjust once within "
            "the evidence (a lower entry still in the support zone, or a "
            "further resistance zone as target, as `get_support_resistance` "
            "names them), never only to make the numbers pass. If it still "
            "refuses, that's evidence against a buy.\n"
            "- Report the plan's size, position share, risk, reward-to-risk "
            "after costs and the limit that set the size, or the refusal and "
            "its reason."
        ),
    ),
    Step(
        7,
        "Execution",
        "Can the stock be traded at a fair price right now?",
        ("get_quote", RISK_CONFIG_TOOL),
        detail=(
            "Read the spread and whether the market is open. A buy becomes a "
            "paper limit order at the plan's entry, filled by the server from "
            "the daily candles. No real order is placed. A spread wider than "
            "the slippage the risk config assumes per leg "
            "(`slippage_per_leg`) is evidence against."
        ),
    ),
    Step(
        8,
        "Monitoring and review",
        "Make the call and put it in the journal, so it is scored later.",
        (RECORD_TOOL,),
        detail=(
            "Only now form a view: weigh the evidence of steps 5 to 7 by the "
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
    # A plan or the portfolio's room is evidence, but no journal signal.
    PORTFOLIO_TOOL: (),
    PLAN_TOOL: (),
    RISK_CONFIG_TOOL: (),
}

REPORT_SECTIONS: tuple[str, ...] = (
    "The call",
    "Evidence for",
    "Evidence against",
    "The plan",
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
    (
        "the plan as accepted (shares, entry, stop, target, risk, "
        "reward-to-risk, valid until) or why there is none: refused, with the "
        "limit, or no room in the portfolio."
    ),
    'each as "skipped: no tool yet (epic X)".',
    (
        f"what `{EARLIER_CALLS_TOOL}` returned, and whether the evidence has "
        "changed since an open call."
    ),
    (
        "the journal `id` and `horizon_ends`, for a buy the plan's `valid_for` "
        "window, or that nothing was recorded."
    ),
)


def procedure_tools() -> set[str]:
    """Every tool the procedure names, as the server registers them."""
    return {EARLIER_CALLS_TOOL, *(tool for step in STEPS for tool in step.tools)}


def _call(tool: str, ticker: str) -> str:
    """How the procedure writes a call of `tool` on `ticker`."""
    if tool in (PORTFOLIO_TOOL, RISK_CONFIG_TOOL):
        return f"`{tool}()`"
    if tool == PLAN_TOOL:
        return f'`{tool}(ticker="{ticker}", entry, stop, target, valid_for)`'
    return f'`{tool}(ticker="{ticker}")`'


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
    calls = ", then ".join(_call(tool, ticker) for tool in step.tools)
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
- Say what kind of `pass` it is, in the reasoning's first words; there are \
three kinds: from evidence pointing down, from mixed evidence, or "no plan \
fits" (the evidence favours a buy, but `{PLAN_TOOL}` refused on \
reward-to-risk, also after the one adjustment; name the limit and the value). \
A `pass` is scored as "won't beat IWDA.AS", so the kinds must be told apart \
later.
- If the evidence favours a buy but there's no room in the portfolio (the kill \
switch tripped, the open-position limit reached, or not enough free cash), \
record nothing: it's about the portfolio, not the stock. The report says which \
limit.
- If `weekly` is null (a recent listing), a `buy` is still possible, but the \
reasoning says the weekly trend couldn't be judged.

## Recording the call
Call `{RECORD_TOOL}` with ticker="{ticker}", the direction, horizon={HORIZON} \
(sessions), the reasoning, and `signals`: each piece of evidence the call \
used, by the journal's names below, each `for` or `against` the call. The \
evidence against goes in as `against`, not left out. A `buy` also passes \
`plan`: the `entry`, `stop`, `target` and `valid_for` that `{PLAN_TOOL}` \
accepted; the server refuses a buy without one. A `pass` has no plan.

{_signals_text()}

## Report
Report to the operator in these sections, in this order:
{report}
"""
