"""
The operator's reset of a tripped kill switch, outside the agent's reach.

`task reset-kill-switch` runs `python -m fartt.kill_switch reset [--note
TEXT]`. It reads the assets folder and the clock as the server does
(FARTT_ASSETS_DIR from the environment or `.env`, and FARTT_NOW), values
the portfolio on the cached data only, without calling the data provider
(the cache is as current as the last get_portfolio), and:

- tripped: appends a `reset` line to the journal, prints it, exits 0;
- not tripped: says so with the current drawdown, writes nothing, exits 1;
- a journal, cache or risk config it can't read: prints why, writes
  nothing, exits 2.

No MCP tool resets the switch, and the analysis session has no shell.

"""

import argparse
import json
import sys
from datetime import date

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from fartt.data import History, Listing, ProviderUnavailable, Quote
from fartt.journal import Journal, Reset
from fartt.risk_config import RISK_FILE, RiskConfig, RiskConfigError, load_risk_config
from fartt.server.server import (
    assets_dir_from_env,
    clock_from_env,
    portfolio_on_cache,
)
from fartt.stock_cache import StockCache

RESET = 0
NOT_TRIPPED = 1
FAILED = 2
TOOL = "reset-kill-switch"


class _Offline:
    """A data provider that never answers: the reset uses the cache only."""

    def listing(self, ticker: str) -> Listing:
        raise ProviderUnavailable("The kill switch reset uses cached data only")

    def history(self, ticker: str, start: date | None) -> History:
        raise ProviderUnavailable("The kill switch reset uses cached data only")

    def quote(self, ticker: str) -> Quote:
        raise ProviderUnavailable("The kill switch reset uses cached data only")


def reset(
    cache: StockCache, journal: Journal, risk_config: RiskConfig, note: str | None
) -> tuple[int, str]:
    """Reset the kill switch if it has tripped: the exit code and what to print."""
    try:
        portfolio, _ = portfolio_on_cache(TOOL, cache, journal, risk_config)
    except ToolError as error:
        return FAILED, str(error)
    switch, totals = portfolio.kill_switch, portfolio.totals
    if (
        not switch.tripped
        or switch.tripped_on is None
        or portfolio.newest_valuation is None
    ):
        return NOT_TRIPPED, (
            f"The kill switch isn't tripped: the portfolio is at a drawdown of "
            f"{totals.drawdown} below its peak of €{totals.peak:,.2f} "
            f"(threshold {switch.threshold}). Nothing was written."
        )
    try:
        line = Reset(
            reset_at=cache.clock().replace(microsecond=0),
            session=portfolio.newest_valuation,
            total_value=totals.total_value,
            peak=totals.peak,
            drawdown=totals.drawdown,
            tripped_on=switch.tripped_on,
            note=note,
        )
        journal.append_reset(line)
    except (ValidationError, ValueError, OSError) as error:
        return FAILED, f"Nothing was written: {error}"
    return RESET, (
        f"The kill switch, tripped on {switch.tripped_on.isoformat()} at a "
        f"drawdown of {switch.drawdown_at_trip}, is reset at a drawdown of "
        f"{totals.drawdown}; the peak starts again from €{totals.total_value:,.2f}. "
        f"Written to {journal.path}:\n"
        + json.dumps({"kind": "reset", **line.model_dump(mode="json")})
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fartt.kill_switch",
        description="Reset a tripped kill switch (the operator's, never the agent's).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("reset", help="Reset a tripped kill switch.")
    command.add_argument("--note", help="Why, in up to 200 characters.")
    arguments = parser.parse_args(argv)

    assets_dir = assets_dir_from_env()
    try:
        risk_config = load_risk_config(assets_dir / RISK_FILE)
    except RiskConfigError as error:
        print(error, file=sys.stderr)
        return FAILED
    clock = clock_from_env()
    cache = StockCache(_Offline(), assets_dir, clock=clock)
    code, text = reset(
        cache, Journal(assets_dir / "journal"), risk_config, arguments.note
    )
    print(text, file=sys.stdout if code != FAILED else sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
