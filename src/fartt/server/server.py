import os
import sys
from pathlib import Path
from typing import Annotated

from loguru import logger
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from fartt.data import (
    Candle,
    ProviderUnavailable,
    UnknownTicker,
    UnsupportedExchange,
    YFinanceProvider,
    missing_sessions,
    normalise_ticker,
)
from fartt.stock_cache import StockCache, UpdateResult

# Claude Code caps MCP tool output at about 25k tokens by default, and
# numbers cost roughly one token per three characters. 200 compact rows stay
# well under that, and the agent needs recent context, not years of history.
MAX_CANDLES = 200

COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# The cache lives in the repository's assets/ wherever Claude Code starts the
# server from; FARTT_ASSETS_DIR points it elsewhere (the end-to-end test does).
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
ASSETS_DIR_ENV = "FARTT_ASSETS_DIR"

INSTRUCTIONS = (
    "Fartt's stock analysis server. Tools take one ticker per call, in "
    "Yahoo's format: AAPL for a US listing, ASML.AS for Euronext Amsterdam "
    "(.PA Paris, .BR Brussels, .LS Lisbon, .MI Milan, .IR Dublin, .OL Oslo). "
    "A ticker the server hasn't seen is fetched on the spot. Prices are daily "
    "and split-adjusted, not dividend-adjusted: they are prices that traded. "
    "Returns, costs and rates are fractions: 0.004 means 0.4%. If is_current "
    "is false the data is stale: read the warning before relying on it."
)

Row = tuple[str, float, float, float, float, float]


class CandlesResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    name: str = Field(description="The company's name.")
    currency: str = Field(description="The currency prices are quoted in.")
    exchange: str = Field(
        description="The exchange's calendar code, e.g. XAMS or XNYS."
    )
    columns: list[str] = Field(
        default=COLUMNS, description="Names of the values in each row, in order."
    )
    rows: list[Row] = Field(
        description=(
            "Closed daily sessions, oldest first: date (the session's date on "
            "the exchange, ISO 8601), open, high, low, close, volume. Prices "
            "are split-adjusted. The last row is the newest."
        )
    )
    new_candles: int = Field(
        description="Sessions fetched from the provider and cached by this call."
    )
    rebuilt: str | None = Field(
        default=None,
        description=(
            "Why this call rewrote the ticker's whole cached history (a split "
            "or an amended price), if it did."
        ),
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Sessions of the exchange missing between the rows, ISO 8601.",
    )
    is_current: bool = Field(
        description=(
            "True when the last row is the exchange's most recently closed "
            "session. False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(
        default=None, description="What's stale or missing, if anything."
    )


def _row(candle: Candle) -> Row:
    return (
        candle.date.isoformat(),
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
    )


def _cache_error(ticker: str, error: Exception) -> ToolError:
    # A folder it can't write, or a file it can't parse: retrying won't help.
    logger.error(f"get_candles {ticker}: {error}")
    return ToolError(
        f"The stock cache for {ticker} can't be read or written: {error}. "
        "Retrying won't help until the operator fixes or deletes the ticker's "
        "cache folder."
    )


def build_server(cache: StockCache) -> MCPServer:
    """
    Build the MCP server on `cache`.

    A factory rather than a module-level server, so tests can pass a cache
    over an in-memory provider and connect with the SDK's in-memory client.

    """
    server = MCPServer("fartt", instructions=INSTRUCTIONS)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_candles(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
        count: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_CANDLES,
                description="How many of the newest closed sessions to return.",
            ),
        ] = 60,
    ) -> CandlesResult:
        """
        Return a ticker's newest closed daily sessions, oldest first, after
        bringing its local cache up to date. A ticker not cached yet is
        fetched in full first.
        """
        # The SDK reports any exception other than ToolError to the agent as
        # a bare "Error executing tool", so every failure is turned into a
        # ToolError that says what to do.
        try:
            ticker = normalise_ticker(ticker)
        except UnknownTicker as error:
            raise ToolError(str(error)) from error
        try:
            update: UpdateResult | None = cache.update(ticker)
        except UnsupportedExchange as error:
            raise ToolError(str(error)) from error
        except UnknownTicker as error:
            raise ToolError(
                f"The data provider doesn't know {ticker}. Check the ticker: "
                "Yahoo's format is AAPL for a US listing, ASML.AS for Euronext "
                "Amsterdam."
            ) from error
        except ProviderUnavailable as error:
            logger.warning(f"get_candles {ticker}: {error}")
            update = None
        except (OSError, ValueError, KeyError) as error:
            raise _cache_error(ticker, error) from error

        try:
            candles = cache.latest(ticker, count)
            listing = cache.listing(ticker)
        except (OSError, ValueError, KeyError) as error:
            raise _cache_error(ticker, error) from error
        if update is None and (not candles or listing is None):
            raise ToolError(
                f"Nothing is cached for {ticker} yet and the data provider is "
                "unreachable. Try again later."
            )
        if not candles or listing is None:
            raise ToolError(
                f"The data provider has no closed sessions for {ticker} yet, "
                "for instance because it listed today. Try again after its "
                "first session closes."
            )

        newest = cache.newest_closed_session(ticker)
        last = candles[-1].date
        is_current = last == newest
        dates = [c.date for c in candles]
        gaps = missing_sessions(listing.exchange, dates)
        warnings: list[str] = []
        if update is None and not is_current:
            behind = len(missing_sessions(listing.exchange, [last, newest])) + 1
            warnings.append(
                f"The data provider is unreachable, so these are cached sessions "
                f"through {last.isoformat()}, {behind} "
                f"session{'s' if behind != 1 else ''} behind. Try again later."
            )
        elif not is_current:
            warnings.append(
                f"The provider hasn't published {newest.isoformat()}'s session "
                f"yet; the last row is {last.isoformat()}. Try again later."
            )
        if gaps:
            warnings.append(
                "The provider has no data for these sessions: "
                + ", ".join(day.isoformat() for day in gaps)
                + "."
            )

        new_candles = update.new_candles if update else 0
        rebuilt = update.rebuilt if update else None
        logger.info(
            f"get_candles {ticker}: {len(candles)} rows, {new_candles} new"
            + (f", rebuilt ({rebuilt})" if rebuilt else "")
            + f", current={is_current}"
        )
        return CandlesResult(
            ticker=listing.ticker,
            name=listing.name,
            currency=listing.currency,
            exchange=listing.exchange,
            rows=[_row(c) for c in candles],
            new_candles=new_candles,
            rebuilt=rebuilt,
            gaps=[day.isoformat() for day in gaps],
            is_current=is_current,
            warning=" ".join(warnings) or None,
        )

    return server


def main() -> None:
    """
    Run the server over stdio: Claude Code starts it from `.mcp.json`
    (`uv run fartt`). Logs go to stderr, since stdout carries the protocol.

    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    assets_dir = Path(os.environ.get(ASSETS_DIR_ENV) or DEFAULT_ASSETS_DIR)
    logger.info(f"Serving stocks cached in {assets_dir} over stdio")
    build_server(StockCache(YFinanceProvider(), assets_dir)).run()
