import os
import sys
from pathlib import Path
from typing import Annotated, NamedTuple

from dotenv import dotenv_values
from loguru import logger
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from fartt.data import (
    Candle,
    Listing,
    ProviderUnavailable,
    UnknownTicker,
    UnsupportedExchange,
    YFinanceProvider,
    exchange_date,
    market_status,
    missing_sessions,
    normalise_ticker,
)
from fartt.stock_cache import StockCache, UpdateResult

# Claude Code caps MCP tool output at about 25k tokens by default, and
# numbers cost roughly one token per three characters. 200 compact rows stay
# well under that, and the agent needs recent context, not years of history.
MAX_CANDLES = 200

COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# Paths are relative to the repository, wherever Claude Code starts the server
# from. This assumes an editable install (`uv sync`), as the project runs.
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS_DIR = ROOT / "assets"
# FARTT_ASSETS_DIR moves the cache, set in the environment (the end-to-end
# test does) or in the repository's .env (see .env.example).
ASSETS_DIR_ENV = "FARTT_ASSETS_DIR"
ENV_FILE = ROOT / ".env"

INSTRUCTIONS = (
    "Fartt's stock analysis server. Tools take one ticker per call, in "
    "Yahoo's format: AAPL for a US listing, ASML.AS for Euronext Amsterdam "
    "(.PA Paris, .BR Brussels, .LS Lisbon, .MI Milan, .IR Dublin, .OL Oslo). "
    "A ticker the server hasn't seen is fetched on the spot. Prices are daily "
    "and split-adjusted, not dividend-adjusted: they are prices that traded, "
    "with dividends kept separately. "
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


AVERAGE_SESSIONS = 20


class QuoteResult(BaseModel):
    ticker: str = Field(description="The ticker, in Yahoo's format.")
    currency: str = Field(description="The currency prices are quoted in.")
    last: float = Field(description="The last traded price.")
    bid: float | None = Field(
        description="The best bid. Null when unknown; then so are ask and spread."
    )
    ask: float | None = Field(
        description="The best ask. Null when unknown; then so are bid and spread."
    )
    spread: float | None = Field(
        description=(
            "Ask minus bid as a fraction of their midpoint: 0.002 means 0.2%. "
            "Null when unknown, never 0."
        )
    )
    volume: float | None = Field(
        description="Shares traded in the current (or last) session so far."
    )
    average_volume_20d: float | None = Field(
        description=(
            "Mean daily volume over the 20 cached sessions before the quote's session."
        )
    )
    volume_ratio: float | None = Field(
        description=(
            "volume / average_volume_20d. Not normalised for time of day: "
            "during the session it compares a partial day with full days, so "
            "it reads low early on."
        )
    )
    market_open: bool = Field(
        description="Whether the ticker's exchange is trading now."
    )
    next_open: str | None = Field(
        description=(
            "When the next session opens, ISO 8601 in UTC. Null while the "
            "exchange is open."
        )
    )
    quote_time: str = Field(description="When the quote was struck, ISO 8601 in UTC.")
    delay_minutes: int | None = Field(
        description="How many minutes the provider delays this exchange's quotes."
    )
    note: str | None = Field(
        default=None, description="What's missing or unusual, if anything."
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


def _cache_error(tool: str, ticker: str, error: Exception) -> ToolError:
    # A folder it can't write, or a file it can't parse: retrying won't help.
    logger.error(f"{tool} {ticker}: {error}")
    return ToolError(
        f"The stock cache for {ticker} can't be read or written: {error}. "
        "Retrying won't help until the operator fixes or deletes the ticker's "
        "cache folder."
    )


# What a cache folder it can't write or a file it can't parse raises.
CACHE_ERRORS = (OSError, ValueError, KeyError)


def _unknown_ticker(ticker: str) -> str:
    return (
        f"The data provider doesn't know {ticker}. Check the ticker: Yahoo's "
        "format is AAPL for a US listing, ASML.AS for Euronext Amsterdam."
    )


def _update(
    tool: str, cache: StockCache, ticker: str
) -> tuple[str, UpdateResult | None]:
    """
    The ticker normalised, and the result of bringing its cache up to date,
    or `None` when the provider is unreachable. Every other failure becomes
    a ToolError: the SDK reports any other exception to the agent as a bare
    "Error executing tool", so each one says what to do.

    """
    try:
        ticker = normalise_ticker(ticker)
    except UnknownTicker as error:
        raise ToolError(str(error)) from error
    try:
        return ticker, cache.update(ticker)
    except UnsupportedExchange as error:
        raise ToolError(str(error)) from error
    except UnknownTicker as error:
        raise ToolError(_unknown_ticker(ticker)) from error
    except ProviderUnavailable as error:
        logger.warning(f"{tool} {ticker}: {error}")
        return ticker, None
    except CACHE_ERRORS as error:
        raise _cache_error(tool, ticker, error) from error


class Snapshot(NamedTuple):
    listing: Listing
    candles: list[Candle]  # Oldest first
    update: UpdateResult | None  # None when the provider was unreachable
    is_current: bool
    warning: str | None  # What's stale, if anything


def _snapshot(tool: str, cache: StockCache, ticker: str, count: int) -> Snapshot:
    """
    Up to `count` of the ticker's newest cached candles after bringing its
    cache up to date, and whether they're current. An unreachable provider
    serves what's cached, with a warning saying how far behind it is; with
    nothing cached, or no closed sessions yet, it's a ToolError.

    """
    ticker, update = _update(tool, cache, ticker)
    try:
        candles = cache.latest(ticker, count)
        listing = cache.listing(ticker)
    except CACHE_ERRORS as error:
        raise _cache_error(tool, ticker, error) from error
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
    warning = None
    if update is None and not is_current:
        behind = len(missing_sessions(listing.exchange, [last, newest])) + 1
        warning = (
            f"The data provider is unreachable, so these are cached sessions "
            f"through {last.isoformat()}, {behind} "
            f"session{'s' if behind != 1 else ''} behind. Try again later."
        )
    elif not is_current:
        warning = (
            f"The provider hasn't published {newest.isoformat()}'s session "
            f"yet; the last session is {last.isoformat()}. Try again later."
        )
    return Snapshot(listing, candles, update, is_current, warning)


def _unreachable(ticker: str) -> ToolError:
    return ToolError(
        f"The data provider is unreachable, so there's no quote for {ticker}. "
        "Try again later."
    )


def build_server(cache: StockCache) -> MCPServer:
    """
    Build the MCP server on `cache`, which also supplies the provider (for
    quotes) and the clock (for market hours).

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
        listing, candles, update, is_current, warning = _snapshot(
            "get_candles", cache, ticker, count
        )
        ticker = listing.ticker
        gaps = missing_sessions(listing.exchange, [c.date for c in candles])
        warnings = [warning] if warning else []
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

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_quote(
        ticker: Annotated[
            str,
            Field(description="One ticker in Yahoo's format: AAPL, ASML.AS."),
        ],
    ) -> QuoteResult:
        """
        Return a ticker's latest quote: last price, bid, ask and spread,
        today's volume against its 20-session average, whether its exchange
        is open, and how delayed the quote is. Use it to judge liquidity and
        where a limit order should sit.
        """
        ticker, update = _update("get_quote", cache, ticker)
        if update is None:
            raise _unreachable(ticker)
        try:
            quote = cache.provider.quote(ticker)
        except UnknownTicker as error:
            raise ToolError(_unknown_ticker(ticker)) from error
        except ProviderUnavailable as error:
            logger.warning(f"get_quote {ticker}: {error}")
            raise _unreachable(ticker) from error
        try:
            listing = cache.listing(ticker)
            candles = cache.latest(ticker, AVERAGE_SESSIONS + 1)
        except CACHE_ERRORS as error:
            raise _cache_error("get_quote", ticker, error) from error
        if listing is None:
            raise _unreachable(ticker)

        notes: list[str] = []
        bid, ask, spread = quote.bid, quote.ask, None
        if bid is None or ask is None:
            bid = ask = None
            notes.append("The spread is unknown: the provider has no bid or ask.")
        elif ask <= bid:
            # Matching orders would have traded: a crossed or locked quote is
            # stale or delayed data, not a free spread.
            bid = ask = None
            notes.append(
                "The spread is unknown: the provider's bid and ask are crossed "
                "or locked (stale or delayed data)."
            )
        else:
            spread = (ask - bid) / ((ask + bid) / 2)

        # Sessions before the quote's own, so today is never in its average.
        today = exchange_date(listing.exchange, quote.time)
        prior = [c for c in candles if c.date < today][-AVERAGE_SESSIONS:]
        if len(prior) < AVERAGE_SESSIONS and len(candles) > AVERAGE_SESSIONS:
            # A quote older than the newest cached sessions: read further back.
            try:
                candles = cache.latest(ticker, sys.maxsize)
            except CACHE_ERRORS as error:
                raise _cache_error("get_quote", ticker, error) from error
            prior = [c for c in candles if c.date < today][-AVERAGE_SESSIONS:]
        last_closed = cache.newest_closed_session(ticker)
        if today < last_closed:
            notes.append(
                f"The quote is from {today.isoformat()}, before the exchange's "
                f"last closed session ({last_closed.isoformat()}), so it may be "
                "stale, for instance because trading is suspended."
            )
        average = sum(c.volume for c in prior) / len(prior) if prior else None
        if 0 < len(prior) < AVERAGE_SESSIONS:
            notes.append(
                f"The average volume covers only {len(prior)} sessions, all that "
                "are cached."
            )
        elif not prior:
            notes.append("No earlier sessions are cached to average volume over.")
        ratio = quote.volume / average if quote.volume is not None and average else None

        status = market_status(listing.exchange, cache.clock())
        logger.info(
            f"get_quote {ticker}: {quote.last} {listing.currency}, open={status.open}"
        )
        return QuoteResult(
            ticker=listing.ticker,
            currency=listing.currency,
            last=quote.last,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=quote.volume,
            average_volume_20d=average,
            volume_ratio=ratio,
            market_open=status.open,
            next_open=status.next_open.isoformat() if status.next_open else None,
            quote_time=quote.time.isoformat(),
            delay_minutes=quote.delay_minutes,
            note=" ".join(notes) or None,
        )

    return server


def main() -> None:
    """
    Run the server over stdio: Claude Code starts it from `.mcp.json`
    (`uv run fartt`). Logs go to stderr, since stdout carries the protocol.

    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    setting = os.environ.get(ASSETS_DIR_ENV) or dotenv_values(ENV_FILE).get(
        ASSETS_DIR_ENV
    )
    assets_dir = ROOT / setting if setting else DEFAULT_ASSETS_DIR
    logger.info(f"Serving stocks cached in {assets_dir} over stdio")
    build_server(StockCache(YFinanceProvider(), assets_dir)).run()
