import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from fartt.candle_cache import DEFAULT_HISTORY_START_MS, CandleCache
from fartt.data import (
    ProviderUnavailable,
    UnknownTicker,
    UnsupportedExchange,
    YFinanceProvider,
)
from fartt.exchange import CcxtExchange, ExchangeUnavailable
from fartt.forecast import RepeatLastReturn, Settings
from fartt.server import build_server
from fartt.stock_cache import StockCache

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def _add_cache_options(command: argparse.ArgumentParser, interval: str) -> None:
    command.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="folder the candle cache lives in (default: %(default)s)",
    )
    command.add_argument(
        "--exchange", default="bitvavo", help="ccxt exchange id (default: %(default)s)"
    )
    command.add_argument(
        "--market",
        default="BTC/EUR",
        help="market in ccxt's format, e.g. BTC/EUR (default: %(default)s)",
    )
    command.add_argument(
        "--interval",
        default=interval,
        help="candle interval the exchange offers, e.g. 1m, 30m, 1h, 4h, 1d "
        "(default: %(default)s)",
    )


def _add_analysis_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--fee",
        type=float,
        default=Settings.fee,
        help="trading fee per leg, as a fraction: 0.0025 is 0.25%% (default: %(default)s)",
    )
    command.add_argument(
        "--slippage",
        type=float,
        default=Settings.slippage,
        help="slippage per leg, as a fraction: 0.001 is 0.1%% (default: %(default)s)",
    )
    command.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum expected return to act on, as a fraction: 0.007 is 0.7%% "
        "(default: the round-trip cost, 2 * (fee + slippage))",
    )
    command.add_argument(
        "--hit-rate-window",
        type=int,
        default=Settings.hit_rate_window,
        help="recent candles the hit rate covers (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fartt", description="Fartt: Financial Analysis Real Time Trading."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download",
        help="download a ticker's daily candles into the local cache",
        description=(
            "Download a ticker's split-adjusted daily candles, dividends and "
            "splits into the local cache, fetching only what's new."
        ),
    )
    download.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="folder the cache lives in (default: %(default)s)",
    )
    download.add_argument(
        "--ticker",
        required=True,
        help="ticker in Yahoo's format: AAPL for a US listing, ASML.AS for "
        "Euronext Amsterdam",
    )

    serve = commands.add_parser(
        "serve",
        help="run the MCP server over stdio (started by Claude Code from .mcp.json)",
        description="Run the MCP server for one market and interval over stdio.",
    )
    _add_cache_options(serve, interval="1h")
    _add_analysis_options(serve)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    logger.add("logs/cli.log", rotation="1 MB", level="INFO", format=LOG_FORMAT)

    if args.command == "download":
        _download(parser, args.assets_dir, args.ticker)
    elif args.command == "serve":
        settings = _settings(parser, args)
        _serve(
            parser, args.assets_dir, args.exchange, args.market, args.interval, settings
        )


def _open_cache(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
    require_exchange: bool,
) -> CandleCache:
    # Validate the options before anything touches the cache, so a typo
    # exits with a one-line message rather than a traceback.
    try:
        client = CcxtExchange(exchange_id=exchange)
    except ValueError:
        parser.error(
            f"unknown exchange '{exchange}'; use a ccxt exchange id, e.g. 'bitvavo'"
        )

    supported = client.supported_intervals()
    if interval not in supported:
        parser.error(
            f"interval '{interval}' is not offered by exchange '{exchange}' "
            f"(supported: {', '.join(supported)})"
        )

    try:
        known = client.has_market(market)
    except ExchangeUnavailable as error:
        if require_exchange:
            parser.exit(1, f"fartt: error: {error}\n")
        logger.warning(f"{error}; starting anyway without checking market '{market}'")
        known = True
    if not known:
        parser.error(
            f"market '{market}' not found on exchange '{exchange}'. "
            "Markets use ccxt's format, e.g. 'BTC/EUR' rather than 'BTC-EUR'."
        )

    try:
        return CandleCache(
            exchange=client,
            assets_dir=assets_dir,
            market=market,
            interval=interval,
            history_start_ms=DEFAULT_HISTORY_START_MS,
        )
    except ValueError:
        # The exchange offers it, but the cache can't size it (e.g. a
        # lowercase `1w` or seconds-based `1s`); see issue #59.
        parser.error(
            f"interval '{interval}' is offered by exchange '{exchange}' but not "
            "supported by the candle cache (units: m, h, d, W, M)"
        )


def _settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Settings:
    # argparse's float accepts "nan" and "inf", so check finiteness too.
    for option, value in (("--fee", args.fee), ("--slippage", args.slippage)):
        if not (math.isfinite(value) and value >= 0):
            parser.error(f"{option} must be a fraction >= 0, e.g. 0.0025 for 0.25%")
    if args.threshold is not None and not (
        math.isfinite(args.threshold) and args.threshold >= 0
    ):
        parser.error("--threshold must be a fraction >= 0, e.g. 0.007 for 0.7%")
    if args.hit_rate_window < 1:
        parser.error("--hit-rate-window must be at least 1")

    settings = Settings(
        fee=args.fee,
        slippage=args.slippage,
        threshold=args.threshold,
        hit_rate_window=args.hit_rate_window,
    )
    if args.threshold is not None and args.threshold < settings.round_trip_cost:
        logger.warning(
            f"--threshold {args.threshold:.2%} is below the round-trip cost "
            f"{settings.round_trip_cost:.2%}: entries can lose money on costs alone"
        )
    return settings


def _download(parser: argparse.ArgumentParser, assets_dir: Path, ticker: str) -> None:
    cache = StockCache(YFinanceProvider(), assets_dir)
    try:
        result = cache.update(ticker)
    except (UnknownTicker, UnsupportedExchange) as error:
        parser.error(str(error))
    except ProviderUnavailable as error:
        parser.exit(1, f"fartt: error: {error}; the cache is unchanged\n")

    directory = cache.directory(ticker)
    if not result.fetched:
        logger.info(f"{directory.name} is already up to date in {directory}")
    elif result.rebuilt is not None:
        logger.info(
            f"Rebuilt {directory.name}'s history in {directory} "
            f"({result.rebuilt}): {result.new_candles} candles"
        )
    else:
        logger.info(
            f"{directory.name}: {result.new_candles} new candles in {directory}"
        )


def _serve(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
    settings: Settings,
) -> None:
    cache = _open_cache(
        parser, assets_dir, exchange, market, interval, require_exchange=False
    )
    logger.info(f"Serving {market} {interval} candles from {cache.filepath} over stdio")
    threshold = (
        settings.round_trip_cost if settings.threshold is None else settings.threshold
    )
    logger.info(
        f"Analysis settings: fee {settings.fee:.2%} per leg, slippage "
        f"{settings.slippage:.2%} per leg, threshold {threshold:.2%}, "
        f"hit rate over {settings.hit_rate_window} candles"
    )
    build_server(cache, market, interval, RepeatLastReturn(), settings).run()


if __name__ == "__main__":
    main()
