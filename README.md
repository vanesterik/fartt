![Fartt](./fartt.avif)

# Fartt

**Fartt** is an [MCP](https://modelcontextprotocol.io) server that gives an AI agent the tools to analyse stocks on Euronext and US exchanges: market data, technical indicators, support and resistance, fundamentals, market context and charts, with a journal that scores every call afterwards. [Claude Code](https://claude.com/claude-code) is the analyst and the operator's interface. It stands for **F**inancial **A**nalysis **R**eal **T**ime **T**rading.

## Motivation

Fartt is decision support for personal investing: it helps make disciplined decisions about stocks, it doesn't sell signals, and it isn't a product. The agent combines many signals into an informed view rather than trusting a single model, and whether that view has any edge is measured in the journal, not assumed. Paper trading comes first; real money only if the journal shows a real hit rate.

## Name

The name **Fartt** is an abbreviation play on the title "Financial Analysis Real Time Trading". It is also a nod to Fart, the gaseous being who appeared in the episode [Mortynight Run](https://www.imdb.com/title/tt4832254/) of the Rick and Morty series. There is a bit of musical part in the episode, which I really enjoyed. The name **Fartt** is a homage to that episode.

## Project Status

This is a solo research project. The plan is in the [PRD](docs/product/stock-analysis-prd.md), and the work is delivered in epics on the [project board](https://github.com/users/vanesterik/projects/3), starting with [Data Foundation](https://github.com/vanesterik/fartt/issues/88). Epics A to D form the walking skeleton: a full analysis running end to end on one ticker.

## Installation

The project's commands run through [Task](https://taskfile.dev) and [uv](https://docs.astral.sh/uv/), on Python 3.14 (pinned in `.python-version`; uv installs it if needed). With both installed (for example `mise use -g aqua:go-task/task`), install the project and its git hooks with:

```bash
task setup
```

`task --list` shows the other commands, such as `task check` to run every check the git hooks run.

## Usage

Open Claude Code in the repository and ask about a stock. The MCP server is registered in `.mcp.json`, so Claude Code starts it (`uv run fartt`) and calls its tools; there is no command to run by hand.

```
> How has ASML.AS traded over the last three months?
```

The server's one tool so far, `get_candles`, returns a ticker's newest closed daily sessions. Tickers use Yahoo's format: `AAPL` for a US listing, `ASML.AS` for Euronext Amsterdam (`.PA`, `.BR`, `.LS`, `.MI`, `.IR` and `.OL` for the other Euronext markets). The first call for a ticker fetches its full history from Yahoo Finance (through [yfinance](https://github.com/ranaroussi/yfinance)) into `assets/stocks/<TICKER>/`; later calls fetch only what's new, and rebuild the history when a split appears. Prices are split-adjusted, not dividend-adjusted; dividends and splits are kept separately. No API keys are needed. `FARTT_ASSETS_DIR` moves the cache elsewhere.

## Project Structure

The project follows the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) layout, adapted to how this project actually works:

```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── assets             <- The stock cache: assets/stocks/<TICKER>/ (not in git).
    │
    ├── docs
    │   ├── product        <- The PRD for the stock analysis MCP.
    │   ├── specs          <- Design docs for individual pieces of work.
    │   └── plans          <- Implementation plans.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-kve-exploratory-data-analysis`.
    │
    ├── tests              <- Unit tests, mirroring the src/fartt layout.
    │
    ├── pyproject.toml     <- Project + dependency config (managed with `uv`).
    ├── .python-version    <- The pinned Python version (3.14).
    ├── Taskfile.yml       <- Everyday commands (`task --list`).
    ├── .mcp.json          <- Registers the MCP server for Claude Code.
    │
    └── src/fartt           <- Source code for use in this project.
        ├── stock_cache.py <- Per-ticker cache of daily candles, dividends, splits and listing.
        ├── constants.py   <- Shared column names, indicator labels, colour palette.
        │
        ├── data           <- Market data, behind an interface so the provider stays swappable.
        │   ├── provider.py          <- The DataProvider Protocol and its errors.
        │   ├── types.py             <- Candle, Dividend, Split, History, Listing, Quote.
        │   ├── sessions.py          <- Ticker suffix to exchange calendar; closed sessions.
        │   └── yfinance_provider.py <- Its yfinance implementation.
        │
        ├── server         <- The MCP server.
        │   └── server.py      <- build_server(), its tools, and main() (the `fartt` console script).
        │
        ├── features       <- Calculations over Polars DataFrames and return series.
        │   ├── calculate_technical_indicators.py  <- Bollinger Bands, EMA, MACD, RSI via TA-Lib.
        │   ├── technical_indicators_config.py
        │   ├── calculate_candle_returns.py        <- Signed percent change per candle.
        │   └── calculate_trade_returns.py         <- A long-only backtest with costs and stops.
        │
        └── visualization  <- Matplotlib plotting helpers.
            ├── candlestick_chart.py  <- Candles with indicator overlays (mplfinance).
            └── plot_styles.py        <- Shared rcParams.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Rick and Morty](https://www.imdb.com/title/tt2861424/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
