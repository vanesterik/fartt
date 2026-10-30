![Fartt](./fartt.avif)

# Fartt

**Fartt** is an [MCP](https://modelcontextprotocol.io) server that gives an AI agent the tools to analyse stocks on Euronext and US exchanges: market data, technical indicators, support and resistance, fundamentals, market context and charts, with a journal that scores every call afterwards. [Claude Code](https://claude.com/claude-code) is the analyst and the operator's interface. It stands for **F**inancial **A**nalysis **R**eal **T**ime **T**rading.

## Motivation

Fartt is decision support for personal investing: it helps make disciplined decisions about stocks, it doesn't sell signals, and it isn't a product. The agent combines many signals into an informed view rather than trusting a single model, and whether that view has any edge is measured in the journal, not assumed. Paper trading comes first; real money only if the journal shows a real hit rate.

## Name

The name **Fartt** is an abbreviation play on the title "Financial Analysis Real Time Trading". It is also a nod to Fart, the gaseous being who appeared in the episode [Mortynight Run](https://www.imdb.com/title/tt4832254/) of the Rick and Morty series. There is a bit of musical part in the episode, which I really enjoyed. The name **Fartt** is a homage to that episode.

## Project Status

This is a solo research project. The plan is in the [PRD](docs/product/stock-analysis-prd.md), and the work is delivered in epics on the [project board](https://github.com/users/vanesterik/projects/3), starting with [Data Foundation](https://github.com/vanesterik/fartt/issues/88). Epics A to D form the walking skeleton: a full analysis running end to end on one ticker.

The code is being moved over from an earlier direction one story at a time, so parts of it (the ccxt exchange layer, the forecast tools, the model pipeline) are still in the tree until the story that replaces them lands.

## Installation

The project's commands run through [Task](https://taskfile.dev) and [uv](https://docs.astral.sh/uv/), on Python 3.14 (pinned in `.python-version`; uv installs it if needed). With both installed (for example `mise use -g aqua:go-task/task`), install the project and its git hooks with:

```bash
task setup
```

`task --list` shows the other commands, such as `task check` to run every check the git hooks run.

## Usage

Two commands work today: `fartt download` fills a ticker's local cache, and `fartt serve` runs the MCP server that Claude Code starts.

```bash
uv run fartt download --ticker ASML.AS
```

`download` fetches a ticker's split-adjusted daily candles, dividends and splits from Yahoo Finance (through [yfinance](https://github.com/ranaroussi/yfinance)) into `assets/stocks/<TICKER>/`. The first run fetches the full history; later runs fetch only what's new, and rebuild the history when a split appears. Tickers use Yahoo's format: `AAPL` for a US listing, `ASML.AS` for Euronext Amsterdam. No API keys are needed.

The MCP server is registered in `.mcp.json`, so a Claude Code session in this directory starts it (`uv run fartt serve`).

`fartt serve` also takes the analysis settings: `--fee`, `--slippage`, `--threshold` and `--hit-rate-window`, all fractions except the window (see `uv run fartt serve --help`).

There is no `train` command. Training and evaluation run from the notebooks, one per candidate architecture: `notebooks/2.0-kve-data-analysis-mlp.ipynb` (MLP) and `notebooks/2.1-kve-data-analysis-cnn.ipynb` (CNN). Each loads the cached data, computes the target (the candle return: the signed percent change as a fraction), builds sliding lag windows with a chronological 60/20/20 train/val/test split, fits its model (see [Project Status](#project-status)), and reports directional accuracy, RMSE and MAE. Neither saves a checkpoint yet.

Run `uv run fartt --help` for the full set of options.

## Project Structure

The project follows the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) layout, adapted to how this project actually works:

```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── assets             <- Cached candle data, one CSV per market/interval.
    │
    ├── artifacts          <- Versioned, trained model checkpoints.
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
        ├── candle_cache.py <- Append-only CSV candle cache, kept up to date through the exchange layer.
        ├── stock_cache.py <- Per-ticker cache of daily candles, dividends and splits.
        ├── cli.py         <- argparse entrypoint (the `fartt` console script).
        ├── constants.py   <- Shared column names, indicator labels, colour palette.
        ├── utils.py       <- Path helpers (project root, candle/model file paths).
        │
        ├── exchange       <- Market data, behind an interface so the exchange stays swappable.
        │   ├── exchange.py        <- The Exchange Protocol and ExchangeUnavailable.
        │   ├── ccxt_exchange.py   <- Its ccxt implementation (Bitvavo by default).
        │   └── candle.py          <- The Candle type.
        │
        ├── forecast       <- The Forecaster interface, the naive baseline and the analysis.
        │   ├── forecaster.py      <- Forecaster Protocol, Forecast, forecast().
        │   ├── baseline.py        <- RepeatLastReturn.
        │   └── analysis.py        <- Settings, Analysis, analyze().
        │
        ├── server         <- The MCP server.
        │   └── server.py      <- build_server() and the four tools.
        │
        ├── features       <- Feature engineering over Polars DataFrames.
        │   ├── calculate_technical_indicators.py
        │   ├── calculate_candle_returns.py
        │   └── calculate_trade_returns.py
        │
        ├── model          <- Regression pipeline epic B (Model Selection) builds on.
        │   ├── prepare_datasets.py   <- Loads candles, computes candle returns, builds lag windows + split.
        │   ├── builder.py            <- ModelBuilder Protocol every architecture's builder satisfies.
        │   ├── blocks.py             <- Reusable linear/conv blocks.
        │   ├── mlp_config.py         <- MLP hyperparameters.
        │   ├── mlp_builder.py        <- Assembles the MLP from its config.
        │   ├── cnn_config.py         <- CNN hyperparameters.
        │   ├── cnn_builder.py        <- Assembles the CNN from its config.
        │   ├── train_model.py        <- Fits an already-built model; records per-epoch loss history.
        │   ├── evaluate_model.py     <- Directional accuracy, RMSE and MAE on train/test.
        │   └── persist_model.py      <- Checkpoint save/load.
        │
        └── visualization  <- Matplotlib/seaborn plotting helpers for notebooks.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Rick and Morty](https://www.imdb.com/title/tt2861424/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
