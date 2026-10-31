# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fartt (Financial Analysis Real Time Trading) is an MCP server that gives an agent the tools to analyse stocks on Euronext and US exchanges, with Claude Code as the analyst and the operator's interface. It's decision support for personal use: the agent combines many signals (market data, technical indicators, support/resistance, fundamentals, market context, charts) into a view, and a journal scores every call afterwards against a world index ETF. The server owns data, risk limits and state; the agent can read the limits but never change them. It's a solo research project, not production infra.

The plan is the PRD, `docs/product/stock-analysis-prd.md` (problem, users including the agent, the eight-step analysis procedure, tool surface, the three stages from paper trading to server-placed orders and their gates, the risk config, risks and open questions). It's delivered in epics on the board, in this order: **A** Data Foundation (#88), **B** Technical Analysis (#89), **C** Journal & Evaluation (#90), **D** Analysis Pipeline (#91; A–D are the walking skeleton, a full analysis on one ticker), then **E** Visualization (#92), **F** Market Context (#93), **G** Fundamentals & Screening (#94), **H** Trade Plan & Risk (#95), **I** Validation (#96), **J** Live Orders (#101). Stories are written just in time, one epic at a time. Epic designs live in `docs/specs/`, implementation plans in `docs/plans/`.

The project follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) layout: `docs/` for the PRD, designs and plans, `src/fartt` for library code, `tests/` for unit tests.

## Current state of the codebase (important)

**What exists and works** (epic A, stories A1–A4; epic B, story B1):

- **Stock data provider.** `fartt/data/` holds the `DataProvider` Protocol (`listing`, `history(ticker, start)`, `quote`) and `YFinanceProvider`, the only module that imports yfinance. Prices are split-adjusted, not dividend-adjusted; dividends and splits come out as separate records.
- **Stock cache.** `fartt/stock_cache.py` keeps each ticker's daily candles, dividends, splits and listing under `assets/stocks/<TICKER>/` and brings them up to date on demand.
- **The MCP server.** Claude Code starts it from `.mcp.json` (`uv run fartt`). It has three tools, `get_candles(ticker, count)`, `get_quote(ticker)` and `get_indicators(ticker)`.
- **Indicators.** `fartt/technical/indicators.py` computes moving averages, RSI, MACD, ATR and volume with their states over plain candles.
- **Helpers later epics build on**, not called by the server yet: the DataFrame indicators behind the chart, a candlestick chart (epic E) and a long-only backtest (`calculate_trade_returns`, epic I).

**What doesn't exist yet:** the rest of epics B–J: trend and support/resistance, the journal, the analysis pipeline, charts, market context, fundamentals and screening, trade plans and risk limits, validation and live orders.

`uv run pytest` collects and passes cleanly; don't assume the suite is broken.

## Commands

The everyday commands are tasks in `Taskfile.yml`; `task --list` names them. `task setup` installs the project from `uv.lock` and wires up the git hooks, and `task check` (lint + typecheck + test) is the same command the pre-push hook runs, so the two cannot drift apart. `task` comes from the global `mise` setup (`mise use -g aqua:go-task/task`); it cannot install itself, so it is the one prerequisite `task setup` does not cover.

```bash
task setup                       # uv sync + install the lefthook hooks
task check                       # lint + typecheck + test, as the pre-push gate runs them
task lint                        # ruff format --check + ruff check, without writing
task format                      # ruff format + ruff check --fix
task typecheck                   # pyright (strict mode, src/ only — tests/ excluded)
task test                        # pytest, offline
task test:network                # live Yahoo tests + the end-to-end check of the server over stdio (deselected by default)
task pre-commit                  # the pre-commit hook's checks over staged files
task outdated                    # direct dependencies with a newer release
```

Tool versions are upgraded on purpose, never because a run nags: ruff's rule set is pinned in `pyproject.toml`, and pyright's new-release warning is silenced. Instead, Dependabot (`.github/dependabot.yml`) opens weekly pull requests for newer releases: minor and patch updates grouped into one for dev tools and one for runtime dependencies, each major update on its own. There's no CI, so check a Dependabot pull request out and run `task check` (or push to it) before merging. `task outdated` is the on-demand version.

Everything below remains the direct route, and is what the Taskfile calls. Dependency management is via `uv` (see `uv.lock`), run from the repository root. The Python version is pinned in `.python-version` (3.14) and matched by `requires-python` in `pyproject.toml`; Dependabot doesn't track it, so moving to a new Python release is a deliberate change to both.

```bash
uv sync                          # install dependencies
uv run fartt                     # run the MCP server over stdio (Claude Code does this from .mcp.json)
uv run pytest                    # run test suite
uv run pytest tests/test_server.py::test_count_beyond_the_history_returns_what_is_cached  # single test
uv run pytest -m network         # run the live tests (deselected by default)
uv run ruff format .             # format
uv run ruff check . --fix        # lint
uv run pyright                   # type check (strict mode, src/ only — tests/ excluded)
```

There is no CLI: the agent calls the server's tools, and a ticker's cache fills on its first call. The server reads `FARTT_ASSETS_DIR` for the cache's folder from the environment or else the repository's `.env` (copy `.env.example`), defaulting to `assets/` in the repository wherever it was started from; a relative path is relative to the repository. It logs to stderr only, since stdout carries the protocol.

`tests/e2e/test_server_stdio.py` checks the server end to end: it starts the server from `.mcp.json`'s command with `FARTT_ASSETS_DIR` set to a temporary folder, calls the tools, and asserts stdout carries only JSON-RPC, including anything written at shutdown. Each new tool joins it: add its name to `EXPECTED_TOOLS` and its call to `test_tools_over_stdio`.

Hooks are managed by `lefthook` (`.lefthook.yml`) and stand in for CI, which this project doesn't have. On every commit, staged Python files get `ruff format` + `ruff check --fix`, then `pyright` and `pytest` run. On every push, `task check` runs the full suite on the whole repository (`ruff format --check .`, `ruff check .`, `pyright`, `pytest`), because pre-commit only sees staged files and a commit touching only `pyproject.toml`, `uv.lock` or `.mcp.json` would otherwise skip tests and type checks. `git push --no-verify` skips the gate for one push; use it deliberately, never by default. Tests marked `network` call Yahoo and are deselected by default; run them with `task test:network`.

## Development workflow and pull requests

The general workflow and pull request flow come from the global `~/.claude/CLAUDE.md` (the `vanesterik/claude-config` repository). What's specific to this repository:

- There's no CI. Lefthook is the gate: pre-commit checks staged files and runs the tests, and pre-push runs the full suite on the whole repository (see Commands).
- The repository allows rebase merging only and has `deleteBranchOnMerge` set, so only the local branch needs deleting after a merge.

## User stories / issue tracking

User stories for this project are tracked as GitHub issues in `vanesterik/fartt` on the project board at https://github.com/users/vanesterik/projects/3:

```bash
gh issue create --repo vanesterik/fartt --title "..." --body "..."
gh project item-add 3 --owner vanesterik --url <issue-url>
```

## Architecture

- **`fartt/data/`** — the provider layer. `DataProvider` takes tickers in Yahoo's format (`AAPL`, `ASML.AS`); its `NamedTuple` types are `Candle` (with a session `date`), `Dividend`, `Split`, `History`, `Listing` and `Quote`, and its errors `UnknownTicker`, `ProviderUnavailable` and `UnsupportedExchange`. `sessions.py` normalises tickers (`normalise_ticker`: stripped, upper-cased, letters, digits, `.` and `-` only, so no ticker escapes a cache folder), maps the suffix to an `exchange_calendars` code (`calendar_for`; no suffix is XNYS, then `.AS` XAMS, `.PA` XPAR, `.BR` XBRU, `.LS` XLIS, `.MI` XMIL, `.IR` XDUB, `.OL` XOSL; anything else is `UnsupportedExchange`), and answers `newest_closed_session` (a 30-minute `CLOSE_MARGIN` after the close), `missing_sessions`, `exchange_date` and `market_status` (open now, or when the next session opens). `YFinanceProvider(ticker_factory, clock)` returns closed sessions only, maps yfinance's missing-ticker errors and Yahoo's HTTP 404 to `UnknownTicker` and everything else to `ProviderUnavailable`, and turns a bid/ask of 0 into `None`. It sets `yfinance.config.debug.hide_exceptions = False` (the per-call `raise_errors` is deprecated). yfinance, exchange_calendars and mplfinance have minimal stubs under `typings/`.
- **`fartt/stock_cache.py`** — `StockCache(provider, assets_dir, clock)`: each ticker's split-adjusted daily candles (`candles.csv`), dividends and splits (`events.csv`) and listing (`listing.json`) under `assets_dir/stocks/<TICKER>/`. `update(ticker) -> UpdateResult(new_candles, rebuilt, fetched)`: a first update fetches the full history; later ones re-fetch from the last cached session and append what's newer; a split after it, or any change to the last cached candle's prices (volume aside), rebuilds the whole history (files swapped in with `os.replace`, candles last); a cache already holding the newest closed session doesn't call the provider. A torn last line is healed first. Reads (`latest`, `events`, `listing`) take a shared `fcntl` lock on the directory's `.lock`, updates an exclusive one.
- **`fartt/server/`** — `build_server(cache)` returns the MCP SDK v2 `MCPServer` (the cache also supplies the provider for quotes and the clock for market hours), and `main()` (the `fartt` console script) builds it on `StockCache(YFinanceProvider(), assets_dir)` and runs it over stdio. `get_candles(ticker, count=60)` (`count` 1–200: 200 rows stay well under Claude Code's MCP output limit of about 25k tokens) updates the ticker's cache and returns structured output: `ticker`, `name`, `currency`, `exchange`, `columns`, `rows` as `[date, open, high, low, close, volume]` arrays oldest first (rounded to 4 decimals, as `get_indicators` is), `new_candles`, `rebuilt`, `gaps` (sessions missing between the rows), `is_current` and `warning`. An unreachable provider degrades to the cached rows with `is_current: false` and a warning saying how many sessions behind they are, or a `ToolError` when nothing is cached. An unknown ticker, an exchange outside the universe or a cache folder it can't write are `ToolError`s saying what to do; the SDK would otherwise report any other exception as a bare "Error executing tool". `get_quote(ticker)` updates the cache too, then returns `ticker`, `currency`, `last`, `bid`, `ask`, `spread` (a fraction of the midpoint), `volume`, `average_volume_20d` (the 20 cached sessions before the quote's session, so today never averages with itself), `volume_ratio` (not normalised for time of day), `market_open` and `next_open` (from the exchange calendar), `quote_time`, `delay_minutes` and `note`. A missing or crossed bid/ask makes `bid`, `ask` and `spread` all null, never a zero or negative spread. An unreachable provider is a `ToolError`: a cached quote would be stale by definition. `get_indicators(ticker)` reads the whole cached history and returns `ticker`, `currency`, `based_on` (the session its values are as of), `close`, `sma_20`, `sma_50`, `sma_200`, `ema_20`, `rsi`, `macd`, `atr`, `volume`, `note` (which indicators are null for too short a history), `is_current` and `warning`. The cache update, read and staleness check all candle-based tools share are in `_snapshot`.
- **`fartt/technical/`** — the technical analysis as pure functions over `list[Candle]` (oldest first), no I/O: `indicators.py::calculate_indicators(candles, ex_dividend)` returns pydantic blocks the server returns as they are. Each value comes with its state (above/below, rising/falling/flat, the RSI zone and how many sessions it's held, the last MACD cross, ATR stretch from the SMA 20, volume ratios with ex-dividend sessions left out of the down-day ratio). An indicator needing more sessions than are cached is `None` with a note; RSI and MACD periods come from `TechnicalIndicatorsConfig`. Floats are rounded to 4 decimals.
- **`fartt/features/`** — calculations over Polars DataFrames and return series: `calculate_technical_indicators.py` computes Bollinger Bands / EMA / MACD / RSI via `talib` for the chart, configured through `technical_indicators_config.py`; `calculate_candle_returns.py` computes the signed percent change per candle; `calculate_trade_returns.py` is a functional long-only backtest over a candle-return series (cost, default 0.05% per leg; slippage, threshold, maximum holding period and stop loss; an optional `predicted_candle_returns` argument backtests a signal known one tick ahead against realised prices).
- **`fartt/visualization/`** — `candlestick_chart.py` (Bollinger/EMA/MACD/RSI overlays via `mplfinance`) and `plot_styles.py::apply_plot_styles` (shared tradingview-style `rcParams`).
- **`fartt/constants.py`** — centralized string labels (column names, indicator labels) and a small colour palette, shared by the features and visualization code to avoid stringly-typed column name drift.
