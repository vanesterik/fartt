# get_candles per ticker, served without a CLI (A3, #99) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, in this session, on branch `a3-stock-server`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The MCP server answers `get_candles(ticker, count)` from the stock cache for any Euronext or US ticker, Claude Code starts it with `uv run fartt` and no CLI, and every piece of the BTC forecasting direction is gone.

**Architecture:** `fartt/server/server.py` gets `build_server(cache: StockCache)` with one tool, `get_candles`, and a `main()` that the `fartt` console script points at: it reads `FARTT_ASSETS_DIR` (default `assets/` under the repository root), logs to stderr only, and runs the server over stdio. `fartt/cli.py` is deleted. Then everything the server no longer imports goes: the forecast and model packages, the exchange layer, the old candle cache, their tests, notebooks and dependencies.

**Tech Stack:** Python 3.14, MCP Python SDK v2 (`MCPServer`, `ToolError`, structured output via pydantic models, as the current server uses), `fartt.data`, `fartt.stock_cache`, loguru, pytest with anyio, pyright strict.

**Spec:** `docs/specs/2026-10-30-data-foundation-design.md`, sections "The server" (`get_candles`, "Launching it", server instructions, the end-to-end test) and the removal table's A3 row.

## Global Constraints

- One tool in this story: `get_candles(ticker: str, count: int = 60)`, `count` from 1 to 200 (`MAX_CANDLES = 200`). `get_quote` is A4.
- Output fields, in this order: `ticker`, `name`, `currency`, `exchange`, `columns` (`["date", "open", "high", "low", "close", "volume"]`), `rows` (`[date ISO, open, high, low, close, volume]`, oldest first), `new_candles`, `rebuilt` (`str | None`), `gaps` (ISO dates), `is_current`, `warning` (`str | None`).
- `is_current` is true when the last row is the newest closed session. `gaps` are `missing_sessions(listing.exchange, row dates)`; when there are any, `warning` mentions them.
- `ProviderUnavailable` with candles cached → cached rows, `is_current: false`, and a warning saying how many sessions behind they are. With nothing cached → `ToolError` saying the provider is unreachable and to try again later.
- `UnknownTicker` → `ToolError` naming the ticker and the format (`AAPL` for the US, `ASML.AS` for Euronext Amsterdam). `UnsupportedExchange` → `ToolError` with its message (it lists the suffixes). `OSError` writing the cache → `ToolError` saying the operator must act (the cache folder can't be written).
- `pyproject.toml`: `[project.scripts] fartt = "fartt.server:main"`. `.mcp.json`: `"command": "uv", "args": ["run", "fartt"]`.
- `main()` logs to stderr only; nothing but JSON-RPC reaches stdout.
- `calculate_trade_returns`, `calculate_candle_returns`, `calculate_technical_indicators` (with `technical_indicators_config.py`), `candlestick_chart.py`, `plot_styles.py` and `constants.py` stay (spec: epics B, E and I build on them).

## Review Focus

1. **A ticker the cache has never seen, asked for while the provider is down** → `ToolError` about the provider, not an empty result. Tested in Task 1.
2. **`count` larger than the cached history** (a ticker listed last week) → all cached rows, no error. Tested in Task 1.
3. **The provider down for a ticker cached two sessions ago** → warning says "2 sessions behind". Tested in Task 1.
4. **A ticker like `..` or `asml.as `** reaching the tool → rejected or normalised exactly as the cache does. Tested in Task 1.
5. **`FARTT_ASSETS_DIR` unset** → the cache lands in the repository's `assets/`, wherever the process was started from. Tested in Task 1.

---

### Task 1: The server on the stock cache, launched without a CLI

**Files:**
- Modify: `src/fartt/server/server.py` (rewrite), `src/fartt/server/__init__.py` (export `build_server`, `main`)
- Modify: `pyproject.toml` (`[project.scripts]`), `.mcp.json`
- Delete: `src/fartt/cli.py`, `tests/test_cli.py`
- Modify: `tests/test_server.py` (rewrite), `tests/e2e/test_server_stdio.py` (rewrite)

This task changes about ten files; the commit message says so.

**Interfaces:**
- Consumes: `StockCache` (`update`, `latest`, `listing`, `newest_closed_session`, `UpdateResult`) from `fartt.stock_cache`; `YFinanceProvider`, `UnknownTicker`, `UnsupportedExchange`, `ProviderUnavailable`, `missing_sessions` from `fartt.data`; `FakeProvider`, `stock_candle` from `tests/fakes.py`.
- Produces: `build_server(cache: StockCache) -> MCPServer`; `main() -> None`; `DEFAULT_ASSETS_DIR: Path` (`Path(__file__).resolve().parents[3] / "assets"`); `ASSETS_DIR_ENV = "FARTT_ASSETS_DIR"`; `CandlesResult` (the pydantic output model). A4 adds `get_quote` to the same server.

**Server instructions** (exact copy):

> Fartt's stock analysis server. Tools take one ticker per call, in Yahoo's format: AAPL for a US listing, ASML.AS for Euronext Amsterdam (.PA Paris, .BR Brussels, .LS Lisbon, .MI Milan, .IR Dublin, .OL Oslo). A ticker the server hasn't seen is fetched on the spot. Prices are daily and split-adjusted, not dividend-adjusted: they are prices that traded. Returns, costs and rates are fractions: 0.004 means 0.4%. If is_current is false the data is stale: read the warning before relying on it.

- [ ] **Step 1: Rewrite the in-process tests** in `tests/test_server.py`, with a `FakeProvider` holding `ASML.AS` (XAMS) and `AAPL` (XNYS) and a `StockCache` on `tmp_path` whose clock is the provider's `now`:

```python
async def test_get_candles_returns_the_newest_closed_sessions() -> None:
    content = (await call("get_candles", {"ticker": "asml.as", "count": 2})).structured_content
    assert content["ticker"] == "ASML.AS"
    assert (content["name"], content["currency"], content["exchange"]) == ("ASML Holding N.V.", "EUR", "XAMS")
    assert content["columns"] == ["date", "open", "high", "low", "close", "volume"]
    assert [row[0] for row in content["rows"]] == ["2026-10-28", "2026-10-29"]
    assert content["is_current"] is True and content["warning"] is None and content["gaps"] == []

async def test_count_defaults_to_60_and_is_capped_at_200() -> None: ...       # 61 cached sessions → 60 rows; count=201 → validation error
async def test_count_beyond_the_history_returns_what_is_cached() -> None: ...  # 3 sessions, count=50 → 3 rows
async def test_first_call_reports_the_fill_and_later_calls_do_not() -> None:
    # first: new_candles == len(history); second: new_candles == 0
async def test_a_rebuild_is_reported() -> None: ...                            # amended last candle → rebuilt == "last candle amended"
async def test_gaps_are_reported_in_a_warning() -> None:
    # history missing 2026-10-27 → gaps == ["2026-10-27"], "2026-10-27" in warning
async def test_an_unreachable_provider_serves_the_cache_with_a_warning() -> None:
    # cached through 10-27, provider down, now after 10-29's close
    assert content["is_current"] is False and "2 sessions behind" in content["warning"]
async def test_an_unreachable_provider_and_no_cache_is_a_tool_error() -> None: ...  # is_error, "unreachable" in text
async def test_an_unknown_ticker_is_a_tool_error_with_the_format() -> None: ...     # "NOSUCH", "ASML.AS" in text
async def test_an_unsupported_exchange_is_a_tool_error() -> None: ...              # "VOD.L" → ".AS" in text
async def test_a_ticker_that_names_no_exchange_is_a_tool_error() -> None: ...      # ".."
async def test_an_unwritable_cache_is_a_tool_error_for_the_operator() -> None: ... # cache.update raises OSError → "operator" in text
async def test_the_only_tool_is_get_candles() -> None: ...

def test_main_uses_the_repository_assets_by_default(monkeypatch) -> None:
    # FARTT_ASSETS_DIR unset, cwd elsewhere; build_server and YFinanceProvider monkeypatched:
    # main() builds the cache on DEFAULT_ASSETS_DIR == ROOT / "assets"
def test_main_reads_the_assets_dir_from_the_environment(monkeypatch, tmp_path) -> None: ...
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL (the server still takes a `CandleCache`, a market and an interval)

- [ ] **Step 3: Rewrite `server.py`**, delete `cli.py` and `tests/test_cli.py`, point `[project.scripts]` at `fartt.server:main`, and change `.mcp.json`. In `main()`, `logger.remove()` then `logger.add(sys.stderr, level="INFO")`. Sessions behind = the sessions after the last cached row up to and including the newest closed one (`missing_sessions(exchange, [last, newest])` plus one).

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: all pass

- [ ] **Step 5: Rewrite the end-to-end test** (`tests/e2e/test_server_stdio.py`): the command and args come from `.mcp.json` unchanged; `FARTT_ASSETS_DIR=<tmp_path>` goes in the environment (`StdioServerParameters(env=...)` merged with `os.environ`, and the same for the raw `subprocess` stdout check). `EXPECTED_TOOLS = ["get_candles"]`; the cycle calls `get_candles({"ticker": "ASML.AS", "count": 5})` and asserts 5 rows, `ticker == "ASML.AS"`, `exchange == "XAMS"`. Keep `test_stdout_carries_only_the_protocol` as it is apart from the environment. Update the module docstring: each tool story adds its tool to `EXPECTED_TOOLS` and its call to the test.

Run: `uv run pytest tests/e2e -m network -v`
Expected: 2 passed

- [ ] **Step 6: Run the whole gate and commit**

Run: `task check` (Expected: all pass; `tests/test_candle_cache.py` etc. still pass, since the old modules still exist), then:

```bash
git add -A src/fartt/server src/fartt/cli.py tests/test_cli.py tests/test_server.py tests/e2e pyproject.toml uv.lock .mcp.json
git commit -m "feat: serve get_candles per ticker from the stock cache, without a CLI"
```

### Task 2: Remove the forecasting direction

**Files:**
- Delete: `src/fartt/forecast/`, `src/fartt/model/`, `src/fartt/exchange/`, `src/fartt/candle_cache.py`, `src/fartt/utils.py`, `src/fartt/features/align_predicted_signal.py`, `src/fartt/features/parse_timestamp_to_datetime.py`, `src/fartt/features/sort_and_deduplicate.py`, `src/fartt/visualization/{candle_returns,diverging_bar_chart,evaluation_line_chart,learning_curve_chart,missing_value_heatmap,predicted_vs_actual_scatter,trade_returns}.py`, the three notebooks (keep `notebooks/.gitkeep`), `typings/ccxt/`
- Delete: their tests: `tests/forecast/`, `tests/model/`, `tests/exchange/`, `tests/utils/`, `tests/test_candle_cache.py`, and the tests of the deleted feature and visualization modules; `FakeExchange` and `candle` in `tests/fakes.py`
- Modify: `pyproject.toml` and `uv.lock`: remove `ccxt`, `torch` (with its `[tool.uv.sources]` and `[[tool.uv.index]]` entries), `jupyter`, `scikit-learn`, `seaborn`, `tabulate`, `tqdm`, `pyarrow`, `pillow` (each only if `grep` finds no remaining import); add `pandas` as a direct dependency (`yfinance_provider.py` imports it)
- Modify: `Taskfile.yml` (the `torch` comment and anything else about it), `.vscode/settings.json` (drop "bitvavo"), `.lefthook.yml` only if a hook names a deleted path
- Modify: `src/fartt/features/calculate_trade_returns.py` and its test: `cost_pct` defaults to `0.0005`
- Local only (not in git): delete `assets/BTC-EUR-*` and `logs/`

This task deletes far more than ten files; the commit message says so.

**Interfaces:**
- Consumes: Task 1's server, which no longer imports any of the deleted modules.
- Produces: nothing new.

- [ ] **Step 1: Change the fee default, test first.** In the trade-returns test, the case that relies on the default cost now expects `0.0005` (0.05% per leg: roughly IBKR's tiered commission on Euronext; PRD open question 3). Run `uv run pytest tests/features/test_calculate_trade_returns.py -v`; Expected: that case FAILS. Change the default and its docstring ("Defaults to 0.05%, roughly Interactive Brokers' tiered commission on Euronext"); run again; Expected: PASS.

- [ ] **Step 2: Delete the modules, tests and notebooks listed above.**

- [ ] **Step 3: Prove nothing refers to them.**

Run: `grep -rnE "fartt\.(forecast|model|exchange|candle_cache|utils)|ccxt|torch|bitvavo|Bitvavo|BTC" src tests pyproject.toml Taskfile.yml .vscode .mcp.json .lefthook.yml`
Expected: no output

- [ ] **Step 4: Remove the dependencies** (`uv remove ...`, `uv add pandas`), checking each with `grep -rn "import <name>\|from <name>" src tests` first.

- [ ] **Step 5: Run the whole gate**

Run: `task check`
Expected: all pass

- [ ] **Step 6: Delete the local BTC files and logs, and commit**

```bash
rm -f assets/BTC-EUR-*; rm -rf logs
git add -A src tests notebooks typings pyproject.toml uv.lock Taskfile.yml .vscode
git commit -m "chore: remove the forecasting direction's code, tests and dependencies"
```

### Task 3: Describe the codebase as it now is

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Rewrite `CLAUDE.md`'s "Current state", "Commands" and "Architecture"** for the stock codebase: `fartt/data/`, `stock_cache.py`, `server/` (`get_candles`, `main`, `FARTT_ASSETS_DIR`), `features/` and `visualization/` (what's left and which epic uses it), `constants.py`. No `fartt serve`/`download`, no forecast, model or exchange sections, and no "pivoted from" paragraph: the codebase no longer carries the earlier direction.
- [ ] **Step 2: Rewrite `README.md`'s "Usage" and "Project Structure"**: open Claude Code in the repository and ask; `.mcp.json` starts the server with `uv run fartt`; the remaining modules.
- [ ] **Step 3: Check for leftovers**

Run: `grep -rnE "BTC|ccxt|Bitvavo|bitvavo|forecast|fartt serve|fartt download|candle_cache|notebooks/2" CLAUDE.md README.md`
Expected: no output, except "forecast" where it means epic I or a future signal

- [ ] **Step 4: Run `task check` and commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: describe the stock analysis codebase"
```

## After the last task

- Run the final branch review, then scrub the epic A design's removal table and the "BTC direction" goal (the last references, now that A3 has used them) in a `docs:` commit on this branch.
- Update #104's description with the commits and mark it ready.
