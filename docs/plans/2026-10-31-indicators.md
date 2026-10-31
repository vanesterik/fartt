# get_indicators (B1, #107) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, in this session, on branch `b1-indicators`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The agent calls `get_indicators(ticker)` and learns whether a dip is overdone: moving averages, RSI, MACD, ATR and volume, each with its state, as of the last cached session.

**Architecture:** The server's cache update, read and staleness check move out of `get_candles` into one helper (Task 1). A new package, `fartt/technical/`, computes the indicators as a pure function over `list[Candle]` with TA-Lib (Task 2). `get_indicators` reads the whole cached history through the helper, calls the function, and adds `ticker`, `based_on`, `is_current` and `warning` (Task 3).

**Tech Stack:** Python 3.14, TA-Lib (`SMA`, `EMA`, `RSI`, `MACD`, `ATR`, typed stubs ship with it) on numpy arrays, pydantic models, MCP Python SDK v2, pytest with anyio, pyright strict.

**Spec:** `docs/specs/2026-10-31-technical-analysis-design.md`, sections "Shared staleness", "One computation module per tool", "Each tool reads the whole cached history", "`get_indicators(ticker)`", "Ex-dividend sessions", "Server instructions" and "The end-to-end test".

## Global Constraints

- Periods: SMA 20, 50 and 200, EMA 20, RSI 14, MACD (12, 26, 9), ATR 14. RSI and MACD periods and the RSI levels (30, 70) come from `TechnicalIndicatorsConfig`; `calculate_technical_indicators` doesn't change.
- Moving average `slope` compares with the value 5 sessions earlier; RSI `direction` with the value 3 sessions earlier.
- An indicator with too few sessions is `null`, and `note` names each one with the sessions it needs: SMA period + 5, EMA 20 + 5 = 25, RSI 15, MACD 34, ATR 15, volume 21 (the last plus the 20 before it).
- `down_day_ratio`: the mean volume on down days (close below the previous close) over the last 10 sessions divided by the mean on up days, leaving ex-dividend sessions out. `null` when there are no up days or no down days among them.
- `stretch = (close − SMA 20) / ATR`; `distance = (close − average) / average`; `fraction = ATR / close`.
- Floats in the output are rounded to 4 decimals (my call: the size budget, and digits past that mean nothing to the agent).
- The output for one ticker stays under 2,000 characters of JSON.
- Staleness follows `get_candles`: an unreachable provider serves the cache with the "N sessions behind" warning; nothing cached is a `ToolError`.

## Review Focus

1. **A flat series** (a suspended stock, a test fixture): average equals the close, the slope is neither rising nor falling. Expect `price_vs: above` (ties count as above), `slope: flat`, and an RSI that TA-Lib may leave undefined (no gains, no losses) reported as `null` with a note rather than NaN in the JSON. Tested in Task 2.
2. **All-zero or missing volume** (some Euronext listings report 0): the average is 0, so `ratio` and `down_day_ratio` are `null`, never a division error or infinity. Tested in Task 2.
3. **A short history** (a recent listing, 30 sessions): SMA 50/200, MACD and volume behave per their own thresholds; nothing computed from too few sessions slips out. Tested in Task 2 and Task 3.
4. **A NaN reaching pydantic** would serialise as `null` silently or fail validation; every value comes from the last element of a TA-Lib array, which is NaN until warm-up. The thresholds above guarantee warm-up; the short-history tests pin them.
5. **The refactor changing `get_candles`' messages**: the existing tests assert the warning text ("2 sessions behind", "hasn't published", "unreachable", "no closed sessions"). They stay as they are and must pass unchanged in Task 1.

---

### Task 1: Share the cache read and staleness check between tools

**Files:**
- Modify: `src/fartt/server/server.py`

**Interfaces:**
- Produces: in `server.py`,

```python
class Snapshot(NamedTuple):
    listing: Listing
    candles: list[Candle]  # oldest first
    update: UpdateResult | None  # None when the provider was unreachable
    is_current: bool
    warning: str | None  # the staleness sentence, if any

def _snapshot(tool: str, cache: StockCache, ticker: str, count: int) -> Snapshot
```

`_snapshot` does what `get_candles` does today up to and including the staleness warnings: `_update`, `cache.latest(ticker, count)` and `cache.listing`, the two `ToolError`s (nothing cached and unreachable; no closed sessions yet), and the "N sessions behind" / "hasn't published" sentence. It takes the normalised ticker from `_update` and returns it as `listing.ticker`. The gaps warning stays in `get_candles`, which appends it after `snapshot.warning`.

- [ ] **Step 1: Refactor.** `get_candles` calls `_snapshot("get_candles", cache, ticker, count)`; its output doesn't change. No new tests: this is a refactor, and the existing `get_candles` tests (Review Focus 5) are the proof.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_server.py -v`
Expected: all pass, unchanged

- [ ] **Step 3: Gate and commit.** `task check`, then:

```bash
git add src/fartt/server/server.py
git commit -m "refactor: share the cache read and staleness check between tools"
```

### Task 2: Indicators over plain candles

**Files:**
- Create: `src/fartt/technical/__init__.py`, `src/fartt/technical/indicators.py`
- Test: `tests/technical/__init__.py`, `tests/technical/test_indicators.py`

**Interfaces:**
- Consumes: `Candle` from `fartt.data`; `TechnicalIndicatorsConfig` from `fartt.features.technical_indicators_config`.
- Produces, exported from `fartt.technical`: pydantic models with a `Field(description=...)` on every field, since Task 3 returns them as structured output:

```python
Direction = Literal["rising", "falling", "flat"]

class MovingAverage(BaseModel):  # value, price_vs: Literal["above", "below"], distance, slope: Direction, days: int
class Rsi(BaseModel):            # value, zone: Literal["oversold", "neutral", "overbought"], days: int, direction: Direction | None
class Cross(BaseModel):          # kind: Literal["bullish", "bearish"], sessions_ago: int (0 = the last session)
class Macd(BaseModel):           # line, signal, histogram, histogram_direction: Direction | None (vs the previous session), last_cross: Cross | None
class Atr(BaseModel):            # value, fraction, stretch: float | None (null below 20 sessions)
class Volume(BaseModel):         # last, average_20d, ratio: float | None, down_day_ratio: float | None
class Indicators(BaseModel):     # close, sma_20, sma_50, sma_200, ema_20: MovingAverage | None, rsi: Rsi | None, macd: Macd | None, atr: Atr | None, volume: Volume | None, note: str | None

def calculate_indicators(
    candles: list[Candle],
    ex_dividend: frozenset[date] = frozenset(),
    config: TechnicalIndicatorsConfig = TechnicalIndicatorsConfig(),
) -> Indicators
```

`days` counts the last session (at least 1). `slope`/`direction` are `flat` when equal. RSI `direction` and MACD `histogram_direction` are `null` when the earlier value is still in warm-up; `last_cross` is `null` when the line never crossed the signal in the history. The ex-dividend dates go in here, not in B3, because `down_day_ratio` is computed here and B3 reads it (spec: "not recomputed").

- [ ] **Step 1: Write the failing tests** in `tests/technical/test_indicators.py`, on candles built with `tests.fakes.stock_candle` over consecutive dates:

```python
def test_sma_matches_a_hand_computed_mean():   # 30 closes 1..30 → sma_20.value == mean(11..30) == 20.5
def test_ema_matches_a_hand_computation():     # seed = SMA of the first 20, then α = 2/21
def test_rsi_matches_wilders_smoothing():      # hand Wilder RSI 14 on a fixed 20-close series, approx 1e-6
def test_atr_matches_wilders_smoothing():      # hand TR and Wilder ATR 14 on 20 candles
def test_macd_histogram_is_line_minus_signal(): # 60 rising closes → line > 0, histogram == line − signal
def test_a_steady_uptrend():                   # 250 rising closes → every MA price_vs "above", slope "rising",
                                               #   days == sessions since it became defined; rsi zone "overbought"
def test_a_dip_reads_oversold_and_stretched(): # 60 flat-ish then 10 falling → rsi "oversold", days ≥ 1,
                                               #   atr.stretch < 0, sma_20.price_vs "below"
def test_a_macd_cross_is_dated():              # rising then falling → last_cross.kind "bearish", sessions_ago matches
def test_volume_against_the_20_before_the_last(): # 21 sessions, last volume 3000, others 1000 → ratio 3.0
def test_down_day_ratio_weighs_selling():      # down days volume 2000, up days 1000 → 2.0
def test_an_ex_dividend_session_is_not_a_down_day(): # the only heavy down day is ex-dividend → ratio excludes it
def test_zero_volume_leaves_the_ratios_null(): # Review Focus 2
def test_a_flat_series(): # Review Focus 1: price_vs "above", slope "flat"; rsi None and "rsi" in note when TA-Lib gives NaN
def test_a_short_history_is_null_with_a_note(): # 30 sessions → sma_50, sma_200, macd None; "sma_200 needs 205" in note;
                                                #   sma_20, rsi, atr, volume present
def test_each_threshold_is_exact():            # parametrised: 204 → sma_200 None, 205 → present; 33/34 macd; 14/15 rsi, atr; 20/21 volume
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/technical -v`
Expected: FAIL (`fartt.technical` doesn't exist)

- [ ] **Step 3: Implement `calculate_indicators`.** Arrays as `np.array([...], dtype=float)`; each value is the last element of the TA-Lib output, and a NaN there becomes `None` with a note (Review Focus 4). Notes read "`<name>` needs N sessions, M are cached." joined with a space.

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/technical -v`
Expected: all pass

- [ ] **Step 5: Gate and commit.** `task check`, then:

```bash
git add src/fartt/technical tests/technical
git commit -m "feat: compute indicators and their states over plain candles"
```

### Task 3: The get_indicators tool

**Files:**
- Modify: `src/fartt/server/server.py`, `src/fartt/server/__init__.py` (export `IndicatorsResult`)
- Modify: `tests/test_server.py`, `tests/e2e/test_server_stdio.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `_snapshot` (Task 1); `calculate_indicators`, `Indicators` and its blocks (Task 2); `StockCache.events`.
- Produces: `IndicatorsResult(BaseModel)`: `ticker`, `currency`, `based_on` (ISO date of the last cached session), `close`, `sma_20`, `sma_50`, `sma_200`, `ema_20`, `rsi`, `macd`, `atr`, `volume`, `note`, `is_current`, `warning`.

- [ ] **Step 1: Write the failing tests** in `tests/test_server.py` (a `history` with 250 sessions of rising closes for the full case):

```python
async def test_get_indicators_returns_every_block(tmp_path: Path) -> None:
    # 250 XAMS sessions → ticker "ASML.AS", currency "EUR", based_on "2026-10-29",
    #   every block present, note None, is_current True, warning None
async def test_get_indicators_stays_within_its_size_budget(tmp_path: Path) -> None:
    # len(json.dumps(result.structured_content)) < 2000
async def test_get_indicators_on_a_short_history_says_what_is_missing(tmp_path: Path) -> None:
    # 30 sessions → sma_200 None, "sma_200 needs 205" in note
async def test_get_indicators_leaves_ex_dividend_sessions_out(tmp_path: Path) -> None:
    # a History with a Dividend on the one heavy down day → volume.down_day_ratio as if it weren't there
async def test_get_indicators_with_the_provider_down_serves_the_cache(tmp_path: Path) -> None:
    # as test_an_unreachable_provider_serves_the_cache_with_a_warning → is_current False, "2 sessions behind"
async def test_get_indicators_of_an_unknown_ticker_is_a_tool_error(tmp_path: Path) -> None:
async def test_the_instructions_say_indicators_are_evidence() -> None:
    # "evidence, not signals" in INSTRUCTIONS
```

`test_the_tools_are_get_candles_and_get_quote` becomes `test_the_tools_are_listed` with the three names.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL (`get_indicators` doesn't exist)

- [ ] **Step 3: Implement `get_indicators(ticker)`**: `_snapshot("get_indicators", cache, ticker, sys.maxsize)`, `cache.events(ticker)` for the dividend dates (a cache error maps through `_cache_error`), `calculate_indicators`, then `IndicatorsResult`. The docstring says it answers "is the dip overdone?" and that values are as of `based_on`. `INSTRUCTIONS` gains: "Indicators, trend and support/resistance zones are evidence, not signals: weigh them together."

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: all pass

- [ ] **Step 5: End to end.** `EXPECTED_TOOLS` gains `"get_indicators"`; after `get_quote`, call `get_indicators({"ticker": "ASML.AS"})` and assert no error, `ticker == "ASML.AS"`, `sma_200` not None (ASML has decades of history), `len(json.dumps(structured_content)) < 2000`.

Run: `uv run pytest tests/e2e -m network -v`
Expected: 2 passed

- [ ] **Step 6: Docs, gate and commit.** `CLAUDE.md` (current state: three tools; architecture: `fartt/technical/` and `get_indicators`) and `README.md` (the tool list). `task check`, then:

```bash
git add src/fartt/server tests/test_server.py tests/e2e CLAUDE.md README.md
git commit -m "feat: get_indicators with moving averages, RSI, MACD, ATR and volume"
```

## After the last task

- Final branch review (opus), fix pass, PR description with the commit list, mark ready.
