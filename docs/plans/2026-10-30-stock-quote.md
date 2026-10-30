# get_quote (A4, #100) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, in this session, on branch `a4-stock-quote`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The agent calls `get_quote(ticker)` and learns whether a stock is liquid enough to trade and where a limit order should sit: last price, bid, ask and spread, today's volume against its 20-session average, whether the market is open (and when it next opens), and how delayed the quote is.

**Architecture:** `sessions.py` gains `market_status(exchange, now)` from the exchange calendar. `build_server(cache)` adds `get_quote`, taking the provider and clock from the cache (`cache.provider`, `cache.clock`, made public), which updates the ticker's cache (for the volume average), asks the provider for the quote, and combines the two.

**Tech Stack:** Python 3.14, MCP Python SDK v2, `exchange_calendars` (`is_open_on_minute`, `next_open`), pydantic output models, pytest with anyio, pyright strict.

**Spec:** `docs/specs/2026-10-30-data-foundation-design.md`, section "The server", `get_quote(ticker)` and "Server instructions".

## Global Constraints

- Output fields, in this order: `ticker`, `currency`, `last`, `bid`, `ask`, `spread`, `volume`, `average_volume_20d`, `volume_ratio`, `market_open`, `next_open`, `quote_time`, `delay_minutes`, `note`.
- `spread = (ask - bid) / ((ask + bid) / 2)`. A missing bid or ask makes `bid`, `ask` and `spread` all `null` and the note says the spread is unknown. Never 0.
- `average_volume_20d` is the mean volume of the last 20 cached sessions *before* the quote's session, so today's volume is never compared with itself. `volume_ratio = volume / average_volume_20d`, `null` when either is missing.
- `market_open` and `next_open` (ISO 8601 UTC, `null` while open) come from the calendar at the server's clock, not from Yahoo.
- An unreachable provider is a `ToolError` (a cached quote would be stale by definition). Unknown ticker, unsupported exchange and cache errors map exactly as in `get_candles`.
- The server instructions gain one clause: dividends are kept separately (the spec asks for it; A3 left it out).

## Review Focus

1. **Fewer than 20 cached sessions** (a recent listing) → the average uses what's there and the note says over how many sessions. Tested in Task 2.
2. **Quote struck after the close, cache already holding today's session** → that session is excluded from the average. Tested in Task 2.
3. **A crossed quote** (ask below bid, which delayed Yahoo data shows) → treated as an unknown spread, never a negative one. Tested in Task 2.
4. **Weekend or holiday** → `market_open: false`, `next_open` the next session's open, a Monday or the day after the holiday. Tested in Task 1.
5. **The volume ratio early in the session** reads low; the field description says it isn't normalised for time of day (spec). Checked in Task 2 against the output schema.

---

### Task 1: Market status from the exchange calendar

**Files:**
- Modify: `src/fartt/data/sessions.py`, `src/fartt/data/__init__.py`, `typings/exchange_calendars/__init__.pyi` (`is_open_on_minute`, `next_open`)
- Test: `tests/data/test_sessions.py`

**Interfaces:**
- Produces: `MarketStatus(NamedTuple)`: `open: bool`, `next_open: datetime | None` (UTC; `None` while open). `market_status(exchange: str, now: datetime) -> MarketStatus`, exported from `fartt.data`.

- [ ] **Step 1: Write the failing tests.** NYSE runs 13:30–20:00 UTC until US daylight saving time ends on 2026-11-01, and 14:30–21:00 UTC after:

```python
def test_market_status_during_the_session():  # Thu 2026-10-29 18:00 UTC → MarketStatus(True, None)
def test_market_status_after_the_close():     # Thu 20:30 UTC → (False, Fri 2026-10-30 13:30 UTC)
def test_market_status_on_a_weekend():        # Sat 2026-10-31 12:00 UTC → next_open Mon 2026-11-02 14:30 UTC
def test_market_status_on_a_holiday():        # Thanksgiving 2026-11-26 15:00 UTC → next_open 2026-11-27 14:30 UTC
def test_market_status_in_amsterdam():        # XAMS Thu 2026-10-29 10:00 UTC → open
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/data/test_sessions.py -v`
Expected: FAIL (`market_status` doesn't exist)

- [ ] **Step 3: Implement `market_status`** with `calendar.is_open_on_minute(minute)` and `calendar.next_open(minute)` on `now` floored to the minute, converted to a `datetime` in UTC. Add both methods to the stub.

- [ ] **Step 4: Run them and see them pass**, then `task check`, then commit:

```bash
git add src/fartt/data typings/exchange_calendars tests/data/test_sessions.py
git commit -m "feat: tell whether an exchange is open and when it next opens"
```

### Task 2: The get_quote tool

**Files:**
- Modify: `src/fartt/server/server.py`, `src/fartt/server/__init__.py` (export `QuoteResult`)
- Modify: `tests/fakes.py` (`FakeProvider` gains `quotes: dict[str, Quote] | None = None`; `quote()` checks reachability and the ticker as `listing` does, and returns the stored quote)
- Modify: `tests/test_server.py`, `tests/e2e/test_server_stdio.py`
- Modify: `CLAUDE.md`, `README.md` (the tool list)

**Interfaces:**
- Consumes: `market_status`, `MarketStatus` (Task 1); `StockCache.update`, `latest`, `listing`; `DataProvider.quote`.
- Produces: `build_server(cache: StockCache) -> MCPServer` (unchanged signature); `StockCache.provider` and `StockCache.clock`; `exchange_date(exchange, moment) -> date` in `sessions.py`; `QuoteResult`.

- [ ] **Step 1: Write the failing tests** in `tests/test_server.py` (the `call` helper takes the tool name):

```python
async def test_get_quote_during_the_session() -> None:
    # 25 cached sessions with volumes 1000..; quote last 102.0, bid 101.9, ask 102.1, volume 500
    # → spread == pytest.approx(0.2 / 102.0), average over the 20 before today,
    #   volume_ratio == 500 / average, market_open True, next_open None, note None,
    #   quote_time ISO UTC, delay_minutes 15
async def test_get_quote_outside_the_session() -> None:      # after the close → market_open False, next_open the next open
async def test_today_is_not_in_its_own_average() -> None:    # cache holds today's session with volume 10_000 → excluded
async def test_a_short_history_averages_what_is_cached() -> None:  # 5 sessions → average of 5, "5 sessions" in note
async def test_a_missing_bid_or_ask_makes_the_spread_unknown() -> None:  # bid None → bid/ask/spread None, "spread is unknown" in note
async def test_a_crossed_quote_makes_the_spread_unknown() -> None:       # ask < bid → same as missing, note says crossed
async def test_get_quote_with_the_provider_down_is_a_tool_error() -> None:  # "unreachable"
async def test_get_quote_of_an_unknown_ticker_is_a_tool_error() -> None:   # "NOSUCH", "ASML.AS"
async def test_the_volume_ratio_says_it_is_not_normalised() -> None:  # "time of day" in the output schema's volume_ratio description
```

`test_the_only_tool_is_get_candles` becomes `test_the_tools_are_get_candles_and_get_quote`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL (`get_quote` doesn't exist)

- [ ] **Step 3: Implement `get_quote`.** The cache update's error mapping is shared with `get_candles` (one helper turning `UnknownTicker`, `UnsupportedExchange`, `OSError`/`ValueError`/`KeyError` into `ToolError`s); `get_quote` adds `ProviderUnavailable` → `ToolError` ("unreachable, try again later"). The quote's session is the exchange-local date of `quote.time`; the average reads `cache.latest(ticker, 21)` and keeps the last 20 dated before it. Notes are sentences joined with a space, `null` when there are none. Add "dividends kept separately" to `INSTRUCTIONS`.

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: all pass

- [ ] **Step 5: End to end.** `EXPECTED_TOOLS = ["get_candles", "get_quote"]`; after `get_candles`, call `get_quote({"ticker": "ASML.AS"})` and assert no error, `ticker == "ASML.AS"`, `last > 0`, `isinstance(market_open, bool)`.

Run: `uv run pytest tests/e2e -m network -v`
Expected: 2 passed

- [ ] **Step 6: Docs, gate and commit.** `CLAUDE.md` (current state, server architecture) and `README.md` (the tool list) describe `get_quote`. Run `task check`, then:

```bash
git add src/fartt/server tests/fakes.py tests/test_server.py tests/e2e CLAUDE.md README.md
git commit -m "feat: get_quote with spread, volume against its average and market hours"
```

## After the last task

- Final branch review, fix pass, update #100 (its end-to-end scenario still says `fartt serve`) and PR description, mark ready.
