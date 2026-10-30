# DataProvider with yfinance (A1, #97) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, in this session, on branch `a1-data-provider`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `fartt/data/`: the `DataProvider` Protocol, its value types and errors, exchange sessions, and a yfinance implementation that returns split-adjusted daily candles, dividends, splits, listings and quotes for Euronext and US tickers.

**Architecture:** A Protocol, `NamedTuple` value types, error types, and one implementation, `YFinanceProvider`, which is the only module that imports yfinance and takes an injectable ticker factory and clock for tests. `sessions.py` maps a ticker's suffix to an `exchange_calendars` calendar and decides which sessions have closed. Nothing outside `fartt/data/` changes in this story; the cache (A2) and the server (A3) start using it later.

**Tech Stack:** Python 3.14, yfinance (`auto_adjust=False, actions=True, raise_errors=True`), exchange_calendars, pandas (only inside the yfinance module), pytest, pyright strict.

**Spec:** `docs/specs/2026-10-30-data-foundation-design.md`, sections "`fartt/data/`: the provider layer" and "Sessions".

## Global Constraints

- yfinance is imported by `src/fartt/data/yfinance_provider.py` only; no yfinance or pandas type appears in any signature outside it.
- Prices are split-adjusted, not dividend-adjusted: `Ticker.history(interval="1d", auto_adjust=False, actions=True)`; `Adj Close` is dropped.
- `history` returns closed sessions only. A session counts as closed 30 minutes after its scheduled close in the exchange calendar, early closes included (`CLOSE_MARGIN = timedelta(minutes=30)`).
- Suffix → calendar: none → `XNYS`, `.AS` → `XAMS`, `.PA` → `XPAR`, `.BR` → `XBRU`, `.LS` → `XLIS`, `.MI` → `XMIL`, `.IR` → `XDUB`, `.OL` → `XOSL`. Any other suffix raises `UnsupportedExchange`.
- A bid or ask of `0` or missing becomes `None`, never `0`.
- No yfinance exception leaves the provider: unknown ticker → `UnknownTicker`; anything else from yfinance or the network → `ProviderUnavailable`.
- Before using a yfinance or exchange_calendars call, check its current signature with context7 (global CLAUDE.md). pyright runs strict on `src/`; if a library lacks usable types, add a minimal stub under `typings/`, declaring only what's called.
- Live tests are marked `network` and stay deselected by default.

## Review Focus

1. **Yahoo returns today's forming candle** before the session (plus margin) has closed → it's dropped. Tested in Task 2.
2. **A bid or ask of 0** → `None`, so a spread is never computed from it. Tested in Task 3.
3. **A lower-case ticker** (`asml.as`) → same calendar as `ASML.AS`. Tested in Task 1.
4. **A reverse split** (`Stock Splits` = 0.1 for 1-for-10) → kept as `ratio=0.1`, not dropped as "less than one". Tested in Task 2.
5. **A row with NaN prices, or a row dated before `start`** → skipped. Tested in Task 2.

---

### Task 1: Value types, errors, the Protocol and sessions

**Files:**
- Create: `src/fartt/data/__init__.py` (re-exports everything below)
- Create: `src/fartt/data/types.py`
- Create: `src/fartt/data/provider.py`
- Create: `src/fartt/data/sessions.py`
- Modify: `pyproject.toml`, `uv.lock` (`uv add exchange_calendars`)
- Test: `tests/data/__init__.py`, `tests/data/test_sessions.py`

**Interfaces:**
- Produces, in `types.py`: `Candle(date: date, open, high, low, close, volume: float)`, `Dividend(date: date, amount: float)`, `Split(date: date, ratio: float)`, `History(candles: list[Candle], dividends: list[Dividend], splits: list[Split])`, `Listing(ticker: str, name: str, currency: str, exchange: str)`, `Quote(last: float, bid: float | None, ask: float | None, volume: float | None, time: datetime, delay_minutes: int | None)`. All `NamedTuple`, fields exactly as the spec's code block.
- Produces, in `provider.py`: `class UnknownTicker(Exception)`, `class ProviderUnavailable(Exception)`, `class UnsupportedExchange(ValueError)` (its message names the supported suffixes), and `class DataProvider(Protocol)` with `listing(ticker: str) -> Listing`, `history(ticker: str, start: date | None) -> History`, `quote(ticker: str) -> Quote`. Docstrings carry the spec's contract: `start=None` is the full history, `start` is inclusive, closed sessions only, which errors each method raises.
- Produces, in `sessions.py`: `CLOSE_MARGIN: timedelta`, `SUFFIX_CALENDARS: dict[str, str]` (key `""` for no suffix), `calendar_for(ticker: str) -> str` (case-insensitive; raises `UnsupportedExchange`), `newest_closed_session(exchange: str, now: datetime) -> date` (`now` must be timezone-aware).

- [ ] **Step 1: Write the failing tests** in `tests/data/test_sessions.py`

```python
def test_calendar_for_maps_suffixes() -> None:
    assert calendar_for("AAPL") == "XNYS"
    assert calendar_for("ASML.AS") == "XAMS"
    assert calendar_for("MC.PA") == "XPAR"
    assert calendar_for("asml.as") == "XAMS"

def test_calendar_for_rejects_unsupported_suffix() -> None:
    with pytest.raises(UnsupportedExchange, match=r"\.AS"):
        calendar_for("VOD.L")

def test_session_is_not_closed_within_the_margin() -> None:
    # XNYS closes 16:00 New York = 20:00 UTC on 2026-10-29 (EDT).
    assert newest_closed_session("XNYS", utc(2026, 10, 29, 20, 29)) == date(2026, 10, 28)
    assert newest_closed_session("XNYS", utc(2026, 10, 29, 20, 30)) == date(2026, 10, 29)

def test_weekend_falls_back_to_friday() -> None:
    assert newest_closed_session("XAMS", utc(2026, 10, 25, 12, 0)) == date(2026, 10, 23)

def test_holidays_are_not_sessions() -> None:
    # Good Friday 2026-04-03 on Euronext; Thanksgiving 2026-11-26 on NYSE.
    assert newest_closed_session("XAMS", utc(2026, 4, 3, 20, 0)) == date(2026, 4, 2)
    assert newest_closed_session("XNYS", utc(2026, 11, 26, 23, 0)) == date(2026, 11, 25)

def test_early_close_counts() -> None:
    # NYSE closes at 13:00 New York (18:00 UTC) on 2026-11-27, the day after Thanksgiving.
    assert newest_closed_session("XNYS", utc(2026, 11, 27, 18, 30)) == date(2026, 11, 27)
```

`utc(...)` is a local helper returning `datetime(..., tzinfo=UTC)`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/data/test_sessions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fartt.data'`

- [ ] **Step 3: Add the dependency and implement the modules**

Run `uv add exchange_calendars`. In `sessions.py`, `newest_closed_session` takes the session on or before `now`'s date in the calendar's time zone (`date_to_session(..., direction="previous")`), and steps back one session (`previous_session`) while `session_close(session) + CLOSE_MARGIN > now`. Cache calendars per code (`functools.cache`), because `get_calendar` is slow.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/data/test_sessions.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole gate**

Run: `task check`
Expected: lint, pyright (0 errors) and pytest all pass

- [ ] **Step 6: Commit**

```bash
git add src/fartt/data tests/data pyproject.toml uv.lock typings
git commit -m "feat: add the DataProvider interface and exchange sessions"
```

### Task 2: `YFinanceProvider.listing` and `.history`

**Files:**
- Create: `src/fartt/data/yfinance_provider.py`
- Modify: `src/fartt/data/__init__.py` (export `YFinanceProvider`)
- Modify: `pyproject.toml`, `uv.lock` (`uv add yfinance`)
- Create (if pyright needs it): `typings/yfinance/__init__.pyi`
- Test: `tests/data/test_yfinance_provider.py`, `tests/data/test_yfinance_provider_network.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `YFinanceProvider(ticker_factory: Callable[[str], YahooTicker] = yfinance.Ticker, clock: Callable[[], datetime] = <now in UTC>)`, satisfying `DataProvider`. `YahooTicker` is a Protocol in the same module declaring what's used: `history(**kwargs) -> pandas.DataFrame` and `info: dict[str, Any]`.

Behaviour:
- `listing(ticker)`: `calendar_for(ticker)` first (so `UnsupportedExchange` comes before any network call), then `info`. `name` is `longName`, else `shortName`, else the ticker; `currency` is `info["currency"]`. An `info` without `currency` or `quoteType` → `UnknownTicker`. The ticker is upper-cased.
- `history(ticker, start)`: `start=None` → `period="max"`, otherwise `start=start.isoformat()`; always `interval="1d", auto_adjust=False, actions=True, raise_errors=True`. Each row's date is the index's date in its own (exchange) time zone. Rows dated after `newest_closed_session(calendar_for(ticker), clock())`, rows before `start`, and rows with a NaN in Open/High/Low/Close are dropped; a NaN volume becomes `0.0`. A non-zero `Dividends` value becomes a `Dividend`, and a non-zero `Stock Splits` value a `Split`. All three lists are ascending.
- Errors: yfinance's missing-ticker errors (verify the names in the installed version, e.g. `YFTickerMissingError`, `YFPricesMissingError`) → `UnknownTicker` when `start is None`, and an empty `History` when `start` is given (nothing new). Any other exception from the ticker factory or its calls → `ProviderUnavailable`, chained (`raise ... from error`).

- [ ] **Step 1: Write the failing tests** in `tests/data/test_yfinance_provider.py`

Use a `FakeTicker` with an `info` dict and a `history(**kwargs)` that records its kwargs and returns a canned frame (or raises a canned error). Frames are built with a tz-aware `DatetimeIndex` in `America/New_York` or `Europe/Amsterdam` and the columns `Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits`. The clock is fixed at `2026-10-29 21:00 UTC` (NYSE closed, plus margin, for 2026-10-29).

```python
def test_history_requests_split_adjusted_daily_prices() -> None:
    # start=None → period="max"; start given → start="2026-10-01"
    assert ticker.calls[0] == {"period": "max", "interval": "1d", "auto_adjust": False,
                               "actions": True, "raise_errors": True}

def test_history_maps_rows_to_candles_on_the_exchange_date() -> None:
    # A row at 2026-10-28 00:00 America/New_York → Candle(date(2026, 10, 28), ...)
    assert history.candles == [Candle(date(2026, 10, 28), 10.0, 11.0, 9.0, 10.5, 1000.0)]

def test_history_drops_the_forming_session() -> None:
    # clock 2026-10-29 20:00 UTC (NYSE still open): the 2026-10-29 row is dropped
    assert [c.date for c in history.candles] == [date(2026, 10, 28)]

def test_history_drops_rows_before_start_and_rows_with_nan_prices() -> None: ...

def test_history_extracts_dividends_and_splits() -> None:
    # Dividends 0.25 on 10-27; Stock Splits 4.0 on 10-26 and 0.1 on 10-23
    assert history.dividends == [Dividend(date(2026, 10, 27), 0.25)]
    assert history.splits == [Split(date(2026, 10, 23), 0.1), Split(date(2026, 10, 26), 4.0)]

def test_unknown_ticker_raises_unknown_ticker() -> None: ...           # missing-ticker error, start=None
def test_missing_prices_after_start_is_an_empty_history() -> None: ... # same error, start given
def test_other_errors_raise_provider_unavailable() -> None: ...       # e.g. a rate-limit error, ConnectionError

def test_listing_maps_info() -> None:
    assert provider.listing("asml.as") == Listing("ASML.AS", "ASML Holding N.V.", "EUR", "XAMS")

def test_listing_of_unknown_ticker_raises_unknown_ticker() -> None: ...  # info == {}

def test_unsupported_suffix_is_rejected_before_any_call() -> None:
    with pytest.raises(UnsupportedExchange):
        provider.listing("VOD.L")
    assert factory_calls == []
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/data/test_yfinance_provider.py -v`
Expected: FAIL with `ImportError` for `YFinanceProvider`

- [ ] **Step 3: `uv add yfinance` and implement `listing` and `history`** in `src/fartt/data/yfinance_provider.py`

The class docstring states that this is the only module importing yfinance, and why (the provider stays swappable for IBKR).

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/data/test_yfinance_provider.py -v`
Expected: all pass

- [ ] **Step 5: Add the live tests** in `tests/data/test_yfinance_provider_network.py`, marked `network`

```python
@pytest.mark.network
@pytest.mark.parametrize("ticker", ["AAPL", "ASML.AS"])
def test_history_returns_recent_closed_sessions(ticker: str) -> None:
    history = YFinanceProvider().history(ticker, date.today() - timedelta(days=30))
    dates = [c.date for c in history.candles]
    assert dates and dates == sorted(dates)
    assert all(c.low <= c.close <= c.high for c in history.candles)

@pytest.mark.network
def test_unknown_ticker_is_reported() -> None:
    with pytest.raises(UnknownTicker):
        YFinanceProvider().history("NOSUCHTICKERXYZ", None)
```

Run: `uv run pytest tests/data/test_yfinance_provider_network.py -m network -v`
Expected: 3 passed. If the unknown-ticker test fails, fix the error mapping in Step 3 to match what yfinance actually raises, and add that case to the unit tests.

- [ ] **Step 6: Run the whole gate**

Run: `task check`
Expected: all pass, with yfinance imported only in `src/fartt/data/yfinance_provider.py` (`grep -rn "import yfinance\|from yfinance" src tests` lists that file and the two test files only)

- [ ] **Step 7: Commit**

```bash
git add src/fartt/data tests/data pyproject.toml uv.lock typings
git commit -m "feat: fetch split-adjusted daily history through yfinance"
```

### Task 3: `YFinanceProvider.quote`

**Files:**
- Modify: `src/fartt/data/yfinance_provider.py`
- Modify: `CLAUDE.md` (see "After the last task")
- Test: `tests/data/test_yfinance_provider.py`, `tests/data/test_yfinance_provider_network.py`

**Interfaces:**
- Consumes: `Quote`, `UnknownTicker`, `ProviderUnavailable`, `calendar_for` (Task 1); `YFinanceProvider` and its error mapping (Task 2).
- Produces: `YFinanceProvider.quote(ticker: str) -> Quote`. A4 (#100) builds `get_quote` on it.

Behaviour, all from `info`: `last` = `regularMarketPrice`; `bid`/`ask` = `bid`/`ask`, with `0`, `None` or missing → `None`; `volume` = `regularMarketVolume` or `None`; `time` = `regularMarketTime` (epoch seconds) as a UTC `datetime`; `delay_minutes` = `exchangeDataDelayedBy` or `None`. An `info` without `regularMarketPrice` → `UnknownTicker`. Errors map as in Task 2.

- [ ] **Step 1: Write the failing tests**

```python
def test_quote_maps_info() -> None:
    assert provider.quote("AAPL") == Quote(last=270.5, bid=270.4, ask=270.6, volume=1_000_000.0,
                                           time=datetime(2026, 10, 29, 20, 0, tzinfo=UTC), delay_minutes=0)

def test_zero_or_missing_bid_and_ask_become_none() -> None:
    # info with bid=0 and no "ask"
    assert (quote.bid, quote.ask) == (None, None)

def test_quote_of_unknown_ticker_raises_unknown_ticker() -> None: ...
def test_quote_errors_raise_provider_unavailable() -> None: ...
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/data/test_yfinance_provider.py -k quote -v`
Expected: FAIL with `AttributeError: 'YFinanceProvider' object has no attribute 'quote'`

- [ ] **Step 3: Implement `quote`**

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/data/test_yfinance_provider.py -v`
Expected: all pass

- [ ] **Step 5: Add a live test** marked `network`: `YFinanceProvider().quote("AAPL")` has `last > 0` and a timezone-aware `time`.

Run: `uv run pytest tests/data/test_yfinance_provider_network.py -m network -v`
Expected: 4 passed

- [ ] **Step 6: Run the whole gate**

Run: `task check`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/fartt/data tests/data CLAUDE.md
git commit -m "feat: fetch quotes through yfinance"
```

## After the last task

- Update `CLAUDE.md`'s "Current state" and "Architecture" sections with `fartt/data/` (one paragraph, in Task 3's commit).
- Mark #102 ready for review; its description lists the commits.
