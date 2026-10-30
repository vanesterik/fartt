# Stock candle cache (A2, #98) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, in this session, on branch `a2-stock-cache`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local cache of each ticker's split-adjusted daily candles, dividends, splits and listing, kept up to date through the `DataProvider`, and `fartt download --ticker` filling it.

**Architecture:** `StockCache` in `src/fartt/stock_cache.py` serves any ticker from one directory per ticker under `assets_dir/stocks/`. An update fetches only what's newer than the last cached session, rebuilds the whole history when a split appears or the last cached candle's prices changed, and skips the provider when the cache already holds the newest closed session. `sessions.py` gains ticker normalisation and gap detection. `fartt download` switches to the cache; `fartt serve` and the old `CandleCache` stay until A3.

**Tech Stack:** Python 3.14, `fartt.data` (A1), exchange_calendars, `fcntl` locks, CSV via the `csv` module, pytest, pyright strict.

**Spec:** `docs/specs/2026-10-30-data-foundation-design.md`, sections "Sessions" and "`fartt/stock_cache.py`: the cache", and the removal table's A2 row.

## Global Constraints

- Layout: `assets_dir/stocks/<TICKER>/candles.csv` (`Date,Open,High,Low,Close,Volume`, ISO dates, ascending), `events.csv` (`Date,Type,Value`, `Type` is `dividend` or `split`, ascending), `listing.json` (the `Listing` fields), and a sidecar `.lock`.
- Prices are stored as the provider returns them (split-adjusted). A dividend never changes a stored price.
- Rebuild when a returned split is dated after the last cached date, or when the re-fetched last candle's open, high, low or close differs from the cached one by a relative difference above `1e-6` (volume ignored). A rebuild fetches the full history, writes each file to a temporary file next to it, and swaps it in with `os.replace`. The reason is reported: `split on <date> (<ratio> new shares per old)` or `last candle amended`.
- No provider call when the last cached date is the newest closed session (and `listing.json` exists).
- Updates hold an exclusive `fcntl` lock on the ticker's `.lock`; `latest`, `events` and `listing` hold a shared one.
- Gaps are sessions missing between the first and last given date, counted only from the calendar's first session; weekends and the exchange's holidays are never gaps.
- No yfinance, ccxt or pandas import in anything this plan touches. The cache sees only `fartt.data`.
- Tickers are stripped and upper-cased; anything but letters, digits, `.` and `-` is rejected before it reaches a file path.

## Review Focus

1. **A ticker that would escape `assets_dir`** (`../x`, `A/B`) → rejected before any path is built. Tested in Task 1.
2. **A dividend on the last cached date, returned again by the re-fetch** → not recorded twice. Tested in Task 3.
3. **A torn last line in `candles.csv`** (a killed update) → healed, and the update still compares against the last whole candle. Tested in Task 3.
4. **The provider failing mid-rebuild** → the old files stay intact (nothing is swapped in). Tested in Task 3.
5. **A history older than the calendar's first session** (AAPL back to 1980) → gap detection ignores the part before it instead of raising. Tested in Task 1.

---

### Task 1: Ticker normalisation and gap detection

**Files:**
- Modify: `src/fartt/data/sessions.py`, `src/fartt/data/__init__.py` (export the two new functions)
- Modify: `typings/exchange_calendars/__init__.pyi` (declare `first_session`, `sessions_in_range`)
- Test: `tests/data/test_sessions.py`

**Interfaces:**
- Produces: `normalise_ticker(ticker: str) -> str` (strip, upper-case; raises `UnknownTicker` naming the ticker when it holds anything but `A-Z`, `0-9`, `.`, `-`, or is empty), and `missing_sessions(exchange: str, dates: Sequence[date]) -> list[date]` (sessions of `exchange` from `max(min(dates), first session)` to `max(dates)` that aren't in `dates`, ascending; `[]` for fewer than two dates).
- `calendar_for` calls `normalise_ticker` first, so `" asml.as "` maps to `XAMS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_normalise_ticker_strips_and_upper_cases() -> None:
    assert normalise_ticker(" asml.as ") == "ASML.AS"
    assert normalise_ticker("brk-b") == "BRK-B"

@pytest.mark.parametrize("ticker", ["", "../X", "A/B", "AAPL;rm"])
def test_normalise_ticker_rejects_what_cannot_be_a_ticker(ticker: str) -> None:
    with pytest.raises(UnknownTicker):
        normalise_ticker(ticker)

def test_calendar_for_ignores_surrounding_whitespace() -> None:
    assert calendar_for(" asml.as ") == "XAMS"

def test_holidays_and_weekends_are_not_gaps() -> None:
    # XAMS 2026-04-02 (Thu) → 2026-04-07 (Tue): Good Friday and Easter Monday closed
    assert missing_sessions("XAMS", [date(2026, 4, 2), date(2026, 4, 7)]) == []
    # XNYS 2026-11-25 (Wed) → 2026-11-27 (Fri): Thanksgiving closed
    assert missing_sessions("XNYS", [date(2026, 11, 25), date(2026, 11, 27)]) == []

def test_a_missing_trading_day_is_a_gap() -> None:
    dates = [date(2026, 10, 26), date(2026, 10, 28)]
    assert missing_sessions("XNYS", dates) == [date(2026, 10, 27)]

def test_dates_before_the_calendar_starts_are_not_checked() -> None:
    dates = [date(1980, 12, 12), date(2026, 10, 26), date(2026, 10, 27)]
    assert missing_sessions("XNYS", dates)  # no DateOutOfBounds; only the calendar's range is checked
```

The last test asserts that the call returns (the range from the calendar's first session to 2026-10-27 is full of gaps here, which is fine).

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/data/test_sessions.py -v`
Expected: FAIL with `ImportError` for `normalise_ticker`

- [ ] **Step 3: Implement both functions** in `sessions.py`; extend the stub with `first_session: datetime` and `sessions_in_range(start, end) -> Sequence[datetime]` (check the signature with context7).

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/data -v`
Expected: all pass

- [ ] **Step 5: Run the whole gate and commit**

Run: `task check` (Expected: all pass), then:

```bash
git add src/fartt/data tests/data typings
git commit -m "feat: normalise tickers and find missing sessions"
```

### Task 2: `StockCache` storage, first fill and reads

**Files:**
- Create: `src/fartt/stock_cache.py`
- Modify: `tests/fakes.py` (add `FakeProvider`)
- Test: `tests/test_stock_cache.py`

**Interfaces:**
- Consumes: `DataProvider`, `Candle`, `Dividend`, `Split`, `History`, `Listing`, `ProviderUnavailable`, `calendar_for`, `newest_closed_session`, `normalise_ticker` from `fartt.data`.
- Produces:
  - `UpdateResult(NamedTuple)`: `new_candles: int`, `rebuilt: str | None` (the reason), `fetched: bool` (whether the provider was called).
  - `StockCache(provider: DataProvider, assets_dir: Path, clock: Callable[[], datetime] = <now in UTC>)` with `directory(ticker: str) -> Path`, `update(ticker: str) -> UpdateResult`, `latest(ticker: str, count: int) -> list[Candle]` (oldest first; `[]` when nothing is cached), `events(ticker: str) -> tuple[list[Dividend], list[Split]]`, `listing(ticker: str) -> Listing | None` (from `listing.json`; `None` when not cached), `newest_closed_session(ticker: str) -> date`.
  - Every method normalises the ticker first. Provider errors (`UnknownTicker`, `ProviderUnavailable`, `UnsupportedExchange`) propagate unchanged.
- In this task `update` handles only the first fill (no `candles.csv`): fetch `listing` and `history(ticker, None)`, write all three files via temporary files and `os.replace`, return `UpdateResult(len(candles), None, True)`.
- `FakeProvider(histories: dict[str, History], listings: dict[str, Listing], now: datetime)` in `tests/fakes.py`: `history` filters the stored history by `start` (inclusive) and by `newest_closed_session(listing.exchange, now)`; `reachable: bool` (raises `ProviderUnavailable` when false); records `history_calls: list[tuple[str, date | None]]`; unknown tickers raise `UnknownTicker`. Tests mutate `now` and the stored history to simulate the next day, a split or an amendment.

- [ ] **Step 1: Write the failing tests** in `tests/test_stock_cache.py`

```python
def test_first_update_writes_the_full_history(tmp_path: Path) -> None:
    result = cache.update("aapl")
    assert result == UpdateResult(new_candles=3, rebuilt=None, fetched=True)
    lines = (tmp_path / "stocks" / "AAPL" / "candles.csv").read_text().splitlines()
    assert lines[0] == "Date,Open,High,Low,Close,Volume"
    assert lines[1].startswith("2026-10-26,")

def test_latest_returns_the_newest_candles_oldest_first(tmp_path: Path) -> None:
    assert [c.date for c in cache.latest("AAPL", 2)] == [date(2026, 10, 27), date(2026, 10, 28)]

def test_latest_of_an_uncached_ticker_is_empty(tmp_path: Path) -> None: ...

def test_events_and_listing_are_cached(tmp_path: Path) -> None:
    assert cache.events("AAPL") == ([Dividend(date(2026, 10, 27), 0.26)], [])
    assert cache.listing("AAPL") == Listing("AAPL", "Apple Inc.", "USD", "XNYS")

def test_first_update_never_writes_the_forming_session(tmp_path: Path) -> None:
    # provider.now during the 2026-10-29 NYSE session; the stored history has a 10-29 row
    assert cache.latest("AAPL", 10)[-1].date == date(2026, 10, 28)

def test_an_unreachable_provider_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ProviderUnavailable):
        cache.update("AAPL")
    assert not (tmp_path / "stocks" / "AAPL").exists() or not any((tmp_path / "stocks" / "AAPL").glob("*.csv"))

def test_a_reader_waits_for_a_running_update(tmp_path: Path) -> None:
    # A FakeProvider whose history() blocks on a threading.Event; a reader thread calling
    # latest() mid-update returns only after the event is set and the update has written.
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_stock_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fartt.stock_cache'`

- [ ] **Step 3: Implement** `src/fartt/stock_cache.py` (first fill, reads, locks). `listing.json` is written with `json.dumps(listing._asdict())`.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_stock_cache.py -v`
Expected: all pass

- [ ] **Step 5: Run the whole gate and commit**

Run: `task check` (Expected: all pass), then:

```bash
git add src/fartt/stock_cache.py tests/fakes.py tests/test_stock_cache.py
git commit -m "feat: cache a ticker's full daily history"
```

### Task 3: Incremental updates and rebuilds

**Files:**
- Modify: `src/fartt/stock_cache.py`
- Test: `tests/test_stock_cache.py`

**Interfaces:**
- Consumes: Task 2's `StockCache`, `UpdateResult`, `FakeProvider`.
- Produces: `StockCache.update` with all five rules of the spec. Rebuilds also rewrite `listing.json`.

Rules, in order, after healing a torn last line of `candles.csv` (a last line that isn't a header or six fields with an ISO date is cut; a whole last line missing its newline gets one):
1. No `candles.csv` (or only a header) → first fill (Task 2).
2. Last cached date ≥ newest closed session and `listing.json` exists → `UpdateResult(0, None, False)`.
3. `history(ticker, last_cached_date)`; rebuild if a returned split is dated after the last cached date, or the returned candle for the last cached date differs in open/high/low/close by a relative difference above `1e-6`.
4. Otherwise append candles dated after the last cached date, and events dated after it.
5. An empty returned history appends nothing: `UpdateResult(0, None, True)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_next_day_appends_only_the_new_session(tmp_path: Path) -> None:
    # fill; provider.now moves past the 10-29 close; stored history gains 10-29
    assert cache.update("AAPL") == UpdateResult(1, None, True)
    assert provider.history_calls[-1] == ("AAPL", date(2026, 10, 28))

def test_an_up_to_date_cache_does_not_call_the_provider(tmp_path: Path) -> None:
    assert cache.update("AAPL") == UpdateResult(0, None, False)
    assert len(provider.history_calls) == 1

def test_a_new_split_rebuilds_the_history(tmp_path: Path) -> None:
    # provider's history replaced by a 4-for-1 split on 10-29, earlier prices divided by 4
    result = cache.update("AAPL")
    assert result.rebuilt == "split on 2026-10-29 (4.0 new shares per old)"
    assert cache.latest("AAPL", 4)[0].close == pytest.approx(original_close / 4)

def test_an_amended_last_candle_rebuilds_the_history(tmp_path: Path) -> None:
    assert cache.update("AAPL").rebuilt == "last candle amended"

def test_a_volume_only_change_is_not_an_amendment(tmp_path: Path) -> None:
    assert cache.update("AAPL").rebuilt is None

def test_a_new_dividend_is_appended_and_no_price_changes(tmp_path: Path) -> None:
    candles_before = cache.latest("AAPL", 10)
    cache.update("AAPL")  # history gains 10-29 with a 0.26 dividend
    assert cache.latest("AAPL", 10)[:-1] == candles_before
    assert cache.events("AAPL")[0][-1] == Dividend(date(2026, 10, 29), 0.26)

def test_a_dividend_on_the_last_cached_date_is_not_recorded_twice(tmp_path: Path) -> None:
    # dividend on 10-28 (cached), re-fetched with start=10-28
    assert cache.events("AAPL")[0].count(Dividend(date(2026, 10, 28), 0.26)) == 1

def test_a_torn_last_line_is_healed_before_updating(tmp_path: Path) -> None:
    # append "2026-10-29,10" without newline; update heals and appends 10-29 properly
    assert [c.date for c in cache.latest("AAPL", 2)] == [date(2026, 10, 28), date(2026, 10, 29)]

def test_a_failed_rebuild_leaves_the_old_files(tmp_path: Path) -> None:
    # split triggers rebuild; provider fails on the full-history call
    with pytest.raises(ProviderUnavailable):
        cache.update("AAPL")
    assert cache.latest("AAPL", 10) == candles_before
```

`FakeProvider` gets an optional `fail_on_full_history: bool` for the last test.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_stock_cache.py -v`
Expected: the new tests FAIL (the cache only knows the first fill)

- [ ] **Step 3: Implement the update rules** in `stock_cache.py`

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_stock_cache.py -v`
Expected: all pass

- [ ] **Step 5: Run the whole gate and commit**

Run: `task check` (Expected: all pass), then:

```bash
git add src/fartt/stock_cache.py tests/fakes.py tests/test_stock_cache.py
git commit -m "feat: update the stock cache incrementally and rebuild on splits"
```

### Task 4: `fartt download --ticker`

**Files:**
- Modify: `src/fartt/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `CLAUDE.md`, `README.md` (the `download` command's description and example)

**Interfaces:**
- Consumes: `StockCache`, `UpdateResult` (Tasks 2–3), `YFinanceProvider`, `UnknownTicker`, `UnsupportedExchange`, `ProviderUnavailable` from `fartt.data`.
- Produces: `fartt download --ticker TICKER [--assets-dir DIR]`. `--ticker` is required. `download` loses `--exchange`, `--market` and `--interval`; `serve` keeps its options (A3 replaces them), so `_add_cache_options` is used only by `serve`.

Behaviour: logs the ticker, the cache directory and the result (new candles, rebuilt reason, or "already up to date"). `UnknownTicker` and `UnsupportedExchange` → `parser.error(...)` (exit 2) with the error's message. `ProviderUnavailable` → `parser.exit(1, ...)`. The module-level name `YFinanceProvider` is what tests monkeypatch, as `CcxtExchange` is now.

- [ ] **Step 1: Replace the download tests** in `tests/test_cli.py` (the ccxt-based ones: `test_download_is_a_subcommand`, `..._rejects_unknown_market_with_hint`, `..._rejects_an_interval_the_exchange_does_not_offer`, `..._rejects_an_unknown_exchange`, `..._exits_cleanly_when_the_exchange_is_unreachable`, `..._rejects_an_offered_interval_the_cache_cannot_handle`) with:

```python
def test_download_fills_the_ticker_cache(fake_provider: FakeProvider, tmp_path: Path) -> None:
    cli.main(["download", "--assets-dir", str(tmp_path), "--ticker", "aapl"])
    assert (tmp_path / "stocks" / "AAPL" / "candles.csv").exists()

def test_download_rejects_an_unknown_ticker(fake_provider: FakeProvider, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--ticker", "NOSUCH"])
    assert exit_info.value.code == 2

def test_download_rejects_an_unsupported_exchange(...) -> None: ...   # "VOD.L" → exit 2, message names .AS
def test_download_exits_1_when_the_provider_is_unreachable(...) -> None: ...
def test_download_requires_a_ticker(...) -> None: ...                 # no --ticker → exit 2
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: the new download tests FAIL (`--ticker` is unknown); the serve tests still pass

- [ ] **Step 3: Implement** the new `download` in `cli.py`

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Try it live** (not a test): `uv run fartt download --assets-dir <tmp> --ticker ASML.AS`, then run it again.
Expected: the first run writes the full history; the second says it is already up to date, or appends one session.

- [ ] **Step 6: Update the docs.** `CLAUDE.md`: the `fartt download` paragraph in "Commands" and a `stock_cache.py` entry in "Architecture"; `README.md`: the Usage section's `download` example. Then run `task check` (Expected: all pass) and commit:

```bash
git add src/fartt/cli.py tests/test_cli.py CLAUDE.md README.md
git commit -m "feat: download a ticker's history into the stock cache"
```

## After the last task

Open the pull request for #98 (`Closes #98`), listing the commits; run the final branch review before marking it ready.
