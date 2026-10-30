# Epic A: Data Foundation — design

**Date:** 2026-10-30
**Epic:** #88 (stories #97 A1, #98 A2, #99 A3, #100 A4)
**PRD:** [stock-analysis-prd.md](../product/stock-analysis-prd.md) §5, §7, §8

## Intent

Every later tool (indicators, support/resistance, journal scoring, charts, trade plans) reads prices from this layer, so it has to be right before it's clever. Epic A gives the server daily candles and quotes for any Euronext or US ticker. The data comes from yfinance behind an interface, so IBKR's feed can replace it later without touching the cache or the tools.

The operator works with a handful of tickers a day: the ones they hold or are looking at, during a check-in after the US close or an analysis when something comes up. Nothing here has to be fast at scale. It has to be correct, say when it isn't current, and fetch a new ticker on the spot.

## Goals

- `get_candles(ticker, count)` and `get_quote(ticker)` for tickers on Euronext and US exchanges, one ticker per call.
- A local cache of split-adjusted daily candles per ticker, plus the ticker's dividends and splits, kept correct across splits.
- Trading sessions and exchange holidays respected: no gaps on days the exchange was closed, and no candle for a session that hasn't closed yet.
- A `DataProvider` interface with a yfinance implementation as the only module that imports yfinance.
- The BTC direction's code removed by the end of the epic (PRD §8).

## Non-goals

- More than one ticker per call, and bulk fetching. Epic G's screen will need that, and it gets designed there.
- Intraday candles, or weekly candles in the cache. Weekly candles are derived from daily ones where epic B needs them.
- Fundamentals, events and market context (epics F and G).
- The IBKR provider. It's a later story, and the interface just keeps it possible.
- Total-return (dividend-adjusted) prices for the agent.

## Anti-goals

These would make the epic a failure even if every tool technically works:

- **A price the agent sees that never traded.** For example, a dividend-adjusted level, or a level left unadjusted after a split. Support/resistance zones, stops and limit orders all use these prices.
- **Stale or missing data that looks current or complete.** For example, a zero spread when the bid/ask is unknown, a weekend counted as a gap, or a forming candle served as closed.
- **yfinance leaking past the provider.** Its exceptions, its column names or its ticker quirks reaching the cache or the server.

## Decisions

### Prices: split-adjusted, with dividends kept separately (decided with the operator)

The cache stores **split-adjusted** OHLCV, the prices a broker's chart shows. A support level at €640 means the stock really traded around €640 once splits are accounted for. Dividends and splits are stored as **separate event records**, not folded into the prices. When epic C scores a call against the world index ETF, it adds the dividends paid during the holding period to the stock's return. That keeps the comparison with an accumulating ETF fair without shifting any price the agent sees.

yfinance gives exactly this with `Ticker.history(interval="1d", auto_adjust=False, actions=True)`. In that mode, OHLC is split-adjusted but not dividend-adjusted, and the `Dividends` and `Stock Splits` columns carry the events. `Adj Close` is dropped.

A dividend never changes the stored prices. **Only a new split does**: it rescales every earlier candle, so the cache rebuilds that ticker's history (see Cache).

### One ticker per call, fetched on the spot (decided with the operator)

Tools take one ticker. A ticker the cache has never seen is fetched when it's first asked for, with its full available history (`period="max"`). Later calls fetch only what's newer. There's no watchlist and no setup step. `fartt download --ticker` exists for filling a cache by hand, through the same code path.

### yfinance first (decided with the operator)

The operator has no IBKR account yet, and yfinance needs no account, login or running gateway. Its costs: it's unofficial and can break or be rate-limited, its quotes are delayed, and its bid/ask is sometimes missing. The interface contains those costs, and the tools report them rather than hide them.

### Tickers use Yahoo's format (my call)

`AAPL`, `ASML.AS`, `MC.PA`: the suffix names the exchange, and no suffix means a US listing. This is the server's one naming scheme. The cache's file names and the tools' arguments use it, and a future IBKR provider translates its own symbols to it. Tickers are upper-cased on input.

## Components

### `fartt/data/`: the provider layer (A1, #97)

It replaces `fartt/exchange/` in shape: a Protocol, the value types, one error type, and one implementation.

```python
class Candle(NamedTuple):
    date: datetime.date      # the trading session, in the exchange's own calendar
    open: float
    high: float
    low: float
    close: float
    volume: float

class Dividend(NamedTuple):
    date: datetime.date      # ex-dividend date
    amount: float            # per share, in the ticker's currency

class Split(NamedTuple):
    date: datetime.date
    ratio: float             # new shares per old share, e.g. 4.0 for a 4-for-1 split

class History(NamedTuple):
    candles: list[Candle]
    dividends: list[Dividend]
    splits: list[Split]

class Listing(NamedTuple):
    ticker: str
    name: str
    currency: str            # ISO 4217, e.g. "EUR", "USD"
    exchange: str            # the exchange_calendars code, e.g. "XAMS", "XNYS"

class Quote(NamedTuple):
    last: float
    bid: float | None
    ask: float | None
    volume: float | None     # today's volume so far
    time: datetime.datetime  # when the quote was struck, UTC
    delay_minutes: int | None

class UnknownTicker(Exception): ...
class ProviderUnavailable(Exception): ...

class DataProvider(Protocol):
    def listing(self, ticker: str) -> Listing: ...
    def history(self, ticker: str, start: datetime.date | None) -> History: ...
    def quote(self, ticker: str) -> Quote: ...
```

- `history(ticker, None)` returns the full history. `history(ticker, start)` returns candles and events from `start` onwards, inclusive.
- `history` returns closed sessions only. It drops a row for a session that hasn't closed yet (see Sessions). This is the old `fetch_closed_candles` rule, moved from the interval clock to the exchange calendar.
- Every method raises `UnknownTicker` for a ticker the provider doesn't list, and `ProviderUnavailable` when it can't answer: network failure, rate limit, or an unexpected response. No yfinance exception gets past the module.
- **`YFinanceProvider`** (`fartt/data/yfinance_provider.py`) is the only module that imports yfinance:
  - `listing` comes from `Ticker.info`, or `fast_info` where it has the field. The exchange code comes from the ticker's suffix (see Sessions).
  - `history` comes from `Ticker.history(..., auto_adjust=False, actions=True)`.
  - `quote` comes from `fast_info` (last price, volume) and `info` (`bid`, `ask`, `exchangeDataDelayedBy`).
  - A bid or ask of `0` or missing becomes `None`. Yahoo uses 0 for "unknown".
  - *Builder's call:* how yfinance signals an unknown ticker in the installed version (an empty frame, a specific exception, or a missing `info`). Verify it with context7 and map it to `UnknownTicker`.

### Sessions (A2, #98)

`exchange_calendars` decides which days are sessions and when each one closes. The exchange comes from the ticker's suffix:

| Suffix | Exchange | Calendar |
|---|---|---|
| (none) | US (NYSE, Nasdaq) | `XNYS` |
| `.AS` | Euronext Amsterdam | `XAMS` |
| `.PA` | Euronext Paris | `XPAR` |
| `.BR` | Euronext Brussels | `XBRU` |
| `.LS` | Euronext Lisbon | `XLIS` |
| `.MI` | Euronext Milan (Borsa Italiana) | `XMIL` |
| `.IR` | Euronext Dublin | `XDUB` |
| `.OL` | Euronext Oslo | `XOSL` |

A ticker with any other suffix is outside the universe (PRD §5). Tools reject it with an error that names the supported suffixes. That's a separate error from an unknown ticker.

**A session counts as closed** 30 minutes after its scheduled close in the exchange calendar, early closes included. The margin lets Yahoo finalise the day's candle. *My call:* 30 minutes. It's a named constant, and the check-in after the US close is well past it.

**The newest closed session** is the one every freshness check compares against. Calendar days that aren't sessions (weekends, holidays) are never gaps. A session missing from the cache between its first and last candle is a gap. Yahoo occasionally has one. The cache doesn't fail on a gap; it reports it (see `get_candles`). Gaps are only checked from the calendar's first session onwards, because `exchange_calendars` doesn't reach as far back as Yahoo's longest histories.

### `fartt/stock_cache.py`: the cache (A2, #98)

It replaces `candle_cache.py`. The layout under `assets_dir`:

```
assets/stocks/ASML.AS/candles.csv    # Date,Open,High,Low,Close,Volume — split-adjusted, ascending
assets/stocks/ASML.AS/events.csv     # Date,Type,Value — Type is "dividend" or "split"
assets/stocks/ASML.AS/listing.json   # the Listing, cached so tools don't ask the provider each call
```

`StockCache(provider, assets_dir, clock)` serves any ticker. Its update rules:

1. **First fill.** No `candles.csv` exists, so call `history(ticker, None)` and write everything.
2. **Update.** Call `history(ticker, last_cached_date)`. This re-fetches the last cached candle, as the old cache did.
3. **Rebuild** when the returned splits include one dated after the last cached date, *or* the re-fetched last candle's open, high, low or close differs from the cached one by more than float noise (a relative difference above 1e-6; volume is ignored). Rebuilding means fetching the full history, writing it to a temporary file next to the original, and swapping it in with `os.replace`. The update reports the reason: `split` (with its date and ratio) or `last candle amended`.
4. **Otherwise** append the new candles and events.
5. **Up to date already.** The last cached date is the newest closed session, so there's no call to the provider at all.

*My call on rule 3:* any price amendment rebuilds, rather than a threshold that tries to tell an amendment from a split. Small-ratio splits (a 21-for-20 stock dividend) sit too close to real price revisions for any threshold to separate them. A rebuild is one call returning a few thousand rows, and it's always correct, so being wrong in the cautious direction costs a second. This drops the old cache's "replace the last line" path.

Rule 5 matters: a check-in that reads five indicators for a ticker doesn't hit Yahoo five times.

Kept from the old cache:
- the exclusive and shared `fcntl` locks on a sidecar `.lock` file, one per ticker directory
- healing a torn last line
- an injectable `clock` for tests

Dropped from it:
- the interval arithmetic
- the Bitvavo history start
- paging. yfinance returns a full daily history in one call.

`latest(ticker, count)` returns the newest `count` cached candles. `events(ticker)` returns its dividends and splits. Both take the shared lock.

*My call:* CSV, not Parquet. A ticker's full daily history is a few thousand rows, CSV matches the rest of `assets/`, and it can be inspected by hand.

### The server (A3 #99, A4 #100)

`build_server(cache, provider)` drops the fixed market and interval. `fartt serve` keeps `--assets-dir` and loses `--exchange`, `--market`, `--interval`, `--fee`, `--slippage`, `--threshold` and `--hit-rate-window`. The cost settings return in epic H's risk config.

**`get_candles(ticker, count=60)`**, with `count` from 1 to 200 as before:
- It updates the cache for the ticker (fetching on the spot if it's new), then returns `ticker`, `name`, `currency`, `exchange`, `columns`, `rows` (`[date, open, high, low, close, volume]`), `new_candles`, `rebuilt` (with the reason, if any), `gaps`, `is_current` and `warning`.
- `is_current` is true when the last row is the newest closed session.
- `gaps` lists the missing sessions within the returned rows (dates), empty when there are none, and `warning` mentions them. *My call:* surfaced to the agent rather than only logged, because missing data that looks complete is an anti-goal, and epic B's indicators will read the same rows.
- If the provider is unreachable, it serves the cached candles with `is_current: false` and a warning saying how many sessions behind they are. With nothing cached, it raises a `ToolError` saying the data provider is unreachable and to try again later.
- An unknown ticker gets a `ToolError` naming it and the expected format (`ASML.AS` for Euronext Amsterdam, `AAPL` for the US). An unsupported suffix gets a `ToolError` listing the supported ones.
- *My call:* the default count is 60, about three months of sessions. That's enough context for a glance; epic B's tools compute their indicators server-side from the cache, not from this output.

**`get_quote(ticker)`**:
- It returns `last`, `bid`, `ask`, `spread` (as a fraction of the mid price), `volume`, `average_volume_20d`, `volume_ratio`, `market_open`, `next_open` (when closed), `quote_time`, `delay_minutes` and `note`.
- `average_volume_20d` comes from the cache's last 20 sessions, so `get_quote` updates the cache first, as `get_candles` does. That way the two tools never disagree about volume.
- `market_open` and `next_open` come from the calendar, not from Yahoo.
- During the session, `volume_ratio` compares the day's volume so far with full-day averages, so it reads low early in the day. The field description says so; it isn't normalised for time of day.
- A missing bid or ask makes `bid`, `ask` and `spread` `null`, with a note saying the spread is unknown. It's never zero.
- An unreachable provider is a `ToolError`. A quote from the cache would be stale by definition, so there's nothing to degrade to.

**Server instructions** describe the stock tools: one ticker per call in Yahoo's format, prices split-adjusted, dividends kept separately, fractions as before, and `is_current: false` meaning stale data.

**The end-to-end test** (`tests/e2e/test_server_stdio.py`) starts `fartt serve` from `.mcp.json` with a temporary `--assets-dir`, lists the tools (`get_candles`, `get_quote`), and calls both on one ticker. It's marked `network`, as now, and asserts stdout carries only JSON-RPC.

## What leaves the codebase, and when

The stories are built in order, and each one leaves `task check` passing:

| Story | Adds | Removes |
|---|---|---|
| A1 #97 | `fartt/data/` (Protocol, types, `YFinanceProvider`), `yfinance` and `exchange_calendars` dependencies | — |
| A2 #98 | `stock_cache.py`, sessions; `fartt download --ticker` uses them | `fartt download`'s ccxt options |
| A3 #99 | `get_candles(ticker, count)` on the new cache; new server instructions | the forecast tools, `fartt/forecast/`, `fartt/model/`, the model-only visualization helpers, `fartt/exchange/`, `candle_cache.py`, the notebooks, `torch`, `ccxt` and other dependencies only they used, the BTC files in `assets/`, the old `serve` options |
| A4 #100 | `get_quote` | — |

This changes three stories, which get updated to match when this design is approved. A1 (#97): removing ccxt moves to A3, because until then the old cache and server still use it. A2 (#98): its adjustment scenario says a split *or dividend* rewrites the candles; only a split (or an amended last candle) does, and dividends are added as event records. A3 (#99): it gains the removal of ccxt and the old cache.

`calculate_trade_returns`, `calculate_candle_returns` and `calculate_technical_indicators` stay, because epics B and I build on them. So do `candlestick_chart.py` and `plot_styles.py`, which epic E builds on. *Builder's call:* any other module is removed in A3 if nothing left imports it.

## Testing

- **Provider:** unit tests mock yfinance at the module boundary, covering the mapping to `Candle`/`Dividend`/`Split`/`Quote`, `0` bid/ask → `None`, and errors → `UnknownTicker`/`ProviderUnavailable`. One `network` test per method hits Yahoo for `AAPL` and `ASML.AS`.
- **Cache:** tests use an in-memory fake `DataProvider` and a fixed clock. They cover first fill, update, a rebuild on a split event, a rebuild on a mismatched last candle, a no-op when up to date, a forming session not stored, weekend and holiday (Good Friday on XAMS, Thanksgiving on XNYS) not counted as gaps, and concurrent readers during an update.
- **Server:** in-process tool tests with the fake provider (as now), plus the stdio end-to-end test above.

## Left to the builder

- How yfinance signals an unknown ticker, and which of its exceptions count as unreachable (verify against the installed version).
- Field descriptions and the exact wording of warnings and errors. Errors follow the house rule: say what went wrong and what to do next.
- Whether `listing.json` is refreshed (for example, monthly) or only written on first fill.
