# Epic B: Technical Analysis — design

**Date:** 2026-10-31
**Epic:** #89 (stories #107 B1, #108 B2, #109 B3)
**PRD:** [stock-analysis-prd.md](../product/stock-analysis-prd.md) §5 (principles, step 5 of the analysis procedure, tool surface)

## Intent

The operator buys dips: "now" is when price drops to a level where it has turned before and looks set to swing up again. The bad case they've lived through is the dip that broke through the level and kept falling, and they want the tools to show the signs of that before buying.

Epic B gives the agent three tools for step 5 of the analysis procedure (timing):

- **`get_indicators`:** is the dip overdone?
- **`get_trend`:** is it a dip in an uptrend, or the start of a fall?
- **`get_support_resistance`:** where has price turned before, and is the nearest support likely to hold?

Each answers in a few clear numbers with states ("RSI 28, below 30 for 3 days"), not tables of prices. The agent weighs them, and epic H turns a zone into an entry, a stop below it and a target at the next resistance.

## Goals

- `get_indicators(ticker)`, `get_trend(ticker)` and `get_support_resistance(ticker)`, one ticker per call, computed on the server from the stock cache.
- Every value comes with its state, and every tool names the session it's based on.
- Support and resistance come as zones (price ranges) whose strength rests mostly on how often price turned there.
- For the nearest support, a list of signs pointing to "holding" and signs pointing to "breaking" (decided with the operator).
- Each tool's output for one ticker fits in a few hundred tokens: about 2,000 characters of JSON for `get_indicators` and `get_trend`, and 3,500 for `get_support_resistance`, whose zones and signs carry more (my call). A sign's `detail` stays under 80 characters.

## Non-goals

- Buy or sell signals, scores that combine the signs, or a recommendation. The signs are evidence; the agent and the operator decide (PRD principle "signals inform, they don't decide alone").
- Intraday timeframes. Daily and weekly only.
- Market context (index trend, sector strength) and news or earnings: epics F and G.
- Charts: epic E. Its candlestick chart can draw these zones later.
- Tuning the parameters on past data: that's validation (epic I).

## Anti-goals

- **A value computed from too little history that looks real.** For example, a "200-day average" over 120 sessions. Missing is `null` with a note.
- **An invented level.** A resistance zone above an all-time high made up from nothing, or a zone price never turned at.
- **A one-word verdict hiding the evidence.** The signs list exists so the agent can see why the support might break, not just that it might.
- **Stale data that looks current.** Each tool updates the cache first and flags staleness exactly as `get_candles` does.

## Decisions

### Signs, not a score (decided with the operator)

For the nearest support zone, `get_support_resistance` lists each sign with the side it points to (`holding` or `breaking`) and a short detail. It never sums them. A sign that's inconclusive is left out rather than listed as neutral.

### Turns weigh most (from the operator's description)

The operator's reason to buy is "price turned near that level before". A zone's strength is driven first by the number of times price turned there, then by recency, then by the volume traded in it, then by how many independent sources found it.

### One computation module per tool, over plain candles (my call)

A new package, `fartt/technical/`, holds the calculations as pure functions over `list[Candle]` (oldest first) from the cache. They don't do I/O or know about MCP, so they're tested with hand-built candles. The tools in `fartt/server/` stay thin: update the cache, read the candles, call the function, shape the output.

- `indicators.py` (B1): moving averages, RSI, MACD, ATR, volume and their states.
- `trend.py` (B2): weekly candles from daily ones, and the trend on each timeframe.
- `levels.py` (B3): zones, their strength, and the signs.

Indicator values come from TA-Lib (`SMA`, `EMA`, `RSI`, `MACD`, `ATR`) on numpy arrays. RSI and MACD take their periods from `TechnicalIndicatorsConfig`, so the tools and the candlestick chart agree. `calculate_technical_indicators` stays as it is, for the chart (epic E).

### Each tool reads the whole cached history (my call)

A ticker's full daily history is a few thousand rows, and the weekly 200-period average needs about four years of it. Each tool reads it all. That costs milliseconds, and no tool has to work out how much warm-up its indicators need.

### Shared staleness (my call)

The update, the error mapping and the staleness check move out of `get_candles` into one helper all four candle-based tools use. Each tool returns `ticker`, `based_on` (the session date its values come from), `is_current` and `warning`. The rules are the same as `get_candles`:

- An unreachable provider serves the cached sessions with a "N sessions behind" warning.
- With nothing cached, the tool raises a `ToolError`.

B1 does this refactor.

## Components

### `get_indicators(ticker)` (B1, #107)

Computed on the daily candles, as of the last cached session. All prices are in the ticker's currency.

- **Moving averages:** SMA 20, 50 and 200, and EMA 20. Each reports:
  - `value`
  - `price_vs`: `above` or `below`
  - `distance`: (close − average) / average, as a fraction
  - `slope`: `rising` or `falling`, compared with the value 5 sessions earlier
  - `days`: how many sessions in a row the close has stayed on the same side
- **RSI 14:**
  - `value`
  - `zone`: `oversold` below 30, `overbought` above 70, else `neutral`
  - `days`: sessions in that zone
  - `direction`: `rising` or `falling` over the last 3 sessions
- **MACD (12, 26, 9):**
  - `line`, `signal` and `histogram`
  - `histogram_direction`
  - `last_cross`: `bullish` or `bearish`, with `sessions_ago`
- **ATR 14:**
  - `value`
  - `fraction`: ATR / close
  - `stretch`: (close − SMA 20) / ATR, how far price has run from its 20-day average in ATRs. It's how "overdone" a dip is.
- **Volume:**
  - `last`
  - `average_20d`: the 20 sessions before the last
  - `ratio`
  - `down_day_ratio`: the mean volume on down days over the last 10 sessions divided by the mean on up days. Above 1 means heavier selling.

An indicator that needs more sessions than are cached is `null`, and `note` names each one with the sessions it needs. MACD needs 34, RSI 15, ATR 15 and the SMA its period plus 5 for the slope.

`calculate_technical_indicators` isn't extended. B1 adds `indicators.py` beside it.

### `get_trend(ticker)` (B2, #108)

Two timeframes, `daily` and `weekly`, each reporting:

- `direction`:
  - `up` when close > SMA 50 > SMA 200 and the SMA 50 is rising
  - `down` when close < SMA 50 < SMA 200 and the SMA 50 is falling
  - `sideways` otherwise
- the evidence: `close`, `sma_50`, `sma_200`, both slopes (rising or falling, compared with 5 periods earlier), and `last_cross` (`golden` or `death`, with its date: the period the SMA 50 crossed the SMA 200)

The output adds `aligned` (true only when both are `up` or both are `down`; two `sideways` timeframes aren't a trend, my call) and `note`.

**Weekly candles** are built from the daily ones:

- A week is the exchange's sessions in one ISO week (Monday to Sunday), so a holiday week has fewer sessions.
- Each weekly candle takes the first open, the highest high, the lowest low, the last close and the summed volume. It's dated by its last session.
- The current week counts only once its last scheduled session has closed (from the exchange calendar). Until then, the weekly figures are as of the previous week, and `note` says so.

With fewer than 205 weekly candles, the weekly SMA 200 is `null`. The weekly direction is then judged on close against SMA 50 and its slope alone, and `note` says so. The daily timeframe follows the same rule with daily candles.

### `get_support_resistance(ticker)` (B3, #109)

**Discovery comes first.** B3 starts with a throwaway script, not committed, that prints the zones for three tickers and renders `candlestick_chart` with the zones drawn as bands. The operator checks the zones by eye. The algorithm below is the starting point. Its numbers are expected to move after that check, and the B3 plan records what changed. The output shape is fixed only after the check.

**Window:** the last 500 sessions (about two years), or all that's cached if fewer.

**Candidate levels:**
- **Swing highs and lows:** a high above the 5 sessions on each side of it, or a low below them.
- **Volume profile:** the volume traded at each price, binned into bins half an ATR wide; the bins in the top 10% by volume are candidates.
- **Pivots:** classic floor pivots (P, S1, R1) from the last completed week.
- **The SMA 50 and SMA 200:** their current values.
- **Round numbers:** multiples of a step that depends on the price (1 below 20, 5 below 100, 10 below 500, 50 above). They're used only as a tie-breaker source; on their own they never form a zone.

**Zones:**
- Candidates within half an ATR of each other merge into one zone.
- The zone spans its candidates' lowest to highest price, at least half an ATR wide around its centre.
- **A zone needs at least one turn** (decided with the operator). A level price never turned at, such as the SMA 200 alone, a weekly pivot or a high-volume bin, isn't listed, even above every swing high or below every swing low. The other sources only add to a zone that turns already found, raising its strength and naming themselves in `sources`.

**For each zone:**
- `low` and `high`.
- `sources`: which kinds of candidates found it.
- `touches`: episodes where a session's range entered the zone. Consecutive sessions in the zone count as one episode.
- `turns`: the touches after which price closed back out on the side it came from, within 3 sessions of the episode's last session. The side it came from is where the close before the episode was; a gap straight across the zone is a touch but not a turn (my call).
- `last_touch`: a date.
- `volume_share`: the zone's share of the window's volume.
- `strength`: 0–100. Turns dominate; then recency, then volume share, then the number of sources. The exact weights are set after discovery and live in one function.

**Output:**
- `last`: the last close.
- `support`: up to 3 zones below the last close, nearest first.
- `resistance`: up to 3 zones above it, nearest first.
- For each zone, its distance from the last close as a fraction and in ATRs.
- A zone that contains the last close is listed as support, with `distance` 0, and the note says price is in it.
- Price above every high in the whole cached history (not just the window): `resistance` is empty and the note says there's no resistance from history above the price. Below every low, the same applies to support. When the 500-session window has no zone above the price but older history does, the search for that side reaches back through the full history (my call). If that still finds no zone with a turn, the side is empty and the note says price never turned above (or below) it.

**Signs for the nearest support zone** (`signs`, a list of `{sign, points, detail}`, where `points` is `holding` or `breaking`):

| Sign | Holding when | Breaking when |
|---|---|---|
| Daily trend | `up` | `down` |
| Weekly trend | `up` | `down` |
| Price vs SMA 200 | above | below |
| Volume on the drop | `down_day_ratio` below 0.8 | above 1.2 |
| RSI divergence | of the two most recent swing lows in the last 20 sessions (the same swing detection as the zones, with 3 sessions on each side), the later is lower in price but higher in RSI (my call) | — |
| Momentum | RSI and MACD histogram both rising | both falling |
| Retests | — | the zone was touched 3 or more times in the last 60 sessions |
| Recent break | — | a close below the zone in the last 20 sessions |
| Turns | 3 or more turns in the window | — (every zone has at least one turn; 1 or 2 is inconclusive and left out, my call) |

Every threshold here is a named constant. The trend and indicator values come from B1's and B2's functions, not recomputed.

**Ex-dividend sessions** (my call). Prices aren't dividend-adjusted, so an ex-dividend gap looks like a down day. `levels.py` takes the cached dividend dates beside the candles. It leaves ex-dividend sessions out of `down_day_ratio` and out of the Recent break sign, and doesn't count a swing low on an ex-dividend date.

### Server instructions

The instructions gain one sentence: indicators, trend and zones are evidence, not signals, and `get_support_resistance`'s signs list which way each points.

### The end-to-end test

Each story adds its tool to `EXPECTED_TOOLS` and calls it on ASML.AS after `get_candles`.

## Delivery order

| Story | Adds |
|---|---|
| B1 #107 | `fartt/technical/indicators.py`, the shared staleness helper, `get_indicators` |
| B2 #108 | `fartt/technical/trend.py` (weekly candles, trend), `get_trend` |
| B3 #109 | the discovery script (not committed), `fartt/technical/levels.py`, `get_support_resistance` |

## Testing

- **Calculations:** unit tests on hand-built candles whose answers are known:
  - a steady uptrend for `up`, and a rising-then-falling series for a cross
  - a series dipping into an earlier swing low twice for turns
  - an all-time high for the empty resistance list
  - fewer sessions than an indicator needs for the `null`s
  - TA-Lib values cross-checked against a hand computation once per indicator
- **Weekly candles:** a holiday week, and a current week that hasn't closed.
- **Tools:** in-process tests with the fake provider for output shape, staleness and errors, plus a size check against each tool's budget (above).
- **End to end:** each tool over stdio against live Yahoo, marked `network`.

## Left to the builder

- Field names inside each block, as long as they say what the value is and its state.
- How the shared staleness helper is split between functions.
- The strength formula's weights, after B3's discovery.
