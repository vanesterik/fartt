# Stock Analysis MCP — PRD

**Date:** 2026-10-29
**Status:** Approved (2026-10-29)
**Owner:** Koen van Esterik

Gaps are tagged inline: 🔶 **Assumption** (plausible, not yet validated) and 🔵 **Open question** (unknown, needs a decision or discovery). Section 10 collects the open questions.

## 1. Executive Summary

We're building an MCP server that turns Claude Code into a disciplined stock analyst for one private investor, who today picks Euronext and US stocks partly on gut feeling. The agent follows the same eight-step procedure on every analysis (market context, sector, screening, fundamentals, technicals, trade plan, execution, review). Each analysis ends in a sized trade plan within risk limits that only the server can change, and in a journal entry that is scored automatically against a world index ETF. Paper trading comes first. Real money follows only when the journal shows a positive excess return after costs over at least 30 closed plans. Orders placed by the server, each approved by hand, come last and behind a second gate.

## 2. Problem Statement

### Who has this problem?

A private investor in the Netherlands (the operator) who buys individual stocks on Euronext and US exchanges alongside, or instead of, index funds.

### What is the problem?

Picks are made partly on gut feeling, without a repeatable process, and they go wrong in five recurring ways:

1. **No exit plan.** Losers are held too long, waiting for a recovery; winners are sold too early.
2. **Position size by feel.** One bad pick hurts more than it should.
3. **Bad timing.** The stock was right, but the entry came at a local top or just before earnings.
4. **Too little research.** Acting on a headline, a tip or a chart without checking fundamentals or the wider market.
5. **No record.** Nothing is written down at decision time, so there's no way to tell whether the picks beat simply holding a world index ETF.

### Why is it painful?

The first three cost money directly. The fourth makes it likely. The fifth hides all of them: without a record, a run of lucky calls feels like skill, and nothing shows whether the time spent picking stocks is worth more than buying the index.

### Evidence

- The operator's own account of how picks go wrong (the five points above, confirmed 2026-10-29).
- Forecasting returns from price history alone is hard to trust: a single model rarely shows a lasting edge after costs. That's a reason to distrust any single signal, and to measure rather than assume an edge.
- An LLM is good at building a convincing story from mixed signals. Without scoring, the agent's analysis could fix point 4 while making points 1–3 feel safer than they are.

## 3. Target Users & Personas

### Primary user: the operator (you)

- A solo private investor with a technical background who runs Claude Code and this server on their own machine.
- **Rhythm:** a short check-in every trading day after the US close (positions against their plans, calls whose horizon passed), plus deeper analyses whenever a ticker or the market calls for one.
- **Wants:** a procedure that's followed every time, plans with an exit and a size before any money moves, and an honest record that says whether this beats the index.
- **Doesn't want:** a product, customers, or signals to sell. This is for personal use only.

### Secondary user: the agent (Claude Code)

- It runs the analysis procedure as an MCP prompt and calls the server's tools. It combines signals into a view, proposes a trade plan and records every call.
- **Needs from the tools:** compact, structured output (key numbers with a clear state, such as "RSI 28, below 30 for 3 days"), not raw tables or prose; errors that say what went wrong and what to do next; and a clear flag when data is stale or missing.
- **Constraints:** it can read the risk limits but never change them. In the analysis session's permission profile it can call only this server's tools.

## 4. Strategic Context

### Why now?

The operator picks stocks today, with their own money, and the five problems above cost money on every pick. An MCP server with Claude Code as the analyst is cheap to build for one person, and a journal scored against the index answers within months whether the effort beats simply holding it.

### End state

The server places real orders through Interactive Brokers, each approved by the operator, for trade plans the agent proposed and the server checked against the risk limits. It gets there in three stages, each opened by a gate in section 6:

1. **Paper:** every trade plan is paper traded, and the journal scores it.
2. **Small real positions, placed by hand:** the operator places the agent's plans in IBKR personally, in small amounts. The server never touches the real account.
3. **Server-placed orders, approved by hand:** epic J places the orders through the IBKR API after the operator approves each one.

### Broker and data

- **Broker: Interactive Brokers** (IBKR Ireland for Dutch residents). It has mature Web and TWS APIs, access to Euronext and US markets, a free paper trading account, and paid market data per exchange (subscribe only to what's used). **Saxo** is the fallback: it has an OpenAPI with a free simulation environment, but higher fees. **DEGIRO** is out because it has no official API.
- **Development data: yfinance** (free, delayed), behind a `DataProvider` interface. It switches to IBKR's feed once the tooling works.

## 5. Solution Overview

An MCP server (`fartt serve`, started by Claude Code from `.mcp.json`) that owns all data, risk limits and state. The agent analyses; the server measures and enforces.

### Principles

- **Walking skeleton first.** A full analysis runs end to end on one ticker (epics A–D) before the tool set grows.
- **The server owns data, limits and state.** The agent can read the risk limits, never change them.
- **Compact, structured tool output.** Key numbers and clear states, not raw tables or prose.
- **Errors say what went wrong and what to do next.**
- **Every call goes in the journal** and is scored automatically after its horizon.
- **The data provider sits behind an interface**, so the source can be swapped.
- **A dedicated permission profile**: the analysis session can call only this server's tools.
- **Signals inform, they don't decide alone.** Support and resistance set entries, stops and targets, not buy or sell signals. Technical indicators on their own don't give a reliable edge.

### The analysis procedure

Fundamentals decide **what**, technicals decide **when**, and risk management decides **how much**. The procedure is published as an MCP prompt, so the agent follows the same order every time:

1. **Market context:** the index above or below its 200-day moving average, volatility, upcoming macro events.
2. **Sector selection:** each sector's relative strength against the index.
3. **Screening:** filter the universe on liquidity, market cap, profitability and debt.
4. **Fundamental analysis:** quality, growth, valuation against peers and own history, value-trap checks, earnings dates.
5. **Technical analysis (timing):** trend, support/resistance zones, momentum (RSI, MACD), volume confirmation.
6. **Trade plan and risk:** entry, a stop below support, a target at the next resistance, a size within the risk limits, and a reward-to-risk of at least 2.
7. **Execution:** limit orders; avoid the first and last minutes of the session; watch the spread.
8. **Monitoring and review:** track positions against their plans, log every decision, and review which steps add value.

### Universe

All of Euronext and the US exchanges. The screen's liquidity and market-cap filters do most of the narrowing. Small caps have patchier data, so every tool reports missing values as missing, never as zero.

### Tool surface

| Epic | Tool | Returns |
|---|---|---|
| A | `get_candles(ticker, count)` | Recent closed daily candles, from the cache |
| A | `get_quote(ticker)` | Last price, bid/ask and spread, volume against its average, delay |
| B | `get_indicators(ticker)` | Moving averages, RSI, MACD, ATR, volume, as values with states |
| B | `get_support_resistance(ticker)` | Price zones with a strength score (touches, volume traded) |
| B | `get_trend(ticker)` | Trend direction on the daily and weekly timeframes |
| C | `record_decision(...)` | Stores the call, horizon, reasoning and signals used |
| C | `score_decisions()` | Scores calls whose horizon has passed against realised prices |
| C | `get_performance()` | Hit rate and excess return against the world index ETF |
| D | prompt `analyze_ticker` | The eight-step procedure |
| E | `render_chart(ticker, ...)` | Candles, moving averages, zones, volume, RSI/MACD panels, as an image |
| F | `get_market_regime()` | Index against its 200-day average, plus volatility |
| F | `get_sector_strength()` | Sectors ranked by relative strength against the index |
| G | `get_fundamentals(ticker)` | Margins, debt, growth, valuation ratios, with source and age |
| G | `compare_peers(ticker)` | Valuation against the sector and the stock's own history |
| G | `screen(criteria)` | A short list from the universe, with why each ticker passed |
| G | `get_events(ticker)` | Earnings and dividend dates |
| G | `estimate_value(ticker, assumptions)` | A simple DCF with visible, adjustable assumptions |
| H | `get_risk_config()` | The risk limits (read only) |
| H | `plan_trade(...)` | Entry, stop, target, size, reward-to-risk; rejects plans that break a limit |
| H | `get_portfolio()` | Paper positions against their plans, P&L after costs |
| I | `backtest_rule(rule)` | Return, hit rate, drawdown and trades after costs, with a walk-forward split |
| J | `place_order(plan)` | Places an approved plan's limit order and its stop through IBKR |

Tool names and outputs are settled in each epic's design; this table is the intent.

### User journey

**Daily check-in (after the US close).** The server has already settled paper positions whose stop or target was hit in the day's candles. The operator asks for a check-in, and the agent calls `get_portfolio`, `score_decisions` and `get_market_regime`. It then reports positions against their plans, calls whose horizon passed and how they scored, and any change in the market regime.

**Analysis.** The operator invokes `analyze_ticker` with a ticker, or with none to start from the screen. The agent works through the eight steps. A step without a tool yet is named as skipped. The analysis ends in a trade plan (`plan_trade`) or a decision not to trade, and both go into the journal (`record_decision`). In stage 1 an accepted plan opens a paper position. In stage 2 the operator places it in IBKR by hand. In stage 3 `place_order` places it after the operator approves.

## 6. Success Metrics

### Primary metric

**Excess return after costs**: the average return of closed trade plans, after fees and slippage, minus the return of a world index ETF over the same holding periods. 🔵 VWCE or IWDA as the benchmark is decided in epic C's design.

### Secondary metrics

- **Hit rate:** the share of closed plans that reached their target before their stop. It's read together with the average reward-to-risk, because a 40% hit rate at 2:1 has positive expectancy.
- **Procedure adherence:** the share of analyses that ran every available step in order and ended in the journal.
- **Step value:** for each step, whether calls that used it scored better than calls that didn't (reviewed, not optimised).

### Gate: stage 1 (paper) → stage 2 (small real positions, placed by hand)

- At least 3 months of paper trading, and at least 30 closed trade plans.
- A positive average excess return after costs.
- A hit rate and average reward-to-risk (at least 2) that together give positive expectancy.

### Gate: stage 2 → stage 3 (server-placed orders, approved by hand)

- The stage 1 gate's results hold for 3 more months with real, small positions.

### Guardrails (the default risk config, enforced by the server)

| Limit | Default |
|---|---|
| Starting paper capital | €10,000 |
| Maximum risk per position (loss if the stop is hit) | 2% of the portfolio |
| Maximum position size | 20% of the portfolio |
| Maximum open positions | 8 |
| Minimum reward-to-risk | 2 |
| Kill switch | blocks new plans at 16% below the portfolio's peak, until the operator resets it outside the agent |
| Costs | IBKR's tiered commission per exchange, plus 0.1% slippage per leg 🔶 |

Only the operator can change these, in a config file outside the agent's reach. The kill switch matches the maximum open risk: 8 positions at 2% each put 16% at risk, so every stop being hit as planned brings the portfolio to the threshold but not past it. Only losing more than the plans allowed trips it (see risk 5).

## 7. User Stories & Requirements

### Epic hypothesis

If we give Claude Code a fixed analysis procedure, tools that turn data into compact signals, server-enforced risk limits, and a journal that scores every call against a world index ETF, then the operator's stock picks will be planned, sized and timed rather than made on gut feeling. Whether that beats the index will be known from evidence before real money is involved.

### Epics

Built in this order. Each epic is a GitHub issue on the [project board](https://github.com/users/vanesterik/projects/3) in the epic-hypothesis format.

| Epic | Issue | Scope |
|---|---|---|
| A. Data Foundation | #88 | `DataProvider` (yfinance), stock candle cache, `get_candles`, `get_quote` |
| B. Technical Analysis | #89 | `get_indicators`, `get_support_resistance`, `get_trend` |
| C. Journal & Evaluation | #90 | `record_decision`, `score_decisions`, `get_performance` |
| D. Analysis Pipeline | #91 | The procedure as an MCP prompt; the permission profile |
| | | **Milestone: walking skeleton.** A full analysis runs end to end on one ticker. |
| E. Visualization | #92 | `render_chart` with indicator panels |
| F. Market Context | #93 | `get_market_regime`, `get_sector_strength` |
| G. Fundamentals & Screening | #94 | `get_fundamentals`, `compare_peers`, `screen`, `get_events`, `estimate_value` |
| H. Trade Plan & Risk | #95 | Risk config, `plan_trade`, paper portfolio, kill switch |
| I. Validation | #96 | `backtest_rule`, walk-forward checks |
| J. Live Orders | #101 | IBKR order placement with approval, behind the stage 2 → 3 gate |

🔵 Paper trading starts once epic H lands, so the 3-month clock for the first gate starts after H, not after the walking skeleton. Before then, decisions are journalled and scored, but not as sized plans. Moving H before E–G would start that clock sooner (see section 10).

### Stories

Stories are written just in time, one epic at a time, as GitHub issues in the Use Case + Gherkin format. Epic A's stories are #97–#100.

### Constraints

- Daily candles only. Holding periods are days to weeks.
- The cache must handle trading sessions, exchange holidays, and split and dividend adjustments, which change every earlier candle's adjusted price.
- yfinance data is delayed and unofficial. Each tool reports where its data came from and how old it is.
- Positions in USD stocks are held from a EUR account. 🔶 Paper P&L is reported in EUR at the day's rate, and FX costs are part of the trading costs.

## 8. Out of Scope

- **Forecast models.** A price forecast may return later as one signal among many, not as the basis of a decision.
- **Crypto,** and any market outside Euronext and the US exchanges.
- **Order book (L2) data.** It's expensive and of little value for holding periods of days to weeks. Revisit only if the journal shows a real hit rate.
- **Intraday trading** and timeframes shorter than a day.
- **News and sentiment signals, and agent-driven retraining.**
- **Short selling, options and leverage.**
- **Unattended operation.** The agent runs in a session the operator starts. No order is placed without approval.
- **Anyone but the operator.** No product, no customers, no selling signals.

## 9. Dependencies & Risks

### Dependencies

- **yfinance** for development data. It's unofficial and can break or be rate-limited without notice.
- **An IBKR account and its paper trading account**, and later market data subscriptions for Euronext and the US. Open them while developing on yfinance.
- **The MCP SDK** and Claude Code's MCP support.

### Risks and mitigations

1. **Convincing stories, no edge.** The agent builds a persuasive case from mixed signals. *Mitigation:* every call is journalled with the signals used, scored automatically and benchmarked against the index. The gates are set on the score, not on the reasoning.
2. **Overfitting rules to history.** A rule the agent proposes looks good only in hindsight. *Mitigation:* `backtest_rule` always reports a walk-forward split and flags a large in-sample/out-of-sample gap.
3. **Bad data looks like a signal.** Unadjusted splits, missing fundamentals or stale prices produce false levels and ratios. *Mitigation:* the cache rebuilds a ticker's history when an adjustment appears, and tools report missing values as missing and stale data as stale.
4. **Patchy small-cap data.** The universe includes small caps, where yfinance coverage is thin. *Mitigation:* the screen filters on liquidity and data completeness; a ticker without enough data is reported, not analysed.
5. **Losses beyond the plan.** A stock can gap through its stop (overnight news, earnings), losing more than 2%, and the drawdown counts open profit given back since the peak. Either can reach the 16% kill switch even though no plan broke a limit. *Mitigation:* that is the switch's job: losing more than the plans allowed is the moment to stop and look. Each position records the slippage on its stop, so the journal shows how often gaps happen; `get_events` warns when earnings fall inside a plan's horizon.
6. **The agent widens its own permissions.** It calls tools outside the server (shell, web, file edits) and changes state the server should own. *Mitigation:* the analysis session's permission profile (epic D) allows only this server's tools, and the risk config lives outside the agent's reach.
7. **Survivorship and look-ahead in scoring.** Scoring against prices that weren't known at decision time, or only on tickers that still exist. *Mitigation:* each decision stores the prices at decision time, and scoring uses only candles after it.

## 10. Open Questions

1. 🔵 **Benchmark:** VWCE or IWDA. Decided in epic C's design.
2. 🔵 **Order of epics E–H.** Moving H (paper portfolio) before E–G starts the 3-month paper clock sooner. Decide at the walking-skeleton milestone.
3. 🔵 **Costs per exchange:** IBKR's tiered commissions for Euronext and the US, and FX costs, verified against the IBKR account once it's open.
4. 🔵 **Horizon of a call:** fixed per call (chosen by the agent), or the plan's stop/target with a maximum holding period. Decided in epic C's design.
5. 🔵 **Macro events** for step 1: which source, given that yfinance doesn't provide an economic calendar. Decided in epic F's design.
