"""
Scoring a call: how the stock did against the benchmark over its horizon.

Every return is in EUR, the account's currency and IWDA.AS's: a price in
another currency is divided by that day's `EUR<CCY>=X` close (units of the
currency per euro). The stock's return runs from `decision_session` to
`horizon_ends`, with the dividends going ex after the first and up to and
including the second added at face value, converted at the end's rate. The
benchmark's runs from its own stamped session to its last session on or
before `horizon_ends`; it's accumulating and priced in EUR, so its price is
its whole return.

Only what was known after the decision counts: closes come from the cached
candles of those exact sessions (split-adjusted, so a split inside the
horizon rescales both ends alike), and nothing after `horizon_ends` is read.
A `buy` is right when the excess return is above 0, a `pass` when it's 0 or
below. Pure functions, no I/O.

"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta

from fartt.data import (
    FX_SUFFIX,
    Candle,
    Dividend,
    calendar_for,
    session_on_or_before,
)
from fartt.journal import BENCHMARK, BenchmarkScore, Decision, Leg, Score

# How far back a rate may come from when Yahoo has no FX candle on the day.
RATE_LOOKBACK_DAYS = 5

PRICE_DECIMALS = 4
RATE_DECIMALS = 6
RETURN_DECIMALS = 6


class Unscorable(Exception):
    """
    A call that can't be scored yet: `str()` is the reason, and `ticker`
    the stock, benchmark or exchange rate whose close is missing.

    """

    def __init__(self, ticker: str, reason: str) -> None:
        super().__init__(reason)
        self.ticker = ticker


def score_call(
    decision: Decision,
    candles: Sequence[Candle],
    dividends: Sequence[Dividend],
    benchmark: Sequence[Candle],
    rates: Sequence[Candle] | None,
    scored_at: datetime,
) -> Score:
    """
    Score `decision` from the stock's cached `candles` and `dividends`, the
    benchmark's cached candles and, for a stock not priced in EUR, the cached
    candles of its exchange rate (`None` for a EUR stock). Raises
    `Unscorable` naming the close or rate that's missing.

    """
    start, end = decision.decision_session, decision.horizon_ends
    start_close = _close(decision.ticker, candles, start)
    end_close = _close(decision.ticker, candles, end)
    benchmark_end = session_on_or_before(calendar_for(BENCHMARK), end)
    benchmark_start_close = _close(BENCHMARK, benchmark, decision.benchmark.session)
    benchmark_end_close = _close(BENCHMARK, benchmark, benchmark_end)
    start_rate = end_rate = None
    if rates is not None:
        start_rate = rate_on(decision.currency, rates, start)
        end_rate = rate_on(decision.currency, rates, end)

    paid = round(
        sum(d.amount for d in dividends if start < d.date <= end), PRICE_DECIMALS
    )
    stock_return = round(
        (end_close + paid) / (end_rate or 1) / (start_close / (start_rate or 1)) - 1,
        RETURN_DECIMALS,
    )
    benchmark_return = round(
        benchmark_end_close / benchmark_start_close - 1, RETURN_DECIMALS
    )
    excess = round(stock_return - benchmark_return, RETURN_DECIMALS)
    return Score(
        decision_id=decision.id,
        scored_at=scored_at,
        start=Leg(session=start, close=start_close, rate=start_rate),
        end=Leg(session=end, close=end_close, rate=end_rate),
        dividends=paid,
        return_=stock_return,
        benchmark=BenchmarkScore(
            ticker=BENCHMARK,
            start=Leg(session=decision.benchmark.session, close=benchmark_start_close),
            end=Leg(session=benchmark_end, close=benchmark_end_close),
            return_=benchmark_return,
        ),
        excess_return=excess,
        right=excess > 0 if decision.direction == "buy" else excess <= 0,
    )


def _close(ticker: str, candles: Sequence[Candle], day: date) -> float:
    for candle in candles:
        if candle.date == day:
            return round(candle.close, PRICE_DECIMALS)
    raise Unscorable(
        ticker, f"{ticker}'s close for {day.isoformat()} isn't available yet"
    )


def rate_on(currency: str, rates: Sequence[Candle], day: date) -> float:
    """
    The rate's close on `day`, or the latest in the `RATE_LOOKBACK_DAYS`
    before it when Yahoo has none on the day, once the rate's candles have
    reached `day` (before that, the day's own close may still come).

    """
    if not rates or max(c.date for c in rates) < day:
        raise Unscorable(
            f"EUR{currency}{FX_SUFFIX}",
            f"the EUR/{currency} rate for {day.isoformat()} isn't available yet",
        )
    earliest = day - timedelta(days=RATE_LOOKBACK_DAYS)
    within = [c for c in rates if earliest <= c.date <= day]
    if not within:
        raise Unscorable(
            f"EUR{currency}{FX_SUFFIX}",
            f"there's no EUR/{currency} rate within {RATE_LOOKBACK_DAYS} days "
            f"before {day.isoformat()}",
        )
    return round(max(within, key=lambda c: c.date).close, RATE_DECIMALS)
