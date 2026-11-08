"""
Paper trading a plan on the daily candles.

A recorded buy's plan is a limit buy at its entry, live from the first
session after the plan's `based_on` session for `valid_for` sessions of the
stock's exchange, never past the call's `horizon_ends`. The first window
session whose low reaches the entry fills it: at the open when the session
opened at or below the entry, else at the entry. A window without such a
session expires the order once its last session is cached.

A filled position closes on the first session that, in this order, opens
at or below the stop (at the open, with the slippage beyond the stop) or at
or above the target (at the open), reaches the stop with its low (at the
stop, also when the high reached the target: daily candles don't say which
came first) or the target with its high (at the target), or is the call's
`horizon_ends` (at its close). On the fill session itself the target
doesn't count, since its high may have come before the dip that filled the
order; a fill at an open already at or below the stop closes at once at
that open.

Each leg costs what `trade_plan.leg_costs` says on its actual value and is
converted to EUR at its own session's rate, by scoring's rules. Dividends
going ex after the fill session up to and including the exit session are
added gross, at the exit's rate. Pure functions over cached data, no I/O.

"""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field

from fartt.data import Candle, Dividend, Split, session_after
from fartt.journal import Decision, Exit, ExitReason, Expiry, Fill
from fartt.risk_config import Costs
from fartt.scoring import PRICE_DECIMALS, Unscorable, rate_on
from fartt.trade_plan import CENTS, TradePlan, leg_costs

PositionState = Literal["waiting", "open", "expired", "closed"]


class Position(BaseModel):
    """A call's paper position as the journal holds it."""

    state: PositionState = Field(
        description=(
            "waiting: the limit order is live, not filled; open: filled; "
            "closed: exited; expired: not filled within its window."
        )
    )
    fill: Fill | None = Field(description="The fill; null until filled.")
    exit: Exit | None = Field(description="The exit; null until closed.")
    expiry: Expiry | None = Field(
        description="The expiry; null unless the order expired."
    )


class Blocked(NamedTuple):
    """Why a plan can't advance: `ticker` is the stock or rate that's missing."""

    ticker: str
    reason: str


class Settled(NamedTuple):
    """What settling a plan found: the lines to write, and why it stopped short."""

    fill: Fill | None
    expiry: Expiry | None
    exit: Exit | None
    blocked: Blocked | None


def position_state(
    fill: Fill | None, expiry: Expiry | None, exit: Exit | None
) -> PositionState:
    if exit is not None:
        return "closed"
    if expiry is not None:
        return "expired"
    return "open" if fill is not None else "waiting"


def order_window(decision: Decision, plan: TradePlan) -> tuple[date, date]:
    """The first and last session the plan's limit order is live."""
    exchange = decision.exchange
    first = session_after(exchange, plan.based_on, 1)
    last = session_after(exchange, plan.based_on, plan.valid_for)
    return first, min(last, decision.horizon_ends)


def _price(value: float) -> float:
    return round(value, PRICE_DECIMALS)


def _exit_rule(
    candle: Candle, plan: TradePlan, fill: Fill, horizon_ends: date
) -> tuple[ExitReason, float, bool] | None:
    """The reason, price and whether at the open, if `candle` closes the position."""
    open_, high, low, close = (_price(p) for p in candle[1:5])
    stop, target = plan.stop, plan.target
    if candle.date == fill.session:
        if fill.price <= stop:
            return "stop", fill.price, True
        if low <= stop:
            return "stop", stop, False
    else:
        if open_ <= stop:
            return "stop", open_, True
        if open_ >= target:
            return "target", open_, True
        if low <= stop:
            return "stop", stop, False
        if high >= target:
            return "target", target, False
    if candle.date == horizon_ends:
        return "horizon", close, False
    return None


def settle_plan(
    decision: Decision,
    plan: TradePlan,
    fill: Fill | None,
    candles: Sequence[Candle],
    dividends: Sequence[Dividend],
    splits: Sequence[Split],
    rates: Sequence[Candle] | None,
    costs: Costs,
    newest: date,
    settled_at: datetime,
) -> Settled:
    """
    Walk `plan` session by session, from its window's first session or the
    stored `fill`'s session, up to `newest` (the exchange's newest closed
    session) or the call's `horizon_ends`. Returns the fill, expiry or exit
    the cached sessions settle (a fill only when it's new), and why the walk
    stopped short, if it did: a candle or rate not cached, or a split after
    the plan was made. `rates` is the `fx_pair`'s candles, `None` for EUR.

    """
    ticker, exchange = decision.ticker, decision.exchange
    first, until = order_window(decision, plan)
    day = fill.session if fill is not None else first
    last = min(newest, decision.horizon_ends)
    if day > last:
        return Settled(None, None, None, None)
    split = next((s for s in splits if plan.based_on < s.date <= last), None)
    if split is not None:
        return Settled(
            None,
            None,
            None,
            Blocked(
                ticker,
                f"{ticker} split {split.ratio:g}-for-1 on {split.date.isoformat()}, "
                f"after the plan was made on {plan.based_on.isoformat()}; "
                "settling across a split isn't supported",
            ),
        )

    leg_exchange = costs.exchanges[exchange]

    def rate(session: date) -> float | None:
        return None if rates is None else rate_on(decision.currency, rates, session)

    def values(price: float, shares: int, rate: float | None) -> tuple[float, float]:
        """The leg's value in the stock's currency, and in EUR to the cent."""
        value = shares * price
        return value, round(value / (rate or 1.0), CENTS)

    by_day = {candle.date: candle for candle in candles}
    new_fill: Fill | None = None
    while day <= last:
        candle = by_day.get(day)
        if candle is None:
            return Settled(
                new_fill,
                None,
                None,
                Blocked(
                    ticker,
                    f"{ticker}'s candle for {day.isoformat()} isn't available yet",
                ),
            )
        if fill is None:
            if _price(candle.low) <= plan.entry:
                at_open = _price(candle.open) <= plan.entry
                price = _price(candle.open) if at_open else plan.entry
                try:
                    fill_rate = rate(day)
                except Unscorable as error:
                    return Settled(None, None, None, Blocked(error.ticker, str(error)))
                value, eur = values(price, plan.shares, fill_rate)
                fill = new_fill = Fill(
                    decision_id=decision.id,
                    settled_at=settled_at,
                    session=day,
                    price=price,
                    at_open=at_open,
                    shares=plan.shares,
                    rate=fill_rate,
                    value=eur,
                    costs=leg_costs(
                        value, fill_rate, decision.currency, leg_exchange, costs, "buy"
                    ),
                )
            elif day == until:
                expiry = Expiry(
                    decision_id=decision.id, settled_at=settled_at, session=day
                )
                return Settled(None, expiry, None, None)
        if fill is not None:
            rule = _exit_rule(candle, plan, fill, decision.horizon_ends)
            if rule is not None:
                reason, price, at_open = rule
                try:
                    exit_rate = rate(day)
                except Unscorable as error:
                    return Settled(
                        new_fill, None, None, Blocked(error.ticker, str(error))
                    )
                value, eur = values(price, fill.shares, exit_rate)
                sell = leg_costs(
                    value, exit_rate, decision.currency, leg_exchange, costs, "sell"
                )
                paid = _price(
                    sum(d.amount for d in dividends if fill.session < d.date <= day)
                )
                income = round(fill.shares * paid / (exit_rate or 1.0), CENTS)
                exit = Exit(
                    decision_id=decision.id,
                    settled_at=settled_at,
                    session=day,
                    reason=reason,
                    price=price,
                    at_open=at_open,
                    slippage=_price(plan.stop - price)
                    if reason == "stop" and at_open
                    else 0.0,
                    rate=exit_rate,
                    value=eur,
                    costs=sell,
                    dividends=income,
                    pnl=round(
                        eur - fill.value + income - fill.costs.total - sell.total, CENTS
                    ),
                )
                return Settled(new_fill, None, exit, None)
        day = session_after(exchange, day, 1)
    return Settled(new_fill, None, None, None)
