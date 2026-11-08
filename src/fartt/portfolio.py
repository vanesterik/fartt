"""
The paper portfolio: cash, open positions against their plans, the plans
waiting for their entry, and the closed positions, all in EUR.

Everything is worked out from the journal's fills and exits and the cached
candles, dividends and rates; nothing is stored. Cash is the starting
capital less every fill (its value and buy costs), plus every exit (its
value less sell costs, plus its dividends), plus the dividends an open
position has gone ex on so far, counted as received. An open position is
valued at what selling it at its newest cached close would bring after the
costs to close, at that close's EUR rate, so cash plus the open positions'
value is the total, and the total is the starting capital plus the realised
and unrealised P&L, to the cent. A plan waiting for its entry reserves its
position value plus buy costs out of the cash. Each amount is rounded to the
cent and every sum is of rounded amounts.

The kill switch is worked out the same way, never stored: it trips on the
first valuation day since the operator's last reset whose value is at or
below the running peak less `kill_switch_drawdown`, and stays tripped
until the next reset, which starts the peak again from the value at the
reset. The open-position limit counts open positions and waiting plans
together. Pure functions, no I/O.

"""

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import date, datetime
from typing import NamedTuple

from pydantic import BaseModel, Field

from fartt.data import (
    Candle,
    Dividend,
    fx_pair,
    newest_closed_session,
    sessions_in_range,
)
from fartt.journal import Decision, Exit, ExitReason, Fill, Positions, Reset
from fartt.positions import order_window
from fartt.risk_config import Costs, Limits
from fartt.scoring import PRICE_DECIMALS, RATE_DECIMALS, Unscorable, rate_on
from fartt.technical import calculate_indicators
from fartt.trade_plan import CENTS, PlanRefused, TradePlan, leg_costs

# How many of the most recent exits the portfolio lists.
MAX_RECENT = 5
DISTANCE_DECIMALS = 4


class Market(NamedTuple):
    """The cached data the portfolio is valued on, per ticker (rates by pair)."""

    candles: Mapping[str, Sequence[Candle]]  # Oldest first
    dividends: Mapping[str, Sequence[Dividend]]


class Totals(BaseModel):
    starting_capital: float = Field(description="The operator's starting capital.")
    cash: float = Field(
        description=(
            "Starting capital less every fill and its buy costs, plus every "
            "exit less its sell costs with its dividends, plus the dividends "
            "open positions have gone ex on so far. Can dip below 0 when a "
            "US fill's rate was worse than the plan's."
        )
    )
    reserved: float = Field(
        description="Cash set aside for the waiting plans: position value plus buy costs."
    )
    free_cash: float = Field(description="cash − reserved: what a new plan may use.")
    open_value: float = Field(
        description="What selling every open position at its last close would bring, after costs."
    )
    total_value: float = Field(description="cash + open_value.")
    realised_pnl: float = Field(description="The closed positions' P&L after costs.")
    unrealised_pnl: float = Field(
        description="The open positions' P&L after the costs to close, with dividends."
    )
    peak: float = Field(
        description=(
            "The highest portfolio value at any session close since the last "
            "kill switch reset (the first fill without one), and at least the "
            "value at the reset (the starting capital without one)."
        )
    )
    peak_session: date | None = Field(
        description=(
            "The session of the peak: the reset's session while the value at "
            "the reset is it; null while the starting capital is it."
        )
    )
    drawdown: float = Field(
        description="(peak − total_value) / peak: 0.05 means 5% below the peak."
    )
    open_risk: float = Field(
        description="The further loss if every open stop is hit as planned, after costs."
    )
    waiting_risk: float = Field(
        description="The waiting plans' planned risk, apart from open_risk."
    )
    open_positions: int = Field(description="How many positions are open.")
    waiting_plans: int = Field(description="How many plans wait for their entry.")


class Holding(BaseModel):
    """An open position against its plan."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    currency: str = Field(description="The currency its prices are in.")
    shares: int = Field(description="Shares held.")
    entry: float = Field(description="The fill price.")
    stop: float = Field(description="The plan's stop.")
    target: float = Field(description="The plan's target.")
    last_close: float = Field(description="The newest cached close.")
    based_on: date = Field(description="That close's session.")
    rate: float | None = Field(
        description="The EUR rate it's valued at (units per euro); null for EUR."
    )
    value: float = Field(
        description="What selling at the last close would bring, after the costs to close, in EUR."
    )
    dividends: float = Field(
        description="Dividends gone ex since the fill, gross, in EUR at this rate."
    )
    unrealised_pnl: float = Field(
        description="value + dividends − the fill's value and buy costs, in EUR."
    )
    to_stop: float = Field(description="(last close − stop) / last close.")
    to_stop_atr: float | None = Field(
        description="(last close − stop) in ATRs; null without an ATR."
    )
    to_target: float = Field(description="(target − last close) / last close.")
    to_target_atr: float | None = Field(
        description="(target − last close) in ATRs; null without an ATR."
    )
    sessions_held: int = Field(
        description="Sessions of its exchange after the fill session, up to based_on."
    )
    horizon_ends: date = Field(
        description="The session it closes on at the latest, at its close."
    )
    is_current: bool = Field(
        description="Whether based_on is its exchange's newest closed session."
    )
    warning: str | None = Field(description="What's stale, if anything.")


class WaitingPlan(BaseModel):
    """A plan whose limit buy hasn't filled."""

    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    entry: float = Field(description="The limit price.")
    stop: float = Field(description="The plan's stop.")
    target: float = Field(description="The plan's target.")
    shares: int = Field(description="The plan's shares.")
    reserved: float = Field(
        description="Cash set aside: the plan's position value plus buy costs, in EUR."
    )
    valid_until: date = Field(description="The order window's last session.")


class ReasonTotal(BaseModel):
    count: int = Field(description="How many positions closed this way.")
    pnl: float = Field(description="Their realised P&L, in EUR.")


class ByReason(BaseModel):
    stop: ReasonTotal
    target: ReasonTotal
    horizon: ReasonTotal


class RecentExit(BaseModel):
    id: int = Field(description="The call's id.")
    ticker: str = Field(description="The stock.")
    session: date = Field(description="The session it closed on.")
    reason: ExitReason = Field(description="stop, target or horizon.")
    pnl: float = Field(description="The realised P&L after costs, in EUR.")


class ClosedSummary(BaseModel):
    count: int = Field(description="How many positions have closed.")
    realised_pnl: float = Field(description="Their P&L after costs, in EUR.")
    by_reason: ByReason = Field(description="Count and P&L per exit reason.")
    recent: list[RecentExit] = Field(
        description=f"The {MAX_RECENT} most recent exits, newest first."
    )


class LastReset(BaseModel):
    """The operator's latest reset of the kill switch."""

    reset_at: datetime = Field(description="When it was reset (UTC).")
    session: date = Field(description="The valuation day it was worked out on.")
    drawdown: float = Field(description="The drawdown at the reset.")
    note: str | None = Field(description="The operator's note, if any.")


class KillSwitch(BaseModel):
    """Whether new plans are blocked for a drawdown, until the operator resets it."""

    tripped: bool = Field(
        description=(
            "Whether a session closed at or beyond the threshold below the "
            "peak since the last reset: no new plans until the operator resets it."
        )
    )
    tripped_on: date | None = Field(
        description="The first session that closed at or beyond it; null if none."
    )
    drawdown_at_trip: float | None = Field(
        description="The drawdown on that session; null if not tripped."
    )
    threshold: float = Field(
        description="The operator's kill_switch_drawdown: 0.16 means 16% below the peak."
    )
    drawdown: float = Field(
        description="The drawdown now, from the peak since the reset."
    )
    last_reset: LastReset | None = Field(
        description="The operator's latest reset; null if never reset."
    )


class PositionLimit(BaseModel):
    """The open-position limit, counting waiting plans too."""

    max_open_positions: int = Field(description="The operator's limit.")
    open_positions: int = Field(description="How many positions are open.")
    waiting_plans: int = Field(description="How many plans wait for their entry.")
    room: int = Field(description="How many more plans fit.")


class Portfolio(BaseModel):
    based_on: date | None = Field(
        description="The newest session an open position is valued at; null with none open."
    )
    totals: Totals
    open: list[Holding]
    waiting: list[WaitingPlan]
    closed: ClosedSummary
    stale: list[str] = Field(description="Tickers valued on a stale close.")
    skipped_days: int = Field(
        description=(
            "Valuation days left out of the peak because a close or rate of "
            "theirs isn't cached yet."
        )
    )
    kill_switch: KillSwitch
    limits: PositionLimit
    newest_valuation: date | None = Field(
        description=(
            "The newest valuation day valued since the last reset; the "
            "reset's session without one since, null without either."
        )
    )


def _cents(value: float) -> float:
    return round(value, CENTS)


def _sale(
    shares: int,
    price: float,
    rate: float | None,
    decision: Decision,
    costs: Costs,
) -> float:
    """What selling `shares` at `price` brings in EUR, after the sell leg's costs."""
    value = shares * price
    sell = leg_costs(
        value,
        rate,
        decision.currency,
        costs.exchanges[decision.exchange],
        costs,
        "sell",
    )
    return _cents(_cents(value / (rate or 1.0)) - sell.total)


def _dividends(
    dividends: Sequence[Dividend],
    shares: int,
    after: date,
    until: date,
    rate: float | None,
) -> float:
    """Dividends going ex after `after` up to `until`, times `shares`, in EUR."""
    paid = round(
        sum(d.amount for d in dividends if after < d.date <= until), PRICE_DECIMALS
    )
    return _cents(shares * paid / (rate or 1.0))


def _current_rate(
    decision: Decision, rates: Sequence[Candle], day: date, fallback: float | None
) -> tuple[float | None, str | None]:
    """The rate of `day`, or the newest cached before it with a warning."""
    try:
        return rate_on(decision.currency, rates, day), None
    except Unscorable:
        before = [c for c in rates if c.date <= day]
        if not before:
            return fallback, (
                f"There's no cached EUR/{decision.currency} rate on or before "
                f"{day.isoformat()}; valued at the fill's rate."
            )
        newest = before[-1]
        return round(newest.close, RATE_DECIMALS), (
            f"The EUR/{decision.currency} rate for {day.isoformat()} isn't "
            f"available yet; valued at {newest.date.isoformat()}'s rate."
        )


def _staleness(
    decision: Decision, last: date, now: datetime, not_updated: AbstractSet[str]
) -> str | None:
    newest = newest_closed_session(decision.exchange, now)
    if last >= newest:
        return None
    ticker = decision.ticker
    if ticker in not_updated:
        behind = len(sessions_in_range(decision.exchange, last, newest)) - 1
        return (
            f"The data provider couldn't bring {ticker} up to date, so it's "
            f"valued at its close of {last.isoformat()}, {behind} "
            f"session{'s' if behind != 1 else ''} behind."
        )
    return (
        f"The provider hasn't published {newest.isoformat()}'s session for "
        f"{ticker} yet; it's valued at its close of {last.isoformat()}."
    )


class _Valued(NamedTuple):
    holding: Holding
    risk: float


def _hold(
    decision: Decision,
    plan: TradePlan,
    fill: Fill,
    market: Market,
    costs: Costs,
    now: datetime,
    not_updated: AbstractSet[str],
) -> _Valued:
    candles = market.candles.get(decision.ticker, [])
    warnings: list[str] = []
    if candles:
        last_session = candles[-1].date
        close = round(candles[-1].close, PRICE_DECIMALS)
        stale = _staleness(decision, last_session, now, not_updated)
    else:
        # The stock cache is safe to delete; an unreachable provider can't
        # refill it, so the fill is the newest price known.
        last_session, close = fill.session, fill.price
        stale = (
            f"Nothing is cached for {decision.ticker} and it couldn't be "
            f"brought up to date; it's valued at its fill price of "
            f"{fill.session.isoformat()}."
        )
    if stale is not None:
        warnings.append(stale)
    rate: float | None = None
    pair = fx_pair(decision.currency)
    if pair is not None:
        rate, warning = _current_rate(
            decision, market.candles.get(pair, []), last_session, fill.rate
        )
        if warning is not None:
            warnings.append(warning)
    value = _sale(fill.shares, close, rate, decision, costs)
    dividends = _dividends(
        market.dividends.get(decision.ticker, []),
        fill.shares,
        fill.session,
        last_session,
        rate,
    )
    atr_block = calculate_indicators(list(candles)).atr if candles else None
    atr = atr_block.value if atr_block is not None else None

    def in_atr(distance: float) -> float | None:
        return round(distance / atr, DISTANCE_DECIMALS) if atr else None

    holding = Holding(
        id=decision.id,
        ticker=decision.ticker,
        currency=decision.currency,
        shares=fill.shares,
        entry=fill.price,
        stop=plan.stop,
        target=plan.target,
        last_close=close,
        based_on=last_session,
        rate=rate,
        value=value,
        dividends=dividends,
        unrealised_pnl=_cents(value + dividends - fill.value - fill.costs.total),
        to_stop=round((close - plan.stop) / close, DISTANCE_DECIMALS),
        to_stop_atr=in_atr(close - plan.stop),
        to_target=round((plan.target - close) / close, DISTANCE_DECIMALS),
        to_target_atr=in_atr(plan.target - close),
        sessions_held=max(
            len(sessions_in_range(decision.exchange, fill.session, last_session)) - 1,
            0,
        ),
        horizon_ends=decision.horizon_ends,
        is_current=stale is None,
        warning=" ".join(warnings) or None,
    )
    risk = _cents(value - _sale(fill.shares, plan.stop, rate, decision, costs))
    return _Valued(holding, risk)


def _closed(decisions: Mapping[int, Decision], exits: Sequence[Exit]) -> ClosedSummary:
    def total(reason: ExitReason) -> ReasonTotal:
        mine = [e for e in exits if e.reason == reason]
        return ReasonTotal(count=len(mine), pnl=_cents(sum(e.pnl for e in mine)))

    newest = sorted(exits, key=lambda e: (e.session, e.decision_id), reverse=True)
    return ClosedSummary(
        count=len(exits),
        realised_pnl=_cents(sum(e.pnl for e in exits)),
        by_reason=ByReason(
            stop=total("stop"), target=total("target"), horizon=total("horizon")
        ),
        recent=[
            RecentExit(
                id=e.decision_id,
                ticker=decisions[e.decision_id].ticker,
                session=e.session,
                reason=e.reason,
                pnl=e.pnl,
            )
            for e in newest[:MAX_RECENT]
        ],
    )


class _Held(NamedTuple):
    """A filled position, for valuing the portfolio on past sessions."""

    decision: Decision
    fill: Fill
    exit: Exit | None


def _value_on(
    day: date,
    held: Sequence[_Held],
    starting_capital: float,
    market: Market,
    costs: Costs,
    sessions: Mapping[str, set[date]],
) -> float | None:
    """
    The portfolio's value at `day`'s close: cash after its fills and exits
    plus each position then open at its latest close on or before the day
    and the day's rate, with its dividends so far. `None` when a session's
    close or a rate isn't cached.

    """
    total = starting_capital
    for decision, fill, exit in held:
        if fill.session > day:
            continue
        total -= fill.value + fill.costs.total
        if exit is not None and exit.session <= day:
            total += exit.value - exit.costs.total + exit.dividends
            continue
        candles = market.candles[decision.ticker]
        index = bisect_right(candles, day, key=lambda c: c.date)
        if not index or (
            day in sessions[decision.exchange] and candles[index - 1].date != day
        ):
            return None
        rate: float | None = None
        pair = fx_pair(decision.currency)
        if pair is not None:
            try:
                rate = rate_on(decision.currency, market.candles.get(pair, []), day)
            except Unscorable:
                return None
        close = round(candles[index - 1].close, PRICE_DECIMALS)
        total += _sale(fill.shares, close, rate, decision, costs)
        total += _dividends(
            market.dividends.get(decision.ticker, []),
            fill.shares,
            fill.session,
            day,
            rate,
        )
    return _cents(total)


class _Watch(NamedTuple):
    """The peak and the kill switch over the valuation days since a reset."""

    peak: float
    peak_session: date | None
    skipped: int
    tripped_on: date | None
    drawdown_at_trip: float | None
    newest: date | None


def _watch(
    held: Sequence[_Held],
    starting_capital: float,
    since: Reset | None,
    threshold: float,
    market: Market,
    costs: Costs,
    now: datetime,
) -> _Watch:
    """
    The highest value at a valuation day's close (every session since the
    first fill of an exchange with a position then open, through its exit
    session) after `since`'s session, and at least the value at `since`
    (the starting capital without one); its session;
    how many of those days were skipped; the first day whose value is at
    or below the running peak less `threshold`, with its drawdown; and the
    newest day valued.

    """
    sessions: dict[str, set[date]] = {}
    for decision, fill, exit in held:
        last = (
            exit.session
            if exit is not None
            else newest_closed_session(decision.exchange, now)
        )
        sessions.setdefault(decision.exchange, set()).update(
            sessions_in_range(decision.exchange, fill.session, last)
        )
    days = {day for exchange_days in sessions.values() for day in exchange_days}
    after = since.session if since is not None else date.min
    peak = since.total_value if since is not None else starting_capital
    peak_session = since.session if since is not None else None
    newest = peak_session
    skipped = 0
    tripped_on: date | None = None
    drawdown_at_trip: float | None = None
    for day in sorted(d for d in days if d > after):
        value = _value_on(day, held, starting_capital, market, costs, sessions)
        if value is None:
            skipped += 1
            continue
        newest = day
        if value > peak:
            peak, peak_session = value, day
        if tripped_on is None and value <= _cents(peak * (1 - threshold)):
            tripped_on = day
            drawdown_at_trip = round((peak - value) / peak, DISTANCE_DECIMALS)
    return _Watch(peak, peak_session, skipped, tripped_on, drawdown_at_trip, newest)


def calculate_portfolio(
    decisions: Sequence[Decision],
    positions: Positions,
    market: Market,
    costs: Costs,
    limits: Limits,
    now: datetime,
    not_updated: AbstractSet[str] = frozenset(),
    reset: Reset | None = None,
) -> Portfolio:
    """
    The portfolio from the journal's `decisions` and `positions` and the
    cached `market`, as of `now`, held to the operator's `limits`; the
    peak and the kill switch run from the latest `reset`. `not_updated`
    names the tickers whose cache couldn't be brought up to date, for the
    warnings.

    """
    starting_capital = limits.starting_capital
    fills, expiries, exits = positions
    by_id = {d.id: d for d in decisions}
    held: list[_Valued] = []
    waiting: list[WaitingPlan] = []
    waiting_risk = 0.0
    for decision in decisions:
        plan = decision.plan
        if plan is None or decision.id in exits or decision.id in expiries:
            continue
        fill = fills.get(decision.id)
        if fill is not None:
            held.append(_hold(decision, plan, fill, market, costs, now, not_updated))
            continue
        waiting.append(
            WaitingPlan(
                id=decision.id,
                ticker=decision.ticker,
                entry=plan.entry,
                stop=plan.stop,
                target=plan.target,
                shares=plan.shares,
                reserved=_cents(plan.position_value + plan.costs.buy.total),
                valid_until=order_window(decision, plan)[1],
            )
        )
        waiting_risk += plan.risk

    holdings = [v.holding for v in held]
    exit_lines = [exits[i] for i in sorted(exits)]
    cash = _cents(
        starting_capital
        - sum(f.value + f.costs.total for f in fills.values())
        + sum(e.value - e.costs.total + e.dividends for e in exit_lines)
        + sum(h.dividends for h in holdings)
    )
    reserved = _cents(sum(w.reserved for w in waiting))
    open_value = _cents(sum(h.value for h in holdings))
    closed = _closed(by_id, exit_lines)
    total_value = _cents(cash + open_value)
    based_on = max((h.based_on for h in holdings), default=None)
    watch = _watch(
        [_Held(by_id[i], f, exits.get(i)) for i, f in sorted(fills.items())],
        starting_capital,
        reset,
        limits.kill_switch_drawdown,
        market,
        costs,
        now,
    )
    peak, peak_session = watch.peak, watch.peak_session
    if total_value > peak:
        peak, peak_session = total_value, based_on
    drawdown = round((peak - total_value) / peak, DISTANCE_DECIMALS)
    totals = Totals(
        starting_capital=starting_capital,
        cash=cash,
        reserved=reserved,
        free_cash=_cents(cash - reserved),
        open_value=open_value,
        total_value=total_value,
        realised_pnl=closed.realised_pnl,
        unrealised_pnl=_cents(sum(h.unrealised_pnl for h in holdings)),
        peak=peak,
        peak_session=peak_session,
        drawdown=drawdown,
        open_risk=_cents(sum(v.risk for v in held)),
        waiting_risk=_cents(waiting_risk),
        open_positions=len(holdings),
        waiting_plans=len(waiting),
    )
    return Portfolio(
        based_on=based_on,
        totals=totals,
        open=holdings,
        waiting=waiting,
        closed=closed,
        stale=list(dict.fromkeys(h.ticker for h in holdings if not h.is_current)),
        skipped_days=watch.skipped,
        kill_switch=KillSwitch(
            tripped=watch.tripped_on is not None,
            tripped_on=watch.tripped_on,
            drawdown_at_trip=watch.drawdown_at_trip,
            threshold=limits.kill_switch_drawdown,
            drawdown=drawdown,
            last_reset=(
                LastReset(
                    reset_at=reset.reset_at,
                    session=reset.session,
                    drawdown=reset.drawdown,
                    note=reset.note,
                )
                if reset is not None
                else None
            ),
        ),
        limits=PositionLimit(
            max_open_positions=limits.max_open_positions,
            open_positions=len(holdings),
            waiting_plans=len(waiting),
            room=max(limits.max_open_positions - len(holdings) - len(waiting), 0),
        ),
        newest_valuation=watch.newest,
    )


def refuse_beyond_free_cash(plan: TradePlan, totals: Totals) -> None:
    """
    `PlanRefused` when the plan's position value plus buy costs is more
    than the free cash, naming the free cash, the reservations and by how
    much it misses.

    """
    buy = plan.costs.buy.total
    need = _cents(plan.position_value + buy)
    if need <= totals.free_cash:
        return
    raise PlanRefused(
        f"The plan needs €{need:,.2f} (€{plan.position_value:,.2f} of shares "
        f"plus €{buy:,.2f} buy costs), more than the free cash of "
        f"€{totals.free_cash:,.2f} (cash €{totals.cash:,.2f} less "
        f"€{totals.reserved:,.2f} reserved for {totals.waiting_plans} waiting "
        f"plan{'s' if totals.waiting_plans != 1 else ''}): it misses by "
        f"€{_cents(need - totals.free_cash):,.2f}. Wait for a waiting plan to "
        "fill or expire, or a position to close."
    )


def refuse_when_tripped(kill_switch: KillSwitch) -> None:
    """
    `PlanRefused` when the kill switch has tripped, naming the session, the
    drawdown and the threshold, and that only the operator can reset it.

    """
    if not kill_switch.tripped or kill_switch.tripped_on is None:
        return
    raise PlanRefused(
        f"The kill switch tripped on {kill_switch.tripped_on.isoformat()} at a "
        f"drawdown of {kill_switch.drawdown_at_trip} below the portfolio's "
        f"peak (threshold {kill_switch.threshold}), so no new plan is made. "
        "Open positions and waiting plans run on. Only the operator can reset "
        "it, with `task reset-kill-switch` outside the analysis session; "
        "meanwhile, record nothing for a buy and say why in the report."
    )


def refuse_beyond_open_positions(limits: PositionLimit) -> None:
    """
    `PlanRefused` when the open positions and waiting plans together leave
    no room under the limit, naming the limit and both counts.

    """
    if limits.room > 0:
        return
    taken = limits.open_positions + limits.waiting_plans
    raise PlanRefused(
        f"The open-position limit of {limits.max_open_positions} is reached: "
        f"{limits.open_positions} position{'s are' if limits.open_positions != 1 else ' is'} "
        f"open and {limits.waiting_plans} plan"
        f"{'s wait' if limits.waiting_plans != 1 else ' waits'} for their entry, "
        f"{taken} together. Wait for a position to close or a waiting plan to "
        "expire."
    )
