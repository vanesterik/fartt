"""
A trade plan: a long position from entry, stop and target, sized against the
operator's risk limits after costs.

The agent picks the prices; this module owns the arithmetic. Each leg (the
buy at the entry, a sale at the stop or at the target) costs commission,
slippage, currency conversion for a stock not in EUR, the exchange's tax or
fee on that side and its fixed fee per trade, all in EUR at one exchange
rate (units of the stock's currency per euro). The size is the largest whole
number of shares whose loss at the stop, every cost of both legs counted,
stays within `max_risk_per_position` of the portfolio and whose value stays
within `max_position_size` of it. A plan that can't hold a share, or whose
reward-to-risk after costs is below `min_reward_to_risk`, is refused with
`PlanRefused`, whose message says why. Pure functions, no I/O.

"""

import math
from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fartt.risk_config import Costs, ExchangeCosts, Limits

EUR = "EUR"
# How far above the last close a limit entry may sit, in ATRs: a limit far
# above the market is a market order in disguise.
MAX_ENTRY_ATRS = 1.0
DEFAULT_VALID_FOR = 5
MAX_VALID_FOR = 10

CENTS = 2
SHARE_DECIMALS = 4
REFUSAL_DECIMALS = 6

Side = Literal["buy", "sell"]
SizedBy = Literal["max_risk_per_position", "max_position_size"]


class PlanRefused(ValueError):
    """A plan that breaks a check or a limit; the message says which and by how much."""


class LegCosts(BaseModel):
    """What one trade costs, in EUR, each to the cent."""

    # The journal can't hold a NaN or infinity its readers could return.
    model_config = ConfigDict(allow_inf_nan=False)

    commission: float = Field(description="The broker's commission.")
    slippage: float = Field(description="The assumed slippage on the fill.")
    conversion: float = Field(
        description="Currency conversion to and from EUR; 0 for a EUR stock."
    )
    tax: float = Field(
        description="The tax on a purchase, or the fee on a sale (the SEC fee)."
    )
    fee: float = Field(description="The exchange's fixed fee per trade.")
    total: float = Field(description="The sum of the above.")


class PlanCosts(BaseModel):
    buy: LegCosts = Field(description="Buying at the entry.")
    sell_at_stop: LegCosts = Field(description="Selling at the stop.")
    sell_at_target: LegCosts = Field(description="Selling at the target.")


class Tariff(BaseModel):
    """The costs the plan was worked out with, from the risk config."""

    name: str = Field(description="The broker's tariff the costs come from.")
    currency_conversion: float = Field(
        description="Charged on each leg of a trade not in EUR."
    )
    slippage_per_leg: float = Field(description="Assumed on each leg.")
    exchange: ExchangeCosts = Field(description="The stock's exchange's costs.")


class TradePlan(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    ticker: str = Field(description="The ticker, in Yahoo's format.")
    currency: str = Field(description="The currency the prices are in.")
    exchange: str = Field(description="The exchange's calendar code, e.g. XAMS.")
    based_on: date = Field(description="The last closed session the plan uses.")
    last_close: float = Field(description="That session's close.")
    atr: float = Field(description="ATR 14 as of that session.")
    entry: float = Field(description="The limit buy price.")
    stop: float = Field(description="The stop, below the entry.")
    target: float = Field(description="The target, above the entry.")
    valid_for: int = Field(
        description="Sessions of the stock's exchange the limit order stays open."
    )
    rate: float | None = Field(
        description=(
            "The EUR/CCY rate used (units of the currency per euro); null for EUR."
        )
    )
    # Defaults for plans stored before these fields existed.
    rate_session: date | None = Field(
        default=None,
        description="The session of the rate's close; null for EUR.",
    )
    warning: str | None = Field(
        default=None,
        description=(
            "Set when the rate is from a session before based_on, saying why. "
            "The plan is still sized on it."
        ),
    )
    shares: int = Field(description="Whole shares to buy.")
    position_value: float = Field(description="shares × entry, in EUR.")
    # Default for plans stored before the portfolio (#145).
    free_cash: float | None = Field(
        default=None,
        description=(
            "The portfolio's free cash when the plan was made (cash less what "
            "waiting plans reserve), in EUR; the plan's position value plus "
            "buy costs fit within it."
        ),
    )
    portfolio_value: float = Field(
        description=(
            "The portfolio's total value the limits are shares of, in EUR: "
            "cash plus the open positions after the costs to close."
        )
    )
    position_share: float = Field(description="position_value / portfolio_value.")
    risk: float = Field(
        description=(
            "The loss in EUR if the stop is hit as planned, every cost of "
            "both legs counted."
        )
    )
    risk_share: float = Field(description="risk / portfolio_value.")
    reward: float = Field(
        description="The gain in EUR at the target, every cost of both legs counted."
    )
    reward_to_risk: float = Field(description="reward / risk, after costs.")
    reward_to_risk_before_costs: float = Field(
        description="(target − entry) / (entry − stop)."
    )
    sized_by: SizedBy = Field(description="The limit that set the size.")
    costs: PlanCosts = Field(description="Each leg's costs, itemised, in EUR.")
    limits: Limits = Field(description="The limits the plan was checked against.")
    tariff: Tariff = Field(description="The costs the plan was worked out with.")


class PlanSummary(BaseModel):
    """A stored plan in brief, as get_decisions lists it."""

    entry: float = Field(description="The limit buy price.")
    stop: float = Field(description="The stop, below the entry.")
    target: float = Field(description="The target, above the entry.")
    valid_for: int = Field(
        description="Sessions of the stock's exchange the limit order stays open."
    )
    shares: int = Field(description="Whole shares to buy.")
    position_value: float = Field(description="shares × entry, in EUR.")
    risk: float = Field(
        description="The loss in EUR if the stop is hit, every cost counted."
    )
    reward_to_risk: float = Field(description="reward / risk, after costs.")
    sized_by: SizedBy = Field(description="The limit that set the size.")
    based_on: date = Field(description="The last closed session the plan uses.")

    @classmethod
    def of(cls, plan: TradePlan) -> PlanSummary:
        return cls.model_validate(plan.model_dump(include=set(cls.model_fields)))


def _cents(value: float) -> float:
    return round(value, CENTS)


def _to_eur(amount: float, in_currency: str, currency: str, rate: float) -> float:
    """`amount` in `in_currency`, the stock's `currency` or EUR, in EUR."""
    if in_currency == EUR:
        return amount
    if in_currency == currency:
        return amount / rate
    raise PlanRefused(
        f"The tariff charges in {in_currency}, neither EUR nor the stock's "
        f"currency ({currency}), and the server has no rate for it. The "
        "operator must check the risk config."
    )


def leg_costs(
    value: float,
    rate: float | None,
    currency: str,
    exchange: ExchangeCosts,
    costs: Costs,
    side: Side,
) -> LegCosts:
    """
    The costs of a trade worth `value` in the stock's `currency`, at `rate`
    (units per euro, `None` for EUR), each to the cent.

    """
    r = rate or 1.0
    eur = value / r
    paid_in = exchange.commission_currency
    base = value if paid_in == currency else eur
    commission = max(base * exchange.commission, exchange.commission_min)
    if exchange.commission_max is not None:
        commission = min(commission, exchange.commission_max)
    rate_on_side = exchange.tax_on_buy if side == "buy" else exchange.fee_on_sell
    items = (
        _cents(_to_eur(commission, paid_in, currency, r)),
        _cents(eur * costs.slippage_per_leg),
        _cents(eur * costs.currency_conversion) if currency != EUR else 0.0,
        _cents(eur * rate_on_side),
        _cents(_to_eur(exchange.fee_per_trade, paid_in, currency, r)),
    )
    commission_eur, slippage, conversion, tax, fee = items
    return LegCosts(
        commission=commission_eur,
        slippage=slippage,
        conversion=conversion,
        tax=tax,
        fee=fee,
        total=_cents(sum(items)),
    )


def _check_prices(
    entry: float,
    stop: float,
    target: float,
    valid_for: int,
    last_close: float,
    atr: float | None,
) -> float:
    """The ATR, once the prices make a long plan; else `PlanRefused`."""
    for name, price in (("entry", entry), ("stop", stop), ("target", target)):
        # NaN and infinity pass every comparison below unnoticed.
        if not math.isfinite(price):
            raise PlanRefused(f"{name} must be a finite number; it is {price}.")
        if price <= 0:
            raise PlanRefused(f"{name} must be above 0; it is {price}.")
    if not 1 <= valid_for <= MAX_VALID_FOR:
        raise PlanRefused(
            f"valid_for must be 1 to {MAX_VALID_FOR} sessions; it is {valid_for}."
        )
    if stop >= entry:
        raise PlanRefused(
            f"The stop ({stop}) must be below the entry ({entry}) for a long plan."
        )
    if target <= entry:
        raise PlanRefused(
            f"The target ({target}) must be above the entry ({entry}) for a long plan."
        )
    if atr is None:
        raise PlanRefused(
            "There is no ATR for this ticker yet (too short a history), so the "
            "entry can't be checked against the last close. No plan can be made."
        )
    highest = last_close + MAX_ENTRY_ATRS * atr
    if round(entry - highest, 10) > 0:
        raise PlanRefused(
            f"The entry ({entry}) is more than 1 ATR ({atr}) above the last "
            f"close ({last_close}): at most {round(highest, 4)}. A limit that "
            "far above the market is a market order; wait for the price or "
            "lower the entry."
        )
    return atr


def _check_finite_at(target: float, *amounts: float) -> None:
    """`PlanRefused` unless every amount worked out at `target` is finite."""
    if not all(math.isfinite(amount) for amount in amounts):
        raise PlanRefused(
            f"The target ({target}) is too large: the plan's amounts at it "
            "aren't finite numbers. Set the target where the price can go."
        )


def _rate_warning(
    rate_session: date | None, rate_is_current: bool, based_on: date
) -> str | None:
    """Why the rate is from a session before the stock's, if it is."""
    if rate_session is None or rate_session >= based_on:
        return None
    why = (
        "the data provider couldn't bring it up to date"
        if not rate_is_current
        else f"the rate's {based_on.isoformat()} session hasn't closed yet "
        "(exchange rates close at midnight UTC)"
    )
    return (
        f"The exchange rate is from {rate_session.isoformat()}, before the "
        f"stock's last close ({based_on.isoformat()}): {why}. The plan is "
        "sized on it."
    )


def _largest(fits: Callable[[int], bool], high: int) -> int:
    """The largest `n` in 0..`high` with `fits(n)`, for `fits` true up to a point."""
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


def calculate_plan(
    *,
    ticker: str,
    currency: str,
    exchange_code: str,
    based_on: date,
    last_close: float,
    atr: float | None,
    entry: float,
    stop: float,
    target: float,
    valid_for: int,
    rate: float | None,
    rate_session: date | None,
    rate_is_current: bool,
    portfolio_value: float,
    limits: Limits,
    costs: Costs,
) -> TradePlan:
    """
    Size and check a long plan, or raise `PlanRefused` saying which check
    or limit it breaks. Prices are in the stock's `currency`; `rate` is
    EUR/`currency` (`None` for EUR), the close of `rate_session`, and
    `rate_is_current` says whether that is the pair's newest closed session.

    """
    atr = _check_prices(entry, stop, target, valid_for, last_close, atr)
    exchange = costs.exchanges[exchange_code]
    r = rate or 1.0

    def leg(price: float, shares: int, side: Side) -> LegCosts:
        return leg_costs(shares * price, rate, currency, exchange, costs, side)

    def risk(shares: int) -> float:
        loss = shares * (entry - stop) / r
        return loss + leg(entry, shares, "buy").total + leg(stop, shares, "sell").total

    max_risk = _cents(limits.max_risk_per_position * portfolio_value)
    max_position = _cents(limits.max_position_size * portfolio_value)

    def within_position(shares: int) -> bool:
        return _cents(shares * entry / r) <= max_position

    def within_risk(shares: int) -> bool:
        return _cents(risk(shares)) <= max_risk

    # A positive entry near zero allows more shares than a float can count.
    most = max_position * r / entry
    if not math.isfinite(most):
        raise PlanRefused(
            f"The entry ({entry}) is too small: the shares the position limit "
            "allows at it aren't a finite number. Set the entry where the "
            "price can go."
        )
    by_position = _largest(within_position, math.ceil(most) + 1)
    if within_risk(by_position):
        shares, sized_by = by_position, "max_position_size"
    else:
        shares, sized_by = _largest(within_risk, by_position), "max_risk_per_position"

    if shares == 0:
        broken: list[str] = []
        if not within_risk(1):
            broken.append(
                f"it would risk €{_cents(risk(1)):,.2f} after costs, above "
                f"max_risk_per_position's €{max_risk:,.2f} "
                f"({limits.max_risk_per_position:g} of €{portfolio_value:,.2f})"
            )
        if not within_position(1):
            broken.append(
                f"it costs €{_cents(entry / r):,.2f}, above max_position_size's "
                f"€{max_position:,.2f} ({limits.max_position_size:g} of "
                f"€{portfolio_value:,.2f})"
            )
        raise PlanRefused(
            "One share already breaks the limits: "
            + "; and ".join(broken)
            + ". The plan rounds down to 0 shares."
        )

    buy = leg(entry, shares, "buy")
    at_stop = leg(stop, shares, "sell")
    # The entry and stop are bounded by the last close; a finite target near
    # the float maximum isn't, and its amounts overflow to infinity and NaN,
    # which would pass the comparison below unnoticed.
    _check_finite_at(target, shares * target / r)
    at_target = leg(target, shares, "sell")
    loss = _cents(shares * (entry - stop) / r + buy.total + at_stop.total)
    gain = _cents(shares * (target - entry) / r - buy.total - at_target.total)
    before_costs = (target - entry) / (entry - stop)
    _check_finite_at(target, gain, gain / loss, before_costs)
    # Compared unrounded: the limit is the operator's, rounding is for display.
    # The refusal shows more decimals, so a ratio just short doesn't read as
    # the limit itself.
    ratio = gain / loss
    if ratio < limits.min_reward_to_risk:
        short = round(limits.min_reward_to_risk - ratio, REFUSAL_DECIMALS)
        raise PlanRefused(
            f"The plan's reward-to-risk after costs is "
            f"{round(ratio, REFUSAL_DECIMALS)} "
            f"(reward €{gain:,.2f}, risk €{loss:,.2f} for {shares} shares): "
            f"below min_reward_to_risk ({limits.min_reward_to_risk}); it falls "
            f"short by {short}. Move the target up, the stop up or the entry down."
        )

    position_value = _cents(shares * entry / r)
    return TradePlan(
        ticker=ticker,
        currency=currency,
        exchange=exchange_code,
        based_on=based_on,
        last_close=last_close,
        atr=atr,
        entry=entry,
        stop=stop,
        target=target,
        valid_for=valid_for,
        rate=rate,
        rate_session=rate_session,
        warning=_rate_warning(rate_session, rate_is_current, based_on),
        shares=shares,
        position_value=position_value,
        portfolio_value=portfolio_value,
        position_share=round(position_value / portfolio_value, SHARE_DECIMALS),
        risk=loss,
        risk_share=round(loss / portfolio_value, SHARE_DECIMALS),
        reward=gain,
        reward_to_risk=round(ratio, SHARE_DECIMALS),
        reward_to_risk_before_costs=round(before_costs, SHARE_DECIMALS),
        sized_by=sized_by,
        costs=PlanCosts(buy=buy, sell_at_stop=at_stop, sell_at_target=at_target),
        limits=limits,
        tariff=Tariff(
            name=costs.tariff,
            currency_conversion=costs.currency_conversion,
            slippage_per_leg=costs.slippage_per_leg,
            exchange=exchange,
        ),
    )
