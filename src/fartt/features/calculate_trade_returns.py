import numpy as np
import numpy.typing as npt


def calculate_trade_returns(
    candle_returns: npt.ArrayLike,
    predicted_candle_returns: npt.ArrayLike | None = None,
    initial_capital: float = 500,
    cost_pct: float = 0.0005,
    slippage_pct: float = 0.0,
    threshold: float | None = None,
    max_holding_period: int | None = None,
    stop_loss_pct: float | None = None,
) -> tuple[list[float], list[float]]:
    """
    Backtest a long-only trading strategy over a series of signed
    percent-change candle_returns (see `calculate_candle_returns`), deciding entry and
    exit off `predicted_candle_returns` while realized profit/loss is always
    computed from the actual `candle_returns`. Passing `predicted_candle_returns`
    lets a signal known one tick in advance act *before* that tick's move
    happens -- e.g. an upper-bound "perfect foresight" backtest via
    `predicted_candle_returns = candle_return_series.shift(-1)`, so `predicted[i]`
    equals the actual `candle_returns[i + 1]` -- as opposed to omitting it, which
    reduces to a purely reactive rule that only ever acts on a move already
    reflected in that same candle's `candle_returns` value.

    A position is opened while flat when the signal clears `threshold`, and
    closed while in a position when the signal drops below `-threshold`,
    after `max_holding_period` candles, or once the price has moved against
    the position by more than `stop_loss_pct` since entry -- whichever comes
    first. Because a trade can span multiple candles, its holding-period
    return is reconstructed by compounding the actual candle_returns between
    entry and exit, then reduced by `cost_pct` and `slippage_pct` applied on
    both the entry and exit leg.

    Parameters
    ----------
    - candle_returns (npt.ArrayLike): Signed percent-change candle_returns, one per
      candle (e.g. `calculate_candle_returns`'s `Return` column) -- the actual
      price path, always used to compute realized return. A leading `NaN`
      (no prior candle) is treated as no signal.
    - predicted_candle_returns (Optional[npt.ArrayLike]): Signal used to decide
      entry/exit, same length as `candle_returns`. If not passed, `candle_returns`
      itself is used as the signal (today's reactive behavior). A `NaN`
      (e.g. the trailing value of a `.shift(-1)` series) is treated as no
      signal.
    - initial_capital (float): Starting capital used to compound `profits`
      across sequential trades.
    - cost_pct (float): Trading cost percentage charged on each leg (entry
      and exit) of a trade. Defaults to 0.05%, roughly Interactive Brokers'
      tiered commission on Euronext.
    - slippage_pct (float): Additional one-way haircut applied on each leg,
      on top of `cost_pct`.
    - threshold (Optional[float]): Minimum absolute signal value required to
      open/close a position. Defaults to `2 * (cost_pct + slippage_pct)`,
      the round-trip cost, so a trade is never taken below break-even.
    - max_holding_period (Optional[int]): Maximum number of candles a
      position may stay open before being force-closed regardless of
      signal, capping how far an adverse move can run against an open
      trade. If not passed, a position stays open until the close signal
      fires (or the data ends).
    - stop_loss_pct (Optional[float]): Maximum adverse price move, as a
      positive fraction, tolerated since entry before a position is
      force-closed -- based on raw price movement (compounded candle_returns),
      before `cost_pct`/`slippage_pct` are applied. If not passed, no
      stop-loss is applied.

    Returns
    -------
    - Tuple[List[float], List[float]]: `(returns, profits)`. `returns` are
      each trade's net return as a fraction, independent of capital and
      trade order. `profits` are each trade's currency profit, compounding
      `initial_capital` across trades in order.

    """
    values = np.asarray(candle_returns, dtype=np.float64)
    signal_values = (
        np.asarray(predicted_candle_returns, dtype=np.float64)
        if predicted_candle_returns is not None
        else values
    )
    if signal_values.shape != values.shape:
        raise ValueError(
            f"predicted_candle_returns must be the same length as candle_returns "
            f"({len(signal_values)} != {len(values)})."
        )

    if threshold is None:
        threshold = 2 * (cost_pct + slippage_pct)

    cost_factor = (1 - cost_pct - slippage_pct) ** 2
    trade_boundaries = _find_trade_boundaries(
        values,
        signal_values,
        threshold,
        max_holding_period,
        stop_loss_pct,
    )
    returns = [
        _calculate_net_return(values, entry_index, exit_index, cost_factor)
        for entry_index, exit_index in trade_boundaries
    ]

    profits: list[float] = []
    capital = initial_capital
    for net_return in returns:
        profit = capital * net_return
        profits.append(profit)
        capital += profit

    return returns, profits


def _find_trade_boundaries(
    values: npt.NDArray[np.float64],
    signal_values: npt.NDArray[np.float64],
    threshold: float,
    max_holding_period: int | None = None,
    stop_loss_pct: float | None = None,
) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    is_open = False
    entry_index = 0
    running_return = 0.0

    for i, signal in enumerate(signal_values):
        if np.isnan(signal):
            continue

        if not is_open and signal > threshold:
            is_open = True
            entry_index = i
            running_return = 0.0
        elif is_open:
            candle_return = values[i]
            if not np.isnan(candle_return):
                running_return = (1 + running_return) * (1 + candle_return) - 1

            if (
                signal < -threshold
                or (
                    max_holding_period is not None
                    and i - entry_index >= max_holding_period
                )
                or (stop_loss_pct is not None and running_return <= -stop_loss_pct)
            ):
                boundaries.append((entry_index, i))
                is_open = False

    if is_open:
        boundaries.append((entry_index, len(values) - 1))

    return boundaries


def _calculate_net_return(
    values: npt.NDArray[np.float64],
    entry_index: int,
    exit_index: int,
    cost_factor: float,
) -> float:
    gross_return = float(np.prod(1 + values[entry_index + 1 : exit_index + 1]))
    return gross_return * cost_factor - 1
