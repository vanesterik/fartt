from datetime import date
from itertools import groupby

from fartt.data import Candle, last_session_of_week


def _week(candle: Candle) -> tuple[int, int]:
    year, week, _ = candle.date.isocalendar()
    return year, week


def _aggregate(sessions: list[Candle]) -> Candle:
    return Candle(
        sessions[-1].date,
        sessions[0].open,
        max(c.high for c in sessions),
        min(c.low for c in sessions),
        sessions[-1].close,
        sum(c.volume for c in sessions),
    )


def weekly_candles(
    candles: list[Candle], exchange: str
) -> tuple[list[Candle], date | None]:
    """
    Weekly candles (ISO weeks, oldest first) from daily `candles` (oldest
    first): the first open, highest high, lowest low, last close and summed
    volume, dated by the week's last session. A holiday week simply has fewer
    sessions. Closed weeks only: the newest week counts once its last
    scheduled session is among the candles, or when the calendar can't place
    it. Returns the weeks, and the date of the newest cached session when its
    week was left out (else `None`).

    """
    weeks = [list(group) for _, group in groupby(candles, key=_week)]
    if not weeks:
        return [], None
    newest = weeks[-1]
    last = last_session_of_week(exchange, newest[-1].date)
    unclosed = None
    if last is not None and last > newest[-1].date:
        unclosed = newest[-1].date
        weeks.pop()
    return [_aggregate(week) for week in weeks], unclosed
