import re
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from functools import cache
from typing import NamedTuple

import exchange_calendars as xcals
from pandas.errors import Pandas4Warning

from fartt.data.provider import UnknownTicker, UnsupportedExchange

# How long after its scheduled close a session counts as closed, so the
# provider has finalised the day's candle.
CLOSE_MARGIN = timedelta(minutes=30)

# Ticker suffix (Yahoo's format) to exchange_calendars code. No suffix is a
# US listing; NYSE's calendar covers Nasdaq's sessions too.
SUFFIX_CALENDARS = {
    "": "XNYS",
    ".AS": "XAMS",
    ".PA": "XPAR",
    ".BR": "XBRU",
    ".LS": "XLIS",
    ".MI": "XMIL",
    ".IR": "XDUB",
    ".OL": "XOSL",
}


# Letters, digits, dots and dashes: every ticker in the universe, and
# nothing that could step outside a cache directory.
_TICKER = re.compile(r"[A-Z0-9.\-]+")


def normalise_ticker(ticker: str) -> str:
    """
    A ticker stripped and upper-cased, in Yahoo's format. Raises
    `UnknownTicker` for anything that can't be one (empty, or holding a
    character other than a letter, digit, `.` or `-`), before it reaches a
    request or a file path.

    """
    normalised = ticker.strip().upper()
    if not _TICKER.fullmatch(normalised):
        raise UnknownTicker(
            f"'{ticker}' isn't a ticker. Use Yahoo's format: AAPL for a US "
            "listing, ASML.AS for Euronext Amsterdam."
        )
    return normalised


def calendar_for(ticker: str) -> str:
    """The exchange_calendars code for a ticker, from its suffix."""
    ticker = normalise_ticker(ticker)
    dot = ticker.rfind(".")
    suffix = ticker[dot:] if dot != -1 else ""
    try:
        return SUFFIX_CALENDARS[suffix]
    except KeyError:
        supported = ", ".join(s for s in SUFFIX_CALENDARS if s)
        raise UnsupportedExchange(
            f"'{ticker}' is on an exchange outside the universe. Supported: US "
            f"tickers without a suffix (AAPL), and Euronext suffixes {supported}."
        ) from None


@contextmanager
def _without_pandas_deprecations() -> Generator[None]:
    # exchange_calendars (4.13.2, the latest) reads Series.values and
    # DatetimeIndex.values on time-zone-aware data, which pandas 3.1
    # deprecates, when it builds a calendar and when it first builds its
    # table of trading minutes. The warning is theirs and says nothing about
    # fartt, so it's silenced around those two calls only; drop this once a
    # release reads the values another way.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Pandas4Warning)
        yield


@cache
def exchange_calendar(exchange: str) -> xcals.ExchangeCalendar:
    """The exchange_calendars calendar for an exchange code, built once."""
    # get_calendar builds the whole session table, which is slow.
    with _without_pandas_deprecations():
        return xcals.get_calendar(exchange)


def newest_closed_session(exchange: str, now: datetime) -> date:
    """
    The most recent session of `exchange` that closed (plus `CLOSE_MARGIN`)
    at or before `now`, which must be timezone-aware. Weekends and the
    exchange's holidays are never sessions; early closes count.

    """
    calendar = exchange_calendar(exchange)
    local_date = now.astimezone(calendar.tz).date()
    session = calendar.date_to_session(local_date, direction="previous")
    while calendar.session_close(session) + CLOSE_MARGIN > now:
        session = calendar.previous_session(session)
    return session.date()


def missing_sessions(exchange: str, dates: Sequence[date]) -> list[date]:
    """
    Sessions of `exchange` between the earliest and latest of `dates` that
    aren't in `dates`, ascending. Weekends and the exchange's holidays are
    never missing. Dates before the calendar's first session (it reaches
    back about 20 years) aren't checked.

    """
    if len(dates) < 2:
        return []
    calendar = exchange_calendar(exchange)
    start = max(min(dates), calendar.first_session.date())
    end = max(dates)
    if start > end:
        return []
    present = set(dates)
    return [
        session.date()
        for session in calendar.sessions_in_range(start, end)
        if session.date() not in present
    ]


def last_session_of_week(exchange: str, day: date) -> date | None:
    """
    The last scheduled session of `exchange` in the ISO week (Monday to
    Sunday) that holds `day`. `None` when that week has no session, or when
    it lies outside the calendar's range (about 20 years back, a year ahead).

    """
    monday = day - timedelta(days=day.weekday())
    try:
        sessions = exchange_calendar(exchange).sessions_in_range(
            monday, monday + timedelta(days=6)
        )
    except ValueError:  # exchange_calendars' DateOutOfBounds
        return None
    return sessions[-1].date() if len(sessions) else None


def session_after(exchange: str, day: date, count: int) -> date:
    """
    The `count`-th session of `exchange` after `day` (a session itself),
    skipping weekends and the exchange's holidays.

    """
    return exchange_calendar(exchange).session_offset(day, count).date()


def exchange_date(exchange: str, moment: datetime) -> date:
    """The date `moment` (timezone-aware) falls on in the exchange's time zone."""
    return moment.astimezone(exchange_calendar(exchange).tz).date()


class MarketStatus(NamedTuple):
    open: bool
    next_open: datetime | None  # UTC; None while the exchange is open


def market_status(exchange: str, now: datetime) -> MarketStatus:
    """
    Whether `exchange` is trading at `now` (timezone-aware), and if not,
    when its next session opens. Weekends, holidays and early closes come
    from the exchange calendar.

    """
    calendar = exchange_calendar(exchange)
    minute = now.replace(second=0, microsecond=0)
    with _without_pandas_deprecations():
        is_open = calendar.is_open_on_minute(minute)
    if is_open:
        return MarketStatus(True, None)
    return MarketStatus(False, calendar.next_open(minute).astimezone(UTC))
