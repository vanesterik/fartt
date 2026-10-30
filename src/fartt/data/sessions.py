import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from functools import cache

import exchange_calendars as xcals

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


@cache
def _calendar(exchange: str) -> xcals.ExchangeCalendar:
    # get_calendar builds the whole session table, which is slow.
    return xcals.get_calendar(exchange)


def newest_closed_session(exchange: str, now: datetime) -> date:
    """
    The most recent session of `exchange` that closed (plus `CLOSE_MARGIN`)
    at or before `now`, which must be timezone-aware. Weekends and the
    exchange's holidays are never sessions; early closes count.

    """
    calendar = _calendar(exchange)
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
    calendar = _calendar(exchange)
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
