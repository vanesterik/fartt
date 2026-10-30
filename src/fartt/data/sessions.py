from datetime import date, datetime, timedelta
from functools import cache

import exchange_calendars as xcals

from fartt.data.provider import UnsupportedExchange

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


def calendar_for(ticker: str) -> str:
    """The exchange_calendars code for a ticker, from its suffix."""
    ticker = ticker.upper()
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
