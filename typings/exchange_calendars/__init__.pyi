# Only what fartt calls. Sessions and closes are pandas Timestamps, which
# subclass datetime; they are declared as datetime so pandas' types stay out.
from collections.abc import Sequence
from datetime import date, datetime, tzinfo
from typing import Literal

class ExchangeCalendar:
    tz: tzinfo
    first_session: datetime
    def sessions_in_range(
        self, start: date | datetime | str, end: date | datetime | str
    ) -> Sequence[datetime]: ...
    def date_to_session(
        self,
        date: date | datetime | str,
        direction: Literal["next", "previous", "none"] = ...,
    ) -> datetime: ...
    def session_close(self, session_label: date | datetime | str) -> datetime: ...
    def previous_session(self, session: date | datetime | str) -> datetime: ...

def get_calendar(name: str) -> ExchangeCalendar: ...
