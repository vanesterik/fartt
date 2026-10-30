import csv
import fcntl
import json
import os
from collections import deque
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import NamedTuple

from fartt.data import (
    Candle,
    DataProvider,
    Dividend,
    History,
    Listing,
    Split,
    calendar_for,
    newest_closed_session,
    normalise_ticker,
)

CANDLE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
EVENT_COLUMNS = ["Date", "Type", "Value"]
DIVIDEND = "dividend"
SPLIT = "split"


class UpdateResult(NamedTuple):
    new_candles: int  # Candles written by this update
    rebuilt: str | None  # Why the whole history was rewritten, if it was
    fetched: bool  # Whether the provider was asked at all


def _now() -> datetime:
    return datetime.now(UTC)


class StockCache:
    """
    Each ticker's split-adjusted daily candles, dividends, splits and
    listing, cached under `assets_dir/stocks/<TICKER>/` and kept up to date
    through a `DataProvider`:

    - `candles.csv`: `Date,Open,High,Low,Close,Volume`, ascending
    - `events.csv`: `Date,Type,Value`, `Type` being `dividend` or `split`
    - `listing.json`: the ticker's `Listing`

    Updates hold an exclusive lock on the directory's sidecar `.lock` file
    and reads a shared one, so readers never see a half-written file.
    `clock` returns the current time (UTC) and decides which sessions have
    closed.

    """

    def __init__(
        self,
        provider: DataProvider,
        assets_dir: Path,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._provider = provider
        self._root = assets_dir / "stocks"
        self._clock = clock

    def directory(self, ticker: str) -> Path:
        return self._root / normalise_ticker(ticker)

    def newest_closed_session(self, ticker: str) -> date:
        return newest_closed_session(calendar_for(ticker), self._clock())

    def update(self, ticker: str) -> UpdateResult:
        """
        Bring the ticker's cache up to date with the provider. A ticker the
        cache hasn't seen is fetched in full. Provider errors propagate, and
        leave the cached files as they were.

        """
        ticker = normalise_ticker(ticker)
        calendar_for(ticker)  # an unsupported exchange fails before any I/O
        with self._lock(ticker, fcntl.LOCK_EX):
            listing = self._provider.listing(ticker)
            history = self._provider.history(ticker, None)
            self._write_all(ticker, listing, history)
            return UpdateResult(len(history.candles), None, True)

    def latest(self, ticker: str, count: int) -> list[Candle]:
        """Up to `count` of the newest cached candles, oldest first."""
        with self._read_lock(ticker) as directory:
            path = directory / "candles.csv"
            if not path.exists():
                return []
            with path.open(newline="", encoding="utf-8") as file:
                rows = deque(csv.DictReader(file), maxlen=count)
        return [_candle(row) for row in rows]

    def events(self, ticker: str) -> tuple[list[Dividend], list[Split]]:
        """The ticker's cached dividends and splits, each ascending."""
        with self._read_lock(ticker) as directory:
            path = directory / "events.csv"
            if not path.exists():
                return [], []
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        dividends = [
            Dividend(date.fromisoformat(row["Date"]), float(row["Value"]))
            for row in rows
            if row["Type"] == DIVIDEND
        ]
        splits = [
            Split(date.fromisoformat(row["Date"]), float(row["Value"]))
            for row in rows
            if row["Type"] == SPLIT
        ]
        return dividends, splits

    def listing(self, ticker: str) -> Listing | None:
        """The ticker's cached listing, or `None` before its first update."""
        with self._read_lock(ticker) as directory:
            path = directory / "listing.json"
            if not path.exists():
                return None
            fields = json.loads(path.read_text(encoding="utf-8"))
        return Listing(**fields)

    @contextmanager
    def _lock(self, ticker: str, operation: int) -> Generator[Path]:
        directory = self.directory(ticker)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / ".lock", "a") as lock:
            fcntl.flock(lock, operation)
            yield directory

    @contextmanager
    def _read_lock(self, ticker: str) -> Generator[Path]:
        # Check files only once the lock is held, so a read during a first
        # fill waits for it. Without a directory no update has ever started
        # (updates create it first), so there's nothing to wait for, and a
        # read doesn't leave an empty directory behind.
        directory = self.directory(ticker)
        if not directory.exists():
            yield directory
            return
        with self._lock(ticker, fcntl.LOCK_SH):
            yield directory

    def _write_all(self, ticker: str, listing: Listing, history: History) -> None:
        # Each file goes to a temporary sibling first and is swapped in
        # whole, so a failure part-way leaves the previous version intact.
        directory = self.directory(ticker)
        _replace(
            directory / "candles.csv", CANDLE_COLUMNS, map(_candle_row, history.candles)
        )
        _replace(directory / "events.csv", EVENT_COLUMNS, _event_rows(history))
        temporary = directory / "listing.json.tmp"
        temporary.write_text(json.dumps(listing._asdict()), encoding="utf-8")
        os.replace(temporary, directory / "listing.json")


def _candle(row: dict[str, str]) -> Candle:
    return Candle(
        date=date.fromisoformat(row["Date"]),
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=float(row["Close"]),
        volume=float(row["Volume"]),
    )


def _candle_row(candle: Candle) -> list[str]:
    return [candle.date.isoformat(), *(str(value) for value in candle[1:])]


def _event_rows(history: History) -> list[list[str]]:
    rows = [[d.date.isoformat(), DIVIDEND, str(d.amount)] for d in history.dividends]
    rows += [[s.date.isoformat(), SPLIT, str(s.ratio)] for s in history.splits]
    return sorted(rows)


def _replace(path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)
