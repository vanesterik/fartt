"""
The integration tests' market data: a checked-in snapshot of the tickers they
read, and the moment the server's clock is pinned to (`FARTT_NOW`) while they
run on it. At that moment the snapshot holds every exchange's newest closed
session, so the server finds each cache current and never calls Yahoo.

Rebuild it with `task test:snapshot` after moving `SNAPSHOT_NOW` forward (to a
moment after the last session you want in it, before the next one closes), and
update the tests whose expectations are dates or values from the snapshot.
"""

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from fartt.data import YFinanceProvider
from fartt.stock_cache import StockCache

# Every ticker an integration test reads from a filled cache.
SNAPSHOT_TICKERS = ("ASML.AS", "AAPL", "IWDA.AS", "EURUSD=X")
# Thursday noon UTC: Wednesday 2026-11-04 is the newest closed session on
# every exchange in the snapshot, Euronext, the US and the 24/5 rates alike.
SNAPSHOT_NOW = datetime(2026, 11, 5, 12, tzinfo=UTC)
SNAPSHOT_DIR = Path(__file__).parent / "snapshot"

# An HTTPS proxy nothing listens on: the provider fails fast instead of
# reaching Yahoo.
UNREACHABLE = "http://127.0.0.1:9"
PROVIDER_DOWN = {name: UNREACHABLE for name in ("HTTPS_PROXY", "https_proxy")}
# The clock of a server on the snapshot. With Yahoo reachable, a test can
# refill a cut snapshot from Yahoo up to the snapshot's moment and no further.
PINNED = {"FARTT_NOW": SNAPSHOT_NOW.isoformat()}
# The environment of a server on the snapshot: its clock pinned and Yahoo out
# of reach, so a test that would fetch anything fails instead of going online.
OFFLINE = {**PINNED, **PROVIDER_DOWN}


def build(target: Path = SNAPSHOT_DIR) -> None:
    """Download SNAPSHOT_TICKERS as they stood at SNAPSHOT_NOW into `target`."""

    def clock() -> datetime:
        return SNAPSHOT_NOW

    shutil.rmtree(target, ignore_errors=True)
    cache = StockCache(YFinanceProvider(clock=clock), target, clock=clock)
    for ticker in SNAPSHOT_TICKERS:
        cache.update(ticker)
        print(f"{ticker}: {len(cache.latest(ticker, 1_000_000))} sessions")
    for lock in target.glob("stocks/*/.lock"):
        lock.unlink()


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else SNAPSHOT_DIR)
