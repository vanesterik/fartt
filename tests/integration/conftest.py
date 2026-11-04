"""
Shared live data for the integration tests. Downloading a ticker's full
history from the data provider takes a second or two, and most tests need the
same few tickers, so they're downloaded once per run and each test gets its
own copy. A test that checks the download itself starts from an empty
`tmp_path` instead.

Under pytest-xdist (`task test:network` runs `-n auto`) every worker has its
own session, so the download goes to a folder all of the run's workers share.
StockCache takes an exclusive lock per ticker while it updates and doesn't
call the provider for a cache that's current, so the first worker downloads
each ticker and the others find it cached.
"""

import shutil
from pathlib import Path

import pytest

from fartt.data import YFinanceProvider
from fartt.stock_cache import StockCache

# Every ticker an integration test reads from a filled cache.
LIVE_TICKERS = ("ASML.AS", "AAPL", "GOOG", "ARM", "IWDA.AS")


@pytest.fixture(scope="session")
def _downloaded(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> Path:
    """An assets folder holding LIVE_TICKERS, filled from the provider once."""
    if worker_id == "master":  # not under xdist: this run's own folder
        assets_dir = tmp_path_factory.mktemp("live-assets")
    else:  # under xdist: the folder above each worker's is the run's
        assets_dir = tmp_path_factory.getbasetemp().parent / "live-assets"
    cache = StockCache(YFinanceProvider(), assets_dir)
    for ticker in LIVE_TICKERS:
        cache.update(ticker)
    return assets_dir


@pytest.fixture
def live_assets(tmp_path: Path, _downloaded: Path) -> Path:
    """
    This test's own assets folder, a copy of the run's download: current, so
    the server doesn't fetch the history again, and free to change.
    """
    shutil.copytree(_downloaded / "stocks", tmp_path / "stocks")
    return tmp_path
