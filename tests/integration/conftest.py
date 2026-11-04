"""
Shared live data for the integration tests. Downloading a ticker's full
history from the data provider takes a second or two, and most tests need the
same few tickers, so they're downloaded once per run and each test gets its
own copy. A test that checks the download itself starts from an empty
`tmp_path` instead.
"""

import shutil
from pathlib import Path

import pytest

from fartt.data import YFinanceProvider
from fartt.stock_cache import StockCache

# Every ticker an integration test reads from a filled cache.
LIVE_TICKERS = ("ASML.AS", "AAPL", "GOOG", "ARM")


@pytest.fixture(scope="session")
def _downloaded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An assets folder holding LIVE_TICKERS, filled from the provider once."""
    assets_dir = tmp_path_factory.mktemp("live-assets")
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
