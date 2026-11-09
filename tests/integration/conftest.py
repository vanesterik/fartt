"""
The integration tests start the server over stdio, as Claude Code does. Most
run it on the checked-in snapshot (`snapshot.py`) with its clock pinned and
Yahoo out of reach: offline, so they're fast and don't fail when Yahoo does.
Every test here is marked `integration`; the one that fills an empty folder
from Yahoo is also marked `network` and runs with `task test:network` instead.
"""

import shutil
from pathlib import Path

import pytest

from tests.integration.risk import copy_risk_config
from tests.integration.snapshot import SNAPSHOT_DIR


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    for item in items:
        if here in item.path.parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def seeded_assets(tmp_path: Path) -> Path:
    """
    This test's own assets folder, a copy of the snapshot: current at the
    pinned clock, so the server fetches nothing, and free to change. It holds
    a copy of the checked-in risk config, which the server needs to start.
    """
    shutil.copytree(SNAPSHOT_DIR / "stocks", tmp_path / "stocks")
    copy_risk_config(tmp_path)
    return tmp_path


@pytest.fixture
def empty_assets(tmp_path: Path) -> Path:
    """
    This test's own assets folder with nothing cached: only a copy of the
    checked-in risk config, which the server needs to start.
    """
    copy_risk_config(tmp_path)
    return tmp_path
