"""
The integration tests start the server over stdio, as Claude Code does. Most
run it on the checked-in snapshot (`snapshot.py`) with its clock pinned and
Yahoo out of reach: offline, so they're fast and don't fail when Yahoo does.
Every test here is marked `integration`; the few that are about Yahoo itself
(a first download, a quote, a ticker Yahoo doesn't list) are also marked
`network` and run with `task test:network` instead.
"""

import shutil
from pathlib import Path

import pytest

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
    pinned clock, so the server fetches nothing, and free to change.
    """
    shutil.copytree(SNAPSHOT_DIR / "stocks", tmp_path / "stocks")
    return tmp_path
