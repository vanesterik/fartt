"""
Every assets folder the server starts on needs a risk config (#149): the
server refuses to start without `<assets folder>/risk.toml`. The tests give
each folder a copy of the repository's checked-in file.
"""

import shutil
from pathlib import Path

from fartt.risk_config import RISK_FILE

ROOT = Path(__file__).resolve().parents[2]
CHECKED_IN = ROOT / "assets" / RISK_FILE


def copy_risk_config(assets_dir: Path) -> Path:
    """Copy the checked-in risk config into `assets_dir`; return its path."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(CHECKED_IN, assets_dir / RISK_FILE))
