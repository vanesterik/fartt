"""
Every assets folder the server starts on needs a risk config (#149): the
server refuses to start without `<assets folder>/risk.toml`. The tests give
each folder a copy of the repository's checked-in file, and read it back for
a server they build in this process.
"""

import shutil
from pathlib import Path

from fartt.risk_config import RISK_FILE, RiskConfig, load_risk_config

ROOT = Path(__file__).resolve().parents[2]
CHECKED_IN = ROOT / "assets" / RISK_FILE


def copy_risk_config(assets_dir: Path) -> Path:
    """Copy the checked-in risk config into `assets_dir`; return its path."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(CHECKED_IN, assets_dir / RISK_FILE))


def checked_in_risk_config() -> RiskConfig:
    """The checked-in risk config, for a server built in this process."""
    return load_risk_config(CHECKED_IN)


def written(assets_dir: Path) -> list[str]:
    """What's in `assets_dir` besides its risk config, by name."""
    return sorted(p.name for p in assets_dir.iterdir() if p.name != RISK_FILE)
