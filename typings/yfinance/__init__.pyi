# Only what fartt calls.
from typing import Any

import pandas as pd

class Ticker:
    def __init__(self, ticker: str) -> None: ...
    @property
    def info(self) -> dict[str, Any]: ...
    def history(self, **kwargs: Any) -> pd.DataFrame: ...

class _Debug:
    hide_exceptions: bool

class _Config:
    debug: _Debug

config: _Config
