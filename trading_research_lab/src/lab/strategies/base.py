from __future__ import annotations

from typing import Dict

import pandas as pd


class Strategy:
    name: str = "base"

    def __init__(self, params: Dict[str, float] | None = None) -> None:
        self.params = params or {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
