from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .schemas import OHLCV_COLUMNS


def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    issues: Dict[str, str] = {}
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        issues["missing_columns"] = ",".join(missing)
    if df.index.duplicated().any():
        issues["duplicate_index"] = "true"
    if not df.index.is_monotonic_increasing:
        issues["index_order"] = "not_monotonic"
    if not missing and df[OHLCV_COLUMNS].isna().any().any():
        issues["nan_values"] = "true"
    return (len(issues) == 0), issues
