from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_slice: slice
    val_slice: slice
    test_slice: slice


def generate_walkforward_splits(
    index: pd.DatetimeIndex,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
) -> List[WalkForwardWindow]:
    n = len(index)
    windows: List[WalkForwardWindow] = []
    start = 0
    while True:
        train_end = start + train_days
        val_end = train_end + val_days
        test_end = val_end + test_days
        if test_end > n:
            break
        windows.append(
            WalkForwardWindow(
                train_slice=slice(start, train_end),
                val_slice=slice(train_end, val_end),
                test_slice=slice(val_end, test_end),
            )
        )
        start += step_days
    return windows
