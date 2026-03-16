from __future__ import annotations

import pandas as pd

from lab.strategies.base import Strategy


class DonchianBreakoutStrategy(Strategy):
    name = "breakout"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = int(self.params.get("lookback", 20))
        high_roll = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
        low_roll = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
        close = df["close"]

        positions = []
        position = 0.0
        for idx in df.index:
            if pd.notna(high_roll.loc[idx]) and close.loc[idx] > high_roll.loc[idx]:
                position = 1.0
            elif pd.notna(low_roll.loc[idx]) and close.loc[idx] < low_roll.loc[idx]:
                position = 0.0
            positions.append(position)
        return pd.Series(positions, index=df.index)
