from __future__ import annotations

import pandas as pd

from lab.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = int(self.params.get("lookback", 20))
        entry_z = float(self.params.get("entry_z", 1.0))
        exit_z = float(self.params.get("exit_z", 0.2))

        close = df["close"]
        mean = close.rolling(lookback, min_periods=lookback).mean()
        std = close.rolling(lookback, min_periods=lookback).std()
        zscore = (close - mean) / std.replace(0, pd.NA)

        positions = []
        position = 0.0
        for idx in df.index:
            z = zscore.loc[idx]
            if pd.notna(z) and z < -entry_z:
                position = 1.0
            elif pd.notna(z) and z > -exit_z:
                position = 0.0
            positions.append(position)
        return pd.Series(positions, index=df.index)
