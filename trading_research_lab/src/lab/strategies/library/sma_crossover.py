from __future__ import annotations

import pandas as pd

from lab.strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    name = "sma_crossover"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short_window = int(self.params.get("short_window", 20))
        long_window = int(self.params.get("long_window", 50))
        close = df["close"]
        short_ma = close.rolling(short_window, min_periods=short_window).mean()
        long_ma = close.rolling(long_window, min_periods=long_window).mean()
        signal = (short_ma > long_ma).astype(float)
        return signal.fillna(0.0)
