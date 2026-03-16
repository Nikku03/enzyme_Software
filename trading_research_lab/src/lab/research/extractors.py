from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd


def parse_stooq_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(
        BytesIO(content),
        parse_dates=["Date"],
        dtype={"Volume": "float64"},
    )
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna().sort_values("date").drop_duplicates("date")
    df = df.set_index("date")
    return df


def filter_dates(
    df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    if start:
        start_dt = datetime.fromisoformat(start)
        df = df[df.index >= start_dt]
    if end:
        end_dt = datetime.fromisoformat(end)
        df = df[df.index <= end_dt]
    return df


def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "weekly":
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        return df.resample("W-FRI").agg(ohlc).dropna()
    return df
