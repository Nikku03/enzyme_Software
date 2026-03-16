import pandas as pd

from lab.strategies.generator import build_strategy
from lab import load_config


def _sample_df(rows: int = 60) -> pd.DataFrame:
    index = pd.date_range(start="2020-01-01", periods=rows, freq="D")
    data = {
        "open": range(rows),
        "high": [x + 1 for x in range(rows)],
        "low": [x - 1 for x in range(rows)],
        "close": range(rows),
        "volume": [100] * rows,
    }
    return pd.DataFrame(data, index=index)


def _assert_no_lookahead(strategy_name: str) -> None:
    config = load_config()
    df = _sample_df()
    strategy = build_strategy(strategy_name, config)
    signals = strategy.generate_signals(df)

    cutoff = df.index[30]
    df_modified = df.copy()
    df_modified.loc[cutoff:, "close"] = df_modified.loc[cutoff:, "close"] * 10
    df_modified.loc[cutoff:, "high"] = df_modified.loc[cutoff:, "high"] * 10
    df_modified.loc[cutoff:, "low"] = df_modified.loc[cutoff:, "low"] * 10

    signals_modified = strategy.generate_signals(df_modified)
    pd.testing.assert_series_equal(
        signals.loc[:cutoff],
        signals_modified.loc[:cutoff],
        check_names=False,
    )


def test_no_lookahead_sma_crossover():
    _assert_no_lookahead("sma_crossover")


def test_no_lookahead_breakout():
    _assert_no_lookahead("breakout")


def test_no_lookahead_mean_reversion():
    _assert_no_lookahead("mean_reversion")
