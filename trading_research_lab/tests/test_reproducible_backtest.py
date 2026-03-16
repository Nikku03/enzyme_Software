from pathlib import Path

import pandas as pd
import pytest

from lab import load_config
from lab.backtest.evaluator import evaluate_strategy
from lab.data.vault import DataVault
from lab.research.extractors import parse_stooq_csv
from lab.strategies.generator import build_strategy


def _metrics_close(a, b):
    for key in [
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "turnover_proxy",
        "trade_count",
        "cost_sensitivity_penalty",
        "low_trade_penalty",
    ]:
        assert a[key] == pytest.approx(b[key])
    for key in ["mean", "median", "pct_positive", "pct_ge_50"]:
        assert a["monthly_stats"][key] == pytest.approx(b["monthly_stats"][key])


def test_reproducible_backtest(tmp_path: Path):
    dates = pd.date_range(start="2020-01-01", periods=120, freq="D")
    content = ["Date,Open,High,Low,Close,Volume"]
    for i, dt in enumerate(dates):
        content.append(
            f"{dt.date()},{100+i},{101+i},{99+i},{100+i},{1000+i}"
        )
    csv_content = ("\n".join(content) + "\n").encode("utf-8")

    vault = DataVault(root=tmp_path, processed_format="parquet")
    metadata = vault.ingest(
        source_url="https://stooq.com/q/d/l/?s=spy.us&i=d",
        content=csv_content,
        parser=parse_stooq_csv,
        symbol="SPY",
        file_ext="csv",
        retrieval_time="2020-01-01T00:00:00",
    )

    df = vault.load(metadata.dataset_id)
    config_path = Path(__file__).resolve().parents[1] / "config" / "lab_config.yaml"
    config = load_config(str(config_path))

    strategy = build_strategy("sma_crossover", config)
    payload1 = evaluate_strategy(
        df=df,
        strategy=strategy,
        config=config,
        dataset_id=metadata.dataset_id,
        walkforward=False,
    )
    payload2 = evaluate_strategy(
        df=df,
        strategy=strategy,
        config=config,
        dataset_id=metadata.dataset_id,
        walkforward=False,
    )

    _metrics_close(payload1["metrics"], payload2["metrics"])
    _metrics_close(payload1["stress_tests"]["metrics"], payload2["stress_tests"]["metrics"])
    assert payload1["failure_tags"] == payload2["failure_tags"]
