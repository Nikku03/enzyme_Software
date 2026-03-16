from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .execution_model import apply_execution_model
from .metrics import compute_metrics


def run_stress_tests(
    prices: pd.Series,
    positions: pd.Series,
    base_metrics: Dict[str, Any],
    slippage_bps: float,
    delay: int,
    fee_bps: float,
) -> Dict[str, Any]:
    stress_exec = apply_execution_model(
        prices=prices,
        positions=positions,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        delay=delay,
    )
    stress_metrics = compute_metrics(
        returns=stress_exec.returns,
        turnover=stress_exec.turnover,
        trade_count=stress_exec.trade_count,
    )
    base_cagr = base_metrics.get("cagr") or 0.0
    stress_cagr = stress_metrics.get("cagr") or 0.0
    delta = {
        "cagr_delta": stress_cagr - base_cagr,
        "total_return_delta": (stress_metrics.get("total_return") or 0.0)
        - (base_metrics.get("total_return") or 0.0),
    }
    return {
        "metrics": stress_metrics,
        "deltas": delta,
    }
