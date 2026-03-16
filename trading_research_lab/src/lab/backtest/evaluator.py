from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from lab.backtest.execution_model import apply_execution_model
from lab.backtest.metrics import compute_failure_tags, compute_metrics
from lab.backtest.walkforward import generate_walkforward_splits

VERSION = "1.0.0"


def ensure_evaluator_seal(artifacts_root: Path, maintenance_mode: bool) -> None:
    seal_path = artifacts_root / "leaderboards" / "evaluator_version.txt"
    if seal_path.exists():
        existing = seal_path.read_text(encoding="utf-8").strip()
        if existing != VERSION and not maintenance_mode:
            raise RuntimeError(
                "Evaluator version changed without maintenance mode. "
                "Set LAB_MAINTENANCE_MODE=true to proceed."
            )
    else:
        seal_path.write_text(VERSION, encoding="utf-8")


def _evaluate_slice(
    df_window: pd.DataFrame,
    test_slice: slice,
    strategy,
    fee_bps: float,
    slippage_bps: float,
    delay: int,
) -> Dict[str, Any]:
    signals = strategy.generate_signals(df_window)
    positions = signals.reindex(df_window.index).fillna(0.0)
    test_index = df_window.index[test_slice]
    test_positions = positions.loc[test_index]
    test_prices = df_window.loc[test_index, "close"]
    exec_result = apply_execution_model(
        prices=test_prices,
        positions=test_positions,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        delay=delay,
    )
    return {
        "returns": exec_result.returns,
        "turnover": exec_result.turnover,
        "trade_count": exec_result.trade_count,
        "positions": test_positions,
        "prices": test_prices,
    }


def evaluate_strategy(
    df: pd.DataFrame,
    strategy,
    config: Dict[str, Any],
    dataset_id: Optional[str] = None,
    walkforward: bool = False,
) -> Dict[str, Any]:
    bt_cfg = config["backtest"]
    fee_bps = float(bt_cfg["fee_bps"])
    slippage_bps = float(bt_cfg["slippage_bps"])
    delay = int(bt_cfg["execution_delay"])
    stress_cfg = bt_cfg["stress"]

    window_cagr: List[float] = []
    combined_returns: List[pd.Series] = []
    combined_stress_returns: List[pd.Series] = []
    total_turnover = 0.0
    total_trades = 0

    if walkforward:
        wf = bt_cfg["walkforward"]
        windows = generate_walkforward_splits(
            df.index,
            train_days=int(wf["train_days"]),
            val_days=int(wf["val_days"]),
            test_days=int(wf["test_days"]),
            step_days=int(wf["step_days"]),
        )
        if not windows:
            walkforward = False
            windows = []
        for window in windows:
            window_start = window.train_slice.start
            df_window = df.iloc[window_start : window.test_slice.stop]
            relative_test = slice(
                window.test_slice.start - window_start,
                window.test_slice.stop - window_start,
            )
            result = _evaluate_slice(
                df_window,
                relative_test,
                strategy,
                fee_bps,
                slippage_bps,
                delay,
            )
            combined_returns.append(result["returns"])
            total_turnover += result["turnover"]
            total_trades += result["trade_count"]

            stress_exec = apply_execution_model(
                prices=result["prices"],
                positions=result["positions"],
                fee_bps=fee_bps,
                slippage_bps=float(stress_cfg["slippage_bps"]),
                delay=int(stress_cfg["delay"]),
            )
            combined_stress_returns.append(stress_exec.returns)

            window_metrics = compute_metrics(
                result["returns"],
                turnover=result["turnover"],
                trade_count=result["trade_count"],
            )
            if window_metrics.get("cagr") is not None:
                window_cagr.append(float(window_metrics["cagr"]))
    if not walkforward:
        result = _evaluate_slice(
            df,
            slice(0, len(df)),
            strategy,
            fee_bps,
            slippage_bps,
            delay,
        )
        combined_returns.append(result["returns"])
        total_turnover = result["turnover"]
        total_trades = result["trade_count"]

        stress_exec = apply_execution_model(
            prices=result["prices"],
            positions=result["positions"],
            fee_bps=fee_bps,
            slippage_bps=float(stress_cfg["slippage_bps"]),
            delay=int(stress_cfg["delay"]),
        )
        combined_stress_returns.append(stress_exec.returns)

    if combined_returns:
        all_returns = pd.concat(combined_returns).sort_index()
    else:
        all_returns = pd.Series(dtype=float)
    metrics = compute_metrics(
        returns=all_returns,
        turnover=total_turnover,
        trade_count=total_trades,
    )

    if combined_stress_returns:
        stress_returns = pd.concat(combined_stress_returns).sort_index()
    else:
        stress_returns = pd.Series(dtype=float)
    stress_metrics = compute_metrics(
        returns=stress_returns,
        turnover=total_turnover,
        trade_count=total_trades,
    )
    stress_result = {
        "metrics": stress_metrics,
        "deltas": {
            "cagr_delta": (stress_metrics.get("cagr") or 0.0) - (metrics.get("cagr") or 0.0),
            "total_return_delta": (stress_metrics.get("total_return") or 0.0)
            - (metrics.get("total_return") or 0.0),
        },
    }

    cost_sensitivity_penalty = max(
        0.0, (metrics.get("cagr") or 0.0) - (stress_metrics.get("cagr") or 0.0)
    )
    min_trades = int(config.get("min_trades", 5))
    low_trade_penalty = max(0.0, (min_trades - total_trades) / max(1, min_trades))

    metrics["cost_sensitivity_penalty"] = cost_sensitivity_penalty
    metrics["low_trade_penalty"] = low_trade_penalty

    failure_tags = compute_failure_tags(
        metrics,
        cost_sensitivity_penalty=cost_sensitivity_penalty,
        window_cagr=window_cagr if window_cagr else None,
        min_trades=min_trades,
    )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "evaluator_version": VERSION,
        "dataset_id": dataset_id,
        "strategy": strategy.name,
        "params": strategy.params,
        "walkforward": walkforward,
        "metrics": metrics,
        "stress_tests": stress_result,
        "failure_tags": failure_tags,
        "window_cagr": window_cagr,
    }


def write_metrics_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
