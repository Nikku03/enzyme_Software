from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(abs(drawdown.min()))


def compute_monthly_stats(equity: pd.Series) -> Dict[str, float]:
    if equity.empty or not isinstance(equity.index, pd.DatetimeIndex):
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "pct_positive": float("nan"),
            "pct_ge_50": float("nan"),
        }
    monthly = equity.resample("M").last().pct_change().dropna()
    if monthly.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "pct_positive": float("nan"),
            "pct_ge_50": float("nan"),
        }
    return {
        "mean": float(monthly.mean()),
        "median": float(monthly.median()),
        "pct_positive": float((monthly > 0).mean()),
        "pct_ge_50": float((monthly >= 0.50).mean()),
    }


def compute_metrics(
    returns: pd.Series,
    turnover: float,
    trade_count: int,
) -> Dict[str, Any]:
    if returns.empty:
        return {
            "total_return": None,
            "cagr": None,
            "max_drawdown": None,
            "sharpe": None,
            "monthly_stats": {
                "mean": None,
                "median": None,
                "pct_positive": None,
                "pct_ge_50": None,
            },
            "turnover_proxy": turnover,
            "trade_count": trade_count,
        }

    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = len(returns) / 252.0
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret != 0 else 0.0
    max_drawdown = compute_max_drawdown(equity)
    monthly_stats = compute_monthly_stats(equity)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "monthly_stats": monthly_stats,
        "turnover_proxy": turnover,
        "trade_count": trade_count,
    }


def compute_failure_tags(
    metrics: Dict[str, Any],
    cost_sensitivity_penalty: float,
    window_cagr: Optional[Iterable[float]] = None,
    min_trades: int = 5,
) -> List[str]:
    tags: List[str] = []
    trade_count = metrics.get("trade_count", 0) or 0
    if trade_count < min_trades:
        tags.append("TOO_FEW_TRADES")

    max_drawdown = metrics.get("max_drawdown")
    if max_drawdown is not None and max_drawdown > 0.4:
        tags.append("REGIME_FRAGILE")

    if cost_sensitivity_penalty > 0.02:
        tags.append("COST_SENSITIVE")

    if window_cagr is not None:
        window_cagr = list(window_cagr)
        if len(window_cagr) >= 2:
            if np.std(window_cagr) > (abs(np.mean(window_cagr)) + 1e-6):
                tags.append("OVERFIT_SUSPECT")

    return tags


def compute_score(metrics: Dict[str, Any], scoring: Dict[str, float]) -> float:
    cagr = metrics.get("cagr") or 0.0
    max_drawdown = metrics.get("max_drawdown") or 0.0
    cost_penalty = metrics.get("cost_sensitivity_penalty") or 0.0
    low_trade_penalty = metrics.get("low_trade_penalty") or 0.0
    return (
        scoring.get("cagr_weight", 1.0) * cagr
        - scoring.get("max_drawdown_weight", 0.7) * abs(max_drawdown)
        - scoring.get("cost_sensitivity_weight", 0.2) * cost_penalty
        - scoring.get("low_trade_weight", 0.1) * low_trade_penalty
    )
