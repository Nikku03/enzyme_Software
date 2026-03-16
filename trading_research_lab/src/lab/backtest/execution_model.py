from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ExecutionResult:
    returns: pd.Series
    turnover: float
    trade_count: int


def apply_execution_model(
    prices: pd.Series,
    positions: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    delay: int = 1,
) -> ExecutionResult:
    if prices.empty:
        return ExecutionResult(returns=pd.Series(dtype=float), turnover=0.0, trade_count=0)

    prices = prices.sort_index()
    positions = positions.reindex(prices.index).fillna(0.0)
    executed_positions = positions.shift(delay).fillna(0.0)

    daily_returns = prices.pct_change().fillna(0.0)
    gross = executed_positions * daily_returns

    position_change = executed_positions.diff().abs().fillna(executed_positions.abs())
    cost = position_change * (fee_bps + slippage_bps) / 10000.0

    net = gross - cost

    trade_count = int(((executed_positions.diff().fillna(0.0) != 0) & (executed_positions != 0)).sum())
    turnover = float(position_change.sum())

    return ExecutionResult(returns=net, turnover=turnover, trade_count=trade_count)
