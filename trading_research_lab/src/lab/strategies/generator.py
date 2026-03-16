from __future__ import annotations

from typing import Dict, List, Type

from lab.strategies.base import Strategy
from lab.strategies.library.breakout import DonchianBreakoutStrategy
from lab.strategies.library.mean_reversion import MeanReversionStrategy
from lab.strategies.library.sma_crossover import SMACrossoverStrategy

STRATEGY_MAP: Dict[str, Type[Strategy]] = {
    "sma_crossover": SMACrossoverStrategy,
    "breakout": DonchianBreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def build_strategy(name: str, config: Dict) -> Strategy:
    key = name.lower()
    if key not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}")
    params = config.get("strategies", {}).get(key, {})
    return STRATEGY_MAP[key](params=params)


def list_strategies() -> List[str]:
    return list(STRATEGY_MAP.keys())
