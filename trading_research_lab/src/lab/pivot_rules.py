from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import random


@dataclass(frozen=True)
class PivotPlan:
    strategy_family: str
    timeframe: str
    universe: List[str]
    risk_sizing: str
    entry_exit: str


def _pick_new(current: str, options: List[str], rng: random.Random) -> str:
    choices = [opt for opt in options if opt != current]
    if not choices:
        return current
    return rng.choice(choices)


def apply_pivot(plan: PivotPlan, config: Dict, rng: random.Random) -> Tuple[PivotPlan, List[str]]:
    strategies = config.get("strategies", {}).get("families", [plan.strategy_family])
    timeframes = config.get("timeframes", [plan.timeframe])
    universes = config.get("loop", {}).get("universe_subsets", [plan.universe])
    risk_options = config.get("risk", {}).get("sizing_options", [plan.risk_sizing])
    entry_exit_options = config.get("entry_exit", {}).get("styles", [plan.entry_exit])

    dimensions = ["strategy_family", "timeframe", "universe", "risk_sizing", "entry_exit"]
    rng.shuffle(dimensions)
    changed: List[str] = []
    updated = plan

    for dim in dimensions:
        if len(changed) >= config.get("loop", {}).get("pivot_change_dimensions", 2):
            break
        if dim == "strategy_family":
            new_value = _pick_new(plan.strategy_family, strategies, rng)
            if new_value != plan.strategy_family:
                updated = replace(updated, strategy_family=new_value)
                changed.append(dim)
        elif dim == "timeframe":
            new_value = _pick_new(plan.timeframe, timeframes, rng)
            if new_value != plan.timeframe:
                updated = replace(updated, timeframe=new_value)
                changed.append(dim)
        elif dim == "universe":
            choices = [u for u in universes if u != plan.universe]
            if choices:
                updated = replace(updated, universe=rng.choice(choices))
                changed.append(dim)
        elif dim == "risk_sizing":
            new_value = _pick_new(plan.risk_sizing, risk_options, rng)
            if new_value != plan.risk_sizing:
                updated = replace(updated, risk_sizing=new_value)
                changed.append(dim)
        elif dim == "entry_exit":
            new_value = _pick_new(plan.entry_exit, entry_exit_options, rng)
            if new_value != plan.entry_exit:
                updated = replace(updated, entry_exit=new_value)
                changed.append(dim)

    return updated, changed
