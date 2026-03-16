import random
from pathlib import Path

from lab import load_config
from lab.loop_controller import LoopController
from lab.pivot_rules import PivotPlan, apply_pivot


def test_anti_loop_pivot_triggers():
    config_path = Path(__file__).resolve().parents[1] / "config" / "lab_config.yaml"
    config = load_config(str(config_path))
    controller = LoopController(config=config)
    history = [
        {"artifact_count": 0, "improved": False},
        {"artifact_count": 0, "improved": False},
    ]
    assert controller._should_pivot(history) is True


def test_pivot_changes_two_dimensions():
    config_path = Path(__file__).resolve().parents[1] / "config" / "lab_config.yaml"
    config = load_config(str(config_path))
    plan = PivotPlan(
        strategy_family="sma_crossover",
        timeframe="daily",
        universe=["SPY", "QQQ"],
        risk_sizing="fixed",
        entry_exit="standard",
    )
    rng = random.Random(1)
    new_plan, changed = apply_pivot(plan, config, rng)
    diff_count = sum(
        [
            plan.strategy_family != new_plan.strategy_family,
            plan.timeframe != new_plan.timeframe,
            plan.universe != new_plan.universe,
            plan.risk_sizing != new_plan.risk_sizing,
            plan.entry_exit != new_plan.entry_exit,
        ]
    )
    assert diff_count >= 2
    assert len(changed) >= 2
