from __future__ import annotations

from typing import Dict

import numpy as np


def analyze_gate_values(gate_values) -> Dict[str, float]:
    if hasattr(gate_values, "detach"):
        array = gate_values.detach().cpu().numpy()
    else:
        array = np.asarray(gate_values)
    return {
        "gate_mean": float(np.mean(array)) if array.size else 0.0,
        "gate_std": float(np.std(array)) if array.size else 0.0,
        "fraction_prefers_liquid": float(np.mean(array > 0.5)) if array.size else 0.0,
        "fraction_prefers_physics": float(np.mean(array <= 0.5)) if array.size else 0.0,
    }
