from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def analyze_tau_history(tau_history, tau_init, bond_classes: Iterable[str]) -> Dict[str, object]:
    tau_arrays = [_to_numpy(tau).reshape(-1) for tau in tau_history]
    tau_init_arr = _to_numpy(tau_init).reshape(-1)
    final_tau = tau_arrays[-1] if tau_arrays else tau_init_arr
    corr = float(np.corrcoef(np.stack([final_tau, tau_init_arr]))[0, 1]) if final_tau.size > 1 else 0.0
    grouped = {}
    for idx, bond_class in enumerate(bond_classes):
        grouped.setdefault(str(bond_class), []).append(float(final_tau[idx]))
    return {
        "tau_layers": tau_arrays,
        "tau_init_correlation": corr,
        "tau_by_class": {key: float(sum(vals) / len(vals)) for key, vals in grouped.items()},
        "tau_mean": float(np.mean(final_tau)) if final_tau.size else 0.0,
        "tau_std": float(np.std(final_tau)) if final_tau.size else 0.0,
    }
