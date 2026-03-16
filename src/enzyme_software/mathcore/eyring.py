from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import random

from enzyme_software.mathcore.uncertainty import percentile

R_KCAL = 0.001987
K_B = 1.380649e-23
H_P = 6.62607015e-34


def estimate_delta_g(
    base_delta_g: float,
    features: Dict[str, Any],
) -> Tuple[float, float]:
    delta_g = float(base_delta_g)
    delta_g += 2.0 * float(features.get("steric_clash", 0.0))
    delta_g += 1.6 * float(features.get("electrostatic_mismatch", 0.0))
    delta_g -= 1.4 * float(features.get("hbond_quality", 0.0))
    delta_g += 1.8 * float(features.get("alignment_error", 0.0))
    delta_g += 1.2 * float(features.get("retention_penalty", 0.0))
    delta_g = max(6.0, delta_g)

    difficulty = features.get("difficulty_label") or features.get("difficulty") or ""
    difficulty = str(difficulty).upper()
    base_std = {"EASY": 0.6, "MEDIUM": 1.0, "HARD": 1.6}.get(difficulty, 1.2)
    data_support = features.get("data_support")
    if not isinstance(data_support, (int, float)):
        data_support = 0.5
    support_factor = 1.4 - 0.8 * max(0.0, min(1.0, float(data_support)))
    delta_g_std = max(0.3, base_std * support_factor)
    return delta_g, delta_g_std


def delta_g_with_conditions(
    base_delta_g: float,
    ph: Optional[float],
    temp_k: Optional[float],
    optimum_hint: Dict[str, Any],
) -> float:
    ph_range = optimum_hint.get("pH_range") or [6.5, 8.0]
    temp_range = optimum_hint.get("temperature_c") or [25.0, 45.0]
    delta_g = float(base_delta_g)
    if ph is not None:
        if ph < ph_range[0]:
            delta_g += min(2.2, (ph_range[0] - ph) * 1.2)
        elif ph > ph_range[1]:
            delta_g += min(2.2, (ph - ph_range[1]) * 1.2)
    if temp_k is not None:
        temp_c = temp_k - 273.15
        if temp_c < temp_range[0]:
            delta_g += min(1.8, (temp_range[0] - temp_c) * 0.08)
        elif temp_c > temp_range[1]:
            delta_g += min(1.8, (temp_c - temp_range[1]) * 0.08)
    return delta_g


def rate_constant(delta_g: float, temp_k: float) -> float:
    temp_k = max(273.15, float(temp_k))
    prefactor = (K_B * temp_k) / H_P
    return prefactor * math.exp(-float(delta_g) / (R_KCAL * temp_k))


def _ensure_rng(rng: Optional[object], seed: Optional[int]) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    if isinstance(rng, (int, float)):
        return random.Random(int(rng))
    if seed is not None:
        return random.Random(int(seed))
    return random.Random()


def _sample_k_pred_stats(
    delta_g_mean: float,
    delta_g_std: float,
    temp_c: float,
    n: int,
    rng: random.Random,
) -> Dict[str, Any]:
    sample_count = max(8, int(n))
    temp_k = float(temp_c) + 273.15
    samples: List[float] = []
    if delta_g_std <= 0:
        samples = [rate_constant(delta_g_mean, temp_k) for _ in range(sample_count)]
    else:
        for _ in range(sample_count):
            delta_g = rng.gauss(float(delta_g_mean), float(delta_g_std))
            samples.append(rate_constant(delta_g, temp_k))
    mean_val = sum(samples) / len(samples)
    variance = sum((value - mean_val) ** 2 for value in samples) / len(samples)
    std_val = math.sqrt(variance)
    low = percentile(samples, 0.05)
    high = percentile(samples, 0.95)
    return {
        "mean": mean_val,
        "std": std_val,
        "ci90": [low, high],
    }


def sample_k_pred(
    delta_g: float,
    temperature: float = 298.15,
    sigma: float = 1.0,
    n: int = 256,
    rng: Optional[object] = None,
    **kwargs: Any,
) -> Any:
    """Sample Eyring rate predictions from delta_g uncertainty.

    Returns a list of k samples for the new signature. For backward compatibility,
    when called with legacy arguments (delta_g_std, temp_C, seed), a stats dict
    with mean/std/ci90 is returned instead.
    """
    seed = kwargs.pop("seed", None)
    rng_obj = _ensure_rng(rng, seed)
    legacy_mode = False
    if isinstance(temperature, (int, float)) and isinstance(sigma, (int, float)):
        if float(temperature) <= 5.0 and float(sigma) >= 10.0:
            legacy_mode = True
    if legacy_mode:
        return _sample_k_pred_stats(
            delta_g_mean=float(delta_g),
            delta_g_std=float(temperature),
            temp_c=float(sigma),
            n=n,
            rng=rng_obj,
        )

    sample_count = max(2, int(n))
    temp_k = float(temperature)
    sigma_val = max(0.0, float(sigma))
    samples: List[float] = []
    if sigma_val <= 0.0:
        samples = [
            max(rate_constant(float(delta_g), temp_k), 1e-300)
            for _ in range(sample_count)
        ]
    else:
        for _ in range(sample_count):
            delta_g_i = rng_obj.gauss(float(delta_g), sigma_val)
            k_val = rate_constant(delta_g_i, temp_k)
            samples.append(max(k_val, 1e-300))

    base = rate_constant(float(delta_g), temp_k)
    mean_val = sum(samples) / len(samples)
    if mean_val > 0.0:
        scale = base / mean_val
        samples = [max(value * scale, 1e-300) for value in samples]
    return samples


def search_optimum_conditions(
    base_delta_g: float,
    optimum_hint: Dict[str, Any],
    ph_bounds: Optional[List[float]] = None,
    temp_bounds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    ph_range = ph_bounds or optimum_hint.get("pH_range") or [6.5, 8.0]
    temp_range = temp_bounds or optimum_hint.get("temperature_c") or [25.0, 45.0]
    ph_candidates = _linspace(ph_range[0], ph_range[1], 4)
    temp_candidates = _linspace(temp_range[0], temp_range[1], 4)

    best = {"score": -1.0, "pH_opt": ph_candidates[0], "T_opt_C": temp_candidates[0]}
    for ph in ph_candidates:
        for temp_c in temp_candidates:
            temp_k = temp_c + 273.15
            dg = delta_g_with_conditions(base_delta_g, ph, temp_k, optimum_hint)
            k_pred = rate_constant(dg, temp_k)
            if k_pred > best["score"]:
                best = {"score": k_pred, "pH_opt": round(ph, 2), "T_opt_C": round(temp_c, 1)}
    return {
        "pH_opt": best["pH_opt"],
        "T_opt_C": best["T_opt_C"],
        "T_opt_K": round(best["T_opt_C"] + 273.15, 2),
        "k_pred": best["score"],
    }


def _linspace(start: float, stop: float, steps: int) -> List[float]:
    if steps <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / (steps - 1)
    return [float(start) + i * step for i in range(steps)]
