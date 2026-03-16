from __future__ import annotations

from typing import Any, Dict, List

from enzyme_software.mathcore.features import FeatureVector
from enzyme_software.mathcore.uncertainty import (
    ProbabilityCalibrator,
    beta_credible_interval,
    calibrate_probability,
    confidence_interval,
    estimate_uncertainty,
)


WEIGHTS: Dict[str, float] = {
    "bond_role_confidence": 1.25,
    "target_resolution_confidence": 1.1,
    "intent_confidence": 0.6,
    "condition_score": 0.5,
    "has_primary_role": 0.4,
    "route_primary_present": 0.35,
    "descriptor_complete": 0.25,
    "mechanism_count": 0.2,
    "data_support": 0.4,
    "warning_count": -0.4,
    "ambiguity_flag": -0.9,
    "job_type_reagent": -0.2,
    "novelty_penalty": -0.6,
}

BIAS = -0.35


def _sigmoid(value: float) -> float:
    if value < -35:
        return 0.0
    if value > 35:
        return 1.0
    return 1.0 / (1.0 + pow(2.718281828, -value))


def predict(feature_vector: FeatureVector) -> Dict[str, Any]:
    values = feature_vector.values
    logit = BIAS
    contributions: Dict[str, float] = {}
    for name, weight in WEIGHTS.items():
        value = values.get(name, 0.0)
        contrib = weight * value
        contributions[name] = contrib
        logit += contrib

    probability = _sigmoid(logit)
    missing_count = len(feature_vector.missing)
    uncertainty = estimate_uncertainty(probability, missing_count)
    heuristic_calibration = calibrate_probability(probability, uncertainty)
    calibrator = ProbabilityCalibrator()
    calibrated = calibrator.predict(probability)
    evidence_strength = max(0.0, 1.0 - 0.12 * missing_count)
    n_eff = max(2.0, 8.0 * evidence_strength)
    interval = beta_credible_interval(calibrated, n_eff)

    drivers = _top_drivers(contributions)

    return {
        "probability": round(calibrated, 3),
        "raw_probability": round(probability, 3),
        "heuristic_probability": round(heuristic_calibration, 3),
        "evidence_strength": round(evidence_strength, 3),
        "n_eff": round(n_eff, 2),
        "uncertainty": uncertainty,
        "uncertainty_90ci": interval,
        "drivers": drivers,
        "diagnostics": {
            "missing_features": feature_vector.missing,
            "logit": round(logit, 3),
        },
    }


def _top_drivers(contributions: Dict[str, float]) -> List[Dict[str, Any]]:
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    drivers: List[Dict[str, Any]] = []
    for name, value in ranked[:5]:
        drivers.append({"feature": name, "contribution": round(value, 3)})
    return drivers
