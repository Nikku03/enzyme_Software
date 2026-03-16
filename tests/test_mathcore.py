from __future__ import annotations

from enzyme_software.mathcore.bayes_dag import predict
from enzyme_software.mathcore.eyring import (
    delta_g_with_conditions,
    estimate_delta_g,
    rate_constant,
    search_optimum_conditions,
)
from enzyme_software.mathcore.features import FeatureVector
from enzyme_software.mathcore.persistent_homology import compute_signature, score_signature


def test_bayes_dag_deterministic():
    features = FeatureVector(
        values={
            "bond_role_confidence": 0.9,
            "target_resolution_confidence": 0.92,
            "intent_confidence": 0.7,
            "condition_score": 0.8,
            "has_primary_role": 1.0,
            "route_primary_present": 1.0,
            "descriptor_complete": 1.0,
            "mechanism_count": 0.5,
            "data_support": 0.8,
            "warning_count": 0.0,
            "ambiguity_flag": 0.0,
            "job_type_reagent": 0.0,
            "novelty_penalty": 0.1,
        },
        missing=[],
        source="test",
    )
    first = predict(features)
    second = predict(features)
    assert first == second


def test_eyring_monotonic_delta_g():
    temp_k = 310.0
    low = rate_constant(12.0, temp_k)
    high = rate_constant(18.0, temp_k)
    assert low > high


def test_optimum_conditions_improve_rate():
    base_dg, _ = estimate_delta_g(20.0, {"alignment_error": 0.2})
    optimum = search_optimum_conditions(
        base_dg,
        {"pH_range": [6.5, 8.5], "temperature_c": [25.0, 45.0]},
    )
    k_opt = optimum["k_pred"]
    temp_k = 298.15
    k_base = rate_constant(
        delta_g_with_conditions(base_dg, 6.0, temp_k, {"pH_range": [6.5, 8.5], "temperature_c": [25.0, 45.0]}),
        temp_k,
    )
    assert k_opt >= k_base


def test_persistent_homology_signature_deterministic():
    signature = compute_signature(
        point_cloud=[[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
        tunnel_summary={"bottleneck_radius": 1.5, "path_length": 12.0, "curvature_proxy": 0.2},
        mode="standard",
    )
    signature2 = compute_signature(
        point_cloud=[[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
        tunnel_summary={"bottleneck_radius": 1.5, "path_length": 12.0, "curvature_proxy": 0.2},
        mode="standard",
    )
    assert signature == signature2
    score, robustness = score_signature(signature)
    assert 0.0 <= score <= 1.0
    assert 0.0 <= robustness <= 1.0
