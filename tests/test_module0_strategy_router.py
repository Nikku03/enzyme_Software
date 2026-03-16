from __future__ import annotations

from typing import Any, Dict, List

from enzyme_software.context import OperationalConstraints
from enzyme_software.modules import module0_strategy_router as m0


def _physics_priors() -> Dict[str, Dict[str, Any]]:
    return {
        "P450": {
            "prior_feasibility": 0.72,
            "prior_success_probability": 0.61,
            "route_prior_target_specific": 0.61,
            "substrate_compatibility_multiplier": 0.95,
            "route_bias_multiplier": 1.0,
            "cofactor_penalty": 0.0,
            "mechanism_mismatch_penalty": 0.05,
            "protonation_factor": 0.85,
            "mechanism_eligibility": "SUPPORTED",
        },
        "non_heme_iron": {
            "prior_feasibility": 0.68,
            "prior_success_probability": 0.58,
            "route_prior_target_specific": 0.58,
            "substrate_compatibility_multiplier": 0.92,
            "route_bias_multiplier": 1.0,
            "cofactor_penalty": 0.0,
            "mechanism_mismatch_penalty": 0.06,
            "protonation_factor": 0.82,
            "mechanism_eligibility": "SUPPORTED",
        },
        "radical_SAM": {
            "prior_feasibility": 0.18,
            "prior_success_probability": 0.11,
            "route_prior_target_specific": 0.11,
            "substrate_compatibility_multiplier": 0.55,
            "route_bias_multiplier": 0.8,
            "cofactor_penalty": 0.25,
            "mechanism_mismatch_penalty": 0.25,
            "protonation_factor": 0.60,
            "mechanism_eligibility": "REQUIRE_QUORUM",
        },
    }


def _router_prediction(entries: List[Dict[str, Any]], *, router_empty: bool = False) -> Dict[str, Any]:
    return {
        "chosen_route": entries[0]["route"],
        "route_posteriors": entries,
        "data_support": 0.72 if not router_empty else 0.0,
        "evidence_strength": 0.66 if not router_empty else 0.0,
        "router_empty": router_empty,
    }


def test_route_posteriors_are_normalized_and_finite() -> None:
    debug = m0.score_all_routes(
        route_candidates=["P450", "non_heme_iron", "radical_SAM"],
        physics_route_priors=_physics_priors(),
        router_prediction=_router_prediction(
            [
                {"route": "P450", "posterior": 0.64},
                {"route": "non_heme_iron", "posterior": 0.29},
                {"route": "radical_SAM", "posterior": 0.07},
            ]
        ),
    )
    evaluated = debug["evaluated_routes"]
    total = sum(float(row["posterior"]) for row in evaluated)
    assert evaluated
    assert abs(total - 1.0) < 1e-6
    assert all(0.0 <= float(row["posterior"]) <= 1.0 for row in evaluated)


def test_ambiguity_flag_when_top_routes_close() -> None:
    debug = m0.score_all_routes(
        route_candidates=["P450", "non_heme_iron"],
        physics_route_priors=_physics_priors(),
        router_prediction=_router_prediction(
            [
                {"route": "P450", "posterior": 0.51},
                {"route": "non_heme_iron", "posterior": 0.49},
            ]
        ),
    )
    assert debug["ambiguity_flag"] is True
    assert len(debug["top_routes"]) == 2
    assert debug["route_gap"] is not None


def test_confidence_reduced_by_fallback_and_conflicts() -> None:
    strong = m0.score_all_routes(
        route_candidates=["P450", "non_heme_iron"],
        physics_route_priors=_physics_priors(),
        router_prediction=_router_prediction(
            [
                {"route": "P450", "posterior": 0.75},
                {"route": "non_heme_iron", "posterior": 0.25},
            ]
        ),
    )
    conflicted_priors = _physics_priors()
    conflicted_priors["P450"]["mechanism_mismatch_penalty"] = 0.4
    conflicted_priors["P450"]["cofactor_penalty"] = 0.3
    weak = m0.score_all_routes(
        route_candidates=["P450", "non_heme_iron"],
        physics_route_priors=conflicted_priors,
        router_prediction=_router_prediction(
            [
                {"route": "P450", "posterior": 0.55},
                {"route": "non_heme_iron", "posterior": 0.45},
            ],
            router_empty=True,
        ),
        fallback_used=True,
    )
    assert strong["confidence_components"]["confidence"] > weak["confidence_components"]["confidence"]
    assert "fallback_override_used" in weak["evidence_conflicts"]


def test_debug_report_contains_evidence_calibration_and_top_routes() -> None:
    debug = m0.score_all_routes(
        route_candidates=["P450", "non_heme_iron"],
        physics_route_priors=_physics_priors(),
        router_prediction=_router_prediction(
            [
                {"route": "P450", "posterior": 0.6},
                {"route": "non_heme_iron", "posterior": 0.4},
            ]
        ),
    )
    assert "evaluated_routes" in debug
    assert "calibration" in debug
    assert "confidence_components" in debug
    assert debug["top_routes"]
    first = debug["evaluated_routes"][0]
    for key in (
        "chemistry_score",
        "physics_score",
        "biology_score",
        "compatibility_score",
        "cofactor_score",
        "eligibility_score",
        "mismatch_penalty",
        "final_score_pre_calibration",
        "final_score_post_calibration",
        "evidence_notes",
    ):
        assert key in first


def test_route_job_preserves_existing_keys_and_adds_router_debug() -> None:
    result = m0.route_job(
        smiles="CC(=O)OCC",
        target_bond="ester__acyl_o",
        requested_output=None,
        trap_target=None,
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
        strict_validation=False,
    )
    job_card = result["job_card"]
    assert "chosen_route" in job_card
    assert "route_posteriors" in job_card
    assert "physics_audit" in job_card
    assert "router_debug" in job_card
    assert "route_audit" in job_card
    assert "confidence_calibrated" in job_card
    assert "top_routes" in job_card
