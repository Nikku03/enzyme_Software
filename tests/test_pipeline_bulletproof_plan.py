"""Priority pipeline validation harness from the bulletproof test plan.

This suite focuses on:
1. Priority chemistry runs across C-H / ester / amide / halide / X-H paths.
2. Universal consistency checks on physics + probability contracts.
3. Regression checks for known historical failures.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pytest

from enzyme_software.pipeline import run_pipeline


PRIORITY_RUNS = [
    {
        "id": "amide",
        "smiles": "CC(=O)NC",
        "target_bond": "amide",
        "expected_tracks": {"nucleophilic"},
    },
    {
        "id": "aliphatic_ch",
        "smiles": "CCCC",
        "target_bond": "C-H",
        "expected_tracks": {"radical_hat"},
    },
    {
        "id": "ester_with_cf3",
        "smiles": "CC(=O)OCC(F)(F)F",
        "target_bond": "ester__acyl_o",
        "expected_tracks": {"nucleophilic"},
    },
    {
        "id": "benzene_ch",
        "smiles": "c1ccccc1",
        "target_bond": "C-H",
        "expected_tracks": {"radical_hat"},
    },
    {
        "id": "no_ch_available",
        "smiles": "FC(F)(F)C(F)(F)F",
        "target_bond": "C-H",
        "expected_tracks": {"radical_hat"},
        "expect_no_match": True,
    },
    {
        "id": "cysteine_hetero_rich",
        "smiles": "O=C(O)C(N)CS",
        "target_bond": "C-H",
        "expected_tracks": {"radical_hat"},
    },
    {
        "id": "alkyl_halide",
        "smiles": "CCCl",
        "target_bond": "C-Cl",
        "expected_tracks": {"nucleophilic"},
    },
    {
        "id": "oh_bond",
        "smiles": "CCO",
        "target_bond": "O-H",
        "expected_tracks": {"radical_hat"},
    },
]


def _module_minus1_bundle(ctx) -> Dict[str, Any]:
    shared = ctx.data.get("shared_io") or {}
    m1 = (shared.get("outputs") or {}).get("module_minus1") or {}
    result = m1.get("result") or (ctx.data.get("module_minus1") or {})
    cpt = m1.get("cpt") or result.get("cpt_scores") or {}
    return {"result": result, "cpt": cpt}


def _assert_universal_checks(ctx, *, allow_missing_physics: bool = False) -> None:
    job_card = ctx.data.get("job_card") or {}
    physics_audit = job_card.get("physics_audit") or {}
    energy = job_card.get("energy_ledger") or {}
    confidence = job_card.get("confidence") or {}

    # UC7 / UC9 bounds on confidence-like values
    for key in (
        "route",
        "feasibility_if_specified",
        "target_resolution",
        "wetlab_prior",
        "wetlab_prior_target_spec",
        "wetlab_prior_any_activity",
    ):
        val = confidence.get(key)
        if isinstance(val, (int, float)):
            assert 0.0 <= float(val) <= 1.0, f"{key} out of bounds: {val}"

    # Some runs intentionally have no match and should not have complete physics.
    delta_g = energy.get("deltaG_dagger_kJ")
    k_eff = energy.get("k_eff_s_inv")
    if not allow_missing_physics:
        assert isinstance(delta_g, (int, float)), "missing deltaG_dagger_kJ"
        assert isinstance(k_eff, (int, float)), "missing k_eff_s_inv"
        assert isinstance(physics_audit.get("deltaG_dagger_kJ_per_mol"), (int, float))
        assert isinstance(physics_audit.get("k_eff_s_inv"), (int, float))

    # UC1: energy ledger vs physics audit consistency.
    if isinstance(delta_g, (int, float)) and isinstance(
        physics_audit.get("deltaG_dagger_kJ_per_mol"), (int, float)
    ):
        assert abs(float(delta_g) - float(physics_audit["deltaG_dagger_kJ_per_mol"])) < 1e-3
    if isinstance(k_eff, (int, float)) and isinstance(physics_audit.get("k_eff_s_inv"), (int, float)):
        assert abs(float(k_eff) - float(physics_audit["k_eff_s_inv"])) < 1e-6

    # UC2: Eyring equation check.
    k_ey = energy.get("eyring_k_s_inv")
    temp_k = physics_audit.get("temperature_K")
    if isinstance(delta_g, (int, float)) and isinstance(k_ey, (int, float)) and isinstance(temp_k, (int, float)):
        k_expected = 6.212e12 * math.exp(-float(delta_g) / (8.314e-3 * float(temp_k)))
        if k_expected > 0:
            rel_err = abs(float(k_ey) - k_expected) / k_expected
            assert rel_err < 0.05, f"Eyring mismatch rel_err={rel_err:.3f}"

    # UC4: p(success) consistency.
    p_success = energy.get("p_success_horizon")
    horizon_s = energy.get("horizon_s")
    if isinstance(k_eff, (int, float)) and isinstance(p_success, (int, float)) and isinstance(horizon_s, (int, float)):
        p_expected = 1.0 - math.exp(-float(k_eff) * float(horizon_s))
        assert abs(float(p_success) - p_expected) < 0.01
        assert 0.0 <= float(p_success) <= 1.0

    # UC8: basic sanity bounds.
    if isinstance(delta_g, (int, float)):
        assert 0.0 < float(delta_g) < 200.0
    if isinstance(k_eff, (int, float)):
        assert 0.0 <= float(k_eff) < 1e10

    # Route posterior bounds.
    for row in job_card.get("route_posteriors") or []:
        p = row.get("posterior")
        if isinstance(p, (int, float)):
            assert 0.0 <= float(p) <= 1.0
        ci90 = row.get("ci90")
        if isinstance(ci90, list) and len(ci90) == 2:
            assert float(ci90[0]) <= float(ci90[1])


def _assert_track_consistency(ctx, target_bond: str) -> None:
    m1 = _module_minus1_bundle(ctx)
    cpt = m1["cpt"]
    track = str(cpt.get("track") or "").lower()
    token = str(target_bond or "").lower()

    if token in {"c-h", "n-h", "o-h"}:
        assert track == "radical_hat"
    if "ester" in token or "amide" in token:
        # Nucleophilic path: track may be empty, but layered outputs should exist.
        assert isinstance(cpt.get("level2"), dict)
        assert isinstance(cpt.get("epav"), dict)


@pytest.mark.parametrize("case", PRIORITY_RUNS, ids=[c["id"] for c in PRIORITY_RUNS])
def test_priority_runs(case: Dict[str, Any]) -> None:
    ctx = run_pipeline(
        case["smiles"],
        case["target_bond"],
        requested_output=case.get("requested_output"),
        trap_target=case.get("trap_target"),
    )
    job_card = ctx.data.get("job_card") or {}
    summary = ctx.data.get("pipeline_summary") or {}
    decision = (summary.get("results") or {}).get("decision")
    halt_reason = (summary.get("results") or {}).get("halt_reason")

    m1 = _module_minus1_bundle(ctx)
    resolved = (m1["result"] or {}).get("resolved_target") or {}
    match_count = int(resolved.get("match_count") or 0)

    if case.get("expect_no_match"):
        assert match_count == 0
        assert decision in {"NO_GO", "HALT_NEED_SELECTION"}
        assert halt_reason in {"M0_NO_MATCH", "M0_TARGET_RESOLUTION_LOW", "M0_TARGET_CLARIFICATION_REQUIRED"}
        _assert_universal_checks(ctx, allow_missing_physics=True)
        return

    assert match_count >= 1, f"no target match for case={case['id']}"
    assert decision in {"GO", "LOW_CONF_GO", "NO_GO", "HALT_NEED_SELECTION"}
    _assert_track_consistency(ctx, case["target_bond"])
    _assert_universal_checks(ctx, allow_missing_physics=False)

    # Regression: no flat route posteriors for C-H route evaluations.
    if str(case["target_bond"]).lower() == "c-h":
        route_post = job_card.get("route_posteriors") or []
        vals = [round(float(row.get("posterior")), 4) for row in route_post if isinstance(row.get("posterior"), (int, float))]
        if len(vals) > 1:
            assert len(set(vals)) > 1


def test_regression_toluene_no_halt_selection() -> None:
    ctx = run_pipeline("Cc1ccccc1", "C-H")
    job_card = ctx.data.get("job_card") or {}
    assert job_card.get("decision") != "HALT_NEED_SELECTION"

    m1 = _module_minus1_bundle(ctx)
    resolved = (m1["result"] or {}).get("resolved_target") or {}
    assert str(resolved.get("resolution_policy") or "").startswith("lowest_BDE_auto")
    assert (resolved.get("next_input_required") or []) == []

