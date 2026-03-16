from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.mechanism_registry import resolve_mechanism
from enzyme_software.modules.module0_strategy_router import Chem
from enzyme_software.modules.module2_active_site_refinement import (
    _mechanism_mismatch_from_contract,
    _module2_variant_physics_audit,
    _resolve_mechanism_evidence,
)
from enzyme_software.modules.module3_experiment_designer import _plan_score_physics
from enzyme_software.pipeline import run_pipeline
from enzyme_software.reporting import render_scientist


def test_report_single_baseline_values():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = run_pipeline(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=7.25, ph_max=7.25, temperature_c=30.0),
    )
    report = render_scientist(ctx.to_dict())
    assert report.count("Baseline ΔG‡") == 1
    assert report.count("Baseline k_Eyring") == 1


def test_mechanism_unverified_status():
    contract = resolve_mechanism("serine_hydrolase").to_dict()
    evidence = _resolve_mechanism_evidence(contract, {}, None)
    mismatch = _mechanism_mismatch_from_contract(contract, None)
    assert evidence.get("status") == "UNVERIFIED"
    assert mismatch.get("status") == "UNVERIFIED"


def test_plan_phys_scaling_with_low_signal():
    ctx = PipelineContext(smiles="CCO", target_bond="C-O")
    ctx.data["job_card"] = {
        "confidence": {"route": 0.3, "target_resolution": 1.0, "wetlab_prior": 1.0}
    }
    ctx.data["shared_io"] = {
        "energy_ledger": {"p_success_horizon": 0.2, "k_eff_s_inv": 1e-3, "horizon_s": 3600}
    }
    ctx.data["unity_state"] = {"mechanism": {"evidence": {"status": "UNVERIFIED"}}}
    score, audit = _plan_score_physics(ctx, {"arms": []})
    assert audit.get("overall_signal") is not None
    assert audit.get("overall_signal") < 0.4
    assert audit.get("expected_signal") < 0.95
    assert audit.get("plan_phys") < 1.0
    assert score >= 0.0


def test_module2_uses_energy_ledger_baseline():
    job_card = {"energy_ledger": {"deltaG_dagger_kJ": 90.0}, "condition_profile": {"temperature_K": 303.15}}
    audit = _module2_variant_physics_audit(
        selected={"delta_g_mean": 10.0},
        best_variant=None,
        job_card=job_card,
        condition_assessment={"given_conditions": {"temperature_c": 30.0}},
        nucleophile_geometry="serine_og",
    )
    assert audit.get("baseline_source") == "energy_ledger"
    assert audit.get("deltaG_dagger_baseline_kJ_per_mol") == 90.0
