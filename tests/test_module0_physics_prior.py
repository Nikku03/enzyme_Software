from __future__ import annotations

import pytest

from enzyme_software.physicscore import (
    estimate_deltaG_dagger_for_bond,
    physics_prior_success_probability,
)
from enzyme_software.modules.module0_strategy_router import Module0StrategyRouter, Chem
from enzyme_software.context import OperationalConstraints, PipelineContext


def test_physics_prior_ester_vs_amide():
    temp_k = 303.15
    dg_ester = estimate_deltaG_dagger_for_bond("ester", "hydrolysis", {})
    dg_amide = estimate_deltaG_dagger_for_bond("amide", "hydrolysis", {})
    p_ester = physics_prior_success_probability(dg_ester, temp_k, 3600.0)
    p_amide = physics_prior_success_probability(dg_amide, temp_k, 3600.0)
    assert p_ester > p_amide
    assert p_ester > 0.1
    assert p_amide < 0.1


def test_module0_physics_audit_aspirin():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = PipelineContext(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    ctx = Module0StrategyRouter().run(ctx)
    job_card = ctx.data.get("job_card") or {}
    physics_audit = job_card.get("physics_audit") or {}
    prior = physics_audit.get("prior_success_probability")
    assert prior is not None
    assert float(prior) > 0.1
