from __future__ import annotations

import math
import pytest

from enzyme_software.context import OperationalConstraints
from enzyme_software.pipeline import run_pipeline
from enzyme_software.modules.module0_strategy_router import Chem


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def test_aspirin_pipeline_physics_audit():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = run_pipeline(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=7.25, ph_max=7.25, temperature_c=30.0),
    )
    job_card = ctx.data.get("job_card") or {}
    physics_audit = job_card.get("physics_audit") or {}
    assert physics_audit.get("temperature_K") == pytest.approx(303.15, rel=1e-3)
    assert physics_audit.get("deltaG_dagger_kJ_per_mol") is not None
    assert physics_audit.get("eyring_k_s_inv") is not None
    for key in ("temperature_K", "deltaG_dagger_kJ_per_mol", "eyring_k_s_inv"):
        assert _is_finite(physics_audit.get(key))

    prior = physics_audit.get("prior_success_probability")
    assert prior is not None
    assert 0.0 <= float(prior) <= 1.0
    assert float(prior) != 0.5

    module2 = ctx.data.get("module2_active_site_refinement") or {}
    module2_physics = module2.get("module2_physics_audit") or {}
    assert module2_physics.get("k_variant_s_inv") is not None
    assert _is_finite(module2_physics.get("k_variant_s_inv"))
