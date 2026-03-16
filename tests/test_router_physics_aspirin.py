from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.modules.module0_strategy_router import Module0StrategyRouter, Chem


def test_router_physics_layer_aspirin():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = PipelineContext(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    ctx = Module0StrategyRouter().run(ctx)
    job_card = ctx.data.get("job_card") or {}
    physics = job_card.get("physics_layer") or {}
    assert physics.get("kT_kj_per_mol") is not None
    assert physics.get("temperature_K") is not None
