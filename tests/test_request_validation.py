from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.modules.module0_strategy_router import Module0StrategyRouter, Chem


def test_bond_token_not_present_halts():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = PipelineContext(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="amide_C-N",
        requested_output=None,
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    ctx = Module0StrategyRouter().run(ctx)
    job_card = ctx.data.get("job_card") or {}
    assert job_card.get("decision") == "HALT"
    assert job_card.get("pipeline_halt_reason") == "BOND_TOKEN_NOT_PRESENT"


def test_requested_output_mismatch_halts():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = PipelineContext(
        smiles="CC(=O)NC",
        target_bond="amide_C-N",
        requested_output="salicylic acid",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    ctx = Module0StrategyRouter().run(ctx)
    job_card = ctx.data.get("job_card") or {}
    assert job_card.get("decision") == "HALT"
    assert job_card.get("pipeline_halt_reason") == "REQUEST_OUTPUT_MISMATCH"
