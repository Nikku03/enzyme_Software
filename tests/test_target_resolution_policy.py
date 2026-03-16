from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.modules.module0_strategy_router import Module0StrategyRouter, Chem


def test_target_resolution_policy_allows_single_match():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = PipelineContext(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    ctx = Module0StrategyRouter().run(ctx)
    job_card = ctx.data.get("job_card") or {}
    resolved = job_card.get("resolved_target") or {}
    assert resolved.get("match_count") == 1
    selected = resolved.get("selected_bond") or {}
    atom_indices = selected.get("atom_indices") or []
    assert set(atom_indices) == {1, 3}
    token_audit = job_card.get("token_resolution_audit") or {}
    assert token_audit.get("canonical_token") == "ester__acyl_o__acetyl"
    assert token_audit.get("smarts_used")
    confidence = (job_card.get("confidence") or {}).get("target_resolution")
    assert confidence is not None
    assert confidence > 0.85
    decision = str(job_card.get("decision") or "")
    assert not decision.startswith("HALT")


def test_target_bond_indices_override_confidence():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = PipelineContext(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="[1,3]",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    ctx = Module0StrategyRouter().run(ctx)
    job_card = ctx.data.get("job_card") or {}
    resolved = job_card.get("resolved_target") or {}
    assert resolved.get("selection_mode") == "atom_indices"
    assert resolved.get("match_count") == 1
    confidence = (job_card.get("confidence") or {}).get("target_resolution")
    assert confidence == pytest.approx(0.99, rel=1e-6)
