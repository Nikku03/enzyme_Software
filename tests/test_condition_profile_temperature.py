from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints
from enzyme_software.pipeline import run_pipeline
from enzyme_software.modules.module0_strategy_router import Chem


def test_condition_profile_temperature_from_constraints():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = run_pipeline(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=7.2, ph_max=7.3, temperature_c=30.0),
    )
    shared = ctx.data.get("shared_io") or {}
    condition_profile = (shared.get("input") or {}).get("condition_profile") or {}
    assert condition_profile.get("temperature_K") == pytest.approx(303.15, rel=1e-3)
    assert condition_profile.get("temperature_C") == pytest.approx(30.0, rel=1e-3)
