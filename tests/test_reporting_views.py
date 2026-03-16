from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints
from enzyme_software.pipeline import run_pipeline
from enzyme_software.reporting import render_demo
from enzyme_software.modules.module0_strategy_router import Chem


def test_demo_view_line_count():
    if Chem is None:
        pytest.skip("RDKit not available")
    ctx = run_pipeline(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(ph_min=7.25, ph_max=7.25, temperature_c=30.0),
    )
    report = render_demo(ctx.to_dict())
    assert len(report.splitlines()) < 150
