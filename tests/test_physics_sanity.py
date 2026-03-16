from __future__ import annotations

import math

from enzyme_software.context import OperationalConstraints
from enzyme_software.pipeline import run_pipeline
from enzyme_software.physicscore import (
    eyring_rate_constant,
    fraction_deprotonated_acid,
    fraction_protonated_base,
    kj_to_j,
    smoluchowski_kdiff,
)


def test_eyring_monotonicity() -> None:
    temp_k = 298.15
    low_barrier = eyring_rate_constant(kj_to_j(60.0), temp_k)
    high_barrier = eyring_rate_constant(kj_to_j(100.0), temp_k)
    assert low_barrier > high_barrier


def test_hh_fraction_bounds() -> None:
    frac_acid = fraction_deprotonated_acid(7.0, 5.0)
    frac_base = fraction_protonated_base(7.0, 9.0)
    assert 0.0 <= frac_acid <= 1.0
    assert 0.0 <= frac_base <= 1.0


def test_diffusion_cap_positive() -> None:
    kdiff = smoluchowski_kdiff(
        298.15, 1.0e-3, radius_m=2.0e-10, encounter_radius_m=4.0e-10
    )
    assert math.isfinite(kdiff)
    assert kdiff > 0.0


def test_module0_physics_smoke_aspirin() -> None:
    ctx = run_pipeline(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        requested_output="salicylic acid",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    assert "job_card" in ctx.data
    assert "shared_io" in ctx.data
    assert "module0_strategy_router" in ctx.data
    job_card = ctx.data["job_card"]
    for key in ("decision", "confidence", "mechanism_route", "compute_plan"):
        assert key in job_card
    assert "physics" in job_card
    assert "physics_audit" in job_card
    delta_g = job_card["physics_audit"].get("deltaG_dagger_kJ_per_mol")
    assert delta_g is not None
    assert 60.0 <= float(delta_g) <= 110.0
