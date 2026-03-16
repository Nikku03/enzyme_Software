from __future__ import annotations

from enzyme_software.mechanism_registry import resolve_mechanism
from enzyme_software.modules import module2_active_site_refinement as m2
from enzyme_software.modules.module0_strategy_router import Chem
from enzyme_software.pipeline import run_pipeline
from enzyme_software.context import OperationalConstraints


def test_serine_contract_geometry_is_serine_og():
    contract = resolve_mechanism("serine_hydrolase").to_dict()
    geometry = m2._geometry_from_contract(contract)
    assert geometry == "serine_og"
    assert geometry != "cysteine_thiol"


def test_aspirin_serine_route_uses_serine_geometry():
    if Chem is None:
        return
    ctx = run_pipeline(
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "acetyl_ester_C-O",
        requested_output="salicylic acid",
        constraints=OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0),
    )
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    selected = module2.get("selected_scaffold") or {}
    reach_summary = selected.get("reach_summary") or {}
    assert reach_summary.get("nucleophile_geometry") == "serine_og"
