from __future__ import annotations

from typing import Any, Dict

import pytest

from enzyme_software.context import OperationalConstraints
from enzyme_software.modules import module0_strategy_router as m0


def _minimal_inputs() -> Dict[str, Any]:
    bond_context = {
        "bond_class": "ester",
        "primary_role": "ester__acyl_o",
        "primary_role_confidence": 0.9,
        "polarity": "polar",
    }
    structure_summary = {"atom_count": 10, "hetero_atoms": 3, "ring_count": 0}
    reaction_intent = {"intent_type": "hydrolysis"}
    constraints = OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0)
    return {
        "bond_context": bond_context,
        "structure_summary": structure_summary,
        "reaction_intent": reaction_intent,
        "constraints": constraints,
    }


def test_module0_uses_physics_prior_when_no_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _minimal_inputs()
    condition_profile = m0._condition_profile_from_constraints(inputs["constraints"])

    def _no_calibration(*args: Any, **kwargs: Any):
        return None, None, None, {}

    monkeypatch.setattr(m0, "_calibrated_any_activity_probability", _no_calibration)
    audit = m0._physics_route_audit(
        smiles="CC(=O)O",
        target_bond="ester__acyl_o",
        bond_context=inputs["bond_context"],
        structure_summary=inputs["structure_summary"],
        reaction_intent=inputs["reaction_intent"],
        condition_profile=condition_profile,
        mechanism_family="serine_hydrolase",
        known_scaffold=False,
    )
    assert audit.get("prior_any_activity_calibrated") is None
    assert audit.get("route_prior_any_activity") == audit.get("prior_any_activity_heuristic")


def test_module0_uses_evidence_calibrated_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _minimal_inputs()
    condition_profile = m0._condition_profile_from_constraints(inputs["constraints"])

    def _fake_calibration(*args: Any, **kwargs: Any):
        return 0.9, "artifacts/calibration_module0_v1.json", "v1", {
            "metrics": {"sources": {"ord": 10}, "n_samples": 10}
        }

    monkeypatch.setattr(m0, "_calibrated_any_activity_probability", _fake_calibration)
    audit = m0._physics_route_audit(
        smiles="CC(=O)O",
        target_bond="ester__acyl_o",
        bond_context=inputs["bond_context"],
        structure_summary=inputs["structure_summary"],
        reaction_intent=inputs["reaction_intent"],
        condition_profile=condition_profile,
        mechanism_family="serine_hydrolase",
        known_scaffold=False,
    )
    assert audit.get("prior_any_activity_calibrated") is not None
    assert audit.get("calibration_sources") == {"ord": 10}
    assert audit.get("route_prior_any_activity") != audit.get("prior_any_activity_heuristic")
