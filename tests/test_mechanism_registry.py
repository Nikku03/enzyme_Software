from __future__ import annotations

import json

from enzyme_software.mechanism_registry import resolve_mechanism
from enzyme_software.unity_layer import build_shared_state, export_shared_io_patch, set_mechanism_contract


def test_mechanism_registry_serine_expected():
    spec = resolve_mechanism("serine_hydrolase")
    assert spec.expected_nucleophile == "Ser"
    assert "serine_og" in spec.allowed_nucleophile_geometries


def test_mechanism_contract_serializes():
    shared_state = build_shared_state(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        requested_output=None,
        trap_target=None,
        constraints={},
    )
    contract = resolve_mechanism("serine_hydrolase").to_dict()
    set_mechanism_contract(shared_state, contract, {"status": "pending"}, source="test")
    payload = export_shared_io_patch(shared_state)
    assert "mechanism" in payload
    json.dumps(payload)
