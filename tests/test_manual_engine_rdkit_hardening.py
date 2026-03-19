from __future__ import annotations

import pytest


def test_atomic_truth_registry_handles_repaired_smiles():
    pytest.importorskip("rdkit")
    from enzyme_software.modules.sre_atr import AtomicTruthRegistry

    atr = AtomicTruthRegistry.from_smiles("cC")
    assert atr is not None
    assert len(atr._atoms) > 0


def test_module_minus1_hub_handles_repaired_smiles():
    pytest.importorskip("rdkit")
    from enzyme_software.modules.module_minus1_reactivity_hub import run_module_minus1_reactivity_hub

    result = run_module_minus1_reactivity_hub(
        smiles="cC",
        target_bond="C-H",
        requested_output=None,
        constraints={},
    )
    assert isinstance(result, dict)
    assert result.get("status") in {"FAIL", "OK", "PASS", "WARN"}
