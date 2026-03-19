from __future__ import annotations

import pytest

Chem = pytest.importorskip("rdkit.Chem")

from enzyme_software.utils.smiles_utils import safe_mol_from_smiles


def test_safe_mol_from_smiles_valid():
    mol, warnings = safe_mol_from_smiles("CCO")
    assert mol is not None
    assert warnings == []


def test_safe_mol_from_smiles_invalid_aromatic_quarantines_or_repairs():
    mol, warnings = safe_mol_from_smiles("c")
    if mol is None:
        assert any("QUARANTINED" in warning for warning in warnings)
    else:
        assert warnings


def test_safe_mol_from_smiles_malformed_fails_cleanly():
    mol, warnings = safe_mol_from_smiles("not_a_smiles")
    assert mol is None
    assert any("QUARANTINED" in warning for warning in warnings)
