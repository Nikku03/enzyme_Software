from __future__ import annotations

from typing import Optional, Tuple

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


def sanitize_smiles(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    result = prepare_mol(smiles)
    if result.mol is None:
        return None, result.error or "Failed to parse or sanitize SMILES"
    return result.canonical_smiles, None


def safe_mol_from_smiles(smiles: str):
    result = prepare_mol(smiles)
    return result.mol, result.status
