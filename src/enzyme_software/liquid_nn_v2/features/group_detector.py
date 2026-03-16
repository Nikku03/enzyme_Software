from __future__ import annotations

from typing import Dict, List

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

from enzyme_software.liquid_nn_v2.data.smarts_patterns import FUNCTIONAL_GROUP_SMARTS


def detect_functional_groups(mol) -> Dict[str, List[int]]:
    results = {name: [] for name in FUNCTIONAL_GROUP_SMARTS}
    if Chem is None or mol is None:
        return results
    for group_name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            for atom_idx in match:
                if atom_idx not in results[group_name]:
                    results[group_name].append(int(atom_idx))
    return results


def get_group_membership_vector(atom_idx: int, group_assignments: Dict[str, List[int]]) -> List[float]:
    return [1.0 if atom_idx in group_assignments.get(group_name, []) else 0.0 for group_name in FUNCTIONAL_GROUP_SMARTS]
