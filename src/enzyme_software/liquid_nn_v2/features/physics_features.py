from __future__ import annotations

from collections import deque
from typing import Dict, List

from enzyme_software.liquid_nn_v2.data.bde_table import BDE_TABLE
from enzyme_software.liquid_nn_v2.data.smarts_patterns import FUNCTIONAL_GROUP_SMARTS
from enzyme_software.liquid_nn_v2.features.bond_classifier import EN_TABLE, bond_class_one_hot, classify_bond
from enzyme_software.liquid_nn_v2.features.group_detector import get_group_membership_vector


def _radical_stability_score(bond_class: str) -> float:
    if bond_class in {"benzylic", "allylic"}:
        return 0.9
    if bond_class == "tertiary_CH":
        return 0.7
    if bond_class == "secondary_CH":
        return 0.5
    if bond_class == "primary_CH":
        return 0.3
    return 0.4


def _nucleophilicity(atom) -> float:
    symbol = atom.GetSymbol()
    return {"N": 0.85, "O": 0.7, "S": 0.9, "C": 0.25}.get(symbol, 0.15)


def _electrophilicity(atom) -> float:
    symbol = atom.GetSymbol()
    base = {"C": 0.5, "N": 0.25, "O": 0.15, "S": 0.35}.get(symbol, 0.2)
    if atom.GetFormalCharge() > 0:
        base += 0.2
    return max(0.0, min(1.0, base))


def _heteroatom_distance(mol, atom_idx: int) -> float:
    if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() in {7, 8, 16}:
        return 0.0
    seen = {atom_idx}
    queue = deque([(atom_idx, 0)])
    while queue:
        current, dist = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        if current != atom_idx and atom.GetAtomicNum() in {7, 8, 16}:
            return min(10.0, float(dist)) / 10.0
        for nbr in atom.GetNeighbors():
            idx = nbr.GetIdx()
            if idx not in seen:
                seen.add(idx)
                queue.append((idx, dist + 1))
    return 1.0


def compute_atom_physics_features(mol, atom_idx: int, group_assignments: Dict[str, List[int]]) -> Dict[str, object]:
    atom = mol.GetAtomWithIdx(atom_idx)
    bond_class = classify_bond(atom, mol)
    bde = float(BDE_TABLE.get(bond_class, BDE_TABLE["other"]))
    return {
        "bond_class": bond_class,
        "bond_class_vector": bond_class_one_hot(bond_class),
        "bde": bde,
        "electronegativity": float(EN_TABLE.get(atom.GetSymbol(), 2.5)),
        "is_aromatic": float(atom.GetIsAromatic()),
        "functional_groups": get_group_membership_vector(atom_idx, group_assignments),
        "radical_stability": _radical_stability_score(bond_class),
        "nucleophilicity": _nucleophilicity(atom),
        "electrophilicity": _electrophilicity(atom),
        "heteroatom_distance": _heteroatom_distance(mol, atom_idx),
    }


def compute_molecule_physics_features(mol, group_assignments: Dict[str, List[int]]) -> Dict[str, object]:
    atom_payloads = [compute_atom_physics_features(mol, idx, group_assignments) for idx in range(mol.GetNumAtoms())]
    return {
        "bde_values": [row["bde"] for row in atom_payloads],
        "bond_classes": [row["bond_class_vector"] for row in atom_payloads],
        "electronegativity": [row["electronegativity"] for row in atom_payloads],
        "is_aromatic": [row["is_aromatic"] for row in atom_payloads],
        "functional_groups": [row["functional_groups"] for row in atom_payloads],
        "radical_stability": [row["radical_stability"] for row in atom_payloads],
        "nucleophilicity": [row["nucleophilicity"] for row in atom_payloads],
        "electrophilicity": [row["electrophilicity"] for row in atom_payloads],
        "heteroatom_distance": [row["heteroatom_distance"] for row in atom_payloads],
        "bond_class_labels": [row["bond_class"] for row in atom_payloads],
        "group_names": list(FUNCTIONAL_GROUP_SMARTS.keys()),
    }
