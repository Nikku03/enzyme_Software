from __future__ import annotations

from typing import List

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Other"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "Other"]
BOND_CLASSES = [
    "benzylic",
    "allylic",
    "alpha_hetero",
    "aryl",
    "primary_CH",
    "secondary_CH",
    "tertiary_CH",
    "amine_NH",
    "amide_NH",
    "alcohol_OH",
    "phenol_OH",
    "other",
]


EN_TABLE = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "P": 2.19}


def one_hot(value: str, choices: List[str]) -> List[float]:
    target = value if value in choices else choices[-1]
    return [1.0 if choice == target else 0.0 for choice in choices]


def atom_type_one_hot(atom) -> List[float]:
    symbol = atom.GetSymbol() if atom is not None else "Other"
    if symbol not in ATOM_TYPES[:-1]:
        symbol = "Other"
    return one_hot(symbol, ATOM_TYPES)


def hybridization_one_hot(atom) -> List[float]:
    hyb = str(atom.GetHybridization()) if atom is not None else "Other"
    hyb = hyb if hyb in HYBRIDIZATIONS[:-1] else "Other"
    return one_hot(hyb, HYBRIDIZATIONS)


def bond_class_one_hot(bond_class: str) -> List[float]:
    return one_hot(bond_class if bond_class in BOND_CLASSES[:-1] else "other", BOND_CLASSES)


def classify_bond(atom, mol) -> str:
    if atom is None:
        return "other"
    symbol = atom.GetSymbol()
    neighbors = list(atom.GetNeighbors())
    total_h = int(atom.GetTotalNumHs())

    if symbol == "O" and total_h > 0:
        return "phenol_OH" if any(n.GetIsAromatic() for n in neighbors) else "alcohol_OH"

    if symbol == "N" and total_h > 0:
        if any(
            nbr.GetAtomicNum() == 6 and any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(nbr).GetAtomicNum() == 8 for b in nbr.GetBonds())
            for nbr in neighbors
        ):
            return "amide_NH"
        return "amine_NH"

    if symbol != "C":
        return "other"

    aromatic_neighbor = any(n.GetIsAromatic() for n in neighbors)
    if atom.GetIsAromatic() and total_h > 0:
        return "aryl"
    if aromatic_neighbor and total_h > 0:
        return "benzylic"

    if any(
        n.GetAtomicNum() == 6 and n.GetHybridization() == Chem.HybridizationType.SP2 and not n.GetIsAromatic()
        for n in neighbors
    ) and total_h > 0:
        return "allylic"

    if any(n.GetAtomicNum() in {7, 8, 16} for n in neighbors):
        return "alpha_hetero"

    heavy_degree = sum(1 for n in neighbors if n.GetAtomicNum() > 1)
    if total_h > 0:
        if heavy_degree <= 1:
            return "primary_CH"
        if heavy_degree == 2:
            return "secondary_CH"
        return "tertiary_CH"

    return "other"
