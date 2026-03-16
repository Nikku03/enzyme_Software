from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None

from enzyme_software.liquid_nn_v2.data.bde_table import BDE_MAX, BDE_MIN
from enzyme_software.liquid_nn_v2.features.bond_classifier import atom_type_one_hot, classify_bond, hybridization_one_hot
from enzyme_software.liquid_nn_v2.features.group_detector import get_group_membership_vector
from enzyme_software.liquid_nn_v2.features.physics_features import _electrophilicity, _heteroatom_distance, _nucleophilicity, _radical_stability_score


def _safe_partial_charges(mol) -> List[float]:
    try:
        work = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(work)
        return [float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0 for atom in work.GetAtoms()]
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def _morgan_bit_info(mol, radius: int = 2, n_bits: int = 64) -> Dict[int, List[Tuple[int, int]]]:
    info: Dict[int, List[Tuple[int, int]]] = {}
    try:
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=info)
    except Exception:
        return {}
    return info


def get_atom_morgan_bits(atom_idx: int, bit_info: Dict[int, List[Tuple[int, int]]], n_bits: int = 64) -> List[float]:
    bits = [0.0] * n_bits
    for bit_idx, centers in bit_info.items():
        if any(center_atom == atom_idx for center_atom, _ in centers):
            bits[int(bit_idx)] = 1.0
    return bits


def _local_environment_features(mol, atom_idx: int) -> List[float]:
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = list(atom.GetNeighbors())
    counts = {
        "C": 0,
        "N": 0,
        "O": 0,
        "halogen": 0,
        "aromatic": 0,
        "sp3": 0,
        "sp2": 0,
        "ring": 0,
    }
    en_values: List[float] = []
    bond_order_sum = 0.0
    single = double = aromatic_bonds = 0
    for nbr in neighbors:
        symbol = nbr.GetSymbol()
        if symbol == "C":
            counts["C"] += 1
        if symbol == "N":
            counts["N"] += 1
        if symbol == "O":
            counts["O"] += 1
        if symbol in {"F", "Cl", "Br", "I"}:
            counts["halogen"] += 1
        if nbr.GetIsAromatic():
            counts["aromatic"] += 1
        if str(nbr.GetHybridization()) == "SP3":
            counts["sp3"] += 1
        if str(nbr.GetHybridization()) == "SP2":
            counts["sp2"] += 1
        if nbr.IsInRing():
            counts["ring"] += 1
        bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
        order = float(bond.GetBondTypeAsDouble()) if bond is not None else 1.0
        bond_order_sum += order
        if bond is not None and bond.GetIsAromatic():
            aromatic_bonds += 1
        elif abs(order - 1.0) < 1e-6:
            single += 1
        elif abs(order - 2.0) < 1e-6:
            double += 1
        en_values.append({"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "P": 2.19}.get(symbol, 2.5))

    avg_en = sum(en_values) / len(en_values) if en_values else 0.0
    max_en = max(en_values) if en_values else 0.0
    donor_score = 0.0
    withdrawer_score = 0.0
    for nbr in neighbors:
        delta = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "P": 2.19}.get(nbr.GetSymbol(), 2.5) - 2.55
        if delta < 0:
            donor_score += abs(delta)
        else:
            withdrawer_score += delta

    ring_bonds = [bond for bond in atom.GetBonds() if bond.IsInRing()]
    is_bridgehead = float(atom.IsInRing() and len(ring_bonds) >= 3)
    is_ring_junction = float(atom.IsInRing() and len({nbr.GetIdx() for nbr in neighbors if nbr.IsInRing()}) >= 2)
    return [
        counts["C"] / 4.0,
        counts["N"] / 4.0,
        counts["O"] / 4.0,
        counts["halogen"] / 4.0,
        counts["aromatic"] / 4.0,
        counts["sp3"] / 4.0,
        counts["sp2"] / 4.0,
        counts["ring"] / 4.0,
        avg_en / 4.0,
        max_en / 4.0,
        bond_order_sum / 6.0,
        single / 4.0,
        double / 2.0,
        aromatic_bonds / 3.0,
        is_ring_junction,
        is_bridgehead,
        max(-1.0, min(1.0, donor_score / 2.0)),
        max(-1.0, min(1.0, withdrawer_score / 2.0)),
    ]


def extract_atom_features(mol, group_assignments: Dict[str, List[int]]) -> Tuple[np.ndarray, Dict[str, object]]:
    if Chem is None or mol is None:
        raise RuntimeError("RDKit is required for feature extraction")
    partial_charges = _safe_partial_charges(mol)
    bit_info = _morgan_bit_info(mol)
    rows: List[List[float]] = []
    bond_classes: List[str] = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        bond_class = classify_bond(atom, mol)
        bond_classes.append(bond_class)
        bde = {"benzylic": 375.5, "allylic": 371.5, "alpha_hetero": 385.0, "tertiary_CH": 403.8, "secondary_CH": 412.5, "primary_CH": 423.0, "aryl": 472.2, "amine_NH": 386.0, "amide_NH": 440.0, "alcohol_OH": 435.7, "phenol_OH": 362.8, "other": 410.0}.get(bond_class, 410.0)
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings() if atom_idx in ring]
        row: List[float] = []
        row.extend(atom_type_one_hot(atom))
        row.extend(hybridization_one_hot(atom))
        row.extend([1.0 if bond_class == name else 0.0 for name in ["benzylic", "allylic", "alpha_hetero", "aryl", "primary_CH", "secondary_CH", "tertiary_CH", "amine_NH", "amide_NH", "alcohol_OH", "phenol_OH", "other"]])
        row.extend(get_group_membership_vector(atom_idx, group_assignments))
        row.extend([
            max(0.0, min(1.0, (bde - BDE_MIN) / (BDE_MAX - BDE_MIN))),
            partial_charges[atom_idx],
            {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "P": 2.19}.get(atom.GetSymbol(), 2.5) / 4.0,
            atom.GetDegree() / 4.0,
            atom.GetTotalNumHs() / 4.0,
            atom.GetFormalCharge() / 2.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            (min(ring_sizes) if ring_sizes else 0.0) / 8.0,
            len(ring_sizes) / 3.0,
            _radical_stability_score(bond_class),
            _nucleophilicity(atom),
            _electrophilicity(atom),
            _heteroatom_distance(mol, atom_idx),
        ])
        row.extend(get_atom_morgan_bits(atom_idx, bit_info))
        row.extend(_local_environment_features(mol, atom_idx))
        rows.append(row)
    array = np.asarray(rows, dtype=np.float32)
    return array, {"bond_classes": bond_classes, "feature_dim": int(array.shape[1])}
