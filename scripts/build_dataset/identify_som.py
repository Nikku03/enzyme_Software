from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2.features.bond_classifier import classify_bond


def identify_som_from_metabolite(substrate_smiles: str, metabolite_smiles: str) -> List[int]:
    """
    Identify likely modified atom indices in the substrate via MCS comparison.
    """
    sub_mol = Chem.MolFromSmiles(substrate_smiles)
    met_mol = Chem.MolFromSmiles(metabolite_smiles)
    if sub_mol is None or met_mol is None:
        return []

    mcs = rdFMCS.FindMCS(
        [sub_mol, met_mol],
        timeout=10,
        matchValences=False,
        ringMatchesRingOnly=True,
    )
    if not mcs.smartsString:
        return []

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return []

    sub_match = set(sub_mol.GetSubstructMatch(mcs_mol))
    modified = [idx for idx in range(sub_mol.GetNumAtoms()) if idx not in sub_match]

    som_candidates = set(modified)
    for atom_idx in modified:
        atom = sub_mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "C":
                som_candidates.add(neighbor.GetIdx())
    return sorted(som_candidates)


def identify_som_from_reaction_type(
    smiles: str,
    reaction_type: str,
    cyp: str | None = None,
) -> List[Tuple[int, str]]:
    """
    Heuristic SoM identification from substrate structure and reaction type.
    """
    del cyp  # reserved for future route-specific heuristics
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    som_candidates: List[Tuple[int, str, float]] = []
    reaction = (reaction_type or "").lower()

    if reaction == "hydroxylation":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C" or atom.GetTotalNumHs() <= 0:
                continue
            bond_class = classify_bond(atom, mol)
            score = {
                "benzylic": 0.9,
                "allylic": 0.85,
                "tertiary_CH": 0.75,
                "secondary_CH": 0.65,
                "primary_CH": 0.55,
                "aryl": 0.4,
            }.get(bond_class, 0.3)
            som_candidates.append((atom.GetIdx(), bond_class, score))

    elif reaction == "n_dealkylation":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            if any(n.GetSymbol() == "N" for n in atom.GetNeighbors()):
                som_candidates.append((atom.GetIdx(), "alpha_hetero", 0.85))

    elif reaction == "o_dealkylation":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            if any(n.GetSymbol() == "O" and not n.GetIsAromatic() for n in atom.GetNeighbors()):
                som_candidates.append((atom.GetIdx(), "alpha_hetero", 0.85))

    elif reaction == "n_oxidation":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N":
                som_candidates.append((atom.GetIdx(), "other", 0.7))

    elif reaction == "s_oxidation":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "S":
                som_candidates.append((atom.GetIdx(), "other", 0.7))

    deduped: List[Tuple[int, str]] = []
    seen = set()
    for atom_idx, bond_class, _score in sorted(som_candidates, key=lambda item: item[2], reverse=True):
        if atom_idx in seen:
            continue
        seen.add(atom_idx)
        deduped.append((atom_idx, bond_class))
    return deduped[:3]


def label_som_indices(smiles: str, atom_indices: Sequence[int]) -> List[dict]:
    """Attach bond-class labels to atom indices for storage."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    labeled = []
    for atom_idx in atom_indices:
        if 0 <= int(atom_idx) < mol.GetNumAtoms():
            atom = mol.GetAtomWithIdx(int(atom_idx))
            labeled.append({"atom_idx": int(atom_idx), "bond_class": classify_bond(atom, mol)})
    return labeled
