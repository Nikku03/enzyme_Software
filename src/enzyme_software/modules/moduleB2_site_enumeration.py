from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def _is_allylic(atom: Any) -> bool:
    if atom.GetIsAromatic():
        return False
    for nbr in atom.GetNeighbors():
        for bond in nbr.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(nbr)
                if other.GetIdx() != atom.GetIdx() and other.GetSymbol() == "C":
                    return True
    return False


def _ch_site_class(atom: Any) -> str:
    if atom.GetIsAromatic():
        return "aromatic_ch"
    if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
        return "benzylic_ch"
    if _is_allylic(atom):
        return "allylic_ch"
    if any(nbr.GetSymbol() in {"O", "N", "S"} for nbr in atom.GetNeighbors()):
        return "alpha_hetero_ch"
    carbon_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C")
    if carbon_neighbors >= 3:
        return "aliphatic_tertiary_ch"
    if carbon_neighbors == 2:
        return "aliphatic_secondary_ch"
    return "aliphatic_primary_ch"


def _reaction_class_for_site(site_class: str) -> str:
    mapping = {
        "benzylic_ch": "benzylic_hydroxylation",
        "allylic_ch": "allylic_hydroxylation",
        "aromatic_ch": "aromatic_hydroxylation",
        "alpha_hetero_ch": "alpha_hetero_hydroxylation",
        "aliphatic_primary_ch": "aliphatic_hydroxylation",
        "aliphatic_secondary_ch": "aliphatic_hydroxylation",
        "aliphatic_tertiary_ch": "aliphatic_hydroxylation",
        "o_demethyl": "o_demethylation",
        "o_dealkyl": "o_dealkylation",
        "n_demethyl": "n_demethylation",
        "n_dealkyl": "n_dealkylation",
    }
    return mapping.get(site_class, "oxidation")


def _dealkyl_site_class(hetero_atom: Any, carbon_atom: Any) -> str:
    symbol = hetero_atom.GetSymbol()
    degree_h = sum(1 for nbr in carbon_atom.GetNeighbors() if nbr.GetSymbol() == "H")
    if degree_h <= 0:
        degree_h = int(carbon_atom.GetTotalNumHs())
    if symbol == "O" and degree_h >= 3:
        return "o_demethyl"
    if symbol == "N" and degree_h >= 3:
        return "n_demethyl"
    if symbol == "O":
        return "o_dealkyl"
    if symbol == "N":
        return "n_dealkyl"
    return "dealkylation"


def enumerate_metabolism_targets(mol: Any, mode: str = "default") -> List[Dict[str, Any]]:
    """Enumerate candidate metabolism sites (atom/bond level)."""
    if mol is None or Chem is None:
        return []

    candidates: List[Dict[str, Any]] = []
    mol_h = Chem.AddHs(Chem.Mol(mol))

    # C-H candidates (one site per heavy atom bearing hydrogens)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        atom_h = mol_h.GetAtomWithIdx(atom.GetIdx())
        h_neighbors = [nbr for nbr in atom_h.GetNeighbors() if nbr.GetSymbol() == "H"]
        if not h_neighbors:
            continue
        site_class = _ch_site_class(atom)
        first_h = h_neighbors[0].GetIdx()
        candidates.append(
            {
                "site_id": f"atom:{atom.GetIdx()}",
                "site_type": "atom",
                "atom_indices": [int(atom.GetIdx())],
                "bond_indices": [int(atom.GetIdx()), int(first_h)],
                "anchor_atom_indices": [int(atom.GetIdx())],
                "site_class": site_class,
                "reaction_class": _reaction_class_for_site(site_class),
                "h_count": len(h_neighbors),
                "feasibility_flags": [],
            }
        )

    # O/N dealkylation candidates (hetero-carbon bond sites)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in {"O", "N"}:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() != "C":
                continue
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                continue
            if nbr.GetIsAromatic():
                continue
            nbr_h = mol_h.GetAtomWithIdx(nbr.GetIdx())
            h_count = sum(1 for h_nbr in nbr_h.GetNeighbors() if h_nbr.GetSymbol() == "H")
            if h_count < 1 and int(nbr_h.GetTotalNumHs()) < 1:
                continue
            site_class = _dealkyl_site_class(atom, nbr_h)
            candidates.append(
                {
                    "site_id": f"bond:{atom.GetIdx()}-{nbr.GetIdx()}",
                    "site_type": "bond",
                    "atom_indices": [int(atom.GetIdx()), int(nbr.GetIdx())],
                    "bond_indices": [int(atom.GetIdx()), int(nbr.GetIdx())],
                    "anchor_atom_indices": [int(atom.GetIdx()), int(nbr.GetIdx())],
                    "site_class": site_class,
                    "reaction_class": _reaction_class_for_site(site_class),
                    "feasibility_flags": [],
                }
            )

    if not candidates:
        return [
            {
                "site_id": "none",
                "site_type": "atom",
                "atom_indices": [],
                "bond_indices": [],
                "anchor_atom_indices": [],
                "site_class": "none",
                "reaction_class": "none",
                "feasibility_flags": ["no_metabolism_site_detected"],
            }
        ]

    # Deterministic ordering
    candidates.sort(key=lambda c: (c.get("site_type", ""), str(c.get("site_id", ""))))
    return candidates


def enumerate_metabolism_targets_from_smiles(smiles: str, mode: str = "default") -> List[Dict[str, Any]]:
    if Chem is None:
        return []
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    return enumerate_metabolism_targets(mol, mode=mode)
