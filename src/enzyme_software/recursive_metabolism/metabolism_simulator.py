from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


class MetabolismType(str, Enum):
    HYDROXYLATION = "hydroxylation"
    N_DEALKYLATION = "n_dealkylation"
    O_DEALKYLATION = "o_dealkylation"
    EPOXIDATION = "epoxidation"
    N_OXIDATION = "n_oxidation"
    SULFOXIDATION = "sulfoxidation"
    UNKNOWN = "unknown"


@dataclass
class MetabolismResult:
    success: bool
    parent_smiles: str
    metabolite_smiles: Optional[str]
    site_atom_idx: int
    metabolism_type: MetabolismType
    atoms_removed: int
    atoms_added: int
    error: Optional[str] = None


class MetabolismSimulator:
    def __init__(self):
        if Chem is not None:
            self.patterns = {
                "n_methyl": Chem.MolFromSmarts("[NX3;H0,H1][CH3]"),
                "o_methyl": Chem.MolFromSmarts("[OX2][CH3]"),
                "alkene": Chem.MolFromSmarts("[C]=[C]"),
            }
        else:  # pragma: no cover
            self.patterns = {}

    def metabolize(
        self,
        smiles: str,
        atom_idx: int,
        metabolism_type: Optional[MetabolismType] = None,
    ) -> MetabolismResult:
        if Chem is None:
            return MetabolismResult(False, smiles, None, atom_idx, MetabolismType.UNKNOWN, 0, 0, error="RDKit unavailable")
        prep = prepare_mol(smiles)
        if prep.mol is None:
            return MetabolismResult(False, smiles, None, atom_idx, MetabolismType.UNKNOWN, 0, 0, error=prep.error or "invalid_smiles")
        mol = prep.mol
        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            return MetabolismResult(False, smiles, None, atom_idx, MetabolismType.UNKNOWN, 0, 0, error="site_atom_out_of_range")

        chosen_type = metabolism_type or self.detect_metabolism_type(mol, atom_idx)
        try:
            metabolite = self._apply_reaction(mol, atom_idx, chosen_type)
        except Exception as exc:
            return MetabolismResult(False, smiles, None, atom_idx, chosen_type, 0, 0, error=str(exc))
        if metabolite is None:
            return MetabolismResult(False, smiles, None, atom_idx, chosen_type, 0, 0, error="reaction_failed")
        try:
            metabolite_smiles = Chem.MolToSmiles(metabolite, canonical=True)
        except Exception as exc:
            return MetabolismResult(False, smiles, None, atom_idx, chosen_type, 0, 0, error=str(exc))
        return MetabolismResult(
            success=True,
            parent_smiles=prep.canonical_smiles or smiles,
            metabolite_smiles=metabolite_smiles,
            site_atom_idx=int(atom_idx),
            metabolism_type=chosen_type,
            atoms_removed=max(0, mol.GetNumAtoms() - metabolite.GetNumAtoms()),
            atoms_added=max(0, metabolite.GetNumAtoms() - mol.GetNumAtoms()),
            error=None,
        )

    def detect_metabolism_type(self, mol, atom_idx: int) -> MetabolismType:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        symbol = atom.GetSymbol()
        if symbol == "N" and atom.GetTotalDegree() >= 3 and atom.GetTotalNumHs() == 0:
            return MetabolismType.N_OXIDATION
        if symbol == "S":
            return MetabolismType.SULFOXIDATION
        if symbol == "C":
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == "N":
                    return MetabolismType.N_DEALKYLATION
                if neighbor.GetSymbol() == "O":
                    return MetabolismType.O_DEALKYLATION
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetSymbol() == "C":
                    return MetabolismType.EPOXIDATION
            return MetabolismType.HYDROXYLATION
        return MetabolismType.HYDROXYLATION

    def _largest_fragment(self, rw_mol):
        try:
            mol = rw_mol.GetMol()
            Chem.SanitizeMol(mol)
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                return None
            return max(frags, key=lambda frag: frag.GetNumHeavyAtoms())
        except Exception:
            return None

    def _apply_reaction(self, mol, atom_idx: int, metabolism_type: MetabolismType):
        if metabolism_type == MetabolismType.N_DEALKYLATION:
            return self._n_dealkylate(mol, atom_idx)
        if metabolism_type == MetabolismType.O_DEALKYLATION:
            return self._o_dealkylate(mol, atom_idx)
        if metabolism_type == MetabolismType.N_OXIDATION:
            return self._n_oxidize(mol, atom_idx)
        if metabolism_type == MetabolismType.EPOXIDATION:
            return self._epoxidate(mol, atom_idx)
        if metabolism_type == MetabolismType.SULFOXIDATION:
            return self._sulfoxidate(mol, atom_idx)
        return self._hydroxylate(mol, atom_idx)

    def _hydroxylate(self, mol, atom_idx: int):
        rw = Chem.RWMol(Chem.Mol(mol))
        atom = rw.GetAtomWithIdx(int(atom_idx))
        if atom.GetSymbol() not in {"C", "N", "S"} or atom.GetTotalNumHs() <= 0:
            return None
        o_idx = rw.AddAtom(Chem.Atom("O"))
        rw.AddBond(int(atom_idx), int(o_idx), Chem.BondType.SINGLE)
        oxygen = rw.GetAtomWithIdx(int(o_idx))
        oxygen.SetNumExplicitHs(1)
        oxygen.SetNoImplicit(True)
        try:
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return None

    def _detach_alkyl_from_neighbor(self, mol, atom_idx: int, hetero_symbol: str):
        rw = Chem.RWMol(Chem.Mol(mol))
        atom = rw.GetAtomWithIdx(int(atom_idx))
        neighbor = None
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == hetero_symbol:
                neighbor = nbr
                break
        if neighbor is None:
            return None
        rw.RemoveBond(int(atom_idx), int(neighbor.GetIdx()))
        neighbor_atom = rw.GetAtomWithIdx(int(neighbor.GetIdx()))
        neighbor_atom.SetNumExplicitHs(neighbor_atom.GetTotalNumHs() + 1)
        neighbor_atom.SetNoImplicit(False)
        return self._largest_fragment(rw)

    def _n_dealkylate(self, mol, atom_idx: int):
        return self._detach_alkyl_from_neighbor(mol, atom_idx, "N")

    def _o_dealkylate(self, mol, atom_idx: int):
        return self._detach_alkyl_from_neighbor(mol, atom_idx, "O")

    def _n_oxidize(self, mol, atom_idx: int):
        rw = Chem.RWMol(Chem.Mol(mol))
        atom = rw.GetAtomWithIdx(int(atom_idx))
        if atom.GetSymbol() != "N":
            return None
        o_idx = rw.AddAtom(Chem.Atom("O"))
        rw.AddBond(int(atom_idx), int(o_idx), Chem.BondType.SINGLE)
        atom.SetFormalCharge(1)
        rw.GetAtomWithIdx(int(o_idx)).SetFormalCharge(-1)
        try:
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return None

    def _sulfoxidate(self, mol, atom_idx: int):
        rw = Chem.RWMol(Chem.Mol(mol))
        atom = rw.GetAtomWithIdx(int(atom_idx))
        if atom.GetSymbol() != "S":
            return None
        o_idx = rw.AddAtom(Chem.Atom("O"))
        rw.AddBond(int(atom_idx), int(o_idx), Chem.BondType.DOUBLE)
        try:
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return None

    def _epoxidate(self, mol, atom_idx: int):
        rw = Chem.RWMol(Chem.Mol(mol))
        atom = rw.GetAtomWithIdx(int(atom_idx))
        partner = None
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "C":
                    partner = other
                    bond.SetBondType(Chem.BondType.SINGLE)
                    break
        if partner is None:
            return None
        o_idx = rw.AddAtom(Chem.Atom("O"))
        rw.AddBond(int(atom_idx), int(o_idx), Chem.BondType.SINGLE)
        rw.AddBond(int(partner.GetIdx()), int(o_idx), Chem.BondType.SINGLE)
        try:
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return None

    def get_all_metabolism_sites(self, smiles: str) -> List[Dict[str, object]]:
        if Chem is None:
            return []
        prep = prepare_mol(smiles)
        if prep.mol is None:
            return []
        mol = prep.mol
        sites: list[dict[str, object]] = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0:
                priority = 1.0
                if self._is_benzylic(mol, idx):
                    priority = 0.25
                elif self._is_allylic(mol, idx):
                    priority = 0.35
                elif self._is_alpha_hetero(mol, idx):
                    priority = 0.45
                sites.append(
                    {
                        "atom_idx": int(idx),
                        "metabolism_type": self.detect_metabolism_type(mol, idx),
                        "priority": float(priority),
                    }
                )
            elif atom.GetSymbol() == "N" and atom.GetTotalDegree() >= 3 and atom.GetTotalNumHs() == 0:
                sites.append(
                    {
                        "atom_idx": int(idx),
                        "metabolism_type": MetabolismType.N_OXIDATION,
                        "priority": 0.55,
                    }
                )
            elif atom.GetSymbol() == "S":
                sites.append(
                    {
                        "atom_idx": int(idx),
                        "metabolism_type": MetabolismType.SULFOXIDATION,
                        "priority": 0.5,
                    }
                )
        sites.sort(key=lambda item: (float(item["priority"]), int(item["atom_idx"])))
        return sites

    def _is_benzylic(self, mol, atom_idx: int) -> bool:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        return any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())

    def _is_allylic(self, mol, atom_idx: int) -> bool:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        for neighbor in atom.GetNeighbors():
            for bond in neighbor.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(neighbor).GetSymbol() == "C":
                    return True
        return False

    def _is_alpha_hetero(self, mol, atom_idx: int) -> bool:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        return any(neighbor.GetSymbol() in {"N", "O", "S"} for neighbor in atom.GetNeighbors())
