from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _RDKIT_OK = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None
    _RDKIT_OK = False


def _require_rdkit() -> None:
    if not _RDKIT_OK:
        raise ImportError("RDKit is required for ZaretzkiMetabolicDataset")


class ZaretzkiMetabolicDataset(Dataset):
    """
    SDF-backed metabolic dataset with ETKDGv3 conformer generation and basic
    per-atom physical tensors for downstream geometric models.

    The current NEXUS trainer is still fundamentally SMILES-driven, so this
    dataset also emits canonical SMILES and a scalar `true_som_idx` whenever
    it can recover one from the SDF metadata.
    """

    _SOM_KEYS = (
        "SOM_IDX",
        "SoM_IDX",
        "SOM",
        "SOM_INDEX",
        "PRIMARY_SOM",
        "SECONDARY_SOM",
        "SITE_OF_METABOLISM",
    )

    def __init__(self, sdf_file_path: str | Path) -> None:
        _require_rdkit()
        self.sdf_file_path = str(sdf_file_path)
        suppl = Chem.SDMolSupplier(self.sdf_file_path, removeHs=False)
        self.mols = [mol for mol in suppl if mol is not None]
        print(f"Loaded {len(self.mols)} valid molecules from {self.sdf_file_path}")

    def __len__(self) -> int:
        return len(self.mols)

    def _embed_if_needed(self, mol):
        if mol.GetNumConformers() > 0:
            return mol
        work = Chem.Mol(mol)
        if not any(atom.GetAtomicNum() == 1 for atom in work.GetAtoms()):
            work = Chem.AddHs(work)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = False
        status = AllChem.EmbedMolecule(work, params)
        if int(status) != 0:
            AllChem.EmbedMolecule(work, randomSeed=42)
        return work

    def _maybe_optimize(self, mol) -> None:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass

    def _compute_charges(self, mol) -> None:
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass

    def _extract_som_idx(self, mol) -> Optional[int]:
        for key in self._SOM_KEYS:
            if not mol.HasProp(key):
                continue
            raw = str(mol.GetProp(key)).strip()
            if not raw:
                continue
            token = raw.split(",")[0].split(";")[0].strip()
            try:
                idx = int(float(token)) - 1
            except Exception:
                continue
            if 0 <= idx < mol.GetNumAtoms():
                return idx
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mol = Chem.Mol(self.mols[idx])
        mol = self._embed_if_needed(mol)
        self._maybe_optimize(mol)
        self._compute_charges(mol)

        num_atoms = mol.GetNumAtoms()
        coords = np.zeros((num_atoms, 3), dtype=np.float32)
        masses = np.zeros((num_atoms,), dtype=np.float32)
        charges = np.zeros((num_atoms,), dtype=np.float32)
        atomic_numbers = np.zeros((num_atoms,), dtype=np.int64)

        conf = mol.GetConformer()
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]

            atom = mol.GetAtomWithIdx(i)
            masses[i] = atom.GetMass()
            atomic_numbers[i] = atom.GetAtomicNum()

            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                charges[i] = charge if not np.isnan(charge) else 0.0
            except Exception:
                charges[i] = 0.0

        target_dag = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
        som_idx = self._extract_som_idx(mol)
        if som_idx is not None:
            target_dag[som_idx, som_idx] = 1.0

        smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
        item: Dict[str, Any] = {
            "smiles": smiles,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "masses": torch.tensor(masses, dtype=torch.float32),
            "charges": torch.tensor(charges, dtype=torch.float32),
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
            "target_dag": target_dag,
            "num_atoms": int(num_atoms),
        }
        if som_idx is not None:
            item["true_som_idx"] = torch.tensor(som_idx, dtype=torch.long)
        return item


def geometric_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-sized molecular tensors to the largest molecule in the batch.

    This collator is future-facing for geometry-first training.  The current
    causal trainer still expects effectively single-sample batches for the live
    SMILES-driven dynamics path.
    """
    coords = pad_sequence([item["coords"] for item in batch], batch_first=True, padding_value=0.0)
    masses = pad_sequence([item["masses"] for item in batch], batch_first=True, padding_value=0.0)
    charges = pad_sequence([item["charges"] for item in batch], batch_first=True, padding_value=0.0)
    atomic_numbers = pad_sequence([item["atomic_numbers"] for item in batch], batch_first=True, padding_value=0)

    max_atoms = coords.size(1)
    batch_size = len(batch)
    padded_dags = torch.zeros((batch_size, max_atoms, max_atoms), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    smiles = []
    true_som_idx = []

    for i, item in enumerate(batch):
        n = int(item["num_atoms"])
        padded_dags[i, :n, :n] = item["target_dag"]
        attention_mask[i, :n] = True
        smiles.append(item["smiles"])
        true_som_idx.append(item.get("true_som_idx"))

    out: Dict[str, Any] = {
        "smiles": smiles,
        "coords": coords,
        "masses": masses,
        "charges": charges,
        "atomic_numbers": atomic_numbers,
        "target_dag": padded_dags,
        "attention_mask": attention_mask,
        "num_atoms": torch.tensor([item["num_atoms"] for item in batch], dtype=torch.long),
    }
    if all(idx is not None for idx in true_som_idx):
        out["true_som_idx"] = torch.stack([idx.view(()) for idx in true_som_idx], dim=0)
    return out


__all__ = ["ZaretzkiMetabolicDataset", "geometric_collate_fn"]
