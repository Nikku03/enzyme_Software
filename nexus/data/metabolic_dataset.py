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
    from rdkit.Chem.rdmolops import GetMolFrags

    _RDKIT_OK = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None
    GetMolFrags = None
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
        "PRIMARY_SOM",
        "SOM_IDX",
        "SoM_IDX",
        "SOM",
        "SOM_INDEX",
        "SECONDARY_SOM",
        "SITE_OF_METABOLISM",
    )

    def __init__(self, sdf_file_path: str | Path, max_molecules: int = 0) -> None:
        _require_rdkit()
        self.sdf_file_path = str(sdf_file_path)
        
        suppl = Chem.SDMolSupplier(self.sdf_file_path, removeHs=False)
        
        print(f"Screening molecules from {self.sdf_file_path}...")
        self.mols = []
        
        raw_mols = []
        for mol in suppl:
            if mol is not None:
                raw_mols.append(mol)

        for mol in raw_mols:
            # Trap 4: Keep only the largest fragment
            frags = GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                mol = max(frags, default=mol, key=lambda m: m.GetNumAtoms())

            # Pre-screen for SoM index
            som_idx = self._extract_som_idx(mol)
            if som_idx is None:
                continue

            # Pre-screen for 3D embeddability
            mol_3d = Chem.AddHs(mol)
            mol_3d.RemoveAllConformers()
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            embed_status = AllChem.EmbedMolecule(mol_3d, params)
            if embed_status != 0:
                # Try fallback embedding
                if AllChem.EmbedMolecule(mol_3d, randomSeed=42) != 0:
                    continue # Discard if embedding fails

            self.mols.append(mol)
            if max_molecules > 0 and len(self.mols) >= max_molecules:
                break
        
        print(f"Loaded {len(self.mols)} valid, filtered molecules from {self.sdf_file_path}")

    def __len__(self) -> int:
        return len(self.mols)

    def _embed_3d(self, mol):
        """Generate fresh ETKDGv3 3D coordinates.

        Always clears existing conformers and re-embeds so that:
        - Explicit Hs added by the caller are included in the geometry.
        - 2D-only conformers from the SDF file are replaced by proper 3D ones.
        Caller must have already called Chem.AddHs() to avoid collinear/degenerate
        heavy-atom-only geometries that cause 1/r Coulomb singularities (NaN).
        """
        work = Chem.Mol(mol)
        work.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = False
        if int(AllChem.EmbedMolecule(work, params)) != 0:
            if int(AllChem.EmbedMolecule(work, randomSeed=42)) != 0:
                # Both deterministic passes failed — use random coordinates as last resort
                params2 = AllChem.ETKDGv3()
                params2.randomSeed = 0
                params2.useRandomCoords = True
                AllChem.EmbedMolecule(work, params2)
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
            # Handle space-, comma-, and semicolon-separated multi-SoM values (e.g. "8 4", "3,7").
            # Take the first listed site as the primary label.
            token = raw.split()[0].split(",")[0].split(";")[0].strip()
            try:
                idx = int(float(token)) - 1  # SDF uses 1-based atom indices
            except Exception:
                continue
            if 0 <= idx < mol.GetNumAtoms():
                return idx
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mol = Chem.Mol(self.mols[idx])

        # Largest fragment is already selected in __init__, but we do it again
        # to ensure consistency if the object is modified after init.
        frags = GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, default=mol, key=lambda m: m.GetNumAtoms())

        som_idx = self._extract_som_idx(mol)
        assert som_idx is not None, f"Molecule at index {idx} should have a SoM after pre-screening."

        mol = Chem.AddHs(mol)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = False
        if int(AllChem.EmbedMolecule(mol, params)) != 0:
            if int(AllChem.EmbedMolecule(mol, randomSeed=42)) != 0:
                params2 = AllChem.ETKDGv3()
                params2.randomSeed = 0
                params2.useRandomCoords = True
                AllChem.EmbedMolecule(mol, params2)
        
        assert mol.GetNumConformers() > 0, f"Molecule at index {idx} should have a 3D conformer after pre-screening."

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
        target_dag[som_idx, som_idx] = 1.0

        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
        except Exception:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
        item: Dict[str, Any] = {
            "smiles": smiles,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "masses": torch.tensor(masses, dtype=torch.float32),
            "charges": torch.tensor(charges, dtype=torch.float32),
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
            "target_dag": target_dag,
            "num_atoms": int(num_atoms),
            "true_som_idx": torch.tensor(som_idx, dtype=torch.long),
        }
        return item


def geometric_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-sized molecular tensors to the largest molecule in the batch.

    This collator is future-facing for geometry-first training.  The current
    causal trainer still expects effectively single-sample batches for the live
    SMILES-driven dynamics path.
    (No longer needs to filter Nones as the dataset is pre-cleaned in __init__).
    """
    if not batch:
        return {
            "smiles": [],
            "coords": torch.empty(0, 0, 3),
            "masses": torch.empty(0, 0),
            "charges": torch.empty(0, 0),
            "atomic_numbers": torch.empty(0, 0),
            "target_dag": torch.empty(0, 0, 0),
            "attention_mask": torch.empty(0, 0, dtype=torch.bool),
            "num_atoms": torch.empty(0, dtype=torch.long),
        }

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
