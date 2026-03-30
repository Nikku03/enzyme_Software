from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


REACTION_TAXONOMY: Dict[str, int] = {
    "aliphatic_hydroxylation": 0,
    "aromatic_hydroxylation": 1,
    "dealkylation": 2,
    "epoxidation": 3,
    "oxidation_n_s": 4,
}
NUM_MORPHISM_CLASSES = len(REACTION_TAXONOMY)

_REACTION_PROP_KEYS = (
    "REACTIONS",
    "REACTION",
    "REACTION_TYPE",
    "REACTION_TYPES",
    "METABOLISM_TYPE",
    "METABOLISM_TYPES",
)

_LABEL_SOURCE_KEYS = (
    "LABEL_SOURCE",
    "SITE_SOURCE",
    "ANNOTATION_SOURCE",
    "SOURCE",
)


def _tokenize_prop(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    normalized = text.replace(";", ",").replace("|", ",").replace("/", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _normalize_reaction_name(name: str) -> str:
    value = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "hydroxylation": "hydroxylation",
        "aliphatic_hydroxylation": "aliphatic_hydroxylation",
        "aromatic_hydroxylation": "aromatic_hydroxylation",
        "n_dealkylation": "dealkylation",
        "o_dealkylation": "dealkylation",
        "dealkylation": "dealkylation",
        "n_oxidation": "oxidation_n_s",
        "s_oxidation": "oxidation_n_s",
        "oxidation_n_s": "oxidation_n_s",
        "epoxidation": "epoxidation",
    }
    return aliases.get(value, value)


def _label_confidence_from_source(source: str | None) -> float:
    value = str(source or "").strip().lower()
    if not value:
        return 0.0
    if any(token in value for token in ("assay", "clinical", "validated", "manual", "curated")):
        return 1.0
    if any(token in value for token in ("drugbank", "metxbio", "literature", "reported")):
        return 0.75
    if any(token in value for token in ("heuristic", "smarts", "inferred", "predicted", "rule")):
        return 0.5
    return 0.5


def _infer_morphism_classes_for_atom(atom) -> List[int]:
    classes: List[int] = []
    seen = set()

    def add(name: str) -> None:
        idx = REACTION_TAXONOMY.get(name)
        if idx is not None and idx not in seen:
            classes.append(idx)
            seen.add(idx)

    symbol = atom.GetSymbol()
    is_aromatic = bool(atom.GetIsAromatic())
    hybrid = str(atom.GetHybridization())
    neighbors = list(atom.GetNeighbors())
    neighbor_symbols = {nbr.GetSymbol() for nbr in neighbors}

    if symbol in {"N", "S"}:
        add("oxidation_n_s")

    if symbol in {"N", "O"} and len(neighbors) >= 2 and any(nbr.GetSymbol() == "C" for nbr in neighbors):
        add("dealkylation")

    has_c_c_double = any(
        bond.GetBondTypeAsDouble() >= 1.9
        and bond.GetBeginAtom().GetSymbol() == "C"
        and bond.GetEndAtom().GetSymbol() == "C"
        and not bond.GetIsAromatic()
        for bond in atom.GetBonds()
    )
    in_unsaturated_ring = bool(atom.IsInRing()) and hybrid == "SP2" and any(
        nbr.GetSymbol() == "C" and not bond.GetIsAromatic()
        for nbr, bond in ((bond.GetOtherAtom(atom), bond) for bond in atom.GetBonds())
    )
    if symbol == "C" and has_c_c_double:
        add("epoxidation")
    elif symbol == "C" and in_unsaturated_ring:
        add("epoxidation")

    if symbol == "C":
        if is_aromatic:
            add("aromatic_hydroxylation")
        else:
            if hybrid == "SP3":
                add("aliphatic_hydroxylation")
            if neighbor_symbols & {"N", "O", "S"}:
                add("dealkylation")

    return classes


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

    def _extract_reaction_names(self, mol) -> List[str]:
        reactions: List[str] = []
        for key in _REACTION_PROP_KEYS:
            if not mol.HasProp(key):
                continue
            reactions.extend(_tokenize_prop(mol.GetProp(key)))
        deduped = []
        seen = set()
        for reaction in reactions:
            normalized = _normalize_reaction_name(reaction)
            if normalized and normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    def _extract_label_source(self, mol) -> Optional[str]:
        for key in _LABEL_SOURCE_KEYS:
            if mol.HasProp(key):
                value = str(mol.GetProp(key)).strip()
                if value:
                    return value
        return None

    @staticmethod
    def _morphism_classes_for_atom(atom, reactions: Iterable[str]) -> List[int]:
        classes: List[int] = []
        seen = set()
        for reaction in reactions:
            canonical = _normalize_reaction_name(reaction)
            if canonical == "hydroxylation":
                if atom.GetIsAromatic():
                    canonical = "aromatic_hydroxylation"
                else:
                    canonical = "aliphatic_hydroxylation"
            elif canonical == "oxidation_n_s":
                if atom.GetSymbol() not in {"N", "S"}:
                    continue
            elif canonical == "epoxidation":
                if atom.GetSymbol() != "C":
                    continue
            elif canonical == "dealkylation":
                if atom.GetSymbol() not in {"C", "N", "O"}:
                    continue

            class_idx = REACTION_TAXONOMY.get(canonical)
            if class_idx is None or class_idx in seen:
                continue
            classes.append(class_idx)
            seen.add(class_idx)
        return classes

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
        som_target = torch.zeros((num_atoms,), dtype=torch.float32)
        som_target[som_idx] = 1.0

        morphism_target = torch.zeros((num_atoms, NUM_MORPHISM_CLASSES), dtype=torch.float32)
        morphism_loss_mask = torch.zeros((num_atoms, NUM_MORPHISM_CLASSES), dtype=torch.float32)
        reaction_names = self._extract_reaction_names(mol)
        label_source = self._extract_label_source(mol)
        label_confidence = _label_confidence_from_source(label_source)
        morphism_class_indices = self._morphism_classes_for_atom(
            mol.GetAtomWithIdx(int(som_idx)),
            reaction_names,
        )
        has_morphism_label = len(morphism_class_indices) > 0
        if not has_morphism_label:
            morphism_class_indices = _infer_morphism_classes_for_atom(mol.GetAtomWithIdx(int(som_idx)))
            has_morphism_label = len(morphism_class_indices) > 0
            if has_morphism_label:
                if not reaction_names:
                    inverse_taxonomy = {value: key for key, value in REACTION_TAXONOMY.items()}
                    reaction_names = [inverse_taxonomy[idx] for idx in morphism_class_indices if idx in inverse_taxonomy]
                label_source = label_source or "heuristic_atom_type"
                label_confidence = max(label_confidence, 0.25)
        if has_morphism_label:
            if label_confidence <= 0.0:
                label_confidence = 0.5
            morphism_target[som_idx, morphism_class_indices] = 1.0
            # Only supervise the labeled SoM row to avoid turning missing
            # site-level mechanism labels into false negatives elsewhere.
            morphism_loss_mask[som_idx, :] = 1.0

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
            "som_target": som_target,
            "morphism_target": morphism_target,
            "morphism_loss_mask": morphism_loss_mask,
            "has_morphism_label": torch.tensor(has_morphism_label, dtype=torch.bool),
            "label_confidence": torch.tensor(label_confidence, dtype=torch.float32),
            "reaction_names": reaction_names,
            "label_source": label_source or "",
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
            "som_target": torch.empty(0, 0),
            "morphism_target": torch.empty(0, 0, NUM_MORPHISM_CLASSES),
            "morphism_loss_mask": torch.empty(0, 0, NUM_MORPHISM_CLASSES),
            "has_morphism_label": torch.empty(0, dtype=torch.bool),
            "label_confidence": torch.empty(0),
            "reaction_names": [],
            "label_source": [],
            "attention_mask": torch.empty(0, 0, dtype=torch.bool),
            "num_atoms": torch.empty(0, dtype=torch.long),
        }

    coords = pad_sequence([item["coords"] for item in batch], batch_first=True, padding_value=0.0)
    masses = pad_sequence([item["masses"] for item in batch], batch_first=True, padding_value=0.0)
    charges = pad_sequence([item["charges"] for item in batch], batch_first=True, padding_value=0.0)
    atomic_numbers = pad_sequence([item["atomic_numbers"] for item in batch], batch_first=True, padding_value=0)
    som_target = pad_sequence([item["som_target"] for item in batch], batch_first=True, padding_value=0.0)
    morphism_target = pad_sequence([item["morphism_target"] for item in batch], batch_first=True, padding_value=0.0)
    morphism_loss_mask = pad_sequence([item["morphism_loss_mask"] for item in batch], batch_first=True, padding_value=0.0)

    max_atoms = coords.size(1)
    batch_size = len(batch)
    padded_dags = torch.zeros((batch_size, max_atoms, max_atoms), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    smiles = []
    true_som_idx = []
    reaction_names = []
    label_source = []

    for i, item in enumerate(batch):
        n = int(item["num_atoms"])
        padded_dags[i, :n, :n] = item["target_dag"]
        attention_mask[i, :n] = True
        smiles.append(item["smiles"])
        true_som_idx.append(item.get("true_som_idx"))
        reaction_names.append(list(item.get("reaction_names", [])))
        label_source.append(str(item.get("label_source", "")))

    out: Dict[str, Any] = {
        "smiles": smiles,
        "coords": coords,
        "masses": masses,
        "charges": charges,
        "atomic_numbers": atomic_numbers,
        "target_dag": padded_dags,
        "som_target": som_target,
        "morphism_target": morphism_target,
        "morphism_loss_mask": morphism_loss_mask,
        "has_morphism_label": torch.stack([item["has_morphism_label"].view(()) for item in batch], dim=0),
        "label_confidence": torch.stack([item["label_confidence"].view(()) for item in batch], dim=0),
        "reaction_names": reaction_names,
        "label_source": label_source,
        "attention_mask": attention_mask,
        "num_atoms": torch.tensor([item["num_atoms"] for item in batch], dtype=torch.long),
    }
    if all(idx is not None for idx in true_som_idx):
        out["true_som_idx"] = torch.stack([idx.view(()) for idx in true_som_idx], dim=0)
    return out


__all__ = [
    "NUM_MORPHISM_CLASSES",
    "REACTION_TAXONOMY",
    "ZaretzkiMetabolicDataset",
    "geometric_collate_fn",
]
