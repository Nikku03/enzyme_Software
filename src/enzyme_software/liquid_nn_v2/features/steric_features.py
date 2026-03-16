from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem

    _RDKIT = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    RDLogger = None
    AllChem = None
    _RDKIT = False

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


def canonicalize_smiles(smiles: str) -> str:
    if not _RDKIT:
        return str(smiles or "").strip()
    with mol_provenance_context(module_triggered="steric features", source_category="steric features", parsed_smiles=str(smiles or "").strip()):
        prep = prepare_mol(str(smiles or "").strip())
    if prep.mol is None or prep.canonical_smiles is None:
        return str(smiles or "").strip()
    return prep.canonical_smiles


def resolve_default_structure_sdf(root: Optional[str | Path] = None) -> Optional[Path]:
    base = Path(root) if root is not None else Path.cwd()
    candidates = [
        base / "3D structures.sdf",
        base / "structures.sdf",
        base / "data" / "drugbank_all_3d_structures.sdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _embed_fallback_conformer(mol):
    if not _RDKIT or mol is None:
        return None
    mol = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    if RDLogger is not None:
        RDLogger.DisableLog("rdApp.*")
    try:
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        return None
    finally:
        if RDLogger is not None:
            RDLogger.EnableLog("rdApp.*")
    return Chem.RemoveHs(mol)


def _has_nontrivial_3d(structure_mol) -> bool:
    if not _RDKIT or structure_mol is None or structure_mol.GetNumConformers() == 0:
        return False
    conf = structure_mol.GetConformer()
    z_values = [float(conf.GetAtomPosition(i).z) for i in range(structure_mol.GetNumAtoms())]
    return any(abs(z) > 1.0e-4 for z in z_values)


def _heavy_atom_match(mol_2d, structure_mol) -> Optional[tuple[int, ...]]:
    if not _RDKIT or mol_2d is None or structure_mol is None:
        return None
    try:
        target = Chem.RemoveHs(Chem.Mol(structure_mol))
    except Exception:
        return None
    if target is None or target.GetNumConformers() == 0:
        return None
    match = tuple(target.GetSubstructMatch(mol_2d))
    if len(match) == mol_2d.GetNumAtoms():
        return match
    reverse = tuple(mol_2d.GetSubstructMatch(target))
    if len(reverse) == target.GetNumAtoms() == mol_2d.GetNumAtoms():
        inverse = [0] * len(reverse)
        for src_idx, dst_idx in enumerate(reverse):
            inverse[int(dst_idx)] = int(src_idx)
        return tuple(inverse)
    if target.GetNumAtoms() == mol_2d.GetNumAtoms():
        return tuple(range(target.GetNumAtoms()))
    return None


def compute_atom_3d_features(mol_2d, structure_mol) -> Optional[np.ndarray]:
    """Compute lightweight per-atom steric descriptors aligned to the 2D atom order."""
    if not _RDKIT or mol_2d is None or structure_mol is None:
        return None
    match = _heavy_atom_match(mol_2d, structure_mol)
    if match is None:
        return None
    target = Chem.RemoveHs(Chem.Mol(structure_mol))
    if target.GetNumConformers() == 0:
        return None
    conf = target.GetConformer()
    coords = np.asarray(
        [
            [
                float(conf.GetAtomPosition(struct_idx).x),
                float(conf.GetAtomPosition(struct_idx).y),
                float(conf.GetAtomPosition(struct_idx).z),
            ]
            for struct_idx in match
        ],
        dtype=np.float32,
    )
    if coords.shape[0] != mol_2d.GetNumAtoms():
        return None

    center = coords.mean(axis=0, keepdims=True)
    centered = coords - center
    radial = np.linalg.norm(centered, axis=1)
    max_radial = float(max(np.max(radial), 1.0))

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    finite = np.where(np.isfinite(distances), distances, 0.0)
    denom = max(coords.shape[0] - 1, 1)

    near_25 = (distances < 2.5).sum(axis=1).astype(np.float32) / float(denom)
    near_40 = (distances < 4.0).sum(axis=1).astype(np.float32) / float(denom)
    inv_sum = (1.0 / np.clip(distances, 1.0e-3, None))
    inv_sum[~np.isfinite(inv_sum)] = 0.0
    crowding = inv_sum.sum(axis=1).astype(np.float32) / float(denom)
    nearest = np.where(np.isfinite(distances), distances, 99.0).min(axis=1).astype(np.float32)
    mean_dist = finite.sum(axis=1).astype(np.float32) / float(denom)
    std_dist = finite.std(axis=1).astype(np.float32)
    buried = crowding * (1.0 - (radial / max_radial))

    features = np.stack(
        [
            (radial / max_radial).astype(np.float32),
            near_25,
            near_40,
            np.tanh(crowding).astype(np.float32),
            np.clip(nearest / 5.0, 0.0, 1.0).astype(np.float32),
            np.clip(mean_dist / 8.0, 0.0, 1.0).astype(np.float32),
            np.clip(std_dist / 4.0, 0.0, 1.0).astype(np.float32),
            np.tanh(buried).astype(np.float32),
        ],
        axis=1,
    )
    return features.astype(np.float32)


@dataclass
class StructureLibrary:
    path: Path
    by_smiles: Dict[str, object]

    @classmethod
    def from_sdf(cls, sdf_path: str | Path) -> Optional["StructureLibrary"]:
        if not _RDKIT:
            return None
        path = Path(sdf_path)
        if not path.exists():
            return None
        if RDLogger is not None:
            RDLogger.DisableLog("rdApp.*")
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        by_smiles: Dict[str, object] = {}
        for mol in supplier:
            if mol is None:
                continue
            try:
                key = canonicalize_smiles(Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True))
            except Exception:
                continue
            if key and key not in by_smiles and mol.GetNumConformers() > 0 and _has_nontrivial_3d(mol):
                by_smiles[key] = mol
        if RDLogger is not None:
            RDLogger.EnableLog("rdApp.*")
        return cls(path=path, by_smiles=by_smiles)

    def get(self, smiles: str):
        return self.by_smiles.get(canonicalize_smiles(smiles))


def compute_atom_3d_features_for_smiles(smiles: str, structure_library: Optional[StructureLibrary] = None) -> Optional[np.ndarray]:
    if not _RDKIT:
        return None
    with mol_provenance_context(module_triggered="steric features", source_category="steric features", parsed_smiles=str(smiles or "")):
        prep = prepare_mol(str(smiles or ""))
    mol_2d = prep.mol
    if mol_2d is None:
        return None
    structure_mol = structure_library.get(prep.canonical_smiles or smiles) if structure_library is not None else None
    if structure_mol is None:
        structure_mol = _embed_fallback_conformer(mol_2d)
    if structure_mol is None:
        return None
    return compute_atom_3d_features(mol_2d, structure_mol)
