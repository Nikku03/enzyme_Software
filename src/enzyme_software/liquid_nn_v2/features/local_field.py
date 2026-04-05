from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _RDKIT = True
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    _RDKIT = False


_VDW_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}


def _rational_kernel(
    r: np.ndarray,
    *,
    a0: float = 1.0,
    a1: float = 0.15,
    a2: float = 0.02,
    b1: float = 0.55,
    b2: float = 0.18,
    b3: float = 0.025,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32)
    numer = a0 + (a1 * r) + (a2 * np.square(r))
    denom = 1.0 + (b1 * r) + (b2 * np.square(r)) + (b3 * np.power(r, 3))
    return numer / np.clip(denom, 1.0e-6, None)


def _embed_fallback_conformer(mol):
    if not _RDKIT or mol is None:
        return None
    target = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    try:
        if AllChem.EmbedMolecule(target, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(target, maxIters=200)
    except Exception:
        return None
    return Chem.RemoveHs(target)


def resolve_atom_coordinates(mol_2d, structure_mol=None) -> Optional[np.ndarray]:
    if not _RDKIT or mol_2d is None:
        return None
    target = structure_mol
    if target is None or target.GetNumConformers() == 0:
        target = _embed_fallback_conformer(mol_2d)
    if target is None or target.GetNumConformers() == 0:
        return None
    if target.GetNumAtoms() != mol_2d.GetNumAtoms():
        try:
            match = tuple(Chem.RemoveHs(Chem.Mol(target)).GetSubstructMatch(mol_2d))
        except Exception:
            return None
        if len(match) != mol_2d.GetNumAtoms():
            return None
    else:
        match = tuple(range(mol_2d.GetNumAtoms()))
    conf = target.GetConformer()
    coords = np.asarray(
        [
            [
                float(conf.GetAtomPosition(int(idx)).x),
                float(conf.GetAtomPosition(int(idx)).y),
                float(conf.GetAtomPosition(int(idx)).z),
            ]
            for idx in match
        ],
        dtype=np.float32,
    )
    return coords if coords.shape[0] == mol_2d.GetNumAtoms() else None


def compute_local_field_features(
    atom_coords: Optional[np.ndarray],
    atomic_numbers: np.ndarray,
    initial_charges: np.ndarray,
    *,
    neighbor_radius: float = 4.5,
    damping_delta: float = 0.35,
) -> dict[str, np.ndarray]:
    num_atoms = int(len(atomic_numbers))
    zeros = np.zeros((num_atoms, 1), dtype=np.float32)
    if atom_coords is None or num_atoms <= 0:
        return {
            "steric_score": zeros,
            "electro_score": zeros,
            "field_score": zeros,
            "access_proxy": zeros,
            "crowding": zeros,
            "neighbor_count": zeros,
            "valid_mask": zeros,
        }

    coords = np.asarray(atom_coords, dtype=np.float32).reshape(num_atoms, 3)
    charges = np.asarray(initial_charges, dtype=np.float32).reshape(num_atoms)
    atom_z = np.asarray(atomic_numbers, dtype=np.int64).reshape(num_atoms)
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1).astype(np.float32)
    np.fill_diagonal(distances, np.inf)
    local_mask = distances < float(neighbor_radius)

    sigma = np.asarray([_VDW_RADII.get(int(z), 1.75) for z in atom_z], dtype=np.float32)
    sigma_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    inv_term = np.square(sigma_ij) / (np.square(distances) + float(damping_delta) ** 2)
    lj = 4.0 * 0.12 * (np.power(inv_term, 6) - np.power(inv_term, 3))
    lj = np.maximum(lj, 0.0) * local_mask
    steric_score = np.tanh(lj.sum(axis=1, keepdims=True)).astype(np.float32)

    screened = _rational_kernel(np.where(np.isfinite(distances), distances, 0.0))
    electro = (screened * charges[None, :] * local_mask).sum(axis=1, keepdims=True)
    electro_score = np.tanh(electro).astype(np.float32)

    access_decay = _rational_kernel(
        np.where(np.isfinite(distances), distances, 0.0),
        a0=0.4,
        a1=0.10,
        a2=0.01,
        b1=0.80,
        b2=0.30,
        b3=0.04,
    )
    crowding = (access_decay * local_mask).sum(axis=1, keepdims=True)
    crowding = np.tanh(crowding / max(float(num_atoms), 1.0)).astype(np.float32)
    access_proxy = (1.0 / (1.0 + np.maximum(crowding, 0.0))).astype(np.float32)

    field_score = np.tanh((0.70 * electro_score) - (0.35 * steric_score) + (0.20 * access_proxy)).astype(np.float32)
    neighbor_count = local_mask.sum(axis=1, keepdims=True).astype(np.float32)
    neighbor_count = neighbor_count / max(float(max(num_atoms - 1, 1)), 1.0)
    valid_mask = np.ones((num_atoms, 1), dtype=np.float32)
    return {
        "steric_score": steric_score,
        "electro_score": electro_score,
        "field_score": field_score,
        "access_proxy": access_proxy,
        "crowding": crowding,
        "neighbor_count": neighbor_count.astype(np.float32),
        "valid_mask": valid_mask,
    }
