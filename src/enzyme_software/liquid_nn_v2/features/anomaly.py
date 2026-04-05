from __future__ import annotations

import numpy as np

try:
    from rdkit import Chem

    _RDKIT = True
except Exception:  # pragma: no cover
    Chem = None
    _RDKIT = False


def compute_anomaly_features(mol, *, num_atoms: int) -> dict[str, np.ndarray]:
    feature_dim = 7
    zeros = np.zeros((1, feature_dim), dtype=np.float32)
    if not _RDKIT or mol is None:
        return {
            "features": zeros,
            "score": np.zeros((1, 1), dtype=np.float32),
            "flag": np.zeros((1, 1), dtype=np.float32),
        }

    ring_count = float(mol.GetRingInfo().NumRings())
    aromatic_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))
    heavy_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
    halogens = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53}))
    formal_charged = float(sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0))
    fused_indicator = 1.0 if ring_count >= 3.0 and aromatic_atoms >= 6.0 else 0.0
    bridgehead = float(sum(1 for atom in mol.GetAtoms() if atom.IsInRing() and atom.GetDegree() >= 3))
    aromatic_ratio = aromatic_atoms / max(heavy_atoms, 1.0)
    features = np.asarray(
        [[ring_count, halogens, formal_charged, fused_indicator, aromatic_ratio, heavy_atoms, bridgehead]],
        dtype=np.float32,
    )
    mu = np.asarray([[1.5, 0.3, 0.2, 0.1, 0.35, max(18.0, float(num_atoms)), 0.5]], dtype=np.float32)
    sigma = np.asarray([[1.5, 0.8, 0.8, 0.3, 0.20, 10.0, 1.5]], dtype=np.float32)
    z = (features - mu) / np.clip(sigma, 1.0e-3, None)
    score = np.sum(np.square(z), axis=1, keepdims=True).astype(np.float32)
    flag = (score > 12.0).astype(np.float32)
    return {"features": features, "score": score, "flag": flag}
