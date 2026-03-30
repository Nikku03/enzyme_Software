from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import torch

try:
    from rdkit import Chem

    _RDKIT_OK = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    _RDKIT_OK = False


@dataclass
class ParityFeatureBundle:
    quadruplets: List[Tuple[int, int, int, int]]
    pseudo_scalar_per_atom: torch.Tensor
    pseudo_vector_per_atom: torch.Tensor
    center_mask: torch.Tensor
    center_indices: List[int]


def triple_product(pos: torch.Tensor, center: int, idx1: int, idx2: int, idx3: int) -> torch.Tensor:
    r1 = pos[idx1] - pos[center]
    r2 = pos[idx2] - pos[center]
    r3 = pos[idx3] - pos[center]
    return torch.dot(r1, torch.cross(r2, r3, dim=0))


def _nearest_neighbors(pos: torch.Tensor, center_idx: int, max_neighbors: int = 4) -> List[int]:
    center = pos[center_idx]
    delta = pos - center
    distance = delta.square().sum(dim=-1)
    order = torch.argsort(distance)
    neighbors = [int(idx.item()) for idx in order if int(idx.item()) != int(center_idx)]
    return neighbors[:max_neighbors]


def _rdkit_chiral_centers(smiles: str) -> Dict[int, List[int]]:
    if not _RDKIT_OK:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    centers: Dict[int, List[int]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        tag = str(atom.GetChiralTag())
        if tag == "CHI_UNSPECIFIED" and atom.GetDegree() < 4:
            continue
        neighbors = [int(neighbor.GetIdx()) for neighbor in atom.GetNeighbors()]
        if len(neighbors) >= 3:
            centers[int(atom.GetIdx())] = neighbors
    return centers


def extract_parity_bundle(manifold) -> ParityFeatureBundle:
    pos = torch.nan_to_num(manifold.pos, nan=0.0, posinf=0.0, neginf=0.0)
    n_atoms = int(pos.shape[0])
    pseudo_scalar = torch.zeros(n_atoms, dtype=pos.dtype, device=pos.device)
    pseudo_vector = torch.zeros_like(pos)
    center_mask = torch.zeros(n_atoms, dtype=torch.bool, device=pos.device)
    quadruplets: List[Tuple[int, int, int, int]] = []
    center_indices: List[int] = []
    chirality_codes = getattr(getattr(manifold, "seed", None), "chirality_codes", None)

    centers = _rdkit_chiral_centers(getattr(manifold.seed, "smiles", ""))
    if not centers:
        for atom_idx, atomic_num in enumerate(manifold.species.tolist()):
            if int(atomic_num) == 6 and n_atoms >= 4:
                centers[int(atom_idx)] = _nearest_neighbors(pos, atom_idx, max_neighbors=4)

    for center_idx, neighbors in centers.items():
        if len(neighbors) < 3:
            continue
        center_mask[center_idx] = True
        center_indices.append(int(center_idx))
        local_chiral_code = 0
        if chirality_codes is not None and int(center_idx) < int(chirality_codes.numel()):
            local_chiral_code = int(chirality_codes[int(center_idx)].item())
        if local_chiral_code != 0:
            ordered = sorted(neighbors[:4])
            combos = [tuple(ordered[:3])]
        else:
            combos = list(combinations(neighbors[:4], 3))
        if not combos:
            continue
        for idx1, idx2, idx3 in combos:
            quadruplets.append((center_idx, idx1, idx2, idx3))
            v1 = pos[idx1] - pos[center_idx]
            v2 = pos[idx2] - pos[center_idx]
            v3 = pos[idx3] - pos[center_idx]
            local_scale = (
                (v1.norm() + v2.norm() + v3.norm()) / 3.0
            ).clamp_min(1.0e-6)
            scalar = torch.nan_to_num(
                torch.dot(v1, torch.cross(v2, v3, dim=0)) / (local_scale ** 3),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            scalar = torch.tanh(scalar)
            if local_chiral_code != 0:
                scalar = torch.sign(torch.tensor(float(local_chiral_code), dtype=pos.dtype, device=pos.device)) * scalar.abs()
            pseudo_scalar[center_idx] = pseudo_scalar[center_idx] + scalar
            cross = torch.cross(v1, v2, dim=0) / (local_scale ** 2)
            cross = torch.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)
            cross = torch.tanh(cross)
            if local_chiral_code != 0:
                cross = torch.sign(torch.tensor(float(local_chiral_code), dtype=pos.dtype, device=pos.device)) * cross
            pseudo_vector[center_idx] = pseudo_vector[center_idx] + cross

    return ParityFeatureBundle(
        quadruplets=quadruplets,
        pseudo_scalar_per_atom=pseudo_scalar,
        pseudo_vector_per_atom=pseudo_vector,
        center_mask=center_mask,
        center_indices=center_indices,
    )
