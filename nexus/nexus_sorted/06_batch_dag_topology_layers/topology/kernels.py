from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from rdkit import Chem

    _RDKIT_OK = True
except Exception:  # pragma: no cover
    Chem = None
    _RDKIT_OK = False


def _safe_pairwise_distances(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rel = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = rel.square().sum(dim=-1).clamp_min(1.0e-8).sqrt()
    return rel, dist


def _functional_group_assignments(smiles: str, n_atoms: int, device: torch.device) -> torch.Tensor:
    if not _RDKIT_OK or not smiles:
        return torch.arange(n_atoms, device=device, dtype=torch.long)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != n_atoms:
        return torch.arange(n_atoms, device=device, dtype=torch.long)

    assignments = torch.full((n_atoms,), -1, device=device, dtype=torch.long)
    group_id = 0
    patterns = [
        ("carbonyl", "[CX3]=[OX1]"),
        ("hetero_alpha", "[CH2,CH3][O,N,S]"),
        ("aromatic_hetero", "[a][N,O,S]"),
        ("halogenated", "[C,c][F,Cl,Br,I]"),
        ("sulfur_center", "[SX2,SX4,SX6]"),
        ("phosphorus_center", "[PX3,PX4,PX5]"),
    ]
    for _, smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            idxs = [int(v) for v in match]
            unassigned = [idx for idx in idxs if assignments[idx] < 0]
            if not unassigned:
                continue
            assignments[idxs] = group_id
            group_id += 1
    for idx in range(n_atoms):
        if assignments[idx] < 0:
            assignments[idx] = group_id
            group_id += 1
    return assignments


@dataclass
class TopologyKernelOutput:
    local_rbf: torch.Tensor
    global_rbf: torch.Tensor
    atomic_features: torch.Tensor
    functional_group_features: torch.Tensor
    conformer_features: torch.Tensor
    functional_group_assignments: torch.Tensor
    distance_matrix: torch.Tensor
    masks: Dict[str, torch.Tensor]


class DenseLocalGaussianRBF(nn.Module):
    def __init__(self, n_basis: int = 24, r_min: float = 1.0, r_max: float = 2.5) -> None:
        super().__init__()
        centers = torch.linspace(float(r_min), float(r_max), steps=int(n_basis))
        width = (float(r_max) - float(r_min)) / max(int(n_basis) - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(width * width, 1.0e-6)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff.pow(2))


class NonVanishingEwaldRBF(nn.Module):
    def __init__(self, n_basis: int = 24, r_min: float = 2.5, r_max: float = 12.0, alpha: float = 0.18) -> None:
        super().__init__()
        centers = torch.linspace(float(r_min), float(r_max), steps=int(n_basis))
        self.register_buffer("centers", centers)
        self.alpha = float(alpha)
        self.soft_tail = float(r_max)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        gaussian = torch.exp(-self.alpha * diff.pow(2))
        ewald_tail = 1.0 / (1.0 + (distances.unsqueeze(-1) / self.soft_tail).pow(2))
        screened = torch.erfc(0.35 * distances.unsqueeze(-1))
        return gaussian * screened + 0.2 * ewald_tail


class MultiScale_Topology_Kernel(nn.Module):
    def __init__(
        self,
        local_basis_dim: int = 24,
        global_basis_dim: int = 24,
        cutoff: float = 12.0,
    ) -> None:
        super().__init__()
        self.local_basis = DenseLocalGaussianRBF(n_basis=local_basis_dim, r_min=1.0, r_max=2.5)
        self.global_basis = NonVanishingEwaldRBF(n_basis=global_basis_dim, r_min=2.5, r_max=cutoff)
        self.cutoff = float(cutoff)
        self.atomic_proj = nn.Sequential(
            nn.Linear(local_basis_dim + global_basis_dim + 4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.group_proj = nn.Sequential(
            nn.Linear(128 + local_basis_dim + global_basis_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.conformer_proj = nn.Sequential(
            nn.Linear(128 + global_basis_dim + 6, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

    def _aggregate_groups(self, atomic_features: torch.Tensor, assignments: torch.Tensor, local_pool: torch.Tensor, global_pool: torch.Tensor) -> torch.Tensor:
        n_groups = int(assignments.max().item()) + 1 if assignments.numel() else 0
        if n_groups == 0:
            return atomic_features.new_zeros((0, 128))
        rows = []
        for group_idx in range(n_groups):
            mask = assignments == group_idx
            atom_mean = atomic_features[mask].mean(dim=0)
            local_mean = local_pool[mask].mean(dim=0)
            global_mean = global_pool[mask].mean(dim=0)
            rows.append(self.group_proj(torch.cat([atom_mean, local_mean, global_mean], dim=0)))
        return torch.stack(rows, dim=0)

    def forward(self, manifold) -> TopologyKernelOutput:
        pos = manifold.pos
        species = manifold.species.to(device=pos.device, dtype=pos.dtype)
        forces = manifold.forces.to(device=pos.device, dtype=pos.dtype)
        n_atoms = int(pos.shape[0])
        rel, dist = _safe_pairwise_distances(pos)

        eye_mask = torch.eye(n_atoms, device=pos.device, dtype=torch.bool)
        local_mask = (dist >= 1.0) & (dist <= 2.5) & (~eye_mask)
        global_mask = (dist > 2.5) & (dist <= self.cutoff) & (~eye_mask)

        local_rbf = self.local_basis(dist) * local_mask.unsqueeze(-1).to(dtype=pos.dtype)
        global_rbf = self.global_basis(dist) * global_mask.unsqueeze(-1).to(dtype=pos.dtype)

        local_pool = local_rbf.sum(dim=1)
        global_pool = global_rbf.sum(dim=1)
        coord_stats = torch.cat(
            [
                pos,
                species.unsqueeze(-1) / 60.0,
            ],
            dim=-1,
        )
        atomic_features = self.atomic_proj(torch.cat([local_pool, global_pool, coord_stats], dim=-1))

        assignments = _functional_group_assignments(getattr(manifold.seed, "smiles", ""), n_atoms, pos.device)
        group_features = self._aggregate_groups(atomic_features, assignments, local_pool, global_pool)

        centroid = pos.mean(dim=0, keepdim=True)
        conformer_input = torch.cat(
            [
                atomic_features.mean(dim=0),
                global_pool.mean(dim=0),
                centroid.squeeze(0),
                forces.mean(dim=0),
            ],
            dim=0,
        )
        conformer_features = self.conformer_proj(conformer_input).unsqueeze(0)

        return TopologyKernelOutput(
            local_rbf=local_rbf,
            global_rbf=global_rbf,
            atomic_features=atomic_features,
            functional_group_features=group_features,
            conformer_features=conformer_features,
            functional_group_assignments=assignments,
            distance_matrix=dist,
            masks={"local": local_mask, "global": global_mask},
        )
