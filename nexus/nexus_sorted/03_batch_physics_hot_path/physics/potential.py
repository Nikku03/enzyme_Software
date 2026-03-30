from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from rdkit import Chem

    _RDKIT_OK = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    _RDKIT_OK = False

try:
    import mace  # type: ignore
    import e3nn  # type: ignore

    _MACE_OK = True
except Exception:  # pragma: no cover - optional dependency
    mace = None
    e3nn = None
    _MACE_OK = False


_SUPPORTED_Z = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
_COVALENT_RADII = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}
_VDW_RADII = {
    1: 1.20,
    5: 1.92,
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
_EPSILON = {
    1: 0.020,
    5: 0.070,
    6: 0.090,
    7: 0.100,
    8: 0.120,
    9: 0.080,
    15: 0.110,
    16: 0.130,
    17: 0.090,
    35: 0.100,
    53: 0.110,
}


@lru_cache(maxsize=4096)
def _bond_template_from_smiles(smiles: str) -> Tuple[Tuple[int, int, float, float], ...]:
    if not _RDKIT_OK or not smiles:
        return tuple()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()
    rows: List[Tuple[int, int, float, float]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        z_i = int(mol.GetAtomWithIdx(begin).GetAtomicNum())
        z_j = int(mol.GetAtomWithIdx(end).GetAtomicNum())
        order = float(bond.GetBondTypeAsDouble())
        r0 = float(_COVALENT_RADII.get(z_i, 0.8) + _COVALENT_RADII.get(z_j, 0.8) - 0.08 * max(order - 1.0, 0.0))
        rows.append((begin, end, r0, order))
    return tuple(rows)


class MACEOFFPotential(nn.Module):
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        supported_atomic_numbers: Sequence[int] | None = None,
        softcore_distance: float = 1.0,
        max_force_norm: float = 25.0,
    ) -> None:
        super().__init__()
        self.model_path = str(model_path or "")
        self.supported_atomic_numbers = list(supported_atomic_numbers or _SUPPORTED_Z)
        self.backend = "surrogate_off"
        self.mace_model = None
        self.softcore_distance = float(max(1.0e-3, softcore_distance))
        self.max_force_norm = float(max(1.0e-3, max_force_norm))

        if self.model_path and _MACE_OK:
            path = Path(self.model_path)
            if path.exists():
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(payload, nn.Module):
                        self.mace_model = payload
                        self.backend = "mace_off23"
                except Exception:
                    self.mace_model = None
                    self.backend = "surrogate_off"

    def _pairwise_nonbonded(self, pos: torch.Tensor, z: torch.Tensor, bonded_mask: torch.Tensor) -> torch.Tensor:
        n_atoms = int(pos.shape[0])
        if n_atoms <= 1:
            return pos.new_zeros(())

        z_list = [int(v) for v in z.detach().cpu().tolist()]
        sigma = pos.new_tensor([_VDW_RADII.get(v, 1.7) for v in z_list], dtype=pos.dtype)
        epsilon = pos.new_tensor([_EPSILON.get(v, 0.1) for v in z_list], dtype=pos.dtype)

        rel = pos.unsqueeze(1) - pos.unsqueeze(0)
        raw_dist = rel.square().sum(dim=-1).clamp_min(1.0e-6).sqrt()
        dist = torch.sqrt(raw_dist.square() + raw_dist.new_tensor(self.softcore_distance ** 2))
        sigma_ij = 0.5 * (sigma.unsqueeze(0) + sigma.unsqueeze(1))
        epsilon_ij = torch.sqrt(epsilon.unsqueeze(0) * epsilon.unsqueeze(1))

        ratio = sigma_ij / dist
        repulsion = ratio.pow(12)
        attraction = 0.15 * ratio.pow(6)
        pair_energy = epsilon_ij * (repulsion - attraction)

        short_contact = torch.relu(raw_dist.new_tensor(self.softcore_distance) - raw_dist)
        softcore_penalty = 4.0 * epsilon_ij * short_contact.square()
        pair_energy = pair_energy + softcore_penalty

        mask = torch.triu(torch.ones((n_atoms, n_atoms), device=pos.device, dtype=torch.bool), diagonal=1)
        mask = mask & (~bonded_mask)
        return pair_energy.masked_select(mask).sum()

    def _bonded_energy(self, pos: torch.Tensor, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        template = _bond_template_from_smiles(smiles)
        n_atoms = int(pos.shape[0])
        bonded_mask = torch.zeros((n_atoms, n_atoms), dtype=torch.bool, device=pos.device)
        if not template:
            return pos.new_zeros(()), bonded_mask

        begin = torch.tensor([row[0] for row in template], dtype=torch.long, device=pos.device)
        end = torch.tensor([row[1] for row in template], dtype=torch.long, device=pos.device)
        r0 = pos.new_tensor([row[2] for row in template], dtype=pos.dtype)
        order = pos.new_tensor([row[3] for row in template], dtype=pos.dtype)

        diff = pos[begin] - pos[end]
        dist = diff.square().sum(dim=-1).clamp_min(1.0e-8).sqrt()
        k_bond = 18.0 + 6.0 * (order - 1.0)
        bond_energy = 0.5 * k_bond * (dist - r0).pow(2)

        bonded_mask[begin, end] = True
        bonded_mask[end, begin] = True
        return bond_energy.sum(), bonded_mask

    def surrogate_energy(self, pos: torch.Tensor, z: torch.Tensor, smiles: str = "") -> torch.Tensor:
        bond_energy, bonded_mask = self._bonded_energy(pos, smiles)
        nonbonded = self._pairwise_nonbonded(pos, z, bonded_mask)
        centroid = pos.mean(dim=0, keepdim=True)
        radius_penalty = 0.0025 * (pos - centroid).pow(2).sum()
        return bond_energy + nonbonded + radius_penalty

    def forward(self, pos: torch.Tensor, z: torch.Tensor, *, smiles: str = "") -> torch.Tensor:
        if self.backend == "mace_off23" and self.mace_model is not None:
            return self.mace_model(pos=pos, z=z)
        return self.surrogate_energy(pos, z, smiles=smiles)

    def clip_forces(self, forces: torch.Tensor) -> torch.Tensor:
        norm = forces.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(self.max_force_norm / norm, max=1.0)
        return forces * scale

    def energy_and_forces(self, pos: torch.Tensor, z: torch.Tensor, *, smiles: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.forward(pos, z, smiles=smiles)
        grad = torch.autograd.grad(
            outputs=energy,
            inputs=pos,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        forces = self.clip_forces(-grad)
        return energy, forces
