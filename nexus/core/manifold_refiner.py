from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _RDKIT_OK = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None
    _RDKIT_OK = False

from nexus.core.generative_agency import NEXUS_Seed
from nexus.physics.potential import MACEOFFPotential


@dataclass
class Refined_NEXUS_Manifold:
    pos: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    species: torch.Tensor
    seed: NEXUS_Seed
    base_refiner: nn.Module | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "Refined_NEXUS_Manifold":
        return Refined_NEXUS_Manifold(
            pos=self.pos.to(device),
            energy=self.energy.to(device),
            forces=self.forces.to(device),
            species=self.species.to(device),
            seed=self.seed.to(device),
            base_refiner=self.base_refiner,
            metadata=dict(self.metadata),
        )


class MACE_OFF_Refiner(nn.Module):
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        relaxation_steps: int = 8,
        step_size: float = 0.001,
        manifold_lock_weight: float = 0.025,
        supported_atomic_numbers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.relaxation_steps = max(1, int(relaxation_steps))
        self.step_size = float(step_size)
        self.manifold_lock_weight = float(manifold_lock_weight)
        self.potential = MACEOFFPotential(
            model_path=model_path,
            supported_atomic_numbers=supported_atomic_numbers,
        )

    def apply_metabolic_pressure(
        self,
        forces: torch.Tensor,
        metabolic_pressure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if metabolic_pressure is None:
            return forces
        pressure = metabolic_pressure.to(device=forces.device, dtype=forces.dtype)
        if pressure.shape != forces.shape:
            raise ValueError(f"metabolic_pressure shape {tuple(pressure.shape)} does not match forces {tuple(forces.shape)}")
        return forces + pressure

    def _step_size(self, step_idx: int) -> float:
        decay = 0.85 ** step_idx
        return self.step_size * decay

    def _fallback_etkdg_pos(self, seed: NEXUS_Seed) -> torch.Tensor:
        if not _RDKIT_OK:
            return seed.pos
        mol = Chem.MolFromSmiles(str(seed.smiles))
        if mol is None:
            return seed.pos
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1
        try:
            status = AllChem.EmbedMolecule(mol, params)
            if int(status) != 0:
                return seed.pos
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
            mol = Chem.RemoveHs(mol)
            conf = mol.GetConformer()
            coords = []
            for idx in range(mol.GetNumAtoms()):
                atom_pos = conf.GetAtomPosition(idx)
                coords.append([float(atom_pos.x), float(atom_pos.y), float(atom_pos.z)])
            pos = torch.tensor(coords, dtype=seed.pos.dtype, device=seed.pos.device)
            return pos.requires_grad_(True)
        except Exception:
            return seed.pos

    def _is_finite_scalar(self, value: torch.Tensor) -> bool:
        return bool(torch.isfinite(value).all().item())

    def _relax_single(
        self,
        seed: NEXUS_Seed,
        metabolic_pressure: torch.Tensor | None = None,
    ) -> Refined_NEXUS_Manifold:
        pos = seed.pos
        anchor = seed.pos
        z = seed.z.to(pos.device)
        last_energy = pos.new_zeros(())
        last_forces = torch.zeros_like(pos)

        for step_idx in range(self.relaxation_steps):
            potential_energy = self.potential(pos, z, smiles=seed.smiles)
            lock_energy = self.manifold_lock_weight * (pos - anchor).pow(2).mean()
            total_energy = potential_energy + lock_energy
            if not self._is_finite_scalar(total_energy) or not bool(torch.isfinite(pos).all().item()):
                pos = self._fallback_etkdg_pos(seed)
                anchor = pos
                potential_energy = self.potential(pos, z, smiles=seed.smiles)
                lock_energy = self.manifold_lock_weight * (pos - anchor).pow(2).mean()
                total_energy = potential_energy + lock_energy
                if not self._is_finite_scalar(total_energy):
                    total_energy = torch.zeros((), dtype=pos.dtype, device=pos.device, requires_grad=True)
                    last_forces = torch.zeros_like(pos)
                    break
            grad = torch.autograd.grad(
                outputs=total_energy,
                inputs=pos,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )[0]
            base_forces = self.potential.clip_forces(-grad)
            total_forces = self.apply_metabolic_pressure(base_forces, metabolic_pressure)
            total_forces = self.potential.clip_forces(total_forces)
            pos = pos + self._step_size(step_idx) * total_forces
            pos = torch.nan_to_num(pos, nan=0.0, posinf=5.0, neginf=-5.0)
            last_energy = total_energy
            last_forces = total_forces

        final_potential = self.potential(pos, z, smiles=seed.smiles)
        final_lock = self.manifold_lock_weight * (pos - anchor).pow(2).mean()
        final_energy = final_potential + final_lock
        if not self._is_finite_scalar(final_energy) or not bool(torch.isfinite(pos).all().item()):
            pos = self._fallback_etkdg_pos(seed)
            anchor = pos
            final_potential = self.potential(pos, z, smiles=seed.smiles)
            final_lock = self.manifold_lock_weight * (pos - anchor).pow(2).mean()
            final_energy = final_potential + final_lock
            if not self._is_finite_scalar(final_energy):
                final_energy = torch.zeros((), dtype=pos.dtype, device=pos.device, requires_grad=True)
                final_forces = torch.zeros_like(pos)
                pos = pos.requires_grad_(True)
                return Refined_NEXUS_Manifold(
                    pos=pos,
                    energy=final_energy,
                    forces=final_forces,
                    species=z,
                    seed=seed,
                    base_refiner=self,
                    metadata={
                        "backend": self.potential.backend,
                        "relaxation_steps": self.relaxation_steps,
                        "step_size": self.step_size,
                        "manifold_lock_weight": self.manifold_lock_weight,
                        "initial_energy": float(last_energy.detach().item()) if last_energy.numel() == 1 and torch.isfinite(last_energy).all() else None,
                        "used_etkdg_fallback": True,
                    },
                )
        final_grad = torch.autograd.grad(
            outputs=final_energy,
            inputs=pos,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        final_forces = self.potential.clip_forces(self.apply_metabolic_pressure(-final_grad, metabolic_pressure))
        pos = pos.requires_grad_(True)

        return Refined_NEXUS_Manifold(
            pos=pos,
            energy=final_energy,
            forces=final_forces,
            species=z,
            seed=seed,
            base_refiner=self,
            metadata={
                "backend": self.potential.backend,
                "relaxation_steps": self.relaxation_steps,
                "step_size": self.step_size,
                "manifold_lock_weight": self.manifold_lock_weight,
                "initial_energy": float(last_energy.detach().item()) if last_energy.numel() == 1 and torch.isfinite(last_energy).all() else None,
                "used_etkdg_fallback": False,
            },
        )

    def forward(
        self,
        seeds: NEXUS_Seed | Sequence[NEXUS_Seed],
        metabolic_pressure: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> Refined_NEXUS_Manifold | List[Refined_NEXUS_Manifold]:
        if isinstance(seeds, NEXUS_Seed):
            single_pressure = metabolic_pressure if isinstance(metabolic_pressure, torch.Tensor) or metabolic_pressure is None else None
            return self._relax_single(seeds, metabolic_pressure=single_pressure)

        seed_list = list(seeds)
        if metabolic_pressure is None:
            pressures: List[torch.Tensor | None] = [None] * len(seed_list)
        elif isinstance(metabolic_pressure, torch.Tensor):
            pressures = [metabolic_pressure for _ in seed_list]
        else:
            pressures = list(metabolic_pressure)
            if len(pressures) != len(seed_list):
                raise ValueError("metabolic_pressure list must match number of seeds")
        return [
            self._relax_single(seed, metabolic_pressure=pressure)
            for seed, pressure in zip(seed_list, pressures)
        ]


Physical_Refiner = MACE_OFF_Refiner
