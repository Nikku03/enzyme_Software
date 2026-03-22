from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nexus.core.field_engine import FSHN_Field_Generator
from nexus.core.generative_agency import NEXT_Mol_Generative_Agency, NEXUS_Seed
from nexus.core.manifold_refiner import MACE_OFF_Refiner, Refined_NEXUS_Manifold
from nexus.field.siren_base import SIREN_OMEGA_0
from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState
from .clifford_math import embed_coordinates
from .constants import ATOMIC_MASSES


@dataclass
class HamiltonianTerms:
    kinetic: torch.Tensor
    physical: torch.Tensor
    reactive: torch.Tensor
    total: torch.Tensor


class NEXUS_Hamiltonian(nn.Module):
    def __init__(
        self,
        agency: Optional[NEXT_Mol_Generative_Agency] = None,
        refiner: Optional[MACE_OFF_Refiner] = None,
        field_engine: Optional[FSHN_Field_Generator] = None,
        coupling_lambda: float = 0.25,
        reactive_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.agency = agency or NEXT_Mol_Generative_Agency("", "")
        self.refiner = refiner or MACE_OFF_Refiner()
        self.field_engine = field_engine or FSHN_Field_Generator()
        self.coupling_lambda = nn.Parameter(torch.tensor(float(coupling_lambda)))
        self.reactive_scale = nn.Parameter(torch.tensor(float(reactive_scale_init)))
        mass_table = torch.ones(128, dtype=torch.float32) * 12.0
        for z, mass in ATOMIC_MASSES.items():
            mass_table[z] = float(mass)
        self.register_buffer("mass_table", mass_table)
        self.register_buffer("reactive_reference", torch.tensor(10.0))

    def _field_dtype(self) -> torch.dtype:
        for param in self.field_engine.parameters():
            return param.dtype
        return torch.float32

    def atomic_masses(self, species: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        idx = species.to(dtype=torch.long).clamp(min=0, max=127)
        return self.mass_table.to(device=device, dtype=dtype)[idx]

    def _field_manifold(self, manifold: Refined_NEXUS_Manifold) -> Refined_NEXUS_Manifold:
        field_dtype = self._field_dtype()
        if manifold.pos.dtype == field_dtype:
            return manifold
        field_seed = NEXUS_Seed(
            pos=manifold.seed.pos.to(dtype=field_dtype),
            z=manifold.seed.z,
            latent_blueprint=manifold.seed.latent_blueprint,
            smiles=manifold.seed.smiles,
            atom_symbols=list(manifold.seed.atom_symbols),
            chirality_codes=manifold.seed.chirality_codes,
            jacobian_hook=manifold.seed.jacobian_hook,
            metadata=dict(manifold.seed.metadata),
        )
        return Refined_NEXUS_Manifold(
            pos=manifold.pos.to(dtype=field_dtype),
            energy=manifold.energy.to(dtype=field_dtype),
            forces=manifold.forces.to(dtype=field_dtype),
            species=manifold.species,
            seed=field_seed,
            base_refiner=manifold.base_refiner,
            metadata=dict(manifold.metadata),
        )

    def _compute_reactive_from_field(self, field, field_manifold: Refined_NEXUS_Manifold) -> torch.Tensor:
        centroid = field_manifold.pos.mean(dim=0, keepdim=True)
        direction = field_manifold.forces + 0.1 * (field_manifold.pos - centroid)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        query_points = field_manifold.pos + 1.0e-2 * direction
        psi_atoms = field.query(query_points)
        psi_raw = torch.nn.functional.softplus(psi_atoms).mean()
        target_ref = psi_raw.detach().abs().clamp_min(1.0)
        self.reactive_reference.mul_(0.95).add_(0.05 * target_ref.to(self.reactive_reference.device))
        return self.reactive_scale.to(device=psi_raw.device, dtype=psi_raw.dtype) * (
            psi_raw / self.reactive_reference.to(device=psi_raw.device, dtype=psi_raw.dtype).clamp_min(1.0)
        )

    def compute_zora_factor(
        self,
        q: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        field=None,
        manifold: Optional[Refined_NEXUS_Manifold] = None,
    ) -> torch.Tensor:
        manifold = manifold or self.build_manifold(smiles, q=q, species=species)
        if field is None:
            field = self.field_engine(self._field_manifold(manifold))
        zora = field.quantum_enforcer.compute_zora_correction(
            field.atom_coords,
            field.atom_coords,
            field.atomic_numbers,
        ).squeeze(-1)
        return zora.to(device=q.device, dtype=q.dtype)

    def compute_kinetic_energy(
        self,
        p: torch.Tensor,
        species: torch.Tensor,
        *,
        q: Optional[torch.Tensor] = None,
        smiles: Optional[str] = None,
        zora_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        masses = self.atomic_masses(species, p.dtype, p.device).unsqueeze(-1)
        if zora_factor is None and q is not None and smiles is not None:
            zora_factor = self.compute_zora_factor(q, smiles=smiles, species=species).detach()
        if zora_factor is not None:
            zora = zora_factor.to(device=p.device, dtype=p.dtype).unsqueeze(-1).clamp_min(1.0)
            kinetic = 0.5 * (p.pow(2) / (masses.clamp_min(1.0e-6) * zora)).sum()
        else:
            kinetic = 0.5 * (p.pow(2) / masses.clamp_min(1.0e-6)).sum()
        if not bool(torch.isfinite(kinetic).all().item()):
            raise RuntimeError("Modified kinetic energy is non-finite after ZORA scaling")
        return kinetic

    def compute_force(
        self,
        q: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        q_eval = q.clone().requires_grad_(True)
        physical, reactive, _ = self.compute_potential_energy(
            q_eval,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        potential = physical + self.coupling_lambda * reactive
        return -torch.autograd.grad(
            outputs=potential,
            inputs=q_eval,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]

    def compute_velocity(
        self,
        p: torch.Tensor,
        *,
        q: torch.Tensor,
        smiles: str,
        species: torch.Tensor,
    ) -> torch.Tensor:
        p_eval = p.clone().requires_grad_(True)
        kinetic = self.compute_kinetic_energy(
            p_eval,
            species,
            q=q,
            smiles=smiles,
        )
        return torch.autograd.grad(
            outputs=kinetic,
            inputs=p_eval,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]

    def compute_clifford_force(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        velocity = self.compute_velocity(p, q=q, smiles=smiles, species=species)
        force = self.compute_force(
            q,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        mv = embed_coordinates(velocity)
        mv[..., 4:7] = force
        mv[..., 7] = (velocity * force).sum(dim=-1)
        return mv

    def compute_accessibility_gate(
        self,
        q: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        if accessibility_field is None:
            return torch.ones((), dtype=q.dtype, device=q.device)
        if ddi_occupancy is None:
            return accessibility_field.gate_scalar(q).to(device=q.device, dtype=q.dtype)
        return accessibility_field.gate_scalar_with_occupancy(q, ddi_occupancy).to(device=q.device, dtype=q.dtype)

    def _seed_from_inputs(
        self,
        smiles: str,
        q: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
    ) -> NEXUS_Seed:
        seed = self.agency(smiles)
        if q is None and species is None:
            return seed
        return NEXUS_Seed(
            pos=(q if q is not None else seed.pos).requires_grad_(True),
            z=(species if species is not None else seed.z).to(device=seed.z.device),
            latent_blueprint=seed.latent_blueprint,
            smiles=seed.smiles,
            atom_symbols=list(seed.atom_symbols),
            chirality_codes=seed.chirality_codes,
            jacobian_hook=seed.jacobian_hook,
            metadata=dict(seed.metadata),
        )

    def build_manifold(
        self,
        smiles: str,
        q: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
    ) -> Refined_NEXUS_Manifold:
        seed = self._seed_from_inputs(smiles, q=q, species=species)
        if q is None:
            return self.refiner(seed)
        q = q.requires_grad_(True)
        z = (species if species is not None else seed.z).to(device=q.device)
        physical_energy, forces = self.refiner.potential.energy_and_forces(q, z, smiles=smiles)
        return Refined_NEXUS_Manifold(
            pos=q,
            energy=physical_energy,
            forces=forces,
            species=z,
            seed=seed,
            base_refiner=self.refiner,
            metadata={"backend": self.refiner.potential.backend, "hamiltonian_override": True},
        )

    def compute_potential_energy(
        self,
        q: torch.Tensor,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        *,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
        return_field: bool = False,
    ):
        manifold = self.build_manifold(smiles, q=q, species=species)
        field_manifold = self._field_manifold(manifold)
        field = self.field_engine(field_manifold)
        reactive = self._compute_reactive_from_field(field, field_manifold)
        reactive = reactive * self.compute_accessibility_gate(
            q,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        if return_field:
            return manifold.energy, reactive.to(dtype=manifold.energy.dtype), manifold, field
        return manifold.energy, reactive.to(dtype=manifold.energy.dtype), manifold

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
        return_terms: bool = False,
    ):
        if abs(float(self.field_engine.omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"Field engine omega_0 must remain locked at {SIREN_OMEGA_0}")
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        physical, reactive, manifold, field = self.compute_potential_energy(
            q,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
            return_field=True,
        )
        zora_factor = self.compute_zora_factor(
            q,
            smiles=smiles,
            species=manifold.species,
            field=field,
            manifold=manifold,
        ).detach()
        kinetic = self.compute_kinetic_energy(
            p,
            manifold.species,
            q=q,
            smiles=smiles,
            zora_factor=zora_factor,
        )
        total = kinetic + physical + self.coupling_lambda * reactive
        if return_terms:
            return HamiltonianTerms(
                kinetic=kinetic,
                physical=physical,
                reactive=reactive,
                total=total,
            )
        return total
