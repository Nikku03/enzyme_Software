from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState

from .hamiltonian import HamiltonianTerms, NEXUS_Hamiltonian
from .trajectory import NEXUS_Trajectory


@dataclass
class ThermostatState:
    enabled: bool
    kinetic_threshold: float
    damping: float


class SymplecticLeapfrog(nn.Module):
    def __init__(
        self,
        hamiltonian_engine: NEXUS_Hamiltonian,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> None:
        super().__init__()
        self.H = hamiltonian_engine
        self.smiles = smiles
        self.species = species
        self.accessibility_field = accessibility_field
        self.ddi_occupancy = ddi_occupancy

    def compute_force(self, q: torch.Tensor) -> torch.Tensor:
        q_eval = q.clone().requires_grad_(True)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            physical, reactive, _ = self.H.compute_potential_energy(
                q_eval,
                smiles=self.smiles,
                species=self.species,
                accessibility_field=self.accessibility_field,
                ddi_occupancy=self.ddi_occupancy,
            )
        potential = physical + self.H.coupling_lambda * reactive
        force = torch.autograd.grad(
            outputs=potential,
            inputs=q_eval,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        return -force

    def compute_velocity(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_eval = p.clone().requires_grad_(True)
        species = self.species
        if species is None:
            raise ValueError("species must be provided to compute velocity")
        kinetic = self.H.compute_kinetic_energy(
            p_eval,
            species,
            q=q,
            smiles=self.smiles,
        )
        velocity = torch.autograd.grad(
            outputs=kinetic,
            inputs=p_eval,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        return velocity


class Symplectic_ODE_Solver(nn.Module):
    def __init__(
        self,
        thermostat: bool = True,
        kinetic_threshold: float = 25.0,
        damping: float = 0.995,
    ) -> None:
        super().__init__()
        self.thermostat = ThermostatState(
            enabled=bool(thermostat),
            kinetic_threshold=float(kinetic_threshold),
            damping=float(damping),
        )

    def _hamiltonian_terms(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q: torch.Tensor,
        p: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> HamiltonianTerms:
        return hamiltonian(
            q,
            p,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
            return_terms=True,
        )

    def _apply_thermostat(self, momentum: torch.Tensor, kinetic: torch.Tensor) -> torch.Tensor:
        if not self.thermostat.enabled:
            return momentum
        if float(kinetic.detach().cpu().item()) <= self.thermostat.kinetic_threshold:
            return momentum
        return momentum * self.thermostat.damping

    def integrate(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        *,
        steps: int,
        dt: float,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NEXUS_Trajectory:
        q = q_init.clone().requires_grad_(True)
        p = p_init.clone().requires_grad_(True)
        dt_t = torch.as_tensor(float(dt), dtype=q.dtype, device=q.device)
        leapfrog = SymplecticLeapfrog(
            hamiltonian,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )

        q_hist = [q]
        p_hist = [p]
        h_hist = []
        kinetic_hist = []
        physical_hist = []
        reactive_hist = []
        lagrangian_terms = []

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            terms0 = self._hamiltonian_terms(
                hamiltonian,
                q,
                p,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
        h_hist.append(terms0.total)
        kinetic_hist.append(terms0.kinetic)
        physical_hist.append(terms0.physical)
        reactive_hist.append(terms0.reactive)
        lagrangian_terms.append(terms0.kinetic - (terms0.physical + hamiltonian.coupling_lambda * terms0.reactive))
        current_force = leapfrog.compute_force(q)

        for _ in range(int(steps)):
            q = q_hist[-1]
            p = p_hist[-1]

            p_half = p + 0.5 * dt_t * current_force
            velocity = leapfrog.compute_velocity(p_half, q)
            q_next = (q + dt_t * velocity).requires_grad_(True)
            next_force = leapfrog.compute_force(q_next)
            p_next = (p_half + 0.5 * dt_t * next_force).requires_grad_(True)
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                terms_pred = self._hamiltonian_terms(
                    hamiltonian,
                    q_next,
                    p_next,
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
            p_next = self._apply_thermostat(p_next, terms_pred.kinetic).requires_grad_(True)
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                terms_final = self._hamiltonian_terms(
                    hamiltonian,
                    q_next,
                    p_next,
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
            current_force = next_force

            q_hist.append(q_next)
            p_hist.append(p_next)
            h_hist.append(terms_final.total)
            kinetic_hist.append(terms_final.kinetic)
            physical_hist.append(terms_final.physical)
            reactive_hist.append(terms_final.reactive)
            lagrangian_terms.append(
                terms_final.kinetic - (terms_final.physical + hamiltonian.coupling_lambda * terms_final.reactive)
            )

        q_path = torch.stack(q_hist, dim=0)
        p_path = torch.stack(p_hist, dim=0)
        h_path = torch.stack(h_hist, dim=0)
        kinetic_path = torch.stack(kinetic_hist, dim=0)
        physical_path = torch.stack(physical_hist, dim=0)
        reactive_path = torch.stack(reactive_hist, dim=0)
        action_integral = dt_t * torch.stack(lagrangian_terms, dim=0).sum()
        hamiltonian_drift = h_path[-1] - h_path[0]

        return NEXUS_Trajectory(
            q_path=q_path,
            p_path=p_path,
            h_path=h_path,
            hamiltonian_drift=hamiltonian_drift,
            action_integral=action_integral,
            kinetic_path=kinetic_path,
            physical_path=physical_path,
            reactive_path=reactive_path,
        )

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        *,
        steps: int,
        dt: float,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NEXUS_Trajectory:
        return self.integrate(
            hamiltonian,
            q_init,
            p_init,
            steps=steps,
            dt=dt,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )


__all__ = ["SymplecticLeapfrog", "Symplectic_ODE_Solver", "ThermostatState"]
