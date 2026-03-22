from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from nexus.core.manifold_refiner import Refined_NEXUS_Manifold
from nexus.core.multiscale_engine import MultiScale_Topology_Engine
from nexus.physics.activation import CatalyticActivationState, CatalyticStateActivation
from nexus.symmetry.engine import O3_Symmetry_Engine


@dataclass
class Reactive_NEXUS_Manifold:
    pos: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    species: torch.Tensor
    reactivity_field: torch.Tensor
    reaction_pathway: torch.Tensor
    base_manifold: Refined_NEXUS_Manifold
    master_jacobian: torch.Tensor
    activation: CatalyticActivationState
    metadata: Dict[str, object] = field(default_factory=dict)


class JacobianTracker(nn.Module):
    def __init__(self, symmetry_engine: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.symmetry_engine = symmetry_engine or O3_Symmetry_Engine()

    def reactivity_field(self, manifold: Refined_NEXUS_Manifold) -> torch.Tensor:
        engine_output = self.symmetry_engine(manifold)
        if hasattr(engine_output, "fused_features"):
            features = engine_output.fused_features
            entropy_loss = engine_output.entropy_loss
        else:
            features = engine_output
            entropy_loss = None
        psi = (
            0.25 * features["0e"].pow(2).mean()
            + 0.35 * features["0o"].abs().mean()
            + 0.15 * features["1o"].pow(2).mean()
            + 0.10 * features["1e"].pow(2).mean()
            + 0.10 * features["2e"].pow(2).mean()
            + 0.05 * features["2o"].abs().mean()
            + 0.20 * features["parity_pseudoscalar"].abs().mean()
        )
        if "0e_topology" in features:
            psi = psi + 0.10 * features["0e_topology"].pow(2).mean()
        if "1o_topology" in features:
            psi = psi + 0.05 * features["1o_topology"].pow(2).mean()
        if "2e_topology" in features:
            psi = psi + 0.05 * features["2e_topology"].pow(2).mean()
        if entropy_loss is not None:
            psi = psi - 0.05 * entropy_loss
        return psi

    def master_jacobian(
        self,
        manifold: Refined_NEXUS_Manifold,
        reactivity_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        psi = reactivity_field if reactivity_field is not None else self.reactivity_field(manifold)
        grad = torch.autograd.grad(
            outputs=psi,
            inputs=manifold.pos,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        return grad

    def seed_jacobian(
        self,
        manifold: Refined_NEXUS_Manifold,
        reactivity_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        psi = reactivity_field if reactivity_field is not None else self.reactivity_field(manifold)
        grad = torch.autograd.grad(
            outputs=psi,
            inputs=manifold.base_seed.pos if hasattr(manifold, "base_seed") else manifold.seed.pos,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        return grad


class PoseOptimizer(nn.Module):
    def __init__(
        self,
        symmetry_engine: Optional[nn.Module] = None,
        activation_engine: Optional[CatalyticStateActivation] = None,
        max_displacement: float = 0.2,
    ) -> None:
        super().__init__()
        self.symmetry_engine = symmetry_engine or MultiScale_Topology_Engine()
        self.jacobian_tracker = JacobianTracker(self.symmetry_engine)
        self.activation_engine = activation_engine or CatalyticStateActivation()
        self.max_displacement = float(max_displacement)

    def _cap_displacement(self, displacement: torch.Tensor) -> torch.Tensor:
        norm = displacement.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(self.max_displacement / norm, max=1.0)
        return displacement * scale

    def optimize_reactive_pose(
        self,
        manifold: Refined_NEXUS_Manifold,
        eta: float = 0.01,
        gamma: float = 0.0025,
    ) -> Reactive_NEXUS_Manifold:
        psi = self.jacobian_tracker.reactivity_field(manifold)
        grad_psi = self.jacobian_tracker.master_jacobian(manifold, psi)
        grad_E = torch.autograd.grad(
            outputs=manifold.energy,
            inputs=manifold.pos,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        activation = self.activation_engine(manifold, grad_psi, grad_E)
        displacement = eta * grad_psi - gamma * grad_E + activation.activation_bias
        displacement = self._cap_displacement(displacement)
        reactive_pos = manifold.pos + displacement

        reactive_energy, reactive_forces = manifold.base_refiner.potential.energy_and_forces(
            reactive_pos,
            manifold.species,
            smiles=manifold.seed.smiles,
        ) if hasattr(manifold, "base_refiner") else (manifold.energy, manifold.forces)

        return Reactive_NEXUS_Manifold(
            pos=reactive_pos,
            energy=reactive_energy,
            forces=reactive_forces,
            species=manifold.species,
            reactivity_field=psi,
            reaction_pathway=displacement,
            base_manifold=manifold,
            master_jacobian=grad_psi,
            activation=activation,
            metadata={
                "eta": float(eta),
                "gamma": float(gamma),
                "spin_state": activation.spin_state,
                "mean_displacement_A": float(displacement.norm(dim=-1).mean().detach().cpu().item()),
                "max_displacement_A": float(displacement.norm(dim=-1).max().detach().cpu().item()),
            },
        )
