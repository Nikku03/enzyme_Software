from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nexus.core.recursive_metabolism import RecursiveMetabolismTree


@dataclass
class NCFAFluxResult:
    node_abundance: torch.Tensor
    edge_flux: torch.Tensor
    normalized_adjacency: torch.Tensor
    residual_mass: torch.Tensor
    total_input_mass: torch.Tensor
    total_output_mass: torch.Tensor
    flux_consistency_loss: torch.Tensor
    intervention_mask: Optional[torch.Tensor] = None
    kinetic_gate: Optional[torch.Tensor] = None


class NCFAFluxPropagator(nn.Module):
    def __init__(
        self,
        eps: float = 1.0e-8,
        *,
        temperature_kelvin: float = 310.15,
        kb_kcal_per_mol_k: float = 0.00198720425864083,
        rate_reference: float = 6.2e12,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.temperature_kelvin = float(temperature_kelvin)
        self.kb_kcal_per_mol_k = float(kb_kcal_per_mol_k)
        self.rate_reference = float(max(rate_reference, 1.0))

    def build_kinetic_gate(
        self,
        *,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        gate = None

        if barrier_energy is not None:
            barrier = torch.as_tensor(barrier_energy)
            thermal = barrier.new_tensor(self.kb_kcal_per_mol_k * self.temperature_kelvin).clamp_min(self.eps)
            barrier = barrier.clamp(min=0.0, max=40.0)
            barrier_gate = torch.exp((-barrier / thermal).clamp(min=-60.0, max=0.0))
            gate = barrier_gate if gate is None else gate * barrier_gate

        if transmission is not None:
            transmission_t = torch.as_tensor(transmission).clamp(min=0.0, max=1.0)
            gate = transmission_t if gate is None else gate * transmission_t

        if metabolic_rate is not None:
            rate_t = torch.as_tensor(metabolic_rate).clamp_min(self.eps)
            rate_gate = (rate_t / rate_t.new_tensor(self.rate_reference)).clamp(min=self.eps, max=1.0)
            gate = rate_gate if gate is None else gate * rate_gate

        if gate is None:
            return None
        return gate.clamp(min=self.eps, max=1.0)

    def _broadcast_gate(self, adjacency: torch.Tensor, kinetic_gate: torch.Tensor) -> torch.Tensor:
        gate = kinetic_gate.to(dtype=adjacency.dtype, device=adjacency.device)
        if gate.ndim == 0:
            return gate
        if adjacency.ndim == 2:
            if gate.shape == adjacency.shape:
                return gate
            if gate.ndim == 1 and gate.numel() == adjacency.size(0):
                return gate.unsqueeze(-1)
            if gate.ndim == 1 and gate.numel() == adjacency.size(1):
                return gate.unsqueeze(0)
        if adjacency.ndim == 3:
            if gate.shape == adjacency.shape:
                return gate
            if gate.ndim == 1 and gate.numel() == adjacency.size(0):
                return gate.view(-1, 1, 1)
            if gate.ndim == 1 and gate.numel() == adjacency.size(1):
                return gate.view(1, -1, 1)
            if gate.ndim == 1 and gate.numel() == adjacency.size(2):
                return gate.view(1, 1, -1)
            if gate.ndim == 2 and gate.shape == adjacency.shape[:2]:
                return gate.unsqueeze(-1)
            if gate.ndim == 2 and gate.shape == adjacency.shape[1:]:
                return gate.unsqueeze(0)
        raise ValueError(
            f"kinetic_gate shape {tuple(gate.shape)} is not broadcastable to adjacency {tuple(adjacency.shape)}"
        )

    def apply_kinetic_gating(
        self,
        adjacency: torch.Tensor,
        *,
        kinetic_gate: Optional[torch.Tensor] = None,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        gate = kinetic_gate
        if gate is None:
            gate = self.build_kinetic_gate(
                barrier_energy=barrier_energy,
                transmission=transmission,
                metabolic_rate=metabolic_rate,
            )
        if gate is None:
            return adjacency, None
        broadcast_gate = self._broadcast_gate(adjacency, gate)
        return adjacency * broadcast_gate, gate.to(dtype=adjacency.dtype, device=adjacency.device)

    def _node_count(self, tree: RecursiveMetabolismTree) -> int:
        return max((node.node_id for node in tree.nodes), default=-1) + 1

    def build_structural_adjacency(self, tree: RecursiveMetabolismTree) -> torch.Tensor:
        n = self._node_count(tree)
        if n == 0:
            return torch.zeros(0, 0)
        device = tree.root.edge_weight.device
        dtype = tree.root.edge_weight.dtype
        adjacency = torch.zeros((n, n), dtype=dtype, device=device)
        for node in tree.nodes:
            if node.parent_node_id is None:
                continue
            adjacency[node.parent_node_id, node.node_id] = node.edge_weight.to(dtype=dtype, device=device)
        return adjacency

    def normalize_flux_adjacency(
        self,
        adjacency: torch.Tensor,
        *,
        kinetic_gate: Optional[torch.Tensor] = None,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        gated_adjacency, used_gate = self.apply_kinetic_gating(
            adjacency,
            kinetic_gate=kinetic_gate,
            barrier_energy=barrier_energy,
            transmission=transmission,
            metabolic_rate=metabolic_rate,
        )
        row_sums = gated_adjacency.sum(dim=-1, keepdim=True)
        scale = torch.maximum(row_sums, torch.ones_like(row_sums))
        return gated_adjacency / scale.clamp_min(self.eps), used_gate

    def apply_inhibition(
        self,
        adjacency: torch.Tensor,
        inhibitor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inhibitor_mask is None:
            return adjacency
        mask = inhibitor_mask.to(dtype=adjacency.dtype, device=adjacency.device)
        if mask.shape != adjacency.shape:
            raise ValueError(f"inhibitor_mask shape {tuple(mask.shape)} does not match adjacency {tuple(adjacency.shape)}")
        return adjacency * (1.0 - mask)

    def apply_intervention(
        self,
        adjacency: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if edge_mask is None:
            return adjacency, None
        mask = edge_mask.to(dtype=adjacency.dtype, device=adjacency.device)
        if mask.shape != adjacency.shape:
            raise ValueError(f"edge_mask shape {tuple(mask.shape)} does not match adjacency {tuple(adjacency.shape)}")
        intervened = adjacency * mask
        return intervened, mask

    def compute_flux_consistency_loss(
        self,
        W_struct: torch.Tensor,
        *,
        kinetic_gate: Optional[torch.Tensor] = None,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gated_flux, _ = self.apply_kinetic_gating(
            W_struct,
            kinetic_gate=kinetic_gate,
            barrier_energy=barrier_energy,
            transmission=transmission,
            metabolic_rate=metabolic_rate,
        )
        outgoing_flux = torch.sum(gated_flux, dim=-1)
        consistency_error = torch.relu(outgoing_flux - 1.0)
        return torch.mean(consistency_error)

    def _propagate_flux_matrix(
        self,
        W: torch.Tensor,
        initial_dose: torch.Tensor,
        *,
        generations: int = 4,
        kinetic_gate: Optional[torch.Tensor] = None,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> NCFAFluxResult:
        if W.ndim != 3:
            raise ValueError("W must have shape [batch, n_nodes, n_nodes]")
        if initial_dose.ndim == 1:
            initial_dose = initial_dose.unsqueeze(0).expand(W.size(0), -1)
        elif initial_dose.ndim != 2:
            raise ValueError("initial_dose must have shape [batch, n_nodes] or [n_nodes]")
        normalized, used_gate = self.normalize_flux_adjacency(
            W,
            kinetic_gate=kinetic_gate,
            barrier_energy=barrier_energy,
            transmission=transmission,
            metabolic_rate=metabolic_rate,
        )
        current_abundance = initial_dose.unsqueeze(-1)
        total_tree_flux = [current_abundance.squeeze(-1)]
        for _ in range(int(generations)):
            next_abundance = torch.bmm(normalized.transpose(1, 2), current_abundance)
            total_tree_flux.append(next_abundance.squeeze(-1))
            current_abundance = next_abundance
        abundance_path = torch.stack(total_tree_flux, dim=1)
        final_abundance = abundance_path[:, -1]
        edge_flux = normalized * initial_dose.unsqueeze(-1)
        residual_mass = torch.relu(initial_dose - edge_flux.sum(dim=-1))
        total_input = initial_dose.sum(dim=-1)
        total_output = final_abundance.sum(dim=-1) + residual_mass.sum(dim=-1)
        flux_loss = self.compute_flux_consistency_loss(normalized)
        return NCFAFluxResult(
            node_abundance=abundance_path,
            edge_flux=edge_flux,
            normalized_adjacency=normalized,
            residual_mass=residual_mass,
            total_input_mass=total_input,
            total_output_mass=total_output,
            flux_consistency_loss=flux_loss,
            intervention_mask=None,
            kinetic_gate=used_gate,
        )

    def propagate_flux(
        self,
        tree_or_W: RecursiveMetabolismTree | torch.Tensor,
        initial_dose: torch.Tensor | float,
        *,
        generations: int = 4,
        edge_mask: Optional[torch.Tensor] = None,
        kinetic_gate: Optional[torch.Tensor] = None,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> NCFAFluxResult:
        if torch.is_tensor(tree_or_W):
            W = self.apply_inhibition(tree_or_W, inhibitor_mask=edge_mask)
            return self._propagate_flux_matrix(
                W,
                torch.as_tensor(initial_dose, dtype=W.dtype, device=W.device),
                generations=generations,
                kinetic_gate=kinetic_gate,
                barrier_energy=barrier_energy,
                transmission=transmission,
                metabolic_rate=metabolic_rate,
            )

        tree = tree_or_W
        adjacency = self.build_structural_adjacency(tree)
        adjacency, used_mask = self.apply_intervention(adjacency, edge_mask=edge_mask)
        normalized, used_gate = self.normalize_flux_adjacency(
            adjacency,
            kinetic_gate=kinetic_gate,
            barrier_energy=barrier_energy,
            transmission=transmission,
            metabolic_rate=metabolic_rate,
        )

        n = normalized.size(0)
        if n == 0:
            zero = torch.zeros(())
            return NCFAFluxResult(
                node_abundance=torch.zeros(0),
                edge_flux=torch.zeros(0, 0),
                normalized_adjacency=normalized,
                residual_mass=torch.zeros(0),
                total_input_mass=zero,
                total_output_mass=zero,
                flux_consistency_loss=zero,
                intervention_mask=used_mask,
                kinetic_gate=used_gate,
            )

        dose = torch.as_tensor(initial_dose, dtype=normalized.dtype, device=normalized.device)
        abundance = torch.zeros(n, dtype=normalized.dtype, device=normalized.device)
        residual = torch.zeros(n, dtype=normalized.dtype, device=normalized.device)
        edge_flux = torch.zeros_like(normalized)

        abundance[tree.root.node_id] = dose
        node_map = {node.node_id: node for node in tree.nodes}

        for generation in tree.generations:
            for node_id in generation:
                parent_mass = abundance[node_id]
                if parent_mass.abs().item() <= self.eps:
                    continue
                outgoing = normalized[node_id]
                distributed = parent_mass * outgoing
                edge_flux[node_id] = distributed
                abundance = abundance + distributed
                residual[node_id] = parent_mass * (1.0 - outgoing.sum()).clamp_min(0.0)
                if node_id in node_map:
                    abundance[node_id] = abundance[node_id] - distributed.sum()

        total_output = abundance.sum()
        flux_loss = self.compute_flux_consistency_loss(normalized)
        return NCFAFluxResult(
            node_abundance=abundance,
            edge_flux=edge_flux,
            normalized_adjacency=normalized,
            residual_mass=residual,
            total_input_mass=dose,
            total_output_mass=total_output,
            flux_consistency_loss=flux_loss,
            intervention_mask=used_mask,
            kinetic_gate=used_gate,
        )

    def forward(
        self,
        tree_or_W: RecursiveMetabolismTree | torch.Tensor,
        initial_dose: torch.Tensor | float,
        *,
        generations: int = 4,
        edge_mask: Optional[torch.Tensor] = None,
        kinetic_gate: Optional[torch.Tensor] = None,
        barrier_energy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        metabolic_rate: Optional[torch.Tensor] = None,
    ) -> NCFAFluxResult:
        return self.propagate_flux(
            tree_or_W,
            initial_dose,
            generations=generations,
            edge_mask=edge_mask,
            kinetic_gate=kinetic_gate,
            barrier_energy=barrier_energy,
            transmission=transmission,
            metabolic_rate=metabolic_rate,
        )


__all__ = ["NCFAFluxPropagator", "NCFAFluxResult"]
