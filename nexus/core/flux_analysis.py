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


class NCFAFluxPropagator(nn.Module):
    def __init__(self, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = float(eps)

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

    def normalize_flux_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        row_sums = adjacency.sum(dim=-1, keepdim=True)
        scale = torch.maximum(row_sums, torch.ones_like(row_sums))
        return adjacency / scale.clamp_min(self.eps)

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

    def compute_flux_consistency_loss(self, W_struct: torch.Tensor) -> torch.Tensor:
        outgoing_flux = torch.sum(W_struct, dim=-1)
        consistency_error = torch.relu(outgoing_flux - 1.0)
        return torch.mean(consistency_error)

    def _propagate_flux_matrix(
        self,
        W: torch.Tensor,
        initial_dose: torch.Tensor,
        *,
        generations: int = 4,
    ) -> NCFAFluxResult:
        if W.ndim != 3:
            raise ValueError("W must have shape [batch, n_nodes, n_nodes]")
        if initial_dose.ndim == 1:
            initial_dose = initial_dose.unsqueeze(0).expand(W.size(0), -1)
        elif initial_dose.ndim != 2:
            raise ValueError("initial_dose must have shape [batch, n_nodes] or [n_nodes]")
        normalized = self.normalize_flux_adjacency(W)
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
        )

    def propagate_flux(
        self,
        tree_or_W: RecursiveMetabolismTree | torch.Tensor,
        initial_dose: torch.Tensor | float,
        *,
        generations: int = 4,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> NCFAFluxResult:
        if torch.is_tensor(tree_or_W):
            W = self.apply_inhibition(tree_or_W, inhibitor_mask=edge_mask)
            return self._propagate_flux_matrix(
                W,
                torch.as_tensor(initial_dose, dtype=W.dtype, device=W.device),
                generations=generations,
            )

        tree = tree_or_W
        adjacency = self.build_structural_adjacency(tree)
        adjacency, used_mask = self.apply_intervention(adjacency, edge_mask=edge_mask)
        normalized = self.normalize_flux_adjacency(adjacency)

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
        )

    def forward(
        self,
        tree_or_W: RecursiveMetabolismTree | torch.Tensor,
        initial_dose: torch.Tensor | float,
        *,
        generations: int = 4,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> NCFAFluxResult:
        return self.propagate_flux(tree_or_W, initial_dose, generations=generations, edge_mask=edge_mask)


__all__ = ["NCFAFluxPropagator", "NCFAFluxResult"]
