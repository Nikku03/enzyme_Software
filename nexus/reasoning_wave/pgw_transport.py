"""
Wave-engine transport.

This fork keeps the PGW contract intact while adding an explicit electronic
distance term derived from the live node multivectors. The goal is to keep
spatial compatibility from the classic transporter, but make wave-supervised
features materially influence the atom-to-atom coupling cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from nexus.pocket.pga import PGA_DIM
from nexus.reasoning.pgw_transport import (
    PGWTransportResult,
    PGWTransporter as ClassicPGWTransporter,
)


@dataclass
class WaveTransportDiagnostics:
    structural_cost_mean: float
    electronic_cost_mean: float


class PGWTransporter(ClassicPGWTransporter):
    def __init__(
        self,
        *,
        structural_cost_weight: float = 0.35,
        electronic_cost_weight: float = 0.65,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.structural_cost_weight = float(max(structural_cost_weight, 0.0))
        self.electronic_cost_weight = float(max(electronic_cost_weight, 0.0))
        self.metric_structural_weight = 0.55
        self.metric_electronic_weight = 0.45

    @staticmethod
    def _canonicalize_multivectors(multivectors: torch.Tensor) -> torch.Tensor:
        mv = torch.as_tensor(multivectors)
        if mv.ndim != 2:
            raise ValueError("wave transport expects [N, D] multivectors")
        if mv.size(-1) == PGA_DIM:
            return mv
        if mv.size(-1) != 8:
            raise ValueError(
                f"wave transport expects trailing multivector dimension 8 or {PGA_DIM}, got {mv.size(-1)}"
            )
        out = torch.zeros(mv.shape[:-1] + (PGA_DIM,), dtype=mv.dtype, device=mv.device)
        out[..., 0] = mv[..., 0]
        out[..., 1:4] = mv[..., 1:4]
        out[..., 5] = mv[..., 4]
        out[..., 6] = mv[..., 6]
        out[..., 7] = mv[..., 5]
        out[..., 11] = mv[..., 7]
        return out

    def _wave_feature_signature(self, multivectors: torch.Tensor) -> torch.Tensor:
        mv = self._canonicalize_multivectors(multivectors).to(dtype=torch.float64)
        mv_centered = mv - mv.mean(dim=0, keepdim=True)
        mv_norm = F.normalize(mv_centered, p=2, dim=-1)
        magnitude = mv.norm(p=2, dim=-1, keepdim=True)
        signed_scalar = mv[:, : min(4, mv.size(-1))]
        abs_signature = torch.abs(mv[:, : min(4, mv.size(-1))])
        return torch.cat([mv_norm, magnitude, signed_scalar, abs_signature], dim=-1)

    def _electronic_cost(
        self,
        query_multivectors: Optional[torch.Tensor],
        retrieved_multivectors: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        q_mv = self._prepare_multivectors(query_multivectors)
        r_mv = self._prepare_multivectors(retrieved_multivectors)
        if q_mv is None or r_mv is None:
            return None
        if q_mv.size(0) == 0 or r_mv.size(0) == 0:
            return None
        q_mv = self._canonicalize_multivectors(q_mv)
        r_mv = self._canonicalize_multivectors(r_mv)
        q_sig = self._wave_feature_signature(q_mv)
        r_sig = self._wave_feature_signature(r_mv)
        elec = torch.cdist(q_sig, r_sig, p=2).to(dtype=torch.float64)
        scale = elec.max().clamp_min(1.0)
        return elec / scale

    def _combined_metric_tensor(self, mol, multivectors: torch.Tensor) -> torch.Tensor:
        structural_metric = self._distance_matrix(mol).to(dtype=torch.float64, device=self.device)
        multivectors = self._canonicalize_multivectors(multivectors)
        electronic_metric = self.compute_z_kernel_matrix(multivectors).to(dtype=torch.float64, device=self.device)
        total_weight = self.metric_structural_weight + self.metric_electronic_weight
        if total_weight <= 0.0:
            return electronic_metric
        combined = (
            self.metric_structural_weight * structural_metric
            + self.metric_electronic_weight * electronic_metric
        ) / total_weight
        scale = combined.max().clamp_min(1.0e-8)
        return combined / scale

    def _anchor_metric_cost(
        self,
        query_mol,
        retrieved_mol,
        q_metric: torch.Tensor,
        r_metric: torch.Tensor,
        cross_cost: torch.Tensor,
    ) -> torch.Tensor:
        q_anchor = self._anchor_indices(query_mol, device=q_metric.device)
        r_anchor = self._anchor_indices(retrieved_mol, device=r_metric.device)
        k = min(int(q_anchor.numel()), int(r_anchor.numel()))
        q_sig = q_metric.index_select(1, q_anchor[:k])
        r_sig = r_metric.index_select(1, r_anchor[:k])
        metric_cost = torch.cdist(q_sig, r_sig, p=2).to(dtype=torch.float64)
        return 0.5 * metric_cost + 0.5 * cross_cost

    def _cross_feature_cost(
        self,
        query_mol,
        retrieved_mol,
        *,
        query_multivectors: Optional[torch.Tensor] = None,
        retrieved_multivectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        structural = super()._cross_feature_cost(
            query_mol,
            retrieved_mol,
            query_multivectors=query_multivectors,
            retrieved_multivectors=retrieved_multivectors,
        )
        electronic = self._electronic_cost(query_multivectors, retrieved_multivectors)
        if electronic is None:
            return structural
        total_weight = self.structural_cost_weight + self.electronic_cost_weight
        if total_weight <= 0.0:
            return electronic
        blended = (
            self.structural_cost_weight * structural
            + self.electronic_cost_weight * electronic
        ) / total_weight
        return blended.to(dtype=torch.float64)

    def _exact_multivector_coupling(
        self,
        query_mol,
        retrieved_mol,
        q_mv: torch.Tensor,
        r_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        q_mv = self._canonicalize_multivectors(q_mv)
        r_mv = self._canonicalize_multivectors(r_mv)
        q_metric = self._combined_metric_tensor(query_mol, q_mv)
        r_metric = self._combined_metric_tensor(retrieved_mol, r_mv)
        cross_cost = self._cross_feature_cost(
            query_mol,
            retrieved_mol,
            query_multivectors=q_mv,
            retrieved_multivectors=r_mv,
        )
        transported_mass = 1.0 - self.dustbin_epsilon
        if max(q_mv.size(0), r_mv.size(0)) > self.linearize_above_atoms:
            anchor_cost = self._anchor_metric_cost(
                query_mol,
                retrieved_mol,
                q_metric,
                r_metric,
                cross_cost,
            )
            return self._partial_sinkhorn(anchor_cost, transported_mass=transported_mass), "wave_gw_linearized"
        return self._partial_z_gw(q_metric, r_metric, cross_cost=cross_cost), "wave_gw_exact"


__all__ = ["PGWTransportResult", "PGWTransporter", "WaveTransportDiagnostics"]
