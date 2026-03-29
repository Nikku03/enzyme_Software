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

    @staticmethod
    def _wave_feature_signature(multivectors: torch.Tensor) -> torch.Tensor:
        mv = torch.as_tensor(multivectors, dtype=torch.float64)
        if mv.ndim != 2:
            raise ValueError("wave transport expects [N, D] multivectors")
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
        q_sig = self._wave_feature_signature(q_mv)
        r_sig = self._wave_feature_signature(r_mv)
        elec = torch.cdist(q_sig, r_sig, p=2).to(dtype=torch.float64)
        scale = elec.max().clamp_min(1.0)
        return elec / scale

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


__all__ = ["PGWTransportResult", "PGWTransporter", "WaveTransportDiagnostics"]
