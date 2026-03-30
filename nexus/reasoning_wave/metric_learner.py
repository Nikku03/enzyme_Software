"""
Wave-engine metric utilities.

This fork keeps the classic retrieval math available while adding a local
quantum distillation head for the wave/equivariant path. The head is trained
against offline xTB targets so the live continuous atom embeddings learn
charge/Fukui/gap structure before analogical routing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.reasoning.metric_learner import (
    HGNNProjection,
    MechanismEncoder,
    PoincareMath,
    _som_class,
    encoder_supervision_loss,
    hyperbolic_supervision_loss,
    mechanism_contrastive_loss,
)


class WaveQuantumDistillationHead(nn.Module):
    """
    Lightweight multi-task decoder over per-atom continuous multivectors.

    Heads:
      - partial charge (per atom)
      - Fukui f(0) radical susceptibility (per atom)
      - HOMO/LUMO gap (pooled global scalar)
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.atom_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=float(max(dropout, 0.0))),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.charge_head = nn.Linear(hidden_dim, 1)
        self.fukui_head = nn.Linear(hidden_dim, 1)
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        atom_multivectors: torch.Tensor,
        *,
        atom_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        x = torch.as_tensor(atom_multivectors, dtype=torch.float32)
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.ndim != 3:
            raise ValueError("WaveQuantumDistillationHead expects [N, D] or [B, N, D] input")

        if atom_mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.float32, device=x.device)
        else:
            mask = torch.as_tensor(atom_mask, dtype=torch.float32, device=x.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != x.shape[:2]:
                raise ValueError("atom_mask must match the first two dimensions of atom_multivectors")

        h = self.atom_proj(x)
        charge = self.charge_head(h).squeeze(-1)
        fukui = self.fukui_head(h).squeeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=-2) / denom
        gap = self.gap_head(pooled).squeeze(-1)

        if squeeze_batch:
            return {
                "predicted_charges": charge.squeeze(0),
                "predicted_fukui": fukui.squeeze(0),
                "predicted_gap": gap.squeeze(0),
            }
        return {
            "predicted_charges": charge,
            "predicted_fukui": fukui,
            "predicted_gap": gap,
        }


def quantum_distillation_loss(
    *,
    predicted_charges: torch.Tensor,
    predicted_fukui: torch.Tensor,
    predicted_gap: torch.Tensor,
    target_charges: torch.Tensor,
    target_fukui: torch.Tensor,
    target_gap: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    charge_weight: float = 0.35,
    fukui_weight: float = 4.0,
    gap_weight: float = 0.02,
    huber_beta: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Robust multi-task quantum distillation loss.

    Huber is used instead of pure MSE so occasional noisy xTB values do not
    dominate early training.
    """

    pred_charge = torch.as_tensor(predicted_charges, dtype=torch.float32)
    pred_fukui = torch.as_tensor(predicted_fukui, dtype=torch.float32, device=pred_charge.device)
    pred_gap = torch.as_tensor(predicted_gap, dtype=torch.float32, device=pred_charge.device)
    tgt_charge = torch.as_tensor(target_charges, dtype=torch.float32, device=pred_charge.device)
    tgt_fukui = torch.as_tensor(target_fukui, dtype=torch.float32, device=pred_charge.device)
    tgt_gap = torch.as_tensor(target_gap, dtype=torch.float32, device=pred_charge.device).view(-1)

    if pred_charge.ndim == 1:
        pred_charge = pred_charge.unsqueeze(0)
    if pred_fukui.ndim == 1:
        pred_fukui = pred_fukui.unsqueeze(0)
    if tgt_charge.ndim == 1:
        tgt_charge = tgt_charge.unsqueeze(0)
    if tgt_fukui.ndim == 1:
        tgt_fukui = tgt_fukui.unsqueeze(0)
    if pred_gap.ndim == 0:
        pred_gap = pred_gap.view(1)

    if atom_mask is None:
        mask = torch.ones_like(pred_charge, dtype=torch.float32, device=pred_charge.device)
    else:
        mask = torch.as_tensor(atom_mask, dtype=torch.float32, device=pred_charge.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != pred_charge.shape:
            raise ValueError("atom_mask must match per-atom prediction shape")

    norm = mask.sum().clamp_min(1.0)
    charge_raw = F.smooth_l1_loss(pred_charge, tgt_charge, reduction="none", beta=huber_beta)
    charge_loss = (charge_raw * mask).sum() / norm
    fukui_raw = F.smooth_l1_loss(pred_fukui, tgt_fukui, reduction="none", beta=huber_beta)
    fukui_raw_loss = (fukui_raw * mask).sum() / norm
    tgt_fukui_pos = (tgt_fukui * mask).clamp_min(0.0)
    tgt_fukui_mass = tgt_fukui_pos.sum(dim=-1, keepdim=True)
    if bool((tgt_fukui_mass > 0.0).any().item()):
        tgt_fukui_dist = tgt_fukui_pos / tgt_fukui_mass.clamp_min(1.0e-8)
        pred_fukui_logprob = F.log_softmax(pred_fukui, dim=-1)
        fukui_kl = F.kl_div(pred_fukui_logprob, tgt_fukui_dist, reduction="none").sum(dim=-1)
        valid_rows = (tgt_fukui_mass.view(-1) > 0.0).to(dtype=torch.float32)
        fukui_dist_loss = (fukui_kl * valid_rows).sum() / valid_rows.sum().clamp_min(1.0)
        fukui_loss = 0.25 * fukui_raw_loss + 0.75 * fukui_dist_loss
    else:
        fukui_loss = fukui_raw_loss
    gap_loss = F.smooth_l1_loss(pred_gap.view(-1), tgt_gap.view(-1), reduction="mean", beta=huber_beta)
    total = (
        float(charge_weight) * charge_loss
        + float(fukui_weight) * fukui_loss
        + float(gap_weight) * gap_loss
    )
    return total, {
        "charge_loss": charge_loss,
        "fukui_loss": fukui_loss,
        "gap_loss": gap_loss,
    }


__all__ = [
    "HGNNProjection",
    "MechanismEncoder",
    "PoincareMath",
    "WaveQuantumDistillationHead",
    "_som_class",
    "encoder_supervision_loss",
    "hyperbolic_supervision_loss",
    "mechanism_contrastive_loss",
    "quantum_distillation_loss",
]
