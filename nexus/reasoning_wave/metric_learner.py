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
        self.harmonic_proj = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
        )
        self.charge_head = nn.Linear(hidden_dim, 1)
        self.fukui_head = nn.Linear(hidden_dim, 1)
        self.harmonic_head = nn.Linear(hidden_dim, 9)
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
        atom_coords: Optional[torch.Tensor] = None,
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
        predicted_harmonics: torch.Tensor | None = None
        if atom_coords is not None:
            sh = real_spherical_harmonics(atom_coords).to(device=x.device, dtype=x.dtype)
            if sh.ndim == 2:
                sh = sh.unsqueeze(0)
            if sh.shape[:2] == h.shape[:2]:
                h = h + self.harmonic_proj(sh)
                predicted_harmonics = self.harmonic_head(h)
        charge = self.charge_head(h).squeeze(-1)
        fukui = self.fukui_head(h).squeeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=-2) / denom
        gap = self.gap_head(pooled).squeeze(-1)

        if squeeze_batch:
            out = {
                "predicted_charges": charge.squeeze(0),
                "predicted_fukui": fukui.squeeze(0),
                "predicted_gap": gap.squeeze(0),
            }
            if predicted_harmonics is not None:
                out["predicted_harmonics"] = predicted_harmonics.squeeze(0)
            return out
        out = {
            "predicted_charges": charge,
            "predicted_fukui": fukui,
            "predicted_gap": gap,
        }
        if predicted_harmonics is not None:
            out["predicted_harmonics"] = predicted_harmonics
        return out


def real_spherical_harmonics(atom_coords: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    coords = torch.as_tensor(atom_coords, dtype=torch.float32)
    squeeze_batch = False
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
        squeeze_batch = True
    if coords.ndim != 3 or coords.size(-1) != 3:
        raise ValueError("real_spherical_harmonics expects [N,3] or [B,N,3] coordinates")
    centered = coords - coords.mean(dim=-2, keepdim=True)
    radius = centered.norm(dim=-1, keepdim=True).clamp_min(eps)
    xyz = centered / radius
    x = xyz[..., 0:1]
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]
    y00 = torch.ones_like(x)
    y1m1 = y
    y10 = z
    y11 = x
    y2m2 = x * y
    y2m1 = y * z
    y20 = 0.5 * (3.0 * z * z - 1.0)
    y21 = x * z
    y22 = 0.5 * (x * x - y * y)
    out = torch.cat([y00, y1m1, y10, y11, y2m2, y2m1, y20, y21, y22], dim=-1)
    return out.squeeze(0) if squeeze_batch else out


def quantum_distillation_loss(
    *,
    predicted_charges: torch.Tensor,
    predicted_fukui: torch.Tensor,
    predicted_gap: torch.Tensor,
    target_charges: torch.Tensor,
    target_fukui: torch.Tensor,
    target_gap: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    charge_weight: float = 0.50,
    fukui_weight: float = 12.0,
    gap_weight: float = 0.02,
    huber_beta: float = 0.25,
    fukui_sharpen_temperature: float = 3.0,
    predicted_harmonics: Optional[torch.Tensor] = None,
    target_harmonics: Optional[torch.Tensor] = None,
    harmonic_weight: float = 0.35,
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
    fukui_raw = F.smooth_l1_loss(pred_fukui, tgt_fukui, reduction="none", beta=0.1)
    fukui_raw_loss = (fukui_raw * mask).sum() / norm
    tgt_fukui_pos = (tgt_fukui * mask).clamp_min(0.0)
    valid_rows = (mask.sum(dim=-1) > 0.0) & (tgt_fukui_pos.sum(dim=-1) > 0.0)
    if bool(valid_rows.any().item()):
        safe_mask = mask > 0.0
        pred_fukui_logits = pred_fukui.masked_fill(~safe_mask, -1.0e9)
        pred_fukui_logprob = F.log_softmax(pred_fukui_logits, dim=-1)
        sharpen_temp = max(float(fukui_sharpen_temperature), 1.0)
        target_logits = (tgt_fukui_pos * sharpen_temp).masked_fill(~safe_mask, -1.0e9)
        target_dist = F.softmax(target_logits, dim=-1)
        fukui_kl = F.kl_div(pred_fukui_logprob, target_dist, reduction="none").sum(dim=-1)
        valid_rows_f = valid_rows.to(dtype=torch.float32, device=pred_charge.device)
        fukui_dist_loss = (fukui_kl * valid_rows_f).sum() / valid_rows_f.sum().clamp_min(1.0)
        fukui_loss = 0.05 * fukui_raw_loss + 0.95 * fukui_dist_loss
    else:
        fukui_dist_loss = pred_charge.new_zeros(())
        fukui_loss = fukui_raw_loss
    gap_loss = F.smooth_l1_loss(pred_gap.view(-1), tgt_gap.view(-1), reduction="mean", beta=huber_beta)
    harmonic_loss = pred_charge.new_zeros(())
    if predicted_harmonics is not None and target_harmonics is not None:
        pred_h = torch.as_tensor(predicted_harmonics, dtype=torch.float32, device=pred_charge.device)
        tgt_h = torch.as_tensor(target_harmonics, dtype=torch.float32, device=pred_charge.device)
        if pred_h.ndim == 2:
            pred_h = pred_h.unsqueeze(0)
        if tgt_h.ndim == 2:
            tgt_h = tgt_h.unsqueeze(0)
        if pred_h.shape[:2] == mask.shape and tgt_h.shape == pred_h.shape:
            h_mask = mask.unsqueeze(-1)
            harmonic_raw = F.smooth_l1_loss(pred_h, tgt_h, reduction="none", beta=huber_beta)
            harmonic_loss = (harmonic_raw * h_mask).sum() / h_mask.sum().clamp_min(1.0)
    total = (
        float(charge_weight) * charge_loss
        + float(fukui_weight) * fukui_loss
        + float(gap_weight) * gap_loss
        + float(harmonic_weight) * harmonic_loss
    )
    return total, {
        "charge_loss": charge_loss,
        "fukui_loss": fukui_loss,
        "fukui_raw_loss": fukui_raw_loss,
        "fukui_kl_loss": fukui_dist_loss,
        "gap_loss": gap_loss,
        "harmonic_loss": harmonic_loss,
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
    "real_spherical_harmonics",
]
