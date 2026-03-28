from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CYP3A4_MORPHISM_ALPHA = torch.tensor(
    [0.25, 0.40, 0.25, 0.75, 0.75],
    dtype=torch.float32,
)


class MorphismFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = float(max(gamma, 0.0))
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32).view(-1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = ((1.0 - pt).clamp_min(0.0) ** self.gamma) * bce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(device=targets.device, dtype=targets.dtype).view(1, 1, -1)
            alpha_factor = targets * alpha_t + (1.0 - targets) * (1.0 - alpha_t)
            loss = loss * alpha_factor
        return loss


class PGWCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = int(max(hidden_dim, 8))
        self.scale = math.sqrt(float(self.hidden_dim))
        self.gamma = nn.Parameter(torch.ones(1))
        self.q_proj = nn.LazyLinear(self.hidden_dim)
        self.k_proj = nn.LazyLinear(self.hidden_dim)
        self.v_proj = nn.LazyLinear(self.hidden_dim)

    def forward(
        self,
        q_fp: torch.Tensor,
        v_ret: torch.Tensor,
        pi_star: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(q_fp)
        k = self.k_proj(v_ret)
        v = self.v_proj(v_ret)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        pi_mask = self.gamma * torch.log(pi_star.clamp_min(1.0e-9))
        attn_weights = F.softmax(attn_logits + pi_mask, dim=-1)
        return torch.matmul(attn_weights, v)


class _NodeProjection(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NexusDualDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_morphism_classes: int = 5) -> None:
        super().__init__()
        self.fp_feature_proj = _NodeProjection(hidden_dim=hidden_dim)
        self.ana_feature_proj = _NodeProjection(hidden_dim=hidden_dim)
        self.fp_som_head = nn.Linear(hidden_dim, 1)
        self.fp_morph_head = nn.Linear(hidden_dim, num_morphism_classes)
        self.ana_som_head = nn.Linear(hidden_dim, 1)
        self.ana_morph_head = nn.Linear(hidden_dim, num_morphism_classes)

    def forward(
        self,
        q_fp: torch.Tensor,
        q_ana: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_fp = self.fp_feature_proj(q_fp)
        h_ana = self.ana_feature_proj(q_ana)
        y_hat_fp_som = self.fp_som_head(h_fp).squeeze(-1)
        y_hat_fp_morph = self.fp_morph_head(h_fp)
        y_hat_ana_som = self.ana_som_head(h_ana).squeeze(-1)
        y_hat_ana_morph = self.ana_morph_head(h_ana)
        return y_hat_fp_som, y_hat_fp_morph, y_hat_ana_som, y_hat_ana_morph


class HomoscedasticArbiterLoss(nn.Module):
    def __init__(
        self,
        *,
        morphism_alpha: torch.Tensor | None = None,
        morphism_gamma: float = 2.0,
        burn_in_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.log_var_fp = nn.Parameter(torch.zeros(1))
        self.log_var_ana = nn.Parameter(torch.zeros(1))
        self.burn_in_epochs = int(max(burn_in_epochs, 0))
        self.morphism_criterion = MorphismFocalLoss(
            alpha=DEFAULT_CYP3A4_MORPHISM_ALPHA if morphism_alpha is None else morphism_alpha,
            gamma=morphism_gamma,
        )

    def forward(
        self,
        y_hat_fp_som: torch.Tensor,
        y_hat_fp_morph: torch.Tensor,
        y_hat_ana_som: torch.Tensor,
        y_hat_ana_morph: torch.Tensor,
        som_target: torch.Tensor,
        morph_target: torch.Tensor,
        morph_mask: torch.Tensor,
        *,
        label_confidence: torch.Tensor | float | None = None,
        has_morphism_label: torch.Tensor | bool | None = None,
        bridge_loss: torch.Tensor | None = None,
        bridge_weight: float = 0.5,
        current_epoch: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if y_hat_fp_som.ndim != 2 or y_hat_ana_som.ndim != 2:
            raise ValueError("SoM dual decoder logits must have shape [B, N]")
        if y_hat_fp_morph.ndim != 3 or y_hat_ana_morph.ndim != 3:
            raise ValueError("Morphism dual decoder logits must have shape [B, N, C]")
        device = y_hat_fp_som.device
        som_target = som_target.to(device=device, dtype=torch.float32)
        morph_target = morph_target.to(device=device, dtype=torch.float32)
        morph_mask = morph_mask.to(device=device, dtype=torch.float32)
        if label_confidence is None:
            confidence_t = torch.ones((), dtype=torch.float32, device=device)
        else:
            confidence_t = torch.as_tensor(label_confidence, dtype=torch.float32, device=device).view(())
        if has_morphism_label is None:
            has_morph_t = torch.ones((), dtype=torch.float32, device=device)
        else:
            has_morph_t = torch.as_tensor(has_morphism_label, dtype=torch.float32, device=device).view(())

        loss_fp_som = F.binary_cross_entropy_with_logits(y_hat_fp_som, som_target)
        loss_ana_som = F.binary_cross_entropy_with_logits(y_hat_ana_som, som_target)

        raw_fp_morph = self.morphism_criterion(
            y_hat_fp_morph,
            morph_target,
        )
        raw_ana_morph = self.morphism_criterion(
            y_hat_ana_morph,
            morph_target,
        )
        masked_fp_morph = raw_fp_morph * morph_mask
        masked_ana_morph = raw_ana_morph * morph_mask
        valid_morph = morph_mask.sum().clamp_min(1.0)
        loss_fp_morph = masked_fp_morph.sum() / valid_morph
        loss_ana_morph = masked_ana_morph.sum() / valid_morph
        morph_scale = confidence_t * has_morph_t

        bridge_term = (
            torch.as_tensor(bridge_loss, dtype=torch.float32, device=device).view(())
            if bridge_loss is not None
            else torch.zeros((), dtype=torch.float32, device=device)
        )
        bridge_term = bridge_term * float(max(bridge_weight, 0.0))

        in_burn_in = int(current_epoch or 0) < self.burn_in_epochs
        if in_burn_in:
            total_loss = (
                loss_fp_som
                + loss_ana_som
                + (loss_fp_morph * morph_scale)
                + (loss_ana_morph * morph_scale)
                + bridge_term
            )
            precision_fp = torch.ones((), dtype=torch.float32, device=device)
            precision_ana = torch.ones((), dtype=torch.float32, device=device)
            warmup_progress = float(int(current_epoch or 0) + 1) / float(max(self.burn_in_epochs, 1))
            warmup_cap = min(max(warmup_progress, 0.0), 1.0) * 0.15
            warmup_signal = confidence_t.clamp(0.0, 1.0) * has_morph_t.clamp(0.0, 1.0)
            weight_ana = torch.as_tensor(warmup_cap, dtype=torch.float32, device=device) * warmup_signal
            weight_fp = 1.0 - weight_ana
        else:
            precision_fp = torch.exp(-self.log_var_fp)
            precision_ana = torch.exp(-self.log_var_ana)

            loss_fp_scaled = 0.5 * precision_fp * (loss_fp_morph * morph_scale) + 0.5 * self.log_var_fp
            loss_ana_scaled = 0.5 * precision_ana * ((loss_ana_morph * morph_scale) + bridge_term) + 0.5 * self.log_var_ana
            total_loss = loss_fp_som + loss_ana_som + loss_fp_scaled + loss_ana_scaled
            weight_ana = precision_ana / (precision_fp + precision_ana).clamp_min(1.0e-9)
            weight_fp = 1.0 - weight_ana

        return total_loss.squeeze(), {
            "loss_fp_som": loss_fp_som.detach(),
            "loss_ana_som": loss_ana_som.detach(),
            "loss_fp_morph": loss_fp_morph.detach(),
            "loss_ana_morph": loss_ana_morph.detach(),
            "weight_fp": weight_fp.detach(),
            "weight_ana": weight_ana.detach(),
            "sigma_fp": torch.exp(0.5 * self.log_var_fp).detach(),
            "sigma_ana": torch.exp(0.5 * self.log_var_ana).detach(),
            "has_morphism_label": has_morph_t.detach(),
            "label_confidence": confidence_t.detach(),
            "bridge_loss": bridge_term.detach(),
            "burn_in_active": torch.as_tensor(1.0 if in_burn_in else 0.0, dtype=torch.float32, device=device),
        }


__all__ = [
    "HomoscedasticArbiterLoss",
    "NexusDualDecoder",
    "PGWCrossAttention",
]
