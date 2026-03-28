from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _NodeDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        inner = max(hidden_dim // 2, 8)
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NexusDualDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.decoder_fp = _NodeDecoder(hidden_dim=hidden_dim)
        self.decoder_ana = _NodeDecoder(hidden_dim=hidden_dim)

    def forward(self, q_fp: torch.Tensor, q_ana: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder_fp(q_fp), self.decoder_ana(q_ana)


class HomoscedasticArbiterLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_var_fp = nn.Parameter(torch.zeros(1))
        self.log_var_ana = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        y_hat_fp: torch.Tensor,
        y_hat_ana: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if y_hat_fp.ndim != 2 or y_hat_ana.ndim != 2:
            raise ValueError("Dual decoder logits must have shape [B, N]")
        target = target_idx.view(-1).to(dtype=torch.long, device=y_hat_fp.device)
        loss_fp_raw = F.cross_entropy(y_hat_fp, target)
        loss_ana_raw = F.cross_entropy(y_hat_ana, target)

        precision_fp = torch.exp(-self.log_var_fp)
        precision_ana = torch.exp(-self.log_var_ana)

        loss_fp_scaled = 0.5 * precision_fp * loss_fp_raw + 0.5 * self.log_var_fp
        loss_ana_scaled = 0.5 * precision_ana * loss_ana_raw + 0.5 * self.log_var_ana
        total_loss = loss_fp_scaled + loss_ana_scaled

        weight_ana = precision_ana / (precision_fp + precision_ana).clamp_min(1.0e-9)
        weight_fp = 1.0 - weight_ana

        return total_loss.squeeze(), {
            "loss_fp_raw": loss_fp_raw.detach(),
            "loss_ana_raw": loss_ana_raw.detach(),
            "weight_fp": weight_fp.detach(),
            "weight_ana": weight_ana.detach(),
            "sigma_fp": torch.exp(0.5 * self.log_var_fp).detach(),
            "sigma_ana": torch.exp(0.5 * self.log_var_ana).detach(),
        }


__all__ = [
    "HomoscedasticArbiterLoss",
    "NexusDualDecoder",
    "PGWCrossAttention",
]
