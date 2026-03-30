from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.reasoning.analogical_fusion import (
    DEFAULT_CYP3A4_MORPHISM_ALPHA,
    HomoscedasticArbiterLoss,
    MorphismFocalLoss,
)


class _WaveNodeProjection(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PGWCrossAttention(nn.Module):
    """
    Wave-aware cross attention.

    Structural compatibility and electronic compatibility are scored separately,
    then fused with the transport prior instead of collapsing everything into a
    single raw dot product.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = int(max(hidden_dim, 8))
        self.scale = math.sqrt(float(self.hidden_dim))
        self.transport_gamma = nn.Parameter(torch.ones(1))
        self.electronic_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.q_struct_proj = nn.LazyLinear(self.hidden_dim)
        self.k_struct_proj = nn.LazyLinear(self.hidden_dim)
        self.v_struct_proj = nn.LazyLinear(self.hidden_dim)

        self.q_elec_proj = nn.LazyLinear(self.hidden_dim)
        self.k_elec_proj = nn.LazyLinear(self.hidden_dim)
        self.v_elec_proj = nn.LazyLinear(self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        q_fp: torch.Tensor,
        v_ret: torch.Tensor,
        pi_star: torch.Tensor,
    ) -> torch.Tensor:
        q_struct = self.q_struct_proj(q_fp)
        k_struct = self.k_struct_proj(v_ret)
        v_struct = self.v_struct_proj(v_ret)

        q_elec = self.q_elec_proj(q_fp)
        k_elec = self.k_elec_proj(v_ret)
        v_elec = self.v_elec_proj(v_ret)

        structural_logits = torch.matmul(q_struct, k_struct.transpose(-2, -1)) / self.scale
        electronic_distance = torch.cdist(q_elec, k_elec, p=2)
        electronic_scale = electronic_distance.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-4)
        electronic_logits = -(electronic_distance / electronic_scale)

        pi_mask = self.transport_gamma * torch.log(pi_star.clamp_min(1.0e-9))
        final_logits = structural_logits + (self.electronic_weight * electronic_logits) + pi_mask
        attn_weights = F.softmax(final_logits, dim=-1)

        context_struct = torch.matmul(attn_weights, v_struct)
        context_elec = torch.matmul(attn_weights, v_elec)
        fused = torch.cat(
            [
                context_struct,
                context_elec,
                context_struct - context_elec,
                context_struct * context_elec,
            ],
            dim=-1,
        )
        return self.out_proj(fused)


class NexusDualDecoder(nn.Module):
    """
    Wave-aware dual decoder.

    Structural and electronic streams are encoded separately and mixed through a
    learned gate, instead of relying on a single flat concatenation.
    """

    def __init__(self, hidden_dim: int = 32, num_morphism_classes: int = 5) -> None:
        super().__init__()
        self.fp_struct_proj = _WaveNodeProjection(hidden_dim=hidden_dim)
        self.fp_elec_proj = _WaveNodeProjection(hidden_dim=hidden_dim)
        self.ana_struct_proj = _WaveNodeProjection(hidden_dim=hidden_dim)
        self.ana_elec_proj = _WaveNodeProjection(hidden_dim=hidden_dim)

        self.struct_gate_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.elec_gate_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fp_context_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
        )
        self.ana_context_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.fp_som_head = nn.Linear(hidden_dim, 1)
        self.fp_morph_head = nn.Linear(hidden_dim, num_morphism_classes)
        self.ana_som_head = nn.Linear(hidden_dim, 1)
        self.ana_morph_head = nn.Linear(hidden_dim, num_morphism_classes)

    def forward(
        self,
        q_fp: torch.Tensor,
        q_ana: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fp_struct = self.fp_struct_proj(q_fp)
        fp_elec = self.fp_elec_proj(q_fp)
        ana_struct = self.ana_struct_proj(q_ana)
        ana_elec = self.ana_elec_proj(q_ana)

        fp_hidden = self.fp_context_proj(torch.cat([fp_struct, fp_elec, fp_struct * fp_elec], dim=-1))

        struct_gate = torch.sigmoid(
            self.struct_gate_proj(torch.cat([ana_struct, fp_struct, ana_struct - fp_struct], dim=-1))
        )
        elec_gate = torch.sigmoid(
            self.elec_gate_proj(torch.cat([ana_elec, fp_elec, torch.abs(ana_elec - fp_elec)], dim=-1))
        )

        mixed_struct = ana_struct + struct_gate * fp_struct
        mixed_elec = ana_elec + elec_gate * fp_elec
        ana_hidden = self.ana_context_proj(
            torch.cat(
                [
                    mixed_struct,
                    mixed_elec,
                    mixed_struct - mixed_elec,
                    mixed_struct * mixed_elec,
                ],
                dim=-1,
            )
        )

        y_hat_fp_som = self.fp_som_head(fp_hidden).squeeze(-1)
        y_hat_fp_morph = self.fp_morph_head(fp_hidden)
        y_hat_ana_som = self.ana_som_head(ana_hidden).squeeze(-1)
        y_hat_ana_morph = self.ana_morph_head(ana_hidden)
        return y_hat_fp_som, y_hat_fp_morph, y_hat_ana_som, y_hat_ana_morph


__all__ = [
    "DEFAULT_CYP3A4_MORPHISM_ALPHA",
    "HomoscedasticArbiterLoss",
    "MorphismFocalLoss",
    "NexusDualDecoder",
    "PGWCrossAttention",
]
