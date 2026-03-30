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
        self.scalar_proj = nn.Linear(1, hidden_dim)
        self.vector_proj = nn.Linear(4, hidden_dim)
        self.bivector_proj = nn.Linear(6, hidden_dim)
        self.trivector_proj = nn.Linear(4, hidden_dim)
        self.pseudo_proj = nn.Linear(1, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def _pad_to_pga(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 16:
            return x
        out = x.new_zeros(x.shape[:-1] + (16,))
        take = min(int(x.size(-1)), 16)
        out[..., :take] = x[..., :take]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_pga(torch.as_tensor(x, dtype=torch.float32, device=x.device))
        scalar = self.scalar_proj(x[..., 0:1])
        vector = self.vector_proj(x[..., 1:5])
        bivector = self.bivector_proj(x[..., 5:11])
        trivector = self.trivector_proj(x[..., 11:15])
        pseudo = self.pseudo_proj(x[..., 15:16])
        return self.fuse(torch.cat([scalar, vector, bivector, trivector, pseudo], dim=-1))


class PGWCrossAttention(nn.Module):
    """
    Decoupled wave-native cross attention.

    Structural geometry and thermodynamic wave channels are scored separately:
    - structural branch uses scaled dot-product attention
    - wave branch uses negative squared Euclidean distance
    A learned blend parameter then mixes the two logits before softmax.
    """

    def __init__(
        self,
        spatial_dim: int = 8,
        wave_dim: int = 3,
        heads: int = 4,
        hidden_dim: int = 64,
        wave_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_dim = int(max(spatial_dim, 1))
        self.wave_dim = int(max(wave_dim, 1))
        self.heads = int(max(heads, 1))
        self.hidden_dim = int(max(hidden_dim, self.heads))
        if self.hidden_dim % self.heads != 0:
            self.hidden_dim = self.heads * math.ceil(self.hidden_dim / self.heads)
        self.head_dim = self.hidden_dim // self.heads
        self.wave_temperature = float(max(wave_temperature, 1.0e-4))

        self.q_spatial = nn.Linear(self.spatial_dim, self.hidden_dim)
        self.k_spatial = nn.Linear(self.spatial_dim, self.hidden_dim)
        self.q_wave = nn.Linear(self.wave_dim, self.hidden_dim)
        self.k_wave = nn.Linear(self.wave_dim, self.hidden_dim)
        self.v_proj = nn.LazyLinear(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.wave_blend = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def _split_streams(self, mv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(mv, dtype=torch.float32)
        feature_dim = int(x.size(-1))
        spatial_end = min(self.spatial_dim, feature_dim)
        wave_end = min(self.spatial_dim + self.wave_dim, feature_dim)

        spatial = x[..., :spatial_end]
        if spatial_end < self.spatial_dim:
            spatial = F.pad(spatial, (0, self.spatial_dim - spatial_end))

        wave = x[..., self.spatial_dim:wave_end]
        if wave_end <= self.spatial_dim:
            wave = x.new_zeros(*x.shape[:-1], self.wave_dim)
        elif int(wave.size(-1)) < self.wave_dim:
            wave = F.pad(wave, (0, self.wave_dim - int(wave.size(-1))))
        return spatial, wave

    def forward(
        self,
        q_fp: torch.Tensor,
        v_ret: torch.Tensor,
        pi_star: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(q_fp.size(0))
        q_len = int(q_fp.size(1))
        k_len = int(v_ret.size(1))

        q_spatial, q_wave = self._split_streams(q_fp)
        k_spatial, k_wave = self._split_streams(v_ret)

        q_struct = self.q_spatial(q_spatial).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k_struct = self.k_spatial(k_spatial).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        spatial_logits = torch.matmul(q_struct, k_struct.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        q_wave_proj = self.q_wave(q_wave).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k_wave_proj = self.k_wave(k_wave).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        wave_diff = q_wave_proj.unsqueeze(3) - k_wave_proj.unsqueeze(2)
        wave_dist = wave_diff.square().sum(dim=-1)
        wave_logits = -(wave_dist / self.wave_temperature)

        alpha = torch.sigmoid(self.wave_blend).to(device=q_fp.device, dtype=q_fp.dtype)
        combined_logits = ((1.0 - alpha) * spatial_logits) + (alpha * wave_logits)
        transport_mask = torch.log(pi_star.clamp_min(1.0e-9)).unsqueeze(1)
        attention_weights = F.softmax(combined_logits + transport_mask, dim=-1)

        values = self.v_proj(v_ret).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        context = torch.matmul(attention_weights, values)
        context = context.transpose(1, 2).contiguous().view(batch, q_len, self.hidden_dim)
        return self.out_proj(context)


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
        self.fp_orient_proj = nn.LazyLinear(3)
        self.ana_orient_proj = nn.LazyLinear(3)

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
        fp_orient = F.normalize(self.fp_orient_proj(q_fp), p=2, dim=-1)
        ana_orient = F.normalize(self.ana_orient_proj(q_ana), p=2, dim=-1)
        orient_cross = torch.linalg.norm(torch.cross(fp_orient, ana_orient, dim=-1), dim=-1, keepdim=True)
        orient_dot = (fp_orient * ana_orient).sum(dim=-1, keepdim=True)

        fp_hidden = self.fp_context_proj(torch.cat([fp_struct, fp_elec, fp_struct * fp_elec, fp_orient, orient_dot], dim=-1))

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
                    ana_orient,
                    fp_orient,
                    orient_dot,
                    orient_cross,
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
