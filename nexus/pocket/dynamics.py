from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .attention import ReversedGeometricAttention, ReversedGeometricAttentionOutput
from .encoder import PocketEncoderOutput, SEGNNPocketEncoder
from .hypernetwork import IsoformHyperOutput


@dataclass
class PocketDistanceGeometryOutput:
    distance_matrix: torch.Tensor
    coordinates: torch.Tensor
    alignment_error: torch.Tensor
    overlap_penalty: torch.Tensor


@dataclass
class DynamicPocketState:
    residue_positions: torch.Tensor
    residue_types: torch.Tensor
    encoded_pocket: PocketEncoderOutput
    distance_geometry: PocketDistanceGeometryOutput
    diffusion_score: torch.Tensor
    attention: ReversedGeometricAttentionOutput
    temporal_memory: torch.Tensor
    cryptic_gate: torch.Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        half = max(self.dim // 2, 1)
        device = t.device
        dtype = t.dtype
        scale = torch.linspace(0.0, 1.0, half, device=device, dtype=dtype)
        freq = torch.exp(-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) * scale)
        angles = t * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = torch.cat([emb, torch.zeros(emb.size(0), self.dim - emb.size(-1), device=device, dtype=dtype)], dim=-1)
        return emb


class PocketDiffusionSampler(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.noise_net = nn.Sequential(
            nn.Linear(3 + hidden_dim + hidden_dim + time_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.cryptic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        residue_positions: torch.Tensor,
        pocket: PocketEncoderOutput,
        *,
        drug_context: Optional[torch.Tensor] = None,
        isoform: Optional[IsoformHyperOutput] = None,
        t: Optional[torch.Tensor] = None,
        noise_scale: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_res = residue_positions.size(0)
        device = residue_positions.device
        dtype = residue_positions.dtype
        if t is None:
            t = torch.zeros((), device=device, dtype=dtype)
        t_embed = self.time_embed(t.to(device=device, dtype=dtype).view(1)).expand(n_res, -1)

        if drug_context is None:
            drug_context = residue_positions.new_zeros(pocket.scalar_features.size(-1))
        drug_context = drug_context.to(device=device, dtype=dtype).reshape(1, -1).expand(n_res, -1)
        if isoform is None:
            iso_pocket = residue_positions.new_zeros(pocket.scalar_features.size(-1))
        else:
            iso_pocket = isoform.pocket_embedding.to(device=device, dtype=dtype).reshape(1, -1).expand(n_res, -1)

        local_noise = torch.randn_like(residue_positions) * float(noise_scale)
        motion_gate = pocket.vector_features.norm(dim=-1, keepdim=True)
        net_in = torch.cat(
            [
                residue_positions + local_noise,
                pocket.scalar_features.to(device=device, dtype=dtype),
                drug_context,
                t_embed,
                motion_gate,
            ],
            dim=-1,
        )
        diffusion_score = self.noise_net(net_in)
        updated_positions = residue_positions + 0.1 * diffusion_score + 0.05 * pocket.vector_features.to(device=device, dtype=dtype)
        cryptic_gate = torch.sigmoid(
            self.cryptic_head(
                torch.cat(
                    [
                        pocket.scalar_features.to(device=device, dtype=dtype),
                        iso_pocket,
                        t_embed,
                    ],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        return updated_positions, diffusion_score, cryptic_gate


class NeuralDistanceGeometry(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        overlap_radius: float = 1.2,
    ) -> None:
        super().__init__()
        self.overlap_radius = float(overlap_radius)
        self.distance_refine = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _classical_mds(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        n = distance_matrix.size(0)
        dtype = distance_matrix.dtype
        device = distance_matrix.device
        eye = torch.eye(n, dtype=dtype, device=device)
        ones = torch.ones((n, n), dtype=dtype, device=device) / max(n, 1)
        centering = eye - ones
        gram = -0.5 * centering @ distance_matrix.pow(2) @ centering
        eigvals, eigvecs = torch.linalg.eigh(gram)
        top_vals = eigvals[-3:].clamp_min(0.0)
        top_vecs = eigvecs[:, -3:]
        return top_vecs * torch.sqrt(top_vals).unsqueeze(0)

    def _align_to_reference(self, coords: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref_center = reference.mean(dim=0, keepdim=True)
        coords_center = coords.mean(dim=0, keepdim=True)
        ref0 = reference - ref_center
        coords0 = coords - coords_center
        cov = coords0.transpose(0, 1) @ ref0
        u, _, vh = torch.linalg.svd(cov, full_matrices=False)
        rot = vh.transpose(0, 1) @ u.transpose(0, 1)
        if torch.linalg.det(rot) < 0:
            vh = vh.clone()
            vh[-1] = -vh[-1]
            rot = vh.transpose(0, 1) @ u.transpose(0, 1)
        aligned = coords0 @ rot + ref_center
        error = (aligned - reference).pow(2).sum(dim=-1).mean().sqrt()
        return aligned, error

    def forward(
        self,
        residue_positions: torch.Tensor,
        ligand_points: torch.Tensor,
        *,
        cryptic_gate: Optional[torch.Tensor] = None,
    ) -> PocketDistanceGeometryOutput:
        device = residue_positions.device
        dtype = residue_positions.dtype
        ligand_points = ligand_points.to(device=device, dtype=dtype)
        rr = torch.cdist(residue_positions.unsqueeze(0), residue_positions.unsqueeze(0)).squeeze(0)
        local_gate = torch.ones(residue_positions.size(0), 1, device=device, dtype=dtype)
        if cryptic_gate is not None:
            local_gate = cryptic_gate.to(device=device, dtype=dtype).reshape(-1, 1)
        refine_in = torch.cat(
            [
                rr.unsqueeze(-1),
                (rr <= 12.0).to(dtype).unsqueeze(-1),
                local_gate.unsqueeze(1).expand(-1, rr.size(1), -1),
                local_gate.unsqueeze(0).expand(rr.size(0), -1, -1),
            ],
            dim=-1,
        )
        rr_refined = (rr + 0.1 * self.distance_refine(refine_in).squeeze(-1)).clamp_min(1.0e-4)
        coords = self._classical_mds(rr_refined)
        coords_aligned, alignment_error = self._align_to_reference(coords, residue_positions)
        pair_dist = torch.cdist(coords_aligned.unsqueeze(0), coords_aligned.unsqueeze(0)).squeeze(0)
        eye = torch.eye(pair_dist.size(0), dtype=torch.bool, device=device)
        overlap_penalty = torch.relu(self.overlap_radius - pair_dist.masked_fill(eye, self.overlap_radius)).mean()
        return PocketDistanceGeometryOutput(
            distance_matrix=rr_refined,
            coordinates=coords_aligned,
            alignment_error=alignment_error,
            overlap_penalty=overlap_penalty,
        )


class TimeVaryingReversedAttention(nn.Module):
    def __init__(
        self,
        drug_dim: int = 8,
        pocket_dim: int = 16,
        heads: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.base_attention = ReversedGeometricAttention(
            drug_dim=drug_dim,
            pocket_dim=pocket_dim,
            heads=heads,
            hidden_dim=hidden_dim,
        )
        self.context_proj = nn.Linear(pocket_dim, hidden_dim)
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        drug_multivectors: torch.Tensor,
        pocket: PocketEncoderOutput,
        *,
        previous_memory: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> tuple[ReversedGeometricAttentionOutput, torch.Tensor]:
        base = self.base_attention(
            drug_multivectors,
            pocket,
            residue_mask=residue_mask,
        )
        attended = base.attended_drug
        attended_batch = attended.unsqueeze(0) if attended.ndim == 2 else attended
        context = self.context_proj(attended_batch)
        memory = context.mean(dim=1)
        if previous_memory is None:
            previous_memory = torch.zeros_like(memory)
        t_scalar = torch.zeros(memory.size(0), 1, dtype=memory.dtype, device=memory.device) if t is None else t.reshape(-1, 1).to(device=memory.device, dtype=memory.dtype)
        if t_scalar.size(0) == 1 and memory.size(0) > 1:
            t_scalar = t_scalar.expand(memory.size(0), -1)
        gate = self.memory_gate(torch.cat([memory, previous_memory, t_scalar], dim=-1))
        new_memory = gate * memory + (1.0 - gate) * previous_memory
        return base, new_memory


class DynamicPocketSimulator(nn.Module):
    def __init__(
        self,
        residue_vocab: int = 32,
        hidden_dim: int = 128,
        encoder_layers: int = 2,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = SEGNNPocketEncoder(
            residue_vocab=residue_vocab,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
        )
        self.diffusion = PocketDiffusionSampler(hidden_dim=hidden_dim)
        self.distance_geometry = NeuralDistanceGeometry(hidden_dim=hidden_dim)
        self.temporal_attention = TimeVaryingReversedAttention(
            drug_dim=8,
            pocket_dim=16,
            heads=attention_heads,
            hidden_dim=hidden_dim,
        )

    def step(
        self,
        residue_positions: torch.Tensor,
        residue_types: torch.Tensor,
        drug_multivectors: torch.Tensor,
        ligand_points: torch.Tensor,
        *,
        aromatic_normals: Optional[torch.Tensor] = None,
        cavity_vectors: Optional[torch.Tensor] = None,
        conservation_scores: Optional[torch.Tensor] = None,
        isoform: Optional[IsoformHyperOutput] = None,
        previous_memory: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> DynamicPocketState:
        encoded = self.encoder(
            residue_positions,
            residue_types,
            aromatic_normals=aromatic_normals,
            cavity_vectors=cavity_vectors,
            conservation_scores=conservation_scores,
        )
        drug_context = None
        if drug_multivectors.ndim == 2:
            drug_context = drug_multivectors.mean(dim=0)
            if drug_context.numel() < encoded.scalar_features.size(-1):
                drug_context = torch.cat(
                    [
                        drug_context,
                        residue_positions.new_zeros(encoded.scalar_features.size(-1) - drug_context.numel()),
                    ],
                    dim=0,
                )
            else:
                drug_context = drug_context[: encoded.scalar_features.size(-1)]
        updated_positions, diffusion_score, cryptic_gate = self.diffusion(
            residue_positions,
            encoded,
            drug_context=drug_context,
            isoform=isoform,
            t=t,
        )
        distance_geometry = self.distance_geometry(
            updated_positions,
            ligand_points,
            cryptic_gate=cryptic_gate,
        )
        dynamic_encoded = self.encoder(
            distance_geometry.coordinates,
            residue_types,
            aromatic_normals=aromatic_normals,
            cavity_vectors=cavity_vectors,
            conservation_scores=conservation_scores,
        )
        attention, temporal_memory = self.temporal_attention(
            drug_multivectors,
            dynamic_encoded,
            previous_memory=previous_memory,
            t=t,
        )
        return DynamicPocketState(
            residue_positions=distance_geometry.coordinates,
            residue_types=residue_types,
            encoded_pocket=dynamic_encoded,
            distance_geometry=distance_geometry,
            diffusion_score=diffusion_score,
            attention=attention,
            temporal_memory=temporal_memory,
            cryptic_gate=cryptic_gate,
        )


__all__ = [
    "PocketDistanceGeometryOutput",
    "DynamicPocketState",
    "PocketDiffusionSampler",
    "NeuralDistanceGeometry",
    "TimeVaryingReversedAttention",
    "DynamicPocketSimulator",
]
