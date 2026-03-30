from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nexus.field.siren_base import SIREN_OMEGA_0, SineLayer


_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO*"


@dataclass
class IsoformHyperOutput:
    pocket_embedding: torch.Tensor
    sequence_embedding: torch.Tensor
    variant_embedding: torch.Tensor
    structural_embedding: torch.Tensor
    allosteric_embedding: torch.Tensor
    delta_row_scale: torch.Tensor
    delta_bias_shift: torch.Tensor
    layer_row_scale: torch.Tensor
    layer_bias_shift: torch.Tensor
    output_scale: torch.Tensor
    accessibility_scale: torch.Tensor
    accessibility_bias: torch.Tensor
    sharpness_scale: torch.Tensor
    residue_radius_delta: torch.Tensor
    heme_access_shift: torch.Tensor
    catalytic_activity: torch.Tensor
    accessibility_activity: torch.Tensor


class IsoformSpecificPocketSIREN(nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        omega_0: float = SIREN_OMEGA_0,
    ) -> None:
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"IsoformSpecificPocketSIREN requires omega_0={SIREN_OMEGA_0}")
        self.layers = nn.ModuleList(
            [
                SineLayer(
                    self.coord_dim if idx == 0 else self.hidden_dim,
                    self.hidden_dim,
                    omega_0=omega_0,
                    is_first=idx == 0,
                )
                for idx in range(self.hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, coords: torch.Tensor, hyper: IsoformHyperOutput) -> torch.Tensor:
        x = coords
        for idx, layer in enumerate(self.layers):
            row_scale = hyper.layer_row_scale[idx]
            bias_shift = hyper.layer_bias_shift[idx]
            x = layer(x, row_scale=row_scale, bias_shift=bias_shift)
        return self.output_layer(x) * hyper.output_scale


class IsoformSpecificHyperNetwork(nn.Module):
    def __init__(
        self,
        sequence_dim: int = 256,
        structural_dim: int = 128,
        variant_dim: int = 64,
        allosteric_dim: int = 128,
        hidden_dim: int = 128,
        siren_hidden_dim: int = 128,
        siren_layers: int = 3,
        allele_vocab: int = 128,
    ) -> None:
        super().__init__()
        self.sequence_dim = int(sequence_dim)
        self.structural_dim = int(structural_dim)
        self.variant_dim = int(variant_dim)
        self.allosteric_dim = int(allosteric_dim)
        self.hidden_dim = int(hidden_dim)
        self.siren_hidden_dim = int(siren_hidden_dim)
        self.siren_layers = int(siren_layers)

        self.sequence_token_embedding = nn.Embedding(len(_AA_ALPHABET), 32)
        self.sequence_pool = nn.Sequential(
            nn.Linear(32, sequence_dim),
            nn.SiLU(),
            nn.Linear(sequence_dim, sequence_dim),
        )
        self.variant_token_embedding = nn.Embedding(allele_vocab, variant_dim)
        self.variant_pool = nn.Sequential(
            nn.Linear(variant_dim, variant_dim),
            nn.SiLU(),
            nn.Linear(variant_dim, variant_dim),
        )
        self.structural_encoder = nn.Sequential(
            nn.Linear(structural_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, structural_dim),
        )
        fused_dim = self.sequence_dim + self.structural_dim + self.variant_dim + self.allosteric_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.pocket_head = nn.Linear(hidden_dim, hidden_dim)
        self.layer_row_scale = nn.Linear(hidden_dim, self.siren_layers * self.siren_hidden_dim)
        self.layer_bias_shift = nn.Linear(hidden_dim, self.siren_layers * self.siren_hidden_dim)
        self.output_scale = nn.Linear(hidden_dim, 1)
        self.accessibility_head = nn.Linear(hidden_dim, 3)
        self.radius_head = nn.Linear(hidden_dim, 1)
        self.heme_shift = nn.Linear(hidden_dim, 1)
        self.register_buffer("global_row_scale", torch.ones(self.siren_layers, self.siren_hidden_dim))
        self.register_buffer("global_bias_shift", torch.zeros(self.siren_layers, self.siren_hidden_dim))
        self.register_buffer("global_output_scale", torch.ones(()))
        self.register_buffer("global_accessibility_scale", torch.ones(()))
        self.register_buffer("global_sharpness_scale", torch.ones(()))

    def _variant_effects(
        self,
        variant_ids: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if variant_ids is None:
            zeros = torch.zeros((), device=device, dtype=dtype)
            ones = torch.ones((), device=device, dtype=dtype)
            return {
                "loss_of_function": zeros,
                "gain_of_function": zeros,
                "catalytic_activity": ones,
                "accessibility_activity": ones,
                "heme_shift": zeros,
                "bias_shift": zeros,
            }
        variant_ids = variant_ids.to(device=device, dtype=torch.long).reshape(-1)
        loss_of_function = (variant_ids == 2).to(dtype=dtype).amax()
        gain_of_function = (variant_ids == 17).to(dtype=dtype).amax()
        catalytic_activity = (1.0 - 0.92 * loss_of_function + 0.20 * gain_of_function).clamp(0.02, 1.50)
        accessibility_activity = (1.0 - 0.80 * loss_of_function + 0.35 * gain_of_function).clamp(0.05, 1.75)
        heme_shift = (-3.0 * loss_of_function + 1.0 * gain_of_function).to(dtype=dtype)
        bias_shift = (-0.25 * loss_of_function + 0.10 * gain_of_function).to(dtype=dtype)
        return {
            "loss_of_function": loss_of_function,
            "gain_of_function": gain_of_function,
            "catalytic_activity": catalytic_activity,
            "accessibility_activity": accessibility_activity,
            "heme_shift": heme_shift,
            "bias_shift": bias_shift,
        }

    def predict_variant_potential(
        self,
        isoform_embedding: torch.Tensor,
        *,
        variant_ids: Optional[torch.Tensor] = None,
        variant_embedding: Optional[torch.Tensor] = None,
        reference_params: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        device = isoform_embedding.device
        dtype = isoform_embedding.dtype
        base = {
            "layer_row_scale": self.global_row_scale.to(device=device, dtype=dtype),
            "layer_bias_shift": self.global_bias_shift.to(device=device, dtype=dtype),
            "output_scale": self.global_output_scale.to(device=device, dtype=dtype),
            "accessibility_scale": self.global_accessibility_scale.to(device=device, dtype=dtype),
            "sharpness_scale": self.global_sharpness_scale.to(device=device, dtype=dtype),
        }
        if reference_params is not None:
            for key, value in reference_params.items():
                if key in base:
                    base[key] = value.to(device=device, dtype=dtype)
        variant = self._variant_effects(variant_ids, device=device, dtype=dtype)
        if variant_embedding is None:
            variant_vec = self.encode_variants(variant_ids, None, device=device, dtype=dtype)
        else:
            variant_vec = variant_embedding.to(device=device, dtype=dtype)
        structural = isoform_embedding.reshape(-1).to(device=device, dtype=dtype)
        structural = structural[: self.structural_dim]
        if structural.numel() < self.structural_dim:
            structural = torch.cat(
                [structural, torch.zeros(self.structural_dim - structural.numel(), device=device, dtype=dtype)],
                dim=0,
            )
        fused = self.fusion(
            torch.cat(
                [
                    torch.zeros(self.sequence_dim, device=device, dtype=dtype),
                    structural,
                    variant_vec,
                    torch.zeros(self.allosteric_dim, device=device, dtype=dtype),
                ],
                dim=0,
            )
        )
        row_delta = 0.25 * torch.tanh(
            self.layer_row_scale(fused).view(self.siren_layers, self.siren_hidden_dim)
        )
        bias_delta = 0.10 * torch.tanh(
            self.layer_bias_shift(fused).view(self.siren_layers, self.siren_hidden_dim)
        )
        return {
            "layer_row_scale": (base["layer_row_scale"] + row_delta) * variant["catalytic_activity"],
            "layer_bias_shift": base["layer_bias_shift"] + bias_delta + variant["bias_shift"],
            "output_scale": (
                base["output_scale"] + 0.25 * torch.sigmoid(self.output_scale(fused).squeeze(-1))
            ) * variant["catalytic_activity"],
            "accessibility_scale": (
                base["accessibility_scale"] + 0.25 * variant["accessibility_activity"]
            ),
            "sharpness_scale": base["sharpness_scale"] * (0.9 + 0.2 * variant["accessibility_activity"]),
            "heme_access_shift": 0.25 * torch.tanh(self.heme_shift(fused).squeeze(-1)) + variant["heme_shift"],
        }

    def encode_sequence(
        self,
        sequence: Optional[str] = None,
        sequence_embedding: Optional[torch.Tensor] = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if sequence_embedding is not None:
            return sequence_embedding.to(device=device, dtype=dtype)
        if sequence is None:
            return torch.zeros(self.sequence_dim, device=device, dtype=dtype)
        vocab = {aa: idx for idx, aa in enumerate(_AA_ALPHABET)}
        idx = [vocab.get(ch.upper(), vocab["X"]) for ch in sequence]
        tokens = torch.as_tensor(idx, dtype=torch.long, device=device)
        pooled = self.sequence_token_embedding(tokens).mean(dim=0)
        return self.sequence_pool(pooled).to(dtype=dtype)

    def encode_variants(
        self,
        variant_ids: Optional[torch.Tensor] = None,
        variant_embedding: Optional[torch.Tensor] = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if variant_embedding is not None:
            return variant_embedding.to(device=device, dtype=dtype)
        if variant_ids is None:
            return torch.zeros(self.variant_dim, device=device, dtype=dtype)
        variant_ids = variant_ids.to(device=device, dtype=torch.long)
        pooled = self.variant_token_embedding(variant_ids).mean(dim=0)
        return self.variant_pool(pooled).to(dtype=dtype)

    def derive_structural_embedding(
        self,
        residue_positions: torch.Tensor,
        pocket_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        centered = residue_positions - residue_positions.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / max(residue_positions.size(0), 1)
        eigvals = torch.linalg.eigvalsh(covariance)
        radial = centered.norm(dim=-1)
        stats = torch.cat(
            [
                residue_positions.mean(dim=0),
                residue_positions.std(dim=0),
                eigvals,
                radial.mean().view(1),
                radial.std().view(1),
            ],
            dim=0,
        )
        if pocket_context is None:
            pocket_context = residue_positions.new_zeros(self.structural_dim - stats.numel())
        else:
            pocket_context = pocket_context.reshape(-1)
        if stats.numel() + pocket_context.numel() < self.structural_dim:
            pad = residue_positions.new_zeros(self.structural_dim - stats.numel() - pocket_context.numel())
            structural = torch.cat([stats, pocket_context, pad], dim=0)
        else:
            structural = torch.cat([stats, pocket_context], dim=0)[: self.structural_dim]
        return self.structural_encoder(structural)

    def forward(
        self,
        *,
        residue_positions: torch.Tensor,
        pocket_context: Optional[torch.Tensor] = None,
        sequence: Optional[str] = None,
        sequence_embedding: Optional[torch.Tensor] = None,
        variant_ids: Optional[torch.Tensor] = None,
        variant_embedding: Optional[torch.Tensor] = None,
        structural_embedding: Optional[torch.Tensor] = None,
        allosteric_embedding: Optional[torch.Tensor] = None,
    ) -> IsoformHyperOutput:
        device = residue_positions.device
        dtype = residue_positions.dtype
        seq = self.encode_sequence(sequence, sequence_embedding, device=device, dtype=dtype)
        var = self.encode_variants(variant_ids, variant_embedding, device=device, dtype=dtype)
        if structural_embedding is None:
            structural = self.derive_structural_embedding(residue_positions, pocket_context=pocket_context)
        else:
            structural = structural_embedding.to(device=device, dtype=dtype)
            if structural.numel() != self.structural_dim:
                structural = structural.reshape(-1)[: self.structural_dim]
                if structural.numel() < self.structural_dim:
                    structural = torch.cat(
                        [structural, residue_positions.new_zeros(self.structural_dim - structural.numel())],
                        dim=0,
                    )
        if allosteric_embedding is None:
            allosteric = torch.zeros(self.allosteric_dim, device=device, dtype=dtype)
        else:
            allosteric = allosteric_embedding.to(device=device, dtype=dtype).reshape(-1)
            if allosteric.numel() != self.allosteric_dim:
                allosteric = allosteric[: self.allosteric_dim]
                if allosteric.numel() < self.allosteric_dim:
                    allosteric = torch.cat(
                        [allosteric, residue_positions.new_zeros(self.allosteric_dim - allosteric.numel())],
                        dim=0,
                    )
        fused = self.fusion(torch.cat([seq, structural, var, allosteric], dim=0))
        variant = self._variant_effects(variant_ids, device=device, dtype=dtype)
        delta_row_scale = 0.25 * torch.tanh(
            self.layer_row_scale(fused).view(self.siren_layers, self.siren_hidden_dim)
        )
        delta_bias_shift = 0.10 * torch.tanh(
            self.layer_bias_shift(fused).view(self.siren_layers, self.siren_hidden_dim)
        )
        row_scale = (
            self.global_row_scale.to(device=device, dtype=dtype) + delta_row_scale
        ) * variant["catalytic_activity"]
        bias_shift = (
            self.global_bias_shift.to(device=device, dtype=dtype)
            + delta_bias_shift
            + variant["bias_shift"]
        )
        access = self.accessibility_head(fused)
        output_scale = (
            self.global_output_scale.to(device=device, dtype=dtype)
            + 0.25 * torch.sigmoid(self.output_scale(fused).squeeze(-1))
        ) * variant["catalytic_activity"]
        accessibility_scale = (
            self.global_accessibility_scale.to(device=device, dtype=dtype)
            * (0.5 + torch.sigmoid(access[0]))
            * variant["accessibility_activity"]
        )
        sharpness_scale = (
            self.global_sharpness_scale.to(device=device, dtype=dtype)
            * (0.5 + torch.sigmoid(access[2]))
            * (0.9 + 0.2 * variant["accessibility_activity"])
        )
        heme_access_shift = 0.25 * torch.tanh(self.heme_shift(fused).squeeze(-1)) + variant["heme_shift"]
        return IsoformHyperOutput(
            pocket_embedding=self.pocket_head(fused),
            sequence_embedding=seq,
            variant_embedding=var,
            structural_embedding=structural,
            allosteric_embedding=allosteric,
            delta_row_scale=delta_row_scale,
            delta_bias_shift=delta_bias_shift,
            layer_row_scale=row_scale,
            layer_bias_shift=bias_shift,
            output_scale=output_scale,
            accessibility_scale=accessibility_scale,
            accessibility_bias=0.25 * torch.tanh(access[1]),
            sharpness_scale=sharpness_scale,
            residue_radius_delta=0.25 * torch.tanh(self.radius_head(fused).squeeze(-1)),
            heme_access_shift=heme_access_shift,
            catalytic_activity=variant["catalytic_activity"],
            accessibility_activity=variant["accessibility_activity"],
        )


__all__ = [
    "IsoformHyperOutput",
    "IsoformSpecificHyperNetwork",
    "IsoformSpecificPocketSIREN",
]
