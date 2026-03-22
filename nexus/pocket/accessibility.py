from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .allostery import AllostericEncoderOutput
from .ddi import DDIAccessibilityOutput, DDIOccupancyState
from .encoder import PocketEncoderOutput
from .hypernetwork import IsoformHyperOutput
from .nftm import NFTMMemoryState, NeuralFieldTuringMachine
from .pga import PGA_DIM


@dataclass
class AccessibilityFieldOutput:
    signed_distance: torch.Tensor
    accessibility: torch.Tensor
    steric_loss: torch.Tensor
    anchor_attention: torch.Tensor
    controller_context: torch.Tensor


@dataclass
class AccessibilityFieldState:
    pocket: PocketEncoderOutput
    residue_positions: torch.Tensor
    residue_types: torch.Tensor
    residue_radii: torch.Tensor
    controller_memory: torch.Tensor
    nftm_state: NFTMMemoryState
    field: "NeuralImplicitAccessibilityField"
    isoform: Optional[IsoformHyperOutput] = None

    def query(self, coords: torch.Tensor) -> AccessibilityFieldOutput:
        return self.field.query(coords, self)

    def query_with_occupancy(
        self,
        coords: torch.Tensor,
        occupancy_state: DDIOccupancyState,
    ) -> DDIAccessibilityOutput:
        return self.field.query_with_occupancy(coords, self, occupancy_state)

    def accessibility(self, coords: torch.Tensor) -> torch.Tensor:
        return self.query(coords).accessibility

    def accessibility_with_occupancy(
        self,
        coords: torch.Tensor,
        occupancy_state: DDIOccupancyState,
    ) -> torch.Tensor:
        return self.query_with_occupancy(coords, occupancy_state).total_accessibility

    def integrate_path(self, path: torch.Tensor) -> torch.Tensor:
        return self.field.integrate_path(path, self)

    def integrate_path_with_occupancy(
        self,
        path: torch.Tensor,
        occupancy_state: DDIOccupancyState,
    ) -> torch.Tensor:
        return self.field.integrate_path_with_occupancy(path, self, occupancy_state)

    def gate_scalar(self, coords: torch.Tensor) -> torch.Tensor:
        return self.field.gate_scalar(coords, self)

    def gate_scalar_with_occupancy(
        self,
        coords: torch.Tensor,
        occupancy_state: DDIOccupancyState,
    ) -> torch.Tensor:
        return self.field.gate_scalar_with_occupancy(coords, self, occupancy_state)


class NeuralImplicitAccessibilityField(nn.Module):
    def __init__(
        self,
        residue_vocab: int = 32,
        hidden_dim: int = 64,
        memory_dim: int = 64,
        sharpness: float = 8.0,
    ) -> None:
        super().__init__()
        self.sharpness = float(sharpness)
        self.hidden_dim = int(hidden_dim)
        self.memory_dim = int(memory_dim)
        self.residue_radius = nn.Embedding(residue_vocab, 1)
        with torch.no_grad():
            self.residue_radius.weight.fill_(1.8)
        self.memory_writer = nn.Sequential(
            nn.Linear(hidden_dim + PGA_DIM, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.anchor_key = nn.Linear(PGA_DIM, memory_dim)
        self.anchor_value = nn.Linear(hidden_dim + PGA_DIM, memory_dim)
        self.nftm = NeuralFieldTuringMachine(hidden_dim=hidden_dim, memory_dim=memory_dim)
        self.sdf_residual = nn.Sequential(
            nn.Linear(memory_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def build_state(
        self,
        residue_positions: torch.Tensor,
        residue_types: torch.Tensor,
        pocket: PocketEncoderOutput,
        isoform: Optional[IsoformHyperOutput] = None,
        allosteric: Optional[AllostericEncoderOutput] = None,
    ) -> AccessibilityFieldState:
        radii = self.residue_radius(
            residue_types.to(device=residue_positions.device, dtype=torch.long)
        ).squeeze(-1).to(dtype=residue_positions.dtype)
        if isoform is not None:
            radii = (radii + isoform.residue_radius_delta.to(device=radii.device, dtype=radii.dtype)).clamp_min(0.5)
        memory_input = torch.cat([pocket.scalar_features, pocket.attention_anchors], dim=-1)
        controller_memory = self.memory_writer(memory_input)
        if isoform is not None:
            controller_memory = controller_memory * isoform.accessibility_scale.to(
                device=controller_memory.device,
                dtype=controller_memory.dtype,
            )
            controller_memory = controller_memory + isoform.accessibility_bias.to(
                device=controller_memory.device,
                dtype=controller_memory.dtype,
            )
        nftm_state = self.nftm.build_state(
            pocket,
            residue_positions,
            radii,
            controller_context=controller_memory.mean(dim=0),
            allosteric=allosteric,
        )
        return AccessibilityFieldState(
            pocket=pocket,
            residue_positions=residue_positions,
            residue_types=residue_types,
            residue_radii=radii,
            controller_memory=controller_memory,
            nftm_state=nftm_state,
            field=self,
            isoform=isoform,
        )

    def query(
        self,
        coords: torch.Tensor,
        state: AccessibilityFieldState,
    ) -> AccessibilityFieldOutput:
        original_shape = coords.shape[:-1]
        points = coords.reshape(-1, 3)
        residue_positions = state.residue_positions.to(device=points.device, dtype=points.dtype)
        residue_radii = state.residue_radii.to(device=points.device, dtype=points.dtype)

        dist = torch.cdist(points.unsqueeze(0), residue_positions.unsqueeze(0)).squeeze(0)
        sdf_base = (dist - residue_radii.unsqueeze(0)).amin(dim=-1, keepdim=True)

        q = self.coord_proj(points)
        keys = self.anchor_key(state.pocket.attention_anchors.to(device=points.device, dtype=points.dtype))
        values = self.anchor_value(
            torch.cat(
                [
                    state.pocket.scalar_features.to(device=points.device, dtype=points.dtype),
                    state.pocket.attention_anchors.to(device=points.device, dtype=points.dtype),
                ],
                dim=-1,
            )
        )
        logits = torch.matmul(q, keys.transpose(0, 1)) / (keys.size(-1) ** 0.5)
        anchor_attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(anchor_attention, values)
        nftm_state = state.nftm_state
        if nftm_state.residue_positions.device != points.device or nftm_state.residue_positions.dtype != points.dtype:
            nftm_state = NFTMMemoryState(
                memory_slots=nftm_state.memory_slots.to(device=points.device, dtype=points.dtype),
                anchor_keys=nftm_state.anchor_keys.to(device=points.device, dtype=points.dtype),
                anchor_values=nftm_state.anchor_values.to(device=points.device, dtype=points.dtype),
                residue_positions=nftm_state.residue_positions.to(device=points.device, dtype=points.dtype),
                residue_radii=nftm_state.residue_radii.to(device=points.device, dtype=points.dtype),
                global_context=nftm_state.global_context.to(device=points.device, dtype=points.dtype),
                field=nftm_state.field,
            )
        nftm_readout = self.nftm(points, nftm_state)
        memory_context = nftm_readout.memory_context
        anchor_attention = nftm_readout.read_weights
        context = 0.5 * context + 0.5 * torch.matmul(
            anchor_attention,
            state.controller_memory.to(device=points.device, dtype=points.dtype),
        )

        residual = 0.25 * self.sdf_residual(torch.cat([context, memory_context, sdf_base], dim=-1))
        sharpness = torch.as_tensor(self.sharpness, dtype=points.dtype, device=points.device)
        accessibility_bias = torch.zeros((), dtype=points.dtype, device=points.device)
        if state.isoform is not None:
            sharpness = sharpness * state.isoform.sharpness_scale.to(device=points.device, dtype=points.dtype)
            accessibility_bias = state.isoform.heme_access_shift.to(device=points.device, dtype=points.dtype)
        signed_distance = sdf_base + residual
        accessibility = torch.sigmoid(sharpness * signed_distance + accessibility_bias)
        steric_loss = torch.relu(-signed_distance).mean()
        return AccessibilityFieldOutput(
            signed_distance=signed_distance.reshape(original_shape),
            accessibility=accessibility.reshape(original_shape),
            steric_loss=steric_loss,
            anchor_attention=anchor_attention.reshape(original_shape + (anchor_attention.size(-1),)),
            controller_context=context.reshape(original_shape + (context.size(-1),)),
        )

    def integrate_path(
        self,
        path: torch.Tensor,
        state: AccessibilityFieldState,
    ) -> torch.Tensor:
        access = self.query(path, state).accessibility
        return access.mean(dim=-1)

    def query_with_occupancy(
        self,
        coords: torch.Tensor,
        state: AccessibilityFieldState,
        occupancy_state: DDIOccupancyState,
    ) -> DDIAccessibilityOutput:
        base = self.query(coords, state)
        base_logits = torch.logit(base.accessibility.clamp(1.0e-6, 1.0 - 1.0e-6))
        return occupancy_state.field(base_logits, coords, occupancy_state)

    def integrate_path_with_occupancy(
        self,
        path: torch.Tensor,
        state: AccessibilityFieldState,
        occupancy_state: DDIOccupancyState,
    ) -> torch.Tensor:
        gated = self.query_with_occupancy(path, state, occupancy_state).total_accessibility
        return gated.mean(dim=-1)

    def gate_scalar(
        self,
        coords: torch.Tensor,
        state: AccessibilityFieldState,
    ) -> torch.Tensor:
        return self.query(coords, state).accessibility.mean()

    def gate_scalar_with_occupancy(
        self,
        coords: torch.Tensor,
        state: AccessibilityFieldState,
        occupancy_state: DDIOccupancyState,
    ) -> torch.Tensor:
        return self.query_with_occupancy(coords, state, occupancy_state).total_accessibility.mean()


__all__ = [
    "AccessibilityFieldOutput",
    "AccessibilityFieldState",
    "NeuralImplicitAccessibilityField",
]
