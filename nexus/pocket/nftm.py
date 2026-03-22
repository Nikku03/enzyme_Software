from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn as nn

from .allostery import AllostericEncoderOutput
from .encoder import PocketEncoderOutput


@dataclass
class NFTMMemoryState:
    memory_slots: torch.Tensor
    anchor_keys: torch.Tensor
    anchor_values: torch.Tensor
    residue_positions: torch.Tensor
    residue_radii: torch.Tensor
    global_context: torch.Tensor
    field: "NeuralFieldTuringMachine"


@dataclass
class NFTMReadout:
    memory_context: torch.Tensor
    read_weights: torch.Tensor
    write_gate: torch.Tensor
    symbolic_gate: torch.Tensor
    updated_state: NFTMMemoryState


class NeuralFieldTuringMachine(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        memory_dim: int = 64,
        steps: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.memory_dim = int(memory_dim)
        self.steps = int(steps)
        self.coord_proj = nn.Sequential(
            nn.Linear(3, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.controller_proj = nn.Sequential(
            nn.Linear(hidden_dim + 16 + memory_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.key_proj = nn.Linear(16, memory_dim)
        self.value_proj = nn.Linear(hidden_dim + 16, memory_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim * 2 + 2, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid(),
        )
        self.write_update = nn.Sequential(
            nn.Linear(memory_dim * 2 + 2, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )

    def build_state(
        self,
        pocket: PocketEncoderOutput,
        residue_positions: torch.Tensor,
        residue_radii: torch.Tensor,
        *,
        controller_context: Optional[torch.Tensor] = None,
        allosteric: Optional[AllostericEncoderOutput] = None,
    ) -> NFTMMemoryState:
        device = residue_positions.device
        dtype = residue_positions.dtype
        n_res = residue_positions.size(0)
        if controller_context is None:
            controller_context = residue_positions.new_zeros(self.memory_dim)
        controller_context = controller_context.to(device=device, dtype=dtype).reshape(1, -1).expand(n_res, -1)
        anchor_keys = self.key_proj(pocket.attention_anchors.to(device=device, dtype=dtype))
        anchor_values = self.value_proj(
            torch.cat(
                [
                    pocket.scalar_features.to(device=device, dtype=dtype),
                    pocket.attention_anchors.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
        )
        memory_input = torch.cat(
            [
                pocket.scalar_features.to(device=device, dtype=dtype),
                pocket.attention_anchors.to(device=device, dtype=dtype),
                controller_context,
            ],
            dim=-1,
        )
        memory_slots = self.controller_proj(memory_input)
        global_context = memory_slots.mean(dim=0)
        if allosteric is not None:
            global_context = global_context + 0.5 * allosteric.global_embedding.to(device=device, dtype=dtype)
            anchor_keys = anchor_keys + 0.25 * allosteric.pocket_embedding.to(device=device, dtype=dtype).reshape(1, -1)
        return NFTMMemoryState(
            memory_slots=memory_slots,
            anchor_keys=anchor_keys,
            anchor_values=anchor_values,
            residue_positions=residue_positions,
            residue_radii=residue_radii,
            global_context=global_context,
            field=self,
        )

    def _step(
        self,
        query_points: torch.Tensor,
        query_embed: torch.Tensor,
        state: NFTMMemoryState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, NFTMMemoryState]:
        logits = torch.matmul(query_embed, state.anchor_keys.transpose(0, 1)) / (self.memory_dim ** 0.5)
        read_weights = torch.softmax(logits, dim=-1)
        memory_context = torch.matmul(read_weights, state.memory_slots)

        geo_dist = torch.cdist(query_points.unsqueeze(0), state.residue_positions.to(device=query_embed.device, dtype=query_embed.dtype).unsqueeze(0)).squeeze(0)
        sdf_hint = (geo_dist - state.residue_radii.to(device=query_embed.device, dtype=query_embed.dtype).unsqueeze(0)).amin(dim=-1, keepdim=True)
        symbolic_gate = torch.sigmoid(4.0 * sdf_hint)

        write_in = torch.cat(
            [
                memory_context.unsqueeze(1).expand(-1, state.memory_slots.size(0), -1),
                state.memory_slots.unsqueeze(0).expand(query_embed.size(0), -1, -1),
                read_weights.unsqueeze(-1),
                symbolic_gate.unsqueeze(-1).expand(-1, state.memory_slots.size(0), -1),
            ],
            dim=-1,
        )
        write_gate = self.write_gate(write_in).mean(dim=0)
        delta = self.write_update(write_in).mean(dim=0)
        updated_memory = state.memory_slots + write_gate * delta
        updated_state = replace(state, memory_slots=updated_memory, global_context=updated_memory.mean(dim=0))
        return memory_context, read_weights, write_gate.squeeze(-1), symbolic_gate.squeeze(-1), updated_state

    def forward(
        self,
        coords: torch.Tensor,
        state: NFTMMemoryState,
    ) -> NFTMReadout:
        points = coords.reshape(-1, 3)
        query_embed = self.coord_proj(points)
        updated_state = state
        memory_context = query_embed.new_zeros(points.size(0), self.memory_dim)
        read_weights = query_embed.new_zeros(points.size(0), state.memory_slots.size(0))
        write_gate = query_embed.new_zeros(state.memory_slots.size(0))
        symbolic_gate = query_embed.new_zeros(points.size(0))
        for _ in range(max(self.steps, 1)):
            memory_context, read_weights, write_gate, symbolic_gate, updated_state = self._step(points, query_embed, updated_state)
            query_embed = query_embed + 0.5 * memory_context
        return NFTMReadout(
            memory_context=memory_context,
            read_weights=read_weights,
            write_gate=write_gate,
            symbolic_gate=symbolic_gate,
            updated_state=updated_state,
        )


__all__ = [
    "NFTMMemoryState",
    "NFTMReadout",
    "NeuralFieldTuringMachine",
]
