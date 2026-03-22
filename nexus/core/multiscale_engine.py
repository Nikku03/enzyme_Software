from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from nexus.symmetry.engine import O3_Symmetry_Engine
from nexus.topology.attention import MSGPreAttention, MSGPreAttentionOutput
from nexus.topology.kernels import MultiScale_Topology_Kernel, TopologyKernelOutput


@dataclass
class MultiScaleEngineOutput:
    topology: TopologyKernelOutput
    attention: MSGPreAttentionOutput
    symmetry: Dict[str, torch.Tensor]
    fused_features: Dict[str, torch.Tensor]
    entropy_loss: torch.Tensor


class MultiScale_Topology_Engine(nn.Module):
    def __init__(
        self,
        topology_kernel: Optional[MultiScale_Topology_Kernel] = None,
        attention: Optional[MSGPreAttention] = None,
        symmetry_engine: Optional[O3_Symmetry_Engine] = None,
    ) -> None:
        super().__init__()
        self.topology_kernel = topology_kernel or MultiScale_Topology_Kernel()
        self.attention = attention or MSGPreAttention(feature_dim=128)
        self.symmetry_engine = symmetry_engine or O3_Symmetry_Engine()
        self.scalar_fuse = nn.Sequential(nn.Linear(128 + 128 + 128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.vector_fuse = nn.Sequential(nn.Linear(128 + 1, 128), nn.SiLU(), nn.Linear(128, 128))
        self.tensor_fuse = nn.Sequential(nn.Linear(64 + 1, 64), nn.SiLU(), nn.Linear(64, 64))

    def forward(self, manifold) -> MultiScaleEngineOutput:
        topology = self.topology_kernel(manifold)
        attention = self.attention(
            topology.atomic_features,
            topology.functional_group_features,
            topology.conformer_features,
            topology.distance_matrix,
            topology.masks["local"],
            topology.masks["global"],
            topology.functional_group_assignments,
        )
        symmetry = self.symmetry_engine(manifold)

        conformer_expand = attention.conformer_features.expand(topology.atomic_features.size(0), -1)
        fused_0e = self.scalar_fuse(torch.cat([symmetry["0e"], attention.atomic_features, conformer_expand], dim=-1))
        scalar_gate = attention.scale_weights[0].reshape(1, 1)
        tensor_gate = attention.scale_weights[1].reshape(1, 1)
        fused_1o = self.vector_fuse(
            torch.cat([symmetry["1o"].norm(dim=-1), scalar_gate.expand(symmetry["1o"].size(0), 1)], dim=-1)
        ).unsqueeze(-1) * symmetry["1o"]
        fused_2e = self.tensor_fuse(
            torch.cat([symmetry["2e"].norm(dim=-1), tensor_gate.expand(symmetry["2e"].size(0), 1)], dim=-1)
        ).unsqueeze(-1) * symmetry["2e"]

        fused = dict(symmetry)
        fused["0e_topology"] = fused_0e
        fused["1o_topology"] = fused_1o
        fused["2e_topology"] = fused_2e
        fused["topology_atomic"] = attention.atomic_features
        fused["topology_group"] = attention.group_features
        fused["topology_conformer"] = attention.conformer_features
        fused["scale_weights"] = attention.scale_weights

        return MultiScaleEngineOutput(
            topology=topology,
            attention=attention,
            symmetry=symmetry,
            fused_features=fused,
            entropy_loss=attention.entropy_loss,
        )
