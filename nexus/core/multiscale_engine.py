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
        self.scalar_to_pga = nn.Linear(128, 1)
        self.odd_scalar_to_pga = nn.Linear(128, 1)
        self.vector_to_pga = nn.Linear(128, 1)
        self.even_vector_to_trivector = nn.Linear(128, 1)
        self.tensor_coeff_to_pga = nn.Linear(64, 1)
        self.odd_tensor_coeff_to_pga = nn.Linear(64, 1)
        self.quadrupole_to_bivector = nn.Linear(5, 6)
        self.odd_tensor_to_trivector = nn.Linear(5, 4)

    def _collapse_vector_irrep(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        # x: [N, C, 3] -> [N, 3], preserving the vector basis while collapsing only channels.
        collapsed = linear(x.transpose(1, 2)).squeeze(-1)
        return collapsed

    def _collapse_tensor_irrep(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        # x: [N, C, 5] -> [N, 5], preserving the l=2 coefficient basis while collapsing channels.
        collapsed = linear(x.transpose(1, 2)).squeeze(-1)
        return collapsed

    def _pack_pga_multivector(
        self,
        *,
        scalar_even: torch.Tensor,
        scalar_odd: torch.Tensor,
        vector_odd: torch.Tensor,
        vector_even: torch.Tensor,
        tensor_even: torch.Tensor,
        tensor_odd: torch.Tensor,
    ) -> torch.Tensor:
        scalar = self.scalar_to_pga(scalar_even).squeeze(-1)
        odd_scalar = self.odd_scalar_to_pga(scalar_odd).squeeze(-1)
        vector = self._collapse_vector_irrep(vector_odd, self.vector_to_pga)
        trivector_seed = self._collapse_vector_irrep(vector_even, self.even_vector_to_trivector)
        quad_coeff = self._collapse_tensor_irrep(tensor_even, self.tensor_coeff_to_pga)
        odd_tensor_coeff = self._collapse_tensor_irrep(tensor_odd, self.odd_tensor_coeff_to_pga)
        bivector = self.quadrupole_to_bivector(quad_coeff)
        trivector = self.odd_tensor_to_trivector(odd_tensor_coeff)

        mv = torch.zeros(scalar_even.size(0), 16, dtype=scalar_even.dtype, device=scalar_even.device)
        mv[..., 0] = scalar
        mv[..., 1:4] = vector
        mv[..., 4] = odd_scalar
        mv[..., 5:11] = bivector
        mv[..., 11:14] = trivector[..., :3]
        mv[..., 14] = trivector[..., 3]
        mv[..., 15] = odd_scalar
        mv[..., 11] = mv[..., 11] + trivector_seed[..., 2]
        mv[..., 12] = mv[..., 12] + trivector_seed[..., 0]
        mv[..., 13] = mv[..., 13] + trivector_seed[..., 1]
        return mv

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
        fused["pga_multivector"] = self._pack_pga_multivector(
            scalar_even=symmetry["0e"],
            scalar_odd=symmetry["0o"],
            vector_odd=symmetry["1o"],
            vector_even=symmetry["1e"],
            tensor_even=symmetry["2e"],
            tensor_odd=symmetry["2o"],
        )
        fused["pga_multivector_topology"] = self._pack_pga_multivector(
            scalar_even=fused_0e,
            scalar_odd=symmetry["0o"],
            vector_odd=fused_1o,
            vector_even=symmetry["1e"],
            tensor_even=fused_2e,
            tensor_odd=symmetry["2o"],
        )
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
