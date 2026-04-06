from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class Phase5SparseRelay(nn.Module):
        """Sparse local context relay for proposer embeddings."""

        def __init__(
            self,
            *,
            atom_dim: int,
            phase5_dim: int = 18,
            hidden_dim: int = 96,
            rounds: int = 2,
            radius: float = 4.5,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.atom_dim = int(atom_dim)
            self.phase5_dim = int(phase5_dim)
            self.hidden_dim = max(32, int(hidden_dim))
            self.rounds = max(1, int(rounds))
            self.radius = max(1.0, float(radius))
            self.output_dim = self.atom_dim

            self.phase5_bias = nn.Sequential(
                nn.Linear(self.phase5_dim, self.atom_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.atom_dim, self.atom_dim),
            )
            self.message_proj = nn.Linear(self.atom_dim, self.atom_dim)
            self.edge_gate = nn.Sequential(
                nn.Linear(7, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            self.relay_proj = nn.Sequential(
                nn.Linear((2 * self.atom_dim) + self.phase5_dim, self.atom_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.atom_dim, self.atom_dim),
            )
            self.hidden_norm = nn.LayerNorm(self.atom_dim)
            self.alpha = nn.Parameter(torch.tensor(0.60))
            self.beta = nn.Parameter(torch.tensor(0.35))

        def _radius_edges(
            self,
            atom_coords: torch.Tensor | None,
            batch_index: torch.Tensor,
            *,
            device,
            dtype,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if atom_coords is None or int(atom_coords.numel()) == 0 or int(batch_index.numel()) == 0:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                empty_dist = torch.zeros(0, 1, dtype=dtype, device=device)
                return empty, empty, empty_dist
            coords = atom_coords.to(device=device, dtype=dtype)
            src_parts = []
            dst_parts = []
            dist_parts = []
            num_graphs = int(batch_index.max().item()) + 1
            for mol_idx in range(num_graphs):
                mask = batch_index == mol_idx
                count = int(mask.sum().item())
                if count <= 1:
                    continue
                local_idx = torch.nonzero(mask, as_tuple=False).view(-1)
                local_coords = coords[local_idx]
                distances = torch.cdist(local_coords, local_coords, p=2)
                radius_mask = (distances > 1.0e-4) & (distances <= self.radius)
                pair_idx = torch.nonzero(radius_mask, as_tuple=False)
                if int(pair_idx.numel()) == 0:
                    continue
                src_parts.append(local_idx[pair_idx[:, 0]])
                dst_parts.append(local_idx[pair_idx[:, 1]])
                dist_parts.append(distances[pair_idx[:, 0], pair_idx[:, 1]].unsqueeze(-1))
            if not src_parts:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                empty_dist = torch.zeros(0, 1, dtype=dtype, device=device)
                return empty, empty, empty_dist
            return (
                torch.cat(src_parts, dim=0),
                torch.cat(dst_parts, dim=0),
                torch.cat(dist_parts, dim=0).to(device=device, dtype=dtype),
            )

        def forward(
            self,
            *,
            atom_features: torch.Tensor,
            edge_index: torch.Tensor | None,
            atom_coords: torch.Tensor | None,
            batch_index: torch.Tensor,
            candidate_mask: torch.Tensor | None,
            phase5_atom_features: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            rows = int(atom_features.size(0))
            device = atom_features.device
            dtype = atom_features.dtype
            phase5 = phase5_atom_features.to(device=device, dtype=dtype)
            candidate = (
                candidate_mask.to(device=device, dtype=dtype).view(rows, 1)
                if candidate_mask is not None
                else torch.ones(rows, 1, device=device, dtype=dtype)
            )
            bias = torch.tanh(self.phase5_bias(phase5))
            state = self.hidden_norm(atom_features + (0.15 * bias))

            bond_src = torch.zeros(0, dtype=torch.long, device=device)
            bond_dst = torch.zeros(0, dtype=torch.long, device=device)
            bond_dist = torch.zeros(0, 1, dtype=dtype, device=device)
            if edge_index is not None and int(edge_index.numel()) > 0:
                bond_src = edge_index[0].long().to(device=device)
                bond_dst = edge_index[1].long().to(device=device)
                if atom_coords is not None and int(atom_coords.numel()) > 0:
                    coords = atom_coords.to(device=device, dtype=dtype)
                    bond_dist = torch.norm(coords[bond_src] - coords[bond_dst], dim=-1, keepdim=True)
                else:
                    bond_dist = torch.ones(int(bond_src.numel()), 1, device=device, dtype=dtype)

            radius_src, radius_dst, radius_dist = self._radius_edges(
                atom_coords,
                batch_index,
                device=device,
                dtype=dtype,
            )

            src = torch.cat([bond_src, radius_src], dim=0)
            dst = torch.cat([bond_dst, radius_dst], dim=0)
            distances = torch.cat([bond_dist, radius_dist], dim=0)

            if int(src.numel()) == 0:
                relay = torch.tanh(self.relay_proj(torch.cat([state, bias, phase5], dim=-1)))
                zeros = torch.zeros(rows, 1, device=device, dtype=dtype)
                return {
                    "relay_features": relay,
                    "neighbor_count": zeros,
                    "activity": zeros,
                    "distance_mean": zeros,
                }

            inv_distance = 1.0 / (1.0 + distances)
            alpha = torch.sigmoid(self.alpha).to(device=device, dtype=dtype)
            beta = torch.sigmoid(self.beta).to(device=device, dtype=dtype)
            neighbor_accum = torch.zeros(rows, 1, device=device, dtype=dtype)
            distance_accum = torch.zeros(rows, 1, device=device, dtype=dtype)
            activity_accum = torch.zeros(rows, 1, device=device, dtype=dtype)

            boundary = phase5[:, 0:1]
            access = phase5[:, 8:9] if int(phase5.size(-1)) >= 9 else torch.zeros(rows, 1, device=device, dtype=dtype)

            for _ in range(self.rounds):
                edge_features = torch.cat(
                    [
                        inv_distance,
                        boundary[src],
                        boundary[dst],
                        access[src],
                        access[dst],
                        candidate[src],
                        candidate[dst],
                    ],
                    dim=-1,
                )
                edge_gate = torch.sigmoid(self.edge_gate(edge_features))
                messages = self.message_proj(state[src]) * edge_gate
                agg = torch.zeros_like(state)
                agg.index_add_(0, dst, messages)
                neigh = torch.zeros(rows, 1, device=device, dtype=dtype)
                neigh.index_add_(0, dst, edge_gate)
                dist_sum = torch.zeros(rows, 1, device=device, dtype=dtype)
                dist_sum.index_add_(0, dst, distances * edge_gate)
                activity = torch.tanh(torch.norm(agg, dim=-1, keepdim=True) / max(1.0, float(self.atom_dim) ** 0.5))
                state = self.hidden_norm((alpha * state) + (beta * agg) + (0.20 * bias))
                state = torch.nn.functional.silu(state)
                neighbor_accum = neighbor_accum + neigh
                distance_accum = distance_accum + dist_sum
                activity_accum = activity_accum + activity

            neighbor_mean = neighbor_accum / float(self.rounds)
            distance_mean = distance_accum / torch.clamp(neighbor_accum, min=1.0)
            activity_mean = activity_accum / float(self.rounds)
            relay = torch.tanh(self.relay_proj(torch.cat([state, bias, phase5], dim=-1)))
            return {
                "relay_features": relay,
                "neighbor_count": neighbor_mean,
                "activity": activity_mean,
                "distance_mean": distance_mean,
            }
else:  # pragma: no cover
    class Phase5SparseRelay:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
