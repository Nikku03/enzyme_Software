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
            edge_feature_dim: int = 10,
            hidden_dim: int = 96,
            rounds: int = 2,
            radius: float = 4.5,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.atom_dim = int(atom_dim)
            self.phase5_dim = int(phase5_dim)
            self.edge_feature_dim = int(edge_feature_dim)
            self.hidden_dim = max(32, int(hidden_dim))
            self.rounds = max(1, int(rounds))
            self.radius = max(1.0, float(radius))
            self.output_dim = self.atom_dim
            self.bias_feature_dim = 14

            self.phase5_bias = nn.Sequential(
                nn.Linear(self.bias_feature_dim, self.atom_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.atom_dim, self.atom_dim),
            )
            self.message_proj = nn.Linear(self.atom_dim, self.atom_dim)
            self.edge_feature_proj = nn.Sequential(
                nn.Linear(self.edge_feature_dim, max(8, self.hidden_dim // 2)),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, self.hidden_dim // 2), 1),
            )
            self.relay_proj = nn.Sequential(
                nn.Linear((2 * self.atom_dim) + self.bias_feature_dim, self.atom_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.atom_dim, self.atom_dim),
            )
            self.hidden_norm = nn.LayerNorm(self.atom_dim)
            self.profile_proj = nn.Linear(8, 1)
            self.alpha = nn.Parameter(torch.tensor(0.62))
            self.beta = nn.Parameter(torch.tensor(0.30))
            self.gamma_edge = nn.Parameter(torch.tensor(0.30))
            self.gamma_access = nn.Parameter(torch.tensor(0.50))
            self.gamma_charge = nn.Parameter(torch.tensor(0.45))
            self.gamma_env = nn.Parameter(torch.tensor(0.10))
            self.distance_scale_log = nn.Parameter(torch.tensor(0.0))

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
                if int(mask.sum().item()) <= 1:
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

        def _selected_bias_features(
            self,
            *,
            phase5_atom_features: torch.Tensor,
            local_chem_features: torch.Tensor | None,
        ) -> torch.Tensor:
            phase5 = phase5_atom_features
            rows = int(phase5.size(0))
            device = phase5.device
            dtype = phase5.dtype
            boundary_scalar = phase5[:, 0:1]
            access_cost = phase5[:, 7:8] if int(phase5.size(-1)) > 7 else torch.zeros(rows, 1, device=device, dtype=dtype)
            access_score = phase5[:, 8:9] if int(phase5.size(-1)) > 8 else torch.zeros(rows, 1, device=device, dtype=dtype)
            access_blockage = phase5[:, 9:10] if int(phase5.size(-1)) > 9 else torch.zeros(rows, 1, device=device, dtype=dtype)
            profile = phase5[:, 10:18] if int(phase5.size(-1)) >= 18 else torch.zeros(rows, 8, device=device, dtype=dtype)
            if local_chem_features is None:
                charge_delta = torch.zeros(rows, 1, device=device, dtype=dtype)
                abs_charge_delta = torch.zeros(rows, 1, device=device, dtype=dtype)
            else:
                chem = local_chem_features.to(device=device, dtype=dtype)
                charge_delta = chem[:, 5:6] if int(chem.size(-1)) > 5 else torch.zeros(rows, 1, device=device, dtype=dtype)
                abs_charge_delta = chem[:, 6:7] if int(chem.size(-1)) > 6 else torch.zeros(rows, 1, device=device, dtype=dtype)
            return torch.cat(
                [
                    boundary_scalar,
                    access_cost,
                    access_score,
                    access_blockage,
                    charge_delta,
                    abs_charge_delta,
                    profile,
                ],
                dim=-1,
            )

        def _incoming_softmax(
            self,
            scores: torch.Tensor,
            dst: torch.Tensor,
            rows: int,
            *,
            dtype,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            weights = torch.zeros_like(scores)
            entropy = torch.zeros(rows, 1, device=scores.device, dtype=dtype)
            unique_dst = torch.unique(dst)
            for node in unique_dst.tolist():
                mask = dst == int(node)
                node_scores = scores[mask].view(-1)
                node_weights = torch.softmax(node_scores, dim=0).unsqueeze(-1)
                weights[mask] = node_weights
                if int(node_weights.numel()) > 1:
                    probs = node_weights.view(-1).clamp_min(1.0e-8)
                    node_entropy = -(probs * probs.log()).sum()
                    node_entropy = node_entropy / torch.log(torch.tensor(float(probs.numel()), device=scores.device, dtype=dtype))
                    entropy[int(node), 0] = node_entropy
            return weights, entropy

        def forward(
            self,
            *,
            atom_features: torch.Tensor,
            edge_index: torch.Tensor | None,
            edge_attr: torch.Tensor | None,
            atom_coords: torch.Tensor | None,
            batch_index: torch.Tensor,
            candidate_mask: torch.Tensor | None,
            phase5_atom_features: torch.Tensor,
            local_chem_features: torch.Tensor | None = None,
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
            bias_features = self._selected_bias_features(
                phase5_atom_features=phase5,
                local_chem_features=local_chem_features,
            )
            bias = torch.tanh(self.phase5_bias(bias_features))
            state0 = self.hidden_norm(atom_features + (0.15 * bias))
            state = state0

            bond_src = torch.zeros(0, dtype=torch.long, device=device)
            bond_dst = torch.zeros(0, dtype=torch.long, device=device)
            bond_dist = torch.zeros(0, 1, dtype=dtype, device=device)
            bond_edge = torch.zeros(0, self.edge_feature_dim, dtype=dtype, device=device)
            if edge_index is not None and int(edge_index.numel()) > 0:
                bond_src = edge_index[0].long().to(device=device)
                bond_dst = edge_index[1].long().to(device=device)
                if atom_coords is not None and int(atom_coords.numel()) > 0:
                    coords = atom_coords.to(device=device, dtype=dtype)
                    bond_dist = torch.norm(coords[bond_src] - coords[bond_dst], dim=-1, keepdim=True)
                else:
                    bond_dist = torch.ones(int(bond_src.numel()), 1, device=device, dtype=dtype)
                if edge_attr is not None and int(edge_attr.numel()) > 0:
                    edge_values = edge_attr.to(device=device, dtype=dtype)
                    width = min(int(edge_values.size(-1)), self.edge_feature_dim)
                    bond_edge = torch.zeros(int(edge_values.size(0)), self.edge_feature_dim, device=device, dtype=dtype)
                    bond_edge[:, :width] = edge_values[:, :width]

            radius_src, radius_dst, radius_dist = self._radius_edges(
                atom_coords,
                batch_index,
                device=device,
                dtype=dtype,
            )
            radius_edge = torch.zeros(int(radius_src.numel()), self.edge_feature_dim, device=device, dtype=dtype)

            src = torch.cat([bond_src, radius_src], dim=0)
            dst = torch.cat([bond_dst, radius_dst], dim=0)
            distances = torch.cat([bond_dist, radius_dist], dim=0)
            edge_features = torch.cat([bond_edge, radius_edge], dim=0)

            if int(src.numel()) == 0:
                relay = torch.tanh(self.relay_proj(torch.cat([state, bias, bias_features], dim=-1)))
                zeros = torch.zeros(rows, 1, device=device, dtype=dtype)
                return {
                    "relay_features": relay,
                    "neighbor_count": zeros,
                    "activity": zeros,
                    "distance_mean": zeros,
                    "weight_entropy": zeros,
                }

            access_score = bias_features[:, 2:3]
            charge_delta = bias_features[:, 4:5]
            boundary = bias_features[:, 0:1]
            profile_bias = self.profile_proj(bias_features[:, 6:14])
            env_signal = boundary + (0.10 * profile_bias)

            alpha = torch.sigmoid(self.alpha).to(device=device, dtype=dtype)
            beta = torch.sigmoid(self.beta).to(device=device, dtype=dtype)
            gamma_edge = torch.nn.functional.softplus(self.gamma_edge).to(device=device, dtype=dtype)
            gamma_access = torch.nn.functional.softplus(self.gamma_access).to(device=device, dtype=dtype)
            gamma_charge = torch.nn.functional.softplus(self.gamma_charge).to(device=device, dtype=dtype)
            gamma_env = torch.nn.functional.softplus(self.gamma_env).to(device=device, dtype=dtype)
            sigma_r_sq = torch.square(torch.nn.functional.softplus(self.distance_scale_log).to(device=device, dtype=dtype) + self.radius)

            neighbor_accum = torch.zeros(rows, 1, device=device, dtype=dtype)
            distance_accum = torch.zeros(rows, 1, device=device, dtype=dtype)
            activity_accum = torch.zeros(rows, 1, device=device, dtype=dtype)
            entropy_accum = torch.zeros(rows, 1, device=device, dtype=dtype)

            for _ in range(self.rounds):
                distance_term = -(torch.square(distances) / sigma_r_sq)
                edge_term = gamma_edge * self.edge_feature_proj(edge_features)
                access_term = gamma_access * (access_score[src] - access_score[dst])
                charge_term = gamma_charge * (charge_delta[src] - charge_delta[dst])
                env_term = gamma_env * (env_signal[src] - env_signal[dst])
                raw_scores = distance_term + edge_term + access_term + charge_term + env_term
                weights, entropy = self._incoming_softmax(raw_scores, dst, rows, dtype=dtype)
                weights = weights * (0.85 + (0.15 * candidate[src]))

                messages = self.message_proj(state[src]) * weights
                agg = torch.zeros_like(state)
                agg.index_add_(0, dst, messages)

                neigh = torch.zeros(rows, 1, device=device, dtype=dtype)
                neigh.index_add_(0, dst, (weights > 1.0e-6).to(dtype))
                dist_sum = torch.zeros(rows, 1, device=device, dtype=dtype)
                dist_sum.index_add_(0, dst, distances * weights)
                activity = torch.tanh(torch.norm(agg - state, dim=-1, keepdim=True) / max(1.0, float(self.atom_dim) ** 0.5))

                state = self.hidden_norm((alpha * state) + (beta * agg) + (0.20 * bias))
                state = torch.nn.functional.silu(state)
                neighbor_accum = neighbor_accum + neigh
                distance_accum = distance_accum + dist_sum
                activity_accum = activity_accum + activity
                entropy_accum = entropy_accum + entropy

            neighbor_mean = neighbor_accum / float(self.rounds)
            distance_mean = distance_accum / torch.clamp(neighbor_accum, min=1.0)
            activity_mean = activity_accum / float(self.rounds)
            entropy_mean = entropy_accum / float(self.rounds)
            relay = torch.tanh(self.relay_proj(torch.cat([state, bias, bias_features], dim=-1)))
            return {
                "relay_features": relay,
                "neighbor_count": neighbor_mean,
                "activity": activity_mean,
                "distance_mean": distance_mean,
                "weight_entropy": entropy_mean,
                "feature_delta": relay - state0,
            }
else:  # pragma: no cover
    class Phase5SparseRelay:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
