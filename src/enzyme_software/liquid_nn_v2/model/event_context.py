from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class SparseEventContext(nn.Module):
        """Sparse relay/event context over the atom graph."""

        def __init__(self, *, atom_dim: int, hidden_dim: int = 24, rounds: int = 3, dropout: float = 0.05):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.rounds = max(1, int(rounds))
            self.output_dim = self.hidden_dim + 3
            self.input_proj = nn.Sequential(
                nn.Linear(atom_dim + 5, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.local_drive = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.SiLU(),
            )
            self.message_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.relay_proj = nn.Sequential(
                nn.Linear(self.hidden_dim + 5, self.hidden_dim),
                nn.SiLU(),
            )
            self.strain_head = nn.Linear(self.hidden_dim, 1)
            self.hidden_norm = nn.LayerNorm(self.hidden_dim)
            self.alpha = nn.Parameter(torch.tensor(0.60))
            self.beta = nn.Parameter(torch.tensor(0.35))
            self.threshold = nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            *,
            atom_features: torch.Tensor,
            edge_index: torch.Tensor | None,
            atom_coords: torch.Tensor | None,
            local_chem_features: torch.Tensor | None,
            candidate_mask: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            rows = int(atom_features.size(0))
            device = atom_features.device
            dtype = atom_features.dtype
            if local_chem_features is None:
                local = torch.zeros(rows, 5, device=device, dtype=dtype)
            else:
                chem = local_chem_features.to(device=device, dtype=dtype)
                if int(chem.size(-1)) >= 11:
                    local = torch.cat(
                        [
                            chem[:, 2:3],   # field
                            chem[:, 4:5],   # crowding
                            chem[:, 6:7],   # |dq|
                            chem[:, 9:10],  # etn gap
                            chem[:, 10:11], # etn zscore
                        ],
                        dim=-1,
                    )
                else:
                    local = torch.nn.functional.pad(chem, (0, max(0, 5 - int(chem.size(-1)))))[:, :5]
            cand = candidate_mask.to(device=device, dtype=dtype) if candidate_mask is not None else torch.ones(rows, 1, device=device, dtype=dtype)
            hidden = self.input_proj(torch.cat([atom_features, local], dim=-1))
            hidden = self.hidden_norm(hidden)
            local_drive = self.local_drive(local)
            strain = local[:, 0:1] - (0.35 * local[:, 1:2]) + (0.50 * local[:, 2:3]) + (0.20 * cand)
            active_count = torch.zeros(rows, 1, device=device, dtype=dtype)
            event_depth = torch.zeros(rows, 1, device=device, dtype=dtype)

            if edge_index is None or int(edge_index.numel()) == 0:
                relay = torch.tanh(self.relay_proj(torch.cat([hidden, local], dim=-1)))
                return {
                    "relay_context": relay,
                    "strain": strain,
                    "active_neighbor_count": active_count,
                    "event_depth": event_depth,
                    "features": torch.cat([relay, active_count, event_depth, strain], dim=-1),
                }

            src = edge_index[0].long()
            dst = edge_index[1].long()
            edge_weight = torch.ones(int(src.numel()), 1, device=device, dtype=dtype)
            if atom_coords is not None and int(atom_coords.numel()):
                coords = atom_coords.to(device=device, dtype=dtype)
                distances = torch.norm(coords[src] - coords[dst], dim=-1, keepdim=True)
                edge_weight = torch.exp(-distances / 2.5)

            alpha = torch.sigmoid(self.alpha).to(device=device, dtype=dtype)
            beta = torch.sigmoid(self.beta).to(device=device, dtype=dtype)
            for _ in range(self.rounds):
                active = torch.sigmoid(3.0 * (strain - self.threshold.to(device=device, dtype=dtype)))
                msg = self.message_proj(hidden[src]) * edge_weight * active[src]
                agg = torch.zeros_like(hidden)
                agg.index_add_(0, dst, msg)
                neigh = torch.zeros(rows, 1, device=device, dtype=dtype)
                neigh.index_add_(0, dst, active[src] * edge_weight)
                hidden = self.hidden_norm((alpha * hidden) + (beta * agg) + (0.25 * local_drive))
                hidden = torch.nn.functional.silu(hidden)
                strain_msg = self.strain_head(hidden[src]) * edge_weight
                agg_strain = torch.zeros(rows, 1, device=device, dtype=dtype)
                agg_strain.index_add_(0, dst, strain_msg)
                strain = (0.60 * strain) + (0.30 * agg_strain) + (0.10 * (local[:, 0:1] - local[:, 1:2] + local[:, 2:3]))
                active_count = active_count + neigh
                event_depth = event_depth + active

            active_count = active_count / float(self.rounds)
            event_depth = event_depth / float(self.rounds)
            relay = torch.tanh(self.relay_proj(torch.cat([hidden, local], dim=-1)))
            return {
                "relay_context": relay,
                "strain": strain,
                "active_neighbor_count": active_count,
                "event_depth": event_depth,
                "features": torch.cat([relay, active_count, event_depth, strain], dim=-1),
            }
else:  # pragma: no cover
    class SparseEventContext:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

