from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class AccessibilityHead(nn.Module):
        """Cheap cavity/accessibility approximation over atom coordinates."""

        def __init__(self, *, hidden_dim: int = 16, dropout: float = 0.05):
            super().__init__()
            self.output_dim = 4
            self.feature_proj = nn.Sequential(
                nn.Linear(4, max(8, int(hidden_dim))),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, int(hidden_dim)), 4),
            )

        def forward(
            self,
            *,
            atom_coords: torch.Tensor | None,
            batch_index: torch.Tensor,
            local_chem_features: torch.Tensor | None,
            candidate_mask: torch.Tensor | None,
            event_outputs: dict[str, torch.Tensor] | None = None,
        ) -> dict[str, torch.Tensor]:
            rows = int(batch_index.size(0))
            device = batch_index.device
            dtype = torch.float32
            if local_chem_features is None:
                local = torch.zeros(rows, 5, device=device, dtype=dtype)
            else:
                local = local_chem_features.to(device=device, dtype=dtype)
            steric = local[:, 0:1] if int(local.size(-1)) > 0 else torch.zeros(rows, 1, device=device, dtype=dtype)
            field = local[:, 2:3] if int(local.size(-1)) > 2 else torch.zeros(rows, 1, device=device, dtype=dtype)
            access_prior = local[:, 3:4] if int(local.size(-1)) > 3 else torch.zeros(rows, 1, device=device, dtype=dtype)
            crowding = local[:, 4:5] if int(local.size(-1)) > 4 else torch.zeros(rows, 1, device=device, dtype=dtype)
            candidate = candidate_mask.to(device=device, dtype=dtype) if candidate_mask is not None else torch.ones(rows, 1, device=device, dtype=dtype)
            event_term = torch.sigmoid(event_outputs["strain"]) if event_outputs is not None and "strain" in event_outputs else torch.zeros(rows, 1, device=device, dtype=dtype)

            if atom_coords is None or int(atom_coords.numel()) == 0:
                access_score = torch.sigmoid(access_prior - crowding)
                access_cost = 1.0 - access_score
                path_length = access_cost
                blockage = torch.relu(crowding) + (0.25 * torch.relu(steric))
            else:
                coords = atom_coords.to(device=device, dtype=dtype)
                access_score = torch.zeros(rows, 1, device=device, dtype=dtype)
                access_cost = torch.zeros(rows, 1, device=device, dtype=dtype)
                path_length = torch.zeros(rows, 1, device=device, dtype=dtype)
                blockage = torch.zeros(rows, 1, device=device, dtype=dtype)
                batch_max = int(batch_index.max().item()) + 1 if int(batch_index.numel()) else 0
                source_score = access_prior - (0.5 * crowding) + (0.25 * field) + (0.15 * candidate)
                for mol_idx in range(batch_max):
                    mask = batch_index == mol_idx
                    if not bool(mask.any()):
                        continue
                    mol_coords = coords[mask]
                    mol_source = source_score[mask].view(-1)
                    source_local = int(torch.argmax(mol_source).item())
                    source_coord = mol_coords[source_local : source_local + 1]
                    dist = torch.norm(mol_coords - source_coord, dim=-1, keepdim=True)
                    steric_m = steric[mask]
                    crowd_m = crowding[mask]
                    field_m = field[mask]
                    event_m = event_term[mask]
                    block = torch.relu(crowd_m) + (0.50 * torch.relu(steric_m)) + (0.20 * (1.0 - torch.sigmoid(field_m)))
                    block = block * (1.0 - (0.20 * event_m))
                    cost = dist * (1.0 + block)
                    score = torch.exp(-cost / 3.0)
                    access_score[mask] = score
                    access_cost[mask] = cost
                    path_length[mask] = dist
                    blockage[mask] = block
            raw = torch.cat([access_score, access_cost, path_length, blockage], dim=-1)
            return {
                "score": access_score,
                "cost": access_cost,
                "path_length": path_length,
                "blockage": blockage,
                "features": self.feature_proj(raw),
            }
else:  # pragma: no cover
    class AccessibilityHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

