from __future__ import annotations

from typing import Dict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class _SineLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, *, omega_0: float = 12.0) -> None:
            super().__init__()
            self.linear = nn.Linear(int(in_dim), int(out_dim))
            self.omega_0 = float(omega_0)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            with torch.no_grad():
                bound = 1.0 / max(1, int(self.linear.in_features))
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.omega_0 * self.linear(x))


    class WholeMoleculeWaveField(nn.Module):
        """
        Query-based whole-molecule wave-field surrogate.

        The module evaluates each atom against the continuous field induced by all
        atoms in the same molecule. It uses:
        - a light SIREN-like field network over query coordinates and atom context
        - all-atom Gaussian aggregation using learned multivector amplitudes
        - finite-difference shell probes to estimate local density and curvature
        """

        field_feature_dim: int = 10

        def __init__(
            self,
            *,
            multivector_dim: int = 16,
            hidden_dim: int = 64,
            omega_0: float = 12.0,
        ) -> None:
            super().__init__()
            self.multivector_dim = int(multivector_dim)
            self.hidden_dim = int(max(32, hidden_dim))
            self.atom_param = nn.Sequential(
                nn.Linear(self.multivector_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 8),
            )
            self.coord_context = nn.Sequential(
                _SineLayer(3 + self.multivector_dim + 4, self.hidden_dim, omega_0=omega_0),
                _SineLayer(self.hidden_dim, self.hidden_dim, omega_0=omega_0),
                nn.Linear(self.hidden_dim, 1),
            )
            directions = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=torch.float32,
            )
            radii = torch.tensor([0.0, 0.75, 1.50], dtype=torch.float32)
            self.register_buffer("probe_directions", directions, persistent=False)
            self.register_buffer("probe_radii", radii, persistent=False)

        def _probe_points(self, coords: torch.Tensor) -> torch.Tensor:
            dirs = self.probe_directions.to(device=coords.device, dtype=coords.dtype)
            radii = self.probe_radii.to(device=coords.device, dtype=coords.dtype)
            offsets = []
            for ridx, radius in enumerate(radii):
                if ridx == 0:
                    offsets.append(dirs[:1] * radius)
                else:
                    offsets.append(dirs[1:] * radius)
            probe_offsets = torch.cat(offsets, dim=0)  # [13,3]
            return coords.unsqueeze(1) + probe_offsets.unsqueeze(0)

        def _molecule_field(self, mv: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
            num_atoms = int(coords.size(0))
            probes = self._probe_points(coords)  # [N,Q,3]
            q = probes.size(1)

            params = self.atom_param(mv)
            alpha = torch.nn.functional.softplus(params[:, 0:1]) + 0.25
            scalar_amp = mv[:, 0:1]
            vector_amp = mv[:, 1:4]
            bivec_amp = (mv[:, 5:11].pow(2).sum(dim=-1, keepdim=True) + 1.0e-12).sqrt()
            pseudo_amp = mv[:, 15:16]

            rel = probes.unsqueeze(2) - coords.view(1, 1, num_atoms, 3)
            dist2 = rel.square().sum(dim=-1, keepdim=True)
            kernel = torch.exp(-alpha.view(1, 1, num_atoms, 1) * dist2)
            rel_norm = dist2.sqrt().clamp_min(1.0e-6)
            rel_unit = rel / rel_norm

            scalar_density = (kernel * scalar_amp.view(1, 1, num_atoms, 1)).sum(dim=2).squeeze(-1)
            vector_flow = (kernel * (rel_unit * vector_amp.view(1, 1, num_atoms, 3))).sum(dim=2)
            vector_align = (vector_flow.pow(2).sum(dim=-1) + 1.0e-12).sqrt()
            bivector_density = (kernel * bivec_amp.view(1, 1, num_atoms, 1)).sum(dim=2).squeeze(-1)
            pseudo_density = (kernel * pseudo_amp.view(1, 1, num_atoms, 1)).sum(dim=2).squeeze(-1)

            center = scalar_density[:, 0]
            shell = scalar_density[:, 1:]
            shell_mean = shell.mean(dim=-1)
            shell_max = shell.max(dim=-1).values

            # Opposite-direction finite differences on the inner shell for gradient/curvature proxies.
            grad_x = 0.5 * (scalar_density[:, 1] - scalar_density[:, 2])
            grad_y = 0.5 * (scalar_density[:, 3] - scalar_density[:, 4])
            grad_z = 0.5 * (scalar_density[:, 5] - scalar_density[:, 6])
            grad_norm = torch.sqrt(grad_x.square() + grad_y.square() + grad_z.square() + 1.0e-8)
            curvature = (
                (scalar_density[:, 1] + scalar_density[:, 2] - 2.0 * center)
                + (scalar_density[:, 3] + scalar_density[:, 4] - 2.0 * center)
                + (scalar_density[:, 5] + scalar_density[:, 6] - 2.0 * center)
            ) / 3.0

            local_context = torch.cat(
                [
                    center.unsqueeze(-1),
                    shell_mean.unsqueeze(-1),
                    shell_max.unsqueeze(-1),
                    grad_norm.unsqueeze(-1),
                ],
                dim=-1,
            )
            siren_in = torch.cat(
                [
                    probes,
                    mv.unsqueeze(1).expand(num_atoms, q, mv.size(-1)),
                    local_context.unsqueeze(1).expand(num_atoms, q, local_context.size(-1)),
                ],
                dim=-1,
            )
            siren_density = self.coord_context(siren_in).squeeze(-1)
            siren_center = siren_density[:, 0]
            siren_shell = siren_density[:, 1:].mean(dim=-1)

            field_features = torch.stack(
                [
                    center,
                    shell_mean,
                    shell_max,
                    grad_norm,
                    curvature,
                    vector_align.mean(dim=-1),
                    bivector_density.mean(dim=-1),
                    pseudo_density.mean(dim=-1),
                    siren_center,
                    siren_shell,
                ],
                dim=-1,
            )
            global_density = scalar_density.mean(dim=-1)
            global_gap_proxy = (center - shell_mean).mean().unsqueeze(0)
            return {
                "atom_field_features": field_features,
                "global_density": global_density,
                "global_gap_proxy": global_gap_proxy,
            }

        def forward(self, atom_multivectors: torch.Tensor, atom_coords: torch.Tensor, batch_index: torch.Tensor) -> Dict[str, torch.Tensor]:
            mv = torch.as_tensor(atom_multivectors, dtype=torch.float32)
            coords = torch.as_tensor(atom_coords, dtype=torch.float32, device=mv.device)
            batch = torch.as_tensor(batch_index, dtype=torch.long, device=mv.device)
            if mv.ndim != 2 or coords.ndim != 2 or coords.size(-1) != 3:
                raise ValueError("WholeMoleculeWaveField expects [N,16] multivectors and [N,3] coordinates")
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            atom_features = torch.zeros((mv.size(0), self.field_feature_dim), device=mv.device, dtype=mv.dtype)
            global_density = torch.zeros((mv.size(0),), device=mv.device, dtype=mv.dtype)
            global_gap = torch.zeros((num_molecules,), device=mv.device, dtype=mv.dtype)
            for mol_idx in range(num_molecules):
                mask = batch == mol_idx
                if not bool(mask.any()):
                    continue
                out = self._molecule_field(mv[mask], coords[mask])
                atom_features[mask] = out["atom_field_features"].to(dtype=mv.dtype)
                global_density[mask] = out["global_density"].to(dtype=mv.dtype)
                global_gap[mol_idx] = out["global_gap_proxy"].to(dtype=mv.dtype)[0]
            return {
                "atom_field_features": atom_features,
                "global_density": global_density,
                "global_gap_proxy": global_gap,
            }
else:  # pragma: no cover
    class WholeMoleculeWaveField:  # type: ignore[override]
        field_feature_dim = 10

        def __init__(self, *args, **kwargs):
            require_torch()
