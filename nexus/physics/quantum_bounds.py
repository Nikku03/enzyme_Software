from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn


SPEED_OF_LIGHT_AU = 137.036


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    # eps=1e-4 (0.01 Å minimum): 2nd derivative of sqrt at boundary = 1/sqrt(eps)=100,
    # safe for the Hessian's create_graph=True inner backward.  1e-12 gave 2.5e17,
    # which chains to overflow via the heme-exclusion and cusp-envelope paths.
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    return torch.sqrt(diff.square().sum(dim=-1).clamp_min(eps))


@dataclass
class QuantumBoundingBox:
    minimum: torch.Tensor
    maximum: torch.Tensor


class HohenbergKohn_Field_Enforcer(nn.Module):
    def __init__(
        self,
        integration_resolution: int = 16,
        box_padding: float = 2.5,
        zora_c_au: float = SPEED_OF_LIGHT_AU,
        integration_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.integration_resolution = int(integration_resolution)
        self.box_padding = float(box_padding)
        self.zora_c_au = float(zora_c_au)
        self.integration_chunk_size = int(integration_chunk_size)

    def _as_batched_points(self, tensor: torch.Tensor, point_dim: int) -> tuple[torch.Tensor, bool]:
        if tensor.ndim == point_dim - 1:
            return tensor.unsqueeze(0), True
        if tensor.ndim != point_dim:
            raise ValueError(f"Expected tensor with ndim {point_dim - 1} or {point_dim}, got {tensor.ndim}")
        return tensor, False

    def compute_total_electrons(
        self,
        atomic_numbers: torch.Tensor,
        molecular_charge: torch.Tensor | float | int = 0,
    ) -> torch.Tensor:
        z_batched, squeezed = self._as_batched_points(atomic_numbers, 2)
        charge = torch.as_tensor(molecular_charge, dtype=z_batched.dtype, device=z_batched.device)
        if charge.ndim == 0:
            charge = charge.expand(z_batched.size(0))
        total = z_batched.sum(dim=-1) - charge
        return total[0] if squeezed else total

    def make_bounding_box(self, nuclear_coords: torch.Tensor) -> QuantumBoundingBox:
        coords_batched, squeezed = self._as_batched_points(nuclear_coords, 3)
        mins = coords_batched.min(dim=1).values - self.box_padding
        maxs = coords_batched.max(dim=1).values + self.box_padding
        if squeezed:
            mins = mins[0]
            maxs = maxs[0]
        return QuantumBoundingBox(minimum=mins, maximum=maxs)

    def apply_cusp_envelope(
        self,
        raw_siren_out: torch.Tensor,
        r: torch.Tensor,
        R_A: torch.Tensor,
        Z_A: torch.Tensor,
    ) -> torch.Tensor:
        raw_batched, raw_squeezed = self._as_batched_points(raw_siren_out, 3)
        r_batched, _ = self._as_batched_points(r, 3)
        R_batched, _ = self._as_batched_points(R_A, 3)
        Z_batched, _ = self._as_batched_points(Z_A, 2)

        if raw_batched.size(-1) != 1:
            raw_batched = raw_batched.unsqueeze(-1)
        R_batched = R_batched.to(dtype=r_batched.dtype, device=r_batched.device)
        Z_batched = Z_batched.to(dtype=r_batched.dtype, device=r_batched.device)
        raw_batched = raw_batched.to(dtype=r_batched.dtype, device=r_batched.device)
        # Guard: NaN SIREN output corrupts the 2nd-order Hessian backward via
        # MulBackward0 1th output (grad × raw_batched²). nan_to_num here (not
        # just in _objective_values after field.query) ensures raw_batched² is
        # always finite inside apply_cusp_envelope's computation graph.
        raw_batched = torch.nan_to_num(raw_batched, nan=0.0, posinf=20.0, neginf=-20.0)
        dist_matrix = pairwise_distance(r_batched, R_batched)
        scaled_dist = dist_matrix * Z_batched.unsqueeze(1)
        envelope = torch.exp(-torch.sum(scaled_dist, dim=-1, keepdim=True))
        rho = raw_batched.pow(2) * envelope
        return rho[0] if raw_squeezed else rho

    def _build_integration_grid(
        self,
        bounding_box: QuantumBoundingBox,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mins, mins_squeezed = self._as_batched_points(bounding_box.minimum, 2)
        maxs, _ = self._as_batched_points(bounding_box.maximum, 2)
        grids = []
        dvols = []
        for batch_idx in range(mins.size(0)):
            axes = []
            spacing = []
            for axis in range(3):
                start = mins[batch_idx, axis].to(device=device, dtype=dtype)
                stop = maxs[batch_idx, axis].to(device=device, dtype=dtype)
                axis_values = torch.linspace(
                    start,
                    stop,
                    self.integration_resolution,
                    device=device,
                    dtype=dtype,
                )
                axes.append(axis_values)
                if self.integration_resolution > 1:
                    spacing.append((stop - start) / float(self.integration_resolution - 1))
                else:
                    spacing.append(torch.tensor(1.0, device=device, dtype=dtype))
            mesh = torch.meshgrid(*axes, indexing="ij")
            grid = torch.stack(mesh, dim=-1).reshape(-1, 3)
            dV = spacing[0] * spacing[1] * spacing[2]
            grids.append(grid)
            dvols.append(dV)
        grid_tensor = torch.stack(grids, dim=0)
        dvol_tensor = torch.stack(dvols, dim=0)
        if mins_squeezed:
            return grid_tensor[0], dvol_tensor[0]
        return grid_tensor, dvol_tensor

    def apply_n_electron_constraint(
        self,
        unnormalized_rho: torch.Tensor,
        total_electrons: torch.Tensor,
        norm_factor: torch.Tensor,
    ) -> torch.Tensor:
        rho_batched, rho_squeezed = self._as_batched_points(unnormalized_rho, 3)
        electrons = torch.as_tensor(total_electrons, dtype=rho_batched.dtype, device=rho_batched.device)
        if electrons.ndim == 0:
            electrons = electrons.expand(rho_batched.size(0))
        factors = torch.as_tensor(norm_factor, dtype=rho_batched.dtype, device=rho_batched.device)
        if factors.ndim == 0:
            factors = factors.expand(rho_batched.size(0))
        constrained = rho_batched * factors.view(-1, 1, 1)
        return constrained[0] if rho_squeezed else constrained

    def enforce_electron_conservation(
        self,
        field,
        total_electrons: torch.Tensor,
        squeezed: bool = True,
    ) -> torch.Tensor:
        """
        Analytical electron-conservation approximation for the splatter field.

        This bypasses the dense 3D integration grid by integrating the atomic
        Gaussian basis in closed form. The result is an O(N_atoms) estimate of
        the normalization factor instead of an O(grid^3) volumetric quadrature.
        """
        state = field.splatter_state
        tensor_splatter = field.engine.tensor_splatter

        alpha = state.alpha.to(dtype=state.even_scalar.dtype, device=state.even_scalar.device).clamp_min(1.0e-6)
        gaussian_vol = (torch.as_tensor(math.pi, dtype=alpha.dtype, device=alpha.device) / alpha).pow(1.5)

        c0e = tensor_splatter.readout_even0(state.even_scalar).squeeze(-1)
        c0e = torch.nan_to_num(c0e, nan=0.0, posinf=0.0, neginf=0.0)
        integral_val = (c0e.abs() * gaussian_vol).sum(dim=-1)

        electrons = torch.as_tensor(total_electrons, dtype=integral_val.dtype, device=integral_val.device)
        if electrons.ndim == 0:
            electrons = electrons.expand_as(integral_val)
        factors = electrons / integral_val.clamp_min(1.0e-8)
        return factors[0] if squeezed and factors.ndim > 0 and factors.numel() == 1 else factors

    def compute_normalization_factor(
        self,
        raw_field_fn: Callable[[torch.Tensor], torch.Tensor],
        nuclear_coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_electrons: torch.Tensor,
        bounding_box: QuantumBoundingBox,
    ) -> torch.Tensor:
        coords_batched, squeezed = self._as_batched_points(nuclear_coords, 3)
        z_batched, _ = self._as_batched_points(atomic_numbers, 2)
        electrons = torch.as_tensor(total_electrons, dtype=coords_batched.dtype, device=coords_batched.device)
        if electrons.ndim == 0:
            electrons = electrons.expand(coords_batched.size(0))
        integration_grid, dV = self._build_integration_grid(
            bounding_box,
            device=coords_batched.device,
            dtype=coords_batched.dtype,
        )
        grid_batched, _ = self._as_batched_points(integration_grid, 3)
        rho_chunks = []
        chunk_size = max(int(self.integration_chunk_size), 1)
        for start in range(0, grid_batched.size(1), chunk_size):
            stop = min(start + chunk_size, grid_batched.size(1))
            grid_chunk = grid_batched[:, start:stop]
            raw_chunk = raw_field_fn(grid_chunk)
            # Guard: SIREN output is NaN when hypernetwork conditioning carries
            # NaN for this molecule.  Replace with 0 so those grid points
            # contribute nothing to the integral instead of corrupting it.
            raw_chunk = torch.nan_to_num(raw_chunk, nan=0.0)
            rho_chunk = self.apply_cusp_envelope(raw_chunk, grid_chunk, coords_batched, z_batched)
            rho_chunks.append(rho_chunk)
        rho_grid = torch.cat(rho_chunks, dim=1)
        rho_batched, _ = self._as_batched_points(rho_grid, 3)
        dV = torch.as_tensor(dV, dtype=rho_batched.dtype, device=rho_batched.device)
        if dV.ndim == 0:
            dV = dV.expand(rho_batched.size(0))
        integral_val = rho_batched.sum(dim=1).squeeze(-1) * dV
        factors = electrons / integral_val.clamp_min(1.0e-8)
        return factors[0] if squeezed else factors

    def compute_zora_correction(
        self,
        r: torch.Tensor,
        R_A: torch.Tensor,
        Z_A: torch.Tensor,
    ) -> torch.Tensor:
        r_batched, r_squeezed = self._as_batched_points(r, 3)
        R_batched, _ = self._as_batched_points(R_A, 3)
        Z_batched, _ = self._as_batched_points(Z_A, 2)
        R_batched = R_batched.to(dtype=r_batched.dtype, device=r_batched.device)
        Z_batched = Z_batched.to(dtype=r_batched.dtype, device=r_batched.device)
        dist = pairwise_distance(r_batched, R_batched).clamp_min(1.0e-6)
        nuclear_potential = -(Z_batched.unsqueeze(1) / dist).sum(dim=-1, keepdim=True)
        denom = 1.0 - nuclear_potential / (2.0 * (self.zora_c_au ** 2))
        zora = denom.clamp_min(1.0e-8).reciprocal()
        return zora[0] if r_squeezed else zora


__all__ = ["HohenbergKohn_Field_Enforcer", "QuantumBoundingBox", "SPEED_OF_LIGHT_AU"]
