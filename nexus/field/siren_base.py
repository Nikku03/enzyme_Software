from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from nexus.physics.clifford_math import clifford_geometric_product, embed_coordinates

SIREN_OMEGA_0 = 30.0


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    return torch.sqrt(diff.square().sum(dim=-1).clamp_min(eps))


def siren_init_(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    in_dim: int,
    omega_0: float = SIREN_OMEGA_0,
    is_first: bool = False,
) -> None:
    if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
        raise ValueError(f"SIREN omega_0 must be exactly {SIREN_OMEGA_0}")
    if is_first:
        bound = 1.0 / max(int(in_dim), 1)
    else:
        bound = (6.0 / max(int(in_dim), 1)) ** 0.5 / float(omega_0)
    with torch.no_grad():
        weight.uniform_(-bound, bound)
        if bias is not None:
            bias.uniform_(-bound, bound)


class SineActivation(nn.Module):
    def __init__(self, omega_0: float = SIREN_OMEGA_0) -> None:
        super().__init__()
        if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"SineActivation requires omega_0={SIREN_OMEGA_0}")
        self.omega_0 = float(omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SineLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_0: float = SIREN_OMEGA_0,
        bias: bool = True,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"SineLayer requires omega_0={SIREN_OMEGA_0}")
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        siren_init_(
            self.weight,
            self.bias,
            in_dim=self.in_dim,
            omega_0=self.omega_0,
            is_first=self.is_first,
        )

    def forward(
        self,
        x: torch.Tensor,
        row_scale: torch.Tensor | None = None,
        col_scale: torch.Tensor | None = None,
        bias_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.weight
        if row_scale is not None:
            weight = weight * row_scale.unsqueeze(-1)
        if col_scale is not None:
            weight = weight * col_scale.unsqueeze(0)
        bias = self.bias
        if bias is not None and bias_shift is not None:
            bias = bias + bias_shift
        elif bias is None and bias_shift is not None:
            bias = bias_shift
        return torch.sin(self.omega_0 * torch.nn.functional.linear(x, weight, bias))


class SpatiallyAdaptiveSirenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_base: float = SIREN_OMEGA_0,
        bias: bool = True,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.omega_base = float(omega_base)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        siren_init_(
            self.linear.weight,
            self.linear.bias,
            in_dim=self.in_features,
            omega_0=self.omega_base,
            is_first=self.is_first,
        )
        self.freq_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Softplus(),
            nn.Linear(16, 1),
        )

    def _as_batched_coords(self, coords: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if coords.ndim == 2 and coords.size(-1) == 3:
            return coords.unsqueeze(0), True
        if coords.ndim != 3 or coords.size(-1) != 3:
            raise ValueError("Coordinates must have shape [N, 3] or [B, N, 3]")
        return coords, False

    def _match_atom_coords(self, query_coords: torch.Tensor, atomic_coords: torch.Tensor) -> torch.Tensor:
        atoms, squeezed = self._as_batched_coords(atomic_coords)
        if atoms.size(0) == 1 and query_coords.size(0) > 1:
            atoms = atoms.expand(query_coords.size(0), -1, -1)
        if atoms.size(0) != query_coords.size(0):
            raise ValueError("atomic_coords batch dimension must match query coordinates")
        return atoms

    def _apply_dynamic_frequency(
        self,
        pre_activation: torch.Tensor,
        query_coords: torch.Tensor,
        atomic_coords: torch.Tensor,
    ) -> torch.Tensor:
        query_batched, squeezed = self._as_batched_coords(query_coords)
        atom_batched = self._match_atom_coords(query_batched, atomic_coords)
        local_scale = torch.clamp(self.freq_net(query_batched), min=0.0, max=5.0)
        dist_mat = pairwise_distance(query_batched, atom_batched)
        min_dist = torch.min(dist_mat, dim=-1, keepdim=True).values
        omega = self.omega_base * (1.0 + local_scale * torch.exp(-min_dist))
        if pre_activation.ndim == 2:
            pre_activation = pre_activation.unsqueeze(0)
        activated = torch.sin(omega * pre_activation)
        return activated.squeeze(0) if squeezed else activated

    def forward(
        self,
        x: torch.Tensor,
        query_coords: torch.Tensor,
        atomic_coords: torch.Tensor,
        row_scale: torch.Tensor | None = None,
        col_scale: torch.Tensor | None = None,
        bias_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.linear.weight
        if row_scale is not None:
            weight = weight * row_scale.unsqueeze(-1)
        if col_scale is not None:
            weight = weight * col_scale.unsqueeze(0)
        bias = self.linear.bias
        if bias is not None and bias_shift is not None:
            bias = bias + bias_shift
        elif bias is None and bias_shift is not None:
            bias = bias_shift
        pre_activation = torch.nn.functional.linear(x, weight, bias)
        return self._apply_dynamic_frequency(pre_activation, query_coords, atomic_coords)


class CliffordLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features, 8) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_features, 8)) if bias else None

    def forward(
        self,
        x: torch.Tensor,
        row_scale: torch.Tensor | None = None,
        col_scale: torch.Tensor | None = None,
        bias_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.weight
        if row_scale is not None:
            weight = weight * row_scale.view(-1, 1, 1)
        if col_scale is not None:
            weight = weight * col_scale.view(1, -1, 1)
        products = clifford_geometric_product(
            x.unsqueeze(-3),
            weight.unsqueeze(0),
        )
        out = products.sum(dim=-2)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        if bias_shift is not None:
            scalar_shift = bias_shift.view(*([1] * (out.ndim - 2)), -1, 1)
            shift = torch.zeros_like(out)
            shift[..., :1] = scalar_shift
            out = out + shift
        return out


class CliffordSirenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_base: float = SIREN_OMEGA_0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.omega_base = float(omega_base)
        self.linear = CliffordLinear(in_features, out_features, bias=bias)
        self.freq_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Softplus(),
            nn.Linear(16, 1),
        )

    def _as_batched_coords(self, coords: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if coords.ndim == 2 and coords.size(-1) == 3:
            return coords.unsqueeze(0), True
        if coords.ndim != 3 or coords.size(-1) != 3:
            raise ValueError("Coordinates must have shape [N, 3] or [B, N, 3]")
        return coords, False

    def _match_atom_coords(self, query_coords: torch.Tensor, atomic_coords: torch.Tensor) -> torch.Tensor:
        atoms, _ = self._as_batched_coords(atomic_coords)
        if atoms.size(0) == 1 and query_coords.size(0) > 1:
            atoms = atoms.expand(query_coords.size(0), -1, -1)
        if atoms.size(0) != query_coords.size(0):
            raise ValueError("atomic_coords batch dimension must match query coordinates")
        return atoms

    def forward(
        self,
        x: torch.Tensor,
        query_coords: torch.Tensor,
        atomic_coords: torch.Tensor,
        row_scale: torch.Tensor | None = None,
        col_scale: torch.Tensor | None = None,
        bias_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pre_activation = self.linear(
            x,
            row_scale=row_scale,
            col_scale=col_scale,
            bias_shift=bias_shift,
        )
        query_batched, squeezed = self._as_batched_coords(query_coords)
        atom_batched = self._match_atom_coords(query_batched, atomic_coords)
        local_scale = torch.clamp(self.freq_net(query_batched), min=0.0, max=5.0)
        dist_mat = pairwise_distance(query_batched, atom_batched)
        min_dist = torch.min(dist_mat, dim=-1, keepdim=True).values
        omega = self.omega_base * (1.0 + local_scale * torch.exp(-min_dist))
        if pre_activation.ndim == 3:
            pre_activation = pre_activation.unsqueeze(0)
        activated = torch.sin(omega.unsqueeze(-1) * pre_activation)
        return activated.squeeze(0) if squeezed else activated


@dataclass
class DynamicLayerParams:
    row_scale: torch.Tensor
    col_scale: torch.Tensor
    bias_shift: torch.Tensor


class DynamicSIREN(nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 512,
        hidden_layers: int = 5,
        omega_0: float = SIREN_OMEGA_0,
    ) -> None:
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"DynamicSIREN requires omega_0={SIREN_OMEGA_0}")
        self.omega_0 = float(omega_0)
        self.omega_base = float(omega_0)

        layers: List[CliffordSirenLayer] = []
        in_dim = 1
        for idx in range(self.hidden_layers):
            layers.append(
                CliffordSirenLayer(
                    in_dim,
                    self.hidden_dim,
                    omega_base=self.omega_base,
                )
            )
            in_dim = self.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(self.hidden_dim, 1)
        siren_init_(
            self.output_layer.weight,
            self.output_layer.bias,
            in_dim=self.hidden_dim,
            omega_0=self.omega_0,
            is_first=False,
        )

    def forward(
        self,
        coords: torch.Tensor,
        atomic_coords: torch.Tensor,
        hidden_params: List[DynamicLayerParams],
        output_row_scale: torch.Tensor,
        output_col_scale: torch.Tensor,
        output_bias_shift: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        x = embed_coordinates(coords).unsqueeze(-2)
        for layer, params in zip(self.layers, hidden_params):
            x = layer(
                x,
                coords,
                atomic_coords,
                row_scale=params.row_scale,
                col_scale=params.col_scale,
                bias_shift=params.bias_shift,
            )
        latent = x
        x_scalar = latent[..., 0]
        weight = self.output_layer.weight * output_row_scale.unsqueeze(-1) * output_col_scale.unsqueeze(0)
        bias = self.output_layer.bias + output_bias_shift
        output = torch.nn.functional.linear(x_scalar, weight, bias).squeeze(-1)
        if return_latent:
            return output, latent
        return output
