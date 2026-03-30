from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from nexus.physics.clifford_math import clifford_geometric_product, embed_coordinates
from nexus.physics.pga_math import embed_point

SIREN_OMEGA_0 = 30.0


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    # eps=1e-4: 2nd derivative at boundary = 1/sqrt(eps)=100, safe for 2nd-order backward.
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
        # Guard: hypernetwork scale may be NaN; clamp/scale of NaN stays NaN in forward
        # and produces NaN grad in backward (SinBackward0 → upstream AddmmBackward0).
        weight = torch.nan_to_num(weight, nan=0.0)
        bias = self.bias
        if bias is not None and bias_shift is not None:
            bias = bias + bias_shift
        elif bias is None and bias_shift is not None:
            bias = bias_shift
        if bias is not None:
            bias = torch.nan_to_num(bias, nan=0.0)
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
        # nan_to_num after clamp: torch.clamp does NOT sanitize NaN (NaN passes through).
        local_scale = torch.nan_to_num(
            torch.clamp(self.freq_net(query_batched), min=0.0, max=5.0), nan=0.0
        )
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
        # Guard: hypernetwork scale NaN → NaN weight → NaN pre_activation →
        # sin(NaN) in CliffordSirenLayer → NaN latent → AddmmBackward0 2th output NaN.
        weight = torch.nan_to_num(weight, nan=0.0)
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
        local_scale = torch.nan_to_num(
            torch.clamp(self.freq_net(query_batched), min=0.0, max=5.0), nan=0.0
        )
        dist_mat = pairwise_distance(query_batched, atom_batched)
        min_dist = torch.min(dist_mat, dim=-1, keepdim=True).values
        omega = self.omega_base * (1.0 + local_scale * torch.exp(-min_dist))
        if pre_activation.ndim == 3:
            pre_activation = pre_activation.unsqueeze(0)
        activated = torch.sin(omega.unsqueeze(-1) * pre_activation)
        return activated.squeeze(0) if squeezed else activated


class EquivariantSineLayer(nn.Module):
    """
    Grade-preserving G(3,0,1) SIREN layer.

    Channel mixing is scalar-only, so the 16 PGA grades are preserved.
    The non-linearity acts on the multivector norm and rescales the
    orientation-preserving unit multivector.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        is_first: bool = False,
        omega_0: float = SIREN_OMEGA_0,
    ) -> None:
        super().__init__()
        self.in_channels = int(max(in_channels, 1))
        self.out_channels = int(max(out_channels, 1))
        if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"EquivariantSineLayer requires omega_0={SIREN_OMEGA_0}")
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels, 16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.is_first:
            bound = 1.0 / float(max(self.in_channels, 1))
        else:
            bound = (6.0 / float(max(self.in_channels, 1))) ** 0.5 / self.omega_0
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.bias.zero_()

    def forward(
        self,
        x_mv: torch.Tensor,
        row_scale: torch.Tensor | None = None,
        col_scale: torch.Tensor | None = None,
        bias_shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.weight
        if row_scale is not None:
            weight = weight * row_scale.unsqueeze(-1)
        if col_scale is not None:
            weight = weight * col_scale.unsqueeze(0)
        weight = torch.nan_to_num(weight, nan=0.0)
        out_mv = torch.einsum("...ci,dc->...di", x_mv, weight)

        bias = self.bias
        if bias_shift is not None:
            bias = bias.clone()
            bias[..., 0] = bias[..., 0] + bias_shift
        bias = torch.nan_to_num(bias, nan=0.0)
        out_mv = out_mv + bias

        norm = out_mv.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        phase = self.omega_0 * norm
        return torch.sin(phase) * (out_mv / norm)


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
        self.pga_dim = 16
        layers: List[EquivariantSineLayer] = []
        in_channels = 1
        for idx in range(self.hidden_layers):
            layers.append(
                EquivariantSineLayer(
                    in_channels,
                    self.hidden_dim,
                    is_first=(idx == 0),
                    omega_0=self.omega_base,
                )
            )
            in_channels = self.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.output_weight = nn.Parameter(torch.empty(1, self.hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(1, self.pga_dim))
        with torch.no_grad():
            bound = (6.0 / float(max(self.hidden_dim, 1))) ** 0.5 / self.omega_0
            self.output_weight.uniform_(-bound, bound)
            self.output_bias.zero_()

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
        x = embed_point(coords).unsqueeze(-2)
        for layer, params in zip(self.layers, hidden_params):
            x = layer(
                x,
                row_scale=params.row_scale,
                col_scale=params.col_scale,
                bias_shift=params.bias_shift,
            )
        latent = x
        weight = self.output_weight
        if output_row_scale is not None:
            weight = weight * output_row_scale.unsqueeze(-1)
        if output_col_scale is not None:
            weight = weight * output_col_scale.unsqueeze(0)
        weight = torch.nan_to_num(weight, nan=0.0)
        output = torch.einsum("...ci,dc->...di", latent, weight)
        bias = self.output_bias.clone()
        if output_bias_shift is not None:
            bias[..., 0] = bias[..., 0] + output_bias_shift
        bias = torch.nan_to_num(bias, nan=0.0)
        output = output + bias
        output = output.squeeze(-2)
        if return_latent:
            return output, latent.squeeze(-2) if latent.size(-2) == 1 else latent
        return output
