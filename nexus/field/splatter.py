from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .kernels import GaussianKernelState, LearnedGaussianSplatKernel

try:
    from e3nn import o3 as _e3nn_o3
except Exception:
    _e3nn_o3 = None


def _symmetric_traceless_coefficients(tensor: torch.Tensor) -> torch.Tensor:
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    zz = tensor[..., 2, 2]
    xy = tensor[..., 0, 1]
    xz = tensor[..., 0, 2]
    yz = tensor[..., 1, 2]
    two = torch.tensor(2.0, dtype=tensor.dtype, device=tensor.device)
    six = torch.tensor(6.0, dtype=tensor.dtype, device=tensor.device)
    c0 = (xx - yy) / torch.sqrt(two)
    c1 = (-xx - yy + 2.0 * zz) / torch.sqrt(six)
    c2 = torch.sqrt(two) * xy
    c3 = torch.sqrt(two) * xz
    c4 = torch.sqrt(two) * yz
    return torch.stack([c0, c1, c2, c3, c4], dim=-1)


def _reconstruct_quadrupole(coefficients: torch.Tensor) -> torch.Tensor:
    c0, c1, c2, c3, c4 = coefficients.unbind(dim=-1)
    sq2 = torch.sqrt(torch.tensor(2.0, dtype=coefficients.dtype, device=coefficients.device))
    sq2_3 = torch.sqrt(torch.tensor(2.0 / 3.0, dtype=coefficients.dtype, device=coefficients.device))

    zz = c1 * sq2_3
    xx = 0.5 * (c0 * sq2 - zz)
    yy = 0.5 * (-c0 * sq2 - zz)
    xy = c2 / sq2
    xz = c3 / sq2
    yz = c4 / sq2

    return torch.stack(
        [
            xx, xy, xz,
            xy, yy, yz,
            xz, yz, zz,
        ],
        dim=-1,
    ).reshape(*coefficients.shape[:-1], 3, 3)


def _fallback_real_spherical_harmonics(direction: torch.Tensor, degree: int) -> torch.Tensor:
    if degree == 0:
        return torch.ones(*direction.shape[:-1], 1, dtype=direction.dtype, device=direction.device)
    if degree == 1:
        return direction
    if degree == 2:
        return _symmetric_traceless_coefficients(direction.unsqueeze(-1) * direction.unsqueeze(-2))
    raise ValueError(f"Unsupported spherical harmonic degree: {degree}")


def _real_spherical_harmonics(direction: torch.Tensor, degree: int) -> torch.Tensor:
    norm = direction.norm(dim=-1, keepdim=True)
    safe_direction = torch.where(
        norm > 1.0e-6,
        direction / norm.clamp_min(1.0e-6),
        torch.zeros_like(direction),
    )
    if _e3nn_o3 is not None:
        try:
            output = _e3nn_o3.spherical_harmonics(
                list(range(degree, degree + 1)),
                safe_direction,
                normalize=True,
                normalization="component",
            )
            return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
    return _fallback_real_spherical_harmonics(safe_direction, degree)


@dataclass
class TensorSplatterState:
    atom_coords: torch.Tensor
    species: torch.Tensor
    even_scalar: torch.Tensor
    odd_scalar: torch.Tensor
    odd_vector: torch.Tensor
    even_vector: torch.Tensor
    even_tensor: torch.Tensor
    odd_tensor: torch.Tensor
    alpha: torch.Tensor
    alpha_target: torch.Tensor


class TensorSplatter(nn.Module):
    def __init__(self, reconstruction_dim: int = 64, init_alpha: float = 0.35) -> None:
        super().__init__()
        self.reconstruction_dim = int(reconstruction_dim)
        self.radial_kernel = LearnedGaussianSplatKernel(init_alpha=init_alpha)
        radii = torch.ones(128, dtype=torch.float32) * 1.70
        radii[1] = 1.20
        radii[6] = 1.70
        radii[7] = 1.55
        radii[8] = 1.52
        radii[9] = 1.47
        radii[15] = 1.80
        radii[16] = 1.80
        radii[17] = 1.75
        radii[35] = 1.85
        radii[53] = 1.98
        self.register_buffer("covalent_radii", radii)
        self.alpha_species_bias = nn.Embedding(128, 1)
        nn.init.zeros_(self.alpha_species_bias.weight)

        self.proj_0e = nn.Linear(128, reconstruction_dim)
        self.proj_0o = nn.Linear(128, reconstruction_dim)
        self.proj_1o = nn.Linear(128, reconstruction_dim)
        self.proj_1e = nn.Linear(128, reconstruction_dim)
        self.proj_2e = nn.Linear(64, reconstruction_dim)
        self.proj_2o = nn.Linear(64, reconstruction_dim)

        self.readout_even0 = nn.Linear(reconstruction_dim, 1)
        self.readout_odd0 = nn.Linear(reconstruction_dim, 1)
        self.readout_even1 = nn.Linear(reconstruction_dim, 1)
        self.readout_odd1 = nn.Linear(reconstruction_dim, 1)
        self.readout_even2 = nn.Linear(reconstruction_dim, 1)
        self.readout_odd2 = nn.Linear(reconstruction_dim, 1)
        self.readout_vector_odd = nn.Linear(reconstruction_dim, 1)
        self.readout_vector_even = nn.Linear(reconstruction_dim, 1)
        self.readout_tensor_even = nn.Linear(reconstruction_dim, 1)
        self.readout_tensor_odd = nn.Linear(reconstruction_dim, 1)

    def _project_vector(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        projected = linear(x.transpose(1, 2))
        return projected.transpose(1, 2)

    def _project_tensor(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        projected = linear(x.transpose(1, 2))
        return projected.transpose(1, 2)

    def _readout_vector(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        return linear(x.transpose(1, 2)).squeeze(-1)

    def _readout_tensor(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        return linear(x.transpose(1, 2)).squeeze(-1)

    def _alpha_targets(self, species: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        base_alpha = self.radial_kernel.alpha.to(device=device, dtype=dtype)
        radii = self.covalent_radii.to(device=device, dtype=dtype)[species.to(dtype=torch.long).clamp(min=0, max=127)]
        radius_scale = (1.70 / radii.clamp_min(0.5)).clamp(0.5, 2.0)
        learned = torch.nn.functional.softplus(
            self.alpha_species_bias(species.to(dtype=torch.long).clamp(min=0, max=127)).squeeze(-1)
        )
        learned = learned / torch.nn.functional.softplus(torch.zeros((), device=device, dtype=dtype))
        return (base_alpha * radius_scale * learned).clamp_min(1.0e-4)

    def prepare(self, features: Dict[str, torch.Tensor], atom_coords: torch.Tensor, species: torch.Tensor) -> TensorSplatterState:
        alpha_target = self._alpha_targets(species, atom_coords.dtype, atom_coords.device)
        return TensorSplatterState(
            atom_coords=atom_coords,
            species=species,
            even_scalar=self.proj_0e(features["0e"]),
            odd_scalar=self.proj_0o(features["0o"]),
            odd_vector=self._project_vector(features["1o"], self.proj_1o),
            even_vector=self._project_vector(features["1e"], self.proj_1e),
            even_tensor=self._project_tensor(features["2e"], self.proj_2e),
            odd_tensor=self._project_tensor(features["2o"], self.proj_2o),
            alpha=alpha_target,
            alpha_target=alpha_target,
        )

    def forward(self, query_coords: torch.Tensor, state: TensorSplatterState) -> Dict[str, torch.Tensor]:
        kernel = self.radial_kernel(query_coords, state.atom_coords, alpha_override=state.alpha.unsqueeze(0))
        rel = query_coords.unsqueeze(1) - state.atom_coords.unsqueeze(0)
        dist = kernel.distances.clamp_min(1.0e-8)
        direction = torch.where(
            kernel.distances.unsqueeze(-1) > 1.0e-6,
            rel / dist.unsqueeze(-1),
            torch.zeros_like(rel),
        )
        y1 = _real_spherical_harmonics(direction, degree=1)
        y2 = _real_spherical_harmonics(direction, degree=2)
        weights = kernel.weights
        normalized_weights = kernel.normalized_weights
        weight_sum = kernel.weight_sum

        c0e = self.readout_even0(state.even_scalar).squeeze(-1)
        c0o = self.readout_odd0(state.odd_scalar).squeeze(-1)
        dipole_coeff = self._readout_vector(state.odd_vector, self.readout_vector_odd)
        bivector_coeff = self._readout_vector(state.even_vector, self.readout_vector_even)
        quadrupole_coeff = self._readout_tensor(state.even_tensor, self.readout_tensor_even)
        odd_quadrupole_coeff = self._readout_tensor(state.odd_tensor, self.readout_tensor_odd)

        rel_unit = torch.where(
            dist.unsqueeze(-1) > 1.0e-6,
            rel / dist.unsqueeze(-1),
            torch.zeros_like(rel),
        )
        quadrupole = _reconstruct_quadrupole(quadrupole_coeff)
        odd_quadrupole = _reconstruct_quadrupole(odd_quadrupole_coeff)
        quad_vector = torch.einsum("aij,qaj->qai", quadrupole, rel_unit)
        odd_quad_vector = torch.einsum("aij,qaj->qai", odd_quadrupole, rel_unit)

        scalar_field = torch.einsum("qa,a->q", normalized_weights, c0e)
        pseudoscalar_field = torch.einsum("qa,a->q", normalized_weights, c0o)
        dipole_field = torch.einsum("qa,ai->qi", normalized_weights, dipole_coeff)
        quadrupole_field = (normalized_weights.unsqueeze(-1) * quad_vector).sum(dim=1)
        vector_field = dipole_field + quadrupole_field

        bivector_seed = torch.einsum("qa,ai->qi", normalized_weights, bivector_coeff)
        odd_quadrupole_field = (normalized_weights.unsqueeze(-1) * odd_quad_vector).sum(dim=1)
        bivector_field = bivector_seed + odd_quadrupole_field

        angular_1o = self.readout_even1((state.odd_vector.unsqueeze(0) * y1.unsqueeze(2)).sum(dim=-1)).squeeze(-1)
        angular_1e = self.readout_odd1((state.even_vector.unsqueeze(0) * y1.unsqueeze(2)).sum(dim=-1)).squeeze(-1)
        angular_2e = self.readout_even2((state.even_tensor.unsqueeze(0) * y2.unsqueeze(2)).sum(dim=-1)).squeeze(-1)
        angular_2o = self.readout_odd2((state.odd_tensor.unsqueeze(0) * y2.unsqueeze(2)).sum(dim=-1)).squeeze(-1)
        even_angular = torch.einsum("qa,qa->q", normalized_weights, angular_1o + angular_2e)
        odd_angular = torch.einsum("qa,qa->q", normalized_weights, angular_1e + angular_2o)

        multivector = torch.zeros(query_coords.size(0), 16, dtype=query_coords.dtype, device=query_coords.device)
        multivector[..., 0] = scalar_field
        multivector[..., 1:4] = vector_field
        multivector[..., 5] = bivector_field[..., 2]
        multivector[..., 6] = -bivector_field[..., 1]
        multivector[..., 7] = bivector_field[..., 0]
        multivector[..., 11] = pseudoscalar_field

        even_contrib = scalar_field + even_angular
        odd_contrib = pseudoscalar_field + odd_angular
        total = (
            even_contrib
            + odd_contrib
            + 0.10 * vector_field.norm(dim=-1)
            + 0.05 * bivector_field.norm(dim=-1)
        )
        return {
            "total": total,
            "even": even_contrib,
            "odd": odd_contrib,
            "multivector": multivector,
            "scalar_field": scalar_field,
            "vector_field": vector_field,
            "bivector_field": bivector_field,
            "pseudoscalar_field": pseudoscalar_field,
            "quadrupole_field": quadrupole_field,
            "odd_quadrupole_field": odd_quadrupole_field,
            "weights": weights,
            "normalized_weights": normalized_weights,
            "weight_sum": weight_sum.squeeze(-1),
            "alpha": kernel.alpha,
            "distance": kernel.distances,
            "y1": y1,
            "y2": y2,
        }
