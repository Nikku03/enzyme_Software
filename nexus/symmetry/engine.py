from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .irreps import DEFAULT_IRREPS, parse_irreps
from .parity import extract_parity_bundle


def _symmetric_traceless_coefficients(tensor: torch.Tensor) -> torch.Tensor:
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    zz = tensor[..., 2, 2]
    xy = tensor[..., 0, 1]
    xz = tensor[..., 0, 2]
    yz = tensor[..., 1, 2]
    c0 = (xx - yy) / torch.sqrt(torch.tensor(2.0, dtype=tensor.dtype, device=tensor.device))
    c1 = (-xx - yy + 2.0 * zz) / torch.sqrt(torch.tensor(6.0, dtype=tensor.dtype, device=tensor.device))
    c2 = torch.sqrt(torch.tensor(2.0, dtype=tensor.dtype, device=tensor.device)) * xy
    c3 = torch.sqrt(torch.tensor(2.0, dtype=tensor.dtype, device=tensor.device)) * xz
    c4 = torch.sqrt(torch.tensor(2.0, dtype=tensor.dtype, device=tensor.device)) * yz
    return torch.stack([c0, c1, c2, c3, c4], dim=-1)


def _basis_l2(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    s2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    s6 = torch.sqrt(torch.tensor(6.0, dtype=dtype, device=device))
    basis = torch.zeros((5, 3, 3), dtype=dtype, device=device)
    basis[0, 0, 0] = 1.0 / s2
    basis[0, 1, 1] = -1.0 / s2
    basis[1, 0, 0] = -1.0 / s6
    basis[1, 1, 1] = -1.0 / s6
    basis[1, 2, 2] = 2.0 / s6
    basis[2, 0, 1] = basis[2, 1, 0] = 1.0 / s2
    basis[3, 0, 2] = basis[3, 2, 0] = 1.0 / s2
    basis[4, 1, 2] = basis[4, 2, 1] = 1.0 / s2
    return basis


class O3_Symmetry_Engine(nn.Module):
    def __init__(self, irreps_spec: str = DEFAULT_IRREPS) -> None:
        super().__init__()
        self.irreps = parse_irreps(irreps_spec)
        self.max_degree = max(irrep.degree for irrep in self.irreps)
        if self.max_degree < 2:
            raise ValueError("O3_Symmetry_Engine must support L=2")

        self.scalar_even_proj = nn.Sequential(nn.Linear(5, 128), nn.SiLU(), nn.Linear(128, 128))
        self.scalar_odd_proj = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128))
        self.vector_odd_gate = nn.Sequential(nn.Linear(5, 128), nn.SiLU(), nn.Linear(128, 128))
        self.vector_even_gate = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128))
        self.tensor_even_gate = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64))
        self.tensor_odd_gate = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))

    def extract_parity_features(self, manifold):
        return extract_parity_bundle(manifold)

    def _rotation_matrix_from_axis_angle(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        # norm(dim=-1) normalises each axis independently; keepdim broadcasts correctly
        # for both unbatched [3] and batched [..., 3] inputs.
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        x = axis[..., 0]
        y = axis[..., 1]
        z = axis[..., 2]
        c = torch.cos(torch.tensor(float(angle), dtype=axis.dtype, device=axis.device))
        s = torch.sin(torch.tensor(float(angle), dtype=axis.dtype, device=axis.device))
        C = 1.0 - c
        # dim=-1 on inner stacks: [B] components → [..., 3] rows
        # dim=-2 on outer stack:  [..., 3] rows   → [..., 3, 3] matrix
        return torch.stack(
            [
                torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], dim=-1),
                torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], dim=-1),
                torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1),
            ],
            dim=-2,
        )

    def wigner_d_matrix(self, degree: int, rotation_matrix: torch.Tensor) -> torch.Tensor:
        if degree == 0:
            return torch.ones((1, 1), dtype=rotation_matrix.dtype, device=rotation_matrix.device)
        if degree == 1:
            return rotation_matrix
        if degree != 2:
            raise ValueError(f"Wigner-D only implemented up to l=2, got l={degree}")
        basis = _basis_l2(rotation_matrix.dtype, rotation_matrix.device)
        transformed = torch.einsum("ab,ibc,dc->iad", rotation_matrix, basis, rotation_matrix)
        return torch.einsum("jab,iab->ji", basis, transformed)

    def apply_wigner_d(self, features: Dict[str, torch.Tensor], rotation_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        rotated = dict(features)
        for key, value in list(features.items()):
            if key.startswith("1"):
                d1 = self.wigner_d_matrix(1, rotation_matrix)
                rotated[key] = torch.einsum("ab,nmb->nma", d1, value)
            elif key.startswith("2"):
                d2 = self.wigner_d_matrix(2, rotation_matrix)
                rotated[key] = torch.einsum("ab,nmb->nma", d2, value)
        return rotated

    def forward(self, manifold) -> Dict[str, torch.Tensor]:
        pos = manifold.pos
        species = manifold.species.to(device=pos.device, dtype=pos.dtype)
        forces = manifold.forces.to(device=pos.device, dtype=pos.dtype)
        energy = manifold.energy.to(device=pos.device, dtype=pos.dtype)

        parity = self.extract_parity_features(manifold)
        centroid = pos.mean(dim=0, keepdim=True)
        rel = pos - centroid
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        rel_unit = rel / rel_norm
        force_norm = forces.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        force_unit = forces / force_norm

        pseudo_scalar = parity.pseudo_scalar_per_atom.unsqueeze(-1)
        pseudo_vector = parity.pseudo_vector_per_atom
        seed_chirality = getattr(getattr(manifold, "seed", None), "chirality_codes", None)
        if seed_chirality is not None:
            chirality_sign = seed_chirality.to(device=pos.device, dtype=pos.dtype).sign().unsqueeze(-1)
            fallback_mask = (pseudo_scalar.abs() < 1.0e-6) & (chirality_sign.abs() > 0)
            pseudo_scalar = torch.where(fallback_mask, 0.25 * chirality_sign, pseudo_scalar)
            fallback_pseudovector = torch.cross(rel_unit, force_unit, dim=-1)
            fallback_pseudovector = torch.where(
                fallback_pseudovector.norm(dim=-1, keepdim=True) < 1.0e-6,
                torch.cross(rel_unit, torch.tensor([1.0, 0.0, 0.0], dtype=pos.dtype, device=pos.device).view(1, 3), dim=-1),
                fallback_pseudovector,
            )
            pseudo_vector = torch.where(fallback_mask.expand_as(pseudo_vector), chirality_sign * fallback_pseudovector, pseudo_vector)
        pseudo_norm = pseudo_vector.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        pseudo_unit = pseudo_vector / pseudo_norm

        scalar_even_input = torch.cat(
            [
                species.unsqueeze(-1) / 60.0,
                rel_norm,
                force_norm,
                energy.reshape(1, 1).expand_as(rel_norm),
                parity.center_mask.to(dtype=pos.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_odd_input = torch.cat(
            [
                pseudo_scalar,
                pseudo_norm,
            ],
            dim=-1,
        )

        vector_odd_weights = self.vector_odd_gate(scalar_even_input)
        vector_even_weights = self.vector_even_gate(torch.cat([pseudo_scalar, pseudo_norm, parity.center_mask.to(dtype=pos.dtype).unsqueeze(-1)], dim=-1))

        quadrupole = rel.unsqueeze(-1) * rel.unsqueeze(-2)
        trace = torch.diagonal(quadrupole, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        eye = torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0)
        quadrupole = quadrupole - trace.unsqueeze(-1).unsqueeze(-1) * eye
        quadrupole_coeff = _symmetric_traceless_coefficients(quadrupole)

        pseudo_tensor = rel.unsqueeze(-1) * pseudo_vector.unsqueeze(-2) + pseudo_vector.unsqueeze(-1) * rel.unsqueeze(-2)
        pseudo_trace = torch.diagonal(pseudo_tensor, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        pseudo_tensor = pseudo_tensor - pseudo_trace.unsqueeze(-1).unsqueeze(-1) * eye
        pseudo_tensor_coeff = _symmetric_traceless_coefficients(pseudo_tensor)

        tensor_even_weights = self.tensor_even_gate(scalar_even_input)
        tensor_odd_weights = self.tensor_odd_gate(torch.cat([pseudo_scalar, pseudo_norm, parity.center_mask.to(dtype=pos.dtype).unsqueeze(-1)], dim=-1))

        features = {
            "0e": self.scalar_even_proj(scalar_even_input),
            "0o": self.scalar_odd_proj(scalar_odd_input),
            "1o": vector_odd_weights.unsqueeze(-1) * rel_unit.unsqueeze(1),
            "1e": vector_even_weights.unsqueeze(-1) * pseudo_unit.unsqueeze(1),
            "2e": tensor_even_weights.unsqueeze(-1) * quadrupole_coeff.unsqueeze(1),
            "2o": tensor_odd_weights.unsqueeze(-1) * pseudo_tensor_coeff.unsqueeze(1),
            "parity_pseudoscalar": parity.pseudo_scalar_per_atom,
            "parity_pseudovector": parity.pseudo_vector_per_atom,
            "chiral_center_mask": parity.center_mask,
        }
        return features
