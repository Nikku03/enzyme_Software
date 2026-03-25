from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def fibonacci_sphere(
    n_points: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    idx = torch.arange(n_points, device=device, dtype=dtype)
    offset = torch.tensor(2.0 / n_points, dtype=dtype, device=device)
    increment = torch.tensor(math.pi * (3.0 - math.sqrt(5.0)), dtype=dtype, device=device)
    y = ((idx * offset) - 1.0) + (offset * 0.5)
    r = torch.sqrt((1.0 - y.pow(2)).clamp_min(0.0))
    phi = idx * increment
    x = torch.cos(phi) * r
    z = torch.sin(phi) * r
    return torch.stack([x, y, z], dim=-1)


def generate_shell_grid(
    centers: torch.Tensor,
    radius: float = 2.5,
    n_points: int = 96,
    shell_fractions: Optional[Sequence[float]] = None,
    bias_directions: Optional[torch.Tensor] = None,
    bias_strength: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if centers.ndim != 2 or centers.size(-1) != 3:
        raise ValueError("centers must have shape [N, 3]")

    if shell_fractions is None:
        shell_fractions = (0.35, 0.55, 0.75, 0.90, 1.00)

    base_dirs = fibonacci_sphere(n_points, device=centers.device, dtype=centers.dtype)
    if bias_directions is not None:
        if bias_directions.shape != centers.shape:
            raise ValueError("bias_directions must match centers shape [N, 3]")
        unit_bias = bias_directions / bias_directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        base_dirs = base_dirs.unsqueeze(0).expand(centers.size(0), -1, -1)
        biased = base_dirs + float(bias_strength) * unit_bias.unsqueeze(1)
        base_dirs = biased / biased.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    else:
        base_dirs = base_dirs.unsqueeze(0).expand(centers.size(0), -1, -1)
    shell_radii = torch.as_tensor(shell_fractions, device=centers.device, dtype=centers.dtype) * float(radius)
    points = []
    radii = []
    for shell_radius in shell_radii:
        points.append(centers.unsqueeze(1) + shell_radius * base_dirs)
        radii.append(torch.full((centers.size(0), n_points), shell_radius, device=centers.device, dtype=centers.dtype))
    return torch.cat(points, dim=1), torch.cat(radii, dim=1)


def smooth_steric_mask(
    query_points: torch.Tensor,
    atom_positions: torch.Tensor,
    center_indices: torch.Tensor,
    min_clearance: float = 1.15,
    softness: float = 10.0,
    nudge_vectors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if query_points.ndim != 3 or query_points.size(-1) != 3:
        raise ValueError("query_points must have shape [N, P, 3]")
    if atom_positions.ndim != 2 or atom_positions.size(-1) != 3:
        raise ValueError("atom_positions must have shape [M, 3]")

    rel = query_points.unsqueeze(2) - atom_positions.unsqueeze(0).unsqueeze(0)
    # eps-under-sqrt: when the query point coincides with a non-center atom,
    # norm(0) has undefined gradient (1/0) and undefined 2nd derivative (1/0³).
    # Both blow up in compute_peak_curvature's Hessian (create_graph=True path).
    # Adding eps=1e-4 bounds ∂²/∂x² to 1/sqrt(1e-4)=100 — safe for 2nd-order.
    # (clamp_min on the norm value does NOT fix 2nd-order: 0*NaN=NaN in norm.backward)
    dist = (rel.pow(2).sum(dim=-1) + 1e-4).sqrt()
    atom_mask = 1.0 - torch.nn.functional.one_hot(
        center_indices.to(dtype=torch.long),
        num_classes=atom_positions.size(0),
    ).to(dtype=dist.dtype).unsqueeze(1)
    large = torch.full_like(dist, 1.0e6)
    dist_other = torch.where(atom_mask > 0, dist, large)
    nearest_other = dist_other.min(dim=-1).values
    clearance_mask = torch.sigmoid((nearest_other - float(min_clearance)) * float(softness))

    if nudge_vectors is None:
        return clearance_mask

    nudge_norm = nudge_vectors.norm(dim=-1, keepdim=True)
    safe_nudge = torch.where(
        nudge_norm > 1.0e-6,
        nudge_vectors / nudge_norm.clamp_min(1.0e-6),
        torch.zeros_like(nudge_vectors),
    )
    direction = query_points - atom_positions[center_indices].unsqueeze(1)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    alignment = (direction * safe_nudge.unsqueeze(1)).sum(dim=-1)
    nudge_mask = torch.where(
        nudge_norm.squeeze(-1).unsqueeze(-1) > 1.0e-6,
        torch.sigmoid(4.0 * alignment),
        torch.ones_like(alignment),
    )
    return clearance_mask * nudge_mask


def symmetric_traceless_direction_coefficients(direction: torch.Tensor) -> torch.Tensor:
    if direction.ndim < 1 or direction.size(-1) != 3:
        raise ValueError("direction must end with dimension 3")
    unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    tensor = unit.unsqueeze(-1) * unit.unsqueeze(-2)
    eye = torch.eye(3, dtype=unit.dtype, device=unit.device)
    trace = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
    tensor = tensor - trace.unsqueeze(-1) * eye
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    zz = tensor[..., 2, 2]
    xy = tensor[..., 0, 1]
    xz = tensor[..., 0, 2]
    yz = tensor[..., 1, 2]
    s2 = torch.sqrt(torch.tensor(2.0, dtype=unit.dtype, device=unit.device))
    s6 = torch.sqrt(torch.tensor(6.0, dtype=unit.dtype, device=unit.device))
    return torch.stack(
        [
            (xx - yy) / s2,
            (-xx - yy + 2.0 * zz) / s6,
            s2 * xy,
            s2 * xz,
            s2 * yz,
        ],
        dim=-1,
    )


def ray_cylinder_clearance(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    atom_positions: torch.Tensor,
    exclude_indices: torch.Tensor,
    length: float = 5.0,
    radius: float = 1.2,
    softness: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ray_origins.ndim != 2 or ray_origins.size(-1) != 3:
        raise ValueError("ray_origins must have shape [N, 3]")
    if ray_directions.ndim != 2 or ray_directions.size(-1) != 3:
        raise ValueError("ray_directions must have shape [N, 3]")

    unit = ray_directions / ray_directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    rel = atom_positions.unsqueeze(0) - ray_origins.unsqueeze(1)
    axial = (rel * unit.unsqueeze(1)).sum(dim=-1)
    clamped_axial = axial.clamp(min=0.0, max=float(length))
    closest = ray_origins.unsqueeze(1) + clamped_axial.unsqueeze(-1) * unit.unsqueeze(1)
    radial = (atom_positions.unsqueeze(0) - closest).norm(dim=-1)

    valid_axial = torch.sigmoid((axial - 0.0) * softness) * torch.sigmoid((float(length) - axial) * softness)
    radial_overlap = torch.sigmoid((float(radius) - radial) * softness)
    overlap = valid_axial * radial_overlap

    exclude = 1.0 - torch.nn.functional.one_hot(
        exclude_indices.to(dtype=torch.long),
        num_classes=atom_positions.size(0),
    ).to(dtype=overlap.dtype)
    overlap = overlap * exclude
    max_overlap = overlap.max(dim=1).values
    clearance = (1.0 - max_overlap).clamp_min(0.0)
    return clearance, overlap
