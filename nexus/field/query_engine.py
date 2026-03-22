from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.utils.geometry_utils import (
    generate_shell_grid,
    ray_cylinder_clearance,
    smooth_steric_mask,
    symmetric_traceless_direction_coefficients,
)


@dataclass
class SubAtomicQueryResult:
    atom_indices: torch.Tensor
    atom_centers: torch.Tensor
    grid_points: torch.Tensor
    shell_radii: torch.Tensor
    psi: torch.Tensor
    grad_psi: torch.Tensor
    steric_mask: torch.Tensor
    heme_mask: torch.Tensor
    accessibility_mask: torch.Tensor
    accessible_psi: torch.Tensor
    peak_points: torch.Tensor
    peak_values: torch.Tensor
    peak_indices: torch.Tensor
    refined_peak_points: torch.Tensor
    refined_peak_values: torch.Tensor
    refined_entry_vectors: torch.Tensor
    curvature_hessian: torch.Tensor
    curvature_eigenvalues: torch.Tensor
    anisotropy_score: torch.Tensor
    approach_vectors: torch.Tensor
    path_clearance: torch.Tensor
    path_obstruction_overlap: torch.Tensor
    alignment_tensor: torch.Tensor
    exposure_scores: torch.Tensor
    dielectric_penalties: torch.Tensor
    effective_reactivity: torch.Tensor
    entry_vectors: torch.Tensor
    integrated_reactivity: torch.Tensor
    accessible_fraction: torch.Tensor
    l2_alignment: torch.Tensor
    metadata: Dict[str, object]


class SubAtomicQueryEngine(nn.Module):
    def __init__(
        self,
        radius: float = 2.5,
        n_points: int = 96,
        query_chunk_size: int = 16,
        shell_fractions: Optional[Sequence[float]] = None,
        min_clearance: float = 1.15,
        steric_softness: float = 10.0,
        refine_steps: int = 5,
        refine_step_size: float = 0.05,
        refine_decay: float = 0.5,
        heme_radius: float = 5.0,
        heme_half_thickness: float = 0.75,
        heme_offset: float = 1.5,
        heme_softness: float = 8.0,
        hydration_decay: float = 0.35,
        local_env_sigma: float = 2.5,
        depth_softmax_temperature: float = 6.0,
        polar_relief: float = 0.15,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.n_points = int(n_points)
        self.query_chunk_size = int(query_chunk_size)
        self.shell_fractions = tuple(shell_fractions or (0.35, 0.55, 0.75, 0.90, 1.00))
        self.min_clearance = float(min_clearance)
        self.steric_softness = float(steric_softness)
        self.refine_steps = int(refine_steps)
        self.refine_step_size = float(refine_step_size)
        self.refine_decay = float(refine_decay)
        self.heme_radius = float(heme_radius)
        self.heme_half_thickness = float(heme_half_thickness)
        self.heme_offset = float(heme_offset)
        self.heme_softness = float(heme_softness)
        self.local_env_sigma = float(local_env_sigma)
        self.depth_softmax_temperature = float(depth_softmax_temperature)
        self.hydration_decay_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(hydration_decay)))))
        self.metabolic_nudge_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.15))))
        self.polar_relief_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(polar_relief)))))

    @property
    def hydration_decay(self) -> torch.Tensor:
        return F.softplus(self.hydration_decay_raw)

    @property
    def metabolic_nudge_strength(self) -> torch.Tensor:
        return F.softplus(self.metabolic_nudge_raw)

    @property
    def polar_relief(self) -> torch.Tensor:
        return F.softplus(self.polar_relief_raw)

    def _l2_coefficients_to_matrix(self, coeffs: torch.Tensor) -> torch.Tensor:
        c0, c1, c2, c3, c4 = coeffs.unbind(dim=-1)
        s2 = torch.sqrt(torch.tensor(2.0, dtype=coeffs.dtype, device=coeffs.device))
        s6 = torch.sqrt(torch.tensor(6.0, dtype=coeffs.dtype, device=coeffs.device))
        xx = c0 / s2 - c1 / s6
        yy = -c0 / s2 - c1 / s6
        zz = 2.0 * c1 / s6
        xy = c2 / s2
        xz = c3 / s2
        yz = c4 / s2
        return torch.stack(
            [
                torch.stack([xx, xy, xz], dim=-1),
                torch.stack([xy, yy, yz], dim=-1),
                torch.stack([xz, yz, zz], dim=-1),
            ],
            dim=-2,
        )

    def generate_query_grid(
        self,
        field,
        positions: torch.Tensor,
        species: torch.Tensor,
        radius: Optional[float] = None,
        n_points: Optional[int] = None,
        nudge_vectors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        heavy_mask = species > 1
        atom_indices = heavy_mask.nonzero(as_tuple=False).squeeze(-1)
        if atom_indices.numel() == 0:
            atom_indices = torch.arange(positions.size(0), device=positions.device)
        centers_world = positions[atom_indices]
        centers_internal = field.to_internal_coords(centers_world).to(dtype=positions.dtype)
        nudge_internal = None
        if nudge_vectors is not None:
            nudge_internal = (nudge_vectors[atom_indices].to(dtype=field.frame.dtype) @ field.frame).to(dtype=positions.dtype)
        grid_internal, shell_radii = generate_shell_grid(
            centers_internal,
            radius=self.radius if radius is None else radius,
            n_points=self.n_points if n_points is None else n_points,
            shell_fractions=self.shell_fractions,
            bias_directions=nudge_internal,
            bias_strength=float(self.metabolic_nudge_strength.detach().cpu().item()),
        )
        # grid_internal is already in absolute world space because to_internal_coords is a
        # dtype cast only (no centering).  Adding field.centroid here would double-count
        # the molecular centroid offset.  Apply only the frame rotation (no-op with the
        # current identity frame, but correct for any future non-trivial frame).
        grid_world = (grid_internal.to(dtype=field.frame.dtype) @ field.frame.transpose(0, 1))
        return atom_indices, centers_world, grid_world.to(dtype=positions.dtype), shell_radii

    def probe_reaction_volume(self, field, grid_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        flat_grid = grid_points.reshape(-1, 3).clone().requires_grad_(True)
        component_chunks: Dict[str, list[torch.Tensor]] = {}
        psi_chunks = []
        grad_chunks = []
        chunk_size = max(int(self.query_chunk_size), 1)

        for start in range(0, flat_grid.size(0), chunk_size):
            stop = min(start + chunk_size, flat_grid.size(0))
            flat_chunk = flat_grid[start:stop].clone().requires_grad_(True)
            components = field.query_components(flat_chunk)
            psi_chunk = components["total"]
            grad_chunk = torch.autograd.grad(
                outputs=psi_chunk.sum(),
                inputs=flat_chunk,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )[0]
            psi_chunks.append(psi_chunk)
            grad_chunks.append(grad_chunk)
            for key, value in components.items():
                if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == flat_chunk.size(0):
                    component_chunks.setdefault(key, []).append(value)
                elif key not in component_chunks:
                    component_chunks[key] = [value]

        psi_flat = torch.cat(psi_chunks, dim=0)
        grad_flat = torch.cat(grad_chunks, dim=0)
        psi = psi_flat.view(grid_points.size(0), grid_points.size(1))
        grad = grad_flat.view(grid_points.size(0), grid_points.size(1), 3)
        reshaped = {}
        for key, values in component_chunks.items():
            value = values[0] if len(values) == 1 else torch.cat(values, dim=0)
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == flat_grid.size(0):
                reshaped[key] = value.view(grid_points.size(0), grid_points.size(1), *value.shape[1:])
            else:
                reshaped[key] = value
        return psi, grad, reshaped

    def _l2_alignment_score(
        self,
        field,
        atom_indices: torch.Tensor,
        entry_vectors: torch.Tensor,
    ) -> torch.Tensor:
        features = field.source_output.fused_features
        template = features.get("2e_topology", features.get("2e"))
        if template is None:
            return torch.zeros(atom_indices.numel(), dtype=entry_vectors.dtype, device=entry_vectors.device)
        atom_template = template[atom_indices].mean(dim=1)
        query_template = symmetric_traceless_direction_coefficients(entry_vectors)
        return torch.nn.functional.cosine_similarity(atom_template, query_template, dim=-1, eps=1.0e-8)

    def _heme_exclusion_mask(
        self,
        query_points: torch.Tensor,
        atom_positions: torch.Tensor,
        center_indices: torch.Tensor,
        nudge_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        center_pos = atom_positions[center_indices].unsqueeze(1)
        direction = query_points - center_pos
        direction_norm = direction.norm(dim=-1, keepdim=True)
        if nudge_vectors is not None:
            nudge = nudge_vectors.unsqueeze(1)
            nudge_norm = nudge.norm(dim=-1, keepdim=True)
            safe_nudge = torch.where(
                nudge_norm > 1.0e-6,
                nudge / nudge_norm.clamp_min(1.0e-6),
                torch.zeros_like(nudge),
            )
            direction = torch.where(direction_norm > 1.0e-6, direction, safe_nudge.expand_as(direction))
            direction_norm = direction.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[..., 0] = 1.0
        unit = torch.where(direction_norm > 1.0e-6, direction / direction_norm.clamp_min(1.0e-6), fallback)
        heme_center = query_points + self.heme_offset * unit

        rel = atom_positions.unsqueeze(0).unsqueeze(0) - heme_center.unsqueeze(2)
        axial = (rel * unit.unsqueeze(2)).sum(dim=-1).abs()
        plane = rel - axial.unsqueeze(-1) * unit.unsqueeze(2)
        radial = plane.norm(dim=-1)

        exclude = 1.0 - torch.nn.functional.one_hot(
            center_indices.to(dtype=torch.long),
            num_classes=atom_positions.size(0),
        ).to(dtype=radial.dtype).unsqueeze(1)
        overlap = torch.sigmoid((self.heme_radius - radial) * self.heme_softness) * torch.sigmoid(
            (self.heme_half_thickness - axial) * self.heme_softness
        )
        overlap = overlap * exclude
        max_overlap = overlap.max(dim=-1).values
        return (1.0 - max_overlap).clamp_min(0.0)

    def _point_accessibility_mask(
        self,
        query_points: torch.Tensor,
        manifold,
        center_indices: torch.Tensor,
        nudge_vectors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_grid = query_points.unsqueeze(1)
        steric = smooth_steric_mask(
            point_grid,
            manifold.pos,
            center_indices,
            min_clearance=self.min_clearance,
            softness=self.steric_softness,
            nudge_vectors=nudge_vectors,
        ).squeeze(1)
        heme = self._heme_exclusion_mask(
            point_grid,
            manifold.pos,
            center_indices,
            nudge_vectors=nudge_vectors,
        ).squeeze(1)
        return steric, heme, steric * heme

    def _objective_values(
        self,
        field,
        manifold,
        center_indices: torch.Tensor,
        query_points: torch.Tensor,
        nudge_vectors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = field.query(query_points)
        steric, heme, mask = self._point_accessibility_mask(query_points, manifold, center_indices, nudge_vectors=nudge_vectors)
        accessible = raw * mask
        return raw, steric, heme, accessible

    def refine_peak_coordinates(
        self,
        field,
        manifold,
        atom_indices: torch.Tensor,
        atom_centers: torch.Tensor,
        initial_points: torch.Tensor,
        initial_values: torch.Tensor,
        nudge_vectors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined_points = initial_points
        refined_values = initial_values
        step_size = self.refine_step_size

        for _ in range(self.refine_steps):
            points = refined_points.clone().requires_grad_(True)
            _, _, _, objective = self._objective_values(
                field,
                manifold,
                atom_indices,
                points,
                nudge_vectors=nudge_vectors,
            )
            grad = torch.autograd.grad(
                outputs=objective.sum(),
                inputs=points,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )[0]
            grad_norm = grad.norm(dim=-1, keepdim=True)
            step = step_size * grad / grad_norm.clamp_min(1.0e-8)
            candidate = points + step

            displacement = candidate - atom_centers
            disp_norm = displacement.norm(dim=-1, keepdim=True)
            capped = displacement * torch.clamp((self.radius + 0.5) / disp_norm.clamp_min(1.0e-8), max=1.0)
            candidate = atom_centers + capped

            _, _, _, candidate_value = self._objective_values(
                field,
                manifold,
                atom_indices,
                candidate,
                nudge_vectors=nudge_vectors,
            )
            improve = candidate_value >= refined_values
            refined_points = torch.where(improve.unsqueeze(-1), candidate, refined_points)
            refined_values = torch.where(improve, candidate_value, refined_values)
            step_size *= self.refine_decay

        return refined_points, refined_values

    def _single_point_objective(
        self,
        field,
        manifold,
        center_index: torch.Tensor,
        point: torch.Tensor,
        nudge_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        center_index = center_index.view(1)
        point = point.view(1, 3)
        nudge = None if nudge_vector is None else nudge_vector.view(1, 3)
        _, _, _, value = self._objective_values(
            field,
            manifold,
            center_index,
            point,
            nudge_vectors=nudge,
        )
        return value.squeeze(0)

    def compute_peak_curvature(
        self,
        field,
        manifold,
        atom_indices: torch.Tensor,
        peak_points: torch.Tensor,
        nudge_vectors: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hessians = []
        eigvals = []
        anisotropy = []
        for i in range(atom_indices.numel()):
            center_index = atom_indices[i]
            point = peak_points[i]
            nudge_vector = None if nudge_vectors is None else nudge_vectors[i]

            def scalar_fn(x: torch.Tensor) -> torch.Tensor:
                return self._single_point_objective(field, manifold, center_index, x, nudge_vector=nudge_vector)

            hessian = torch.autograd.functional.hessian(scalar_fn, point, create_graph=True)
            hessian = 0.5 * (hessian + hessian.transpose(0, 1))
            values = torch.linalg.eigvalsh(hessian)
            abs_values = values.abs()
            score = abs_values.max() / abs_values.mean().clamp_min(1.0e-8)
            hessians.append(hessian)
            eigvals.append(values)
            anisotropy.append(score)
        return torch.stack(hessians, dim=0), torch.stack(eigvals, dim=0), torch.stack(anisotropy, dim=0)

    def compute_approach_vectors(
        self,
        field,
        manifold,
        atom_indices: torch.Tensor,
        refined_peak_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = refined_peak_points.clone().requires_grad_(True)
        psi = field.query(points)
        grad = torch.autograd.grad(
            outputs=psi.sum(),
            inputs=points,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        approach_vectors = -grad / grad.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        path_clearance, overlap = ray_cylinder_clearance(
            refined_peak_points,
            approach_vectors,
            manifold.pos,
            atom_indices,
            length=5.0,
            radius=1.2,
            softness=self.steric_softness,
        )

        features = field.source_output.fused_features
        l1_template = features.get("1o_topology", features.get("1o"))
        if l1_template is None:
            l1_alignment = torch.zeros(atom_indices.numel(), dtype=approach_vectors.dtype, device=approach_vectors.device)
        else:
            l1_vector = l1_template[atom_indices].mean(dim=1)
            l1_alignment = torch.nn.functional.cosine_similarity(
                l1_vector,
                approach_vectors,
                dim=-1,
                eps=1.0e-8,
            ).abs()

        l2_template = features.get("2e_topology", features.get("2e"))
        if l2_template is None:
            l2_axis_alignment = torch.zeros_like(l1_alignment)
            l2_overlap = torch.zeros_like(l1_alignment)
        else:
            l2_coeff = l2_template[atom_indices].mean(dim=1)
            l2_matrix = self._l2_coefficients_to_matrix(l2_coeff)
            eigvals, eigvecs = torch.linalg.eigh(l2_matrix)
            principal_axis = eigvecs[..., -1]
            l2_axis_alignment = torch.nn.functional.cosine_similarity(
                principal_axis,
                approach_vectors,
                dim=-1,
                eps=1.0e-8,
            ).abs()
            query_template = symmetric_traceless_direction_coefficients(approach_vectors)
            l2_overlap = torch.nn.functional.cosine_similarity(
                l2_coeff,
                query_template,
                dim=-1,
                eps=1.0e-8,
            ).abs()

        alignment_tensor = torch.stack([l1_alignment, l2_axis_alignment, l2_overlap], dim=-1)
        return approach_vectors, path_clearance, overlap, alignment_tensor

    def compute_environmental_mask(
        self,
        field,
        manifold,
        atom_indices: torch.Tensor,
        peak_points: torch.Tensor,
        path_clearance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centroid = field.centroid
        rel_to_center = peak_points - centroid.unsqueeze(0)
        radial_dir = rel_to_center / rel_to_center.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        atom_rel = manifold.pos - centroid.unsqueeze(0)
        atom_proj = torch.einsum("nd,md->nm", radial_dir, atom_rel)
        peak_proj = (rel_to_center * radial_dir).sum(dim=-1)
        depth_weights = torch.softmax(self.depth_softmax_temperature * atom_proj, dim=-1)
        surface_proj = (depth_weights * atom_proj).sum(dim=-1)
        depth = (surface_proj - peak_proj).clamp_min(0.0)

        rel = manifold.pos.unsqueeze(0) - peak_points.unsqueeze(1)
        dist2 = rel.square().sum(dim=-1)
        env_weights = torch.exp(-dist2 / (2.0 * self.local_env_sigma * self.local_env_sigma))
        env_exclude = 1.0 - torch.nn.functional.one_hot(
            atom_indices.to(dtype=torch.long),
            num_classes=manifold.pos.size(0),
        ).to(dtype=env_weights.dtype)
        env_weights = env_weights * env_exclude
        weight_sum = env_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        mean_rel = (env_weights.unsqueeze(-1) * rel).sum(dim=1) / weight_sum
        centered_rel = rel - mean_rel.unsqueeze(1)
        cov = torch.einsum("nm,nmi,nmj->nij", env_weights, centered_rel, centered_rel) / weight_sum.unsqueeze(-1)
        eigvals = torch.linalg.eigvalsh(0.5 * (cov + cov.transpose(-1, -2)))
        eigvals = eigvals.clamp_min(1.0e-8)
        asphericity = (eigvals[:, -1] - eigvals[:, 0]) / eigvals.sum(dim=-1).clamp_min(1.0e-8)

        local_density = weight_sum.squeeze(-1)
        density_scale = torch.sigmoid(1.5 - 0.15 * local_density)
        depth_scale = torch.exp(-self.hydration_decay.to(dtype=peak_points.dtype, device=peak_points.device) * depth)

        center_species = manifold.species[atom_indices]
        polar_mask = torch.isin(
            center_species.to(dtype=torch.long),
            torch.tensor([7, 8, 15, 16], dtype=torch.long, device=center_species.device),
        ).to(dtype=peak_points.dtype)
        polar_relief = 1.0 + self.polar_relief.to(dtype=peak_points.dtype, device=peak_points.device) * polar_mask

        exposure_scores = (
            0.45 * depth_scale
            + 0.25 * asphericity
            + 0.20 * density_scale
            + 0.10 * path_clearance
        )
        exposure_scores = (exposure_scores * polar_relief).clamp(0.0, 1.0)
        dielectric_penalties = depth_scale * polar_relief
        return exposure_scores, dielectric_penalties, depth

    def find_peak_reactivity(
        self,
        field,
        manifold,
        atom_indices: torch.Tensor,
        atom_centers: torch.Tensor,
        grid_points: torch.Tensor,
        shell_radii: torch.Tensor,
        psi: torch.Tensor,
        grad_psi: torch.Tensor,
    ) -> SubAtomicQueryResult:
        nudge_vectors = getattr(manifold, "reaction_pathway", None)
        if nudge_vectors is not None:
            nudge_vectors = nudge_vectors[atom_indices]
        steric_mask = smooth_steric_mask(
            grid_points,
            manifold.pos,
            atom_indices,
            min_clearance=self.min_clearance,
            softness=self.steric_softness,
            nudge_vectors=nudge_vectors,
        )
        heme_mask = self._heme_exclusion_mask(
            grid_points,
            manifold.pos,
            atom_indices,
            nudge_vectors=nudge_vectors,
        )
        accessibility_mask = steric_mask * heme_mask
        accessible_psi = psi * accessibility_mask
        peak_indices = accessible_psi.argmax(dim=1)
        gather_idx = peak_indices.view(-1, 1, 1).expand(-1, 1, 3)
        peak_points = grid_points.gather(1, gather_idx).squeeze(1)
        peak_values = accessible_psi.gather(1, peak_indices.view(-1, 1)).squeeze(1)
        entry_vectors = peak_points - atom_centers
        refined_peak_points, refined_peak_values = self.refine_peak_coordinates(
            field,
            manifold,
            atom_indices,
            atom_centers,
            peak_points,
            peak_values,
            nudge_vectors=nudge_vectors,
        )
        refined_entry_vectors = refined_peak_points - atom_centers
        curvature_hessian, curvature_eigenvalues, anisotropy_score = self.compute_peak_curvature(
            field,
            manifold,
            atom_indices,
            refined_peak_points,
            nudge_vectors=nudge_vectors,
        )
        approach_vectors, path_clearance, path_obstruction_overlap, alignment_tensor = self.compute_approach_vectors(
            field,
            manifold,
            atom_indices,
            refined_peak_points,
        )
        exposure_scores, dielectric_penalties, depth = self.compute_environmental_mask(
            field,
            manifold,
            atom_indices,
            refined_peak_points,
            path_clearance,
        )

        volume = (4.0 / 3.0) * math.pi * (float(self.radius) ** 3)
        point_volume = volume / float(grid_points.size(1))
        integrated_reactivity = torch.nn.functional.softplus(accessible_psi).sum(dim=1) * point_volume
        accessible_fraction = accessibility_mask.mean(dim=1)
        l2_alignment = self._l2_alignment_score(field, atom_indices, refined_entry_vectors)
        effective_reactivity = (
            torch.nn.functional.softplus(refined_peak_values)
            * exposure_scores
            * dielectric_penalties
            * path_clearance.clamp_min(1.0e-4)
        )

        return SubAtomicQueryResult(
            atom_indices=atom_indices,
            atom_centers=atom_centers,
            grid_points=grid_points,
            shell_radii=shell_radii,
            psi=psi,
            grad_psi=grad_psi,
            steric_mask=steric_mask,
            heme_mask=heme_mask,
            accessibility_mask=accessibility_mask,
            accessible_psi=accessible_psi,
            peak_points=peak_points,
            peak_values=peak_values,
            peak_indices=peak_indices,
            refined_peak_points=refined_peak_points,
            refined_peak_values=refined_peak_values,
            refined_entry_vectors=refined_entry_vectors,
            curvature_hessian=curvature_hessian,
            curvature_eigenvalues=curvature_eigenvalues,
            anisotropy_score=anisotropy_score,
            approach_vectors=approach_vectors,
            path_clearance=path_clearance,
            path_obstruction_overlap=path_obstruction_overlap,
            alignment_tensor=alignment_tensor,
            exposure_scores=exposure_scores,
            dielectric_penalties=dielectric_penalties,
            effective_reactivity=effective_reactivity,
            entry_vectors=entry_vectors,
            integrated_reactivity=integrated_reactivity,
            accessible_fraction=accessible_fraction,
            l2_alignment=l2_alignment,
            metadata={
                "radius_A": self.radius,
                "points_per_shell": self.n_points,
                "n_shells": len(self.shell_fractions),
                "total_points_per_atom": grid_points.size(1),
                "refine_steps": self.refine_steps,
                "refine_step_size_A": self.refine_step_size,
                "heme_disk_radius_A": self.heme_radius,
                "approach_ray_length_A": 5.0,
                "approach_cylinder_radius_A": 1.2,
                "hydration_decay": float(self.hydration_decay.detach().cpu().item()),
                "metabolic_nudge_strength": float(self.metabolic_nudge_strength.detach().cpu().item()),
                "mean_depth_A": float(depth.mean().detach().cpu().item()),
            },
        )

    def forward(
        self,
        field,
        manifold,
        radius: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> SubAtomicQueryResult:
        nudge_vectors = getattr(manifold, "reaction_pathway", None)
        atom_indices, atom_centers, grid_points, shell_radii = self.generate_query_grid(
            field,
            manifold.pos,
            manifold.species,
            radius=radius,
            n_points=n_points,
            nudge_vectors=nudge_vectors,
        )
        psi, grad_psi, _ = self.probe_reaction_volume(field, grid_points)
        return self.find_peak_reactivity(
            field,
            manifold,
            atom_indices,
            atom_centers,
            grid_points,
            shell_radii,
            psi,
            grad_psi,
        )
