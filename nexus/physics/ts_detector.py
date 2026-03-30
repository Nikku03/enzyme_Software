from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState

from .hamiltonian import NEXUS_Hamiltonian


@dataclass
class TransitionStateCandidate:
    q: torch.Tensor
    p: torch.Tensor
    eigenvalues: torch.Tensor
    negative_count: int
    is_transition_state: bool
    potential_energy: torch.Tensor
    atom_indices: torch.Tensor


def _matrix_free_extreme_eigenpair(
    energy_func: Callable[[torch.Tensor], torch.Tensor],
    coords: torch.Tensor,
    *,
    shift: float | torch.Tensor = 0.0,
    orthogonal_to: Optional[torch.Tensor] = None,
    max_iter: int = 30,
    tol: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matrix-free extreme eigenpair of H-shift*I using Vector-Hessian Products.

    Returns the dominant-magnitude eigenpair of the shifted operator. When the
    shift exceeds the Hessian spectral radius, this becomes the lowest mode of
    the original Hessian.
    """
    flat0 = coords.reshape(-1).detach()
    shift_t = torch.as_tensor(shift, dtype=coords.dtype, device=coords.device)

    def _energy_from_flat(flat_coords: torch.Tensor) -> torch.Tensor:
        return energy_func(flat_coords.view_as(coords))

    def hvp_fn(v: torch.Tensor) -> torch.Tensor:
        _, hv = torch.autograd.functional.vhp(
            _energy_from_flat,
            flat0,
            v=v.reshape(-1),
            create_graph=False,
            strict=False,
        )
        return hv.view_as(coords)

    v = torch.randn_like(coords)
    v = v / v.norm().clamp_min(1.0e-8)
    previous = None
    orth = None
    if orthogonal_to is not None:
        orth = orthogonal_to / orthogonal_to.norm().clamp_min(1.0e-8)

    for _ in range(max_iter):
        Hv = hvp_fn(v) - shift_t * v
        if orth is not None:
            Hv = Hv - torch.sum(Hv * orth) * orth
        hv_norm = Hv.norm().clamp_min(1.0e-8)
        v = Hv / hv_norm
        rayleigh = torch.sum(v * (hvp_fn(v) - shift_t * v))
        if previous is not None and torch.abs(rayleigh - previous) < tol:
            previous = rayleigh
            break
        previous = rayleigh

    if previous is None:
        previous = torch.sum(v * (hvp_fn(v) - shift_t * v))
    return previous + shift_t, v


def _matrix_free_lowest_eigenmodes(
    energy_func: Callable[[torch.Tensor], torch.Tensor],
    coords: torch.Tensor,
    *,
    max_iter: int = 30,
    tol: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the two lowest eigenvalues of the full-system Hessian without
    materializing the dense matrix.
    """
    flat0 = coords.reshape(-1).detach()

    def _energy_from_flat(flat_coords: torch.Tensor) -> torch.Tensor:
        return energy_func(flat_coords.view_as(coords))

    def hvp_flat(v_flat: torch.Tensor) -> torch.Tensor:
        _, hv = torch.autograd.functional.vhp(
            _energy_from_flat,
            flat0,
            v=v_flat,
            create_graph=False,
            strict=False,
        )
        return hv

    # Spectral-radius estimate via power iteration on H^2 to get a safe shift.
    v = torch.randn_like(flat0)
    v = v / v.norm().clamp_min(1.0e-8)
    spectral_radius = torch.zeros((), dtype=coords.dtype, device=coords.device)
    for _ in range(12):
        Hv = hvp_flat(v)
        H2v = hvp_flat(Hv)
        denom = v.norm().pow(2).clamp_min(1.0e-8)
        spectral_radius = torch.sqrt(torch.abs(torch.dot(v, H2v)) / denom).to(dtype=coords.dtype)
        v = H2v / H2v.norm().clamp_min(1.0e-8)

    shift = spectral_radius + torch.as_tensor(10.0, dtype=coords.dtype, device=coords.device)
    lowest, mode0 = _matrix_free_extreme_eigenpair(
        energy_func,
        coords,
        shift=shift,
        max_iter=max_iter,
        tol=tol,
    )
    second, _ = _matrix_free_extreme_eigenpair(
        energy_func,
        coords,
        shift=shift,
        orthogonal_to=mode0,
        max_iter=max_iter,
        tol=tol,
    )
    eigenvalues = torch.sort(torch.stack([lowest, second])).values
    return eigenvalues, torch.arange(coords.size(0), device=coords.device, dtype=torch.long)


def compute_micro_hessian(
    q_full: torch.Tensor,
    active_indices: torch.Tensor,
    energy_func: Callable[[torch.Tensor], torch.Tensor],
    *,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a Hessian only on an active subset of atoms while freezing the rest.

    Returns a dense [k*3, k*3] Hessian matrix and the sorted active atom indices.
    """
    if q_full.ndim != 2 or q_full.size(-1) != 3:
        raise ValueError(f"Expected q_full to have shape [N, 3], got {tuple(q_full.shape)}")

    active_indices = active_indices.to(dtype=torch.long, device=q_full.device).view(-1)
    if active_indices.numel() == 0:
        raise ValueError("active_indices must be non-empty")
    active_indices = torch.unique(active_indices, sorted=True)

    n_atoms = q_full.size(0)
    is_active = torch.zeros(n_atoms, dtype=torch.bool, device=q_full.device)
    is_active[active_indices] = True

    q_background = q_full[~is_active].detach()
    q_active = q_full[is_active].detach().requires_grad_(True)

    def masked_energy_fn(q_active_in: torch.Tensor) -> torch.Tensor:
        q_reconstructed = torch.empty_like(q_full)
        q_reconstructed[~is_active] = q_background
        q_reconstructed[is_active] = q_active_in
        return energy_func(q_reconstructed)

    micro_hessian = torch.autograd.functional.hessian(
        masked_energy_fn,
        q_active,
        create_graph=create_graph,
        strict=False,
    )
    matrix_dim = int(q_active.numel())
    return micro_hessian.reshape(matrix_dim, matrix_dim), active_indices


class Transition_State_Detector(nn.Module):
    def __init__(
        self,
        active_atoms: int = 8,
        negative_tolerance: float = -1.0e-5,
        positive_tolerance: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.active_atoms = int(active_atoms)
        self.negative_tolerance = float(negative_tolerance)
        self.positive_tolerance = float(positive_tolerance)

    def _stable_eigvalsh(self, hessian: torch.Tensor) -> torch.Tensor:
        hessian64 = hessian.to(dtype=torch.float64)
        eye = torch.eye(hessian64.size(0), dtype=hessian64.dtype, device=hessian64.device)
        for jitter in (0.0, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
            try:
                return torch.linalg.eigvalsh(hessian64 + jitter * eye).to(dtype=hessian.dtype)
            except RuntimeError:
                continue
        return torch.zeros(hessian.size(0), dtype=hessian.dtype, device=hessian.device)

    def _select_active_atoms(self, q: torch.Tensor, target_point: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_atoms = q.size(0)
        if n_atoms <= self.active_atoms:
            return torch.arange(n_atoms, device=q.device)
        if target_point is None:
            centroid = q.mean(dim=0, keepdim=True)
            scores = (q - centroid).norm(dim=-1)
        else:
            scores = -(q - target_point.view(1, 3)).norm(dim=-1)
        topk = torch.topk(scores, k=self.active_atoms, largest=True).indices
        return torch.sort(topk).values

    def _potential_on_subset(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_subset_flat: torch.Tensor,
        q_full: torch.Tensor,
        atom_indices: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        q_eval = q_full.clone()
        q_eval[atom_indices] = q_subset_flat.view(-1, 3)
        physical, reactive, _ = hamiltonian.compute_potential_energy(
            q_eval,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        return physical + hamiltonian.coupling_lambda * reactive

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q: torch.Tensor,
        p: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        target_point: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> TransitionStateCandidate:
        q_eval = q.clone().requires_grad_(True)

        def potential_fn(q_reconstructed: torch.Tensor) -> torch.Tensor:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                physical, reactive, _ = hamiltonian.compute_potential_energy(
                    q_reconstructed,
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                return physical + hamiltonian.coupling_lambda * reactive

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            eigenvalues, atom_indices = _matrix_free_lowest_eigenmodes(
                potential_fn,
                q_eval,
                max_iter=30,
                tol=1.0e-4,
            )
        negative_count = int((eigenvalues < self.negative_tolerance).sum().detach().cpu().item())
        positive_count = int((eigenvalues > self.positive_tolerance).sum().detach().cpu().item())
        is_transition_state = negative_count == 1 and positive_count >= eigenvalues.numel() - 1
        return TransitionStateCandidate(
            q=q_eval,
            p=p,
            eigenvalues=eigenvalues,
            negative_count=negative_count,
            is_transition_state=bool(is_transition_state),
            potential_energy=potential_fn(q_eval),
            atom_indices=atom_indices,
        )


__all__ = ["TransitionStateCandidate", "Transition_State_Detector", "compute_micro_hessian"]
