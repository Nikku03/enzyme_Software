from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


class Transition_State_Detector(nn.Module):
    def __init__(
        self,
        active_atoms: int = 4,
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
        atom_indices = self._select_active_atoms(q_eval, target_point=target_point)
        q_subset = q_eval[atom_indices].reshape(-1).clone().requires_grad_(True)

        def potential_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                return self._potential_on_subset(
                    hamiltonian,
                    x,
                    q_eval,
                    atom_indices,
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            hessian = torch.autograd.functional.hessian(potential_fn, q_subset, create_graph=True)
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))
        eigenvalues = self._stable_eigvalsh(hessian)
        negative_count = int((eigenvalues < self.negative_tolerance).sum().detach().cpu().item())
        positive_count = int((eigenvalues > self.positive_tolerance).sum().detach().cpu().item())
        is_transition_state = negative_count == 1 and positive_count >= eigenvalues.numel() - 1
        return TransitionStateCandidate(
            q=q_eval,
            p=p,
            eigenvalues=eigenvalues,
            negative_count=negative_count,
            is_transition_state=bool(is_transition_state),
            potential_energy=potential_fn(q_subset),
            atom_indices=atom_indices,
        )


__all__ = ["TransitionStateCandidate", "Transition_State_Detector"]
