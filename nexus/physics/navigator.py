from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState

from .hamiltonian import NEXUS_Hamiltonian
from .ode_solver import Symplectic_ODE_Solver
from .solvers import CliffordLieIntegrator
from .trajectory import NEXUS_Trajectory
from .ts_detector import TransitionStateCandidate, Transition_State_Detector


@dataclass
class NavigatorCandidate:
    loss: torch.Tensor
    terminal_distance: torch.Tensor
    action_term: torch.Tensor
    trajectory: NEXUS_Trajectory
    p_init: torch.Tensor
    target_atom_index: torch.Tensor
    target_point: torch.Tensor
    ts_candidate: Optional[TransitionStateCandidate]


@dataclass
class LeastActionResult:
    best: NavigatorCandidate
    candidates: List[NavigatorCandidate]
    optimization_losses: torch.Tensor


class Least_Action_Navigator(nn.Module):
    def __init__(
        self,
        solver: Optional[nn.Module] = None,
        ts_detector: Optional[Transition_State_Detector] = None,
        action_weight: float = 0.05,
        learning_rate: float = 0.01,
        optimization_steps: int = 6,
        candidate_batch: int = 8,
        momentum_noise: float = 0.01,
        momentum_clip: float = 0.25,
    ) -> None:
        super().__init__()
        self.solver = solver or CliffordLieIntegrator()
        self.ts_detector = ts_detector or Transition_State_Detector()
        self.action_weight = float(action_weight)
        self.learning_rate = float(learning_rate)
        self.optimization_steps = int(optimization_steps)
        self.candidate_batch = int(candidate_batch)
        self.momentum_noise = float(momentum_noise)
        self.momentum_clip = float(momentum_clip)

    def _canonical_seed(
        self,
        q_init: torch.Tensor,
        target_atom_index: torch.Tensor,
        target_point: torch.Tensor,
    ) -> int:
        q_flat = torch.round(q_init.detach().to(dtype=torch.float64).reshape(-1) * 1.0e5).to(dtype=torch.int64)
        t_flat = torch.round(target_point.detach().to(dtype=torch.float64).reshape(-1) * 1.0e5).to(dtype=torch.int64)
        atom_term = target_atom_index.detach().to(dtype=torch.int64).view(-1)
        payload = torch.cat([q_flat, t_flat, atom_term], dim=0)
        weights = torch.arange(1, payload.numel() + 1, dtype=torch.int64, device=payload.device)
        checksum = ((payload + 104729) * weights).sum()
        seed = int(torch.abs(checksum).detach().cpu().item() % 2147483647)
        return max(seed, 1)

    def _path_loss(
        self,
        trajectory: NEXUS_Trajectory,
        target_atom_index: torch.Tensor,
        target_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terminal_distance = (trajectory.q_path[-1, target_atom_index] - target_point).pow(2).sum()
        action_term = trajectory.action_integral.abs()
        total = terminal_distance + self.action_weight * action_term
        return total, terminal_distance, action_term

    def _run_candidate(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        target_point: torch.Tensor,
        steps: int,
        dt: float,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NavigatorCandidate:
        q_boundary = q_init.detach().clone()
        p_boundary = p_init if isinstance(p_init, nn.Parameter) else p_init.detach().clone().requires_grad_(True)
        target_atom_index = target_atom_index.detach().clone()
        target_point = target_point.detach().clone()
        reactive_reference = hamiltonian.reactive_reference.detach().clone()
        try:
            trajectory = self.solver(
                hamiltonian,
                q_boundary,
                p_boundary,
                steps=steps,
                dt=dt,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            loss, terminal_distance, action_term = self._path_loss(trajectory, target_atom_index, target_point)
            if not bool(torch.isfinite(trajectory.h_path).all().item()) or not bool(torch.isfinite(loss).all().item()):
                inf = torch.full((), float("inf"), dtype=q_boundary.dtype, device=q_boundary.device)
                return NavigatorCandidate(
                    loss=inf,
                    terminal_distance=inf,
                    action_term=inf,
                    trajectory=trajectory,
                    p_init=p_boundary.detach(),
                    target_atom_index=target_atom_index,
                    target_point=target_point,
                    ts_candidate=None,
                )
            ts_candidate = self.ts_detector(
                hamiltonian,
                trajectory.q_path[-1],
                trajectory.p_path[-1],
                smiles=smiles,
                species=species,
                target_point=target_point,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            return NavigatorCandidate(
                loss=loss,
                terminal_distance=terminal_distance,
                action_term=action_term,
                trajectory=trajectory,
                p_init=p_init,
                target_atom_index=target_atom_index,
                target_point=target_point,
                ts_candidate=ts_candidate,
            )
        finally:
            hamiltonian.reactive_reference.copy_(reactive_reference)

    def optimize_initial_momentum(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        target_point: torch.Tensor,
        steps: int,
        dt: float,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[NavigatorCandidate, torch.Tensor]:
        p_param = nn.Parameter(torch.zeros_like(q_init))
        optimizer = torch.optim.Adam([p_param], lr=self.learning_rate)
        history = []

        for _ in range(self.optimization_steps):
            optimizer.zero_grad()
            candidate = self._run_candidate(
                hamiltonian,
                q_init,
                p_param,
                smiles=smiles,
                species=species,
                target_atom_index=target_atom_index,
                target_point=target_point,
                steps=steps,
                dt=dt,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            if not bool(torch.isfinite(candidate.loss).all().item()):
                with torch.no_grad():
                    p_param.mul_(0.5)
                history.append(torch.full((), float("inf"), dtype=q_init.dtype, device=q_init.device))
                continue
            grad_p = torch.autograd.grad(
                outputs=candidate.loss,
                inputs=p_param,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            # Finite loss does NOT guarantee finite gradient — NaN can propagate through
            # intermediate tensors and appear only in the gradient.  Sanitize before
            # assigning so Adam never accumulates NaN into its moment estimates.
            grad_p = torch.nan_to_num(grad_p, nan=0.0, posinf=0.0, neginf=0.0)
            p_param.grad = grad_p
            torch.nn.utils.clip_grad_norm_([p_param], max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                # clamp_ does NOT sanitize NaN; nan_to_num first, then clamp.
                p_param.data = torch.nan_to_num(p_param.data, nan=0.0, posinf=0.0, neginf=0.0)
                p_param.clamp_(-self.momentum_clip, self.momentum_clip)
            history.append(candidate.loss.detach())

        final_candidate = self._run_candidate(
            hamiltonian,
            q_init,
            p_param.detach().clone().requires_grad_(True),
            smiles=smiles,
            species=species,
            target_atom_index=target_atom_index,
            target_point=target_point,
            steps=steps,
            dt=dt,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        return final_candidate, torch.stack(history, dim=0)

    def sample_candidates(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        target_point: torch.Tensor,
        steps: int,
        dt: float,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> List[NavigatorCandidate]:
        seed = self._canonical_seed(q_init, target_atom_index, target_point)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        candidates: List[NavigatorCandidate] = [
            self._run_candidate(
                hamiltonian,
                q_init,
                torch.zeros_like(q_init),
                smiles=smiles,
                species=species,
                target_atom_index=target_atom_index,
                target_point=target_point,
                steps=steps,
                dt=dt,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
        ]
        for _ in range(self.candidate_batch):
            p0 = self.momentum_noise * torch.randn(
                q_init.shape,
                generator=generator,
                dtype=q_init.dtype,
                device="cpu",
            ).to(q_init.device)
            candidates.append(
                self._run_candidate(
                    hamiltonian,
                    q_init,
                    p0,
                    smiles=smiles,
                    species=species,
                    target_atom_index=target_atom_index,
                    target_point=target_point,
                    steps=steps,
                    dt=dt,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
            )
        return candidates

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        target_point: torch.Tensor,
        steps: int,
        dt: float,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> LeastActionResult:
        optimized, opt_history = self.optimize_initial_momentum(
            hamiltonian,
            q_init,
            smiles=smiles,
            species=species,
            target_atom_index=target_atom_index,
            target_point=target_point,
            steps=steps,
            dt=dt,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        sampled = self.sample_candidates(
            hamiltonian,
            q_init,
            smiles=smiles,
            species=species,
            target_atom_index=target_atom_index,
            target_point=target_point,
            steps=steps,
            dt=dt,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        candidates = [optimized] + sampled
        finite_candidates = [c for c in candidates if bool(torch.isfinite(c.loss).all().item())]
        best_pool = finite_candidates if finite_candidates else candidates
        best = min(best_pool, key=lambda x: float(torch.nan_to_num(x.loss, nan=float("inf")).detach().cpu().item()))
        return LeastActionResult(best=best, candidates=candidates, optimization_losses=opt_history)


__all__ = ["Least_Action_Navigator", "LeastActionResult", "NavigatorCandidate"]
