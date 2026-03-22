from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn as nn

from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState

from .flow_matching import FlowMatchTSGenerator
from .navigator import LeastActionResult, NavigatorCandidate
from .observables import QuantumDescriptorExtractor, TSDARResult
from .solvers import CliffordLieIntegrator, NeuralPathOptimizationResult, NeuralPathOptimizer
from .ts_detector import TransitionStateCandidate, Transition_State_Detector


@dataclass
class NEXUSTSSearchResult:
    ts_guess: torch.Tensor
    ts_dar: TSDARResult
    path_optimization: NeuralPathOptimizationResult
    saddle_state: TransitionStateCandidate
    refined_navigation: Optional[LeastActionResult]


class NEXUSTSSearch(nn.Module):
    def __init__(
        self,
        flow_matcher: Optional[FlowMatchTSGenerator] = None,
        descriptor_extractor: Optional[QuantumDescriptorExtractor] = None,
        path_optimizer: Optional[NeuralPathOptimizer] = None,
        ts_detector: Optional[Transition_State_Detector] = None,
        dimer_steps: int = 6,
        dimer_step_size: float = 1.0e-2,
        min_mode_eps: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.flow_matcher = flow_matcher or FlowMatchTSGenerator()
        self.descriptor_extractor = descriptor_extractor or QuantumDescriptorExtractor()
        self.path_optimizer = path_optimizer or NeuralPathOptimizer()
        self.ts_detector = ts_detector or Transition_State_Detector()
        self.dimer_steps = int(dimer_steps)
        self.dimer_step_size = float(dimer_step_size)
        self.min_mode_eps = float(min_mode_eps)

    def locate_saddle(
        self,
        hamiltonian,
        solver: CliffordLieIntegrator,
        y_guess: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> TransitionStateCandidate:
        y = y_guess.clone().requires_grad_(True)
        direction = torch.randn_like(y)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        for _ in range(self.dimer_steps):
            q, p = solver.multivector_to_phase_space(y)
            energy = hamiltonian(
                q,
                p,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            grad = torch.autograd.grad(energy, y, create_graph=True, retain_graph=True)[0]
            y_plus = (y + self.min_mode_eps * direction).requires_grad_(True)
            q_plus, p_plus = solver.multivector_to_phase_space(y_plus)
            e_plus = hamiltonian(
                q_plus,
                p_plus,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            g_plus = torch.autograd.grad(e_plus, y_plus, create_graph=True, retain_graph=True)[0]
            hv_dir = (g_plus - grad) / self.min_mode_eps
            direction = hv_dir / hv_dir.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            proj = (grad * direction).sum(dim=-1, keepdim=True) * direction
            modified_grad = grad - 2.0 * proj
            y = (y - self.dimer_step_size * modified_grad).requires_grad_(True)
        q_s, p_s = solver.multivector_to_phase_space(y)
        return self.ts_detector(
            hamiltonian,
            q_s,
            p_s,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )

    def refine_navigation(
        self,
        navigation: LeastActionResult,
        ts_candidate: TransitionStateCandidate,
    ) -> LeastActionResult:
        best = navigation.best
        refined_best = replace(best, ts_candidate=ts_candidate)
        candidates = [refined_best if candidate is best else candidate for candidate in navigation.candidates]
        return replace(navigation, best=refined_best, candidates=candidates)

    def forward(
        self,
        hamiltonian,
        solver: CliffordLieIntegrator,
        navigation: LeastActionResult,
        *,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NEXUSTSSearchResult:
        trajectory = navigation.best.trajectory
        if trajectory.lie_state_path is None:
            raise RuntimeError("NEXUSTSSearch requires a Lie-state trajectory")
        y_reactant = trajectory.lie_state_path[0]
        y_product = trajectory.lie_state_path[-1]
        ts_guess = self.flow_matcher(y_reactant, y_product)
        ts_dar = self.descriptor_extractor.identify_ts_candidates(trajectory.lie_state_path)
        path_optimization = self.path_optimizer.optimize(
            hamiltonian,
            solver,
            y_reactant,
            y_product,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        best_idx = int(torch.argmax(path_optimization.energies).detach().cpu().item())
        guess_from_path = path_optimization.path_points[best_idx]
        saddle_state = self.locate_saddle(
            hamiltonian,
            solver,
            0.5 * (ts_guess + guess_from_path),
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        refined_navigation = self.refine_navigation(navigation, saddle_state)
        return NEXUSTSSearchResult(
            ts_guess=ts_guess,
            ts_dar=ts_dar,
            path_optimization=path_optimization,
            saddle_state=saddle_state,
            refined_navigation=refined_navigation,
        )


__all__ = ["NEXUSTSSearch", "NEXUSTSSearchResult"]
