from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState

from .clifford_math import clifford_geometric_product
from .hamiltonian import HamiltonianTerms, NEXUS_Hamiltonian
from .lie_algebra import clifford_exp, dexp_inv
from .trajectory import NEXUS_Trajectory


@dataclass
class CliffordIntegratorState:
    y: torch.Tensor
    q: torch.Tensor
    p: torch.Tensor


@dataclass
class NeuralPathOptimizationResult:
    path_points: torch.Tensor
    energies: torch.Tensor
    loss: torch.Tensor


@dataclass
class AdjointDiagnostics:
    forward_drift: torch.Tensor
    backward_drift: torch.Tensor
    checkpoint_count: int


class CayleyPropagator(nn.Module):
    def __init__(self, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def _mode_frequencies(
        self,
        n_beads: int,
        omega: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        k = torch.arange(n_beads, dtype=dtype, device=device)
        return 2.0 * omega * torch.sin(math.pi * k / float(max(n_beads, 1)))

    def step(
        self,
        positions: torch.Tensor,
        momenta: torch.Tensor,
        *,
        mass: torch.Tensor,
        omega: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_beads = positions.size(-2)
        dtype = positions.dtype
        device = positions.device
        dt_t = torch.as_tensor(float(dt), dtype=dtype, device=device)
        mass_t = mass.to(dtype=dtype, device=device)
        omega_modes = self._mode_frequencies(n_beads, omega.to(dtype=dtype, device=device), dtype=dtype, device=device)

        q_modes = torch.fft.fft(positions, dim=-2)
        p_modes = torch.fft.fft(momenta, dim=-2)

        omega_sq = omega_modes.pow(2).view(*([1] * (q_modes.ndim - 2)), n_beads, 1)
        mass_view = mass_t.view(*([1] * (q_modes.ndim - 2)), 1, 1)
        a = (0.5 * dt_t).pow(2) * omega_sq
        denom = (1.0 + a).clamp_min(self.eps)
        vel_modes = p_modes / mass_view.clamp_min(self.eps)
        q_next_modes = ((1.0 - a) * q_modes + dt_t * vel_modes) / denom
        v_next_modes = ((1.0 - a) * vel_modes - dt_t * omega_sq * q_modes) / denom
        p_next_modes = v_next_modes * mass_view

        q_next = torch.fft.ifft(q_next_modes, dim=-2).real
        p_next = torch.fft.ifft(p_next_modes, dim=-2).real
        return q_next, p_next

    def forward(
        self,
        positions: torch.Tensor,
        momenta: torch.Tensor,
        *,
        mass: torch.Tensor,
        omega: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(
            positions,
            momenta,
            mass=mass,
            omega=omega,
            dt=dt,
        )


class CliffordLieIntegrator(nn.Module):
    def __init__(
        self,
        derivative_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        h: float = 0.1,
        rolling_shifts: Sequence[int] = (1, 2, 4, 8),
        interaction_scale: float = 0.05,
        thermostat: bool = True,
        kinetic_threshold: float = 25.0,
        damping: float = 0.995,
        dexp_order: int = 4,
        exp_terms: int = 8,
    ) -> None:
        super().__init__()
        self.derivative_fn = derivative_fn
        self.h = float(h)
        self.rolling_shifts = tuple(int(s) for s in rolling_shifts)
        self.interaction_scale = float(interaction_scale)
        self.thermostat = bool(thermostat)
        self.kinetic_threshold = float(kinetic_threshold)
        self.damping = float(damping)
        self.dexp_order = int(dexp_order)
        self.exp_terms = int(exp_terms)
        self.position_clip = 5.0
        self.momentum_clip = 1.0

    def phase_space_to_multivector(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(q.shape[:-1] + (8,), dtype=q.dtype, device=q.device)
        y[..., 1:4] = q
        y[..., 4:7] = p
        y[..., 7] = (q * p).sum(dim=-1)
        return y

    def multivector_to_phase_space(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.nan_to_num(y[..., 1:4], nan=0.0, posinf=self.position_clip, neginf=-self.position_clip)
        p = torch.nan_to_num(y[..., 4:7], nan=0.0, posinf=self.momentum_clip, neginf=-self.momentum_clip)
        q = q.clamp(-self.position_clip, self.position_clip)
        p = p.clamp(-self.momentum_clip, self.momentum_clip)
        return q, p

    def sparse_rolling_interaction(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim < 2:
            raise ValueError("Clifford state must include an atom dimension")
        accum = torch.zeros_like(y)
        if y.size(-2) <= 1:
            return accum
        for shift in self.rolling_shifts:
            rolled = torch.roll(y, shifts=-shift, dims=-2)
            accum = accum + clifford_geometric_product(y, rolled) / float(max(shift, 1))
        return self.interaction_scale * accum / float(max(len(self.rolling_shifts), 1))

    def _hamiltonian_terms(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q: torch.Tensor,
        p: torch.Tensor,
        *,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> HamiltonianTerms:
        terms = hamiltonian(
            q,
            p,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
            return_terms=True,
        )
        return HamiltonianTerms(
            kinetic=torch.nan_to_num(terms.kinetic, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            physical=torch.nan_to_num(terms.physical, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            reactive=torch.nan_to_num(terms.reactive, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            total=torch.nan_to_num(terms.total, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
        )

    def vector_field(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        y: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        if self.derivative_fn is not None:
            base = self.derivative_fn(y, torch.as_tensor(0.0 if t is None else t, dtype=y.dtype, device=y.device))
            return base + self.sparse_rolling_interaction(y)
        q, p = self.multivector_to_phase_space(y)
        return (
            hamiltonian.compute_clifford_force(
                q,
                p,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            + self.sparse_rolling_interaction(y)
        )

    def _left_multiply(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        exp_u = clifford_exp(u, terms=self.exp_terms)
        return clifford_geometric_product(exp_u, y)

    def _algebra_field(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        y0: torch.Tensor,
        u: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        y = self._left_multiply(u, y0)
        w = self.vector_field(
            hamiltonian,
            y,
            t=t,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        return dexp_inv(u, w, order=self.dexp_order)

    def step(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        *,
        hamiltonian: Optional[NEXUS_Hamiltonian] = None,
        dt_t: Optional[torch.Tensor] = None,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        if hamiltonian is None and self.derivative_fn is None:
            raise ValueError("Either hamiltonian or derivative_fn must be provided")
        if dt_t is None:
            dt_t = torch.as_tensor(self.h, dtype=y.dtype, device=y.device)
        zero = torch.zeros_like(y)
        k1 = self._algebra_field(
            hamiltonian,
            y,
            zero,
            t=t,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        u2 = 0.5 * dt_t * k1
        y2 = self._left_multiply(u2, y)
        k2 = dexp_inv(
            u2,
            self.vector_field(
                hamiltonian,
                y2,
                t=t + 0.5 * dt_t,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            ),
            order=self.dexp_order,
        )
        u3 = 0.5 * dt_t * k2
        y3 = self._left_multiply(u3, y)
        k3 = dexp_inv(
            u3,
            self.vector_field(
                hamiltonian,
                y3,
                t=t + 0.5 * dt_t,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            ),
            order=self.dexp_order,
        )
        u4 = dt_t * k3
        y4 = self._left_multiply(u4, y)
        k4 = dexp_inv(
            u4,
            self.vector_field(
                hamiltonian,
                y4,
                t=t + dt_t,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            ),
            order=self.dexp_order,
        )
        u = dt_t * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return self._left_multiply(u, y)

    def _apply_thermostat(self, p: torch.Tensor, kinetic: torch.Tensor) -> torch.Tensor:
        if not self.thermostat:
            return p
        if float(kinetic.detach().cpu().item()) <= self.kinetic_threshold:
            return p
        return p * self.damping

    def integrate(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        *,
        steps: int,
        dt: float,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NEXUS_Trajectory:
        if species is None:
            raise ValueError("species must be provided for CliffordLieIntegrator")
        q = q_init.clone().requires_grad_(True)
        p = p_init.clone().requires_grad_(True)
        y = self.phase_space_to_multivector(q, p).requires_grad_(True)
        dt_t = torch.as_tensor(float(dt), dtype=q.dtype, device=q.device)
        t = torch.zeros((), dtype=q.dtype, device=q.device)

        q_hist = [q]
        p_hist = [p]
        y_hist = [y]
        h_hist = []
        kinetic_hist = []
        physical_hist = []
        reactive_hist = []
        lagrangian_terms = []

        terms0 = self._hamiltonian_terms(
            hamiltonian,
            q,
            p,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        h_hist.append(terms0.total)
        kinetic_hist.append(terms0.kinetic)
        physical_hist.append(terms0.physical)
        reactive_hist.append(terms0.reactive)
        lagrangian_terms.append(terms0.kinetic - (terms0.physical + hamiltonian.coupling_lambda * terms0.reactive))

        for _ in range(int(steps)):
            y = self.step(
                y,
                t,
                hamiltonian=hamiltonian,
                dt_t=dt_t,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            ).requires_grad_(True)
            t = t + dt_t
            q, p = self.multivector_to_phase_space(y)
            terms_pred = self._hamiltonian_terms(
                hamiltonian,
                q,
                p,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            p = self._apply_thermostat(p, terms_pred.kinetic).requires_grad_(True)
            y = self.phase_space_to_multivector(q, p).requires_grad_(True)
            terms_final = self._hamiltonian_terms(
                hamiltonian,
                q,
                p,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )

            q_hist.append(q)
            p_hist.append(p)
            y_hist.append(y)
            h_hist.append(terms_final.total)
            kinetic_hist.append(terms_final.kinetic)
            physical_hist.append(terms_final.physical)
            reactive_hist.append(terms_final.reactive)
            lagrangian_terms.append(
                terms_final.kinetic - (terms_final.physical + hamiltonian.coupling_lambda * terms_final.reactive)
            )

        q_path = torch.stack(q_hist, dim=0)
        p_path = torch.stack(p_hist, dim=0)
        y_path = torch.stack(y_hist, dim=0)
        h_path = torch.stack(h_hist, dim=0)
        kinetic_path = torch.stack(kinetic_hist, dim=0)
        physical_path = torch.stack(physical_hist, dim=0)
        reactive_path = torch.stack(reactive_hist, dim=0)
        action_integral = dt_t * torch.stack(lagrangian_terms, dim=0).sum()
        hamiltonian_drift = h_path[-1] - h_path[0]

        return NEXUS_Trajectory(
            q_path=q_path,
            p_path=p_path,
            h_path=h_path,
            hamiltonian_drift=hamiltonian_drift,
            action_integral=action_integral,
            kinetic_path=kinetic_path,
            physical_path=physical_path,
            reactive_path=reactive_path,
            lie_state_path=y_path,
        )

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        *,
        steps: int,
        dt: float,
        smiles: str,
        species: Optional[torch.Tensor] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NEXUS_Trajectory:
        return self.integrate(
            hamiltonian,
            q_init,
            p_init,
            steps=steps,
            dt=dt,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )


def _integrate_terminal_state(
    integrator: CliffordLieIntegrator,
    hamiltonian: NEXUS_Hamiltonian,
    q_init: torch.Tensor,
    p_init: torch.Tensor,
    t_span: torch.Tensor,
    *,
    smiles: str,
    species: torch.Tensor,
    checkpoint_stride: int = 0,
):
    steps = max(int(t_span.numel()) - 1, 1)
    q = q_init
    p = p_init
    y = integrator.phase_space_to_multivector(q, p)
    t = t_span[0]
    checkpoints = []
    start_terms = integrator._hamiltonian_terms(hamiltonian, q, p, smiles=smiles, species=species)
    if checkpoint_stride > 0:
        checkpoints.append((0, q.detach(), p.detach(), y.detach()))
    for idx in range(steps):
        dt_t = (t_span[idx + 1] - t_span[idx]).to(dtype=q.dtype, device=q.device)
        y = integrator.step(y, t, hamiltonian=hamiltonian, dt_t=dt_t, smiles=smiles, species=species)
        q, p = integrator.multivector_to_phase_space(y)
        terms_pred = integrator._hamiltonian_terms(hamiltonian, q, p, smiles=smiles, species=species)
        p = integrator._apply_thermostat(p, terms_pred.kinetic)
        y = integrator.phase_space_to_multivector(q, p)
        t = t_span[idx + 1]
        if checkpoint_stride > 0 and ((idx + 1) % checkpoint_stride == 0 or idx + 1 == steps):
            checkpoints.append((idx + 1, q.detach(), p.detach(), y.detach()))
    final_terms = integrator._hamiltonian_terms(hamiltonian, q, p, smiles=smiles, species=species)
    return q, p, y, start_terms.total, final_terms.total, checkpoints


class _SymplecticAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q0: torch.Tensor,
        p0: torch.Tensor,
        t_span: torch.Tensor,
        species: torch.Tensor,
        integrator,
        hamiltonian,
        smiles: str,
        drift_threshold: float,
        checkpoint_stride: int,
        *params: torch.Tensor,
    ):
        del params
        ctx.integrator = integrator
        ctx.hamiltonian = hamiltonian
        ctx.smiles = smiles
        ctx.drift_threshold = float(drift_threshold)
        ctx.checkpoint_stride = int(checkpoint_stride)
        ctx.save_for_backward(q0.detach(), p0.detach(), t_span.detach(), species.detach())
        with torch.no_grad():
            qT, pT, _, start_h, end_h, checkpoints = _integrate_terminal_state(
                integrator,
                hamiltonian,
                q0.detach(),
                p0.detach(),
                t_span.detach(),
                smiles=smiles,
                species=species.detach(),
                checkpoint_stride=int(checkpoint_stride),
            )
        ctx.forward_drift = (end_h - start_h).detach()
        ctx.checkpoints = checkpoints
        return qT, pT

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_p: torch.Tensor):
        q0, p0, t_span, species = ctx.saved_tensors
        params = tuple(p for p in ctx.hamiltonian.parameters() if p.requires_grad)
        with torch.enable_grad():
            q0_r = q0.clone().requires_grad_(True)
            p0_r = p0.clone().requires_grad_(True)
            qT, pT, _, start_h, end_h, _ = _integrate_terminal_state(
                ctx.integrator,
                ctx.hamiltonian,
                q0_r,
                p0_r,
                t_span,
                smiles=ctx.smiles,
                species=species,
                checkpoint_stride=0,
            )
            grads = torch.autograd.grad(
                outputs=(qT, pT),
                inputs=(q0_r, p0_r, *params),
                grad_outputs=(grad_q, grad_p),
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
        backward_drift = (end_h - start_h).detach()
        if float(backward_drift.abs().detach().cpu().item()) > ctx.drift_threshold:
            warnings.warn(
                f"Symplectic adjoint drift {float(backward_drift.detach().cpu().item()):.3e} exceeded threshold {ctx.drift_threshold:.1e}",
                RuntimeWarning,
            )
        return (
            grads[0],
            grads[1],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grads[2:],
        )


class SymplecticAdjointSolver(nn.Module):
    def __init__(
        self,
        integrator: Optional[CliffordLieIntegrator] = None,
        drift_threshold: float = 1.0e-5,
        checkpoint_stride: int = 0,
    ) -> None:
        super().__init__()
        self.integrator = integrator or CliffordLieIntegrator()
        self.drift_threshold = float(drift_threshold)
        self.checkpoint_stride = int(checkpoint_stride)
        self.last_diagnostics: Optional[AdjointDiagnostics] = None

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q0: torch.Tensor,
        p0: torch.Tensor,
        *,
        t_span: torch.Tensor,
        smiles: str,
        species: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = tuple(p for p in hamiltonian.parameters() if p.requires_grad)
        stride = self.checkpoint_stride
        if stride <= 0:
            stride = max(1, int(math.sqrt(max(int(t_span.numel()) - 1, 1))))
        qT, pT = _SymplecticAdjointFunction.apply(
            q0,
            p0,
            t_span,
            species,
            self.integrator,
            hamiltonian,
            smiles,
            self.drift_threshold,
            stride,
            *params,
        )
        with torch.no_grad():
            _, _, _, start_h, end_h, checkpoints = _integrate_terminal_state(
                self.integrator,
                hamiltonian,
                q0.detach(),
                p0.detach(),
                t_span.detach(),
                smiles=smiles,
                species=species.detach(),
                checkpoint_stride=stride,
            )
        self.last_diagnostics = AdjointDiagnostics(
            forward_drift=(end_h - start_h).detach(),
            backward_drift=(end_h - start_h).detach(),
            checkpoint_count=len(checkpoints),
        )
        return qT, pT


class PNODEWrapper(nn.Module):
    def __init__(
        self,
        adjoint_solver: Optional[SymplecticAdjointSolver] = None,
        implicit: bool = False,
        implicit_iters: int = 2,
    ) -> None:
        super().__init__()
        self.adjoint_solver = adjoint_solver or SymplecticAdjointSolver()
        self.implicit = bool(implicit)
        self.implicit_iters = int(implicit_iters)

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q0: torch.Tensor,
        p0: torch.Tensor,
        *,
        t_span: torch.Tensor,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qT, pT = self.adjoint_solver(
            hamiltonian,
            q0,
            p0,
            t_span=t_span,
            smiles=smiles,
            species=species,
        )
        if not self.implicit:
            return qT, pT
        q_cur, p_cur = qT, pT
        dt = (t_span[-1] - t_span[-2]).to(dtype=q0.dtype, device=q0.device) if t_span.numel() > 1 else torch.as_tensor(
            self.adjoint_solver.integrator.h,
            dtype=q0.dtype,
            device=q0.device,
        )
        for _ in range(max(self.implicit_iters, 1)):
            force = hamiltonian.compute_force(
                q_cur,
                smiles=smiles,
                species=species,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
            vel = hamiltonian.compute_velocity(p_cur, q=q_cur, smiles=smiles, species=species)
            q_cur = 0.5 * (q_cur + (qT + dt * vel))
            p_cur = 0.5 * (p_cur + (pT + dt * force))
        return q_cur, p_cur


__all__ = ["CliffordIntegratorState", "CliffordLieIntegrator"]


class NeuralPathOptimizer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        samples: int = 9,
        alpha: float = 0.1,
        steps: int = 8,
        learning_rate: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.samples = int(samples)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.path_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )

    def path(
        self,
        y_reactant: torch.Tensor,
        y_product: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        s_scalar = s.reshape(s.size(0), 1)
        s_broadcast = s_scalar.reshape(s.size(0), *([1] * y_reactant.ndim))
        linear = (1.0 - s_broadcast) * y_reactant.unsqueeze(0) + s_broadcast * y_product.unsqueeze(0)
        residual = self.path_net(s_scalar.to(dtype=linear.dtype)).unsqueeze(-2)
        return linear + 0.1 * residual

    def _energy_from_path(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        solver: CliffordLieIntegrator,
        y: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        q, p = solver.multivector_to_phase_space(y)
        terms = hamiltonian(
            q,
            p,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
            return_terms=True,
        )
        return terms.physical + hamiltonian.coupling_lambda * terms.reactive

    def optimize(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        solver: CliffordLieIntegrator,
        y_reactant: torch.Tensor,
        y_product: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NeuralPathOptimizationResult:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        s_grid = torch.linspace(0.0, 1.0, self.samples, dtype=y_reactant.dtype, device=y_reactant.device).unsqueeze(-1)
        final_path = None
        final_energies = None
        final_loss = None
        for _ in range(self.steps):
            optimizer.zero_grad()
            path_points = self.path(y_reactant, y_product, s_grid)
            path_points = path_points.requires_grad_(True)
            energies = []
            grad_perp_terms = []
            for idx in range(path_points.size(0)):
                y_s = path_points[idx]
                energy = self._energy_from_path(
                    hamiltonian,
                    solver,
                    y_s,
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                energies.append(energy)
                grad_e = torch.autograd.grad(
                    energy,
                    y_s,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                if 0 < idx < path_points.size(0) - 1:
                    tangent = path_points[idx + 1] - path_points[idx - 1]
                elif idx == 0:
                    tangent = path_points[idx + 1] - path_points[idx]
                else:
                    tangent = path_points[idx] - path_points[idx - 1]
                tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
                proj = (grad_e * tangent).sum(dim=-1, keepdim=True) * tangent
                grad_perp_terms.append((grad_e - proj).pow(2).mean())
            energies_t = torch.stack(energies, dim=0)
            arc = path_points[1:] - path_points[:-1]
            arc_penalty = ((arc.norm(dim=(-1, -2)) - 1.0) ** 2).mean()
            grad_penalty = torch.stack(grad_perp_terms, dim=0).mean()
            loss = grad_penalty + self.alpha * arc_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            final_path = path_points.detach()
            final_energies = energies_t.detach()
            final_loss = loss.detach()
        return NeuralPathOptimizationResult(
            path_points=final_path,
            energies=final_energies,
            loss=final_loss,
        )


__all__ = [
    "AdjointDiagnostics",
    "CayleyPropagator",
    "CliffordIntegratorState",
    "CliffordLieIntegrator",
    "NeuralPathOptimizationResult",
    "NeuralPathOptimizer",
    "PNODEWrapper",
    "SymplecticAdjointSolver",
]
