from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn

from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState

from .clifford_math import embed_coordinates
from .constants import ATOMIC_MASSES
from .hamiltonian import NEXUS_Hamiltonian
from .navigator import LeastActionResult
from .solvers import CayleyPropagator
from .ts_detector import TransitionStateCandidate, compute_micro_hessian


K_B_KCAL_PER_MOL_K = 0.00198720425864083
PLANCK_TIMES_C_KCAL_PER_MOL_CM = 0.002859144
EYRING_PREFACTOR = 6.2e12


@dataclass
class RingPolymerState:
    bead_positions: torch.Tensor
    bead_momenta: torch.Tensor
    bead_multivectors: torch.Tensor
    centroid_path: torch.Tensor
    spring_energies: torch.Tensor


@dataclass
class KineticBarrierEstimate:
    classical_barrier: torch.Tensor
    zpe_initial: torch.Tensor
    zpe_ts: torch.Tensor
    zpe_correction: torch.Tensor
    delta_g_dagger: torch.Tensor
    imaginary_frequency_cm1: torch.Tensor
    wigner_kappa: torch.Tensor
    effective_delta_g_dagger: torch.Tensor
    metabolic_rate: torch.Tensor
    ts_eigenvalues: torch.Tensor
    initial_frequencies_cm1: torch.Tensor
    ts_frequencies_cm1: torch.Tensor
    quantum_rate_rpmd: torch.Tensor
    transmission_coefficient: torch.Tensor
    instanton_correction: torch.Tensor
    ring_polymer_spring_energy: torch.Tensor
    ring_polymer_state: Optional[RingPolymerState] = None


class PerturbativeCorrection(nn.Module):
    def __init__(self, cubic_scale: float = 0.02, quartic_scale: float = 0.005) -> None:
        super().__init__()
        self.cubic_scale = float(cubic_scale)
        self.quartic_scale = float(quartic_scale)

    def _finite_difference(self, values: torch.Tensor, order: int) -> torch.Tensor:
        out = values
        for _ in range(order):
            if out.numel() < 2:
                return torch.zeros(1, dtype=values.dtype, device=values.device)
            out = out[1:] - out[:-1]
        return out

    def forward(self, energy_profile: torch.Tensor) -> torch.Tensor:
        profile = energy_profile.reshape(-1)
        third = self._finite_difference(profile, 3)
        fourth = self._finite_difference(profile, 4)
        cubic_term = third.pow(2).mean().sqrt() if third.numel() > 0 else profile.new_zeros(())
        quartic_term = fourth.abs().mean() if fourth.numel() > 0 else profile.new_zeros(())
        log_corr = self.cubic_scale * cubic_term + self.quartic_scale * quartic_term
        return torch.exp(log_corr.clamp(min=0.0, max=5.0))


class RingPolymerManager(nn.Module):
    def __init__(
        self,
        n_beads: int = 32,
        temperature_kelvin: float = 310.15,
        bead_spread: float = 0.05,
        spring_scale: float = 1.0,
        propagator: Optional[CayleyPropagator] = None,
        perturbative_correction: Optional[PerturbativeCorrection] = None,
    ) -> None:
        super().__init__()
        self.n_beads = int(n_beads)
        self.temperature_kelvin = float(temperature_kelvin)
        self.bead_spread = float(bead_spread)
        self.spring_scale = float(spring_scale)
        self.propagator = propagator or CayleyPropagator()
        self.perturbative_correction = perturbative_correction or PerturbativeCorrection()

    def atomic_mass(self, atomic_number: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        z_int = int(torch.as_tensor(atomic_number).detach().cpu().item())
        mass = ATOMIC_MASSES.get(z_int, 12.0)
        return torch.as_tensor(mass, dtype=dtype, device=device)

    def ring_frequency(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        beta = 1.0 / (K_B_KCAL_PER_MOL_K * self.temperature_kelvin)
        omega = self.spring_scale * float(self.n_beads) / max(beta, 1.0e-6)
        return torch.as_tensor(omega, dtype=dtype, device=device)

    def spring_potential(
        self,
        bead_positions: torch.Tensor,
        mass: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        shifted = torch.roll(bead_positions, shifts=-1, dims=-2)
        diff = bead_positions - shifted
        return 0.5 * mass * omega.pow(2) * diff.pow(2).sum(dim=-1).sum(dim=-1)

    def _initial_offsets(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        angles = torch.linspace(0.0, 2.0 * math.pi, self.n_beads + 1, dtype=dtype, device=device)[:-1]
        basis = torch.stack(
            [
                torch.cos(angles),
                torch.sin(angles),
                0.5 * torch.cos(2.0 * angles),
            ],
            dim=-1,
        )
        basis = basis - basis.mean(dim=0, keepdim=True)
        return self.bead_spread * basis

    def build_ring_polymer(
        self,
        centroid_coord: torch.Tensor,
        centroid_momentum: torch.Tensor,
        atomic_number: torch.Tensor,
    ) -> RingPolymerState:
        dtype = centroid_coord.dtype
        device = centroid_coord.device
        mass = self.atomic_mass(atomic_number, dtype=dtype, device=device)
        omega = self.ring_frequency(dtype=dtype, device=device)
        offsets = self._initial_offsets(dtype=dtype, device=device) / mass.sqrt().clamp_min(1.0)
        bead_positions = centroid_coord.unsqueeze(0) + offsets
        bead_momenta = centroid_momentum.unsqueeze(0).expand_as(bead_positions).clone()
        bead_multivectors = embed_coordinates(bead_positions)
        bead_multivectors[..., 4:7] = bead_momenta
        bead_multivectors[..., 7] = (bead_positions * bead_momenta).sum(dim=-1)
        spring_energies = self.spring_potential(bead_positions.unsqueeze(0), mass, omega).squeeze(0)
        return RingPolymerState(
            bead_positions=bead_positions.unsqueeze(0),
            bead_momenta=bead_momenta.unsqueeze(0),
            bead_multivectors=bead_multivectors.unsqueeze(0),
            centroid_path=centroid_coord.unsqueeze(0),
            spring_energies=spring_energies.view(1),
        )

    def forward(
        self,
        centroid_path: torch.Tensor,
        centroid_momentum_path: torch.Tensor,
        atomic_number: torch.Tensor,
        *,
        dt: float,
    ) -> RingPolymerState:
        dtype = centroid_path.dtype
        device = centroid_path.device
        mass = self.atomic_mass(atomic_number, dtype=dtype, device=device)
        omega = self.ring_frequency(dtype=dtype, device=device)

        init_state = self.build_ring_polymer(centroid_path[0], centroid_momentum_path[0], atomic_number)
        bead_positions = init_state.bead_positions[0]
        bead_momenta = init_state.bead_momenta[0]

        bead_pos_hist = []
        bead_mom_hist = []
        bead_mv_hist = []
        spring_hist = []
        centroid_hist = []

        for step_idx in range(int(centroid_path.size(0))):
            if step_idx > 0:
                bead_positions, bead_momenta = self.propagator(
                    bead_positions,
                    bead_momenta,
                    mass=mass,
                    omega=omega,
                    dt=dt,
                )
                centroid_delta = centroid_path[step_idx] - centroid_path[step_idx - 1]
                bead_positions = bead_positions + centroid_delta.unsqueeze(0)
                bead_momenta = bead_momenta + (
                    centroid_momentum_path[step_idx] - centroid_momentum_path[step_idx - 1]
                ).unsqueeze(0)
            centroid_hist.append(bead_positions.mean(dim=0))
            bead_pos_hist.append(bead_positions)
            bead_mom_hist.append(bead_momenta)
            bead_mv = embed_coordinates(bead_positions)
            bead_mv[..., 4:7] = bead_momenta
            bead_mv[..., 7] = (bead_positions * bead_momenta).sum(dim=-1)
            bead_mv_hist.append(bead_mv)
            spring_hist.append(self.spring_potential(bead_positions.unsqueeze(0), mass, omega).squeeze(0))

        return RingPolymerState(
            bead_positions=torch.stack(bead_pos_hist, dim=0),
            bead_momenta=torch.stack(bead_mom_hist, dim=0),
            bead_multivectors=torch.stack(bead_mv_hist, dim=0),
            centroid_path=torch.stack(centroid_hist, dim=0),
            spring_energies=torch.stack(spring_hist, dim=0),
        )

    def calculate_quantum_rate(
        self,
        centroid_trajectory: torch.Tensor,
        *,
        classical_rate: torch.Tensor,
        barrier: torch.Tensor,
        energy_profile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del energy_profile
        reaction_coord = (centroid_trajectory - centroid_trajectory[0]).norm(dim=-1)
        forward_flux = torch.relu(reaction_coord[1:] - reaction_coord[:-1]).sum()
        backward_flux = torch.relu(reaction_coord[:-1] - reaction_coord[1:]).sum()
        transmission = forward_flux / (forward_flux + backward_flux + 1.0e-8)
        transmission = transmission.clamp(min=1.0e-3, max=1.0)
        tunneling_gain = torch.exp(torch.relu(-barrier)).clamp(max=10.0)
        quantum_rate = classical_rate * transmission * tunneling_gain
        return transmission, quantum_rate


class Kinetic_Barrier_Estimator(nn.Module):
    def __init__(
        self,
        temperature_kelvin: float = 310.15,
        frequency_scale_cm1: float = 1000.0,
        ring_polymer_manager: Optional[RingPolymerManager] = None,
    ) -> None:
        super().__init__()
        self.temperature_kelvin = float(temperature_kelvin)
        self.frequency_scale_cm1 = float(frequency_scale_cm1)
        self.ring_polymer_manager = ring_polymer_manager or RingPolymerManager(
            temperature_kelvin=temperature_kelvin
        )

    def _eigenvalues_to_frequencies_cm1(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(self.frequency_scale_cm1, dtype=eigenvalues.dtype, device=eigenvalues.device)
        return torch.sign(eigenvalues) * torch.sqrt(eigenvalues.abs().clamp_min(1.0e-12)) * scale

    def _zero_point_energy(self, frequencies_cm1: torch.Tensor) -> torch.Tensor:
        positive = frequencies_cm1[frequencies_cm1 > 0]
        if positive.numel() == 0:
            return torch.zeros((), dtype=frequencies_cm1.dtype, device=frequencies_cm1.device)
        return 0.5 * PLANCK_TIMES_C_KCAL_PER_MOL_CM * positive.sum()

    def _initial_state_frequencies(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        atom_indices: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        q_eval = q_init.clone().requires_grad_(True)

        def potential_fn(q_full: torch.Tensor) -> torch.Tensor:
            physical, reactive, _ = hamiltonian.compute_potential_energy(
                q_full,
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
            hessian, _ = compute_micro_hessian(
                q_eval,
                atom_indices,
                potential_fn,
                create_graph=False,
            )
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))
        hessian64 = hessian.to(dtype=torch.float64)
        eye = torch.eye(hessian64.size(0), dtype=hessian64.dtype, device=hessian64.device)
        eigvals = None
        for jitter in (0.0, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
            try:
                eigvals = torch.linalg.eigvalsh(hessian64 + jitter * eye).to(dtype=hessian.dtype)
                break
            except RuntimeError:
                continue
        if eigvals is None:
            eigvals = torch.zeros(hessian.size(0), dtype=hessian.dtype, device=hessian.device)
        return self._eigenvalues_to_frequencies_cm1(eigvals)

    def compute_classical_barrier(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        q_init: torch.Tensor,
        ts_state: TransitionStateCandidate,
        *,
        smiles: str,
        species: torch.Tensor,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> torch.Tensor:
        zero_init = torch.zeros_like(q_init)
        zero_ts = torch.zeros_like(ts_state.q)
        h_init = hamiltonian(
            q_init,
            zero_init,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        h_ts = hamiltonian(
            ts_state.q,
            zero_ts,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        return h_ts - h_init

    def compute_metabolic_rate(
        self,
        effective_delta_g_dagger: torch.Tensor,
        wigner_kappa: torch.Tensor,
    ) -> torch.Tensor:
        temperature = torch.as_tensor(
            self.temperature_kelvin,
            dtype=effective_delta_g_dagger.dtype,
            device=effective_delta_g_dagger.device,
        )
        kb = torch.as_tensor(
            K_B_KCAL_PER_MOL_K,
            dtype=effective_delta_g_dagger.dtype,
            device=effective_delta_g_dagger.device,
        )
        prefactor = torch.as_tensor(EYRING_PREFACTOR, dtype=effective_delta_g_dagger.dtype, device=effective_delta_g_dagger.device)
        exponent = torch.exp(-effective_delta_g_dagger / (kb * temperature).clamp_min(1.0e-8))
        return wigner_kappa * prefactor * exponent

    def forward(
        self,
        hamiltonian: NEXUS_Hamiltonian,
        navigation: LeastActionResult,
        q_init: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        dt: float = 0.001,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> KineticBarrierEstimate:
        ts_state = navigation.best.ts_candidate
        if ts_state is None:
            zeros = torch.zeros((), dtype=q_init.dtype, device=q_init.device)
            empty = torch.zeros(0, dtype=q_init.dtype, device=q_init.device)
            return KineticBarrierEstimate(
                classical_barrier=zeros,
                zpe_initial=zeros,
                zpe_ts=zeros,
                zpe_correction=zeros,
                delta_g_dagger=zeros,
                imaginary_frequency_cm1=zeros,
                wigner_kappa=torch.ones_like(zeros),
                effective_delta_g_dagger=zeros,
                metabolic_rate=zeros,
                ts_eigenvalues=empty,
                initial_frequencies_cm1=empty,
                ts_frequencies_cm1=empty,
                quantum_rate_rpmd=zeros,
                transmission_coefficient=torch.ones_like(zeros),
                instanton_correction=torch.ones_like(zeros),
                ring_polymer_spring_energy=zeros,
                ring_polymer_state=None,
            )

        classical = self.compute_classical_barrier(
            hamiltonian,
            q_init,
            ts_state,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        ts_freq = self._eigenvalues_to_frequencies_cm1(ts_state.eigenvalues)
        init_freq = self._initial_state_frequencies(
            hamiltonian,
            q_init,
            smiles=smiles,
            species=species,
            atom_indices=ts_state.atom_indices,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )

        zpe_initial = self._zero_point_energy(init_freq)
        zpe_ts = self._zero_point_energy(ts_freq[ts_freq > 0])
        zpe_correction = zpe_ts - zpe_initial
        delta_g = classical + zpe_correction

        negative_freqs = ts_freq[ts_freq < 0]
        imag_freq = negative_freqs.abs().max() if negative_freqs.numel() > 0 else torch.zeros((), dtype=delta_g.dtype, device=delta_g.device)
        temperature = torch.as_tensor(self.temperature_kelvin, dtype=delta_g.dtype, device=delta_g.device)
        kb = torch.as_tensor(K_B_KCAL_PER_MOL_K, dtype=delta_g.dtype, device=delta_g.device)
        hnu = PLANCK_TIMES_C_KCAL_PER_MOL_CM * imag_freq
        wigner = 1.0 + (1.0 / 24.0) * (hnu / (kb * temperature).clamp_min(1.0e-8)).pow(2)
        effective_delta_g = delta_g - kb * temperature * torch.log(wigner.clamp_min(1.0))
        rate = self.compute_metabolic_rate(effective_delta_g, wigner)

        target_atom_index = navigation.best.target_atom_index.to(dtype=torch.long, device=q_init.device)
        atom_idx = int(target_atom_index.detach().cpu().item())
        atom_centroid_path = navigation.best.trajectory.q_path[:, atom_idx]
        atom_momentum_path = navigation.best.trajectory.p_path[:, atom_idx]
        ring_polymer = self.ring_polymer_manager(
            atom_centroid_path,
            atom_momentum_path,
            species[target_atom_index],
            dt=dt,
        )
        instanton_correction = self.ring_polymer_manager.perturbative_correction(navigation.best.trajectory.h_path)
        transmission, quantum_rate = self.ring_polymer_manager.calculate_quantum_rate(
            ring_polymer.centroid_path,
            classical_rate=rate,
            barrier=effective_delta_g,
            energy_profile=navigation.best.trajectory.h_path,
        )
        spring_energy = ring_polymer.spring_energies.mean()
        quantum_rate = quantum_rate * instanton_correction / (1.0 + 0.01 * spring_energy)

        return KineticBarrierEstimate(
            classical_barrier=classical,
            zpe_initial=zpe_initial,
            zpe_ts=zpe_ts,
            zpe_correction=zpe_correction,
            delta_g_dagger=delta_g,
            imaginary_frequency_cm1=imag_freq,
            wigner_kappa=wigner,
            effective_delta_g_dagger=effective_delta_g,
            metabolic_rate=rate,
            ts_eigenvalues=ts_state.eigenvalues,
            initial_frequencies_cm1=init_freq,
            ts_frequencies_cm1=ts_freq,
            quantum_rate_rpmd=quantum_rate,
            transmission_coefficient=transmission,
            instanton_correction=instanton_correction,
            ring_polymer_spring_energy=spring_energy,
            ring_polymer_state=ring_polymer,
        )


__all__ = [
    "Kinetic_Barrier_Estimator",
    "KineticBarrierEstimate",
    "PerturbativeCorrection",
    "RingPolymerManager",
    "RingPolymerState",
]
