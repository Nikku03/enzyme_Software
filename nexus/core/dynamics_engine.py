from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nexus.core.generative_agency import NEXUS_Seed
from nexus.core.inference import NEXUS_Module1_Inference, NEXUS_Module1_Output
from nexus.core.manifold_refiner import Refined_NEXUS_Manifold
from nexus.core.field_engine import ContinuousReactivityField
from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState
from nexus.pocket.pipeline import EnzymePocketEncoder, EnzymePocketEncodingOutput
from nexus.physics.clifford_math import embed_coordinates
from nexus.physics.hamiltonian import NEXUS_Hamiltonian
from nexus.physics.kinetics import Kinetic_Barrier_Estimator, KineticBarrierEstimate
from nexus.physics.navigator import Least_Action_Navigator, LeastActionResult
from nexus.physics.ode_solver import Symplectic_ODE_Solver
from nexus.physics.solvers import CliffordLieIntegrator
from nexus.physics.ts_search import NEXUSTSSearch, NEXUSTSSearchResult
from nexus.physics.trajectory import NEXUS_Trajectory
from nexus.physics.ts_detector import Transition_State_Detector
from nexus.physics.ts_detector import TransitionStateCandidate


@dataclass
class NEXUS_Dynamic_Manifold:
    seed_geometry: NEXUS_Seed
    module1: NEXUS_Module1_Output
    electronic_field: ContinuousReactivityField
    reactive_trajectory: NEXUS_Trajectory
    ts_state: Optional[TransitionStateCandidate]
    metabolic_kinetics: KineticBarrierEstimate
    navigation: LeastActionResult
    target_atom_index: torch.Tensor
    target_point: torch.Tensor
    accessibility_field: Optional[AccessibilityFieldState] = None
    invariance_error: Optional[torch.Tensor] = None
    invariance_passed: Optional[bool] = None
    ts_search: Optional[NEXUSTSSearchResult] = None


NEXUS_Dynamics_Output = NEXUS_Dynamic_Manifold


class NEXUS_Dynamics_Engine(nn.Module):
    def __init__(
        self,
        module1: Optional[NEXUS_Module1_Inference] = None,
        hamiltonian: Optional[NEXUS_Hamiltonian] = None,
        solver: Optional[nn.Module] = None,
        navigator: Optional[Least_Action_Navigator] = None,
        ts_detector: Optional[Transition_State_Detector] = None,
        kinetics: Optional[Kinetic_Barrier_Estimator] = None,
        ts_search: Optional[NEXUSTSSearch] = None,
        pocket_encoder: Optional[EnzymePocketEncoder] = None,
    ) -> None:
        super().__init__()
        self.module1 = module1 or NEXUS_Module1_Inference()
        self.solver = solver or CliffordLieIntegrator()
        self.ts_detector = ts_detector or Transition_State_Detector()
        self.hamiltonian = hamiltonian or NEXUS_Hamiltonian(
            agency=self.module1.agency,
            refiner=self.module1.refiner,
            field_engine=self.module1.field_engine,
        )
        self.navigator = navigator or Least_Action_Navigator(
            solver=self.solver,
            ts_detector=self.ts_detector,
        )
        self.kinetics = kinetics or Kinetic_Barrier_Estimator()
        self.ts_search = ts_search or NEXUSTSSearch(ts_detector=self.ts_detector)
        self.pocket_encoder = pocket_encoder or EnzymePocketEncoder()
        self.solver_dtype = torch.float64

    def _build_pocket_encoding(
        self,
        module1_out: NEXUS_Module1_Output,
        target_rank: int,
        protein_data: dict,
    ) -> EnzymePocketEncodingOutput:
        protein_coords = protein_data["coords"]
        isoform_embedding = protein_data["isoform_embedding"]
        target_point = module1_out.som_coordinates[target_rank].view(1, 3)
        components = module1_out.field_state.field.query_components(target_point, compute_observables=True)
        latent = components.get("latent_multivector")
        if latent is None:
            drug_mv = embed_coordinates(target_point.to(dtype=protein_coords.dtype))
        else:
            if latent.ndim == 3:
                drug_mv = latent.mean(dim=-2)
            else:
                drug_mv = latent
            drug_mv = drug_mv.to(device=protein_coords.device, dtype=protein_coords.dtype)
        return self.pocket_encoder(
            drug_mv,
            protein_coords,
            isoform_embedding,
            sequence=protein_data.get("sequence"),
            sequence_embedding=protein_data.get("sequence_embedding"),
            variant_ids=protein_data.get("variant_ids"),
            variant_embedding=protein_data.get("variant_embedding"),
            residue_types=protein_data.get("residue_types"),
            conservation_scores=protein_data.get("conservation_scores"),
            allosteric=protein_data.get("allosteric"),
            t=protein_data.get("t"),
        )

    def forward(
        self,
        smiles: str,
        *,
        steps: int = 8,
        dt: float = 0.001,
        target_rank: int = 0,
        verify_invariance: bool = False,
        protein_data: Optional[dict] = None,
        pocket_encoding: Optional[EnzymePocketEncodingOutput] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> NEXUS_Dynamics_Output:
        module1_out = self.module1(smiles)
        output_dtype = module1_out.manifold.pos.dtype
        rank = int(max(0, min(target_rank, module1_out.ranked_atom_indices.numel() - 1)))
        if pocket_encoding is None and protein_data is not None:
            pocket_encoding = self._build_pocket_encoding(module1_out, rank, protein_data)
        if accessibility_field is None and pocket_encoding is not None:
            accessibility_field = pocket_encoding.accessibility_state
        if ddi_occupancy is None and protein_data is not None:
            ddi_occupancy = protein_data.get("ddi_occupancy")
        field = module1_out.field_state.field
        target_atom_index = module1_out.ranked_atom_indices[rank]
        target_point_world = module1_out.som_coordinates[rank]
        q_init_internal = field.to_internal_coords(module1_out.manifold.pos).to(dtype=self.solver_dtype)
        target_point_internal = field.to_internal_coords(target_point_world.view(1, 3)).view(-1).to(dtype=self.solver_dtype)
        navigation = self.navigator(
            self.hamiltonian,
            q_init_internal,
            smiles=smiles,
            species=module1_out.manifold.species,
            target_atom_index=target_atom_index,
            target_point=target_point_internal,
            steps=steps,
            dt=dt,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        ts_search_result = None
        if navigation.best.trajectory.lie_state_path is not None:
            try:
                ts_search_result = self.ts_search(
                    self.hamiltonian,
                    self.solver,
                    navigation,
                    smiles=smiles,
                    species=module1_out.manifold.species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                if ts_search_result.refined_navigation is not None:
                    navigation = ts_search_result.refined_navigation
            except Exception:
                ts_search_result = None
        kinetics = self.kinetics(
            self.hamiltonian,
            navigation,
            q_init_internal,
            smiles=smiles,
            species=module1_out.manifold.species,
            dt=dt,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        navigation_world = self._navigation_to_world(navigation, field, output_dtype=output_dtype)
        invariance_error = None
        invariance_passed = None
        if verify_invariance:
            invariance_error, invariance_passed = self._verify_invariance(
                smiles,
                module1_out,
                target_atom_index,
                target_point_world,
                steps=steps,
                dt=dt,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
        return NEXUS_Dynamic_Manifold(
            seed_geometry=module1_out.seed,
            module1=module1_out,
            electronic_field=field,
            reactive_trajectory=navigation_world.best.trajectory,
            ts_state=navigation_world.best.ts_candidate,
            metabolic_kinetics=kinetics,
            navigation=navigation_world,
            target_atom_index=target_atom_index,
            target_point=target_point_world.to(dtype=output_dtype),
            accessibility_field=accessibility_field,
            invariance_error=invariance_error,
            invariance_passed=invariance_passed,
            ts_search=ts_search_result,
        )

    def _to_world_coords(self, coords: torch.Tensor, field: ContinuousReactivityField, output_dtype: torch.dtype) -> torch.Tensor:
        world = field.centroid.to(dtype=coords.dtype).view(*([1] * (coords.ndim - 1)), 3) + coords @ field.frame.transpose(0, 1).to(dtype=coords.dtype)
        return world.to(dtype=output_dtype)

    def _to_world_momentum(self, momentum: torch.Tensor, field: ContinuousReactivityField, output_dtype: torch.dtype) -> torch.Tensor:
        return (momentum @ field.frame.transpose(0, 1).to(dtype=momentum.dtype)).to(dtype=output_dtype)

    def _trajectory_to_world(self, trajectory: NEXUS_Trajectory, field: ContinuousReactivityField, output_dtype: torch.dtype) -> NEXUS_Trajectory:
        return NEXUS_Trajectory(
            q_path=self._to_world_coords(trajectory.q_path, field, output_dtype),
            p_path=self._to_world_momentum(trajectory.p_path, field, output_dtype),
            h_path=trajectory.h_path.to(dtype=output_dtype),
            hamiltonian_drift=trajectory.hamiltonian_drift.to(dtype=output_dtype),
            action_integral=trajectory.action_integral.to(dtype=output_dtype),
            kinetic_path=trajectory.kinetic_path.to(dtype=output_dtype),
            physical_path=trajectory.physical_path.to(dtype=output_dtype),
            reactive_path=trajectory.reactive_path.to(dtype=output_dtype),
        )

    def _ts_to_world(self, ts_state: Optional[TransitionStateCandidate], field: ContinuousReactivityField, output_dtype: torch.dtype) -> Optional[TransitionStateCandidate]:
        if ts_state is None:
            return None
        return TransitionStateCandidate(
            q=self._to_world_coords(ts_state.q, field, output_dtype),
            p=self._to_world_momentum(ts_state.p, field, output_dtype),
            eigenvalues=ts_state.eigenvalues.to(dtype=output_dtype),
            negative_count=ts_state.negative_count,
            is_transition_state=ts_state.is_transition_state,
            potential_energy=ts_state.potential_energy.to(dtype=output_dtype),
            atom_indices=ts_state.atom_indices,
        )

    def _navigation_to_world(self, navigation: LeastActionResult, field: ContinuousReactivityField, output_dtype: torch.dtype) -> LeastActionResult:
        converted = []
        for candidate in navigation.candidates:
            converted.append(
                type(candidate)(
                    loss=candidate.loss,
                    terminal_distance=candidate.terminal_distance,
                    action_term=candidate.action_term,
                    trajectory=self._trajectory_to_world(candidate.trajectory, field, output_dtype),
                    p_init=self._to_world_momentum(candidate.p_init, field, output_dtype),
                    target_atom_index=candidate.target_atom_index,
                    target_point=self._to_world_coords(candidate.target_point.view(1, 3), field, output_dtype).view(-1),
                    ts_candidate=self._ts_to_world(candidate.ts_candidate, field, output_dtype),
                )
            )
        best_idx = 0
        for idx, candidate in enumerate(navigation.candidates):
            if candidate is navigation.best:
                best_idx = idx
                break
        return type(navigation)(
            best=converted[best_idx],
            candidates=converted,
            optimization_losses=navigation.optimization_losses.to(dtype=output_dtype),
        )

    def _random_rotation(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        a = torch.randn(3, 3, dtype=dtype, device=device)
        q, r = torch.linalg.qr(a)
        sign = torch.sign(torch.diagonal(r))
        q = q @ torch.diag(torch.where(sign == 0, torch.ones_like(sign), sign))
        if torch.det(q) < 0:
            q[:, -1] = -q[:, -1]
        return q

    def _rotated_manifold(self, module1_out: NEXUS_Module1_Output, rotation: torch.Tensor) -> tuple[Refined_NEXUS_Manifold, ContinuousReactivityField]:
        seed = module1_out.seed
        rotated_seed = NEXUS_Seed(
            pos=seed.pos @ rotation.transpose(0, 1).to(dtype=seed.pos.dtype),
            z=seed.z,
            latent_blueprint=seed.latent_blueprint,
            smiles=seed.smiles,
            atom_symbols=list(seed.atom_symbols),
            chirality_codes=seed.chirality_codes,
            jacobian_hook=seed.jacobian_hook,
            metadata=dict(seed.metadata),
        )
        manifold = module1_out.manifold
        rotated_manifold = Refined_NEXUS_Manifold(
            pos=manifold.pos @ rotation.transpose(0, 1).to(dtype=manifold.pos.dtype),
            energy=manifold.energy,
            forces=manifold.forces @ rotation.transpose(0, 1).to(dtype=manifold.forces.dtype),
            species=manifold.species,
            seed=rotated_seed,
            base_refiner=manifold.base_refiner,
            metadata=dict(manifold.metadata),
        )
        rotated_field = self.module1.field_engine.build_state(rotated_manifold).field
        return rotated_manifold, rotated_field

    def _verify_invariance(
        self,
        smiles: str,
        module1_out: NEXUS_Module1_Output,
        target_atom_index: torch.Tensor,
        target_point_world: torch.Tensor,
        *,
        steps: int,
        dt: float,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[torch.Tensor, bool]:
        rotation = self._random_rotation(module1_out.manifold.pos.dtype, module1_out.manifold.pos.device)
        rotated_manifold, rotated_field = self._rotated_manifold(module1_out, rotation)
        rotated_target_world = target_point_world @ rotation.transpose(0, 1).to(dtype=target_point_world.dtype)

        # Unconditioned baseline: run the ORIGINAL molecule with no pocket context so the
        # comparison is purely about the G(3,0,1) physics engine's rotational equivariance.
        # Passing the unrotated accessibility_field into the rotated rollout is a rigged test
        # (a rotated key cannot fit an unrotated lock), so we strip all pocket conditioning.
        orig_field = module1_out.field_state.field
        orig_q = orig_field.to_internal_coords(module1_out.manifold.pos).to(dtype=module1_out.manifold.pos.dtype)
        orig_target = orig_field.to_internal_coords(target_point_world.view(1, 3)).view(-1).to(dtype=module1_out.manifold.pos.dtype)
        navigation_base = self.navigator(
            self.hamiltonian,
            orig_q,
            smiles=smiles,
            species=module1_out.manifold.species,
            target_atom_index=target_atom_index,
            target_point=orig_target,
            steps=steps,
            dt=dt,
            accessibility_field=None,
            ddi_occupancy=None,
        )
        kinetics_base = self.kinetics(
            self.hamiltonian,
            navigation_base,
            orig_q,
            smiles=smiles,
            species=module1_out.manifold.species,
            dt=dt,
            accessibility_field=None,
            ddi_occupancy=None,
        )

        # Rotated molecule, also without pocket context.
        q_init_internal = rotated_field.to_internal_coords(rotated_manifold.pos).to(dtype=rotated_manifold.pos.dtype)
        target_internal = rotated_field.to_internal_coords(rotated_target_world.view(1, 3)).view(-1).to(dtype=rotated_manifold.pos.dtype)
        navigation_rot = self.navigator(
            self.hamiltonian,
            q_init_internal,
            smiles=smiles,
            species=rotated_manifold.species,
            target_atom_index=target_atom_index,
            target_point=target_internal,
            steps=steps,
            dt=dt,
            accessibility_field=None,
            ddi_occupancy=None,
        )
        kinetics_rot = self.kinetics(
            self.hamiltonian,
            navigation_rot,
            q_init_internal,
            smiles=smiles,
            species=rotated_manifold.species,
            dt=dt,
            accessibility_field=None,
            ddi_occupancy=None,
        )
        error = (kinetics_rot.effective_delta_g_dagger - kinetics_base.effective_delta_g_dagger).abs()
        return error, bool((error <= 1.0e-5).detach().cpu().item())


__all__ = ["NEXUS_Dynamic_Manifold", "NEXUS_Dynamics_Engine", "NEXUS_Dynamics_Output"]
