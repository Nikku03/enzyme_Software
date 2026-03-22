from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from nexus.core.multiscale_engine import MultiScale_Topology_Engine, MultiScaleEngineOutput
from nexus.field.hypernetwork import FieldConditioning, ReactivityHyperNetwork
from nexus.field.query_engine import SubAtomicQueryEngine, SubAtomicQueryResult
from nexus.field.splatter import TensorSplatter, TensorSplatterState
from nexus.field.siren_base import DynamicSIREN, SIREN_OMEGA_0
from nexus.physics.observables import QuantumDescriptorExtractor
from nexus.physics.quantum_bounds import HohenbergKohn_Field_Enforcer, QuantumBoundingBox


@dataclass
class ContinuousReactivityField:
    conditioning: FieldConditioning
    splatter_state: TensorSplatterState
    centroid: torch.Tensor
    frame: torch.Tensor
    max_radius: torch.Tensor
    engine: "FSHN_Field_Generator"
    source_output: MultiScaleEngineOutput
    atom_coords: torch.Tensor
    atomic_numbers: torch.Tensor
    total_electrons: torch.Tensor
    bounding_box: QuantumBoundingBox
    quantum_norm_factor: torch.Tensor
    quantum_enforcer: HohenbergKohn_Field_Enforcer
    query_engine: Optional[SubAtomicQueryEngine] = None

    def _siren_dtype(self) -> torch.dtype:
        return next(self.engine.siren_field.parameters()).dtype

    def to_internal_coords(self, coords: torch.Tensor) -> torch.Tensor:
        coords64 = coords.to(dtype=torch.float64)
        centered = coords64 - self.centroid.to(dtype=torch.float64).view(*([1] * (coords64.ndim - 1)), 3)
        return centered @ self.frame.to(dtype=torch.float64)

    def to_world_coords(self, coords: torch.Tensor) -> torch.Tensor:
        coords64 = coords.to(dtype=torch.float64)
        world = coords64 @ self.frame.transpose(0, 1).to(dtype=torch.float64)
        return world + self.centroid.to(dtype=torch.float64).view(*([1] * (coords64.ndim - 1)), 3)

    def raw_query(self, coords: torch.Tensor, return_latent: bool = False):
        internal = self.to_internal_coords(coords)
        siren_dtype = self._siren_dtype()
        siren_input = (internal / self.max_radius.clamp_min(1.0e-8)).to(dtype=siren_dtype)
        splat_input = internal.to(dtype=self.splatter_state.atom_coords.dtype)
        siren_out = self.engine.siren_field(
            siren_input,
            self.atom_coords.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.hidden_params.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.output_row_scale.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.output_col_scale.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.output_bias_shift.to(dtype=siren_dtype, device=siren_input.device),
            return_latent=return_latent,
        )
        if return_latent:
            siren, latent = siren_out
        else:
            siren = siren_out
        splat = self.engine.tensor_splatter(splat_input, self.splatter_state)
        total = siren + splat["total"]
        if return_latent:
            return total, latent
        return total

    def query_density(
        self,
        coords: torch.Tensor,
        total_electrons: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        internal = self.to_internal_coords(coords)
        raw = self.raw_query(coords)
        electrons = self.total_electrons if total_electrons is None else torch.as_tensor(
            total_electrons,
            dtype=self.total_electrons.dtype,
            device=self.total_electrons.device,
        )
        if total_electrons is None:
            norm_factor = self.quantum_norm_factor
        else:
            norm_factor = self.quantum_norm_factor * (electrons / self.total_electrons.clamp_min(1.0e-8))
        rho = self.quantum_enforcer.apply_cusp_envelope(
            raw.unsqueeze(-1),
            internal,
            self.atom_coords,
            self.atomic_numbers,
        )
        constrained = self.quantum_enforcer.apply_n_electron_constraint(
            rho,
            electrons,
            norm_factor,
        )
        return constrained.squeeze(-1)

    def query(self, coords: torch.Tensor) -> torch.Tensor:
        return self.query_density(coords)

    def query_components(
        self,
        coords: torch.Tensor,
        compute_observables: bool = False,
    ) -> Dict[str, torch.Tensor]:
        internal = self.to_internal_coords(coords)
        siren_dtype = self._siren_dtype()
        siren_input = (internal / self.max_radius.clamp_min(1.0e-8)).to(dtype=siren_dtype)
        splat_input = internal.to(dtype=self.splatter_state.atom_coords.dtype)
        siren_out = self.engine.siren_field(
            siren_input,
            self.atom_coords.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.hidden_params.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.output_row_scale.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.output_col_scale.to(dtype=siren_dtype, device=siren_input.device),
            self.conditioning.output_bias_shift.to(dtype=siren_dtype, device=siren_input.device),
            return_latent=compute_observables,
        )
        if compute_observables:
            siren, latent = siren_out
        else:
            siren = siren_out
            latent = None
        splat = self.engine.tensor_splatter(splat_input, self.splatter_state)
        out = dict(splat)
        raw_total = siren + splat["total"]
        rho = self.quantum_enforcer.apply_cusp_envelope(
            raw_total.unsqueeze(-1),
            internal,
            self.atom_coords,
            self.atomic_numbers,
        )
        constrained = self.quantum_enforcer.apply_n_electron_constraint(
            rho,
            self.total_electrons,
            self.quantum_norm_factor,
        ).squeeze(-1)
        out["siren"] = siren
        out["raw_total"] = raw_total
        if latent is not None:
            out["latent_multivector"] = latent
        out["cusp_envelope"] = (rho.squeeze(-1) / raw_total.pow(2).clamp_min(1.0e-12))
        out["density"] = constrained
        out["norm_factor"] = torch.as_tensor(
            self.quantum_norm_factor,
            dtype=constrained.dtype,
            device=constrained.device,
        )
        out["total"] = constrained
        if compute_observables:
            observable_bundle = self.engine.observable_extractor(self, internal, latent_features=latent)
            out["observables"] = {
                "f_plus": observable_bundle.f_plus,
                "f_minus": observable_bundle.f_minus,
                "f_dual": observable_bundle.f_dual,
                "mesp": observable_bundle.mesp,
                "density_gradient": observable_bundle.density_gradient,
                "density_laplacian": observable_bundle.density_laplacian,
                "homo_proxy": observable_bundle.homo_proxy,
                "lumo_proxy": observable_bundle.lumo_proxy,
                "homo_lumo_gap": observable_bundle.homo_lumo_gap,
                "pseudo_hamiltonian": observable_bundle.pseudo_hamiltonian,
            }
        return out

    def scan_reaction_volume(
        self,
        manifold,
        radius: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> SubAtomicQueryResult:
        if self.query_engine is None:
            raise RuntimeError("ContinuousReactivityField was built without a SubAtomicQueryEngine")
        return self.query_engine(self, manifold, radius=radius, n_points=n_points)


@dataclass
class NEXUS_Field_State:
    field: ContinuousReactivityField
    manifold: object
    source_output: MultiScaleEngineOutput


class FSHN_Field_Generator(nn.Module):
    def __init__(
        self,
        multiscale_engine: Optional[MultiScale_Topology_Engine] = None,
        query_engine: Optional[SubAtomicQueryEngine] = None,
        quantum_enforcer: Optional[HohenbergKohn_Field_Enforcer] = None,
        hidden_dim: int = 512,
        siren_layers: int = 5,
        omega_0: float = SIREN_OMEGA_0,
    ) -> None:
        super().__init__()
        if abs(float(omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"FSHN_Field_Generator requires omega_0={SIREN_OMEGA_0}")
        self.multiscale_engine = multiscale_engine or MultiScale_Topology_Engine()
        self.hyper_net = ReactivityHyperNetwork(context_dim=640, hidden_dim=hidden_dim, hidden_layers=siren_layers)
        self.siren_field = DynamicSIREN(coord_dim=3, hidden_dim=hidden_dim, hidden_layers=siren_layers, omega_0=omega_0)
        self.tensor_splatter = TensorSplatter(reconstruction_dim=64, init_alpha=0.35)
        self.query_engine = query_engine or SubAtomicQueryEngine()
        self.quantum_enforcer = quantum_enforcer or HohenbergKohn_Field_Enforcer()
        self.observable_extractor = QuantumDescriptorExtractor()
        self.omega_0 = float(omega_0)

    def build_field(
        self,
        manifold,
        source_output: Optional[MultiScaleEngineOutput] = None,
    ) -> ContinuousReactivityField:
        engine_output = source_output if source_output is not None else self.multiscale_engine(manifold)
        features = engine_output.fused_features
        conditioning = self.hyper_net(features, manifold.pos, manifold.species)
        atom_world = manifold.pos.to(dtype=manifold.pos.dtype)
        centroid = conditioning.canonical_centroid.to(dtype=torch.float64, device=atom_world.device)
        frame = conditioning.canonical_frame.to(dtype=torch.float64, device=atom_world.device)
        atom_internal = ((atom_world.to(dtype=torch.float64) - centroid.unsqueeze(0)) @ frame).to(dtype=atom_world.dtype)
        max_radius = atom_internal.to(dtype=torch.float64).norm(dim=-1).max().clamp_min(1.0)
        splatter_state = self.tensor_splatter.prepare(features, atom_internal, manifold.species)
        molecular_charge = manifold.seed.metadata.get("formal_charge", manifold.seed.metadata.get("charge", 0))
        total_electrons = self.quantum_enforcer.compute_total_electrons(
            manifold.species.to(dtype=atom_internal.dtype),
            molecular_charge=molecular_charge,
        )
        bounding_box = self.quantum_enforcer.make_bounding_box(atom_internal)

        def _raw_field_fn(query_internal: torch.Tensor) -> torch.Tensor:
            query_batched = query_internal if query_internal.ndim == 3 else query_internal.unsqueeze(0)
            outputs = []
            for points in query_batched:
                siren_input = (points / max_radius.clamp_min(1.0e-8)).to(dtype=conditioning.molecular_context.dtype)
                splat_input = points.to(dtype=splatter_state.atom_coords.dtype)
                siren = self.siren_field(
                    siren_input,
                    atom_internal.to(dtype=siren_input.dtype, device=siren_input.device),
                    conditioning.hidden_params,
                    conditioning.output_row_scale,
                    conditioning.output_col_scale,
                    conditioning.output_bias_shift,
                )
                splat = self.tensor_splatter(splat_input, splatter_state)
                outputs.append((siren + splat["total"]).unsqueeze(-1))
            return torch.stack(outputs, dim=0)

        quantum_norm_factor = self.quantum_enforcer.compute_normalization_factor(
            _raw_field_fn,
            atom_internal,
            manifold.species.to(dtype=atom_internal.dtype),
            total_electrons,
            bounding_box,
        )
        return ContinuousReactivityField(
            conditioning=conditioning,
            splatter_state=splatter_state,
            centroid=centroid,
            frame=frame,
            max_radius=max_radius,
            engine=self,
            source_output=engine_output,
            atom_coords=atom_internal,
            atomic_numbers=manifold.species.to(dtype=atom_internal.dtype),
            total_electrons=torch.as_tensor(total_electrons, dtype=atom_internal.dtype, device=atom_internal.device),
            bounding_box=bounding_box,
            quantum_norm_factor=torch.as_tensor(
                quantum_norm_factor,
                dtype=atom_internal.dtype,
                device=atom_internal.device,
            ),
            quantum_enforcer=self.quantum_enforcer,
            query_engine=self.query_engine,
        )

    def forward(
        self,
        manifold,
        query_points: Optional[torch.Tensor] = None,
        source_output: Optional[MultiScaleEngineOutput] = None,
    ):
        field = self.build_field(manifold, source_output=source_output)
        if query_points is None:
            return field
        return field.query(query_points)

    def build_state(
        self,
        manifold,
        source_output: Optional[MultiScaleEngineOutput] = None,
    ) -> NEXUS_Field_State:
        engine_output = source_output if source_output is not None else self.multiscale_engine(manifold)
        field = self.build_field(manifold, source_output=engine_output)
        return NEXUS_Field_State(field=field, manifold=manifold, source_output=engine_output)
