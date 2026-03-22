from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from nexus.core.field_engine import NEXUS_Field_State
from nexus.core.generative_agency import NEXUS_Seed
from nexus.core.inference import NEXUS_Module1_Inference, NEXUS_Module1_Output
from nexus.core.manifold_refiner import Refined_NEXUS_Manifold
from nexus.field.query_engine import SubAtomicQueryResult
from nexus.layers.dag_learner import MetabolicDAGLearner, MetabolicDAGOutput
from nexus.layers.operator_library import DifferentiableGeometricOperatorLibrary
from nexus.pocket.accessibility import AccessibilityFieldState
from nexus.pocket.ddi import DDIOccupancyState
from nexus.pocket.pipeline import EnzymePocketEncoder, EnzymePocketEncodingOutput
from nexus.physics.clifford_math import embed_coordinates


@dataclass
class RecursiveMetaboliteNode:
    node_id: int
    generation: int
    parent_node_id: Optional[int]
    source_site_index: Optional[int]
    target_site_index: Optional[int]
    operator_index: Optional[int]
    operator_name: Optional[str]
    edge_weight: torch.Tensor
    module1_output: NEXUS_Module1_Output
    dag_output: Optional[MetabolicDAGOutput] = None


@dataclass
class RecursiveMetabolismTree:
    root: RecursiveMetaboliteNode
    nodes: List[RecursiveMetaboliteNode]
    generations: List[List[int]]


class RecursiveNeuralGraphGenerator(nn.Module):
    def __init__(
        self,
        module1: Optional[NEXUS_Module1_Inference] = None,
        dag_learner: Optional[MetabolicDAGLearner] = None,
        *,
        max_generations: int = 3,
        spawn_threshold: float = 0.1,
        max_children_per_node: int = 4,
        operator_hidden_dim: int = 64,
        pocket_encoder: Optional[EnzymePocketEncoder] = None,
    ) -> None:
        super().__init__()
        self.module1 = module1 or NEXUS_Module1_Inference()
        self.dag_learner = dag_learner or MetabolicDAGLearner()
        self.max_generations = int(max_generations)
        self.spawn_threshold = float(spawn_threshold)
        self.max_children_per_node = int(max_children_per_node)
        self.operator_library = DifferentiableGeometricOperatorLibrary(
            hidden_dim=operator_hidden_dim,
        )
        self.pocket_encoder = pocket_encoder or EnzymePocketEncoder()

    def _assemble_module1_output(
        self,
        seed: NEXUS_Seed,
        manifold: Refined_NEXUS_Manifold,
        field_state: NEXUS_Field_State,
        scan: SubAtomicQueryResult,
    ) -> NEXUS_Module1_Output:
        alignment_score = 0.5 * scan.alignment_tensor.mean(dim=-1) + 0.5 * scan.l2_alignment
        order = torch.argsort(scan.effective_reactivity, descending=True)
        return NEXUS_Module1_Output(
            seed=seed,
            manifold=manifold,
            field_state=field_state,
            scan=scan,
            ranked_atom_indices=scan.atom_indices[order],
            som_coordinates=scan.refined_peak_points[order],
            psi_peak=scan.refined_peak_values[order],
            approach_vector=scan.approach_vectors[order],
            alignment_score=alignment_score[order],
            exposure_score=scan.exposure_scores[order],
            effective_reactivity=scan.effective_reactivity[order],
        )

    def _characterize_seed(self, seed: NEXUS_Seed) -> NEXUS_Module1_Output:
        manifold = self.module1.refiner(seed)
        _ = self.module1.symmetry_engine(manifold)
        field_state = self.module1.field_engine.build_state(manifold)
        scan = field_state.field.scan_reaction_volume(manifold)
        return self._assemble_module1_output(seed, manifold, field_state, scan)

    def _node_multivectors(self, module1_output: NEXUS_Module1_Output) -> torch.Tensor:
        components = module1_output.field_state.field.query_components(
            module1_output.scan.refined_peak_points,
            compute_observables=True,
        )
        latent = components.get("latent_multivector")
        if latent is None:
            return embed_coordinates(module1_output.scan.refined_peak_points)
        if latent.ndim == 3:
            return latent.mean(dim=-2)
        return latent

    def _reaction_accessibility(
        self,
        module1_output: NEXUS_Module1_Output,
        accessibility_field: AccessibilityFieldState,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
        *,
        path_length: float = 2.5,
        path_samples: int = 16,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        peaks = module1_output.scan.refined_peak_points
        site_access = accessibility_field.accessibility(peaks).clamp(0.0, 1.0)
        s = torch.linspace(
            0.0,
            1.0,
            path_samples,
            dtype=peaks.dtype,
            device=peaks.device,
        ).view(1, path_samples, 1)
        path = peaks.unsqueeze(1) + s * path_length * module1_output.scan.approach_vectors.unsqueeze(1)
        if ddi_occupancy is None:
            path_access = accessibility_field.integrate_path(path).clamp(0.0, 1.0)
        else:
            path_access = accessibility_field.integrate_path_with_occupancy(path, ddi_occupancy).clamp(0.0, 1.0)
        return site_access, path_access

    def _build_pocket_encoding(
        self,
        module1_output: NEXUS_Module1_Output,
        protein_data: dict,
    ) -> EnzymePocketEncodingOutput:
        protein_coords = protein_data["coords"]
        isoform_embedding = protein_data["isoform_embedding"]
        drug_mv = self._node_multivectors(module1_output).to(device=protein_coords.device, dtype=protein_coords.dtype)
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

    def pocket_conditioned_step(
        self,
        module1_output: NEXUS_Module1_Output,
        protein_data: dict,
        *,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[MetabolicDAGOutput, EnzymePocketEncodingOutput]:
        pocket_encoding = self._build_pocket_encoding(module1_output, protein_data)
        if ddi_occupancy is None:
            ddi_occupancy = protein_data.get("ddi_occupancy")
        multivectors = self._node_multivectors(module1_output).unsqueeze(0)
        delta_g_activations, delta_g_rxn, binding_affinity, accessibility_mask, path_accessibility = self._reaction_priors(
            module1_output,
            accessibility_field=pocket_encoding.accessibility_state,
            ddi_occupancy=ddi_occupancy,
        )
        dag_output = self.dag_learner(
            multivectors,
            delta_g_activations=delta_g_activations,
            delta_g_rxn=delta_g_rxn,
            binding_affinity=binding_affinity,
            accessibility_mask=accessibility_mask,
            path_accessibility=path_accessibility,
        )
        return dag_output, pocket_encoding

    def _reaction_priors(
        self,
        module1_output: NEXUS_Module1_Output,
        *,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        reactivity = module1_output.scan.effective_reactivity
        exposure = module1_output.scan.exposure_scores
        path_clearance = module1_output.scan.path_clearance
        accessible_fraction = module1_output.scan.accessible_fraction
        pocket_site_access = None
        pocket_path_access = None
        if accessibility_field is not None:
            pocket_site_access, pocket_path_access = self._reaction_accessibility(
                module1_output,
                accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
        reactivity_norm = (reactivity - reactivity.mean()) / reactivity.std().clamp_min(1.0e-6)
        barrier_node = 25.0 - 5.0 * torch.tanh(reactivity_norm) * exposure
        delta_g_rxn_node = 2.0 - 4.0 * torch.sigmoid(reactivity_norm) * path_clearance
        affinity_node = 1.0 - exposure * path_clearance
        accessibility_node = (exposure * path_clearance * accessible_fraction.clamp(0.0, 1.0)).sqrt().clamp(0.0, 1.0)
        if pocket_site_access is not None:
            accessibility_node = (accessibility_node * pocket_site_access).clamp(0.0, 1.0)
        barrier = 0.5 * (barrier_node.unsqueeze(0) + barrier_node.unsqueeze(1))
        delta_g_rxn = 0.5 * (delta_g_rxn_node.unsqueeze(0) + delta_g_rxn_node.unsqueeze(1))
        affinity = 0.5 * (affinity_node.unsqueeze(0) + affinity_node.unsqueeze(1))
        return (
            barrier.unsqueeze(0),
            delta_g_rxn.unsqueeze(0),
            affinity.unsqueeze(0),
            accessibility_node.unsqueeze(0),
            None if pocket_path_access is None else pocket_path_access.unsqueeze(0),
        )

    def generate_child_seed(
        self,
        parent: NEXUS_Module1_Output,
        *,
        target_site_index: int,
        operator_index: int,
        edge_weight: torch.Tensor,
        generation: int,
    ) -> NEXUS_Seed:
        seed = parent.seed
        target_atom = parent.ranked_atom_indices[target_site_index].to(dtype=torch.long)
        atom_idx = int(target_atom.detach().cpu().item())
        row_match = (parent.scan.atom_indices == target_atom).nonzero(as_tuple=False)
        approach = None
        if row_match.numel() > 0:
            row_idx = int(row_match[0, 0].detach().cpu().item())
            approach = parent.scan.approach_vectors[row_idx]
        application = self.operator_library.apply(
            seed,
            target_atom_index=atom_idx,
            operator_index=operator_index,
            edge_weight=edge_weight,
            approach_vector=approach,
        )
        child_seed = application.transformed_seed
        child_seed.metadata = dict(child_seed.metadata)
        child_seed.metadata.update(
            {
                "parent_smiles": seed.smiles,
                "generation": int(generation),
            }
        )
        return child_seed

    def _spawn_children_for_node(
        self,
        node: RecursiveMetaboliteNode,
        next_node_id: int,
        *,
        protein_data: Optional[dict] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[List[RecursiveMetaboliteNode], int]:
        parent = node.module1_output
        pocket_encoding = None
        if protein_data is not None and accessibility_field is None:
            pocket_encoding = self._build_pocket_encoding(parent, protein_data)
            accessibility_field = pocket_encoding.accessibility_state
        if ddi_occupancy is None and protein_data is not None:
            ddi_occupancy = protein_data.get("ddi_occupancy")
        multivectors = self._node_multivectors(parent).unsqueeze(0)
        delta_g_activations, delta_g_rxn, binding_affinity, accessibility_mask, path_accessibility = self._reaction_priors(
            parent,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        dag_output = self.dag_learner(
            multivectors,
            delta_g_activations=delta_g_activations,
            delta_g_rxn=delta_g_rxn,
            binding_affinity=binding_affinity,
            accessibility_mask=accessibility_mask,
            path_accessibility=path_accessibility,
        )
        node.dag_output = dag_output

        def _child_generator(batch_idx: int, src_idx: int, dst_idx: int, op_idx: int, weight: torch.Tensor):
            del batch_idx
            child_seed = self.generate_child_seed(
                parent,
                target_site_index=dst_idx,
                operator_index=op_idx,
                edge_weight=weight,
                generation=node.generation + 1,
            )
            child_output = self._characterize_seed(child_seed)
            nonlocal next_node_id
            child_node = RecursiveMetaboliteNode(
                node_id=next_node_id,
                generation=node.generation + 1,
                parent_node_id=node.node_id,
                source_site_index=src_idx,
                target_site_index=dst_idx,
                operator_index=op_idx,
                operator_name=self.operator_library.operator_name(op_idx),
                edge_weight=weight,
                module1_output=child_output,
                dag_output=None,
            )
            next_node_id += 1
            return child_node

        spawned = self.dag_learner.spawn_generation(
            dag_output.adjacency,
            threshold=self.spawn_threshold,
            max_children=self.max_children_per_node,
            operator_weights=dag_output.operator_weights,
            child_generator=_child_generator,
        )
        return list(spawned), next_node_id

    def forward(
        self,
        smiles: str,
        *,
        max_generations: Optional[int] = None,
        protein_data: Optional[dict] = None,
        pocket_encoding: Optional[EnzymePocketEncodingOutput] = None,
        accessibility_field: Optional[AccessibilityFieldState] = None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> RecursiveMetabolismTree:
        if accessibility_field is None and pocket_encoding is not None:
            accessibility_field = pocket_encoding.accessibility_state
        if ddi_occupancy is None and protein_data is not None:
            ddi_occupancy = protein_data.get("ddi_occupancy")
        root_output = self.module1(smiles)
        if pocket_encoding is None and protein_data is not None:
            pocket_encoding = self._build_pocket_encoding(root_output, protein_data)
        if accessibility_field is None and pocket_encoding is not None:
            accessibility_field = pocket_encoding.accessibility_state
        root = RecursiveMetaboliteNode(
            node_id=0,
            generation=0,
            parent_node_id=None,
            source_site_index=None,
            target_site_index=None,
            operator_index=None,
            operator_name=None,
            edge_weight=torch.ones((), dtype=root_output.manifold.pos.dtype, device=root_output.manifold.pos.device),
            module1_output=root_output,
            dag_output=None,
        )
        nodes = [root]
        generations = [[0]]
        frontier = [root]
        next_node_id = 1
        max_depth = self.max_generations if max_generations is None else int(max_generations)

        for _ in range(max_depth):
            next_frontier: List[RecursiveMetaboliteNode] = []
            gen_ids: List[int] = []
            for parent_node in frontier:
                children, next_node_id = self._spawn_children_for_node(
                    parent_node,
                    next_node_id,
                    protein_data=protein_data,
                    accessibility_field=None,
                    ddi_occupancy=ddi_occupancy,
                )
                for child in children:
                    nodes.append(child)
                    next_frontier.append(child)
                    gen_ids.append(child.node_id)
            if not next_frontier:
                break
            generations.append(gen_ids)
            frontier = next_frontier

        return RecursiveMetabolismTree(
            root=root,
            nodes=nodes,
            generations=generations,
        )


__all__ = [
    "RecursiveMetaboliteNode",
    "RecursiveMetabolismTree",
    "RecursiveNeuralGraphGenerator",
]
