from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from nexus.core.field_engine import FSHN_Field_Generator, NEXUS_Field_State
from nexus.core.generative_agency import NEXT_Mol_Generative_Agency, NEXUS_Seed
from nexus.core.manifold_refiner import MACE_OFF_Refiner, Refined_NEXUS_Manifold
from nexus.field.query_engine import SubAtomicQueryResult
from nexus.symmetry.engine import O3_Symmetry_Engine


@dataclass
class NEXUS_Module1_Output:
    seed: NEXUS_Seed
    manifold: Refined_NEXUS_Manifold
    field_state: NEXUS_Field_State
    scan: SubAtomicQueryResult
    ranked_atom_indices: torch.Tensor
    som_coordinates: torch.Tensor
    psi_peak: torch.Tensor
    approach_vector: torch.Tensor
    alignment_score: torch.Tensor
    exposure_score: torch.Tensor
    effective_reactivity: torch.Tensor


class NEXUS_Module1_Inference(nn.Module):
    def __init__(
        self,
        agency: NEXT_Mol_Generative_Agency | None = None,
        refiner: MACE_OFF_Refiner | None = None,
        symmetry_engine: O3_Symmetry_Engine | None = None,
        field_engine: FSHN_Field_Generator | None = None,
    ) -> None:
        super().__init__()
        self.agency = agency or NEXT_Mol_Generative_Agency("", "")
        self.refiner = refiner or MACE_OFF_Refiner()
        self.symmetry_engine = symmetry_engine or O3_Symmetry_Engine()
        self.field_engine = field_engine or FSHN_Field_Generator()

    def forward(self, smiles: str) -> NEXUS_Module1_Output:
        seed = self.agency(smiles)
        manifold = self.refiner(seed)
        _ = self.symmetry_engine(manifold)
        field_state = self.field_engine.build_state(manifold)
        scan = field_state.field.scan_reaction_volume(manifold)

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

    def as_dict(self, smiles: str) -> Dict[str, torch.Tensor]:
        out = self.forward(smiles)
        return {
            "ranked_atom_indices": out.ranked_atom_indices,
            "som_coordinates": out.som_coordinates,
            "psi_peak": out.psi_peak,
            "approach_vector": out.approach_vector,
            "alignment_score": out.alignment_score,
            "exposure_score": out.exposure_score,
            "effective_reactivity": out.effective_reactivity,
        }
