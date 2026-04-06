from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from enzyme_software.liquid_nn_v2._compat import require_torch, torch


@dataclass
class CandidateSetSample:
    molecule_id: str
    canonical_smiles: str
    source: str
    primary_cyp: str
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_mask: torch.Tensor
    candidate_atom_indices: torch.Tensor
    proposal_scores: torch.Tensor
    proposal_top1_index: int
    proposal_top1_is_true: bool
    true_site_atoms: List[int]
    candidate_local_rival_mask: Optional[torch.Tensor] = None
    candidate_graph_distance: Optional[torch.Tensor] = None
    candidate_3d_distance: Optional[torch.Tensor] = None
    candidate_same_ring_system: Optional[torch.Tensor] = None
    candidate_same_topology_role: Optional[torch.Tensor] = None
    candidate_same_chem_family: Optional[torch.Tensor] = None
    candidate_branch_bulk: Optional[torch.Tensor] = None
    candidate_exposed_span: Optional[torch.Tensor] = None
    candidate_anti_score: Optional[torch.Tensor] = None


class CandidateSetDataset(Dataset):
    def __init__(self, cache_path: str | Path, *, split: str):
        require_torch()
        self.cache_path = Path(cache_path)
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.meta = dict(payload.get("meta") or {})
        split_payload = dict((payload.get("splits") or {}).get(str(split), {}))
        self.split = str(split)
        self.summary = dict(split_payload.get("summary") or {})
        self.samples = list(split_payload.get("samples") or [])

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, idx: int) -> CandidateSetSample:
        row = dict(self.samples[int(idx)])
        return CandidateSetSample(
            molecule_id=str(row.get("molecule_id", "")),
            canonical_smiles=str(row.get("canonical_smiles", "")),
            source=str(row.get("source", "")),
            primary_cyp=str(row.get("primary_cyp", "")),
            candidate_features=torch.as_tensor(row.get("candidate_features"), dtype=torch.float32),
            candidate_mask=torch.as_tensor(row.get("candidate_mask"), dtype=torch.float32),
            target_mask=torch.as_tensor(row.get("target_mask"), dtype=torch.float32),
            candidate_atom_indices=torch.as_tensor(row.get("candidate_atom_indices"), dtype=torch.long),
            proposal_scores=torch.as_tensor(row.get("proposal_scores"), dtype=torch.float32),
            proposal_top1_index=int(row.get("proposal_top1_index", 0)),
            proposal_top1_is_true=bool(row.get("proposal_top1_is_true", False)),
            true_site_atoms=[int(v) for v in list(row.get("true_site_atoms") or [])],
            candidate_local_rival_mask=(
                torch.as_tensor(row.get("candidate_local_rival_mask"), dtype=torch.float32)
                if row.get("candidate_local_rival_mask") is not None
                else None
            ),
            candidate_graph_distance=(
                torch.as_tensor(row.get("candidate_graph_distance"), dtype=torch.float32)
                if row.get("candidate_graph_distance") is not None
                else None
            ),
            candidate_3d_distance=(
                torch.as_tensor(row.get("candidate_3d_distance"), dtype=torch.float32)
                if row.get("candidate_3d_distance") is not None
                else None
            ),
            candidate_same_ring_system=(
                torch.as_tensor(row.get("candidate_same_ring_system"), dtype=torch.float32)
                if row.get("candidate_same_ring_system") is not None
                else None
            ),
            candidate_same_topology_role=(
                torch.as_tensor(row.get("candidate_same_topology_role"), dtype=torch.float32)
                if row.get("candidate_same_topology_role") is not None
                else None
            ),
            candidate_same_chem_family=(
                torch.as_tensor(row.get("candidate_same_chem_family"), dtype=torch.float32)
                if row.get("candidate_same_chem_family") is not None
                else None
            ),
            candidate_branch_bulk=(
                torch.as_tensor(row.get("candidate_branch_bulk"), dtype=torch.float32)
                if row.get("candidate_branch_bulk") is not None
                else None
            ),
            candidate_exposed_span=(
                torch.as_tensor(row.get("candidate_exposed_span"), dtype=torch.float32)
                if row.get("candidate_exposed_span") is not None
                else None
            ),
            candidate_anti_score=(
                torch.as_tensor(row.get("candidate_anti_score"), dtype=torch.float32)
                if row.get("candidate_anti_score") is not None
                else None
            ),
        )


def collate_candidate_sets(batch: List[CandidateSetSample]) -> Dict[str, object]:
    require_torch()
    if not batch:
        raise ValueError("collate_candidate_sets received an empty batch")
    max_candidates = max(int(sample.candidate_features.shape[0]) for sample in batch)
    feature_dim = int(batch[0].candidate_features.shape[-1])

    def _pad_2d(tensor: torch.Tensor, *, fill: float = 0.0) -> torch.Tensor:
        rows = int(tensor.shape[0])
        if rows >= max_candidates:
            return tensor
        pad = torch.full((max_candidates - rows, tensor.shape[1]), fill_value=fill, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    def _pad_1d(tensor: torch.Tensor, *, fill: float = 0.0, dtype=None) -> torch.Tensor:
        rows = int(tensor.shape[0])
        if rows >= max_candidates:
            return tensor
        pad = torch.full((max_candidates - rows,), fill_value=fill, dtype=dtype or tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    def _pad_square(tensor: torch.Tensor, *, fill: float = 0.0) -> torch.Tensor:
        rows = int(tensor.shape[0])
        cols = int(tensor.shape[1])
        out = torch.full((max_candidates, max_candidates), fill_value=fill, dtype=tensor.dtype)
        out[:rows, :cols] = tensor
        return out

    def _stack_optional_1d(name: str, *, fill: float = 0.0):
        values = [getattr(sample, name) for sample in batch]
        if not any(value is not None for value in values):
            return None
        return torch.stack([
            _pad_1d(value.view(-1) if value is not None else torch.zeros((0,), dtype=torch.float32), fill=fill)
            for value in values
        ], dim=0)

    def _stack_optional_square(name: str, *, fill: float = 0.0):
        values = [getattr(sample, name) for sample in batch]
        if not any(value is not None for value in values):
            return None
        return torch.stack([
            _pad_square(value.view(value.shape[0], value.shape[1]) if value is not None else torch.zeros((0, 0), dtype=torch.float32), fill=fill)
            for value in values
        ], dim=0)

    collated = {
        "molecule_id": [sample.molecule_id for sample in batch],
        "canonical_smiles": [sample.canonical_smiles for sample in batch],
        "source": [sample.source for sample in batch],
        "primary_cyp": [sample.primary_cyp for sample in batch],
        "candidate_features": torch.stack([
            _pad_2d(sample.candidate_features.view(-1, feature_dim), fill=0.0) for sample in batch
        ], dim=0),
        "candidate_mask": torch.stack([
            _pad_1d(sample.candidate_mask.view(-1), fill=0.0) for sample in batch
        ], dim=0),
        "target_mask": torch.stack([
            _pad_1d(sample.target_mask.view(-1), fill=0.0) for sample in batch
        ], dim=0),
        "candidate_atom_indices": torch.stack([
            _pad_1d(sample.candidate_atom_indices.view(-1), fill=-1, dtype=torch.long) for sample in batch
        ], dim=0),
        "proposal_scores": torch.stack([
            _pad_1d(sample.proposal_scores.view(-1), fill=-20.0) for sample in batch
        ], dim=0),
        "proposal_top1_index": torch.as_tensor([sample.proposal_top1_index for sample in batch], dtype=torch.long),
        "proposal_top1_is_true": torch.as_tensor([1.0 if sample.proposal_top1_is_true else 0.0 for sample in batch], dtype=torch.float32),
        "true_site_atoms": [list(sample.true_site_atoms) for sample in batch],
    }
    for key, value in {
        "candidate_local_rival_mask": _stack_optional_square("candidate_local_rival_mask", fill=0.0),
        "candidate_graph_distance": _stack_optional_square("candidate_graph_distance", fill=99.0),
        "candidate_3d_distance": _stack_optional_square("candidate_3d_distance", fill=99.0),
        "candidate_same_ring_system": _stack_optional_square("candidate_same_ring_system", fill=0.0),
        "candidate_same_topology_role": _stack_optional_square("candidate_same_topology_role", fill=0.0),
        "candidate_same_chem_family": _stack_optional_square("candidate_same_chem_family", fill=0.0),
        "candidate_branch_bulk": _stack_optional_1d("candidate_branch_bulk", fill=0.0),
        "candidate_exposed_span": _stack_optional_1d("candidate_exposed_span", fill=0.0),
        "candidate_anti_score": _stack_optional_1d("candidate_anti_score", fill=0.0),
    }.items():
        if value is not None:
            collated[key] = value
    return collated
