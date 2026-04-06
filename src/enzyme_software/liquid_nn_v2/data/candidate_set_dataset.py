from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

    return {
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
