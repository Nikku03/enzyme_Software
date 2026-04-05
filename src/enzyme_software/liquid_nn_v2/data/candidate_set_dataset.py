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
    return {
        "molecule_id": [sample.molecule_id for sample in batch],
        "canonical_smiles": [sample.canonical_smiles for sample in batch],
        "source": [sample.source for sample in batch],
        "primary_cyp": [sample.primary_cyp for sample in batch],
        "candidate_features": torch.stack([sample.candidate_features for sample in batch], dim=0),
        "candidate_mask": torch.stack([sample.candidate_mask for sample in batch], dim=0),
        "target_mask": torch.stack([sample.target_mask for sample in batch], dim=0),
        "candidate_atom_indices": torch.stack([sample.candidate_atom_indices for sample in batch], dim=0),
        "proposal_scores": torch.stack([sample.proposal_scores for sample in batch], dim=0),
        "proposal_top1_index": torch.as_tensor([sample.proposal_top1_index for sample in batch], dtype=torch.long),
        "proposal_top1_is_true": torch.as_tensor([1.0 if sample.proposal_top1_is_true else 0.0 for sample in batch], dtype=torch.float32),
        "true_site_atoms": [list(sample.true_site_atoms) for sample in batch],
    }
