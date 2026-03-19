from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from enzyme_software.liquid_nn_v2._compat import require_torch
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset, collate_fn as base_collate_fn
from enzyme_software.liquid_nn_v2.features.xtb_features import attach_xtb_features_to_graph


def _load_drugs(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices"))


def filter_site_labeled_drugs(drugs: List[dict]) -> List[dict]:
    return [drug for drug in drugs if _has_site_labels(drug)]


def split_drugs(drugs: List[dict], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[dict], List[dict], List[dict]]:
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )
    shuffled = list(drugs)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


class MicroPatternExperimentDataset(Dataset):
    def __init__(
        self,
        base_dataset: CYPMetabolismDataset,
        *,
        xtb_cache_dir: str,
        compute_xtb_if_missing: bool = False,
    ):
        self.base_dataset = base_dataset
        self.xtb_cache_dir = xtb_cache_dir
        self.compute_xtb_if_missing = bool(compute_xtb_if_missing)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = dict(self.base_dataset[idx])
        graph = sample.get("graph")
        if graph is None:
            return sample
        sample["graph"] = attach_xtb_features_to_graph(
            graph,
            cache_dir=self.xtb_cache_dir,
            compute_if_missing=self.compute_xtb_if_missing,
        )
        return sample


def create_micropattern_dataloaders_from_drugs(
    train_drugs: List[dict],
    val_drugs: List[dict],
    test_drugs: List[dict],
    *,
    structure_sdf: Optional[str],
    batch_size: int,
    xtb_cache_dir: str,
    compute_xtb_if_missing: bool,
    manual_feature_cache_dir: Optional[str] = None,
    manual_target_bond: Optional[str] = None,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
    drop_failed: bool = True,
):
    require_torch()
    # Reuse the base dataset logic directly rather than the convenience helpers so xTB stays experimental-only.
    train_ds = CYPMetabolismDataset(
        split="train",
        augment=True,
        drugs=train_drugs,
        structure_library=None,
        use_manual_engine_features=True,
        manual_target_bond=manual_target_bond,
        manual_feature_cache_dir=manual_feature_cache_dir,
        allow_partial_sanitize=allow_partial_sanitize,
        allow_aggressive_repair=allow_aggressive_repair,
        drop_failed=drop_failed,
    )
    val_ds = CYPMetabolismDataset(
        split="val",
        augment=False,
        drugs=val_drugs,
        structure_library=None,
        use_manual_engine_features=True,
        manual_target_bond=manual_target_bond,
        manual_feature_cache_dir=manual_feature_cache_dir,
        allow_partial_sanitize=allow_partial_sanitize,
        allow_aggressive_repair=allow_aggressive_repair,
        drop_failed=drop_failed,
    )
    test_ds = CYPMetabolismDataset(
        split="test",
        augment=False,
        drugs=test_drugs,
        structure_library=None,
        use_manual_engine_features=True,
        manual_target_bond=manual_target_bond,
        manual_feature_cache_dir=manual_feature_cache_dir,
        allow_partial_sanitize=allow_partial_sanitize,
        allow_aggressive_repair=allow_aggressive_repair,
        drop_failed=drop_failed,
    )
    # Attach structure library only if provided.
    if structure_sdf:
        from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary

        structure_library = StructureLibrary.from_sdf(structure_sdf)
        train_ds.structure_library = structure_library
        val_ds.structure_library = structure_library
        test_ds.structure_library = structure_library

    train_exp = MicroPatternExperimentDataset(train_ds, xtb_cache_dir=xtb_cache_dir, compute_xtb_if_missing=compute_xtb_if_missing)
    val_exp = MicroPatternExperimentDataset(val_ds, xtb_cache_dir=xtb_cache_dir, compute_xtb_if_missing=compute_xtb_if_missing)
    test_exp = MicroPatternExperimentDataset(test_ds, xtb_cache_dir=xtb_cache_dir, compute_xtb_if_missing=compute_xtb_if_missing)
    train_loader = DataLoader(train_exp, batch_size=batch_size, shuffle=True, collate_fn=base_collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_exp, batch_size=batch_size, shuffle=False, collate_fn=base_collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_exp, batch_size=batch_size, shuffle=False, collate_fn=base_collate_fn, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def print_split_summary(train_drugs: List[dict], val_drugs: List[dict], test_drugs: List[dict]) -> None:
    for split_name, split_drugs in (("train", train_drugs), ("val", val_drugs), ("test", test_drugs)):
        source_counts = Counter(str(d.get("source", "unknown")) for d in split_drugs)
        site_count = sum(1 for d in split_drugs if _has_site_labels(d))
        print(
            f"{split_name}: total={len(split_drugs)} | site_supervised={site_count} | sources={dict(source_counts)}",
            flush=True,
        )
