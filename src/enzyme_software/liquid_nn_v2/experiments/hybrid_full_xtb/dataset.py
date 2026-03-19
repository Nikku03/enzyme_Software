from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from enzyme_software.liquid_nn_v2._compat import require_torch
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset, collate_fn
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM, load_or_compute_full_xtb_features


def split_drugs(drugs: List[Dict[str, object]], seed: int, train_ratio: float, val_ratio: float):
    shuffled = list(drugs)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


class FullXTBHybridDataset(CYPMetabolismDataset):
    def __init__(
        self,
        *args,
        full_xtb_cache_dir: Optional[str] = None,
        compute_full_xtb_if_missing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_xtb_cache_dir = full_xtb_cache_dir
        self.compute_full_xtb_if_missing = bool(compute_full_xtb_if_missing)

    def precompute(self):
        # Cache base features only (32-dim), NOT XTB-augmented (40-dim).
        # __getitem__ adds XTB on top of cached base features → 32+8=40 total.
        # Without this override, precompute caches 40-dim items and __getitem__
        # would add XTB again → 48 dims, causing a dimension mismatch.
        if self._cache is None:
            self._cache = [CYPMetabolismDataset.__getitem__(self, i) for i in range(len(self.drugs))]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        graph = item.get("graph")
        if graph is None or self.full_xtb_cache_dir is None:
            return item
        # Shallow-copy dict and graph so cached objects are never mutated.
        # Without this, the first call appends 8 XTB dims (32→40), and the
        # second call (epoch 2) would read 40-dim and append again → 48.
        item = dict(item)
        graph = copy.copy(graph)
        item["graph"] = graph

        payload = load_or_compute_full_xtb_features(
            graph.canonical_smiles or graph.smiles,
            cache_dir=self.full_xtb_cache_dir,
            compute_if_missing=self.compute_full_xtb_if_missing,
        )
        num_atoms = int(graph.num_atoms)
        raw_features = np.asarray(payload.get("atom_features") or [], dtype=np.float32)
        raw_valid = np.asarray(payload.get("atom_valid_mask") or [], dtype=np.float32)
        if raw_features.size == 0:
            raw_features = np.zeros((num_atoms, FULL_XTB_FEATURE_DIM), dtype=np.float32)
        if raw_valid.size == 0:
            raw_valid = np.zeros((num_atoms, 1), dtype=np.float32)
        raw_features = raw_features[:num_atoms]
        raw_valid = raw_valid[:num_atoms]
        if raw_features.shape[0] < num_atoms:
            pad = np.zeros((num_atoms - raw_features.shape[0], FULL_XTB_FEATURE_DIM), dtype=np.float32)
            raw_features = np.concatenate([raw_features, pad], axis=0)
        if raw_valid.shape[0] < num_atoms:
            pad = np.zeros((num_atoms - raw_valid.shape[0], 1), dtype=np.float32)
            raw_valid = np.concatenate([raw_valid, pad], axis=0)

        base_manual = getattr(graph, "manual_engine_atom_features", None)
        if base_manual is None:
            base_manual = np.zeros((num_atoms, 32), dtype=np.float32)
        base_manual = np.asarray(base_manual, dtype=np.float32)
        if base_manual.shape[0] != num_atoms:
            fixed = np.zeros((num_atoms, base_manual.shape[1] if base_manual.ndim == 2 else 32), dtype=np.float32)
            rows = min(num_atoms, base_manual.shape[0]) if base_manual.ndim == 2 else 0
            if rows:
                fixed[:rows] = base_manual[:rows]
            base_manual = fixed

        graph.manual_engine_atom_features = np.concatenate([base_manual, raw_features], axis=1).astype(np.float32)
        graph.xtb_atom_features = raw_features.astype(np.float32)
        graph.xtb_atom_valid_mask = raw_valid.astype(np.float32)
        graph.xtb_mol_valid = np.asarray([[1.0 if payload.get("xtb_valid") else 0.0]], dtype=np.float32)
        graph.xtb_feature_status = str(payload.get("status") or "missing")
        return item


def create_full_xtb_dataloaders_from_drugs(
    train_drugs: List[Dict[str, object]],
    val_drugs: List[Dict[str, object]],
    test_drugs: List[Dict[str, object]],
    *,
    batch_size: int,
    cyp_classes: Optional[List[str]] = None,
    structure_sdf: Optional[str] = None,
    use_manual_engine_features: bool = True,
    manual_target_bond: Optional[str] = None,
    manual_feature_cache_dir: Optional[str] = None,
    full_xtb_cache_dir: Optional[str] = None,
    compute_full_xtb_if_missing: bool = False,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
    drop_failed: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    require_torch()
    structure_library = StructureLibrary.from_sdf(structure_sdf) if structure_sdf else None
    common = {
        "cyp_classes": cyp_classes,
        "structure_library": structure_library,
        "use_manual_engine_features": use_manual_engine_features,
        "manual_target_bond": manual_target_bond,
        "manual_feature_cache_dir": manual_feature_cache_dir,
        "allow_partial_sanitize": allow_partial_sanitize,
        "allow_aggressive_repair": allow_aggressive_repair,
        "drop_failed": drop_failed,
        "full_xtb_cache_dir": full_xtb_cache_dir,
        "compute_full_xtb_if_missing": compute_full_xtb_if_missing,
    }
    train_ds = FullXTBHybridDataset(split="train", augment=True, drugs=train_drugs, **common)
    val_ds = FullXTBHybridDataset(split="val", augment=False, drugs=val_drugs, **common)
    test_ds = FullXTBHybridDataset(split="test", augment=False, drugs=test_drugs, **common)
    for ds in (train_ds, val_ds, test_ds):
        ds.precompute()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader
