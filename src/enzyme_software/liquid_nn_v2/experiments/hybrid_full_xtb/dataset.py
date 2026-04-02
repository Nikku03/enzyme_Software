from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from enzyme_software.liquid_nn_v2._compat import require_torch
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset, collate_fn
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.topology_features import TOPOLOGY_FEATURE_DIM, compute_atom_topology_features
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM, load_or_compute_full_xtb_features, xtb_status_vector


def _canonical_smiles(drug: Dict[str, object]) -> str:
    return " ".join(str(drug.get("smiles", "") or "").split())


def _safe_num_atoms(drug: Dict[str, object]) -> int:
    value = drug.get("num_atoms")
    if isinstance(value, int) and value > 0:
        return int(value)
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(_canonical_smiles(drug))
        if mol is not None:
            return int(mol.GetNumAtoms())
    except Exception:
        pass
    return 0


def _size_bucket(num_atoms: int) -> str:
    if num_atoms <= 0:
        return "unknown"
    if num_atoms <= 15:
        return "<=15"
    if num_atoms <= 25:
        return "16-25"
    if num_atoms <= 40:
        return "26-40"
    if num_atoms <= 60:
        return "41-60"
    return "61+"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scaffold_key(drug: Dict[str, object]) -> str:
    smiles = _canonical_smiles(drug)
    if not smiles:
        return f"missing::{drug.get('id', '')}"
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"invalid::{smiles}"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or f"acyclic::{smiles}"
    except Exception:
        return f"fallback::{smiles}"


def _target_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = max(0, total - n_train - n_val)
    return {"train": n_train, "val": n_val, "test": n_test}


def _assign_groups_greedily(
    groups: List[List[Dict[str, object]]],
    *,
    seed: int,
    targets: dict[str, int],
) -> dict[str, List[Dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    splits: dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for group in shuffled:
        split_name = min(
            ("train", "val", "test"),
            key=lambda name: (len(splits[name]) - targets[name], len(splits[name])),
        )
        splits[split_name].extend(group)
    return splits


def _group_signature(
    group: List[Dict[str, object]],
    *,
    stratify_size: bool,
) -> dict[str, object]:
    source_counts: dict[str, int] = defaultdict(int)
    size_counts: dict[str, int] = defaultdict(int)
    atom_sum = 0
    for drug in group:
        source_counts[str(drug.get("source", "DrugBank"))] += 1
        atom_sum += int(_safe_num_atoms(drug))
        if stratify_size:
            size_counts[_size_bucket(_safe_num_atoms(drug))] += 1
    return {
        "size": int(len(group)),
        "atom_sum": float(atom_sum),
        "source_counts": dict(source_counts),
        "size_counts": dict(size_counts),
    }


def _global_targets(
    drugs: List[Dict[str, object]],
    *,
    train_ratio: float,
    val_ratio: float,
    stratify_size: bool,
) -> dict[str, dict[str, object]]:
    split_names = ("train", "val", "test")
    total = float(len(drugs))
    base_ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": max(0.0, 1.0 - float(train_ratio) - float(val_ratio)),
    }
    sources = sorted({str(drug.get("source", "DrugBank")) for drug in drugs})
    size_buckets = sorted({_size_bucket(_safe_num_atoms(drug)) for drug in drugs}) if stratify_size else []
    source_totals = Counter(str(drug.get("source", "DrugBank")) for drug in drugs)
    size_totals = Counter(_size_bucket(_safe_num_atoms(drug)) for drug in drugs) if stratify_size else Counter()
    total_atom_sum = float(sum(_safe_num_atoms(drug) for drug in drugs))
    dataset_mean_atoms = (total_atom_sum / total) if total > 0 else 0.0
    heldout_target_mean = _clamp(dataset_mean_atoms, 25.0, 28.0)
    return {
        split: {
            "total": total * base_ratios[split],
            "atom_sum": total_atom_sum * base_ratios[split],
            "target_mean_atoms": heldout_target_mean if split in {"val", "test"} else dataset_mean_atoms,
            "heldout_mean_band": (25.0, 28.0) if split in {"val", "test"} else None,
            "sources": {source: float(source_totals[source]) * base_ratios[split] for source in sources},
            "sizes": {bucket: float(size_totals[bucket]) * base_ratios[split] for bucket in size_buckets},
        }
        for split in split_names
    }


def _assignment_cost(
    *,
    split_name: str,
    current: dict[str, dict[str, object]],
    group_sig: dict[str, object],
    targets: dict[str, dict[str, object]],
    stratify_size: bool,
) -> float:
    projected_total = float(current[split_name]["total"]) + float(group_sig["size"])
    total_target = float(targets[split_name]["total"])
    total_error = abs(projected_total - total_target) / max(1.0, total_target)
    total_overflow = max(0.0, projected_total - total_target) / max(1.0, total_target)

    atom_sum_target = float(targets[split_name].get("atom_sum", 0.0))
    projected_atom_sum = float(current[split_name].get("atom_sum", 0.0)) + float(group_sig.get("atom_sum", 0.0))
    atom_sum_error = abs(projected_atom_sum - atom_sum_target) / max(1.0, atom_sum_target)

    mean_cost = 0.0
    if projected_total > 0:
        projected_mean = projected_atom_sum / projected_total
        target_mean = float(targets[split_name].get("target_mean_atoms", projected_mean))
        mean_cost += abs(projected_mean - target_mean)
        heldout_band = targets[split_name].get("heldout_mean_band")
        if heldout_band is not None:
            lower, upper = heldout_band
            mean_cost += 4.0 * max(0.0, float(lower) - projected_mean)
            mean_cost += 4.0 * max(0.0, projected_mean - float(upper))

    source_cost = 0.0
    for source, add_count in dict(group_sig["source_counts"]).items():
        projected = float(current[split_name]["sources"].get(source, 0.0)) + float(add_count)
        target = float(targets[split_name]["sources"].get(source, 0.0))
        source_cost += abs(projected - target) / max(1.0, target)

    size_cost = 0.0
    if stratify_size:
        for bucket, add_count in dict(group_sig["size_counts"]).items():
            projected = float(current[split_name]["sizes"].get(bucket, 0.0)) + float(add_count)
            target = float(targets[split_name]["sizes"].get(bucket, 0.0))
            size_cost += abs(projected - target) / max(1.0, target)

    return (
        (100.0 * total_overflow)
        + (12.0 * total_error)
        + (3.0 * atom_sum_error)
        + (2.5 * mean_cost)
        + (1.5 * source_cost)
        + (1.25 * size_cost)
    )


def _global_assignment_score(
    *,
    current: dict[str, dict[str, object]],
    targets: dict[str, dict[str, object]],
    stratify_size: bool,
) -> float:
    score = 0.0
    for split_name in ("train", "val", "test"):
        total = float(current[split_name]["total"])
        total_target = float(targets[split_name]["total"])
        score += 12.0 * abs(total - total_target) / max(1.0, total_target)
        score += 3.0 * abs(float(current[split_name].get("atom_sum", 0.0)) - float(targets[split_name].get("atom_sum", 0.0))) / max(
            1.0, float(targets[split_name].get("atom_sum", 0.0))
        )
        if total > 0:
            mean_atoms = float(current[split_name].get("atom_sum", 0.0)) / total
            target_mean = float(targets[split_name].get("target_mean_atoms", mean_atoms))
            score += 2.5 * abs(mean_atoms - target_mean)
            heldout_band = targets[split_name].get("heldout_mean_band")
            if heldout_band is not None:
                lower, upper = heldout_band
                score += 10.0 * max(0.0, float(lower) - mean_atoms)
                score += 10.0 * max(0.0, mean_atoms - float(upper))
        for source, target in dict(targets[split_name]["sources"]).items():
            score += 1.5 * abs(float(current[split_name]["sources"].get(source, 0.0)) - float(target)) / max(1.0, float(target))
        if stratify_size:
            for bucket, target in dict(targets[split_name]["sizes"]).items():
                score += 1.25 * abs(float(current[split_name]["sizes"].get(bucket, 0.0)) - float(target)) / max(1.0, float(target))
    return score


def _move_group_state(
    *,
    current: dict[str, dict[str, object]],
    source_split: str,
    target_split: str,
    sig: dict[str, object],
) -> None:
    current[source_split]["total"] -= float(sig["size"])
    current[target_split]["total"] += float(sig["size"])
    current[source_split]["atom_sum"] -= float(sig.get("atom_sum", 0.0))
    current[target_split]["atom_sum"] += float(sig.get("atom_sum", 0.0))
    for source, count in dict(sig["source_counts"]).items():
        current[source_split]["sources"][source] -= float(count)
        current[target_split]["sources"][source] += float(count)
    for bucket, count in dict(sig["size_counts"]).items():
        current[source_split]["sizes"][bucket] -= float(count)
        current[target_split]["sizes"][bucket] += float(count)


def _rebalance_group_assignments(
    *,
    groups: List[List[Dict[str, object]]],
    group_sigs: List[dict[str, object]],
    assignments: List[str],
    current: dict[str, dict[str, object]],
    targets: dict[str, dict[str, object]],
    stratify_size: bool,
    seed: int,
) -> None:
    rng = random.Random(seed + 7919)
    split_names = ("train", "val", "test")
    best_score = _global_assignment_score(current=current, targets=targets, stratify_size=stratify_size)
    improved = True
    max_passes = 8
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        order = list(range(len(groups)))
        rng.shuffle(order)
        for idx in order:
            current_split = assignments[idx]
            sig = group_sigs[idx]
            local_best_split = current_split
            local_best_score = best_score
            for candidate_split in split_names:
                if candidate_split == current_split:
                    continue
                _move_group_state(current=current, source_split=current_split, target_split=candidate_split, sig=sig)
                candidate_score = _global_assignment_score(
                    current=current,
                    targets=targets,
                    stratify_size=stratify_size,
                )
                _move_group_state(current=current, source_split=candidate_split, target_split=current_split, sig=sig)
                if candidate_score + 1e-9 < local_best_score:
                    local_best_score = candidate_score
                    local_best_split = candidate_split
            if local_best_split != current_split:
                _move_group_state(current=current, source_split=current_split, target_split=local_best_split, sig=sig)
                assignments[idx] = local_best_split
                best_score = local_best_score
                improved = True


def _split_by_global_scaffold_groups(
    drugs: List[Dict[str, object]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    stratify_size: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    scaffold_groups: dict[str, List[Dict[str, object]]] = defaultdict(list)
    for drug in drugs:
        scaffold_groups[_scaffold_key(drug)].append(drug)

    rng = random.Random(seed)
    groups = list(scaffold_groups.values())
    rng.shuffle(groups)
    groups.sort(key=lambda group: len(group), reverse=True)

    targets = _global_targets(drugs, train_ratio=train_ratio, val_ratio=val_ratio, stratify_size=stratify_size)
    current = {
        "train": {"total": 0.0, "atom_sum": 0.0, "sources": defaultdict(float), "sizes": defaultdict(float)},
        "val": {"total": 0.0, "atom_sum": 0.0, "sources": defaultdict(float), "sizes": defaultdict(float)},
        "test": {"total": 0.0, "atom_sum": 0.0, "sources": defaultdict(float), "sizes": defaultdict(float)},
    }
    splits: dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    group_sigs: List[dict[str, object]] = []
    assignments: List[str] = []

    for group in groups:
        sig = _group_signature(group, stratify_size=stratify_size)
        group_sigs.append(sig)
        split_name = min(
            ("train", "val", "test"),
            key=lambda name: (
                _assignment_cost(
                    split_name=name,
                    current=current,
                    group_sig=sig,
                    targets=targets,
                    stratify_size=stratify_size,
                ),
                current[name]["total"],
            ),
        )
        splits[split_name].extend(group)
        assignments.append(split_name)
        current[split_name]["total"] += float(sig["size"])
        current[split_name]["atom_sum"] += float(sig.get("atom_sum", 0.0))
        for source, count in dict(sig["source_counts"]).items():
            current[split_name]["sources"][source] += float(count)
        for bucket, count in dict(sig["size_counts"]).items():
            current[split_name]["sizes"][bucket] += float(count)

    _rebalance_group_assignments(
        groups=groups,
        group_sigs=group_sigs,
        assignments=assignments,
        current=current,
        targets=targets,
        stratify_size=stratify_size,
        seed=seed,
    )

    splits = {"train": [], "val": [], "test": []}
    for group, split_name in zip(groups, assignments):
        splits[split_name].extend(group)

    return splits["train"], splits["val"], splits["test"]


def _split_by_scaffold_groups(
    drugs: List[Dict[str, object]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    stratify_size: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    strata: dict[tuple[str, str], dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for drug in drugs:
        source = str(drug.get("source", "DrugBank"))
        bucket = _size_bucket(_safe_num_atoms(drug)) if stratify_size else "all"
        strata[(source, bucket)][_scaffold_key(drug)].append(drug)

    targets = _target_counts(len(drugs), train_ratio, val_ratio)
    splits: dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for idx, ((_source, _bucket), scaffold_groups) in enumerate(sorted(strata.items())):
        stratum_groups = list(scaffold_groups.values())
        stratum_targets = _target_counts(sum(len(g) for g in stratum_groups), train_ratio, val_ratio)
        assigned = _assign_groups_greedily(stratum_groups, seed=seed + idx, targets=stratum_targets)
        for split_name in splits:
            splits[split_name].extend(assigned[split_name])

    # Coarse rebalance to keep ratios roughly aligned after grouped assignment.
    for donor, receiver in (("train", "val"), ("train", "test"), ("val", "test")):
        while len(splits[donor]) > targets[donor] + 1 and len(splits[receiver]) < targets[receiver]:
            splits[receiver].append(splits[donor].pop())
    return splits["train"], splits["val"], splits["test"]


def split_drugs(
    drugs: List[Dict[str, object]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    *,
    mode: str = "random",
):
    shuffled = list(drugs)
    if mode == "random":
        random.Random(seed).shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)
        return (
            shuffled[:n_train],
            shuffled[n_train : n_train + n_val],
            shuffled[n_train + n_val :],
        )
    if mode == "scaffold_source":
        return _split_by_global_scaffold_groups(
            shuffled,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stratify_size=False,
        )
    if mode == "scaffold_source_size":
        return _split_by_global_scaffold_groups(
            shuffled,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stratify_size=True,
        )
    raise ValueError(f"Unsupported split mode: {mode}")


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
            cached = [CYPMetabolismDataset.__getitem__(self, i) for i in range(len(self.drugs))]
            self._cache = cached
            self._valid_count = sum(1 for item in cached if item.get("graph") is not None and item.get("name") != "INVALID")
            reasons: Dict[str, int] = {}
            for item in cached:
                if item.get("graph") is not None and item.get("name") != "INVALID":
                    continue
                reason = str(item.get("error_reason") or "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
            self._invalid_reasons = reasons
            if reasons:
                top = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
                )
                print(
                    f"[{self.__class__.__name__}:{self.split}] valid={self._valid_count}/{len(cached)} "
                    f"invalid={len(cached) - self._valid_count} | {top}",
                    flush=True,
                )
            else:
                print(
                    f"[{self.__class__.__name__}:{self.split}] valid={self._valid_count}/{len(cached)} invalid=0",
                    flush=True,
                )

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

        # Guard: xtb failures can produce NaN/inf — zero them out so gradients stay finite
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        base_manual  = np.nan_to_num(base_manual,  nan=0.0, posinf=0.0, neginf=0.0)

        graph.manual_engine_atom_features = np.concatenate([base_manual, raw_features], axis=1).astype(np.float32)
        graph.xtb_atom_features = raw_features.astype(np.float32)
        graph.xtb_atom_valid_mask = raw_valid.astype(np.float32)
        graph.xtb_mol_valid = np.asarray([[1.0 if payload.get("xtb_valid") else 0.0]], dtype=np.float32)
        graph.xtb_feature_status = str(payload.get("status") or "missing")
        graph.xtb_status_flags = xtb_status_vector(graph.xtb_feature_status)

        # Global topology features (scaffold/sidechain role, centrality, carbonyl
        # distance, molecule size context).  These give the site arbiter context
        # that the local GNN cannot propagate across large molecules.
        smiles_for_topo = graph.canonical_smiles or graph.smiles
        topo = compute_atom_topology_features(smiles_for_topo)
        if topo is not None and topo.shape[0] == num_atoms:
            graph.topology_atom_features = topo
            graph.topology_atom_valid_mask = np.ones((num_atoms, 1), dtype=np.float32)
            graph.topology_mol_valid = np.asarray([[1.0]], dtype=np.float32)
        else:
            graph.topology_atom_features = np.zeros((num_atoms, TOPOLOGY_FEATURE_DIM), dtype=np.float32)
            graph.topology_atom_valid_mask = np.zeros((num_atoms, 1), dtype=np.float32)
            graph.topology_mol_valid = np.asarray([[0.0]], dtype=np.float32)

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
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if int(getattr(ds, "_valid_count", 0)) <= 0:
            reasons = getattr(ds, "_invalid_reasons", {}) or {}
            top = ", ".join(
                f"{key}={value}"
                for key, value in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
            ) or "unknown"
            raise RuntimeError(
                f"FullXTBHybridDataset produced zero valid graphs for {name} split. Top failure reasons: {top}"
            )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader
