from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

from enzyme_software.liquid_nn_v2._compat import torch


def load_drugs(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return list(payload.get("drugs", payload))


def has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def extract_site_indices(drug: dict) -> list[int]:
    raw = []
    if drug.get("metabolism_sites"):
        raw = drug.get("metabolism_sites") or []
    elif drug.get("som"):
        raw = drug.get("som") or []
    elif drug.get("site_atoms"):
        raw = drug.get("site_atoms") or []
    elif drug.get("site_atom_indices"):
        raw = drug.get("site_atom_indices") or []
    indices: list[int] = []
    for item in raw:
        if isinstance(item, int):
            indices.append(int(item))
            continue
        if isinstance(item, dict):
            idx = item.get("atom_index", item.get("atom_idx", item.get("index", -1)))
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int >= 0:
                indices.append(idx_int)
    return indices


def filter_site_labeled_drugs(drugs: Sequence[dict]) -> list[dict]:
    return [drug for drug in drugs if has_site_labels(drug)]


def split_items(items: Sequence[dict], *, seed: int, train_ratio: float, val_ratio: float):
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def initialized_state_dict(model) -> dict:
    state = {}
    uninitialized_type = getattr(torch.nn.parameter, "UninitializedParameter", ())
    for key, value in model.state_dict().items():
        if isinstance(value, uninitialized_type):
            continue
        state[key] = value.detach().cpu() if hasattr(value, "detach") else value
    return state


def source_weight(source: str, *, ground_truth_weight: float, pseudo_weight: float) -> float:
    return float(ground_truth_weight if source == "ground_truth" else pseudo_weight)


def summarize_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
