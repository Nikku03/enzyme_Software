from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.features.manual_engine_features import attach_manual_engine_features_to_graph
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary, resolve_default_structure_sdf
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


CONFIDENCE_WEIGHTS = {
    "high": 1.0,
    "validated": 1.0,
    "validated_gold": 1.0,
    "validated_literature": 1.0,
    "curated": 1.0,
    "medium": 0.7,
    "low": 0.4,
    "bde_predicted": 0.7,
    "unknown": 0.5,
}


class CYPMetabolismDataset(Dataset):
    CYP_CLASSES = list(MAJOR_CYP_CLASSES)

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        confidence_filter: Optional[List[str]] = None,
        augment: bool = False,
        seed: int = 42,
        cyp_classes: Optional[List[str]] = None,
        drugs: Optional[List[Dict[str, object]]] = None,
        structure_library: Optional[StructureLibrary] = None,
        use_manual_engine_features: bool = False,
        manual_target_bond: Optional[str] = None,
        manual_feature_cache_dir: Optional[str] = None,
        allow_partial_sanitize: bool = True,
        allow_aggressive_repair: bool = False,
        drop_failed: bool = True,
    ):
        require_torch()
        self.augment = augment
        self.cyp_classes = list(cyp_classes or self.CYP_CLASSES)
        self.cyp_to_idx = {name: idx for idx, name in enumerate(self.cyp_classes)}
        self.structure_library = structure_library
        self.use_manual_engine_features = bool(use_manual_engine_features)
        self.manual_target_bond = manual_target_bond
        self.manual_feature_cache_dir = manual_feature_cache_dir
        self.allow_partial_sanitize = bool(allow_partial_sanitize)
        self.allow_aggressive_repair = bool(allow_aggressive_repair)
        self.drop_failed = bool(drop_failed)
        if drugs is None:
            if data_path is None:
                raise ValueError("Either data_path or drugs must be provided")
            payload = json.loads(Path(data_path).read_text())
            source_drugs = list(payload.get("drugs", payload))
            if confidence_filter:
                allowed = set(confidence_filter)
                source_drugs = [d for d in source_drugs if d.get("confidence") in allowed or d.get("source") in allowed]
            supported = set(self.cyp_classes)
            source_drugs = [d for d in source_drugs if str(d.get("cyp") or d.get("primary_cyp") or "") in supported]
            random.seed(seed)
            random.shuffle(source_drugs)
            n_train = int(len(source_drugs) * train_ratio)
            n_val = int(len(source_drugs) * val_ratio)
            if split == "train":
                self.drugs = source_drugs[:n_train]
            elif split == "val":
                self.drugs = source_drugs[n_train : n_train + n_val]
            else:
                self.drugs = source_drugs[n_train + n_val :]
        else:
            supported = set(self.cyp_classes)
            self.drugs = [d for d in drugs if str(d.get("cyp") or d.get("primary_cyp") or "") in supported]
        print(f"Loaded {len(self.drugs)} drugs for {split} split")
        self._cache: Optional[List] = None
        self._cache = [self[i] for i in range(len(self.drugs))]

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx) -> Dict[str, object]:
        if self._cache is not None:
            return self._cache[idx]
        drug = self.drugs[idx]
        smiles = str(drug.get("smiles", ""))
        cyp = str(drug.get("cyp") or drug.get("primary_cyp") or "")
        site_atoms = []
        has_site_supervision = False
        if drug.get("som"):
            for som in drug["som"]:
                atom_idx = som.get("atom_idx", som) if isinstance(som, dict) else som
                if isinstance(atom_idx, int):
                    site_atoms.append(atom_idx)
            has_site_supervision = len(site_atoms) > 0
        elif drug.get("site_atoms"):
            site_atoms = [int(v) for v in drug.get("site_atoms", [])]
            has_site_supervision = len(site_atoms) > 0
        elif drug.get("site_atom_indices"):
            site_atoms = [int(v) for v in drug.get("site_atom_indices", [])]
            has_site_supervision = len(site_atoms) > 0
        try:
            with mol_provenance_context(
                caller_module="dataset loader",
                module_triggered="dataset loader",
                source_category="dataset loader",
                original_smiles=smiles,
                drug_name=str(drug.get("name", "")),
                drug_id=str(drug.get("drug_id", drug.get("id", idx))),
            ):
                structure_mol = self.structure_library.get(smiles) if self.structure_library is not None else None
                graph = smiles_to_graph(
                    smiles,
                    site_atoms=site_atoms if has_site_supervision else None,
                    structure_mol=structure_mol,
                    allow_partial_sanitize=self.allow_partial_sanitize,
                    allow_aggressive_repair=self.allow_aggressive_repair,
                )
                if not self.allow_aggressive_repair and bool(getattr(graph, "aggressive_repair", False)):
                    raise ValueError(f"Aggressive repaired molecule not allowed: {smiles}")
                if self.use_manual_engine_features:
                    graph = attach_manual_engine_features_to_graph(
                        graph,
                        target_bond=self.manual_target_bond,
                        cyp_order=self.cyp_classes,
                        cache_dir=self.manual_feature_cache_dir,
                        allow_partial_sanitize=self.allow_partial_sanitize,
                        allow_aggressive_repair=self.allow_aggressive_repair,
                    )
                graph.cyp_label = int(self.cyp_to_idx[cyp])
                graph.has_site_supervision = bool(has_site_supervision)
        except Exception:
            if self.drop_failed:
                return self._get_dummy()
            raise
        confidence_value = str(drug.get("confidence") or drug.get("source") or "unknown")
        return {
            "graph": graph,
            "smiles": smiles,
            "name": str(drug.get("name", "")),
            "confidence": confidence_value,
            "confidence_weight": float(CONFIDENCE_WEIGHTS.get(confidence_value, 0.5)),
        }

    def _get_dummy(self):
        return {"graph": None, "name": "INVALID", "smiles": "", "confidence": "none", "confidence_weight": 0.0}



def collate_fn(batch: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    require_torch()
    batch = [b for b in batch if b.get("graph") is not None and b.get("name") != "INVALID"]
    if not batch:
        return None
    graphs = [b["graph"] for b in batch]
    merged = collate_molecule_graphs(graphs)
    graph_weights = torch.tensor([float(b["confidence_weight"]) for b in batch], dtype=torch.float32)
    node_weights = []
    for b, g in zip(batch, graphs):
        node_weights.extend([float(b["confidence_weight"])] * g.num_atoms)
    merged["graph_confidence_weights"] = graph_weights
    merged["node_confidence_weights"] = torch.tensor(node_weights, dtype=torch.float32).unsqueeze(-1)
    merged["graph_names"] = [str(b["name"]) for b in batch]
    merged["graph_confidences"] = [str(b["confidence"]) for b in batch]
    merged["num_graphs"] = len(batch)
    return merged



def create_dataloaders(
    data_path: str,
    batch_size: int = 8,
    confidence_filter: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    cyp_classes: Optional[List[str]] = None,
    structure_sdf: Optional[str] = None,
    use_manual_engine_features: bool = False,
    manual_target_bond: Optional[str] = None,
    manual_feature_cache_dir: Optional[str] = None,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
    drop_failed: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    require_torch()
    structure_library = None
    resolved_sdf = structure_sdf or str(resolve_default_structure_sdf(Path(data_path).resolve().parents[1]) or "")
    if resolved_sdf:
        structure_library = StructureLibrary.from_sdf(resolved_sdf)
    common = {
        "data_path": data_path,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "confidence_filter": confidence_filter,
        "seed": seed,
        "cyp_classes": cyp_classes,
        "structure_library": structure_library,
        "use_manual_engine_features": use_manual_engine_features,
        "manual_target_bond": manual_target_bond,
        "manual_feature_cache_dir": manual_feature_cache_dir,
        "allow_partial_sanitize": allow_partial_sanitize,
        "allow_aggressive_repair": allow_aggressive_repair,
        "drop_failed": drop_failed,
    }
    train_ds = CYPMetabolismDataset(split="train", augment=True, **common)
    val_ds = CYPMetabolismDataset(split="val", augment=False, **common)
    test_ds = CYPMetabolismDataset(split="test", augment=False, **common)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def create_dataloaders_from_drugs(
    train_drugs: List[Dict[str, object]],
    val_drugs: List[Dict[str, object]],
    test_drugs: List[Dict[str, object]],
    batch_size: int = 8,
    cyp_classes: Optional[List[str]] = None,
    structure_sdf: Optional[str] = None,
    use_manual_engine_features: bool = False,
    manual_target_bond: Optional[str] = None,
    manual_feature_cache_dir: Optional[str] = None,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
    drop_failed: bool = True,
):
    require_torch()
    structure_library = StructureLibrary.from_sdf(structure_sdf) if structure_sdf else None
    train_ds = CYPMetabolismDataset(split="train", augment=True, cyp_classes=cyp_classes, drugs=train_drugs, structure_library=structure_library, use_manual_engine_features=use_manual_engine_features, manual_target_bond=manual_target_bond, manual_feature_cache_dir=manual_feature_cache_dir, allow_partial_sanitize=allow_partial_sanitize, allow_aggressive_repair=allow_aggressive_repair, drop_failed=drop_failed)
    val_ds = CYPMetabolismDataset(split="val", augment=False, cyp_classes=cyp_classes, drugs=val_drugs, structure_library=structure_library, use_manual_engine_features=use_manual_engine_features, manual_target_bond=manual_target_bond, manual_feature_cache_dir=manual_feature_cache_dir, allow_partial_sanitize=allow_partial_sanitize, allow_aggressive_repair=allow_aggressive_repair, drop_failed=drop_failed)
    test_ds = CYPMetabolismDataset(split="test", augment=False, cyp_classes=cyp_classes, drugs=test_drugs, structure_library=structure_library, use_manual_engine_features=use_manual_engine_features, manual_target_bond=manual_target_bond, manual_feature_cache_dir=manual_feature_cache_dir, allow_partial_sanitize=allow_partial_sanitize, allow_aggressive_repair=allow_aggressive_repair, drop_failed=drop_failed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader
