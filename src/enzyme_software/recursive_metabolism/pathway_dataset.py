from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.features.manual_engine_features import attach_manual_engine_features_to_graph
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.xtb_features import attach_xtb_features_to_graph
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs


@dataclass
class RecursiveSample:
    pathway_index: int
    drug_name: str
    original_drug_smiles: str
    smiles: str
    metabolite_smiles: str
    target_site: int
    step_number: int
    metabolism_type: str
    supervision_source: str
    source_weight: float


def _load_pathways(pathways: str | Path | List[Dict[str, object]]) -> list[dict]:
    if isinstance(pathways, (str, Path)):
        payload = json.loads(Path(pathways).read_text())
        return list(payload)
    return list(pathways)


class RecursiveMetabolismDataset:
    def __init__(
        self,
        pathways: str | Path | List[Dict[str, object]],
        *,
        structure_sdf: Optional[str] = None,
        include_manual_engine_features: bool = True,
        include_xtb_features: bool = True,
        xtb_cache_dir: Optional[str] = None,
        compute_xtb_if_missing: bool = False,
        manual_feature_cache_dir: Optional[str] = None,
        allow_partial_sanitize: bool = True,
        allow_aggressive_repair: bool = False,
        drop_failed: bool = True,
        ground_truth_only: bool = False,
        max_step: Optional[int] = None,
    ):
        require_torch()
        self.pathways = _load_pathways(pathways)
        self.structure_library = StructureLibrary.from_sdf(structure_sdf) if structure_sdf else None
        self.include_manual_engine_features = bool(include_manual_engine_features)
        self.include_xtb_features = bool(include_xtb_features)
        self.xtb_cache_dir = xtb_cache_dir
        self.compute_xtb_if_missing = bool(compute_xtb_if_missing)
        self.manual_feature_cache_dir = manual_feature_cache_dir
        self.allow_partial_sanitize = bool(allow_partial_sanitize)
        self.allow_aggressive_repair = bool(allow_aggressive_repair)
        self.drop_failed = bool(drop_failed)
        self.ground_truth_only = bool(ground_truth_only)
        self.max_step = int(max_step) if max_step is not None else None
        self.samples = self._expand_pathways()
        print(
            f"Expanded {len(self.pathways)} pathways into {len(self.samples)} recursive samples",
            flush=True,
        )

    def _expand_pathways(self) -> List[RecursiveSample]:
        samples: list[RecursiveSample] = []
        for pathway_idx, pathway in enumerate(self.pathways):
            steps = pathway.get("steps", []) or []
            for step in steps:
                step_number = int(step.get("step_number", step.get("step", 0)))
                supervision_source = str(step.get("supervision_source", step.get("source", "heuristic")))
                if self.ground_truth_only and supervision_source != "ground_truth":
                    continue
                if self.max_step is not None and step_number > self.max_step:
                    continue
                parent_smiles = str(step.get("parent_smiles") or step.get("parent") or "")
                metabolite_smiles = str(step.get("metabolite_smiles") or step.get("metabolite") or "")
                target_site = int(step.get("site_atom_idx", step.get("site", -1)))
                if not parent_smiles or target_site < 0:
                    continue
                samples.append(
                    RecursiveSample(
                        pathway_index=pathway_idx,
                        drug_name=str(pathway.get("drug_name", f"pathway_{pathway_idx}")),
                        original_drug_smiles=str(pathway.get("drug_smiles", "")),
                        smiles=parent_smiles,
                        metabolite_smiles=metabolite_smiles,
                        target_site=target_site,
                        step_number=step_number,
                        metabolism_type=str(step.get("metabolism_type", step.get("type", "unknown"))),
                        supervision_source=supervision_source,
                        source_weight=float(step.get("source_weight", 1.0)),
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        try:
            structure_mol = self.structure_library.get(sample.smiles) if self.structure_library is not None else None
            graph = smiles_to_graph(
                sample.smiles,
                site_atoms=[sample.target_site],
                structure_mol=structure_mol,
                allow_partial_sanitize=self.allow_partial_sanitize,
                allow_aggressive_repair=self.allow_aggressive_repair,
            )
            if self.include_manual_engine_features:
                graph = attach_manual_engine_features_to_graph(
                    graph,
                    cache_dir=self.manual_feature_cache_dir,
                    allow_partial_sanitize=self.allow_partial_sanitize,
                    allow_aggressive_repair=self.allow_aggressive_repair,
                )
            if self.include_xtb_features and self.xtb_cache_dir:
                graph = attach_xtb_features_to_graph(
                    graph,
                    cache_dir=self.xtb_cache_dir,
                    compute_if_missing=self.compute_xtb_if_missing,
                )
            if graph.site_labels is None or float(graph.site_labels.sum()) <= 0.0:
                raise ValueError(f"Invalid recursive target site for {sample.smiles}: {sample.target_site}")
            graph.cyp_label = None
            graph.has_site_supervision = True
        except Exception:
            if self.drop_failed:
                return {"graph": None, "valid": False}
            raise

        return {
            "graph": graph,
            "valid": True,
            "step_number": int(sample.step_number),
            "source_weight": float(sample.source_weight),
            "supervision_source": sample.supervision_source,
            "pathway_index": int(sample.pathway_index),
            "drug_name": sample.drug_name,
            "metabolism_type": sample.metabolism_type,
            "parent_smiles": sample.smiles,
            "metabolite_smiles": sample.metabolite_smiles,
            "original_drug_smiles": sample.original_drug_smiles,
        }

    def get_stats(self) -> Dict[str, object]:
        step_counts: dict[int, int] = {}
        source_counts: dict[str, int] = {}
        for sample in self.samples:
            step_counts[sample.step_number] = step_counts.get(sample.step_number, 0) + 1
            source_counts[sample.supervision_source] = source_counts.get(sample.supervision_source, 0) + 1
        return {
            "total_samples": len(self.samples),
            "num_pathways": len(self.pathways),
            "expansion_factor": float(len(self.samples) / max(1, len(self.pathways))),
            "samples_per_step": step_counts,
            "samples_per_source": source_counts,
        }


def collate_recursive_batch(batch: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    require_torch()
    batch = [item for item in batch if item.get("graph") is not None and item.get("valid", True)]
    if not batch:
        return None
    graphs = [item["graph"] for item in batch]
    merged = collate_molecule_graphs(graphs)
    merged["graph_step_numbers"] = torch.as_tensor([int(item["step_number"]) for item in batch], dtype=torch.long)
    merged["graph_source_weights"] = torch.as_tensor([float(item["source_weight"]) for item in batch], dtype=torch.float32)
    merged["graph_supervision_sources"] = [str(item["supervision_source"]) for item in batch]
    merged["graph_pathway_indices"] = torch.as_tensor([int(item["pathway_index"]) for item in batch], dtype=torch.long)
    merged["graph_drug_names"] = [str(item["drug_name"]) for item in batch]
    merged["graph_metabolism_types"] = [str(item["metabolism_type"]) for item in batch]
    merged["graph_parent_smiles"] = [str(item["parent_smiles"]) for item in batch]
    merged["graph_metabolite_smiles"] = [str(item["metabolite_smiles"]) for item in batch]
    merged["graph_original_drug_smiles"] = [str(item["original_drug_smiles"]) for item in batch]
    merged["num_graphs"] = len(batch)
    return merged
