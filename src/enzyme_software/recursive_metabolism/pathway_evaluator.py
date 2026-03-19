from __future__ import annotations

from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.features.manual_engine_features import attach_manual_engine_features_to_graph
from enzyme_software.liquid_nn_v2.features.xtb_features import attach_xtb_features_to_graph
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device
from .metabolism_simulator import MetabolismSimulator


if TORCH_AVAILABLE:
    class RecursivePathwayEvaluator:
        def __init__(self, model, *, device, structure_library=None, xtb_cache_dir: Optional[str] = None, compute_xtb_if_missing: bool = False, manual_feature_cache_dir: Optional[str] = None):
            self.model = model
            self.device = device
            self.structure_library = structure_library
            self.xtb_cache_dir = xtb_cache_dir
            self.compute_xtb_if_missing = bool(compute_xtb_if_missing)
            self.manual_feature_cache_dir = manual_feature_cache_dir
            self.simulator = MetabolismSimulator()

        def _build_batch(self, smiles: str, step_number: int):
            structure_mol = self.structure_library.get(smiles) if self.structure_library is not None else None
            graph = smiles_to_graph(smiles, site_atoms=None, structure_mol=structure_mol)
            graph = attach_manual_engine_features_to_graph(graph, cache_dir=self.manual_feature_cache_dir)
            if self.xtb_cache_dir:
                graph = attach_xtb_features_to_graph(
                    graph,
                    cache_dir=self.xtb_cache_dir,
                    compute_if_missing=self.compute_xtb_if_missing,
                )
            batch = collate_molecule_graphs([graph])
            batch["graph_step_numbers"] = torch.as_tensor([int(step_number)], dtype=torch.long)
            batch["graph_source_weights"] = torch.as_tensor([1.0], dtype=torch.float32)
            batch["graph_supervision_sources"] = ["predicted"]
            batch["graph_pathway_indices"] = torch.as_tensor([0], dtype=torch.long)
            batch["graph_drug_names"] = [smiles]
            batch["graph_metabolism_types"] = ["unknown"]
            batch["graph_parent_smiles"] = [smiles]
            batch["graph_metabolite_smiles"] = [""]
            batch["graph_original_drug_smiles"] = [smiles]
            batch["num_graphs"] = 1
            return move_to_device(batch, self.device)

        def rollout(self, smiles: str, *, max_steps: int = 6) -> List[Dict[str, object]]:
            self.model.eval()
            current_smiles = str(smiles)
            results: list[dict[str, object]] = []
            seen = set()
            with torch.no_grad():
                for step in range(max_steps):
                    if current_smiles in seen:
                        break
                    seen.add(current_smiles)
                    batch = self._build_batch(current_smiles, step)
                    outputs = self.model(batch)
                    logits = outputs["recursive_site_logits"].squeeze(-1)
                    site_idx = int(torch.argmax(logits).item())
                    reaction = self.simulator.metabolize(current_smiles, site_idx)
                    results.append(
                        {
                            "step": int(step),
                            "parent_smiles": current_smiles,
                            "site_atom_idx": site_idx,
                            "predicted_score": float(logits[site_idx].item()),
                            "metabolism_type": reaction.metabolism_type.value,
                            "success": bool(reaction.success),
                            "metabolite_smiles": reaction.metabolite_smiles,
                            "error": reaction.error,
                        }
                    )
                    if not reaction.success or not reaction.metabolite_smiles:
                        break
                    current_smiles = reaction.metabolite_smiles
            return results

        def evaluate_rollouts(self, pathways: List[dict], *, max_steps: int = 6) -> Dict[str, float]:
            compared = 0
            matched_step0 = 0
            exact_prefix = 0
            for pathway in pathways:
                steps = pathway.get("steps", []) or []
                if not steps:
                    continue
                rollout = self.rollout(str(pathway.get("drug_smiles", "")), max_steps=max_steps)
                if not rollout:
                    continue
                compared += 1
                matched_step0 += int(int(rollout[0]["site_atom_idx"]) == int(steps[0].get("site_atom_idx", steps[0].get("site", -1))))
                prefix_ok = True
                for predicted, truth in zip(rollout, steps):
                    if int(predicted["site_atom_idx"]) != int(truth.get("site_atom_idx", truth.get("site", -1))):
                        prefix_ok = False
                        break
                exact_prefix += int(prefix_ok)
            if compared == 0:
                return {"rollout_step0_acc": 0.0, "rollout_prefix_acc": 0.0}
            return {
                "rollout_step0_acc": float(matched_step0 / compared),
                "rollout_prefix_acc": float(exact_prefix / compared),
            }
else:  # pragma: no cover
    class RecursivePathwayEvaluator:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
