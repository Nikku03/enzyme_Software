from __future__ import annotations

import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from contextlib import redirect_stdout

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset, collate_fn
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import FullXTBHybridDataset
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.model import (
    MicroPatternXTBHybridModel,
    load_base_hybrid_checkpoint,
)
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.xtb_features import attach_xtb_features_to_graph
from enzyme_software.liquid_nn_v2.training.utils import move_to_device


@dataclass
class BaseCheckpointSpec:
    name: str
    checkpoint_path: str
    family: str
    source_checkpoint: Optional[str] = None
    xtb_cache_dir: Optional[str] = None
    compute_xtb_if_missing: bool = False


DEFAULT_MODEL_SPECS = {
    "hybrid_lnn": BaseCheckpointSpec(
        name="hybrid_lnn",
        checkpoint_path="checkpoints/hybrid_lnn_best.pt",
        family="hybrid_lnn",
    ),
    "hybrid_full_xtb": BaseCheckpointSpec(
        name="hybrid_full_xtb",
        checkpoint_path="checkpoints/hybrid_full_xtb/hybrid_full_xtb_best.pt",
        family="hybrid_full_xtb",
        xtb_cache_dir="cache/full_xtb",
    ),
    "micropattern_xtb": BaseCheckpointSpec(
        name="micropattern_xtb",
        checkpoint_path="checkpoints/micropattern_xtb/micropattern_xtb_best.pt",
        family="micropattern_xtb",
        source_checkpoint="checkpoints/hybrid_lnn_best.pt",
        xtb_cache_dir="cache/full_xtb",
        compute_xtb_if_missing=True,
    ),
}


def load_default_model_specs(names: Optional[Iterable[str]] = None) -> List[BaseCheckpointSpec]:
    selected = list(names or DEFAULT_MODEL_SPECS.keys())
    specs: List[BaseCheckpointSpec] = []
    for name in selected:
        spec = DEFAULT_MODEL_SPECS.get(str(name))
        if spec is None:
            raise KeyError(f"Unknown model spec: {name}")
        specs.append(spec)
    return specs


def _load_drugs(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _single_item_batch(dataset) -> Optional[Dict[str, object]]:
    item = dataset[0]
    batch = collate_fn([item])
    return batch


def _resolve_device(name: Optional[str]):
    require_torch()
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_hybrid_family_checkpoint(spec: BaseCheckpointSpec, device):
    payload = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    base_cfg = payload.get("config", {}).get("base_model") or ModelConfig.light_advanced(
        use_manual_engine_priors=True,
        use_3d_branch=True,
        return_intermediate_stats=True,
    ).__dict__
    base_model = LiquidMetabolismNetV2(ModelConfig(**base_cfg))
    model = HybridLNNModel(base_model)
    state_dict = payload.get("model_state_dict") or payload
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, payload


def _load_micropattern_checkpoint(spec: BaseCheckpointSpec, device):
    payload = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    config_payload = payload.get("config") or {}
    config = MicroPatternXTBConfig.default()
    for key, value in config_payload.items():
        if hasattr(config, key):
            setattr(config, key, value)
    source_checkpoint = spec.source_checkpoint or payload.get("source_checkpoint") or config.base_checkpoint
    config.base_checkpoint = str(source_checkpoint)
    if spec.xtb_cache_dir:
        config.xtb_cache_dir = str(spec.xtb_cache_dir)
    base_model = load_base_hybrid_checkpoint(config.base_checkpoint, device=device)
    model = MicroPatternXTBHybridModel(base_model, config=config)
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)
    model.to(device)
    model.eval()
    return model, payload


class BaseModelWrapper:
    def __init__(self, spec: BaseCheckpointSpec, *, device: Optional[str] = None, structure_sdf: Optional[str] = None):
        require_torch()
        self.spec = spec
        self.device = _resolve_device(device)
        self.structure_library = StructureLibrary.from_sdf(structure_sdf) if structure_sdf else None
        if spec.family == "micropattern_xtb":
            self.model, self.payload = _load_micropattern_checkpoint(spec, self.device)
        else:
            self.model, self.payload = _load_hybrid_family_checkpoint(spec, self.device)

    def _drug_row(self, drug: Dict[str, object]) -> Dict[str, object]:
        row = dict(drug)
        row.setdefault("source", "meta_learner")
        row.setdefault("confidence", "validated")
        return row

    def _supports_drug(self, drug: Dict[str, object]) -> bool:
        cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "").strip()
        return cyp in set(MAJOR_CYP_CLASSES)

    def build_batch(self, drug: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self._supports_drug(drug):
            return None
        row = self._drug_row(drug)
        if self.spec.family == "hybrid_full_xtb":
            with io.StringIO() as buf, redirect_stdout(buf):
                dataset = FullXTBHybridDataset(
                    split="test",
                    augment=False,
                    drugs=[row],
                    structure_library=self.structure_library,
                    use_manual_engine_features=True,
                    full_xtb_cache_dir=self.spec.xtb_cache_dir,
                    compute_full_xtb_if_missing=self.spec.compute_xtb_if_missing,
                    drop_failed=True,
                )
            return _single_item_batch(dataset)

        with io.StringIO() as buf, redirect_stdout(buf):
            dataset = CYPMetabolismDataset(
                split="test",
                augment=False,
                drugs=[row],
                structure_library=self.structure_library,
                use_manual_engine_features=True,
                drop_failed=True,
            )
        item = dataset[0]
        if self.spec.family == "micropattern_xtb":
            graph = item.get("graph")
            if graph is None:
                return None
            item = dict(item)
            item["graph"] = attach_xtb_features_to_graph(
                graph,
                cache_dir=self.spec.xtb_cache_dir or "cache/micropattern_xtb",
                compute_if_missing=self.spec.compute_xtb_if_missing,
            )
        return collate_fn([item])

    def predict(self, drug: Dict[str, object]) -> Optional[Dict[str, object]]:
        batch = self.build_batch(drug)
        if batch is None:
            return None
        batch = move_to_device(batch, self.device)
        with torch.no_grad():
            if self.spec.family == "micropattern_xtb":
                outputs = self.model(batch)
                site_scores = torch.sigmoid(outputs["reranked_site_logits"]).detach().cpu().squeeze(-1)
                cyp_logits = outputs["base_outputs"]["cyp_logits"].detach().cpu()
                extra = {
                    "base_site_scores": torch.sigmoid(outputs["base_site_logits"]).detach().cpu().squeeze(-1),
                    "stats": {
                        key: float(value)
                        for key, value in (outputs.get("stats") or {}).items()
                        if isinstance(value, (int, float))
                    },
                }
            else:
                outputs = self.model(batch)
                site_scores = outputs["site_scores"].detach().cpu().squeeze(-1)
                cyp_logits = outputs["cyp_logits"].detach().cpu()
                extra = {}
        cyp_probs = torch.softmax(cyp_logits, dim=-1).squeeze(0)
        site_labels = batch.get("site_labels")
        if site_labels is not None:
            site_labels = site_labels.detach().cpu().view(-1)
        cyp_labels = batch.get("cyp_labels")
        cyp_label = int(cyp_labels[0].detach().cpu().item()) if cyp_labels is not None else None
        return {
            "site_scores": site_scores,
            "cyp_probs": cyp_probs,
            "site_labels": site_labels,
            "cyp_label": cyp_label,
            "num_atoms": int(site_scores.shape[0]),
            "smiles": str(batch.get("canonical_smiles", [drug.get("smiles", "")])[0]),
            "model_name": self.spec.name,
            **extra,
        }


class MultiModelPredictor:
    def __init__(
        self,
        specs: Optional[Iterable[BaseCheckpointSpec]] = None,
        *,
        device: Optional[str] = None,
        structure_sdf: Optional[str] = None,
    ):
        require_torch()
        self.device = _resolve_device(device)
        self.wrappers: Dict[str, BaseModelWrapper] = {}
        for spec in (list(specs) if specs is not None else load_default_model_specs()):
            path = Path(spec.checkpoint_path)
            if not path.exists():
                continue
            self.wrappers[spec.name] = BaseModelWrapper(spec, device=str(self.device), structure_sdf=structure_sdf)

    def predict_all(self, drug: Dict[str, object]) -> Dict[str, Optional[Dict[str, object]]]:
        return {name: wrapper.predict(drug) for name, wrapper in self.wrappers.items()}
