from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM


_BASE_GRAPH_ATOM_DIM = 140
_MANUAL_PRIOR_ATOM_DIM = 32


@dataclass(frozen=True)
class MainlineRunConfig:
    preset_name: str
    target_family: str = "CYP3A4"
    train_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_train.json"
    val_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_val.json"
    test_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_test.json"
    strict_val_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_val_exact_clean_eval.json"
    strict_test_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_test_exact_clean_eval.json"
    tiered_val_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_val_tiered_eval.json"
    tiered_test_dataset: str = "data/prepared_training/cyp3a4_phase12_splits/cyp3a4_exact_plus_tiered_test_tiered_eval.json"
    structure_sdf: str = "3D structures.sdf"
    xtb_cache_dir: str = "cache/full_xtb"
    output_dir: str = "artifacts/mainline"
    batch_size: int = 8
    epochs: int = 20
    patience: int = 5
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 5.0
    shortlist_loss_weight: float = 1.0
    winner_loss_weight: float = 0.35
    shortlist_ranking_weight: float = 0.25
    shortlist_rank_window_weight: float = 0.25
    shortlist_hard_negative_weight: float = 0.25
    shortlist_pairwise_margin: float = 0.20
    shortlist_hard_negative_max_per_true: int = 3
    shortlist_candidate_topk: int = 12
    local_winner_topk: int = 6
    shortlist_use_rank_weighting: bool = True
    warm_start_checkpoint: str = ""
    seed: int = 42
    selection_metric: str = "strict_exact.shortlist_recall_at_12"
    high_confidence_thresholds: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)

    def with_overrides(self, **overrides: Any) -> "MainlineRunConfig":
        allowed = {key: value for key, value in overrides.items() if value is not None and hasattr(self, key)}
        return replace(self, **allowed)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["high_confidence_thresholds"] = [float(value) for value in self.high_confidence_thresholds]
        return payload

    @property
    def fixed_cyp_index(self) -> int:
        try:
            return int(MAJOR_CYP_CLASSES.index(str(self.target_family).strip()))
        except ValueError:
            return 0

    @property
    def disable_cyp_task(self) -> bool:
        return str(self.target_family).strip() in set(MAJOR_CYP_CLASSES)

    def build_model_config(self) -> ModelConfig:
        return ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            use_3d_branch=True,
            use_nexus_bridge=False,
            use_nexus_site_arbiter=False,
            use_nexus_sideinfo_features=False,
            use_local_chemistry_path=False,
            disable_cyp_task=bool(self.disable_cyp_task),
            fixed_cyp_index=int(self.fixed_cyp_index),
            candidate_mask_mode="hard",
            candidate_mask_logit_bias=2.0,
            return_intermediate_stats=False,
            manual_atom_feature_dim=_MANUAL_PRIOR_ATOM_DIM + FULL_XTB_FEATURE_DIM,
            atom_input_dim=_BASE_GRAPH_ATOM_DIM + FULL_XTB_FEATURE_DIM,
        )

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[3] / path


MAINLINE_PRESETS: dict[str, MainlineRunConfig] = {
    "mainline_strict_exact_baseline": MainlineRunConfig(
        preset_name="mainline_strict_exact_baseline",
        train_dataset="data/prepared_training/cyp3a4_phase12_splits/cyp3a4_strict_exact_clean_train.json",
        val_dataset="data/prepared_training/cyp3a4_phase12_splits/cyp3a4_strict_exact_clean_val.json",
        test_dataset="data/prepared_training/cyp3a4_phase12_splits/cyp3a4_strict_exact_clean_test.json",
        strict_val_dataset="data/prepared_training/cyp3a4_phase12_splits/cyp3a4_strict_exact_clean_val.json",
        strict_test_dataset="data/prepared_training/cyp3a4_phase12_splits/cyp3a4_strict_exact_clean_test.json",
        tiered_val_dataset="",
        tiered_test_dataset="",
        output_dir="artifacts/mainline_strict_exact_baseline",
        shortlist_rank_window_weight=0.20,
        shortlist_hard_negative_weight=0.20,
        selection_metric="strict_exact.end_to_end_top1",
    ),
    "mainline_exact_plus_tiered": MainlineRunConfig(
        preset_name="mainline_exact_plus_tiered",
        output_dir="artifacts/mainline_exact_plus_tiered",
    ),
    "mainline_exact_plus_tiered_confidence_eval": MainlineRunConfig(
        preset_name="mainline_exact_plus_tiered_confidence_eval",
        output_dir="artifacts/mainline_exact_plus_tiered_confidence_eval",
    ),
}


def get_preset(name: str) -> MainlineRunConfig:
    key = str(name or "").strip()
    if key not in MAINLINE_PRESETS:
        raise KeyError(f"Unknown mainline preset: {name}")
    return MAINLINE_PRESETS[key]
