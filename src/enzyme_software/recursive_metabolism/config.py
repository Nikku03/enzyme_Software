from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RecursiveMetabolismConfig:
    base_checkpoint: str = "checkpoints/hybrid_lnn_latest.pt"
    checkpoint_dir: str = "checkpoints/recursive_metabolism"
    artifact_dir: str = "artifacts/recursive_metabolism"
    pathway_cache_dir: str = "cache/recursive_metabolism"
    xtb_cache_dir: str = "cache/recursive_metabolism/xtb"
    manual_feature_cache_dir: str | None = None
    structure_sdf: str = "3D structures.sdf"
    train_ratio: float = 0.68
    val_ratio: float = 0.16
    seed: int = 42
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    max_steps: int = 6
    min_heavy_atoms: int = 5
    step_weight_decay: float = 0.9
    freeze_base_model: bool = True
    unfreeze_after_epochs: int = 0
    finetune_learning_rate: float | None = None
    step_embedding_dim: int = 16
    recursive_hidden_dim: int = 128
    recursive_dropout: float = 0.1
    recursive_scale_init: float = 0.15
    include_manual_engine_features: bool = True
    include_xtb_features: bool = True
    compute_xtb_if_missing: bool = False
    allow_partial_sanitize: bool = True
    allow_aggressive_repair: bool = False
    drop_failed: bool = True
    pseudo_step_weight: float = 0.35
    ground_truth_step_weight: float = 1.0
    early_stopping_patience: int = 8

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pathway_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.xtb_cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls, **overrides) -> "RecursiveMetabolismConfig":
        cfg = cls()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg
