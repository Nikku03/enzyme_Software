from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetaLearnerConfig:
    checkpoint_dir: str = "checkpoints/meta_learner"
    artifact_dir: str = "artifacts/meta_learner"
    cache_dir: str = "cache/meta_learner"
    hidden_dim: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 50
    patience: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    use_attention: bool = True
    mirank_weight: float = 1.0
    bce_weight: float = 0.3
    listmle_weight: float = 0.0
    focal_weight: float = 0.0
    ranking_margin: float = 1.0
    hard_negative_fraction: float | None = None

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
