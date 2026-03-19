from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from enzyme_software.liquid_nn_v2.config import ModelConfig


@dataclass
class MicroPatternXTBConfig:
    base_checkpoint: str = "checkpoints/hybrid_lnn_latest.pt"
    checkpoint_dir: str = "checkpoints/micropattern_xtb"
    artifact_dir: str = "artifacts/micropattern_xtb"
    xtb_cache_dir: str = "cache/micropattern_xtb"
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            use_3d_branch=True,
            return_intermediate_stats=True,
        )
    )
    top_k_candidates: int = 10
    micropattern_radius: int = 2
    reranker_hidden_dim: int = 128
    reranker_dropout: float = 0.1
    reranker_scale_init: float = 0.2
    candidate_ce_weight: float = 1.0
    pairwise_margin_weight: float = 0.5
    pairwise_margin: float = 0.2
    mirank_weight: float = 0.75
    listmle_weight: float = 0.2
    hard_negative_fraction: Optional[float] = 0.5
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 10
    batch_size: int = 16
    freeze_base_model: bool = True
    unfreeze_after_epochs: int = 0
    finetune_learning_rate: Optional[float] = None
    compute_xtb_if_missing: bool = False
    allow_partial_sanitize: bool = True
    allow_aggressive_repair: bool = False
    drop_failed: bool = True
    evaluation_split: str = "test"

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(self.xtb_cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls, **overrides) -> "MicroPatternXTBConfig":
        cfg = cls()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg
