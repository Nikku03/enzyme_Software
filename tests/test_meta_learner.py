from __future__ import annotations

from pathlib import Path

import torch

from enzyme_software.meta_learner.base_model_wrapper import BaseCheckpointSpec
from enzyme_software.meta_learner.feature_stacker import FeatureStacker
from enzyme_software.meta_learner.meta_model import MetaLearner
from enzyme_software.meta_learner.meta_trainer import MetaLearnerDataset


def test_feature_stacker_shapes():
    stacker = FeatureStacker(["a", "b", "c"], num_cyp=5)
    preds = {
        "a": {"site_scores": torch.tensor([0.1, 0.2, 0.3]), "cyp_probs": torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05]), "site_labels": torch.tensor([0.0, 1.0, 0.0]), "cyp_label": 0, "num_atoms": 3},
        "b": {"site_scores": torch.tensor([0.3, 0.4, 0.1]), "cyp_probs": torch.tensor([0.2, 0.5, 0.1, 0.1, 0.1]), "site_labels": torch.tensor([0.0, 1.0, 0.0]), "cyp_label": 0, "num_atoms": 3},
        "c": {"site_scores": torch.tensor([0.2, 0.8, 0.1]), "cyp_probs": torch.tensor([0.1, 0.2, 0.2, 0.2, 0.3]), "site_labels": torch.tensor([0.0, 1.0, 0.0]), "cyp_label": 0, "num_atoms": 3},
    }
    out = stacker.stack(preds)
    assert out["atom_features"].shape == (3, 11)
    assert out["global_features"].shape == (19,)
    assert out["site_scores_raw"].shape == (3, 3)


def test_meta_learner_forward():
    model = MetaLearner(n_models=3, n_cyp=5, atom_feature_dim=11, global_feature_dim=19, hidden_dim=16, use_attention=True)
    atom_features = torch.randn(4, 11)
    global_features = torch.randn(19)
    site_scores_raw = torch.randn(4, 3)
    site_logits, cyp_logits, stats = model(atom_features, global_features, site_scores_raw)
    assert site_logits.shape == (4,)
    assert cyp_logits.shape == (5,)
    assert "attention_weights" in stats


def test_meta_dataset_reads_cached_predictions(tmp_path: Path):
    cache_path = tmp_path / "predictions.pt"
    torch.save(
        {
            "predictions": {
                "CCO": {
                    "atom_features": torch.zeros((3, 11)),
                    "global_features": torch.zeros((19,)),
                    "site_scores_raw": torch.zeros((3, 3)),
                    "site_labels": torch.tensor([0.0, 1.0, 0.0]),
                    "cyp_label": torch.tensor(1, dtype=torch.long),
                    "num_atoms": torch.tensor(3, dtype=torch.long),
                }
            }
        },
        cache_path,
    )
    ds = MetaLearnerDataset(cache_path, [{"smiles": "CCO"}, {"smiles": "CCC"}])
    assert len(ds) == 1
    row = ds[0]
    assert row["atom_features"].shape == (3, 11)
    assert int(row["cyp_label"].item()) == 1
