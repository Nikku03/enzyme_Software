from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from enzyme_software.meta_learner import MetaLearnerConfig, MetaLearnerDataset
from enzyme_software.meta_learner.multi_head_meta_model import MultiHeadMetaLearner
from enzyme_software.meta_learner.multi_head_trainer import MultiHeadTrainer


def test_multihead_forward_shapes():
    model = MultiHeadMetaLearner(n_models=3, n_cyp=5, atom_feature_dim=11, global_feature_dim=19, hidden_dim=16)
    atom_features = torch.randn(7, 11)
    global_features = torch.randn(19)
    site_scores_raw = torch.randn(7, 3)
    cyp_probs_raw = torch.softmax(torch.randn(3, 5), dim=-1)

    site_scores, cyp_logits, stats = model(atom_features, global_features, site_scores_raw, cyp_probs_raw)

    assert site_scores.shape == (7,)
    assert cyp_logits.shape == (5,)
    assert stats["site_attention"].shape == (3,)
    assert stats["cyp_attention"].shape == (3,)


def test_meta_dataset_exposes_cyp_probs_raw(tmp_path: Path):
    predictions_path = tmp_path / "predictions.pt"
    payload = {
        "predictions": {
            "CCO": {
                "atom_features": torch.randn(4, 11),
                "global_features": torch.randn(19),
                "site_scores_raw": torch.randn(4, 3),
                "cyp_probs_raw": torch.softmax(torch.randn(3, 5), dim=-1),
                "site_labels": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "cyp_label": torch.tensor(2, dtype=torch.long),
                "num_atoms": torch.tensor(4, dtype=torch.long),
            }
        }
    }
    torch.save(payload, predictions_path)
    dataset = MetaLearnerDataset(predictions_path, [{"smiles": "CCO", "metabolism_sites": [1]}])
    row = dataset[0]

    assert row["cyp_probs_raw"].shape == (3, 5)
    assert bool(row["site_supervised"].item()) is True


def test_multihead_trainer_runs_one_epoch(tmp_path: Path):
    predictions_path = tmp_path / "predictions.pt"
    payload = {
        "predictions": {
            "CCO": {
                "atom_features": torch.randn(4, 11),
                "global_features": torch.randn(19),
                "site_scores_raw": torch.randn(4, 3),
                "cyp_probs_raw": torch.softmax(torch.randn(3, 5), dim=-1),
                "site_labels": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "cyp_label": torch.tensor(2, dtype=torch.long),
                "num_atoms": torch.tensor(4, dtype=torch.long),
            },
            "CCC": {
                "atom_features": torch.randn(5, 11),
                "global_features": torch.randn(19),
                "site_scores_raw": torch.randn(5, 3),
                "cyp_probs_raw": torch.softmax(torch.randn(3, 5), dim=-1),
                "site_labels": torch.zeros(5),
                "cyp_label": torch.tensor(1, dtype=torch.long),
                "num_atoms": torch.tensor(5, dtype=torch.long),
            },
        }
    }
    torch.save(payload, predictions_path)
    drugs = [{"smiles": "CCO", "metabolism_sites": [1]}, {"smiles": "CCC"}]
    dataset = MetaLearnerDataset(predictions_path, drugs)
    model = MultiHeadMetaLearner(hidden_dim=8)
    config = MetaLearnerConfig(
        checkpoint_dir=str(tmp_path / "ckpt"),
        artifact_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "cache"),
        epochs=1,
        patience=10,
    )
    trainer = MultiHeadTrainer(model=model, train_dataset=dataset, val_dataset=dataset, config=config, device=torch.device("cpu"))

    payload = trainer.train()

    assert "best_val_top1" in payload
    assert Path(config.checkpoint_dir, "multihead_meta_learner_latest.pt").exists()
