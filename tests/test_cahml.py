from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from enzyme_software.cahml import CAHML, CAHMLConfig, CAHMLDataset, CAHMLTrainer
from enzyme_software.cahml.components.chemistry_encoder import ChemistryFeatureExtractor
from enzyme_software.cahml.components.physics_constraints import PhysicsConstraints


def test_chemistry_feature_extractor_smoke():
    features = ChemistryFeatureExtractor().extract("CCOc1ccccc1")
    assert features is not None
    assert features.num_atoms > 0
    assert features.mol_features_raw.ndim == 1
    assert features.atom_features_raw.shape[0] == features.num_atoms


def test_physics_constraints_block_and_boost():
    extractor = ChemistryFeatureExtractor()
    features = extractor.extract("CC(C)(C)Cc1ccccc1")
    assert features is not None
    constraints = PhysicsConstraints()
    logits = torch.zeros(features.num_atoms)
    constrained, info = constraints.apply_constraints(features.atom_features_raw, features.smarts_matches, logits)
    assert any(idx in info["blocked_atoms"] for idx, row in enumerate(features.atom_features_raw) if row[2].item() == 0.0)
    assert len(info["boosted_atoms"]) >= 1
    assert constrained.shape[0] == features.num_atoms


def test_cahml_trainer_smoke(tmp_path: Path):
    extractor = ChemistryFeatureExtractor()
    chem1 = extractor.extract("CCOc1ccccc1")
    chem2 = extractor.extract("CCN")
    assert chem1 is not None and chem2 is not None
    predictions_path = tmp_path / "predictions.pt"
    payload = {
        "predictions": {
            "CCOc1ccccc1": {
                "site_scores_raw": torch.randn(chem1.num_atoms, 3),
                "cyp_probs_raw": torch.softmax(torch.randn(3, 5), dim=-1),
                "site_labels": torch.tensor([1.0] + [0.0] * (chem1.num_atoms - 1)),
                "cyp_label": torch.tensor(2, dtype=torch.long),
                "num_atoms": torch.tensor(chem1.num_atoms, dtype=torch.long),
            },
            "CCN": {
                "site_scores_raw": torch.randn(chem2.num_atoms, 3),
                "cyp_probs_raw": torch.softmax(torch.randn(3, 5), dim=-1),
                "site_labels": torch.zeros(chem2.num_atoms),
                "cyp_label": torch.tensor(1, dtype=torch.long),
                "num_atoms": torch.tensor(chem2.num_atoms, dtype=torch.long),
            },
        }
    }
    torch.save(payload, predictions_path)
    drugs = [
        {"smiles": "CCOc1ccccc1", "metabolism_sites": [0], "primary_cyp": "CYP2C19"},
        {"smiles": "CCN", "primary_cyp": "CYP2D6"},
    ]
    dataset = CAHMLDataset(predictions_path, drugs)
    model = CAHML(CAHMLConfig(hidden_dim=32, epochs=1, patience=10, checkpoint_dir=str(tmp_path / "ckpt"), artifact_dir=str(tmp_path / "art"), cache_dir=str(tmp_path / "cache")))
    trainer = CAHMLTrainer(model=model, train_dataset=dataset, val_dataset=dataset, config=model.config, device=torch.device("cpu"))
    payload = trainer.train()
    assert "best_val_top1" in payload
    assert Path(model.config.checkpoint_dir, "cahml_latest.pt").exists()
