from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_dataset.collect_500_drugs import collect_500_drugs
from scripts.build_dataset.generate_som_labels import generate_all_som_labels, generate_som_for_drug
from scripts.build_dataset.merge_datasets import merge_datasets
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset, collate_fn, create_dataloaders


@pytest.fixture()
def tiny_dataset_payload():
    return [
        {
            "name": "Ibuprofen",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "cyp": "CYP2C9",
            "som": [{"atom_idx": 7, "bond_class": "benzylic", "confidence": 0.9}],
            "confidence": "high",
        },
        {
            "name": "Codeine",
            "smiles": "COC1=CC2=C(C=C1)C3C4C=CC(C2C3O)N(C)CC4",
            "cyp": "CYP2D6",
            "som": [{"atom_idx": 0, "bond_class": "alpha_hetero", "confidence": 0.8}],
            "confidence": "medium",
        },
        {
            "name": "Caffeine",
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "cyp": "CYP1A2",
            "som": [{"atom_idx": 0, "bond_class": "alpha_hetero", "confidence": 0.75}],
            "confidence": "low",
        },
    ]


def test_collect_500_drugs_mocked(monkeypatch, tmp_path):
    def fake_get_json(url, params=None):
        if url.endswith("activity.json"):
            target = params["target_chembl_id"]
            return {
                "activities": [
                    {"molecule_chembl_id": f"{target}_A", "standard_type": "IC50", "standard_value": "12000"},
                    {"molecule_chembl_id": f"{target}_B", "standard_type": "Ki", "standard_value": "15000"},
                ]
            }
        chembl_id = Path(url).stem
        return {
            "molecule_structures": {
                "canonical_smiles": "CCO" if chembl_id.endswith("A") else "CCN"
            }
        }

    monkeypatch.setattr("scripts.build_dataset.collect_500_drugs._chembl_get_json", fake_get_json)
    monkeypatch.setattr("scripts.build_dataset.collect_500_drugs.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "scripts.build_dataset.collect_500_drugs.DEFAULT_PER_CYP_TARGET",
        {"CYP1A2": 2, "CYP2C9": 2, "CYP2C19": 2, "CYP2D6": 2, "CYP3A4": 2},
    )
    output_path = tmp_path / "raw.json"
    rows = collect_500_drugs(str(output_path))
    assert output_path.exists()
    assert len(rows) >= 2
    assert all("smiles" in row for row in rows)
    assert any(len(row.get("all_cyps", [])) > 1 for row in rows)


def test_generate_som_labels_pipeline(tmp_path):
    raw = [
        {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "cyp": "CYP2C9"},
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "cyp": "CYP1A2"},
    ]
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "labeled.json"
    raw_path.write_text(json.dumps(raw))
    rows = generate_all_som_labels(str(raw_path), str(out_path))
    assert out_path.exists()
    assert len(rows) == 2
    assert all(row["som"] for row in rows)
    assert all("primary_som" in row for row in rows)


def test_generate_som_for_drug_returns_ranked_sites():
    rows = generate_som_for_drug("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CYP2C9")
    assert rows
    assert rows[0]["score"] >= rows[-1]["score"]
    assert "bond_class" in rows[0]


def test_merge_datasets_with_validated_override(tmp_path):
    validated = [
        {
            "name": "Ibuprofen",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "primary_cyp": "CYP2C9",
            "som": [{"atom_idx": 7, "bond_class": "benzylic", "confidence": 1.0}],
        }
    ]
    pseudo = [
        {
            "name": "Ibuprofen pseudo",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "cyp": "CYP2C9",
            "som": [{"atom_idx": 1, "bond_class": "secondary_CH", "confidence": 0.4}],
        },
        {
            "name": "Caffeine",
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "cyp": "CYP1A2",
            "som": [{"atom_idx": 0, "bond_class": "alpha_hetero", "confidence": 0.8}],
        },
    ]
    validated_path = tmp_path / "validated.json"
    pseudo_path = tmp_path / "pseudo.json"
    out_path = tmp_path / "merged.json"
    validated_path.write_text(json.dumps(validated))
    pseudo_path.write_text(json.dumps(pseudo))
    merged = merge_datasets(str(validated_path), str(pseudo_path), str(out_path))
    payload = json.loads(out_path.read_text())
    assert len(merged) == 2
    assert payload["metadata"]["validated_count"] == 1
    assert any(row.get("source") == "validated" for row in payload["drugs"])
    assert any(row.get("name") == "Caffeine" for row in payload["drugs"])


def test_dataset_loader_and_collate(tmp_path, tiny_dataset_payload):
    dataset_path = tmp_path / "training.json"
    dataset_path.write_text(json.dumps({"drugs": tiny_dataset_payload}))
    ds = CYPMetabolismDataset(str(dataset_path), split="train", train_ratio=1.0, val_ratio=0.0, augment=False)
    assert len(ds) == 3
    batch = collate_fn([ds[0], ds[1]])
    assert batch is not None
    assert batch["x"].shape[0] > 0
    assert batch["graph_confidence_weights"].shape[0] == 2
    assert batch["node_confidence_weights"].shape[0] == batch["x"].shape[0]


def test_create_dataloaders(tmp_path, tiny_dataset_payload):
    dataset_path = tmp_path / "training.json"
    dataset_path.write_text(json.dumps({"drugs": tiny_dataset_payload}))
    train_loader, val_loader, test_loader = create_dataloaders(str(dataset_path), batch_size=2)
    assert len(train_loader.dataset) >= 1
    assert len(val_loader.dataset) >= 0
    assert len(test_loader.dataset) >= 0


def test_train_script_smoke(tmp_path, tiny_dataset_payload):
    dataset_path = tmp_path / "training.json"
    dataset_path.write_text(json.dumps({"drugs": tiny_dataset_payload}))
    output_dir = tmp_path / "checkpoints"
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--dataset",
            str(dataset_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved_report=" in result.stdout
    reports = list(output_dir.glob("*_training_report.json"))
    assert reports
