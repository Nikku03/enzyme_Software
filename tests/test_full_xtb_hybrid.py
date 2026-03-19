from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.model_utils import (
    MANUAL_ATOM_PROJECTION_KEY,
    expand_manual_atom_projection,
)
from enzyme_software.liquid_nn_v2.features.xtb_features import (
    FULL_XTB_FEATURE_DIM,
    extract_full_xtb_features,
    load_or_compute_full_xtb_features,
)


def test_extract_full_xtb_features_uses_relaxation_and_confidence():
    module_minus1 = {
        "resolved_target": {
            "candidate_bonds": [
                {
                    "heavy_atom_index": 2,
                    "bde_kj_mol": 380.0,
                }
            ]
        },
        "cpt_scores": {
            "bde": {
                "corrected_kj_mol": 380.0,
                "xtb_bde_vertical_kj_mol": 410.0,
                "xtb_bde_adiabatic_kj_mol": 360.0,
                "xtb_relaxation_energy_kj_mol": 50.0,
                "lookup_bde_kj_mol": 390.0,
                "deviation_kj_mol": 10.0,
                "source": "xtb_validated",
            }
        },
    }

    features = extract_full_xtb_features(module_minus1, num_atoms=5)
    assert tuple(features.shape) == (5, FULL_XTB_FEATURE_DIM)
    assert features[2, 4].item() == pytest.approx(1.0, abs=1e-6)
    assert features[2, 6].item() == pytest.approx(1.0, abs=1e-6)
    assert float(features.sum().item()) > 0.0


def test_expand_manual_atom_projection_preserves_existing_columns():
    state_dict = {
        MANUAL_ATOM_PROJECTION_KEY: torch.arange(128 * 32, dtype=torch.float32).reshape(128, 32),
    }
    expanded = expand_manual_atom_projection(state_dict, new_input_dim=40)
    weight = expanded[MANUAL_ATOM_PROJECTION_KEY]
    assert tuple(weight.shape) == (128, 40)
    assert torch.equal(weight[:, :32], state_dict[MANUAL_ATOM_PROJECTION_KEY])
    assert torch.count_nonzero(weight[:, 32:]).item() == 0


def test_load_or_compute_full_xtb_features_uses_cache(tmp_path, monkeypatch):
    payload = {
        "canonical_smiles": "CCO",
        "xtb_valid": True,
        "atom_valid_mask": [[1.0], [1.0], [0.0]],
        "atom_features": [[0.1] * FULL_XTB_FEATURE_DIM, [0.2] * FULL_XTB_FEATURE_DIM, [0.0] * FULL_XTB_FEATURE_DIM],
        "feature_names": ["x"] * FULL_XTB_FEATURE_DIM,
        "status": "ok",
        "error": None,
    }

    calls = {"count": 0}

    def fake_compute(smiles: str, *, target_bond=None):
        calls["count"] += 1
        return payload

    monkeypatch.setattr("enzyme_software.liquid_nn_v2.features.xtb_features.compute_full_xtb_payload", fake_compute)
    out1 = load_or_compute_full_xtb_features("CCO", cache_dir=tmp_path, compute_if_missing=True)
    out2 = load_or_compute_full_xtb_features("CCO", cache_dir=tmp_path, compute_if_missing=True)

    assert out1["xtb_valid"] is True
    assert out2["status"] == "ok"
    assert calls["count"] == 1
    assert any(Path(tmp_path).glob("*_full.json"))
