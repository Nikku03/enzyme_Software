from __future__ import annotations

from enzyme_software.calibration.drug_metabolism_db import DRUG_DATABASE
from enzyme_software.moduleB.module_b_validation import (
    CYP_ISOFORMS,
    export_confusion_matrix_data,
    export_site_accuracy_data,
    run_full_validation,
    validate_cyp_predictions,
    validate_site_predictions,
)


def test_validate_cyp_predictions_perfect_accuracy():
    preds = {k: v["primary_cyp"] for k, v in DRUG_DATABASE.items()}
    out = validate_cyp_predictions(preds)
    assert out["total"] == len(DRUG_DATABASE)
    assert out["correct"] == len(DRUG_DATABASE)
    assert out["accuracy"] == 1.0
    assert out["met_target"] is True


def test_validate_site_predictions_smoke():
    preds = {}
    for k, v in DRUG_DATABASE.items():
        cls = v.get("expected_bde_class")
        preds[k] = [{"bond_class": cls, "bde_kj_mol": 380.0, "rank": 1}] if cls else []
    out = validate_site_predictions(preds, top_k=3)
    assert out["total"] > 0
    assert out["accuracy"] == 1.0
    assert out["met_target"] is True


def test_run_full_validation_with_mock_predictors():
    def cyp_fn(smiles: str):
        # Deterministic mock: intentionally imperfect.
        return {"predicted_cyp": "CYP3A4"}

    def site_fn(smiles: str):
        return [{"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1}]

    out = run_full_validation(cyp_predictor=cyp_fn, site_predictor=site_fn)
    assert "cyp_validation" in out
    assert "site_validation" in out
    assert "combined_accuracy" in out
    assert out["combined_total"] == out["cyp_validation"]["total"] + out["site_validation"]["total"]


def test_exporters_shape():
    preds = {k: v["primary_cyp"] for k, v in DRUG_DATABASE.items()}
    cyp_out = validate_cyp_predictions(preds)
    cm = export_confusion_matrix_data(cyp_out)
    assert cm["labels"] == CYP_ISOFORMS
    assert len(cm["matrix"]) == len(CYP_ISOFORMS)

    site_preds = {k: [{"bond_class": (v.get("expected_bde_class") or ""), "bde_kj_mol": 390.0, "rank": 1}] for k, v in DRUG_DATABASE.items()}
    site_out = validate_site_predictions(site_preds)
    bars = export_site_accuracy_data(site_out)
    assert len(bars["drugs"]) == site_out["total"]
    assert len(bars["correct"]) == site_out["total"]

