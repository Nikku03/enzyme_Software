from __future__ import annotations

import pytest


def test_moduleb2_validation_top3_not_worse_than_top1_on_subset():
    pytest.importorskip("rdkit")
    from enzyme_software.calibration.drug_metabolism_db import DRUG_DATABASE
    from enzyme_software.modules.moduleB2_validation import run_drug_metabolism_validation

    subset_keys = ["ibuprofen", "codeine", "midazolam"]
    subset = {k: DRUG_DATABASE[k] for k in subset_keys}
    report = run_drug_metabolism_validation(drug_db=subset, topk_list=(1, 3))

    assert report.get("total_drugs") == len(subset)
    metrics = report.get("metrics") or {}
    assert "top1_acc" in metrics
    assert "top3_acc" in metrics
    assert float(metrics["top3_acc"]) >= float(metrics["top1_acc"])
    buckets = report.get("error_buckets") or {}
    for key in ["site_enumeration_missing", "route_mismatch", "ranking_miss", "ground_truth_ambiguous"]:
        assert key in buckets
