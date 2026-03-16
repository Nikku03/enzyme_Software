from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from enzyme_software.calibration.drug_metabolism_db import DRUG_DATABASE
from enzyme_software.moduleB.metabolism_site_predictor import (
    predict_drug_metabolism,
    predict_metabolism_sites,
)

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


def _normalize_predicted_cyp(value: Any) -> str:
    raw = str(value or "").upper().strip()
    if raw in CYP_ISOFORMS:
        return raw
    if raw.endswith("_LIKE"):
        raw = raw.replace("_LIKE", "")
    if raw in CYP_ISOFORMS:
        return raw
    if "3A4" in raw:
        return "CYP3A4"
    if "2D6" in raw:
        return "CYP2D6"
    if "2C9" in raw:
        return "CYP2C9"
    if "2C19" in raw:
        return "CYP2C19"
    if "1A2" in raw:
        return "CYP1A2"
    return "UNKNOWN"


def validate_cyp_predictions(predictions: Dict[str, str]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    confusion = {p: {a: 0 for a in CYP_ISOFORMS} for p in CYP_ISOFORMS}

    for drug_key, drug in DRUG_DATABASE.items():
        expected = str(drug.get("primary_cyp") or "").upper()
        predicted_raw = predictions.get(drug_key, predictions.get(str(drug.get("name", "")).lower(), "unknown"))
        predicted = _normalize_predicted_cyp(predicted_raw)
        is_correct = bool(predicted == expected)
        if is_correct:
            correct += 1
        total += 1
        results.append(
            {
                "drug": drug.get("name"),
                "drug_key": drug_key,
                "expected_cyp": expected,
                "predicted_cyp": predicted,
                "correct": is_correct,
            }
        )
        if predicted in CYP_ISOFORMS and expected in CYP_ISOFORMS:
            confusion[predicted][expected] += 1

    accuracy = float(correct) / float(total) if total > 0 else 0.0
    per_cyp: Dict[str, Dict[str, Any]] = {}
    for cyp in CYP_ISOFORMS:
        tp = int(confusion[cyp][cyp])
        fp = sum(int(confusion[cyp][a]) for a in CYP_ISOFORMS if a != cyp)
        fn = sum(int(confusion[p][cyp]) for p in CYP_ISOFORMS if p != cyp)
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_cyp[cyp] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": total,
        "target": "11/14 (0.786)",
        "met_target": bool(correct >= 11),
        "per_drug": results,
        "confusion_matrix": confusion,
        "per_cyp_metrics": per_cyp,
    }


def _classes_match(expected: str, predicted: str) -> bool:
    e = str(expected or "").lower().strip()
    p = str(predicted or "").lower().strip()
    if not e or not p:
        return False
    if e == p:
        return True
    if e.endswith(p) or p.endswith(e):
        return True
    aliases = {
        "ch__alpha_hetero": ["alpha_hetero", "alpha-hetero", "ch_alpha"],
        "ch__benzylic": ["benzylic"],
        "ch__allylic": ["allylic"],
        "ch__aryl": ["aryl", "aromatic"],
        "ch__primary": ["primary", "ch__aliphatic"],
        "ch__secondary": ["secondary"],
    }
    for canonical, alts in aliases.items():
        if e == canonical and p in alts:
            return True
        if p == canonical and e in alts:
            return True
    return False


def validate_site_predictions(
    predictions: Dict[str, List[Dict[str, Any]]],
    top_k: int = 3,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    for drug_key, drug in DRUG_DATABASE.items():
        expected_class = drug.get("expected_bde_class")
        if expected_class is None:
            continue
        predicted_sites = list(predictions.get(drug_key, []))
        top_classes = [str(site.get("bond_class") or "").lower() for site in predicted_sites[: max(1, int(top_k))]]
        found = False
        matched_rank: Optional[int] = None
        for i, pred_class in enumerate(top_classes):
            if _classes_match(str(expected_class), pred_class):
                found = True
                matched_rank = i + 1
                break
        if found:
            correct += 1
        total += 1
        results.append(
            {
                "drug": drug.get("name"),
                "drug_key": drug_key,
                "expected_class": expected_class,
                "top_k_predicted": top_classes,
                "correct": found,
                "matched_rank": matched_rank,
                "n_sites_predicted": len(predicted_sites),
            }
        )

    accuracy = float(correct) / float(total) if total > 0 else 0.0
    by_class: Dict[str, Dict[str, int]] = {}
    for row in results:
        cls = str(row.get("expected_class") or "")
        if cls not in by_class:
            by_class[cls] = {"correct": 0, "total": 0}
        by_class[cls]["total"] += 1
        if row.get("correct"):
            by_class[cls]["correct"] += 1

    return {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": total,
        "top_k": int(top_k),
        "target": "10/14 (0.714)",
        "met_target": bool(correct >= 10),
        "per_drug": results,
        "accuracy_by_class": by_class,
    }


def _default_cyp_predictor(smiles: str) -> Dict[str, Any]:
    return predict_drug_metabolism(smiles=smiles, topk=5, use_xtb=False)


def _default_site_predictor(smiles: str) -> List[Dict[str, Any]]:
    out = predict_metabolism_sites(smiles=smiles, topk=50, use_xtb=False)
    ranked = list(out.get("all_ranked_sites") or out.get("ranked_sites") or [])
    normalized: List[Dict[str, Any]] = []
    for i, site in enumerate(ranked):
        normalized.append(
            {
                "bond_class": site.get("bond_class"),
                "bde_kj_mol": site.get("bde"),
                "rank": i + 1,
            }
        )
    return normalized


def run_full_validation(
    cyp_predictor: Optional[Callable[[str], Dict[str, Any]]] = None,
    site_predictor: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    use_cyp_predictor = cyp_predictor or _default_cyp_predictor
    use_site_predictor = site_predictor or _default_site_predictor

    cyp_predictions: Dict[str, str] = {}
    site_predictions: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[str] = []

    for drug_key, drug in DRUG_DATABASE.items():
        smiles = str(drug.get("smiles") or "")
        try:
            cyp_result = use_cyp_predictor(smiles)
            cyp_predictions[drug_key] = str((cyp_result or {}).get("predicted_cyp", "unknown"))
        except Exception as exc:
            cyp_predictions[drug_key] = "ERROR"
            errors.append(f"CYP prediction failed for {drug.get('name')}: {exc}")
        try:
            site_result = use_site_predictor(smiles)
            site_predictions[drug_key] = list(site_result) if isinstance(site_result, list) else []
        except Exception as exc:
            site_predictions[drug_key] = []
            errors.append(f"Site prediction failed for {drug.get('name')}: {exc}")

    cyp_validation = validate_cyp_predictions(cyp_predictions)
    site_validation = validate_site_predictions(site_predictions)
    total_checks = int(cyp_validation["total"]) + int(site_validation["total"])
    total_correct = int(cyp_validation["correct"]) + int(site_validation["correct"])
    combined_accuracy = float(total_correct) / float(total_checks) if total_checks > 0 else 0.0

    return {
        "cyp_validation": cyp_validation,
        "site_validation": site_validation,
        "combined_accuracy": round(combined_accuracy, 3),
        "combined_correct": total_correct,
        "combined_total": total_checks,
        "errors": errors,
        "summary": {
            "cyp_accuracy": f"{cyp_validation['correct']}/{cyp_validation['total']}",
            "site_accuracy": f"{site_validation['correct']}/{site_validation['total']}",
            "cyp_target_met": cyp_validation["met_target"],
            "site_target_met": site_validation["met_target"],
        },
    }


def print_validation_report(report: Dict[str, Any]) -> None:
    cyp = report["cyp_validation"]
    site = report["site_validation"]
    print("\n" + "=" * 80)
    print("MODULE B VALIDATION REPORT")
    print("=" * 80)
    print(
        f"\nCYP ISOFORM PREDICTION: {cyp['correct']}/{cyp['total']} "
        f"({cyp['accuracy']:.1%}) {'TARGET MET' if cyp['met_target'] else 'BELOW TARGET'}"
    )
    print(f"{'Drug':<20} {'Expected':<10} {'Predicted':<10} {'Result'}")
    for row in cyp["per_drug"]:
        flag = "OK" if row["correct"] else "MISS"
        print(f"  {row['drug']:<18} {row['expected_cyp']:<10} {row['predicted_cyp']:<10} {flag}")
    print(
        f"\nMETABOLISM SITE PREDICTION: {site['correct']}/{site['total']} "
        f"({site['accuracy']:.1%}) {'TARGET MET' if site['met_target'] else 'BELOW TARGET'}"
    )
    print(f"(correct if known site type in top-{site['top_k']} predicted bonds)")
    print(f"{'Drug':<20} {'Expected class':<20} {'In top-k?':<10} {'Rank'}")
    for row in site["per_drug"]:
        flag = "OK" if row["correct"] else "MISS"
        rank_str = str(row["matched_rank"]) if row["matched_rank"] else "-"
        print(f"  {row['drug']:<18} {row['expected_class']:<20} {flag:<10} {rank_str}")
    print(f"\nCOMBINED: {report['combined_correct']}/{report['combined_total']} ({report['combined_accuracy']:.1%})")
    if report.get("errors"):
        print("\nErrors:")
        for err in report["errors"]:
            print(f"  - {err}")


def export_confusion_matrix_data(cyp_validation: Dict[str, Any]) -> Dict[str, Any]:
    matrix: List[List[int]] = []
    for pred in CYP_ISOFORMS:
        matrix.append([int(cyp_validation["confusion_matrix"][pred][actual]) for actual in CYP_ISOFORMS])
    return {"matrix": matrix, "labels": CYP_ISOFORMS, "title": "CYP Isoform Prediction Confusion Matrix"}


def export_site_accuracy_data(site_validation: Dict[str, Any]) -> Dict[str, Any]:
    drugs: List[str] = []
    correct_flags: List[int] = []
    for row in site_validation["per_drug"]:
        drugs.append(str(row.get("drug") or ""))
        correct_flags.append(1 if row.get("correct") else 0)
    return {
        "drugs": drugs,
        "correct": correct_flags,
        "accuracy": site_validation.get("accuracy"),
        "title": f"Metabolism Site Prediction (top-{site_validation.get('top_k')})",
    }


def export_for_json(report: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def _mock_cyp_predictions() -> Dict[str, str]:
    """Mock CYP predictions for dashboard/demo usage."""
    return {
        "ibuprofen": "CYP2C9",
        "diclofenac": "CYP2C9",
        "warfarin": "CYP2C9",
        "tolbutamide": "CYP2C9",
        "codeine": "CYP2D6",
        "dextromethorphan": "CYP2D6",
        "metoprolol": "CYP2D6",
        "midazolam": "CYP3A4",
        "testosterone": "CYP3A4",
        "nifedipine": "CYP3A4",
        "omeprazole": "CYP3A4",   # intentionally wrong
        "clopidogrel": "CYP3A4",  # intentionally wrong
        "caffeine": "CYP1A2",
        "theophylline": "CYP2C19",  # intentionally wrong
    }


def _mock_site_predictions() -> Dict[str, List[Dict[str, Any]]]:
    """Mock site predictions for dashboard/demo usage."""
    return {
        "ibuprofen": [
            {"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1},
            {"bond_class": "ch__primary", "bde_kj_mol": 423.0, "rank": 2},
            {"bond_class": "ch__aryl", "bde_kj_mol": 472.2, "rank": 3},
        ],
        "diclofenac": [{"bond_class": "ch__aryl", "bde_kj_mol": 472.2, "rank": 1}],
        "warfarin": [
            {"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1},
            {"bond_class": "ch__aryl", "bde_kj_mol": 472.2, "rank": 2},
        ],
        "tolbutamide": [
            {"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1},
            {"bond_class": "ch__primary", "bde_kj_mol": 423.0, "rank": 2},
        ],
        "codeine": [{"bond_class": "ch__alpha_hetero", "bde_kj_mol": 385.0, "rank": 1}],
        "dextromethorphan": [{"bond_class": "ch__alpha_hetero", "bde_kj_mol": 385.0, "rank": 1}],
        "metoprolol": [{"bond_class": "ch__alpha_hetero", "bde_kj_mol": 385.0, "rank": 1}],
        "midazolam": [{"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1}],
        "testosterone": [{"bond_class": "ch__allylic", "bde_kj_mol": 371.5, "rank": 1}],
        "nifedipine": [{"bond_class": "ch__allylic", "bde_kj_mol": 371.5, "rank": 1}],
        "omeprazole": [{"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1}],
        "clopidogrel": [{"bond_class": "ch__aryl", "bde_kj_mol": 472.2, "rank": 1}],
        "caffeine": [{"bond_class": "ch__alpha_hetero", "bde_kj_mol": 385.0, "rank": 1}],
        "theophylline": [{"bond_class": "ch__alpha_hetero", "bde_kj_mol": 385.0, "rank": 1}],
    }


__all__ = [
    "CYP_ISOFORMS",
    "validate_cyp_predictions",
    "validate_site_predictions",
    "run_full_validation",
    "print_validation_report",
    "export_confusion_matrix_data",
    "export_site_accuracy_data",
    "export_for_json",
    "_mock_cyp_predictions",
    "_mock_site_predictions",
]


if __name__ == "__main__":
    report = run_full_validation()
    print_validation_report(report)
