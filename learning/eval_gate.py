from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.evidence_store import load_labeled_examples
from enzyme_software.unity_schema import UnityRecord, build_features

EPS = 1e-12
MIN_AUC = 0.55
MAX_RELIABILITY_ERROR = 0.25


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [val / total for val in exp_scores]


def _predict_calibrator(features: Dict[str, float], model: Dict[str, Any]) -> float:
    order = model.get("feature_order") or []
    weights = model.get("weights") or []
    means = model.get("feature_mean") or []
    scales = model.get("feature_scale") or []
    bias = float(model.get("bias") or 0.0)
    z = bias
    for idx, name in enumerate(order):
        value = float(features.get(name, 0.0))
        mean = float(means[idx]) if idx < len(means) else 0.0
        scale = float(scales[idx]) if idx < len(scales) and scales[idx] else 1.0
        weight = float(weights[idx]) if idx < len(weights) else 0.0
        z += weight * ((value - mean) / scale)
    return _sigmoid(z)


def _predict_failure_mode(features: Dict[str, float], model: Dict[str, Any]) -> str:
    order = model.get("feature_order") or []
    weights = model.get("weights") or []
    bias = model.get("bias") or []
    means = model.get("feature_mean") or []
    scales = model.get("feature_scale") or []
    classes = model.get("classes") or []
    scores = []
    for c_idx in range(len(classes)):
        z = float(bias[c_idx]) if c_idx < len(bias) else 0.0
        for f_idx, name in enumerate(order):
            value = float(features.get(name, 0.0))
            mean = float(means[f_idx]) if f_idx < len(means) else 0.0
            scale = float(scales[f_idx]) if f_idx < len(scales) and scales[f_idx] else 1.0
            weight = float(weights[c_idx][f_idx]) if c_idx < len(weights) else 0.0
            z += weight * ((value - mean) / scale)
        scores.append(z)
    probs = _softmax(scores)
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return classes[idx] if classes else "unknown"


def _logloss(y_true: List[int], y_pred: List[float]) -> float:
    total = 0.0
    for y, p in zip(y_true, y_pred):
        prob = min(1.0 - EPS, max(EPS, p))
        total += -(y * math.log(prob) + (1 - y) * math.log(1 - prob))
    return total / max(1, len(y_true))


def _brier(y_true: List[int], y_pred: List[float]) -> float:
    total = 0.0
    for y, p in zip(y_true, y_pred):
        total += (p - y) ** 2
    return total / max(1, len(y_true))


def _accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def _hash_split(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "test" if bucket < 20 else "train"


def _collect_split(
    examples: List[Tuple[UnityRecord, Dict[str, Any]]]
) -> Tuple[List[Tuple[UnityRecord, Dict[str, Any]]], List[Tuple[UnityRecord, Dict[str, Any]]]]:
    train = []
    test = []
    for record, payload in examples:
        split = _hash_split(record.run_id)
        if split == "test":
            test.append((record, payload))
        else:
            train.append((record, payload))
    return train, test


def _resolve_pack(path: str) -> Dict[str, Path]:
    candidate = Path(path)
    if candidate.is_dir():
        return {
            "calibrator": candidate / "calibration_module0_v1.json",
            "failure_mode": candidate / "failure_mode_v1.json",
        }
    if candidate.is_file():
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if "calibrator" in data or "failure_mode" in data:
            base = candidate.parent
            return {
                "calibrator": (base / data.get("calibrator", "")).resolve(),
                "failure_mode": (base / data.get("failure_mode", "")).resolve(),
            }
    raise ValueError("Candidate pack must be a directory or a manifest JSON.")


def _load_latest_pack(latest_path: Path) -> Optional[Dict[str, Path]]:
    if not latest_path.exists():
        return None
    data = json.loads(latest_path.read_text(encoding="utf-8"))
    pack_path = data.get("path") or data.get("pack_path") or data.get("artifacts_path")
    if not pack_path:
        return None
    return _resolve_pack(pack_path)


def _auc(y_true: List[int], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    paired = sorted(zip(y_pred, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = 0.0
    for idx, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += idx
    return (rank_sum - (n_pos * (n_pos + 1)) / 2) / (n_pos * n_neg)


def _reliability_error(y_true: List[int], y_pred: List[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    bucket = [[] for _ in range(bins)]
    for truth, pred in zip(y_true, y_pred):
        idx = min(bins - 1, int(pred * bins))
        bucket[idx].append((truth, pred))
    total = len(y_true)
    ece = 0.0
    for group in bucket:
        if not group:
            continue
        avg_pred = sum(p for _, p in group) / len(group)
        avg_true = sum(t for t, _ in group) / len(group)
        ece += (len(group) / total) * abs(avg_pred - avg_true)
    return ece


def _monotonic_ok(
    deltas: List[float],
    preds: List[float],
    families: List[str],
    tolerance: float = 0.02,
) -> bool:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for delta_g, pred, family in zip(deltas, preds, families):
        grouped.setdefault(family, []).append((delta_g, pred))
    for _, items in grouped.items():
        items.sort(key=lambda x: x[0])
        for idx in range(1, len(items)):
            if items[idx][1] > items[idx - 1][1] + tolerance:
                return False
    return True


def _calibrator_predictions(
    model: Dict[str, Any], examples: List[Tuple[UnityRecord, Dict[str, Any]]]
) -> Tuple[List[int], List[float], List[float], List[str]]:
    y_true: List[int] = []
    y_pred: List[float] = []
    delta_g: List[float] = []
    families: List[str] = []
    for record, payload in examples:
        outcomes = payload.get("outcomes", [])
        for outcome in outcomes:
            if "any_activity" not in outcome:
                continue
            y_true.append(1 if outcome["any_activity"] else 0)
            features = build_features(record)
            y_pred.append(_predict_calibrator(features, model))
            delta_g.append(float(features.get("deltaG_dagger_kJ", 0.0)))
            if features.get("route_family_serine_hydrolase", 0.0) >= 0.5:
                families.append("serine_hydrolase")
            elif features.get("route_family_metallo_esterase", 0.0) >= 0.5:
                families.append("metallo_esterase")
            else:
                families.append("other")
    return y_true, y_pred, delta_g, families


def _eval_failure_mode(
    model: Dict[str, Any], examples: List[Tuple[UnityRecord, Dict[str, Any]]]
) -> Tuple[float, int]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for record, payload in examples:
        outcomes = payload.get("outcomes", [])
        for outcome in outcomes:
            failure_mode = outcome.get("failure_mode")
            if not failure_mode:
                continue
            y_true.append(str(failure_mode))
            features = build_features(record)
            y_pred.append(_predict_failure_mode(features, model))
    if not y_true:
        return 0.0, 0
    return _accuracy(y_true, y_pred), len(y_true)


def _metric_ok_low(candidate: float, baseline: Optional[float]) -> bool:
    if baseline is None:
        return True
    return candidate <= baseline * 1.01


def _metric_ok_high(candidate: float, baseline: Optional[float]) -> bool:
    if baseline is None:
        return True
    return candidate >= baseline * 0.99


def evaluate(
    db_path: str,
    candidate_pack: str,
    latest_path: str,
) -> int:
    examples = load_labeled_examples(db_path)
    _, test_examples = _collect_split(examples)
    if not test_examples:
        raise ValueError("No held-out examples found for evaluation.")

    candidate_paths = _resolve_pack(candidate_pack)
    if not candidate_paths["calibrator"].is_file():
        raise ValueError("Missing candidate calibration_module0_v1.json.")
    if not candidate_paths["failure_mode"].is_file():
        raise ValueError("Missing candidate failure_mode_v1.json.")

    candidate_cal = json.loads(candidate_paths["calibrator"].read_text(encoding="utf-8"))
    candidate_fail = json.loads(candidate_paths["failure_mode"].read_text(encoding="utf-8"))

    baseline_paths = _load_latest_pack(Path(latest_path))
    baseline_cal = None
    baseline_fail = None
    if baseline_paths:
        if baseline_paths["calibrator"].is_file():
            baseline_cal = json.loads(
                baseline_paths["calibrator"].read_text(encoding="utf-8")
            )
        if baseline_paths["failure_mode"].is_file():
            baseline_fail = json.loads(
                baseline_paths["failure_mode"].read_text(encoding="utf-8")
            )

    cand_y, cand_pred, cand_delta_g, cand_fam = _calibrator_predictions(
        candidate_cal, test_examples
    )
    cand_logloss = _logloss(cand_y, cand_pred)
    cand_brier = _brier(cand_y, cand_pred)
    cand_auc = _auc(cand_y, cand_pred)
    cand_reliability = _reliability_error(cand_y, cand_pred)
    cand_monotonic = _monotonic_ok(cand_delta_g, cand_pred, cand_fam)
    cand_n = len(cand_y)
    cand_acc, cand_n_fail = _eval_failure_mode(candidate_fail, test_examples)

    base_logloss = base_brier = base_auc = base_reliability = None
    base_acc = None
    if baseline_cal:
        base_y, base_pred, base_delta_g, base_fam = _calibrator_predictions(
            baseline_cal, test_examples
        )
        base_logloss = _logloss(base_y, base_pred)
        base_brier = _brier(base_y, base_pred)
        base_auc = _auc(base_y, base_pred)
        base_reliability = _reliability_error(base_y, base_pred)
        _ = _monotonic_ok(base_delta_g, base_pred, base_fam)
    if baseline_fail:
        base_acc, _ = _eval_failure_mode(baseline_fail, test_examples)

    logloss_ok = _metric_ok_low(cand_logloss, base_logloss)
    brier_ok = _metric_ok_low(cand_brier, base_brier)
    auc_ok = _metric_ok_high(cand_auc, base_auc) and cand_auc >= MIN_AUC
    reliability_ok = _metric_ok_low(cand_reliability, base_reliability) and (
        cand_reliability <= MAX_RELIABILITY_ERROR
    )
    acc_ok = _metric_ok_high(cand_acc, base_acc)
    monotonic_ok = cand_monotonic

    passed = logloss_ok and brier_ok and auc_ok and reliability_ok and acc_ok and monotonic_ok
    report = [
        "Evaluation report (held-out split):",
        f"- Calibrator: logloss {cand_logloss:.4f} (baseline {base_logloss}), ok={logloss_ok}",
        f"- Calibrator: brier {cand_brier:.4f} (baseline {base_brier}), ok={brier_ok}",
        f"- Calibrator: auc {cand_auc:.4f} (baseline {base_auc}), ok={auc_ok}",
        f"- Calibrator: reliability {cand_reliability:.4f} (baseline {base_reliability}), ok={reliability_ok}",
        f"- Calibrator: monotonic_ok {cand_monotonic}, ok={monotonic_ok}",
        f"- Failure mode: accuracy {cand_acc:.4f} (baseline {base_acc}), ok={acc_ok}",
        f"- Samples: calibrator n={cand_n}, failure_mode n={cand_n_fail}",
        f"- Decision: {'PASS' if passed else 'FAIL'}",
    ]
    print("\n".join(report))

    if not passed:
        return 1

    artifacts_dir = Path("artifacts")
    pack_dir = artifacts_dir / f"pack_{_now_stamp()}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_paths["calibrator"], pack_dir / "calibration_module0_v1.json")
    shutil.copy2(candidate_paths["failure_mode"], pack_dir / "failure_mode_v1.json")

    latest_payload = {
        "path": str(pack_dir),
        "created_at": _now_stamp(),
        "metrics": {
            "logloss": cand_logloss,
            "brier": cand_brier,
            "auc": cand_auc,
            "reliability_error": cand_reliability,
            "monotonic_ok": cand_monotonic,
            "accuracy": cand_acc,
            "samples": {
                "calibrator": cand_n,
                "failure_mode": cand_n_fail,
            },
        },
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    Path(latest_path).write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate artifact pack and gate deployment.")
    parser.add_argument("--db", required=True, help="Path to evidence SQLite DB.")
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate artifact pack directory or manifest JSON.",
    )
    parser.add_argument(
        "--latest",
        default="artifacts/latest.json",
        help="Path to latest.json baseline pointer.",
    )
    args = parser.parse_args(argv)

    try:
        return evaluate(args.db, args.candidate, args.latest)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
