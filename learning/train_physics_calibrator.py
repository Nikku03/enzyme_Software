from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.evidence_store import load_labeled_examples
from enzyme_software.unity_schema import UnityRecord, build_features

EPS = 1e-12


def predict_proba(features: Dict[str, float], model: Dict[str, Any]) -> float:
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


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _standardize(
    features: List[List[float]],
) -> Tuple[List[List[float]], List[float], List[float]]:
    if not features:
        return [], [], []
    rows = len(features)
    cols = len(features[0])
    means = []
    scales = []
    for j in range(cols):
        col = [features[i][j] for i in range(rows)]
        mean = sum(col) / rows
        var = sum((x - mean) ** 2 for x in col) / rows
        scale = math.sqrt(var)
        if scale < EPS:
            scale = 1.0
        means.append(mean)
        scales.append(scale)
    standardized = [
        [(row[j] - means[j]) / scales[j] for j in range(cols)] for row in features
    ]
    return standardized, means, scales


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


def _train_logistic(
    x: List[List[float]],
    y: List[int],
    lr: float,
    max_iter: int,
    l2: float,
    tol: float,
    patience: int,
) -> Tuple[List[float], float, float, int]:
    if not x:
        return [], 0.0, 0.0, 0
    n_samples = len(x)
    n_features = len(x[0])
    pos_rate = sum(y) / max(1, n_samples)
    pos_rate = min(1.0 - EPS, max(EPS, pos_rate))
    bias = math.log(pos_rate / (1.0 - pos_rate))
    weights = [0.0] * n_features
    best_loss = float("inf")
    best_iter = 0
    no_improve = 0
    for step in range(max_iter):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        preds = []
        for row, target in zip(x, y):
            z = bias + sum(w * v for w, v in zip(weights, row))
            prob = _sigmoid(z)
            preds.append(prob)
            error = prob - target
            grad_b += error
            for j in range(n_features):
                grad_w[j] += error * row[j]
        grad_b /= n_samples
        grad_w = [(g / n_samples) + l2 * weights[j] for j, g in enumerate(grad_w)]
        bias -= lr * grad_b
        weights = [w - lr * g for w, g in zip(weights, grad_w)]
        loss = _logloss(y, preds)
        if loss + tol < best_loss:
            best_loss = loss
            best_iter = step + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return weights, bias, best_loss, best_iter


def _collect_examples(path: str) -> Tuple[List[UnityRecord], List[int]]:
    examples = load_labeled_examples(path)
    records: List[UnityRecord] = []
    labels: List[int] = []
    for record, payload in examples:
        outcomes = payload.get("outcomes", [])
        for outcome in outcomes:
            if "any_activity" not in outcome:
                continue
            records.append(record)
            labels.append(1 if outcome["any_activity"] else 0)
    return records, labels


def _collect_examples_from_corpus(
    corpus_path: str,
) -> Tuple[List[Dict[str, float]], List[int], Dict[str, int]]:
    features: List[Dict[str, float]] = []
    labels: List[int] = []
    sources: Dict[str, int] = {}
    with open(corpus_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            feat = payload.get("features") or {}
            label = payload.get("label")
            if label is None:
                continue
            features.append({str(k): float(v) for k, v in feat.items()})
            labels.append(1 if label else 0)
            provenance = payload.get("provenance") or {}
            source = provenance.get("source")
            if source:
                sources[source] = sources.get(source, 0) + 1
    return features, labels, sources


def _write_model(path: Path, model: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def train(
    path: str,
    out_path: str,
    lr: float,
    max_iter: int,
    l2: float,
    seed: int,
    corpus_path: Optional[str] = None,
) -> int:
    sources: Dict[str, int] = {}
    if corpus_path:
        feature_list, labels, sources = _collect_examples_from_corpus(corpus_path)
    else:
        records, labels = _collect_examples(path)
        feature_list = [build_features(record) for record in records]
    if len(labels) < 2:
        raise ValueError("Need at least 2 labeled outcomes with any_activity.")

    feature_order = sorted(feature_list[0].keys())
    x = [[float(features.get(name, 0.0)) for name in feature_order] for features in feature_list]
    x_std, mean, scale = _standardize(x)

    weights, bias, best_loss, steps = _train_logistic(
        x=x_std,
        y=labels,
        lr=lr,
        max_iter=max_iter,
        l2=l2,
        tol=1e-6,
        patience=15,
    )
    preds = []
    for row in x_std:
        z = bias + sum(w * v for w, v in zip(weights, row))
        preds.append(_sigmoid(z))
    metrics = {
        "logloss": round(_logloss(labels, preds), 6),
        "brier": round(_brier(labels, preds), 6),
        "n_samples": len(labels),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
        "sources": sources,
    }
    model = {
        "model_version": "calibration_module0_v1",
        "feature_order": feature_order,
        "weights": [round(w, 6) for w in weights],
        "bias": round(bias, 6),
        "feature_mean": [round(val, 6) for val in mean],
        "feature_scale": [round(val, 6) for val in scale],
        "training": {
            "learning_rate": lr,
            "max_iter": max_iter,
            "l2": l2,
            "seed": seed,
            "best_iter": steps,
        },
        "metrics": metrics,
    }
    _write_model(Path(out_path), model)
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train physics calibrator for Module 0.")
    parser.add_argument("--db", required=True, help="Path to evidence SQLite DB.")
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional training corpus JSONL (overrides --db labels).",
    )
    parser.add_argument(
        "--out",
        default="artifacts/calibration_module0_v1.json",
        help="Output model JSON path.",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--max-iter", type=int, default=400, help="Maximum iterations.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (reserved).")
    args = parser.parse_args(argv)

    try:
        return train(
            args.db,
            args.out,
            args.lr,
            args.max_iter,
            args.l2,
            args.seed,
            corpus_path=args.corpus,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
