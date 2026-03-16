from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from enzyme_software.evidence_store import load_labeled_examples
from enzyme_software.unity_schema import UnityRecord, build_features

EPS = 1e-12


def predict_class(features: Dict[str, float], model: Dict[str, Any]) -> str:
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


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [val / total for val in exp_scores]


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


def _cross_entropy(y_true: List[int], y_pred: List[List[float]]) -> float:
    total = 0.0
    for label, probs in zip(y_true, y_pred):
        prob = probs[label] if 0 <= label < len(probs) else EPS
        prob = min(1.0 - EPS, max(EPS, prob))
        total += -math.log(prob)
    return total / max(1, len(y_true))


def _train_softmax(
    x: List[List[float]],
    y: List[int],
    n_classes: int,
    lr: float,
    max_iter: int,
    l2: float,
    tol: float,
    patience: int,
) -> Tuple[List[List[float]], List[float], float, int]:
    if not x:
        return [], [], 0.0, 0
    n_samples = len(x)
    n_features = len(x[0])
    weights = [[0.0 for _ in range(n_features)] for _ in range(n_classes)]
    bias = [0.0 for _ in range(n_classes)]
    best_loss = float("inf")
    best_iter = 0
    no_improve = 0

    for step in range(max_iter):
        grad_w = [[0.0 for _ in range(n_features)] for _ in range(n_classes)]
        grad_b = [0.0 for _ in range(n_classes)]
        preds = []

        for row, label in zip(x, y):
            scores = [
                bias[c] + sum(weights[c][j] * row[j] for j in range(n_features))
                for c in range(n_classes)
            ]
            probs = _softmax(scores)
            preds.append(probs)
            for c in range(n_classes):
                error = probs[c] - (1.0 if c == label else 0.0)
                grad_b[c] += error
                for j in range(n_features):
                    grad_w[c][j] += error * row[j]

        for c in range(n_classes):
            grad_b[c] /= n_samples
            for j in range(n_features):
                grad_w[c][j] = (grad_w[c][j] / n_samples) + l2 * weights[c][j]

        for c in range(n_classes):
            bias[c] -= lr * grad_b[c]
            for j in range(n_features):
                weights[c][j] -= lr * grad_w[c][j]

        loss = _cross_entropy(y, preds)
        if loss + tol < best_loss:
            best_loss = loss
            best_iter = step + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return weights, bias, best_loss, best_iter


def _collect_examples(path: str) -> Tuple[List[UnityRecord], List[str]]:
    examples = load_labeled_examples(path)
    records: List[UnityRecord] = []
    labels: List[str] = []
    for record, payload in examples:
        outcomes = payload.get("outcomes", [])
        for outcome in outcomes:
            failure_mode = outcome.get("failure_mode")
            if not failure_mode:
                continue
            records.append(record)
            labels.append(str(failure_mode))
    return records, labels


def _evaluate(
    y_true: List[int], y_pred: List[int], n_classes: int
) -> Dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / len(y_true)

    f1_scores = []
    for c in range(n_classes):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / n_classes if n_classes else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _write_model(path: Path, model: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def train(path: str, out_path: str, lr: float, max_iter: int, l2: float) -> int:
    records, labels = _collect_examples(path)
    if len(labels) < 2:
        raise ValueError("Need at least 2 labeled outcomes with failure_mode.")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("Need at least 2 failure_mode classes to train.")

    feature_list = [build_features(record) for record in records]
    feature_order = sorted(feature_list[0].keys())
    x = [[float(features.get(name, 0.0)) for name in feature_order] for features in feature_list]
    x_std, mean, scale = _standardize(x)

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y = [class_to_idx[label] for label in labels]

    weights, bias, loss, steps = _train_softmax(
        x=x_std,
        y=y,
        n_classes=len(classes),
        lr=lr,
        max_iter=max_iter,
        l2=l2,
        tol=1e-6,
        patience=20,
    )

    preds = []
    for row in x_std:
        scores = [
            bias[c] + sum(weights[c][j] * row[j] for j in range(len(row)))
            for c in range(len(classes))
        ]
        probs = _softmax(scores)
        preds.append(int(max(range(len(classes)), key=lambda i: probs[i])))

    metrics = _evaluate(y, preds, len(classes))
    metrics["loss"] = round(loss, 6)
    metrics["n_samples"] = len(labels)

    model = {
        "model_version": "failure_mode_v1",
        "classes": classes,
        "feature_order": feature_order,
        "weights": [[round(val, 6) for val in row] for row in weights],
        "bias": [round(val, 6) for val in bias],
        "feature_mean": [round(val, 6) for val in mean],
        "feature_scale": [round(val, 6) for val in scale],
        "training": {
            "learning_rate": lr,
            "max_iter": max_iter,
            "l2": l2,
            "best_iter": steps,
        },
        "metrics": {
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "loss": metrics["loss"],
            "n_samples": metrics["n_samples"],
        },
    }
    _write_model(Path(out_path), model)
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train failure mode classifier.")
    parser.add_argument("--db", required=True, help="Path to evidence SQLite DB.")
    parser.add_argument(
        "--out",
        default="artifacts/failure_mode_v1.json",
        help="Output model JSON path.",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--max-iter", type=int, default=500, help="Maximum iterations.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization.")
    args = parser.parse_args(argv)

    try:
        return train(args.db, args.out, args.lr, args.max_iter, args.l2)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
