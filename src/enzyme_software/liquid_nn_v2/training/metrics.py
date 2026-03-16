from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_topk_accuracy(scores, labels, batch, k: int = 1, supervision_mask=None) -> float:
    scores_np = _to_numpy(scores).reshape(-1)
    labels_np = _to_numpy(labels).reshape(-1)
    batch_np = _to_numpy(batch).reshape(-1)
    mask_np = _to_numpy(supervision_mask).reshape(-1) if supervision_mask is not None else None
    if batch_np.size == 0:
        return 0.0
    num_molecules = int(batch_np.max()) + 1
    correct = 0
    total = 0
    for mol_idx in range(num_molecules):
        mask = batch_np == mol_idx
        if mask_np is not None and not np.any(mask_np[mask] > 0.5):
            continue
        mol_scores = scores_np[mask]
        mol_labels = labels_np[mask]
        true_sites = np.where(mol_labels == 1)[0]
        if true_sites.size == 0:
            continue
        if len(mol_scores) <= k:
            topk_idx = np.arange(len(mol_scores))
        else:
            topk_idx = np.argsort(mol_scores)[-k:]
        hit = any(int(t) in set(int(v) for v in topk_idx.tolist()) for t in true_sites.tolist())
        correct += int(hit)
        total += 1
    return correct / total if total > 0 else 0.0


def compute_site_metrics_v2(scores, labels, batch, threshold: float = 0.5, supervision_mask=None) -> Dict[str, float]:
    scores_flat = _to_numpy(scores).reshape(-1)
    labels_flat = _to_numpy(labels).reshape(-1)
    batch_flat = _to_numpy(batch).reshape(-1)
    if supervision_mask is not None:
        mask_flat = _to_numpy(supervision_mask).reshape(-1) > 0.5
        scores_eval = scores_flat[mask_flat]
        labels_eval = labels_flat[mask_flat]
    else:
        mask_flat = None
        scores_eval = scores_flat
        labels_eval = labels_flat
    if labels_eval.size == 0:
        return {
            "site_precision": 0.0,
            "site_recall": 0.0,
            "site_f1": 0.0,
            "site_auc": 0.5,
            "site_top1_acc": 0.0,
            "site_top2_acc": 0.0,
            "site_top3_acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
    pred_binary = (scores_eval > threshold).astype(np.float32)
    tp = float(np.sum((pred_binary == 1) & (labels_eval == 1)))
    fp = float(np.sum((pred_binary == 1) & (labels_eval == 0)))
    fn = float(np.sum((pred_binary == 0) & (labels_eval == 1)))
    tn = float(np.sum((pred_binary == 0) & (labels_eval == 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels_eval, scores_eval))
    except Exception:
        auc = 0.5
    top1_acc = compute_topk_accuracy(scores_flat, labels_flat, batch_flat, k=1, supervision_mask=mask_flat)
    top2_acc = compute_topk_accuracy(scores_flat, labels_flat, batch_flat, k=2, supervision_mask=mask_flat)
    top3_acc = compute_topk_accuracy(scores_flat, labels_flat, batch_flat, k=3, supervision_mask=mask_flat)
    return {
        "site_precision": precision,
        "site_recall": recall,
        "site_f1": f1,
        "site_auc": auc,
        "site_top1_acc": top1_acc,
        "site_top2_acc": top2_acc,
        "site_top3_acc": top3_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_site_metrics(predictions, labels, batch=None, threshold: float = 0.5) -> Dict[str, float]:
    if batch is None:
        preds = _to_numpy(predictions).reshape(-1)
        refs = _to_numpy(labels).reshape(-1)
        pred_binary = (preds > threshold).astype(np.float32)
        tp = float(np.sum((pred_binary == 1) & (refs == 1)))
        fp = float(np.sum((pred_binary == 1) & (refs == 0)))
        fn = float(np.sum((pred_binary == 0) & (refs == 1)))
        tn = float(np.sum((pred_binary == 0) & (refs == 0)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(refs, preds))
        except Exception:
            auc = 0.5
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return compute_site_metrics_v2(predictions, labels, batch, threshold=threshold)


def compute_cyp_metrics(logits, labels) -> Dict[str, object]:
    logits_np = _to_numpy(logits)
    refs = _to_numpy(labels).reshape(-1)
    preds = np.argmax(logits_np, axis=-1)
    accuracy = float(np.mean(preds == refs)) if refs.size else 0.0
    f1_per_class: List[float] = []
    num_classes = logits_np.shape[-1] if logits_np.ndim == 2 else 0
    for cls in range(num_classes):
        tp = float(np.sum((preds == cls) & (refs == cls)))
        fp = float(np.sum((preds == cls) & (refs != cls)))
        fn = float(np.sum((preds != cls) & (refs == cls)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_per_class.append(2.0 * precision * recall / (precision + recall + 1e-8))
    macro = float(sum(f1_per_class) / len(f1_per_class)) if f1_per_class else 0.0
    return {"accuracy": accuracy, "f1_macro": macro, "f1_per_class": f1_per_class}


def analyze_tau(tau_history, tau_init, bond_classes: Iterable[str]) -> Dict[str, object]:
    tau_final = _to_numpy(tau_history[-1]).reshape(-1)
    tau_start = _to_numpy(tau_init).reshape(-1)
    corr = float(np.corrcoef(np.stack([tau_final, tau_start]))[0, 1]) if tau_final.size > 1 else 0.0
    grouped: Dict[str, List[float]] = {}
    for idx, bond_class in enumerate(bond_classes):
        grouped.setdefault(str(bond_class), []).append(float(tau_final[idx]))
    tau_by_class = {key: float(sum(values) / len(values)) for key, values in grouped.items()}
    return {
        "tau_init_correlation": corr,
        "tau_by_class": tau_by_class,
        "tau_mean": float(np.mean(tau_final)) if tau_final.size else 0.0,
        "tau_std": float(np.std(tau_final)) if tau_final.size else 0.0,
    }
