from __future__ import annotations

import contextlib
import io
from typing import Dict, Iterable, List

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_binary_auc(labels, scores) -> float:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def compute_topk_accuracy(
    scores,
    labels,
    batch,
    k: int = 1,
    supervision_mask=None,
    ranking_mask=None,
    *,
    count_missing_positive_as_miss: bool = False,
) -> float:
    scores_np = _to_numpy(scores).reshape(-1)
    labels_np = _to_numpy(labels).reshape(-1)
    batch_np = _to_numpy(batch).reshape(-1)
    mask_np = _to_numpy(supervision_mask).reshape(-1) if supervision_mask is not None else None
    ranking_np = _to_numpy(ranking_mask).reshape(-1) if ranking_mask is not None else None
    if batch_np.size == 0:
        return 0.0
    num_molecules = int(batch_np.max()) + 1
    correct = 0
    total = 0
    for mol_idx in range(num_molecules):
        supervised_mol_mask = batch_np == mol_idx
        if mask_np is not None:
            supervised_mol_mask = supervised_mol_mask & (mask_np > 0.5)
        if not np.any(supervised_mol_mask):
            continue
        true_sites_full = np.where(labels_np[supervised_mol_mask] == 1)[0]
        if true_sites_full.size == 0:
            continue
        ranked_mol_mask = supervised_mol_mask
        if ranking_np is not None:
            ranked_mol_mask = ranked_mol_mask & (ranking_np > 0.5)
        mol_scores = scores_np[ranked_mol_mask]
        mol_labels = labels_np[ranked_mol_mask]
        if mol_scores.size == 0:
            if count_missing_positive_as_miss:
                total += 1
            continue
        true_sites = np.where(mol_labels == 1)[0]
        if true_sites.size == 0:
            if count_missing_positive_as_miss:
                total += 1
            continue
        if len(mol_scores) <= k:
            topk_idx = np.arange(len(mol_scores))
        else:
            topk_idx = np.argsort(mol_scores)[-k:]
        hit = any(int(t) in set(int(v) for v in topk_idx.tolist()) for t in true_sites.tolist())
        correct += int(hit)
        total += 1
    return correct / total if total > 0 else 0.0


def compute_site_metrics_v2(scores, labels, batch, threshold: float = 0.5, supervision_mask=None, ranking_mask=None) -> Dict[str, float]:
    scores_flat = _to_numpy(scores).reshape(-1)
    labels_flat = _to_numpy(labels).reshape(-1)
    batch_flat = _to_numpy(batch).reshape(-1)
    if ranking_mask is not None:
        ranking_flat = _to_numpy(ranking_mask).reshape(-1) > 0.5
    else:
        ranking_flat = None
    supervision_flat = _to_numpy(supervision_mask).reshape(-1) > 0.5 if supervision_mask is not None else None
    if supervision_mask is not None:
        mask_flat = supervision_flat.copy()
        if ranking_flat is not None:
            mask_flat = mask_flat & ranking_flat
        scores_eval = scores_flat[mask_flat]
        labels_eval = labels_flat[mask_flat]
    else:
        mask_flat = ranking_flat
        if ranking_flat is not None:
            scores_eval = scores_flat[ranking_flat]
            labels_eval = labels_flat[ranking_flat]
        else:
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
            "site_top1_acc_all_molecules": 0.0,
            "site_top2_acc_all_molecules": 0.0,
            "site_top3_acc_all_molecules": 0.0,
            "site_candidate_positive_coverage_molecules": 0.0,
            "site_candidate_positive_coverage_atoms": 0.0,
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
    auc = _safe_binary_auc(labels_eval, scores_eval)
    top1_acc = compute_topk_accuracy(scores_flat, labels_flat, batch_flat, k=1, supervision_mask=mask_flat, ranking_mask=ranking_flat)
    top2_acc = compute_topk_accuracy(scores_flat, labels_flat, batch_flat, k=2, supervision_mask=mask_flat, ranking_mask=ranking_flat)
    top3_acc = compute_topk_accuracy(scores_flat, labels_flat, batch_flat, k=3, supervision_mask=mask_flat, ranking_mask=ranking_flat)
    top1_all = compute_topk_accuracy(
        scores_flat,
        labels_flat,
        batch_flat,
        k=1,
        supervision_mask=supervision_flat,
        ranking_mask=None,
        count_missing_positive_as_miss=True,
    )
    top2_all = compute_topk_accuracy(
        scores_flat,
        labels_flat,
        batch_flat,
        k=2,
        supervision_mask=supervision_flat,
        ranking_mask=None,
        count_missing_positive_as_miss=True,
    )
    top3_all = compute_topk_accuracy(
        scores_flat,
        labels_flat,
        batch_flat,
        k=3,
        supervision_mask=supervision_flat,
        ranking_mask=None,
        count_missing_positive_as_miss=True,
    )
    candidate_molecules_total = 0
    candidate_molecules_hit = 0
    candidate_positive_total = 0
    candidate_positive_hit = 0
    num_molecules = int(batch_flat.max()) + 1 if batch_flat.size else 0
    for mol_idx in range(num_molecules):
        mol_mask = batch_flat == mol_idx
        if supervision_flat is not None:
            mol_mask = mol_mask & supervision_flat
        if not np.any(mol_mask):
            continue
        mol_labels = labels_flat[mol_mask]
        positive_mask = mol_labels == 1
        if not np.any(positive_mask):
            continue
        candidate_molecules_total += 1
        if ranking_flat is None:
            candidate_molecules_hit += 1
            candidate_positive_total += int(np.sum(positive_mask))
            candidate_positive_hit += int(np.sum(positive_mask))
            continue
        mol_ranking = ranking_flat[mol_mask]
        candidate_positive_total += int(np.sum(positive_mask))
        candidate_positive_hit += int(np.sum(positive_mask & mol_ranking))
        if bool(np.any(positive_mask & mol_ranking)):
            candidate_molecules_hit += 1
    return {
        "site_precision": precision,
        "site_recall": recall,
        "site_f1": f1,
        "site_auc": auc,
        "site_top1_acc": top1_acc,
        "site_top2_acc": top2_acc,
        "site_top3_acc": top3_acc,
        "site_top1_acc_all_molecules": top1_all,
        "site_top2_acc_all_molecules": top2_all,
        "site_top3_acc_all_molecules": top3_all,
        "site_candidate_positive_coverage_molecules": (
            float(candidate_molecules_hit) / float(candidate_molecules_total) if candidate_molecules_total > 0 else 0.0
        ),
        "site_candidate_positive_coverage_atoms": (
            float(candidate_positive_hit) / float(candidate_positive_total) if candidate_positive_total > 0 else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_reranker_metrics(
    final_scores,
    proposal_scores,
    labels,
    batch,
    proposal_mask,
    supervision_mask=None,
) -> Dict[str, float]:
    final_np = _to_numpy(final_scores).reshape(-1)
    proposal_np = _to_numpy(proposal_scores).reshape(-1)
    labels_np = _to_numpy(labels).reshape(-1)
    batch_np = _to_numpy(batch).reshape(-1)
    proposal_mask_np = _to_numpy(proposal_mask).reshape(-1) > 0.5
    supervision_np = _to_numpy(supervision_mask).reshape(-1) > 0.5 if supervision_mask is not None else None
    if batch_np.size == 0:
        return {}
    corrected = 0
    harmed = 0
    unchanged = 0
    proposal_hit_molecules = 0
    proposal_positive_total = 0
    proposal_positive_hit = 0
    true_deltas = []
    false_deltas = []
    total = 0
    num_molecules = int(batch_np.max()) + 1
    for mol_idx in range(num_molecules):
        mol_mask = batch_np == mol_idx
        if supervision_np is not None:
            mol_mask = mol_mask & supervision_np
        if not np.any(mol_mask):
            continue
        mol_labels = labels_np[mol_mask]
        if not np.any(mol_labels == 1):
            continue
        total += 1
        mol_proposal_mask = proposal_mask_np[mol_mask]
        if np.any((mol_labels == 1) & mol_proposal_mask):
            proposal_hit_molecules += 1
        proposal_positive_total += int(np.sum(mol_labels == 1))
        proposal_positive_hit += int(np.sum((mol_labels == 1) & mol_proposal_mask))
        mol_final = final_np[mol_mask]
        mol_proposal = proposal_np[mol_mask]
        proposal_top = int(np.argmax(mol_proposal))
        final_top = int(np.argmax(mol_final))
        proposal_hit = bool(mol_labels[proposal_top] == 1)
        final_hit = bool(mol_labels[final_top] == 1)
        if (not proposal_hit) and final_hit:
            corrected += 1
        elif proposal_hit and (not final_hit):
            harmed += 1
        else:
            unchanged += 1
        if np.any(mol_proposal_mask):
            true_mask = (mol_labels == 1) & mol_proposal_mask
            false_mask = (mol_labels == 0) & mol_proposal_mask
            if np.any(true_mask):
                true_deltas.extend((mol_final[true_mask] - mol_proposal[true_mask]).tolist())
            if np.any(false_mask):
                false_deltas.extend((mol_final[false_mask] - mol_proposal[false_mask]).tolist())
    if total == 0:
        return {}
    return {
        "proposal_molecule_recall_at_k": float(proposal_hit_molecules) / float(total),
        "proposal_recall_at_k": float(proposal_positive_hit) / float(proposal_positive_total) if proposal_positive_total > 0 else 0.0,
        "reranker_corrected_count": float(corrected),
        "reranker_harmed_count": float(harmed),
        "reranker_top1_unchanged_count": float(unchanged),
        "reranker_corrected_fraction": float(corrected) / float(total),
        "reranker_harmed_fraction": float(harmed) / float(total),
        "reranker_top1_unchanged_fraction": float(unchanged) / float(total),
        "reranker_delta_true_mean": float(np.mean(true_deltas)) if true_deltas else 0.0,
        "reranker_delta_false_mean": float(np.mean(false_deltas)) if false_deltas else 0.0,
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
        auc = _safe_binary_auc(refs, preds)
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


def compute_cyp_metrics(logits, labels, supervision_mask=None) -> Dict[str, object]:
    logits_np = _to_numpy(logits)
    refs = _to_numpy(labels).reshape(-1)
    if supervision_mask is not None:
        mask = _to_numpy(supervision_mask).reshape(-1) > 0.5
        logits_np = logits_np[mask]
        refs = refs[mask]
    if refs.size == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_per_class": []}
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
