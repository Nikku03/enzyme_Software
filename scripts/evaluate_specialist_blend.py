"""
evaluate_specialist_blend.py

Evaluates every combination of specialist routing on the held-out test split:

  1. individual models  (hybrid_lnn, hybrid_full_xtb, micropattern_xtb, cahml)
  2. multi-head meta-learner  (trained end-to-end)
  3. specialist blend        (lnn+xtb avg → top-1 | micropattern → top-2/3 | cahml → CYP)
  4. meta + cahml_cyp        (meta-learner sites | cahml → CYP)

All predictions are read from the pre-computed base_predictions file — no live
model inference at evaluation time (fast).

Usage:
    python scripts/evaluate_specialist_blend.py \\
        --predictions  cache/meta_learner/base_predictions_cahml_oof.pt \\
        --dataset      data/merged_all_sources.json \\
        --checkpoint   checkpoints/meta_learner_multihead/multihead_meta_learner_best.pt \\
        --train-ratio  0.7 --val-ratio 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.meta_learner import MetaLearnerDataset
from enzyme_software.meta_learner.meta_evaluator import evaluate_meta_predictions
from enzyme_software.meta_learner.multi_head_meta_model import MultiHeadMetaLearner


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_drugs(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(
        drug.get("som")
        or drug.get("site_atoms")
        or drug.get("site_atom_indices")
        or drug.get("metabolism_sites")
    )


def _resolve_device(name: Optional[str]):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _top1_from_scores(scores: torch.Tensor) -> int:
    return int(torch.argmax(scores.view(-1)).item())


def _ranking_from_scores(scores: torch.Tensor) -> List[int]:
    return torch.argsort(scores.view(-1), descending=True).tolist()


def _top1_correct(ranking: List[int], labels: torch.Tensor) -> bool:
    positives = set(int(i) for i in torch.where(labels.view(-1) > 0.5)[0].tolist())
    return bool(ranking and ranking[0] in positives)


def _top3_correct(ranking: List[int], labels: torch.Tensor) -> bool:
    positives = set(int(i) for i in torch.where(labels.view(-1) > 0.5)[0].tolist())
    return bool(any(idx in positives for idx in ranking[:3]))


def _cyp_correct(pred_idx: int, true_label: int) -> bool:
    return pred_idx == true_label


# ── specialist blend ──────────────────────────────────────────────────────────

def _specialist_ranking(
    site_scores_raw: torch.Tensor,
    model_names: List[str],
) -> Tuple[List[int], Optional[List[int]]]:
    """
    Returns (top1_ranking, top3_ranking) where:
    - top1_ranking: sorted by avg(hybrid_lnn, hybrid_full_xtb)
    - top3_ranking: top-1 from lnn+xtb, then micropattern ordering for the rest
    Both lists are full-length atom rankings.
    """
    n_models = site_scores_raw.shape[1]
    idx = {name: i for i, name in enumerate(model_names)}

    # Top-1 scores: average of lnn + full_xtb (whichever are present)
    parts = []
    for key in ("hybrid_lnn", "hybrid_full_xtb"):
        if key in idx:
            parts.append(site_scores_raw[:, idx[key]])
    top1_scores = torch.stack(parts).mean(dim=0) if parts else site_scores_raw[:, 0]
    top1_ranking = _ranking_from_scores(top1_scores)

    # Top-3 fill: use micropattern ordering, pin top-1 atom at position 0
    micro_key = "micropattern_xtb"
    if micro_key in idx:
        micro_scores = site_scores_raw[:, idx[micro_key]]
        micro_ranking = _ranking_from_scores(micro_scores)
        top1_atom = top1_ranking[0]
        fill = [a for a in micro_ranking if a != top1_atom]
        specialist_ranking = [top1_atom] + fill
    else:
        specialist_ranking = top1_ranking

    return top1_ranking, specialist_ranking


def _cahml_cyp_idx(cyp_probs_raw: torch.Tensor, model_names: List[str]) -> Optional[int]:
    """Return CAHML's predicted CYP index, or None if CAHML not in model_names."""
    idx = {name: i for i, name in enumerate(model_names)}
    if "cahml" not in idx:
        return None
    probs = cyp_probs_raw[idx["cahml"]]
    return int(torch.argmax(probs).item())


# ── per-model evaluation from site_scores_raw ────────────────────────────────

def _eval_single_model(
    site_scores_raw: torch.Tensor,
    cyp_probs_raw: torch.Tensor,
    model_idx: int,
    site_labels: torch.Tensor,
    cyp_label: int,
) -> Dict[str, float]:
    scores = site_scores_raw[:, model_idx]
    ranking = _ranking_from_scores(scores)
    cyp_idx = int(torch.argmax(cyp_probs_raw[model_idx]).item())
    return {
        "top1": float(_top1_correct(ranking, site_labels)),
        "top3": float(_top3_correct(ranking, site_labels)),
        "cyp":  float(_cyp_correct(cyp_idx, cyp_label)),
    }


# ── accumulator ──────────────────────────────────────────────────────────────

class Accumulator:
    def __init__(self, name: str):
        self.name = name
        self.top1 = self.top3 = self.cyp = self.n = 0

    def add(self, top1: bool, top3: bool, cyp: bool) -> None:
        self.top1 += int(top1)
        self.top3 += int(top3)
        self.cyp  += int(cyp)
        self.n    += 1

    def result(self) -> Dict[str, object]:
        d = max(1, self.n)
        return {
            "model": self.name,
            "n":     self.n,
            "top1":  round(self.top1 / d, 4),
            "top3":  round(self.top3 / d, 4),
            "cyp":   round(self.cyp  / d, 4),
            "top1_count": self.top1,
            "top3_count": self.top3,
            "cyp_count":  self.cyp,
        }

    def print_row(self) -> None:
        r = self.result()
        print(
            f"  {r['model']:<30}  "
            f"top1={r['top1_count']:>3}/{r['n']} ({r['top1']:.1%})  "
            f"top3={r['top3_count']:>3}/{r['n']} ({r['top3']:.1%})  "
            f"cyp={r['cyp_count']:>3}/{r['n']} ({r['cyp']:.1%})"
        )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    require_torch()

    parser = argparse.ArgumentParser(
        description="Evaluate specialist blend vs all models on the held-out test split"
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to base_predictions_cahml_oof.pt")
    parser.add_argument("--dataset", required=True,
                        help="Training dataset JSON (same file used during training)")
    parser.add_argument("--checkpoint", required=True,
                        help="Multi-head meta-learner checkpoint (.pt)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--site-labeled-only", dest="site_labeled_only",
                        action="store_true", default=True)
    parser.add_argument("--no-site-labeled-only", dest="site_labeled_only",
                        action="store_false")
    parser.add_argument("--output-json", default=None,
                        help="Optional path to save per-compound JSON results")
    args = parser.parse_args()

    device = _resolve_device(args.device)

    # ── load predictions file to get model_names ──────────────────────────────
    pred_payload = torch.load(args.predictions, map_location="cpu", weights_only=False)
    predictions_dict = pred_payload.get("predictions") or pred_payload
    model_names: List[str] = list(
        pred_payload.get("model_names") or ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"]
    )
    n_models = len(model_names)
    print(f"Model names in predictions: {model_names}")

    # ── infer feature dims from first record ─────────────────────────────────
    first = next(iter(predictions_dict.values()))
    atom_feature_dim   = int(first["atom_features"].shape[1])
    global_feature_dim = int(first["global_features"].shape[0])
    print(f"atom_feature_dim={atom_feature_dim}  global_feature_dim={global_feature_dim}")

    # ── load meta-learner ─────────────────────────────────────────────────────
    ckpt_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt_payload.get("config") or {}
    meta_model = MultiHeadMetaLearner(
        n_models=n_models,
        n_cyp=5,
        atom_feature_dim=atom_feature_dim,
        global_feature_dim=global_feature_dim,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
    )
    state_dict = ckpt_payload.get("model_state_dict") or ckpt_payload
    current = meta_model.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in current and tuple(v.shape) == tuple(current[k].shape)
    }
    skipped = len(state_dict) - len(compatible)
    if skipped:
        print(f"  [meta] skipped {skipped} incompatible tensors")
    meta_model.load_state_dict(compatible, strict=False)
    meta_model.to(device)
    meta_model.eval()
    print(f"Loaded meta-learner: {args.checkpoint}")

    # ── build test split ─────────────────────────────────────────────────────
    drugs = _load_drugs(Path(args.dataset))
    if args.site_labeled_only:
        drugs = [d for d in drugs if _has_site_labels(d)]
    random.Random(args.seed).shuffle(drugs)
    n_train = int(len(drugs) * args.train_ratio)
    n_val   = int(len(drugs) * args.val_ratio)
    test_drugs = drugs[n_train + n_val:]
    dataset = MetaLearnerDataset(args.predictions, test_drugs)
    print(f"Test set: {len(test_drugs)} drugs  →  {len(dataset)} in predictions cache")

    # ── accumulators ─────────────────────────────────────────────────────────
    acc: Dict[str, Accumulator] = {}
    for name in model_names:
        acc[name] = Accumulator(name)
    acc["meta_learner"]    = Accumulator("meta_learner")
    acc["specialist"]      = Accumulator("specialist  (lnn+xtb→1 / micro→2,3 / cahml→cyp)")
    acc["meta_cahml_cyp"]  = Accumulator("meta+cahml_cyp (meta→sites / cahml→cyp)")

    cahml_available = "cahml" in model_names
    if not cahml_available:
        print("WARNING: 'cahml' not found in model_names — specialist CYP will fall back to meta-learner CYP")

    per_compound = []

    # ── evaluation loop ───────────────────────────────────────────────────────
    for i in range(len(dataset)):
        row = dataset[i]

        # Move to device
        atom_features   = row["atom_features"].to(device)
        global_features = row["global_features"].to(device)
        site_scores_raw = row["site_scores_raw"].to(device)
        cyp_probs_raw   = row["cyp_probs_raw"].to(device)
        site_labels     = row["site_labels"]       # cpu
        cyp_label       = int(row["cyp_label"].item())
        has_sites       = bool(torch.any(site_labels > 0.5))

        if not has_sites:
            continue  # skip compounds without site labels

        # ── individual models ─────────────────────────────────────────────
        for j, mname in enumerate(model_names):
            ev = _eval_single_model(
                site_scores_raw.cpu(), cyp_probs_raw.cpu(), j, site_labels, cyp_label
            )
            acc[mname].add(ev["top1"], ev["top3"], ev["cyp"])

        # ── meta-learner ──────────────────────────────────────────────────
        with torch.no_grad():
            site_logits, cyp_logits, stats = meta_model(
                atom_features, global_features, site_scores_raw, cyp_probs_raw
            )
        meta_ranking = _ranking_from_scores(site_logits.detach().cpu())
        meta_cyp_idx = int(torch.argmax(cyp_logits.detach().cpu()).item())
        meta_ev = evaluate_meta_predictions(
            site_logits.detach().cpu(), site_labels, cyp_logits.detach().cpu(), cyp_label
        )
        acc["meta_learner"].add(meta_ev["site_top1"], meta_ev["site_top3"], meta_ev["cyp_acc"])

        # ── specialist blend ──────────────────────────────────────────────
        _, spec_ranking = _specialist_ranking(site_scores_raw.cpu(), model_names)
        cahml_cyp_idx = _cahml_cyp_idx(cyp_probs_raw.cpu(), model_names)
        spec_cyp_idx  = cahml_cyp_idx if cahml_cyp_idx is not None else meta_cyp_idx

        spec_top1 = _top1_correct(spec_ranking, site_labels)
        spec_top3 = _top3_correct(spec_ranking, site_labels)
        spec_cyp  = _cyp_correct(spec_cyp_idx, cyp_label)
        acc["specialist"].add(spec_top1, spec_top3, spec_cyp)

        # ── meta + cahml CYP ──────────────────────────────────────────────
        mc_top1 = _top1_correct(meta_ranking, site_labels)
        mc_top3 = _top3_correct(meta_ranking, site_labels)
        mc_cyp  = _cyp_correct(spec_cyp_idx, cyp_label)
        acc["meta_cahml_cyp"].add(mc_top1, mc_top3, mc_cyp)

        per_compound.append({
            "smiles": row["smiles"],
            "cyp_label": cyp_label,
            "meta":     {"top1": meta_ev["site_top1"], "top3": meta_ev["site_top3"], "cyp": meta_ev["cyp_acc"]},
            "specialist": {"top1": spec_top1, "top3": spec_top3, "cyp": spec_cyp},
            "meta_cahml_cyp": {"top1": mc_top1, "top3": mc_top3, "cyp": mc_cyp},
            **{
                f"{mname}": {
                    "top1": _eval_single_model(site_scores_raw.cpu(), cyp_probs_raw.cpu(), j, site_labels, cyp_label)["top1"],
                    "top3": _eval_single_model(site_scores_raw.cpu(), cyp_probs_raw.cpu(), j, site_labels, cyp_label)["top3"],
                    "cyp":  _eval_single_model(site_scores_raw.cpu(), cyp_probs_raw.cpu(), j, site_labels, cyp_label)["cyp"],
                }
                for j, mname in enumerate(model_names)
            },
        })

    # ── print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SPECIALIST BLEND EVALUATION — TEST SET")
    print("=" * 72)
    print(f"  {'Model':<30}  {'Top-1':>14}  {'Top-3':>14}  {'CYP':>14}")
    print("  " + "-" * 66)
    for name in model_names:
        acc[name].print_row()
    print("  " + "-" * 66)
    acc["meta_learner"].print_row()
    acc["specialist"].print_row()
    acc["meta_cahml_cyp"].print_row()
    print("=" * 72)

    # ── per-CYP breakdown ────────────────────────────────────────────────────
    cyp_classes = ["CYP1A2", "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4"]
    print()
    print("PER-CYP  (meta_learner / specialist / meta+cahml_cyp)")
    print("  " + "-" * 66)
    for cyp_idx, cyp_name in enumerate(cyp_classes):
        rows = [r for r in per_compound if r["cyp_label"] == cyp_idx]
        if not rows:
            continue
        n = len(rows)
        def _pct(key, sub):
            return sum(r[key][sub] for r in rows) / n
        print(
            f"  {cyp_name:<10}  n={n:>3}  "
            f"meta top1={_pct('meta','top1'):.1%}  "
            f"spec top1={_pct('specialist','top1'):.1%}  "
            f"mc_cyp top1={_pct('meta_cahml_cyp','top1'):.1%}  |  "
            f"meta cyp={_pct('meta','cyp'):.1%}  "
            f"cahml cyp={_pct('specialist','cyp'):.1%}"
        )

    # ── save per-compound results ─────────────────────────────────────────────
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "summary": {k: v.result() for k, v in acc.items()},
            "model_names": model_names,
            "per_compound": per_compound,
        }
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
