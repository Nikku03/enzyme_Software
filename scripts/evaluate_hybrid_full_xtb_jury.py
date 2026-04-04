from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import load_full_xtb_warm_start
from enzyme_software.liquid_nn_v2.training.metrics import compute_site_metrics_v2

from train_hybrid_full_xtb import (
    _build_loaders_with_fallback,
    _has_site_labels,
    _load_drugs,
    _parse_csv_tokens,
    _primary_cyp,
    split_drugs,
)


@dataclass
class GraphPrediction:
    source: str
    scores: torch.Tensor
    labels: torch.Tensor
    top1: int
    top1_score: float
    gap: float


def _load_model_from_checkpoint(checkpoint_path: Path, *, device, manual_atom_feature_dim: int, atom_input_dim: int):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_payload = dict(((payload.get("config") or {}).get("base_model") or {}))
    base_cfg = ModelConfig(**cfg_payload)
    base_model = LiquidMetabolismNetV2(base_cfg)
    model = HybridLNNModel(base_model)
    load_full_xtb_warm_start(
        model,
        checkpoint_path,
        device=device,
        new_manual_atom_dim=manual_atom_feature_dim,
        new_atom_input_dim=atom_input_dim,
    )
    model.to(device)
    model.eval()
    return model


def _prepare_batch(raw_batch, device):
    moved = {}
    for key, value in raw_batch.items():
        if isinstance(value, dict):
            moved[key] = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in value.items()}
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _graph_predictions_from_scores(batch, site_scores: torch.Tensor) -> list[GraphPrediction]:
    labels = batch["site_labels"].detach().cpu().view(-1)
    batch_idx = batch["batch"].detach().cpu().view(-1)
    supervision_mask = batch.get("site_supervision_mask")
    if supervision_mask is None:
        supervision_mask = torch.ones_like(labels, dtype=torch.float32)
    supervision_mask = supervision_mask.detach().cpu().view(-1) > 0.5
    scores = site_scores.detach().cpu().view(-1)
    metadata = list(batch.get("graph_metadata") or [])
    results: list[GraphPrediction] = []
    num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    for graph_idx in range(num_graphs):
        mol_mask = (batch_idx == graph_idx) & supervision_mask
        if not bool(mol_mask.any()):
            continue
        mol_scores = scores[mol_mask]
        mol_labels = labels[mol_mask]
        if int(mol_scores.numel()) == 0:
            continue
        ranked = torch.argsort(mol_scores, descending=True)
        top1 = int(ranked[0].item())
        top1_score = float(mol_scores[top1].item())
        gap = 0.0
        if int(ranked.numel()) > 1:
            gap = float((mol_scores[ranked[0]] - mol_scores[ranked[1]]).item())
        meta = metadata[graph_idx] if graph_idx < len(metadata) else {}
        source = str((meta or {}).get("site_source") or (meta or {}).get("source") or "unknown")
        results.append(
            GraphPrediction(
                source=source,
                scores=mol_scores.clone(),
                labels=mol_labels.clone(),
                top1=top1,
                top1_score=top1_score,
                gap=gap,
            )
        )
    return results


def _collect_split_predictions(loader, *, base_model, alt_model, device):
    base_graphs: list[GraphPrediction] = []
    alt_graphs: list[GraphPrediction] = []
    with torch.no_grad():
        for raw_batch in loader:
            if raw_batch is None:
                continue
            batch = _prepare_batch(raw_batch, device)
            base_out = base_model(batch)
            alt_out = alt_model(batch)
            base_scores = torch.sigmoid(base_out["site_logits"])
            alt_scores = torch.sigmoid(alt_out["site_logits"])
            base_graphs.extend(_graph_predictions_from_scores(batch, base_scores))
            alt_graphs.extend(_graph_predictions_from_scores(batch, alt_scores))
    if len(base_graphs) != len(alt_graphs):
        raise RuntimeError("Base and alt predictions disagree on graph count")
    return list(zip(base_graphs, alt_graphs))


def _apply_rule(
    paired_graphs: list[tuple[GraphPrediction, GraphPrediction]],
    *,
    max_base_gap: float,
    min_alt_gap: float,
    min_delta: float,
    min_alt_score: float,
):
    chosen_scores = []
    chosen_labels = []
    batch_parts = []
    source_breakdown = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "top5": 0, "alt_chosen": 0})
    offset = 0
    alt_chosen_total = 0
    for graph_idx, (base, alt) in enumerate(paired_graphs):
        choose_alt = (
            alt.top1 != base.top1
            and base.gap <= max_base_gap
            and alt.gap >= min_alt_gap
            and (alt.gap - base.gap) >= min_delta
            and alt.top1_score >= min_alt_score
        )
        chosen = alt if choose_alt else base
        alt_chosen_total += int(choose_alt)
        num_atoms = int(chosen.scores.numel())
        chosen_scores.append(chosen.scores)
        chosen_labels.append(chosen.labels)
        batch_parts.append(torch.full((num_atoms,), graph_idx, dtype=torch.long))
        row = source_breakdown[chosen.source]
        row["n"] += 1
        ranked = torch.argsort(chosen.scores, descending=True).tolist()
        true_sites = set(torch.where(chosen.labels > 0.5)[0].tolist())
        row["top1"] += int(bool(ranked[:1] and ranked[0] in true_sites))
        row["top3"] += int(bool(set(ranked[:3]).intersection(true_sites)))
        row["top5"] += int(bool(set(ranked[:5]).intersection(true_sites)))
        row["alt_chosen"] += int(choose_alt)
        offset += num_atoms
    merged_scores = torch.cat(chosen_scores, dim=0)
    merged_labels = torch.cat(chosen_labels, dim=0)
    merged_batch = torch.cat(batch_parts, dim=0)
    metrics = compute_site_metrics_v2(
        merged_scores,
        merged_labels,
        merged_batch,
        supervision_mask=torch.ones_like(merged_labels),
        ranking_mask=None,
    )
    source_report = {
        src: {
            "n": row["n"],
            "top1": (row["top1"] / row["n"]) if row["n"] else 0.0,
            "top3": (row["top3"] / row["n"]) if row["n"] else 0.0,
            "top5": (row["top5"] / row["n"]) if row["n"] else 0.0,
            "alt_chosen_fraction": (row["alt_chosen"] / row["n"]) if row["n"] else 0.0,
        }
        for src, row in sorted(source_breakdown.items())
    }
    return metrics, source_report, alt_chosen_total


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Validation-calibrated jury between two hybrid_full_xtb checkpoints")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", required=True)
    parser.add_argument("--manual-feature-cache-dir", default="")
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--split-mode", default="scaffold_source_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-cyp", default="")
    parser.add_argument("--confidence-allowlist", default="")
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--compute-xtb-if-missing", action="store_true")
    parser.add_argument("--use-candidate-mask", action="store_true")
    parser.add_argument("--balance-train-sources", action="store_true")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--alt-checkpoint", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drugs = _load_drugs(Path(args.dataset))
    if str(args.target_cyp or "").strip():
        target_cyp = str(args.target_cyp).strip()
        drugs = [drug for drug in drugs if _primary_cyp(drug) == target_cyp]
    conf = _parse_csv_tokens(args.confidence_allowlist)
    if conf:
        allowed = {token.lower() for token in conf}
        drugs = [drug for drug in drugs if str(drug.get("confidence") or "").strip().lower() in allowed]
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]

    train_drugs, val_drugs, test_drugs = split_drugs(
        drugs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        mode=args.split_mode,
    )
    loaders, manual_engine_enabled = _build_loaders_with_fallback(train_drugs, val_drugs, test_drugs, args=args)
    if isinstance(loaders, dict):
        val_loader = loaders["val"]
        test_loader = loaders["test"]
    else:
        _train_loader, val_loader, test_loader = loaders

    manual_atom_feature_dim = (32 if manual_engine_enabled else 0) + 8
    atom_input_dim = 140 + 8
    base_model = _load_model_from_checkpoint(Path(args.base_checkpoint), device=device, manual_atom_feature_dim=manual_atom_feature_dim, atom_input_dim=atom_input_dim)
    alt_model = _load_model_from_checkpoint(Path(args.alt_checkpoint), device=device, manual_atom_feature_dim=manual_atom_feature_dim, atom_input_dim=atom_input_dim)

    val_pairs = _collect_split_predictions(val_loader, base_model=base_model, alt_model=alt_model, device=device)
    test_pairs = _collect_split_predictions(test_loader, base_model=base_model, alt_model=alt_model, device=device)

    grid_max_base_gap = [0.02, 0.05, 0.08, 0.12, 0.18]
    grid_min_alt_gap = [0.03, 0.06, 0.10, 0.15]
    grid_min_delta = [0.00, 0.02, 0.05, 0.08]
    grid_min_alt_score = [0.45, 0.55, 0.65, 0.75]

    best = None
    for max_base_gap in grid_max_base_gap:
        for min_alt_gap in grid_min_alt_gap:
            for min_delta in grid_min_delta:
                for min_alt_score in grid_min_alt_score:
                    val_metrics, _val_sources, alt_chosen = _apply_rule(
                        val_pairs,
                        max_base_gap=max_base_gap,
                        min_alt_gap=min_alt_gap,
                        min_delta=min_delta,
                        min_alt_score=min_alt_score,
                    )
                    score = (
                        float(val_metrics.get("site_top1_acc_all_molecules", 0.0)),
                        float(val_metrics.get("site_top3_acc_all_molecules", 0.0)),
                        -abs(float(alt_chosen)),
                    )
                    candidate = {
                        "params": {
                            "max_base_gap": max_base_gap,
                            "min_alt_gap": min_alt_gap,
                            "min_delta": min_delta,
                            "min_alt_score": min_alt_score,
                        },
                        "val_metrics": val_metrics,
                        "score": score,
                    }
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

    if best is None:
        raise RuntimeError("No jury configuration evaluated")

    test_metrics, test_sources, alt_chosen_total = _apply_rule(test_pairs, **best["params"])
    report = {
        "dataset": str(args.dataset),
        "target_cyp": str(args.target_cyp or ""),
        "split_mode": str(args.split_mode),
        "base_checkpoint": str(args.base_checkpoint),
        "alt_checkpoint": str(args.alt_checkpoint),
        "best_rule": best["params"],
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "test_source_breakdown": test_sources,
        "test_alt_override_count": int(alt_chosen_total),
        "test_alt_override_fraction": float(alt_chosen_total) / float(max(1, len(test_pairs))),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
