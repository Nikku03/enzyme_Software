from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from enzyme_software.liquid_nn_v2.training.metrics import compute_cyp_metrics, compute_site_metrics_v2

from train_hybrid_full_xtb import (
    _build_loaders_with_fallback,
    _load_drugs,
    _parse_csv_tokens,
    _primary_cyp,
    _has_site_labels,
    split_drugs,
)


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
    return model, base_cfg


def _source_breakdown(scores, labels, batch_idx, metadata, supervision_mask):
    by_source = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "top5": 0})
    scores = scores.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1)
    batch_idx = batch_idx.detach().cpu().view(-1)
    mask = supervision_mask.detach().cpu().view(-1) > 0.5 if supervision_mask is not None else torch.ones_like(labels, dtype=torch.bool)
    num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    for graph_idx in range(num_graphs):
        mol_mask = (batch_idx == graph_idx) & mask
        if not bool(mol_mask.any()):
            continue
        mol_scores = scores[mol_mask]
        mol_labels = labels[mol_mask]
        true_sites = set(torch.where(mol_labels > 0.5)[0].tolist())
        if not true_sites:
            continue
        ranked = torch.argsort(mol_scores, descending=True).tolist()
        src = str((metadata[graph_idx] or {}).get("source", ""))
        row = by_source[src]
        row["n"] += 1
        row["top1"] += int(bool(ranked[:1] and ranked[0] in true_sites))
        row["top3"] += int(bool(set(ranked[:3]).intersection(true_sites)))
        row["top5"] += int(bool(set(ranked[:5]).intersection(true_sites)))
    return {
        src: {
            "n": row["n"],
            "top1": (row["top1"] / row["n"]) if row["n"] else 0.0,
            "top3": (row["top3"] / row["n"]) if row["n"] else 0.0,
            "top5": (row["top5"] / row["n"]) if row["n"] else 0.0,
        }
        for src, row in sorted(by_source.items())
    }


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of hybrid_full_xtb checkpoints")
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
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, default=[])
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
        test_loader = loaders["test"]
    else:
        _train_loader, _val_loader, test_loader = loaders

    manual_atom_feature_dim = (32 if manual_engine_enabled else 0) + 8
    atom_input_dim = 140 + 8

    ckpts = [Path(p) for p in args.checkpoint]
    weights = [float(v) for v in args.weight] if args.weight else [1.0] * len(ckpts)
    if len(weights) != len(ckpts):
        raise ValueError("Number of --weight entries must match number of --checkpoint entries")
    total_w = sum(weights)
    weights = [v / total_w for v in weights]

    models = []
    configs = []
    for ckpt in ckpts:
        model, cfg = _load_model_from_checkpoint(
            ckpt,
            device=device,
            manual_atom_feature_dim=manual_atom_feature_dim,
            atom_input_dim=atom_input_dim,
        )
        models.append(model)
        configs.append(cfg)

    site_logits_all = []
    site_labels_all = []
    batch_all = []
    supervision_all = []
    cyp_logits_all = []
    cyp_labels_all = []
    cyp_mask_all = []
    batch_offset = 0
    source_counts = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "top5": 0})

    with torch.no_grad():
        for raw_batch in test_loader:
            if raw_batch is None:
                continue
            batch = raw_batch
            moved = {}
            for key, value in batch.items():
                if isinstance(value, dict):
                    moved[key] = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in value.items()}
                elif hasattr(value, "to"):
                    moved[key] = value.to(device)
                else:
                    moved[key] = value
            batch = moved

            site_logits = None
            cyp_logits = None
            for model, weight in zip(models, weights):
                outputs = model(batch)
                sl = outputs["site_logits"]
                cl = outputs["cyp_logits"]
                site_logits = (weight * sl) if site_logits is None else (site_logits + weight * sl)
                cyp_logits = (weight * cl) if cyp_logits is None else (cyp_logits + weight * cl)

            site_scores = torch.sigmoid(site_logits)
            site_logits_all.append(site_scores.detach().cpu())
            site_labels_all.append(batch["site_labels"].detach().cpu())
            supervision_all.append(batch.get("site_supervision_mask", torch.ones_like(batch["site_labels"])).detach().cpu())
            batch_all.append((batch["batch"].detach().cpu() + batch_offset))
            cyp_logits_all.append(cyp_logits.detach().cpu())
            cyp_labels_all.append(batch["cyp_labels"].detach().cpu())
            cyp_mask_all.append(batch.get("cyp_supervision_mask", torch.ones_like(batch["cyp_labels"])).detach().cpu())
            batch_breakdown = _source_breakdown(
                site_scores,
                batch["site_labels"],
                batch["batch"],
                batch.get("graph_metadata") or [],
                batch.get("site_supervision_mask"),
            )
            for src, row in batch_breakdown.items():
                source_counts[src]["n"] += int(row["n"])
                source_counts[src]["top1"] += float(row["top1"]) * int(row["n"])
                source_counts[src]["top3"] += float(row["top3"]) * int(row["n"])
                source_counts[src]["top5"] += float(row["top5"]) * int(row["n"])
            batch_offset += int(batch["cyp_labels"].shape[0])

    merged_scores = torch.cat(site_logits_all, dim=0)
    merged_labels = torch.cat(site_labels_all, dim=0)
    merged_batch = torch.cat(batch_all, dim=0)
    merged_supervision = torch.cat(supervision_all, dim=0)
    merged_cyp_logits = torch.cat(cyp_logits_all, dim=0)
    merged_cyp_labels = torch.cat(cyp_labels_all, dim=0)
    merged_cyp_mask = torch.cat(cyp_mask_all, dim=0)

    site_metrics = compute_site_metrics_v2(
        merged_scores,
        merged_labels,
        merged_batch,
        supervision_mask=merged_supervision,
        ranking_mask=None,
    )
    cyp_metrics = compute_cyp_metrics(
        merged_cyp_logits,
        merged_cyp_labels,
        supervision_mask=merged_cyp_mask,
    )
    source_breakdown = {
        src: {
            "n": row["n"],
            "top1": (row["top1"] / row["n"]) if row["n"] else 0.0,
            "top3": (row["top3"] / row["n"]) if row["n"] else 0.0,
            "top5": (row["top5"] / row["n"]) if row["n"] else 0.0,
        }
        for src, row in sorted(source_counts.items())
    }
    report = {
        "checkpoints": [str(p) for p in ckpts],
        "weights": weights,
        "dataset": str(args.dataset),
        "target_cyp": str(args.target_cyp or ""),
        "split_mode": str(args.split_mode),
        "test_metrics": {**site_metrics, **cyp_metrics},
        "source_breakdown": source_breakdown,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
