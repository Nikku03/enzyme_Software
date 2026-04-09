from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import train_hybrid_full_xtb as train_script
from audit_two_head_hard_sources import (
    _build_audit_rows_for_loader,
    _build_model_and_trainer,
    _load_checkpoint,
    _make_loader_args,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 1/Phase 2 explicit split datasets with separate exact-clean and tier-aware reporting."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--exact-val-dataset", default="")
    parser.add_argument("--exact-test-dataset", default="")
    parser.add_argument("--tiered-val-dataset", default="")
    parser.add_argument("--tiered-test-dataset", default="")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--split-mode", default="scaffold_source_size")
    parser.add_argument("--target-cyp", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/cyp3a4_phase12_eval")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload.get("drugs", payload))
    return list(payload)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_truth_sites(row: dict[str, Any], mode: str) -> list[int]:
    primary = [int(v) for v in list(row.get("primary_site_atoms", []) or [])]
    secondary = [int(v) for v in list(row.get("secondary_site_atoms", []) or [])]
    tertiary = [int(v) for v in list(row.get("tertiary_site_atoms", []) or [])]
    any_labeled = [int(v) for v in list(row.get("all_labeled_site_atoms", []) or row.get("true_site_indices", []) or [])]
    if mode == "primary_only":
        return sorted(set(primary))
    if mode == "primary_plus_secondary":
        return sorted(set(primary + secondary))
    if mode == "any_labeled_site":
        return sorted(set(any_labeled or primary + secondary + tertiary))
    raise ValueError(f"Unsupported tiered evaluation mode: {mode}")


def _summarize_tiered_rows(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {
            "total_examples": 0,
            "end_to_end_top1": 0.0,
            "end_to_end_top3": 0.0,
            "shortlist_recall_at_6": 0.0,
            "shortlist_recall_at_12": 0.0,
            "winner_acc_given_hit": 0.0,
        }
    top1 = 0
    top3 = 0
    hit6 = 0
    hit12 = 0
    hitk = 0
    correct_given_hit = 0
    for row in rows:
        truth = set(_row_truth_sites(row, mode))
        if not truth:
            continue
        predicted = row.get("winner_predicted_atom_index")
        top3_atoms = [int(v) for v in list(row.get("winner_top3_atom_indices", []) or [])]
        shortlist6 = [int(v) for v in list(row.get("shortlist_top6_candidate_indices", []) or [])]
        shortlist12 = [int(v) for v in list(row.get("shortlist_top12_candidate_indices", []) or [])]
        shortlistk = [int(v) for v in list(row.get("shortlist_selected_candidate_indices", []) or [])]
        if predicted in truth:
            top1 += 1
        if any(int(v) in truth for v in top3_atoms):
            top3 += 1
        if any(int(v) in truth for v in shortlist6):
            hit6 += 1
        if any(int(v) in truth for v in shortlist12):
            hit12 += 1
        local_hitk = any(int(v) in truth for v in shortlistk)
        if local_hitk:
            hitk += 1
            if predicted in truth:
                correct_given_hit += 1
    return {
        "total_examples": int(total),
        "end_to_end_top1": float(top1) / float(total),
        "end_to_end_top3": float(top3) / float(total),
        "shortlist_recall_at_6": float(hit6) / float(total),
        "shortlist_recall_at_12": float(hit12) / float(total),
        "winner_acc_given_hit": float(correct_given_hit) / float(max(1, hitk)),
        "shortlist_hit_count_at_train_k": int(hitk),
    }


def _filter_rows(rows: list[dict[str, Any]], regime: str) -> list[dict[str, Any]]:
    if regime == "exact_clean":
        return [
            row
            for row in rows
            if str(row.get("training_regime") or "") == "exact_clean"
            or str(row.get("label_regime") or "") in {"single_exact", "multi_exact"}
        ]
    if regime == "tiered_multisite":
        return [
            row
            for row in rows
            if str(row.get("training_regime") or "") == "tiered_multisite"
            or str(row.get("label_regime") or "") == "tiered_multisite"
        ]
    raise ValueError(f"Unsupported regime: {regime}")


def _build_loader(rows: list[dict[str, Any]], *, args: argparse.Namespace):
    if not rows:
        return None
    loader_args = argparse.Namespace(
        batch_size=max(1, int(args.batch_size) if int(args.batch_size) > 0 else 8),
        seed=42 if int(args.seed) < 0 else int(args.seed),
        structure_sdf=str(args.structure_sdf),
        manual_target_bond=args.manual_target_bond,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        xtb_cache_dir=str(Path(args.xtb_cache_dir)),
        compute_xtb_if_missing=False,
        use_candidate_mask=False,
        target_cyp=str(args.target_cyp or "CYP3A4"),
        balance_train_sources=False,
        split_mode=str(args.split_mode),
    )
    (train_loader, val_loader, _), _ = train_script._build_loaders_with_fallback(
        rows,
        rows,
        [],
        args=loader_args,
    )
    return val_loader


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = _load_checkpoint(checkpoint_path, device=train_script._resolve_device(args.device))
    seed = int(args.seed if int(args.seed) >= 0 else int(checkpoint.get("seed", 42) or 42))
    train_script._apply_reproducibility_lock(seed)
    model, winner_head, trainer, load_summary = _build_model_and_trainer(
        checkpoint,
        device=train_script._resolve_device(args.device),
    )

    train_dataset = Path(args.train_dataset).expanduser()
    val_dataset = Path(args.val_dataset).expanduser()
    test_dataset = Path(args.test_dataset).expanduser()
    train_rows = _load_rows(train_dataset)
    val_rows = _load_rows(val_dataset)
    test_rows = _load_rows(test_dataset)

    exact_val_rows = _load_rows(Path(args.exact_val_dataset).expanduser()) if str(args.exact_val_dataset).strip() else _filter_rows(val_rows, "exact_clean")
    exact_test_rows = _load_rows(Path(args.exact_test_dataset).expanduser()) if str(args.exact_test_dataset).strip() else _filter_rows(test_rows, "exact_clean")
    tiered_val_rows = _load_rows(Path(args.tiered_val_dataset).expanduser()) if str(args.tiered_val_dataset).strip() else _filter_rows(val_rows, "tiered_multisite")
    tiered_test_rows = _load_rows(Path(args.tiered_test_dataset).expanduser()) if str(args.tiered_test_dataset).strip() else _filter_rows(test_rows, "tiered_multisite")

    exact_val_loader = _build_loader(exact_val_rows, args=args)
    exact_test_loader = _build_loader(exact_test_rows, args=args)
    tiered_val_loader = _build_loader(tiered_val_rows, args=args)
    tiered_test_loader = _build_loader(tiered_test_rows, args=args)

    exact_report = {
        "val": trainer.evaluate_loader(exact_val_loader) if exact_val_loader is not None else {},
        "test": trainer.evaluate_loader(exact_test_loader) if exact_test_loader is not None else {},
    }

    tiered_val_audit_rows = (
        _build_audit_rows_for_loader(
            trainer=trainer,
            loader=tiered_val_loader,
            split_name="val",
            hard_source_names=None,
            winner_small_margin_threshold=0.12,
            shortlist_small_margin_threshold=0.05,
            near_cutoff_rank=12,
            near_true_graph_distance=2,
            attnsom_tier_lookup=None,
        )
        if tiered_val_loader is not None
        else []
    )
    tiered_test_audit_rows = (
        _build_audit_rows_for_loader(
            trainer=trainer,
            loader=tiered_test_loader,
            split_name="test",
            hard_source_names=None,
            winner_small_margin_threshold=0.12,
            shortlist_small_margin_threshold=0.05,
            near_cutoff_rank=12,
            near_true_graph_distance=2,
            attnsom_tier_lookup=None,
        )
        if tiered_test_loader is not None
        else []
    )
    tiered_rows = list(tiered_val_audit_rows) + list(tiered_test_audit_rows)
    _write_csv(output_dir / "cyp3a4_phase12_tiered_eval_rows.csv", tiered_rows)

    tiered_report = {
        "val": {
            "primary_only": _summarize_tiered_rows(tiered_val_audit_rows, mode="primary_only"),
            "primary_plus_secondary": _summarize_tiered_rows(tiered_val_audit_rows, mode="primary_plus_secondary"),
            "any_labeled_site": _summarize_tiered_rows(tiered_val_audit_rows, mode="any_labeled_site"),
        },
        "test": {
            "primary_only": _summarize_tiered_rows(tiered_test_audit_rows, mode="primary_only"),
            "primary_plus_secondary": _summarize_tiered_rows(tiered_test_audit_rows, mode="primary_plus_secondary"),
            "any_labeled_site": _summarize_tiered_rows(tiered_test_audit_rows, mode="any_labeled_site"),
        },
    }
    for split_name in ("val", "test"):
        base = tiered_report[split_name]["primary_only"]["end_to_end_top1"]
        tiered_report[split_name]["attnsom_relaxed_gain_primary_plus_secondary"] = (
            tiered_report[split_name]["primary_plus_secondary"]["end_to_end_top1"] - base
        )
        tiered_report[split_name]["attnsom_relaxed_gain_any_labeled_site"] = (
            tiered_report[split_name]["any_labeled_site"]["end_to_end_top1"] - base
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "load_summary": load_summary,
        "split_datasets": {
            "train": str(train_dataset),
            "val": str(val_dataset),
            "test": str(test_dataset),
            "exact_val": str(args.exact_val_dataset or ""),
            "exact_test": str(args.exact_test_dataset or ""),
            "tiered_val": str(args.tiered_val_dataset or ""),
            "tiered_test": str(args.tiered_test_dataset or ""),
        },
        "exact_clean_metrics": exact_report,
        "tiered_multisite_metrics": tiered_report,
        "outputs": {
            "report_json": str(output_dir / "cyp3a4_phase12_eval_report.json"),
            "tiered_rows_csv": str(output_dir / "cyp3a4_phase12_tiered_eval_rows.csv"),
        },
    }
    (output_dir / "cyp3a4_phase12_eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
