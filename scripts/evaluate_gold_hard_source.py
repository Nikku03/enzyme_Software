from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

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
    _load_split_drugs,
    _make_loader_args,
    _normalize_source,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rebuild-family checkpoints on raw and gold hard-source subsets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/prepared_training/main8_cyp3a4_augmented.json")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--split-mode", default="", choices=("", "random", "scaffold_source", "scaffold_source_size"))
    parser.add_argument("--target-cyp", default="")
    parser.add_argument("--hard-sources", default="attnsom,cyp_dbs_external")
    parser.add_argument("--gold-slice-path", default="")
    parser.add_argument("--gold-policy", default="keep_all")
    parser.add_argument("--winner-top1-prob-threshold", type=float, default=-1.0)
    parser.add_argument("--winner-prob-gap-threshold", type=float, default=-1.0)
    parser.add_argument("--winner-candidate-k", type=int, default=-1)
    parser.add_argument("--output-dir", default="artifacts/gold_hard_source_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--winner-small-margin-threshold", type=float, default=0.12)
    parser.add_argument("--shortlist-small-margin-threshold", type=float, default=0.05)
    parser.add_argument("--near-cutoff-rank", type=int, default=12)
    parser.add_argument("--near-true-graph-distance", type=int, default=2)
    return parser.parse_args()


def _parse_json_cell(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    if text[:1] in {"[", "{", "\""}:
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_gold_slice(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold slice file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("split", "")).strip().lower(),
            str(row.get("source", "")).strip().lower(),
            _parse_int(row.get("molecule_key", 0), 0),
        )
        curated_true_site_indices = _parse_json_cell(row.get("curated_true_site_indices", ""))
        if isinstance(curated_true_site_indices, str):
            curated_true_site_indices = _parse_json_cell(curated_true_site_indices)
        mapping[key] = {
            "gold_include": _parse_bool(row.get("gold_include", False)),
            "gold_policy": str(row.get("gold_policy", "") or ""),
            "gold_exclusion_reason": str(row.get("gold_exclusion_reason", "") or ""),
            "curated_true_site_indices": curated_true_site_indices if isinstance(curated_true_site_indices, list) else [],
            "curated_label_available": _parse_bool(row.get("curated_label_available", False)),
            "curated_confidence_label": str(row.get("curated_confidence_label", "") or ""),
        }
    return mapping


def _winner_top1_prob(row: dict[str, Any]) -> float | None:
    values = row.get("winner_probabilities_all")
    if not isinstance(values, list) or not values:
        return None
    try:
        probs = [float(v) for v in values]
    except Exception:
        return None
    return max(probs) if probs else None


def _row_truth_sites(row: dict[str, Any], gold_entry: dict[str, Any] | None) -> list[int]:
    curated = list((gold_entry or {}).get("curated_true_site_indices") or [])
    if curated:
        return [int(v) for v in curated]
    return [int(v) for v in list(row.get("true_site_indices") or [])]


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("split", "")).strip().lower(),
        str(row.get("source", "")).strip().lower(),
        _parse_int(row.get("molecule_key", 0), 0),
    )


def _summarize_subset(
    rows: list[dict[str, Any]],
    *,
    gold_map: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    include_per_source: bool = True,
) -> dict[str, Any]:
    total = len(rows)
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct_top1_count = 0
    shortlist_hit_at_6_count = 0
    shortlist_hit_at_12_count = 0
    shortlist_hit_at_train_k_count = 0
    end_to_end_top3_count = 0
    winner_eval_count = 0
    winner_correct_count = 0
    rescued_by_12_count = 0
    winner_miss_among_rescued_by_12_count = 0
    for row in rows:
        source_rows[str(row.get("source", "unknown"))].append(row)
        gold_entry = (gold_map or {}).get(_row_key(row))
        truth_sites = set(_row_truth_sites(row, gold_entry))
        top6 = set(int(v) for v in list(row.get("shortlist_top6_candidate_indices") or []))
        top12 = set(int(v) for v in list(row.get("shortlist_top12_candidate_indices") or []))
        selected = set(int(v) for v in list(row.get("shortlist_selected_candidate_indices") or []))
        winner_top3 = set(int(v) for v in list(row.get("winner_top3_atom_indices") or []))
        predicted_atom = row.get("winner_predicted_atom_index")
        predicted_atom = None if predicted_atom in {None, ""} else int(predicted_atom)
        hit_at_6 = bool(truth_sites & top6)
        hit_at_12 = bool(truth_sites & top12)
        hit_at_train_k = bool(truth_sites & selected)
        top1_correct = bool(predicted_atom is not None and predicted_atom in truth_sites)
        top3_correct = bool(truth_sites & winner_top3)
        rescued_by_12 = bool(hit_at_12 and not hit_at_6)

        correct_top1_count += int(top1_correct)
        shortlist_hit_at_6_count += int(hit_at_6)
        shortlist_hit_at_12_count += int(hit_at_12)
        shortlist_hit_at_train_k_count += int(hit_at_train_k)
        end_to_end_top3_count += int(top3_correct)
        rescued_by_12_count += int(rescued_by_12)
        if hit_at_train_k:
            winner_eval_count += 1
            winner_correct_count += int(top1_correct)
            if rescued_by_12 and not top1_correct:
                winner_miss_among_rescued_by_12_count += 1

    per_source = {}
    if include_per_source and total > 0:
        for source, subset in sorted(source_rows.items()):
            if subset:
                source_summary = _summarize_subset(subset, gold_map=gold_map, include_per_source=False)
                source_summary.pop("per_source", None)
                per_source[source] = source_summary

    return {
        "total_examples": int(total),
        "shortlist_recall_at_6": float(shortlist_hit_at_6_count) / float(total) if total > 0 else 0.0,
        "shortlist_recall_at_12": float(shortlist_hit_at_12_count) / float(total) if total > 0 else 0.0,
        "shortlist_hit_fraction_at_train_k": float(shortlist_hit_at_train_k_count) / float(total) if total > 0 else 0.0,
        "winner_acc_given_hit": float(winner_correct_count) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
        "winner_acc_given_hit_at_k": float(winner_correct_count) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
        "winner_eval_molecule_count": int(winner_eval_count),
        "end_to_end_top1": float(correct_top1_count) / float(total) if total > 0 else 0.0,
        "end_to_end_top3": float(end_to_end_top3_count) / float(total) if total > 0 else 0.0,
        "shortlist_rescued_by_12_count": int(rescued_by_12_count),
        "shortlist_rescued_by_12_fraction": float(rescued_by_12_count) / float(total) if total > 0 else 0.0,
        "winner_miss_among_rescued_by_12_count": int(winner_miss_among_rescued_by_12_count),
        "per_source": per_source,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    normalized_rows = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                normalized[key] = json.dumps(value, sort_keys=True)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = train_script._resolve_device(args.device)
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    seed = int(args.seed if int(args.seed) >= 0 else int(checkpoint.get("seed", 42) or 42))
    train_script._apply_reproducibility_lock(seed)
    model, winner_head, trainer, load_summary = _build_model_and_trainer(checkpoint, device=device)
    if int(args.winner_candidate_k) > 0:
        trainer.frozen_shortlist_topk = int(args.winner_candidate_k)

    loader_args = _make_loader_args(args, checkpoint)
    train_drugs, val_drugs, test_drugs = _load_split_drugs(args, loader_args)
    (train_loader, val_loader, test_loader), manual_engine_enabled = train_script._build_loaders_with_fallback(
        train_drugs,
        val_drugs,
        test_drugs,
        args=loader_args,
    )

    hard_source_names = sorted(
        {_normalize_source(token) for token in str(args.hard_sources or "").split(",") if str(token).strip()}
    )
    if not hard_source_names:
        raise ValueError("No hard sources configured")

    gold_map = None
    gold_slice_path = Path(args.gold_slice_path).expanduser() if str(args.gold_slice_path).strip() else None
    if gold_slice_path is not None:
        gold_map = _load_gold_slice(gold_slice_path)

    trainer.model.eval()
    trainer.winner_head.eval()
    val_metrics = trainer.evaluate_loader(val_loader)
    test_metrics = trainer.evaluate_loader(test_loader)

    val_rows = _build_audit_rows_for_loader(
        trainer=trainer,
        loader=val_loader,
        split_name="val",
        hard_source_names=set(hard_source_names),
        winner_small_margin_threshold=float(args.winner_small_margin_threshold),
        shortlist_small_margin_threshold=float(args.shortlist_small_margin_threshold),
        near_cutoff_rank=int(args.near_cutoff_rank),
        near_true_graph_distance=int(args.near_true_graph_distance),
    )
    test_rows = _build_audit_rows_for_loader(
        trainer=trainer,
        loader=test_loader,
        split_name="test",
        hard_source_names=set(hard_source_names),
        winner_small_margin_threshold=float(args.winner_small_margin_threshold),
        shortlist_small_margin_threshold=float(args.shortlist_small_margin_threshold),
        near_cutoff_rank=int(args.near_cutoff_rank),
        near_true_graph_distance=int(args.near_true_graph_distance),
    )
    all_rows = sorted(val_rows + test_rows, key=lambda row: (str(row["split"]), str(row["source"]), int(row["molecule_key"])))

    raw_hard_summary = {
        "combined": _summarize_subset(all_rows),
        "val": _summarize_subset(val_rows),
        "test": _summarize_subset(test_rows),
    }

    gold_rows = []
    if gold_map is not None:
        for row in all_rows:
            gold_entry = gold_map.get(_row_key(row))
            if gold_entry and bool(gold_entry.get("gold_include", False)):
                merged = dict(row)
                merged["gold_entry"] = gold_entry
                gold_rows.append(merged)
    else:
        gold_rows = list(all_rows)

    gold_summary = {
        "combined": _summarize_subset(gold_rows, gold_map=gold_map or {}),
        "val": _summarize_subset([row for row in gold_rows if str(row.get("split")) == "val"], gold_map=gold_map or {}),
        "test": _summarize_subset([row for row in gold_rows if str(row.get("split")) == "test"], gold_map=gold_map or {}),
    }

    high_conf_rows = []
    if float(args.winner_top1_prob_threshold) >= 0.0 or float(args.winner_prob_gap_threshold) >= 0.0:
        for row in gold_rows:
            top1_prob = _winner_top1_prob(row)
            prob_gap = row.get("winner_prob_gap")
            try:
                prob_gap = None if prob_gap in {None, ""} else float(prob_gap)
            except Exception:
                prob_gap = None
            meets_top1 = float(args.winner_top1_prob_threshold) < 0.0 or (
                top1_prob is not None and float(top1_prob) >= float(args.winner_top1_prob_threshold)
            )
            meets_gap = float(args.winner_prob_gap_threshold) < 0.0 or (
                prob_gap is not None and float(prob_gap) >= float(args.winner_prob_gap_threshold)
            )
            if meets_top1 and meets_gap:
                high_conf_rows.append(row)
    high_conf_summary = {
        "enabled": bool(float(args.winner_top1_prob_threshold) >= 0.0 or float(args.winner_prob_gap_threshold) >= 0.0),
        "thresholds": {
            "winner_top1_prob_threshold": float(args.winner_top1_prob_threshold),
            "winner_prob_gap_threshold": float(args.winner_prob_gap_threshold),
        },
        "coverage": float(len(high_conf_rows)) / float(len(gold_rows)) if gold_rows else 0.0,
        "combined": _summarize_subset(high_conf_rows, gold_map=gold_map or {}),
        "val": _summarize_subset([row for row in high_conf_rows if str(row.get("split")) == "val"], gold_map=gold_map or {}),
        "test": _summarize_subset([row for row in high_conf_rows if str(row.get("split")) == "test"], gold_map=gold_map or {}),
    }

    rows_with_gold_flags = []
    for row in all_rows:
        gold_entry = (gold_map or {}).get(_row_key(row), {})
        merged = dict(row)
        merged["gold_include"] = bool(gold_entry.get("gold_include", False))
        merged["gold_exclusion_reason"] = str(gold_entry.get("gold_exclusion_reason", "") or "")
        merged["curated_true_site_indices"] = list(gold_entry.get("curated_true_site_indices") or [])
        rows_with_gold_flags.append(merged)

    eval_rows_path = output_dir / "gold_hard_source_eval_rows.csv"
    report_path = output_dir / "gold_hard_source_eval_report.json"
    _write_csv(eval_rows_path, rows_with_gold_flags)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "branch_name": str(load_summary.get("branch_name", "")),
        "load_summary": load_summary,
        "seed": int(seed),
        "split_mode": str(loader_args.split_mode),
        "target_cyp": str(loader_args.target_cyp),
        "batch_size": int(loader_args.batch_size),
        "manual_engine_enabled": bool(manual_engine_enabled),
        "winner_candidate_k": int(getattr(trainer, "frozen_shortlist_topk", 6)),
        "hard_source_names": list(hard_source_names),
        "gold_slice_path": str(gold_slice_path) if gold_slice_path is not None else "",
        "gold_policy": str(args.gold_policy),
        "raw_benchmark": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "raw_hard_source_subset": raw_hard_summary,
        "gold_hard_source_subset": gold_summary,
        "high_confidence_subset": high_conf_summary,
        "output_rows_csv": str(eval_rows_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "Gold hard-source eval complete | "
        f"winner_k={int(getattr(trainer, 'frozen_shortlist_topk', 6))} | "
        f"raw_hard_top1={raw_hard_summary['combined']['end_to_end_top1']:.4f} | "
        f"gold_top1={gold_summary['combined']['end_to_end_top1']:.4f}",
        flush=True,
    )
    if high_conf_summary["enabled"]:
        print(
            "High-confidence subset | "
            f"coverage={high_conf_summary['coverage']:.4f} | "
            f"top1={high_conf_summary['combined']['end_to_end_top1']:.4f}",
            flush=True,
        )
    print(f"Gold eval rows: {eval_rows_path}", flush=True)
    print(f"Gold eval report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
