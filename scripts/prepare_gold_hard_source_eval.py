from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a cleaned gold hard-source eval slice from audit artifacts.")
    parser.add_argument("--audit-csv", required=True, help="Path to hard_source_audit_val_test.csv")
    parser.add_argument("--output-dir", default="artifacts/hard_source_audit")
    parser.add_argument(
        "--gold-policy",
        default="exclude_ambiguous",
        choices=("exclude_ambiguous", "keep_all", "use_curated_labels"),
    )
    parser.add_argument(
        "--curated-labels-path",
        default="",
        help="Optional CSV/JSON/JSONL with split/source/molecule_key join keys plus include/curated_true_site_indices overrides.",
    )
    parser.add_argument(
        "--exclude-review-reasons",
        default="",
        help="Comma-separated review reasons that should cause exclusion under exclude_ambiguous. Empty means exclude any manual-review case.",
    )
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
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_audit_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = dict(row)
            for key in (
                "true_site_indices",
                "shortlist_top6_candidate_indices",
                "shortlist_top12_candidate_indices",
                "shortlist_selected_candidate_indices",
                "winner_top3_atom_indices",
                "winner_top3_logits",
                "winner_top3_probs",
                "winner_probabilities_all",
                "manual_review_reasons",
            ):
                normalized[key] = _parse_json_cell(normalized.get(key, ""))
            for key in (
                "needs_manual_review",
                "winner_correct",
                "shortlist_fail",
                "winner_fail",
                "correct_top1",
                "is_multi_site",
                "multi_positive_case",
                "top_candidate_is_true",
                "top_candidate_is_near_true",
                "winner_margin_small",
                "shortlist_margin_small",
                "true_rank_7_to_12",
            ):
                normalized[key] = _parse_bool(normalized.get(key, ""))
            normalized["molecule_key"] = _parse_int(normalized.get("molecule_key", 0), 0)
            normalized["manual_review_priority"] = _parse_int(normalized.get("manual_review_priority", 0), 0)
            rows.append(normalized)
    return rows


def _load_curated_entries(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Curated labels file not found: {path}")
    suffix = path.suffix.lower()
    entries: list[dict[str, Any]]
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            entries = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else list(payload.get("entries", []))
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for entry in entries:
        key = (
            str(entry.get("split", "")).strip().lower(),
            str(entry.get("source", "")).strip().lower(),
            _parse_int(entry.get("molecule_key", 0), 0),
        )
        curated_true_site_indices = entry.get("curated_true_site_indices")
        if isinstance(curated_true_site_indices, str):
            curated_true_site_indices = _parse_json_cell(curated_true_site_indices)
        result[key] = {
            "include": entry.get("include"),
            "curated_true_site_indices": curated_true_site_indices,
            "notes": str(entry.get("notes", "") or ""),
            "confidence_label": str(entry.get("confidence_label", "") or ""),
        }
    return result


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
    audit_csv = Path(args.audit_csv).expanduser()
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_audit_rows(audit_csv)
    curated_entries = {}
    curated_path = Path(args.curated_labels_path).expanduser() if str(args.curated_labels_path).strip() else None
    if curated_path is not None:
        curated_entries = _load_curated_entries(curated_path)
    if args.gold_policy == "use_curated_labels" and not curated_entries:
        raise ValueError("gold_policy=use_curated_labels requires --curated-labels-path")

    excluded_reason_filter = {
        token.strip()
        for token in str(args.exclude_review_reasons or "").split(",")
        if token.strip()
    }

    prepared_rows: list[dict[str, Any]] = []
    overall_counter = Counter()
    per_source_counter: dict[str, Counter] = defaultdict(Counter)
    per_split_counter: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        key = (
            str(row.get("split", "")).strip().lower(),
            str(row.get("source", "")).strip().lower(),
            _parse_int(row.get("molecule_key", 0), 0),
        )
        curated = dict(curated_entries.get(key) or {})
        review_reasons = [str(reason) for reason in list(row.get("manual_review_reasons") or [])]
        exclude_due_to_review = bool(row.get("needs_manual_review", False))
        if excluded_reason_filter:
            exclude_due_to_review = any(reason in excluded_reason_filter for reason in review_reasons)

        include = True
        exclusion_reason = ""
        if args.gold_policy == "exclude_ambiguous":
            include = not exclude_due_to_review
            if not include:
                exclusion_reason = "manual_review_flagged"
        elif args.gold_policy == "keep_all":
            include = True
        elif args.gold_policy == "use_curated_labels":
            include = bool(curated) and str(curated.get("include", "")).strip() not in {"", "None"}
            include = _parse_bool(curated.get("include", False))
            if not include:
                exclusion_reason = "missing_or_excluded_curated_label"

        if curated:
            if str(curated.get("include", "")).strip():
                include = _parse_bool(curated.get("include"))
                if not include:
                    exclusion_reason = "curated_exclude"
            if include and not exclusion_reason and curated.get("curated_true_site_indices") not in {None, ""}:
                exclusion_reason = ""

        curated_true_site_indices = curated.get("curated_true_site_indices")
        if isinstance(curated_true_site_indices, str):
            curated_true_site_indices = _parse_json_cell(curated_true_site_indices)

        prepared = dict(row)
        prepared.update(
            {
                "gold_include": bool(include),
                "gold_policy": str(args.gold_policy),
                "gold_exclusion_reason": str(exclusion_reason),
                "curated_label_available": bool(curated),
                "curated_true_site_indices": curated_true_site_indices if curated_true_site_indices is not None else [],
                "curated_notes": str(curated.get("notes", "") or ""),
                "curated_confidence_label": str(curated.get("confidence_label", "") or ""),
                "gold_high_confidence_candidate": bool(include and not bool(row.get("needs_manual_review", False))),
            }
        )
        prepared_rows.append(prepared)

        overall_counter["total_rows"] += 1
        overall_counter["included_rows"] += int(bool(include))
        overall_counter["excluded_rows"] += int(not include)
        overall_counter["curated_rows"] += int(bool(curated))
        overall_counter["manual_review_rows"] += int(bool(row.get("needs_manual_review", False)))
        overall_counter["multi_site_rows"] += int(bool(row.get("is_multi_site", False)))
        split_name = str(row.get("split", "unknown"))
        source = str(row.get("source", "unknown"))
        per_source_counter[source]["total_rows"] += 1
        per_source_counter[source]["included_rows"] += int(bool(include))
        per_split_counter[split_name]["total_rows"] += 1
        per_split_counter[split_name]["included_rows"] += int(bool(include))

    gold_eval_rows = [row for row in prepared_rows if bool(row.get("gold_include", False))]
    high_confidence_rows = [row for row in gold_eval_rows if bool(row.get("gold_high_confidence_candidate", False))]

    gold_csv = output_dir / "gold_hard_source_eval_slice.csv"
    summary_json = output_dir / "gold_hard_source_eval_summary.json"
    _write_csv(gold_csv, prepared_rows)
    summary = {
        "audit_csv": str(audit_csv),
        "gold_policy": str(args.gold_policy),
        "curated_labels_path": str(curated_path) if curated_path is not None else "",
        "exclude_review_reasons": sorted(excluded_reason_filter),
        "overall": dict(overall_counter),
        "per_source": {name: dict(counter) for name, counter in sorted(per_source_counter.items())},
        "per_split": {name: dict(counter) for name, counter in sorted(per_split_counter.items())},
        "gold_eval_rows": int(len(gold_eval_rows)),
        "gold_high_confidence_candidate_rows": int(len(high_confidence_rows)),
        "output_csv": str(gold_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Prepared gold hard-source eval slice | "
        f"rows={overall_counter['total_rows']} | included={len(gold_eval_rows)} | "
        f"high_confidence_candidates={len(high_confidence_rows)} | policy={args.gold_policy}",
        flush=True,
    )
    print(f"Gold eval slice: {gold_csv}", flush=True)
    print(f"Gold eval summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
