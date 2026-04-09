from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile reviewed hard-source curation decisions into separate clean-label and hard-failure slices.")
    parser.add_argument("--audit-csv", required=True, help="Path to hard_source_audit_val_test.csv")
    parser.add_argument("--reviewed-csv", required=True, help="Path to reviewed hard_source_curation_template.csv")
    parser.add_argument("--output-dir", default="artifacts/hard_source_curation")
    parser.add_argument("--hard-sources", default="attnsom,cyp_dbs_external")
    parser.add_argument("--min-subset-size", type=int, default=8)
    parser.add_argument("--dominance-threshold", type=float, default=0.8)
    parser.add_argument("--use-suggested-flags-when-missing", action="store_true")
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


def _parse_bool_or_none(value: Any) -> bool | None:
    text = str(value or "").strip()
    if not text:
        return None
    return _parse_bool(value)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("split", "")).strip().lower(),
        str(row.get("source", "")).strip().lower(),
        _parse_int(row.get("molecule_key", 0), 0),
    )


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


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _default_exclusion_reason(row: dict[str, Any]) -> str:
    if _parse_bool(row.get("exclude_from_curated", False)):
        return "explicit_exclude"
    if str(row.get("review_status", "") or "").strip().lower() == "excluded":
        return "review_status_excluded"
    label_conf = str(row.get("label_confidence", "") or "").strip().lower()
    if label_conf == "low":
        return "label_confidence_low"
    atom_conf = str(row.get("atom_mapping_confidence", "") or "").strip().lower()
    if atom_conf == "low":
        return "atom_mapping_confidence_low"
    literature = str(row.get("literature_support_status", "") or "").strip().lower()
    if literature in {"ambiguous", "conflicting", "mapping_unclear"}:
        return f"literature_{literature}"
    if literature == "not_checked":
        return "literature_not_checked"
    return "not_selected"


def _reviewed_bool(
    row: dict[str, Any],
    field: str,
    *,
    fallback_field: str,
    use_suggestion: bool,
) -> bool:
    explicit = _parse_bool_or_none(row.get(field, ""))
    if explicit is not None:
        return bool(explicit)
    if use_suggestion:
        return _parse_bool(row.get(fallback_field, False))
    return False


def _build_warnings(
    *,
    subset_name: str,
    subset_rows: list[dict[str, Any]],
    hard_sources: list[str],
    min_subset_size: int,
    dominance_threshold: float,
) -> list[str]:
    warnings: list[str] = []
    total = len(subset_rows)
    per_source = Counter(str(row.get("source", "") or "") for row in subset_rows)
    if total < int(min_subset_size):
        warnings.append(f"{subset_name}: below_min_size:{total}<{int(min_subset_size)}")
    for source in hard_sources:
        if per_source.get(source, 0) <= 0:
            warnings.append(f"{subset_name}: missing_source:{source}")
    if total > 0 and per_source:
        dominant_source, dominant_count = per_source.most_common(1)[0]
        if float(dominant_count) / float(total) > float(dominance_threshold):
            warnings.append(
                f"{subset_name}: dominated_by_source:{dominant_source}:{dominant_count}/{total}"
            )
    return warnings


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_conf_counter = Counter(str(row.get("label_confidence", "") or "") for row in rows)
    literature_counter = Counter(str(row.get("literature_support_status", "") or "") for row in rows)
    split_counter = Counter(str(row.get("split", "") or "") for row in rows)
    source_counter = Counter(str(row.get("source", "") or "") for row in rows)
    return {
        "total_rows": int(len(rows)),
        "per_source_counts": dict(sorted(source_counter.items())),
        "per_split_counts": dict(sorted(split_counter.items())),
        "single_site_count": int(sum(1 for row in rows if not _parse_bool(row.get("is_multi_site", False)))),
        "multi_site_count": int(sum(1 for row in rows if _parse_bool(row.get("is_multi_site", False)))),
        "excluded_for_ambiguity_count": int(
            sum(
                1
                for row in rows
                if str(row.get("literature_support_status", "") or "").strip().lower() in {"ambiguous", "conflicting"}
            )
        ),
        "excluded_for_mapping_uncertainty_count": int(
            sum(1 for row in rows if str(row.get("atom_mapping_confidence", "") or "").strip().lower() == "low")
        ),
        "confidence_counts": dict(sorted(label_conf_counter.items())),
        "literature_support_counts": dict(sorted(literature_counter.items())),
    }


def main() -> None:
    args = _parse_args()
    audit_csv = Path(args.audit_csv).expanduser()
    reviewed_csv = Path(args.reviewed_csv).expanduser()
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")
    if not reviewed_csv.exists():
        raise FileNotFoundError(f"Reviewed CSV not found: {reviewed_csv}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = _load_csv_rows(audit_csv)
    reviewed_rows = _load_csv_rows(reviewed_csv)
    reviewed_map = {_row_key(row): row for row in reviewed_rows}
    hard_sources = [
        str(token).strip().lower()
        for token in str(args.hard_sources or "").split(",")
        if str(token).strip()
    ]

    master_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    hard_failure_rows: list[dict[str, Any]] = []
    exclusion_counter = Counter()
    source_comp_overall = Counter()
    split_comp_overall = Counter()
    excluded_source_counts = Counter()
    excluded_split_counts = Counter()

    for audit_row in audit_rows:
        key = _row_key(audit_row)
        source = str(audit_row.get("source", "") or "").strip().lower()
        if hard_sources and source not in hard_sources:
            continue
        review_row = dict(reviewed_map.get(key) or {})
        merged = dict(audit_row)
        merged.update(review_row)

        include_clean = _reviewed_bool(
            merged,
            "include_clean_label_eval",
            fallback_field="suggest_include_clean_label_eval",
            use_suggestion=bool(args.use_suggested_flags_when_missing),
        )
        include_hard = _reviewed_bool(
            merged,
            "include_hard_failure_gold",
            fallback_field="suggest_include_hard_failure_gold",
            use_suggestion=bool(args.use_suggested_flags_when_missing),
        )
        if _parse_bool(merged.get("exclude_from_curated", False)):
            include_clean = False
            include_hard = False

        merged["include_clean_label_eval"] = bool(include_clean)
        merged["include_hard_failure_gold"] = bool(include_hard)
        merged["review_status"] = str(merged.get("review_status", "") or "unreviewed")
        merged["label_confidence"] = str(merged.get("label_confidence", "") or "")
        merged["atom_mapping_confidence"] = str(merged.get("atom_mapping_confidence", "") or "")
        merged["literature_support_status"] = str(merged.get("literature_support_status", "") or "")
        merged["curation_reason"] = str(merged.get("curation_reason", "") or "")
        merged["curation_notes"] = str(merged.get("curation_notes", "") or "")
        merged["exclude_from_curated"] = bool(_parse_bool(merged.get("exclude_from_curated", False)))
        merged["needs_literature_check"] = bool(_parse_bool(merged.get("needs_literature_check", False)))
        merged["needs_atom_mapping_review"] = bool(_parse_bool(merged.get("needs_atom_mapping_review", False)))
        merged["multi_site_reviewed"] = bool(_parse_bool(merged.get("multi_site_reviewed", False)))
        merged["multi_site_policy_applied"] = str(merged.get("multi_site_policy_applied", "") or "")
        merged["curated_true_site_indices"] = _parse_json_cell(merged.get("curated_true_site_indices", "")) or []
        merged["curation_key"] = "|".join([key[0], key[1], str(key[2])])

        if not include_clean and not include_hard:
            exclusion_counter[_default_exclusion_reason(merged)] += 1
            excluded_source_counts[source] += 1
            excluded_split_counts[str(merged.get("split", "") or "")] += 1

        master_rows.append(merged)
        source_comp_overall[source] += 1
        split_comp_overall[str(merged.get("split", "") or "")] += 1
        if include_clean:
            clean_rows.append(merged)
        if include_hard:
            hard_failure_rows.append(merged)

    clean_warnings = _build_warnings(
        subset_name="clean_label_eval",
        subset_rows=clean_rows,
        hard_sources=hard_sources,
        min_subset_size=int(args.min_subset_size),
        dominance_threshold=float(args.dominance_threshold),
    )
    hard_warnings = _build_warnings(
        subset_name="hard_failure_gold",
        subset_rows=hard_failure_rows,
        hard_sources=hard_sources,
        min_subset_size=int(args.min_subset_size),
        dominance_threshold=float(args.dominance_threshold),
    )

    master_csv = output_dir / "curation_master_table.csv"
    clean_csv = output_dir / "clean_label_eval_slice.csv"
    hard_csv = output_dir / "hard_failure_gold_slice.csv"
    summary_json = output_dir / "curation_summary.json"
    _write_csv(master_csv, master_rows)
    _write_csv(clean_csv, clean_rows)
    _write_csv(hard_csv, hard_failure_rows)

    summary = {
        "audit_csv": str(audit_csv),
        "reviewed_csv": str(reviewed_csv),
        "hard_sources": list(hard_sources),
        "overall_counts": {
            "total_rows": int(len(master_rows)),
            "clean_label_eval_rows": int(len(clean_rows)),
            "hard_failure_gold_rows": int(len(hard_failure_rows)),
            "overlap_rows": int(sum(1 for row in master_rows if row["include_clean_label_eval"] and row["include_hard_failure_gold"])),
            "excluded_rows": int(sum(1 for row in master_rows if not row["include_clean_label_eval"] and not row["include_hard_failure_gold"])),
        },
        "overall_source_counts": dict(sorted(source_comp_overall.items())),
        "overall_split_counts": dict(sorted(split_comp_overall.items())),
        "clean_label_eval": {
            **_subset_summary(clean_rows),
            "warnings": clean_warnings,
        },
        "hard_failure_gold": {
            **_subset_summary(hard_failure_rows),
            "warnings": hard_warnings,
        },
        "excluded_rows": {
            "count": int(sum(1 for row in master_rows if not row["include_clean_label_eval"] and not row["include_hard_failure_gold"])),
            "top_exclusion_reasons": dict(exclusion_counter.most_common(10)),
            "per_source_counts": dict(sorted(excluded_source_counts.items())),
            "per_split_counts": dict(sorted(excluded_split_counts.items())),
        },
        "output_files": {
            "curation_master_table": str(master_csv),
            "clean_label_eval_slice": str(clean_csv),
            "hard_failure_gold_slice": str(hard_csv),
            "curation_summary": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Curated hard-source slices compiled | "
        f"clean_rows={len(clean_rows)} | "
        f"hard_failure_rows={len(hard_failure_rows)} | "
        f"excluded={summary['excluded_rows']['count']}",
        flush=True,
    )
    for warning in clean_warnings + hard_warnings:
        print(f"WARNING: {warning}", flush=True)
    print(f"Master table: {master_csv}", flush=True)
    print(f"Clean-label slice: {clean_csv}", flush=True)
    print(f"Hard-failure slice: {hard_csv}", flush=True)
    print(f"Curation summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
