from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a row-level curation template for hard-source audit review.")
    parser.add_argument("--audit-csv", required=True, help="Path to hard_source_audit_val_test.csv")
    parser.add_argument("--output-csv", default="artifacts/hard_source_audit/hard_source_curation_template.csv")
    parser.add_argument("--hard-sources", default="attnsom,cyp_dbs_external")
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


def _review_defaults(row: dict[str, Any], *, hard_sources: set[str]) -> dict[str, Any]:
    source = str(row.get("source", "") or "").strip().lower()
    needs_manual_review = _parse_bool(row.get("needs_manual_review", False))
    is_multi_site = _parse_bool(row.get("is_multi_site", False))
    error_type = str(row.get("error_type", "") or "")
    manual_review_reasons = list(_parse_json_cell(row.get("manual_review_reasons", "")) or [])
    true_site_count = _parse_int(row.get("true_site_count", 0), 0)

    label_confidence = "low" if needs_manual_review else "medium"
    atom_mapping_confidence = "low" if is_multi_site or true_site_count != 1 else "medium"
    needs_atom_mapping_review = bool(is_multi_site or true_site_count != 1)
    needs_literature_check = True

    suggest_clean = (
        not needs_manual_review
        and not is_multi_site
        and true_site_count == 1
        and source in hard_sources
    )
    suggest_hard_failure = (
        error_type in {"shortlist_miss", "winner_miss"}
        and not needs_manual_review
        and atom_mapping_confidence != "low"
        and source in hard_sources
    )

    auto_reason_parts: list[str] = []
    if needs_manual_review:
        auto_reason_parts.append("manual_review_flagged")
    if is_multi_site:
        auto_reason_parts.append("multi_site")
    if error_type in {"shortlist_miss", "winner_miss"}:
        auto_reason_parts.append(error_type)
    if not auto_reason_parts:
        auto_reason_parts.append("candidate_clean_label_eval")

    return {
        "review_status": "unreviewed",
        "label_confidence": label_confidence,
        "atom_mapping_confidence": atom_mapping_confidence,
        "literature_support_status": "not_checked",
        "curation_reason": "",
        "curation_notes": "",
        "include_clean_label_eval": False,
        "include_hard_failure_gold": False,
        "exclude_from_curated": False,
        "needs_literature_check": needs_literature_check,
        "needs_atom_mapping_review": needs_atom_mapping_review,
        "multi_site_reviewed": False,
        "multi_site_policy_applied": "",
        "curated_true_site_indices": [],
        "suggest_include_clean_label_eval": bool(suggest_clean),
        "suggest_include_hard_failure_gold": bool(suggest_hard_failure),
        "auto_curation_reason": "|".join(auto_reason_parts),
        "auto_review_reason_count": int(len(manual_review_reasons)),
    }


def main() -> None:
    args = _parse_args()
    audit_csv = Path(args.audit_csv).expanduser()
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")
    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with audit_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    template_rows: list[dict[str, Any]] = []
    hard_sources = {
        str(token).strip().lower()
        for token in str(args.hard_sources or "").split(",")
        if str(token).strip()
    }
    for row in rows:
        source = str(row.get("source", "") or "").strip().lower()
        if hard_sources and source not in hard_sources:
            continue
        prepared = dict(row)
        prepared.update(_review_defaults(row, hard_sources=hard_sources))
        template_rows.append(prepared)

    _write_csv(output_csv, template_rows)
    print(
        "Hard-source curation template complete | "
        f"rows={len(template_rows)} | "
        f"output={output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
