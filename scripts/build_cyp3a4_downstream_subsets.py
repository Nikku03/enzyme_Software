from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build policy-driven downstream subsets from the merged CYP3A4 master dataset."
    )
    parser.add_argument(
        "--input",
        default="data/prepared_training/cyp3a4_merged_dataset_local/cyp3a4_merged_dataset.json",
        help="Merged master dataset JSON produced by build_cyp3a4_merged_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared_training/cyp3a4_downstream_subsets",
        help="Directory where downstream subset files will be written.",
    )
    parser.add_argument(
        "--add-source-family-tags",
        action="store_true",
        help="Annotate each output row with a source_family field derived from molecule_source / merged_from_sources.",
    )
    parser.add_argument(
        "--include-partial-agreement-union-in-strict",
        action="store_true",
        default=True,
        help="Include partial_agreement_union rows in strict_exact_clean when they retain usable exact labels and no blocking conflicts.",
    )
    parser.add_argument(
        "--exclude-partial-agreement-union-in-strict",
        dest="include_partial_agreement_union_in_strict",
        action="store_false",
        help="Exclude all partial_agreement_union rows from strict_exact_clean.",
    )
    return parser.parse_args()


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _source_family_for_name(source_name: Any) -> str:
    token = _normalize_token(source_name)
    if token in {"attnsom", "cyp_dbs_external"}:
        return "attnsom_family"
    if token == "metxbiodb":
        return "metxbiodb_family"
    if token == "peng_external":
        return "peng_family"
    if token == "rudik_external":
        return "rudik_family"
    if token:
        return f"{token}_family"
    return "unknown_family"


def _copy_top_level_template(master: dict[str, Any], rows: list[dict[str, Any]], *, subset_name: str) -> dict[str, Any]:
    payload = dict(master)
    payload["drugs"] = rows
    payload["n_drugs"] = int(len(rows))
    payload["n_site_labeled"] = int(sum(1 for row in rows if list(row.get("all_labeled_site_atoms") or row.get("site_atoms") or [])))
    summary = dict(master.get("summary") or {})
    summary.update(
        {
            "subset_name": subset_name,
            "total_final_merged_rows": int(len(rows)),
        }
    )
    payload["summary"] = summary
    return payload


def _count_source_combinations(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        combo = " | ".join(sorted(str(v) for v in list(row.get("merged_from_sources") or []) if str(v).strip()))
        counter[combo or "unknown"] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _source_breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row.get(field) or "unknown") for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _source_family_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("source_family") or "unknown_family") for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _label_regime_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("label_regime") or "unknown") for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _merge_policy_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("merge_policy_used") or "unknown") for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _basic_subset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    multisite_count = sum(1 for row in rows if bool(row.get("is_multisite")))
    citation_count = sum(1 for row in rows if bool(row.get("citation")) or bool(row.get("citation_available")))
    doi_count = sum(1 for row in rows if bool(row.get("doi")) or bool(row.get("doi_available")))
    empty_site_count = sum(1 for row in rows if not list(row.get("all_labeled_site_atoms") or row.get("site_atoms") or []))
    return {
        "row_count": int(len(rows)),
        "source_breakdown": _source_breakdown(rows, "molecule_source"),
        "source_family_breakdown": _source_family_breakdown(rows),
        "label_regime_breakdown": _label_regime_breakdown(rows),
        "merge_policy_breakdown": _merge_policy_breakdown(rows),
        "citation_count": int(citation_count),
        "doi_count": int(doi_count),
        "multisite_count": int(multisite_count),
        "empty_site_count": int(empty_site_count),
    }


def _strict_exact_clean_allowed(row: dict[str, Any], *, include_partial_agreement_union: bool) -> tuple[bool, str]:
    label_regime = str(row.get("label_regime") or "")
    merge_policy = str(row.get("merge_policy_used") or "")
    conflict_flags = set(str(v) for v in list(row.get("conflict_flags") or []))
    exact_atoms = list(row.get("all_labeled_site_atoms") or row.get("site_atoms") or [])
    if label_regime not in {"single_exact", "multi_exact"}:
        return False, "excluded_non_exact_regime"
    if merge_policy == "conflict_preserved":
        return False, "excluded_conflict_preserved"
    if not exact_atoms:
        return False, "excluded_empty_exact_sites"
    if merge_policy == "partial_agreement_union":
        if not include_partial_agreement_union:
            return False, "excluded_partial_agreement_policy"
        blocking_flags = {
            "contains_broad_region_label",
            "site_set_disjoint",
            "unresolved_conflict",
        }
        if conflict_flags & blocking_flags:
            return False, "excluded_partial_agreement_blocking_conflict"
        return True, "included_partial_agreement_union"
    return True, "included_exact_clean"


def _annotate_source_family(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    source_family = _source_family_for_name(out.get("molecule_source"))
    merged_from = [str(v) for v in list(out.get("merged_from_sources") or []) if str(v).strip()]
    out["source_family"] = source_family
    out["merged_from_source_families"] = sorted({_source_family_for_name(value) for value in merged_from})
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "subset_name",
        "row_count",
        "source",
        "source_count",
        "source_family",
        "source_family_count",
        "label_regime",
        "label_regime_count",
        "merge_policy_used",
        "merge_policy_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    master = json.loads(input_path.read_text())
    if not isinstance(master, dict) or not isinstance(master.get("drugs"), list):
        raise TypeError(f"Merged master dataset must be an object with a 'drugs' list: {input_path}")
    master_rows = [dict(row) for row in list(master.get("drugs") or [])]
    if args.add_source_family_tags:
        master_rows = [_annotate_source_family(row) for row in master_rows]

    strict_rows: list[dict[str, Any]] = []
    tiered_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    strict_decision_counts: Counter[str] = Counter()
    for row in master_rows:
        allowed, decision = _strict_exact_clean_allowed(
            row,
            include_partial_agreement_union=bool(args.include_partial_agreement_union_in_strict),
        )
        strict_decision_counts[decision] += 1
        if allowed:
            strict_rows.append(row)
        if str(row.get("label_regime") or "") == "tiered_multisite":
            tiered_rows.append(row)
        if str(row.get("label_regime") or "") == "broad_region":
            broad_rows.append(row)
        if str(row.get("merge_policy_used") or "") == "conflict_preserved" or list(row.get("conflict_flags") or []):
            conflict_rows.append(row)

    subset_payloads = {
        "cyp3a4_strict_exact_clean.json": _copy_top_level_template(master, strict_rows, subset_name="strict_exact_clean"),
        "cyp3a4_tiered_multisite_eval.json": _copy_top_level_template(master, tiered_rows, subset_name="tiered_multisite_eval"),
        "cyp3a4_broad_region_aux.json": _copy_top_level_template(master, broad_rows, subset_name="broad_region_aux"),
        "cyp3a4_conflict_audit.json": _copy_top_level_template(master, conflict_rows, subset_name="conflict_audit"),
    }
    for filename, payload in subset_payloads.items():
        _write_json(output_dir / filename, payload)

    tier_counts = {
        "primary_only_count": int(sum(1 for row in tiered_rows if list(row.get("primary_site_atoms") or []) and not list(row.get("secondary_site_atoms") or []) and not list(row.get("tertiary_site_atoms") or []))),
        "primary_plus_secondary_count": int(sum(1 for row in tiered_rows if list(row.get("secondary_site_atoms") or []))),
        "primary_plus_secondary_plus_tertiary_count": int(sum(1 for row in tiered_rows if list(row.get("tertiary_site_atoms") or []))),
    }
    broad_region_counts = {
        "empty_exact_site_count": int(sum(1 for row in broad_rows if not list(row.get("all_labeled_site_atoms") or row.get("site_atoms") or []))),
        "rows_with_broad_region_annotations": int(sum(1 for row in broad_rows if list(row.get("broad_region_annotations") or []))),
    }
    conflict_flag_counts = Counter()
    for row in conflict_rows:
        flags = list(row.get("conflict_flags") or [])
        if not flags:
            conflict_flag_counts["no_explicit_flags"] += 1
        else:
            for flag in flags:
                conflict_flag_counts[str(flag)] += 1

    csv_rows: list[dict[str, Any]] = []
    subset_rows_map = {
        "strict_exact_clean": strict_rows,
        "tiered_multisite_eval": tiered_rows,
        "broad_region_aux": broad_rows,
        "conflict_audit": conflict_rows,
    }
    for subset_name, rows in subset_rows_map.items():
        stats = _basic_subset_stats(rows)
        for source, count in stats["source_breakdown"].items():
            csv_rows.append(
                {
                    "subset_name": subset_name,
                    "row_count": len(rows),
                    "source": source,
                    "source_count": count,
                    "source_family": "",
                    "source_family_count": "",
                    "label_regime": "",
                    "label_regime_count": "",
                    "merge_policy_used": "",
                    "merge_policy_count": "",
                }
            )
        for source_family, count in stats["source_family_breakdown"].items():
            csv_rows.append(
                {
                    "subset_name": subset_name,
                    "row_count": len(rows),
                    "source": "",
                    "source_count": "",
                    "source_family": source_family,
                    "source_family_count": count,
                    "label_regime": "",
                    "label_regime_count": "",
                    "merge_policy_used": "",
                    "merge_policy_count": "",
                }
            )
        for label_regime, count in stats["label_regime_breakdown"].items():
            csv_rows.append(
                {
                    "subset_name": subset_name,
                    "row_count": len(rows),
                    "source": "",
                    "source_count": "",
                    "source_family": "",
                    "source_family_count": "",
                    "label_regime": label_regime,
                    "label_regime_count": count,
                    "merge_policy_used": "",
                    "merge_policy_count": "",
                }
            )
        for merge_policy, count in stats["merge_policy_breakdown"].items():
            csv_rows.append(
                {
                    "subset_name": subset_name,
                    "row_count": len(rows),
                    "source": "",
                    "source_count": "",
                    "source_family": "",
                    "source_family_count": "",
                    "label_regime": "",
                    "label_regime_count": "",
                    "merge_policy_used": merge_policy,
                    "merge_policy_count": count,
                }
            )
    _write_csv(output_dir / "cyp3a4_downstream_subset_breakdown.csv", csv_rows)

    summary = {
        "input_master_dataset_path": str(input_path),
        "total_master_rows": int(len(master_rows)),
        "source_family_tagging_enabled": bool(args.add_source_family_tags),
        "strict_exact_clean_policy": (
            "Include only single_exact and multi_exact rows with non-empty exact atom labels; "
            "exclude broad_region, tiered_multisite, and conflict_preserved rows. "
            f"partial_agreement_union rows are {'included' if args.include_partial_agreement_union_in_strict else 'excluded'} "
            "when they retain usable exact labels and do not carry blocking conflicts."
        ),
        "tiered_multisite_eval_policy": (
            "Include only tiered_multisite rows and preserve primary, secondary, tertiary, "
            "and all_labeled_site_atoms without flattening to a deterministic target."
        ),
        "broad_region_aux_policy": (
            "Include only broad_region rows for weak-supervision / auxiliary use; preserve broad_region_annotations "
            "and do not treat this subset as a strict Top-1 benchmark."
        ),
        "conflict_audit_policy": (
            "Include conflict_preserved rows and any row with non-empty conflict_flags for human review and source-conflict analysis."
        ),
        "strict_exact_clean": {
            **_basic_subset_stats(strict_rows),
            "included_single_exact_count": int(sum(1 for row in strict_rows if str(row.get("label_regime") or "") == "single_exact")),
            "included_multi_exact_count": int(sum(1 for row in strict_rows if str(row.get("label_regime") or "") == "multi_exact")),
            "excluded_due_to_conflict_count": int(strict_decision_counts.get("excluded_conflict_preserved", 0) + strict_decision_counts.get("excluded_partial_agreement_blocking_conflict", 0)),
            "excluded_due_to_broad_or_tiered_regime_count": int(strict_decision_counts.get("excluded_non_exact_regime", 0)),
            "included_partial_agreement_union_count": int(strict_decision_counts.get("included_partial_agreement_union", 0)),
            "exact_policy_string_used": (
                "single_exact|multi_exact only; conflict_preserved excluded; non-empty exact sites required; "
                f"partial_agreement_union={'allowed_with_non_blocking_conflicts' if args.include_partial_agreement_union_in_strict else 'excluded'}"
            ),
            "strict_decision_counts": dict(sorted(strict_decision_counts.items())),
        },
        "tiered_multisite_eval": {
            **_basic_subset_stats(tiered_rows),
            **tier_counts,
        },
        "broad_region_aux": {
            **_basic_subset_stats(broad_rows),
            **broad_region_counts,
        },
        "conflict_audit": {
            **_basic_subset_stats(conflict_rows),
            "conflict_flag_counts": dict(sorted(conflict_flag_counts.items(), key=lambda item: (-item[1], item[0]))),
            "counts_by_merge_policy_used": _merge_policy_breakdown(conflict_rows),
            "counts_by_source_combination": _count_source_combinations(conflict_rows),
        },
        "outputs": {
            "strict_exact_clean_json": str(output_dir / "cyp3a4_strict_exact_clean.json"),
            "tiered_multisite_eval_json": str(output_dir / "cyp3a4_tiered_multisite_eval.json"),
            "broad_region_aux_json": str(output_dir / "cyp3a4_broad_region_aux.json"),
            "conflict_audit_json": str(output_dir / "cyp3a4_conflict_audit.json"),
            "summary_json": str(output_dir / "cyp3a4_downstream_subset_summary.json"),
            "breakdown_csv": str(output_dir / "cyp3a4_downstream_subset_breakdown.csv"),
        },
    }
    _write_json(output_dir / "cyp3a4_downstream_subset_summary.json", summary)

    print(
        "Built CYP3A4 downstream subsets | "
        f"strict_exact_clean={len(strict_rows)} | "
        f"tiered_multisite_eval={len(tiered_rows)} | "
        f"broad_region_aux={len(broad_rows)} | "
        f"conflict_audit={len(conflict_rows)}"
    )
    print(f"Summary JSON: {output_dir / 'cyp3a4_downstream_subset_summary.json'}")


if __name__ == "__main__":
    main()
