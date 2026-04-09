from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

from build_cyp3a4_downstream_subsets import (
    _annotate_source_family,
    _basic_subset_stats,
    _count_source_combinations,
    _merge_policy_breakdown,
    _strict_exact_clean_allowed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-family downstream subsets from the family-aware merged master dataset."
    )
    parser.add_argument(
        "--input-master",
        default="data/prepared_training/enzyme_family_merged_master/enzyme_family_merged_master_dataset.json",
        help="Family-aware merged master dataset JSON produced by build_all_family_merged_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared_training/family_downstream_subsets",
        help="Directory where per-family downstream subset folders will be written.",
    )
    parser.add_argument(
        "--add-source-family-tags",
        action="store_true",
        help="Annotate rows with source_family fields if the master dataset does not already carry them.",
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
    parser.add_argument("--family-allowlist", default="", help="Optional comma-separated family allowlist.")
    parser.add_argument("--trainable-min-rows", type=int, default=50)
    parser.add_argument("--benchmarkable-min-exact-rows", type=int, default=30)
    parser.add_argument("--benchmarkable-min-tiered-rows", type=int, default=20)
    return parser.parse_args()


def _csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").replace(";", ",").split(",") if token.strip()]


def _copy_top_level_template(master: dict[str, Any], rows: list[dict[str, Any]], *, subset_name: str, family: str) -> dict[str, Any]:
    payload = dict(master)
    payload["drugs"] = rows
    payload["n_drugs"] = int(len(rows))
    payload["n_site_labeled"] = int(sum(1 for row in rows if list(row.get("all_labeled_site_atoms") or row.get("site_atoms") or [])))
    summary = dict(master.get("summary") or {})
    summary.update(
        {
            "subset_name": subset_name,
            "target_family": family,
            "total_final_merged_rows": int(len(rows)),
        }
    )
    payload["summary"] = summary
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _family_rows(master_rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [
        row
        for row in master_rows
        if str(row.get("target_family") or row.get("enzyme_family") or "") == family
    ]


def _subset_summary_for_family(
    *,
    rows: list[dict[str, Any]],
    family: str,
    include_partial_agreement_union: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    strict_rows: list[dict[str, Any]] = []
    tiered_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    strict_decision_counts: Counter[str] = Counter()

    for row in rows:
        allowed, decision = _strict_exact_clean_allowed(
            row,
            include_partial_agreement_union=include_partial_agreement_union,
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

    subset_rows_map = {
        "strict_exact_clean": strict_rows,
        "tiered_multisite_eval": tiered_rows,
        "broad_region_aux": broad_rows,
        "conflict_audit": conflict_rows,
    }

    csv_rows: list[dict[str, Any]] = []
    for subset_name, subset_rows in subset_rows_map.items():
        stats = _basic_subset_stats(subset_rows)
        for source, count in stats["source_breakdown"].items():
            csv_rows.append(
                {
                    "family": family,
                    "subset_name": subset_name,
                    "row_count": len(subset_rows),
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
                    "family": family,
                    "subset_name": subset_name,
                    "row_count": len(subset_rows),
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
                    "family": family,
                    "subset_name": subset_name,
                    "row_count": len(subset_rows),
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
                    "family": family,
                    "subset_name": subset_name,
                    "row_count": len(subset_rows),
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

    summary = {
        "target_family": family,
        "total_master_rows_for_family": int(len(rows)),
        "strict_exact_clean_policy": (
            "Include only single_exact and multi_exact rows with non-empty exact atom labels; "
            "exclude broad_region, tiered_multisite, and conflict_preserved rows. "
            f"partial_agreement_union rows are {'included' if include_partial_agreement_union else 'excluded'} "
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
    }
    return summary, subset_rows_map, csv_rows


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_master).expanduser()
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
    allowlist = set(_csv_tokens(args.family_allowlist))
    families = sorted({str(row.get("target_family") or row.get("enzyme_family") or "").strip() for row in master_rows if str(row.get("target_family") or row.get("enzyme_family") or "").strip()})
    if allowlist:
        families = [family for family in families if family in allowlist]

    global_summary: dict[str, Any] = {
        "input_master_dataset_path": str(input_path),
        "source_family_tagging_enabled": bool(args.add_source_family_tags),
        "families": {},
        "outputs_root": str(output_dir),
    }
    family_count_rows: list[dict[str, Any]] = []
    trainable_families: list[str] = []
    benchmarkable_families: list[str] = []
    audit_only_families: list[str] = []

    for family in families:
        rows = _family_rows(master_rows, family)
        family_dir = output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        summary, subset_rows_map, csv_rows = _subset_summary_for_family(
            rows=rows,
            family=family,
            include_partial_agreement_union=bool(args.include_partial_agreement_union_in_strict),
        )

        payloads = {
            "strict_exact_clean.json": _copy_top_level_template(master, subset_rows_map["strict_exact_clean"], subset_name="strict_exact_clean", family=family),
            "tiered_multisite_eval.json": _copy_top_level_template(master, subset_rows_map["tiered_multisite_eval"], subset_name="tiered_multisite_eval", family=family),
            "broad_region_aux.json": _copy_top_level_template(master, subset_rows_map["broad_region_aux"], subset_name="broad_region_aux", family=family),
            "conflict_audit.json": _copy_top_level_template(master, subset_rows_map["conflict_audit"], subset_name="conflict_audit", family=family),
        }
        for filename, payload in payloads.items():
            _write_json(family_dir / filename, payload)
        _write_json(family_dir / "downstream_subset_summary.json", summary)
        _write_csv(family_dir / "downstream_subset_breakdown.csv", csv_rows)

        strict_count = int(summary["strict_exact_clean"]["row_count"])
        tiered_count = int(summary["tiered_multisite_eval"]["row_count"])
        broad_count = int(summary["broad_region_aux"]["row_count"])
        total_count = int(len(rows))
        trainable = (strict_count + tiered_count) >= int(args.trainable_min_rows)
        benchmarkable = strict_count >= int(args.benchmarkable_min_exact_rows) or tiered_count >= int(args.benchmarkable_min_tiered_rows)
        if trainable:
            trainable_families.append(family)
        if benchmarkable:
            benchmarkable_families.append(family)
        if not trainable and not benchmarkable:
            audit_only_families.append(family)

        family_summary = dict(summary)
        family_summary["outputs"] = {
            "strict_exact_clean_json": str(family_dir / "strict_exact_clean.json"),
            "tiered_multisite_eval_json": str(family_dir / "tiered_multisite_eval.json"),
            "broad_region_aux_json": str(family_dir / "broad_region_aux.json"),
            "conflict_audit_json": str(family_dir / "conflict_audit.json"),
            "summary_json": str(family_dir / "downstream_subset_summary.json"),
            "breakdown_csv": str(family_dir / "downstream_subset_breakdown.csv"),
        }
        family_summary["trainable"] = bool(trainable)
        family_summary["benchmarkable"] = bool(benchmarkable)
        global_summary["families"][family] = family_summary
        family_count_rows.append(
            {
                "family": family,
                "total_rows": total_count,
                "strict_exact_clean_rows": strict_count,
                "tiered_multisite_rows": tiered_count,
                "broad_region_rows": broad_count,
                "trainable": int(trainable),
                "benchmarkable": int(benchmarkable),
            }
        )

    global_summary["trainable_family_count"] = int(len(trainable_families))
    global_summary["benchmarkable_family_count"] = int(len(benchmarkable_families))
    global_summary["insufficient_data_family_count"] = int(len(audit_only_families))
    global_summary["trainable_families"] = sorted(trainable_families)
    global_summary["benchmarkable_families"] = sorted(benchmarkable_families)
    global_summary["audit_only_families"] = sorted(audit_only_families)
    _write_json(output_dir / "enzyme_family_downstream_subset_summary.json", global_summary)
    _write_csv(output_dir / "enzyme_family_family_counts.csv", family_count_rows)

    print(
        "Built per-family downstream subsets | "
        f"families={len(families)} | "
        f"trainable={len(trainable_families)} | "
        f"benchmarkable={len(benchmarkable_families)}",
        flush=True,
    )
    print(f"Summary JSON: {output_dir / 'enzyme_family_downstream_subset_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
