from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]


def _normalize_source_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_bucket(raw_source: str) -> str:
    normalized = _normalize_source_name(raw_source)
    mapping = {
        "az120": "AZ120",
        "drugbank": "DrugBank",
        "metxbiodb": "MetXBioDB",
        "cyp_dbs_external": "cyp_dbs_external",
    }
    return mapping.get(normalized, "")


def _atom_count_bucket(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    num_atoms = int(mol.GetNumAtoms()) if mol is not None else 0
    if num_atoms <= 0:
        return "unknown"
    if num_atoms <= 15:
        return "<=15"
    if num_atoms <= 25:
        return "16-25"
    if num_atoms <= 40:
        return "26-40"
    if num_atoms <= 60:
        return "41-60"
    return "61+"


def _site_atom_indices(row: dict) -> list[int]:
    if row.get("site_atoms"):
        return sorted({int(v) for v in list(row.get("site_atoms") or [])})
    if row.get("site_atom_indices"):
        return sorted({int(v) for v in list(row.get("site_atom_indices") or [])})
    if row.get("metabolism_sites"):
        return sorted({int(v) for v in list(row.get("metabolism_sites") or [])})
    if row.get("som"):
        out = []
        for item in list(row.get("som") or []):
            atom_idx = item.get("atom_idx", item) if isinstance(item, dict) else item
            if isinstance(atom_idx, int):
                out.append(int(atom_idx))
        return sorted(set(out))
    return []


def _single_multi_label(row: dict) -> str:
    return "multi_site" if len(_site_atom_indices(row)) > 1 else "single_site"


def _confidence_label(row: dict) -> str:
    return str(row.get("confidence") or "unknown").strip().lower() or "unknown"


def _summary_for(rows: list[dict]) -> dict[str, object]:
    return {
        "total_rows": int(len(rows)),
        "per_source_counts": dict(sorted(Counter(str(row.get("source") or "unknown") for row in rows).items())),
        "single_vs_multi_site_counts": dict(sorted(Counter(_single_multi_label(row) for row in rows).items())),
        "atom_count_bucket_counts": dict(sorted(Counter(_atom_count_bucket(str(row.get("smiles") or "")) for row in rows).items())),
        "confidence_counts": dict(sorted(Counter(_confidence_label(row) for row in rows).items())),
        "merged_metxbiodb_rows": int(sum(1 for row in rows if str(row.get("source") or "") == "MetXBioDB")),
    }


def build_no_attnsom_dataset(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    payload = json.loads(input_path.read_text())
    rows = list(payload.get("drugs", payload))

    kept_bucket_names = ("AZ120", "DrugBank", "MetXBioDB", "cyp_dbs_external")
    kept_bucket_set = set(kept_bucket_names)
    output_rows: list[dict] = []
    raw_source_counts = Counter()
    normalized_source_counts = Counter()
    kept_source_counts = Counter()
    excluded_source_counts = Counter()
    excluded_reason_counts = Counter()
    skipped_schema_counts = Counter()

    for row in rows:
        raw_source = str(row.get("source") or row.get("data_source") or "unknown").strip()
        normalized_source = _normalize_source_name(raw_source)
        raw_source_counts[raw_source] += 1
        normalized_source_counts[normalized_source or "unknown"] += 1

        if normalized_source == "attnsom":
            excluded_source_counts[raw_source] += 1
            excluded_reason_counts["attnsom_removed"] += 1
            continue

        canonical_bucket = _canonical_bucket(raw_source)
        if canonical_bucket not in kept_bucket_set:
            excluded_source_counts[raw_source] += 1
            excluded_reason_counts["excluded_tail_source"] += 1
            continue

        site_atoms = _site_atom_indices(row)
        if not site_atoms:
            skipped_schema_counts["missing_site_atoms"] += 1
            excluded_source_counts[raw_source] += 1
            excluded_reason_counts["missing_site_atoms"] += 1
            continue
        if not str(row.get("smiles") or "").strip():
            skipped_schema_counts["missing_smiles"] += 1
            excluded_source_counts[raw_source] += 1
            excluded_reason_counts["missing_smiles"] += 1
            continue

        kept_row = dict(row)
        kept_row["source"] = canonical_bucket
        kept_row["site_source"] = canonical_bucket
        details = [str(v) for v in list(kept_row.get("source_details") or []) if str(v).strip()]
        details.append(raw_source)
        dedup_details = []
        seen = set()
        for detail in details:
            if detail not in seen:
                dedup_details.append(detail)
                seen.add(detail)
        kept_row["source_details"] = dedup_details
        kept_row["no_attnsom_source_bucket"] = canonical_bucket
        kept_row["no_attnsom_source_original"] = raw_source
        output_rows.append(kept_row)
        kept_source_counts[canonical_bucket] += 1

    output_payload = {
        "n_drugs": int(len(output_rows)),
        "n_site_labeled": int(len(output_rows)),
        "summary": _summary_for(output_rows),
        "build_stats": {
            "input_dataset": str(input_path),
            "attnsom_policy": "exclude",
            "tail_source_policy": "exclude",
            "kept_source_buckets": list(kept_bucket_names),
            "raw_source_counts": dict(sorted(raw_source_counts.items())),
            "normalized_source_counts": dict(sorted(normalized_source_counts.items())),
            "kept_source_counts": dict(sorted(kept_source_counts.items())),
            "excluded_source_counts": dict(sorted(excluded_source_counts.items())),
            "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
            "skipped_schema_counts": dict(sorted(skipped_schema_counts.items())),
            "normalized_merges_applied": {
                "metxbiodb": "MetXBioDB",
                "MetXBioDB": "MetXBioDB",
            },
        },
        "drugs": output_rows,
    }

    summary = {
        "input_dataset": str(input_path),
        "output_dataset": str(output_path),
        "original_total_rows": int(len(rows)),
        "rows_removed_due_to_attnsom_exclusion": int(excluded_reason_counts.get("attnsom_removed", 0)),
        "rows_removed_due_to_tail_source_exclusion": int(excluded_reason_counts.get("excluded_tail_source", 0)),
        "final_kept_total": int(len(output_rows)),
        "kept_source_buckets": list(kept_bucket_names),
        "excluded_source_buckets": ["ATTNSOM", "literature", "MetaPred", "validated"],
        "per_source_counts": output_payload["summary"]["per_source_counts"],
        "single_vs_multi_site_counts": output_payload["summary"]["single_vs_multi_site_counts"],
        "atom_count_bucket_counts": output_payload["summary"]["atom_count_bucket_counts"],
        "confidence_counts": output_payload["summary"]["confidence_counts"],
        "merged_metxbiodb_rows": output_payload["summary"]["merged_metxbiodb_rows"],
        "raw_source_counts_before_normalization": dict(sorted(raw_source_counts.items())),
        "normalized_source_counts_before_filtering": dict(sorted(normalized_source_counts.items())),
        "kept_source_counts": dict(sorted(kept_source_counts.items())),
        "excluded_source_counts": dict(sorted(excluded_source_counts.items())),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "skipped_schema_counts": dict(sorted(skipped_schema_counts.items())),
        "source_names_before_normalization": sorted(raw_source_counts.keys()),
        "source_names_after_normalization": sorted(normalized_source_counts.keys()),
        "normalized_merges_applied": {
            "metxbiodb": "MetXBioDB",
            "MetXBioDB": "MetXBioDB",
        },
        "warnings": [],
    }
    if summary["per_source_counts"].get("cyp_dbs_external", 0) == 0:
        summary["warnings"].append("cyp_dbs_external rows missing after filtering")
    if summary["final_kept_total"] <= 0:
        summary["warnings"].append("no rows kept after ATTNSOM exclusion")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CYP3A4 dataset with ATTNSOM removed and stable non-ATTNSOM sources kept.")
    parser.add_argument("--input", default="data/prepared_training/main8_cyp3a4_augmented.json")
    parser.add_argument("--output", default="data/prepared_training/main8_cyp3a4_no_attnsom.json")
    parser.add_argument(
        "--summary-output",
        default="data/prepared_training/main8_cyp3a4_no_attnsom_summary.json",
    )
    args = parser.parse_args()

    summary = build_no_attnsom_dataset(
        input_path=ROOT / args.input,
        output_path=ROOT / args.output,
        summary_path=ROOT / args.summary_output,
    )
    print(
        "CYP3A4 no-ATTNSOM dataset complete | "
        f"rows={summary['final_kept_total']} | "
        f"per_source={summary['per_source_counts']} | "
        f"output={ROOT / args.output}",
        flush=True,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
