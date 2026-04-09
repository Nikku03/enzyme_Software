from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]


def _normalize_source_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_bucket(raw_source: str) -> str:
    normalized = _normalize_source_name(raw_source)
    mapping = {
        "attnsom": "ATTNSOM",
        "az120": "AZ120",
        "drugbank": "DrugBank",
        "metxbiodb": "MetXBioDB",
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


def build_balanced_dataset(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    seed: int,
    per_source_target: int,
) -> dict[str, object]:
    payload = json.loads(input_path.read_text())
    rows = list(payload.get("drugs", payload))

    allowed_bucket_names = ("ATTNSOM", "AZ120", "DrugBank", "MetXBioDB")
    allowed_bucket_set = set(allowed_bucket_names)
    bucket_rows: dict[str, list[dict]] = {name: [] for name in allowed_bucket_names}
    raw_source_counts = Counter()
    normalized_source_counts = Counter()
    discarded_by_raw_source = Counter()
    discarded_by_bucket = Counter()
    skipped_schema_counts = Counter()

    for row in rows:
        raw_source = str(row.get("source") or row.get("data_source") or "unknown").strip()
        raw_source_counts[raw_source] += 1
        canonical_bucket = _canonical_bucket(raw_source)
        normalized_source_counts[canonical_bucket or _normalize_source_name(raw_source) or "unknown"] += 1
        if not canonical_bucket:
            discarded_by_raw_source[raw_source] += 1
            discarded_by_bucket["excluded_tail_source"] += 1
            continue
        site_atoms = _site_atom_indices(row)
        if not site_atoms:
            skipped_schema_counts["missing_site_atoms"] += 1
            discarded_by_raw_source[raw_source] += 1
            discarded_by_bucket[canonical_bucket] += 1
            continue
        if not str(row.get("smiles") or "").strip():
            skipped_schema_counts["missing_smiles"] += 1
            discarded_by_raw_source[raw_source] += 1
            discarded_by_bucket[canonical_bucket] += 1
            continue
        bucket_row = dict(row)
        bucket_row["source"] = canonical_bucket
        bucket_row["site_source"] = canonical_bucket
        details = [str(v) for v in list(bucket_row.get("source_details") or []) if str(v).strip()]
        details.append(raw_source)
        dedup_details = []
        seen = set()
        for detail in details:
            if detail not in seen:
                dedup_details.append(detail)
                seen.add(detail)
        bucket_row["source_details"] = dedup_details
        bucket_row["balanced_source_bucket"] = canonical_bucket
        bucket_row["balanced_source_original"] = raw_source
        bucket_rows[canonical_bucket].append(bucket_row)

    for source_name in allowed_bucket_names:
        available = len(bucket_rows[source_name])
        if available < int(per_source_target):
            raise RuntimeError(
                f"Balanced dataset requires {per_source_target} rows for {source_name}, "
                f"but only {available} are available."
            )

    rng = random.Random(int(seed))
    selected_rows: list[dict] = []
    selection_preview: dict[str, list[str]] = {}
    sampled_original_counts = Counter()
    for source_name in allowed_bucket_names:
        candidates = list(bucket_rows[source_name])
        rng.shuffle(candidates)
        chosen = candidates[: int(per_source_target)]
        selected_rows.extend(chosen)
        selection_preview[source_name] = [str(row.get("id") or row.get("name") or "") for row in chosen[:5]]
        sampled_original_counts.update(str(row.get("balanced_source_original") or "") for row in chosen)
        discarded_by_bucket[source_name] += max(0, len(candidates) - int(per_source_target))

    rng.shuffle(selected_rows)
    output_payload = {
        "n_drugs": int(len(selected_rows)),
        "n_site_labeled": int(len(selected_rows)),
        "summary": _summary_for(selected_rows),
        "build_stats": {
            "input_dataset": str(input_path),
            "seed": int(seed),
            "per_source_target": int(per_source_target),
            "allowed_source_buckets": list(allowed_bucket_names),
            "raw_source_counts": dict(sorted(raw_source_counts.items())),
            "normalized_source_counts": dict(sorted(normalized_source_counts.items())),
            "discarded_by_raw_source": dict(sorted(discarded_by_raw_source.items())),
            "discarded_by_bucket": dict(sorted(discarded_by_bucket.items())),
            "skipped_schema_counts": dict(sorted(skipped_schema_counts.items())),
            "sampled_original_source_counts": dict(sorted(sampled_original_counts.items())),
            "selection_preview": selection_preview,
            "source_balance_ok": all(
                int(output_payload_count) == int(per_source_target)
                for output_payload_count in Counter(str(row.get("source") or "") for row in selected_rows).values()
            ),
        },
        "drugs": selected_rows,
    }

    summary = {
        "input_dataset": str(input_path),
        "output_dataset": str(output_path),
        "seed_used": int(seed),
        "per_source_target": int(per_source_target),
        "total_rows": int(len(selected_rows)),
        "per_source_counts": output_payload["summary"]["per_source_counts"],
        "single_vs_multi_site_counts": output_payload["summary"]["single_vs_multi_site_counts"],
        "atom_count_bucket_counts": output_payload["summary"]["atom_count_bucket_counts"],
        "confidence_counts": output_payload["summary"]["confidence_counts"],
        "merged_metxbiodb_rows": output_payload["summary"]["merged_metxbiodb_rows"],
        "raw_source_counts_before_normalization": dict(sorted(raw_source_counts.items())),
        "normalized_source_counts_before_sampling": dict(sorted(normalized_source_counts.items())),
        "discarded_by_raw_source": dict(sorted(discarded_by_raw_source.items())),
        "discarded_by_bucket": dict(sorted(discarded_by_bucket.items())),
        "skipped_schema_counts": dict(sorted(skipped_schema_counts.items())),
        "sampled_original_source_counts": dict(sorted(sampled_original_counts.items())),
        "source_names_before_normalization": sorted(raw_source_counts.keys()),
        "source_names_after_normalization": sorted(normalized_source_counts.keys()),
        "warnings": [],
    }
    if summary["per_source_counts"].get("ATTNSOM", 0) != per_source_target:
        summary["warnings"].append("ATTNSOM count mismatch")
    if summary["per_source_counts"].get("AZ120", 0) != per_source_target:
        summary["warnings"].append("AZ120 count mismatch")
    if summary["per_source_counts"].get("DrugBank", 0) != per_source_target:
        summary["warnings"].append("DrugBank count mismatch")
    if summary["per_source_counts"].get("MetXBioDB", 0) != per_source_target:
        summary["warnings"].append("MetXBioDB count mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced CYP3A4 dataset with 104 rows from four sources.")
    parser.add_argument("--input", default="data/prepared_training/main8_cyp3a4_augmented.json")
    parser.add_argument("--output", default="data/prepared_training/main8_cyp3a4_balanced_104x4.json")
    parser.add_argument(
        "--summary-output",
        default="data/prepared_training/main8_cyp3a4_balanced_104x4_summary.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-source-target", type=int, default=104)
    args = parser.parse_args()

    summary = build_balanced_dataset(
        input_path=ROOT / args.input,
        output_path=ROOT / args.output,
        summary_path=ROOT / args.summary_output,
        seed=int(args.seed),
        per_source_target=int(args.per_source_target),
    )
    print(
        "Balanced CYP3A4 dataset complete | "
        f"rows={summary['total_rows']} | "
        f"per_source={summary['per_source_counts']} | "
        f"output={ROOT / args.output}",
        flush=True,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
