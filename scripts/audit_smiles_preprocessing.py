from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


def _load_entries(dataset_path: Path):
    payload = json.loads(dataset_path.read_text())
    entries = payload.get("drugs", payload)
    return list(entries)


def audit_entries(entries):
    counts = Counter()
    failures = []
    for idx, entry in enumerate(entries):
        smiles = str(entry.get("smiles", ""))
        result = prepare_mol(smiles)
        counts[result.status] += 1
        if result.mol is None:
            failures.append(
                {
                    "row_index": idx,
                    "name": str(entry.get("name", "")),
                    "drug_id": str(entry.get("drug_id", entry.get("id", ""))),
                    "original_smiles": smiles,
                    "status": result.status,
                    "error": result.error,
                }
            )
    return counts, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RDKit SMILES preprocessing outcomes")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", default="artifacts/smiles_preprocessing_audit.json")
    parser.add_argument("--output-failures", default="artifacts/smiles_preprocessing_failures.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    entries = _load_entries(dataset_path)
    counts, failures = audit_entries(entries)

    total = len(entries)
    summary = {
        "dataset": str(dataset_path),
        "total_molecules": total,
        "ok": int(counts.get("ok", 0)),
        "repaired_full_sanitize": int(counts.get("repaired_full_sanitize", 0)),
        "repaired_partial_sanitize": int(counts.get("repaired_partial_sanitize", 0)),
        "failed": int(counts.get("failed", 0)),
        "failures": len(failures),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "counts": dict(counts), "failed_entries": failures}, indent=2))

    output_failures = Path(args.output_failures)
    output_failures.parent.mkdir(parents=True, exist_ok=True)
    with output_failures.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "name", "drug_id", "original_smiles", "status", "error"])
        writer.writeheader()
        writer.writerows(failures)

    print(f"Total molecules: {total}")
    print(f"ok: {summary['ok']}")
    print(f"repaired_full_sanitize: {summary['repaired_full_sanitize']}")
    print(f"repaired_partial_sanitize: {summary['repaired_partial_sanitize']}")
    print(f"failed: {summary['failed']}")


if __name__ == "__main__":
    main()
