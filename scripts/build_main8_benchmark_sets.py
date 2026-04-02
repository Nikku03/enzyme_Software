from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from rdkit import Chem


def _canonical_smiles(row: Dict[str, object]) -> str:
    text = " ".join(str(row.get("canonical_smiles") or row.get("smiles") or "").split())
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def _site_atoms(row: Dict[str, object]) -> List[int]:
    vals: List[int] = []
    if row.get("som"):
        for som in row["som"]:
            atom = som.get("atom_idx", som) if isinstance(som, dict) else som
            if isinstance(atom, int):
                vals.append(atom)
    elif row.get("site_atoms"):
        for value in row.get("site_atoms", []):
            try:
                vals.append(int(value))
            except Exception:
                continue
    elif row.get("site_atom_indices"):
        for value in row.get("site_atom_indices", []):
            try:
                vals.append(int(value))
            except Exception:
                continue
    return vals


def _bucket(num_atoms: int) -> str:
    if num_atoms <= 15:
        return "<=15"
    if num_atoms <= 25:
        return "16-25"
    if num_atoms <= 40:
        return "26-40"
    if num_atoms <= 60:
        return "41-60"
    return "61+"


def _rdkit_atom_count(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetNumAtoms())


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    atom_counts = []
    for row in rows:
        atom_counts.append(_rdkit_atom_count(_canonical_smiles(row)))
    atom_counts = [count for count in atom_counts if count > 0]
    return {
        "count": len(rows),
        "unique_smiles": len({_canonical_smiles(row) for row in rows}),
        "sources": dict(Counter(str(row.get("source", "?")) for row in rows)),
        "cyps": dict(Counter(str(row.get("primary_cyp") or row.get("cyp") or "?") for row in rows)),
        "confidences": dict(Counter(str(row.get("confidence", "unknown")) for row in rows)),
        "site_count_buckets": {
            "single": sum(1 for row in rows if len(_site_atoms(row)) == 1),
            "multi": sum(1 for row in rows if len(_site_atoms(row)) > 1),
        },
        "atom_buckets": dict(Counter(_bucket(count) for count in atom_counts)),
        "avg_num_atoms": (sum(atom_counts) / len(atom_counts)) if atom_counts else 0.0,
        "min_num_atoms": min(atom_counts) if atom_counts else 0,
        "max_num_atoms": max(atom_counts) if atom_counts else 0,
        "names": [str(row.get("name", "")) for row in rows],
    }


def _write_dataset(path: Path, rows: List[Dict[str, object]], description: str, parent: str) -> None:
    payload = {
        "drugs": rows,
        "metadata": {
            "description": description,
            "derived_from": parent,
            "count": len(rows),
            "unique_smiles": len({_canonical_smiles(row) for row in rows}),
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive benchmark dataset variants from main8 benchmark holdout rows.")
    parser.add_argument(
        "--input",
        default="data/prepared_training/main8_benchmark_holdout_singlecyp.json",
        help="Row-level single-CYP benchmark JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared_training",
        help="Directory for derived benchmark files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(input_path.read_text()).get("drugs", [])

    benchmark_a = list(rows)

    unique_rows_by_smiles: Dict[str, Dict[str, object]] = {}
    for row in rows:
        unique_rows_by_smiles.setdefault(_canonical_smiles(row), row)
    benchmark_b = list(unique_rows_by_smiles.values())

    high_conf = {"validated", "validated_gold", "validated_literature"}
    benchmark_c = [row for row in benchmark_a if str(row.get("confidence", "unknown")) in high_conf]

    outputs = {
        "main8_benchmark_a_row_level_singlecyp.json": (
            benchmark_a,
            "External Benchmark A: row-level single-CYP benchmark absent from main8",
        ),
        "main8_benchmark_b_unique_molecules.json": (
            benchmark_b,
            "External Benchmark B: unique-molecule benchmark derived from A by canonical SMILES deduplication",
        ),
        "main8_benchmark_c_high_confidence.json": (
            benchmark_c,
            "External Benchmark C: high-confidence subset of A (validated / validated_gold / validated_literature)",
        ),
    }

    summary = {}
    for filename, (subset, description) in outputs.items():
        dataset_path = output_dir / filename
        _write_dataset(dataset_path, subset, description, str(input_path))
        summary[filename] = _summary(subset)

    summary_path = output_dir / "main8_benchmark_sets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Wrote benchmark datasets:")
    for filename in outputs:
        print(output_dir / filename)
    print(summary_path)


if __name__ == "__main__":
    main()
