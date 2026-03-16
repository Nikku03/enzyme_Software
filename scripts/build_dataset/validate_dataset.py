from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem


def validate_dataset(dataset_path: str | Path) -> bool:
    with Path(dataset_path).open() as handle:
        data = json.load(handle)
    drugs = data.get("drugs", [])

    print(f"Total drugs: {len(drugs)}\n")

    invalid_smiles = []
    invalid_som = []
    with_som = 0

    cyp_counts = Counter()
    confidence_counts = Counter()

    for drug in drugs:
        smiles = drug.get("smiles")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            invalid_smiles.append(drug.get("name"))
            continue

        cyp_counts[drug.get("primary_cyp")] += 1
        confidence_counts[drug.get("confidence")] += 1

        som = drug.get("som", []) or []
        if som:
            with_som += 1
        for site in som:
            atom_idx = int(site.get("atom_idx", -1))
            if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
                invalid_som.append(drug.get("name"))
                break

    print(f"Invalid SMILES: {len(invalid_smiles)}")
    if invalid_smiles[:5]:
        print(f"  Examples: {invalid_smiles[:5]}")
    print()

    print("CYP Distribution:")
    for cyp, count in sorted(cyp_counts.items()):
        print(f"  {cyp}: {count}")
    print()

    print(f"Drugs with SoM annotations: {with_som}")
    print(f"Drugs without SoM: {len(drugs) - with_som}\n")

    print("Confidence levels:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"  {confidence}: {count}")
    print()

    print(f"Drugs with invalid SoM indices: {len(invalid_som)}")
    if invalid_som[:5]:
        print(f"  Examples: {invalid_som[:5]}")
    print("\n" + "=" * 50)

    ok = len(invalid_smiles) == 0 and len(invalid_som) == 0
    print("✓ Dataset validation PASSED" if ok else "✗ Dataset validation FAILED - fix issues above")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CYP metabolism dataset JSON file.")
    parser.add_argument("dataset_path", nargs="?", default="data/cyp_metabolism_dataset.json")
    args = parser.parse_args()
    validate_dataset(args.dataset_path)


if __name__ == "__main__":
    main()
