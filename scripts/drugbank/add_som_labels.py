from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scripts.build_dataset.generate_som_labels import generate_som_for_drug


def add_som_labels(
    input_path: str = "data/training_dataset_drugbank.json",
    output_path: str = "data/training_dataset_final.json",
) -> List[Dict]:
    print("=" * 60)
    print("Adding SoM Labels")
    print("=" * 60)

    with open(input_path) as f:
        data = json.load(f)
    drugs = data.get("drugs", [])

    stats = defaultdict(int)
    for drug in drugs:
        if drug.get("site_atoms"):
            stats["already_labeled"] += 1
            continue
        smiles = drug.get("smiles")
        cyp = drug.get("primary_cyp")
        if not smiles or not cyp:
            stats["missing_data"] += 1
            continue
        predictions = generate_som_for_drug(smiles, cyp)
        if predictions:
            drug["site_atoms"] = [int(p["atom_idx"]) for p in predictions]
            drug["expected_bond_class"] = str(predictions[0]["bond_class"])
            drug["som_source"] = "bde_predicted"
            stats["predicted"] += 1
        else:
            stats["prediction_failed"] += 1

    print("\nSoM Labeling Stats:")
    for stat, count in sorted(stats.items()):
        print(f"  {stat}: {count}")
    final_som = sum(1 for d in drugs if d.get("site_atoms"))
    print(f"\nFinal SoM coverage: {final_som}/{len(drugs)} ({final_som / len(drugs) * 100 if drugs else 0:.1f}%)")

    data.setdefault("metadata", {})["som_labeled"] = final_som
    data["metadata"]["som_validated"] = stats.get("already_labeled", 0)
    data["metadata"]["som_predicted"] = stats.get("predicted", 0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {output_path}")
    return drugs


if __name__ == "__main__":
    add_som_labels()
