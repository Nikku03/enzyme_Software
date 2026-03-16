from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from scripts.build_dataset.generate_som_labels import generate_som_for_drug


def add_som_labels(input_path: str, output_path: str) -> Dict[str, object]:
    print("=" * 60)
    print("Adding SoM Labels")
    print("=" * 60)
    payload = json.loads(Path(input_path).read_text())
    drugs = list(payload.get("drugs", payload))
    labeled = 0
    skipped = 0
    failed = 0
    for drug in drugs:
        if drug.get("site_atoms") and str(drug.get("som_label_source", "")).lower() in {"validated", "curated"}:
            skipped += 1
            continue
        smiles = drug.get("smiles")
        cyp = drug.get("primary_cyp") or drug.get("cyp")
        if not smiles or not cyp:
            failed += 1
            continue
        predictions = generate_som_for_drug(str(smiles), str(cyp))
        if not predictions:
            failed += 1
            continue
        drug["site_atoms"] = [int(p["atom_idx"]) for p in predictions]
        drug["expected_bond_class"] = str(predictions[0]["bond_class"])
        drug["som"] = [
            {
                "atom_idx": int(p["atom_idx"]),
                "bond_class": str(p["bond_class"]),
                "confidence": float(p["score"]),
            }
            for p in predictions
        ]
        drug["som_label_source"] = "bde_predicted"
        labeled += 1
    metadata = dict(payload.get("metadata", {}))
    metadata["som_labeled"] = labeled
    metadata["som_validated"] = skipped
    metadata["som_failed"] = failed
    output = {"metadata": metadata, "drugs": drugs}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"Already labeled (validated/curated): {skipped}")
    print(f"Newly labeled (BDE): {labeled}")
    print(f"Failed: {failed}")
    print(f"\nSaved to {out}")
    return output


if __name__ == "__main__":
    add_som_labels("data/training_dataset_real.json", "data/training_dataset_real_labeled.json")
