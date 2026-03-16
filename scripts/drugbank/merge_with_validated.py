from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rdkit import Chem

from enzyme_software.liquid_nn_v2.data.curated_50_drugs import CURATED_50_DRUGS
from enzyme_software.liquid_nn_v2.data.training_drugs import TRAINING_DRUGS


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def get_validated_30() -> List[Dict]:
    drugs = []
    for drug in TRAINING_DRUGS:
        drugs.append(
            {
                "name": drug["name"],
                "smiles": drug["smiles"],
                "primary_cyp": drug["primary_cyp"],
                "site_atoms": list(drug["site_atom_indices"]),
                "expected_bond_class": drug["expected_bond_class"],
                "confidence": "validated_gold",
                "source": "manual_curation",
                "som_label_source": "validated",
            }
        )
    return drugs


def get_curated_50() -> List[Dict]:
    drugs = []
    for drug in CURATED_50_DRUGS:
        item = dict(drug)
        item["confidence"] = "validated_literature"
        item["source"] = "literature_curation"
        item["site_atoms"] = list(item.get("site_atoms") or item.get("site_atom_indices") or [])
        item["som_label_source"] = "curated"
        drugs.append(item)
    return drugs


def merge_all_drugs(
    drugbank_path: str = "data/drugbank_standardized.json",
    output_path: str = "data/training_dataset_drugbank.json",
) -> List[Dict]:
    print("=" * 60)
    print("Merging All Drug Sources")
    print("=" * 60)

    all_drugs = []
    seen = set()

    validated_30 = get_validated_30()
    for drug in validated_30:
        key = canonicalize(drug["smiles"])
        if key not in seen:
            seen.add(key)
            all_drugs.append(drug)
    print(f"Validated 30: added {len(validated_30)} drugs")

    added = 0
    for drug in get_curated_50():
        key = canonicalize(drug["smiles"])
        if key not in seen:
            seen.add(key)
            all_drugs.append(drug)
            added += 1
    print(f"Curated 50: added {added} drugs")

    with open(drugbank_path) as f:
        drugbank_data = json.load(f)
    drugbank_drugs = drugbank_data.get("drugs", [])
    added = 0
    for drug in drugbank_drugs:
        key = canonicalize(drug["smiles"])
        if key not in seen:
            seen.add(key)
            all_drugs.append(drug)
            added += 1
    print(f"DrugBank: added {added} drugs")
    print(f"\nTotal unique drugs: {len(all_drugs)}")

    by_cyp = defaultdict(int)
    by_source = defaultdict(int)
    by_confidence = defaultdict(int)
    som_count = 0
    for drug in all_drugs:
        by_cyp[drug["primary_cyp"]] += 1
        by_source[drug.get("source", "unknown")] += 1
        by_confidence[drug.get("confidence", "unknown")] += 1
        if drug.get("site_atoms"):
            som_count += 1

    print("\nCYP Distribution:")
    for cyp, count in sorted(by_cyp.items(), key=lambda x: -x[1]):
        print(f"  {cyp}: {count}")
    print("\nSource Distribution:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    print("\nConfidence Distribution:")
    for conf, count in sorted(by_confidence.items(), key=lambda x: -x[1]):
        print(f"  {conf}: {count}")
    print(f"\nDrugs with SoM labels: {som_count} ({som_count / len(all_drugs) * 100 if all_drugs else 0:.1f}%)")

    output = {
        "metadata": {
            "total_drugs": len(all_drugs),
            "cyp_distribution": dict(by_cyp),
            "source_distribution": dict(by_source),
            "confidence_distribution": dict(by_confidence),
            "som_labeled": som_count,
            "cyp_classes": len(by_cyp),
        },
        "drugs": all_drugs,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    return all_drugs


if __name__ == "__main__":
    merge_all_drugs()
