from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from enzyme_software.liquid_nn_v2.data.curated_50_drugs import CURATED_50_DRUGS
from enzyme_software.liquid_nn_v2.data.drug_database import DRUG_DATABASE

MAJOR_CYPS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


def get_validated_drugs() -> List[Dict[str, object]]:
    drugs: List[Dict[str, object]] = []
    for key, data in DRUG_DATABASE.items():
        drugs.append(
            {
                "name": str(data.get("name", key)),
                "smiles": str(data["smiles"]),
                "primary_cyp": str(data["primary_cyp"]),
                "site_atoms": list(data.get("site_atoms", [])),
                "expected_bond_class": str(data.get("expected_bond_class", "other")),
                "confidence": "validated",
                "source": "manual_curation",
                "cyp_label_source": "manual_curation",
                "som_label_source": "validated",
            }
        )
    return drugs


def _curated_drugs() -> List[Dict[str, object]]:
    enriched = []
    for drug in CURATED_50_DRUGS:
        item = dict(drug)
        item.setdefault("confidence", "curated")
        item.setdefault("source", "curated")
        item.setdefault("cyp_label_source", "curated")
        item.setdefault("som_label_source", "curated")
        if "site_atoms" not in item and item.get("site_atom_indices"):
            item["site_atoms"] = list(item["site_atom_indices"])
        enriched.append(item)
    return enriched


def load_json(path: str) -> List[Dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    return list(payload.get("drugs", payload))


def merge_all_sources(output_path: str = "data/training_dataset_real.json") -> List[Dict[str, object]]:
    print("=" * 60)
    print("Merging All CYP Data Sources")
    print("=" * 60)
    all_drugs: List[Dict[str, object]] = []
    seen_smiles = set()
    sources = [
        ("Validated 30", get_validated_drugs),
        ("Curated 50", _curated_drugs),
        ("SuperCYP", lambda: load_json("data/supercyp_drugs.json")),
        ("GtoPdb", lambda: load_json("data/gtopdb_cyp_drugs.json")),
        ("DrugCentral", lambda: load_json("data/drugcentral_cyp_drugs.json")),
    ]
    for source_name, loader in sources:
        try:
            drugs = loader()
        except Exception as exc:
            print(f"{source_name}: Error - {exc}")
            continue
        added = 0
        for drug in drugs:
            smiles = str(drug.get("smiles", "")).strip()
            if not smiles or smiles in seen_smiles:
                continue
            cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "")
            if cyp not in MAJOR_CYPS:
                continue
            seen_smiles.add(smiles)
            item = dict(drug)
            item["primary_cyp"] = cyp
            all_drugs.append(item)
            added += 1
        print(f"{source_name}: added {added} new drugs")
    by_cyp = defaultdict(int)
    by_source = defaultdict(int)
    by_confidence = defaultdict(int)
    for drug in all_drugs:
        by_cyp[str(drug.get("primary_cyp"))] += 1
        by_source[str(drug.get("source", "unknown"))] += 1
        by_confidence[str(drug.get("confidence", "unknown"))] += 1
    print(f"\nTotal unique drugs: {len(all_drugs)}")
    print("\nCYP Distribution:")
    for cyp in MAJOR_CYPS:
        print(f"  {cyp}: {by_cyp[cyp]}")
    print("\nSource Distribution:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")
    print("\nConfidence Distribution:")
    for conf, count in sorted(by_confidence.items()):
        print(f"  {conf}: {count}")
    payload = {
        "metadata": {
            "total_drugs": len(all_drugs),
            "cyp_distribution": dict(by_cyp),
            "source_distribution": dict(by_source),
            "confidence_distribution": dict(by_confidence),
            "sources": ["validated_30", "curated_50", "SuperCYP", "GtoPdb", "DrugCentral"],
        },
        "drugs": all_drugs,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out}")
    return all_drugs


if __name__ == "__main__":
    merge_all_sources()
