from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem

from enzyme_software.liquid_nn_v2.data.training_drugs import TRAINING_DRUGS


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def load_validated_dataset(path: str | None = None) -> List[Dict[str, object]]:
    if path and Path(path).exists():
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, dict):
            drugs = payload.get("drugs", payload)
        else:
            drugs = payload
        return list(drugs)
    validated = []
    for item in TRAINING_DRUGS:
        validated.append(
            {
                "name": item["name"],
                "smiles": item["smiles"],
                "primary_cyp": item["primary_cyp"],
                "cyp": item["primary_cyp"],
                "som": [
                    {
                        "atom_idx": int(idx),
                        "bond_class": item["expected_bond_class"],
                        "confidence": 1.0,
                    }
                    for idx in item["site_atom_indices"]
                ],
                "confidence": "high",
                "source": "validated",
            }
        )
    return validated


def merge_datasets(
    validated_path: str | None = None,
    pseudo_path: str = "data/500_drugs_with_som.json",
    output_path: str = "data/training_dataset_530.json",
) -> List[Dict[str, object]]:
    validated = load_validated_dataset(validated_path)
    pseudo = json.loads(Path(pseudo_path).read_text())

    print(f"Validated drugs: {len(validated)}")
    print(f"Pseudo-labeled drugs: {len(pseudo)}")

    for drug in validated:
        drug["confidence"] = "high"
        drug["source"] = "validated"

    for drug in pseudo:
        top_score = drug.get("som", [{}])[0].get("confidence", 0.0) if drug.get("som") else 0.0
        drug["confidence"] = "medium" if top_score > 0.7 else "low"
        drug["source"] = "pseudo"

    validated_smiles = {_canonical_smiles(str(d["smiles"])): d for d in validated}
    merged = list(validated)
    for drug in pseudo:
        smiles = _canonical_smiles(str(drug["smiles"]))
        if smiles not in validated_smiles:
            merged.append(drug)

    print(f"Merged total: {len(merged)}")
    by_cyp: Dict[str, int] = {}
    by_confidence = {"high": 0, "medium": 0, "low": 0}
    for drug in merged:
        cyp = str(drug.get("cyp") or drug.get("primary_cyp"))
        by_cyp[cyp] = by_cyp.get(cyp, 0) + 1
        by_confidence[str(drug.get("confidence", "low"))] = by_confidence.get(str(drug.get("confidence", "low")), 0) + 1

    print("\nCYP Distribution:")
    for cyp, count in sorted(by_cyp.items()):
        print(f"  {cyp}: {count}")
    print("\nConfidence Distribution:")
    for conf, count in by_confidence.items():
        print(f"  {conf}: {count}")

    output = {
        "metadata": {
            "version": "1.0",
            "total_drugs": len(merged),
            "validated_count": len(validated),
            "pseudo_count": len(merged) - len(validated),
            "cyp_distribution": by_cyp,
            "confidence_distribution": by_confidence,
        },
        "drugs": merged,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out}")
    return merged


if __name__ == "__main__":
    merge_datasets()
