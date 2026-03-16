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

from enzyme_software.liquid_nn_v2.data.bde_table import BDE_TABLE
from enzyme_software.liquid_nn_v2.features.bond_classifier import classify_bond


def get_bond_class_bde(bond_class: str) -> float:
    return float(BDE_TABLE.get(bond_class, BDE_TABLE["other"]))


def get_cyp_modifier(bond_class: str, cyp: str) -> float:
    cyp_preferences = {
        "CYP3A4": {"benzylic": 1.2, "allylic": 1.3, "tertiary_CH": 1.1, "alpha_hetero": 1.0, "aryl": 0.8},
        "CYP2D6": {"alpha_hetero": 1.4, "benzylic": 1.0, "aryl": 0.9},
        "CYP2C9": {"benzylic": 1.2, "aryl": 1.1, "alpha_hetero": 0.9},
        "CYP2C19": {"benzylic": 1.1, "alpha_hetero": 1.2},
        "CYP1A2": {"aryl": 1.3, "alpha_hetero": 1.1},
    }
    return float(cyp_preferences.get(cyp, {}).get(bond_class, 1.0))


def generate_som_for_drug(smiles: str, primary_cyp: str) -> List[Dict[str, object]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    atom_scores: List[Dict[str, object]] = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() not in {"C", "N", "S"}:
            continue
        bond_class = classify_bond(atom, mol)
        bde = get_bond_class_bde(bond_class)
        bde_min, bde_max = 360.0, 480.0
        reactivity = 1.0 - (bde - bde_min) / (bde_max - bde_min)
        reactivity = max(0.0, min(1.0, reactivity))
        cyp_modifier = get_cyp_modifier(bond_class, primary_cyp)
        final_score = reactivity * cyp_modifier
        atom_scores.append(
            {
                "atom_idx": atom_idx,
                "bond_class": bond_class,
                "bde": bde,
                "reactivity": reactivity,
                "cyp_modifier": cyp_modifier,
                "score": final_score,
            }
        )
    atom_scores.sort(key=lambda item: item["score"], reverse=True)
    return atom_scores[:3]


def generate_all_som_labels(
    input_path: str = "data/500_drugs_raw.json",
    output_path: str = "data/500_drugs_with_som.json",
) -> List[Dict[str, object]]:
    drugs = json.loads(Path(input_path).read_text())
    print(f"Processing {len(drugs)} drugs...")
    labeled_drugs: List[Dict[str, object]] = []
    failed = 0
    for i, drug in enumerate(drugs, start=1):
        smiles = drug.get("smiles")
        cyp = drug.get("cyp") or drug.get("primary_cyp")
        if not smiles or not cyp:
            failed += 1
            continue
        som_predictions = generate_som_for_drug(str(smiles), str(cyp))
        if not som_predictions:
            failed += 1
            continue
        enriched = dict(drug)
        enriched["som"] = [
            {
                "atom_idx": pred["atom_idx"],
                "bond_class": pred["bond_class"],
                "confidence": pred["score"],
            }
            for pred in som_predictions
        ]
        enriched["primary_som"] = som_predictions[0]["atom_idx"]
        enriched["primary_bond_class"] = som_predictions[0]["bond_class"]
        labeled_drugs.append(enriched)
        if i % 100 == 0:
            print(f"  Processed {i}/{len(drugs)}")
    print(f"\nSuccessfully labeled: {len(labeled_drugs)}")
    print(f"Failed: {failed}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labeled_drugs, indent=2))
    print(f"Saved to {out}")
    return labeled_drugs


if __name__ == "__main__":
    generate_all_som_labels()
