from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES

CYP_ALIASES = {
    "CYP3A": "CYP3A4",
    "CYP3A4/5": "CYP3A4",
    "CYP3A5": "CYP3A4",
    "CYP2C": "CYP2C9",
    "CYP2D": "CYP2D6",
}


def normalize_cyp(cyp_str: str) -> Optional[str]:
    if not cyp_str:
        return None
    cyp = cyp_str.strip().upper()
    cyp = re.sub(r"^CYTOCHROME\s*P450\s*", "CYP", cyp)
    cyp = re.sub(r"^P450\s*", "CYP", cyp)
    if not cyp.startswith("CYP"):
        cyp = "CYP" + cyp
    cyp = CYP_ALIASES.get(cyp, cyp)
    if cyp in ALL_CYP_CLASSES:
        return cyp
    for valid_cyp in ALL_CYP_CLASSES:
        if valid_cyp in cyp or cyp in valid_cyp:
            return valid_cyp
    return None


def standardize_drugbank(
    input_path: str = "data/cyp_metabolism_dataset.json",
    output_path: str = "data/drugbank_standardized.json",
) -> List[Dict]:
    print("=" * 60)
    print("Standardizing DrugBank Data")
    print("=" * 60)

    with open(input_path) as f:
        data = json.load(f)
    raw_drugs = data if isinstance(data, list) else data.get("drugs", data.get("compounds", []))
    print(f"Raw drugs: {len(raw_drugs)}")

    standardized = []
    stats = defaultdict(int)
    for raw in raw_drugs:
        stats["total"] += 1
        smiles = raw.get("smiles") or raw.get("SMILES") or raw.get("canonical_smiles") or raw.get("structure")
        if not smiles:
            stats["no_smiles"] += 1
            continue
        if len(str(smiles)) > 400:
            stats["too_large"] += 1
            continue

        all_cyps: List[str] = []
        primary_cyp = None
        cyp_raw = raw.get("primary_cyp") or raw.get("cyp") or raw.get("enzyme") or raw.get("metabolizing_enzyme")
        if isinstance(cyp_raw, list):
            for c in cyp_raw:
                norm = normalize_cyp(str(c))
                if norm and norm not in all_cyps:
                    all_cyps.append(norm)
        elif cyp_raw:
            norm = normalize_cyp(str(cyp_raw))
            if norm:
                all_cyps.append(norm)
                primary_cyp = norm

        multi_cyp = raw.get("all_cyps") or raw.get("enzymes") or raw.get("cyps")
        if isinstance(multi_cyp, list):
            for c in multi_cyp:
                norm = normalize_cyp(str(c))
                if norm and norm not in all_cyps:
                    all_cyps.append(norm)

        if not primary_cyp and all_cyps:
            primary_cyp = all_cyps[0]
        if not primary_cyp:
            stats["no_cyp"] += 1
            continue

        name = raw.get("name") or raw.get("drug_name") or raw.get("title") or raw.get("drugbank_id") or "Unknown"
        metabolism_desc = raw.get("metabolism_description") or raw.get("metabolism") or raw.get("pathway") or raw.get("metabolism_text") or ""
        som = raw.get("som") or []
        site_atoms = raw.get("site_atoms") or raw.get("site_atom_indices")
        if not site_atoms and som:
            extracted = []
            for item in som:
                atom_idx = item.get("atom_idx", item) if isinstance(item, dict) else item
                if isinstance(atom_idx, int):
                    extracted.append(atom_idx)
            site_atoms = extracted
        bond_class = raw.get("expected_bond_class") or raw.get("bond_class")
        if not bond_class and som and isinstance(som[0], dict):
            bond_class = som[0].get("bond_class")

        drug = {
            "name": name,
            "smiles": smiles,
            "primary_cyp": primary_cyp,
            "all_cyps": list(dict.fromkeys(all_cyps)),
            "source": "DrugBank",
            "confidence": "validated",
        }
        if site_atoms:
            drug["site_atoms"] = list(site_atoms)
        if bond_class:
            drug["expected_bond_class"] = bond_class
        if metabolism_desc:
            drug["metabolism_description"] = metabolism_desc
        if som:
            drug["som"] = som
        for field in ["drugbank_id", "cas_number", "inchi", "inchikey", "id", "reactions"]:
            if raw.get(field):
                drug[field] = raw[field]
        standardized.append(drug)
        stats["success"] += 1

    print("\nProcessing Stats:")
    for stat, count in sorted(stats.items()):
        print(f"  {stat}: {count}")

    cyp_counts = defaultdict(int)
    som_count = 0
    for drug in standardized:
        cyp_counts[drug["primary_cyp"]] += 1
        if drug.get("site_atoms"):
            som_count += 1

    print(f"\nCYP Distribution ({len(standardized)} drugs):")
    for cyp in ALL_CYP_CLASSES:
        count = cyp_counts.get(cyp, 0)
        pct = count / len(standardized) * 100 if standardized else 0
        print(f"  {cyp}: {count} ({pct:.1f}%)")
    print(f"\nDrugs with SoM labels: {som_count} ({som_count / len(standardized) * 100 if standardized else 0:.1f}%)")

    output = {
        "metadata": {
            "source": "DrugBank",
            "total_drugs": len(standardized),
            "cyp_classes": len(cyp_counts),
            "cyp_distribution": dict(cyp_counts),
            "som_labeled": som_count,
        },
        "drugs": standardized,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    return standardized


if __name__ == "__main__":
    standardize_drugbank()
