from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2.data.drug_database import DRUG_DATABASE

try:
    from .identify_som import identify_som_from_metabolite, identify_som_from_reaction_type, label_som_indices
    from .parse_drugbank import SUPPORTED_CYP_ISOFORMS, parse_drugbank_xml
except ImportError:  # pragma: no cover - direct script execution
    from identify_som import identify_som_from_metabolite, identify_som_from_reaction_type, label_som_indices
    from parse_drugbank import SUPPORTED_CYP_ISOFORMS, parse_drugbank_xml


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def parse_metxbiodb_csv(csv_path: str | Path) -> List[Dict[str, object]]:
    """
    Parse a flexible MetXBioDB-like CSV/TSV export.

    Expected loose columns:
    - substrate_smiles
    - product_smiles
    - cyp_isoform / enzyme
    - reaction_type
    - substrate_name / name
    """
    path = Path(csv_path)
    dialect = csv.excel_tab if path.suffix.lower() in {".tsv", ".txt"} else csv.excel
    records: List[Dict[str, object]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            cyp = (
                row.get("cyp_isoform")
                or row.get("enzyme")
                or row.get("cyp")
                or row.get("isoform")
                or ""
            ).strip().upper()
            if cyp and not cyp.startswith("CYP"):
                cyp = f"CYP{cyp}"
            if cyp not in SUPPORTED_CYP_ISOFORMS:
                continue
            substrate_smiles = (row.get("substrate_smiles") or row.get("smiles") or "").strip()
            product_smiles = (row.get("product_smiles") or row.get("metabolite_smiles") or "").strip()
            if not substrate_smiles:
                continue
            records.append(
                {
                    "id": row.get("id") or row.get("reaction_id"),
                    "name": row.get("substrate_name") or row.get("name"),
                    "smiles": substrate_smiles,
                    "product_smiles": product_smiles,
                    "primary_cyp": cyp,
                    "all_cyps": [cyp],
                    "reactions": [row.get("reaction_type")] if row.get("reaction_type") else [],
                    "source": "metxbiodb",
                }
            )
    return records


def get_validated_drugs() -> List[Dict[str, object]]:
    """Return the repo's curated high-confidence validation set."""
    validated: List[Dict[str, object]] = []
    for idx, entry in enumerate(DRUG_DATABASE.values(), start=1):
        validated.append(
            {
                "id": f"VAL{idx:03d}",
                "name": entry["name"],
                "smiles": entry["smiles"],
                "primary_cyp": entry["primary_cyp"],
                "all_cyps": [entry["primary_cyp"]],
                "reactions": ["validated"],
                "som": [
                    {"atom_idx": int(atom_idx), "bond_class": str(entry["expected_bond_class"])}
                    for atom_idx in entry["site_atoms"]
                ],
                "source": "validated",
                "confidence": "high",
            }
        )
    return validated


def _build_drugbank_entries(drugbank_drugs: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    final_dataset: List[Dict[str, object]] = []
    for drug in drugbank_drugs:
        enzymes = [c for c in drug.get("enzymes", []) if c in SUPPORTED_CYP_ISOFORMS]
        if not enzymes:
            continue
        primary_cyp = enzymes[0]
        som_info: List[Dict[str, object]] = []
        for reaction in drug.get("reactions", []):
            for atom_idx, bond_class in identify_som_from_reaction_type(str(drug["smiles"]), str(reaction), primary_cyp):
                if all(existing["atom_idx"] != atom_idx for existing in som_info):
                    som_info.append({"atom_idx": atom_idx, "bond_class": bond_class})
        final_dataset.append(
            {
                "id": drug.get("drugbank_id"),
                "name": drug.get("name"),
                "smiles": drug["smiles"],
                "primary_cyp": primary_cyp,
                "all_cyps": enzymes,
                "reactions": list(drug.get("reactions", [])),
                "som": som_info,
                "source": "drugbank",
                "confidence": "low" if not som_info else "medium",
            }
        )
    return final_dataset


def _merge_metxbiodb_entries(final_dataset: List[Dict[str, object]], records: Iterable[Dict[str, object]]) -> None:
    by_smiles = {canonicalize_smiles(str(entry["smiles"])): entry for entry in final_dataset if entry.get("smiles")}
    for record in records:
        canonical = canonicalize_smiles(str(record["smiles"]))
        if canonical is None:
            continue
        som_indices = identify_som_from_metabolite(str(record["smiles"]), str(record.get("product_smiles") or ""))
        som = label_som_indices(str(record["smiles"]), som_indices)
        existing = by_smiles.get(canonical)
        if existing is not None:
            existing["all_cyps"] = sorted(set(existing.get("all_cyps", [])) | set(record.get("all_cyps", [])))
            if existing.get("confidence") != "high":
                existing["primary_cyp"] = record.get("primary_cyp") or existing.get("primary_cyp")
                if som:
                    existing["som"] = som
                    existing["confidence"] = "medium"
            continue
        entry = {
            "id": record.get("id"),
            "name": record.get("name"),
            "smiles": record["smiles"],
            "primary_cyp": record.get("primary_cyp"),
            "all_cyps": list(record.get("all_cyps", [])),
            "reactions": list(record.get("reactions", [])),
            "som": som,
            "source": "metxbiodb",
            "confidence": "medium" if som else "low",
        }
        final_dataset.append(entry)
        by_smiles[canonical] = entry


def build_dataset(
    drugbank_xml: str,
    metxbiodb_csv: str | None = None,
    output_path: str | Path = "data/cyp_metabolism_dataset.json",
) -> Dict[str, object]:
    """
    Build a combined CYP metabolism dataset from DrugBank, optional MetXBioDB CSV, and validated drugs.
    """
    print("Parsing DrugBank...")
    drugbank_drugs = parse_drugbank_xml(drugbank_xml)
    print(f"  Found {len(drugbank_drugs)} drugs with CYP annotations")

    final_dataset = _build_drugbank_entries(drugbank_drugs)

    if metxbiodb_csv:
        print("Parsing MetXBioDB...")
        metx_records = parse_metxbiodb_csv(metxbiodb_csv)
        print(f"  Found {len(metx_records)} reaction records")
        _merge_metxbiodb_entries(final_dataset, metx_records)

    validated = get_validated_drugs()
    by_smiles = {canonicalize_smiles(str(entry["smiles"])): entry for entry in final_dataset if entry.get("smiles")}
    for drug in validated:
        canonical = canonicalize_smiles(str(drug["smiles"]))
        existing = by_smiles.get(canonical)
        if existing is not None:
            existing["som"] = drug["som"]
            existing["confidence"] = "high"
            existing["source"] = "validated"
            existing["all_cyps"] = sorted(set(existing.get("all_cyps", [])) | set(drug.get("all_cyps", [])))
            existing["primary_cyp"] = drug["primary_cyp"]
        else:
            final_dataset.append(drug)
            by_smiles[canonical] = drug

    cyp_distribution = Counter(entry.get("primary_cyp") for entry in final_dataset if entry.get("primary_cyp"))
    confidence_distribution = Counter(entry.get("confidence") for entry in final_dataset)

    output = {
        "metadata": {
            "version": "1.0",
            "total_drugs": len(final_dataset),
            "sources": sorted({str(entry.get("source")) for entry in final_dataset}),
            "cyp_distribution": dict(sorted(cyp_distribution.items())),
            "confidence_distribution": dict(sorted(confidence_distribution.items())),
        },
        "drugs": final_dataset,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2))

    print("\nDrugs per CYP:")
    for cyp, count in sorted(cyp_distribution.items()):
        print(f"  {cyp}: {count}")
    print(f"\nSaved {len(final_dataset)} drugs to {output_file}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CYP metabolism training dataset.")
    parser.add_argument("drugbank_xml", help="Path to DrugBank XML")
    parser.add_argument("--metxbiodb-csv", help="Optional MetXBioDB CSV/TSV export")
    parser.add_argument("--output", default="data/cyp_metabolism_dataset.json", help="Output dataset path")
    args = parser.parse_args()
    build_dataset(args.drugbank_xml, metxbiodb_csv=args.metxbiodb_csv, output_path=args.output)


if __name__ == "__main__":
    main()
