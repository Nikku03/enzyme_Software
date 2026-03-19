#!/usr/bin/env python
from __future__ import annotations

"""
Expand site-labeled drug metabolism data from the repo's local sources.

What this script can do locally:
- merge existing site-labeled DrugBank rows
- infer additional site labels from DrugBank reaction annotations
- add a curated literature panel
- optionally ingest XenoSite CSV exports
- optionally ingest MetXBioDB CSV/TSV exports

What it cannot do by itself:
- scrape MetXBioDB or paper supplements from the web
- guarantee 1000+ site-labeled drugs without extra external files
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD_DATASET = ROOT / "scripts" / "build_dataset"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BUILD_DATASET) not in sys.path:
    sys.path.insert(0, str(BUILD_DATASET))

from enzyme_software.calibration.drug_metabolism_db import DRUG_DATABASE
from identify_som import identify_som_from_metabolite, identify_som_from_reaction_type, label_som_indices


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_json_drugs(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _read_csv_dict_rows(path: Path, dialect=csv.excel) -> List[Dict[str, str]]:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle, dialect=dialect))
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, dialect=dialect))


def extract_site_indices(drug: Dict[str, object]) -> List[int]:
    indices: List[int] = []
    for key in ("site_atoms", "site_atom_indices", "metabolism_sites"):
        raw = drug.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, int):
                    indices.append(int(item))
                elif isinstance(item, dict):
                    atom_idx = item.get("atom_idx", item.get("atom_index", item.get("index", item.get("atom_idx"))))
                    if isinstance(atom_idx, int):
                        indices.append(int(atom_idx))
    raw_som = drug.get("som")
    if isinstance(raw_som, list):
        for item in raw_som:
            if isinstance(item, dict):
                atom_idx = item.get("atom_idx", item.get("atom_index", item.get("index")))
                if isinstance(atom_idx, int):
                    indices.append(int(atom_idx))
    return sorted(set(idx for idx in indices if idx >= 0))


def normalize_entry(
    drug: Dict[str, object],
    *,
    source: str,
    site_source: str,
    confidence: str,
    site_indices: Optional[List[int]] = None,
) -> Optional[Dict[str, object]]:
    smiles = str(drug.get("smiles", "")).strip()
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None

    indices = sorted(set(int(idx) for idx in (site_indices if site_indices is not None else extract_site_indices(drug)) if int(idx) >= 0))
    if not indices:
        return None

    primary_cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "").strip()
    all_cyps = drug.get("all_cyps") or [primary_cyp] if primary_cyp else []
    if not isinstance(all_cyps, list):
        all_cyps = [primary_cyp] if primary_cyp else []

    normalized = {
        "id": drug.get("id") or drug.get("drugbank_id"),
        "name": drug.get("name"),
        "smiles": canonical,
        "primary_cyp": primary_cyp,
        "all_cyps": list(dict.fromkeys(str(cyp) for cyp in all_cyps if cyp)),
        "reactions": list(drug.get("reactions") or []),
        "site_atoms": indices,
        "metabolism_sites": indices,
        "som": label_som_indices(canonical, indices),
        "source": source,
        "site_source": site_source,
        "confidence": confidence,
    }
    extra_keys = ("expected_bond_class", "site_type", "full_xtb_status")
    for key in extra_keys:
        if key in drug and drug.get(key) is not None:
            normalized[key] = drug.get(key)
    return normalized


class SiteDatasetExpander:
    def __init__(self, output_dir: str = "data/expanded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.by_smiles: Dict[str, Dict[str, object]] = {}

    def _priority(self, entry: Dict[str, object]) -> tuple[int, int]:
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        source_rank = {
            "literature": 4,
            "validated": 4,
            "drugbank_existing": 3,
            "combined_existing": 3,
            "xenosite": 3,
            "metxbiodb": 3,
            "drugbank_reaction_inference": 2,
            "metabolite_inference": 2,
        }
        return (
            confidence_rank.get(str(entry.get("confidence", "low")), 0),
            source_rank.get(str(entry.get("site_source", "")), 0),
        )

    def _merge(self, entry: Dict[str, object]) -> None:
        smiles = str(entry["smiles"])
        existing = self.by_smiles.get(smiles)
        if existing is None:
            self.by_smiles[smiles] = entry
            return

        merged_sites = sorted(set(existing.get("site_atoms", [])) | set(entry.get("site_atoms", [])))
        existing["site_atoms"] = merged_sites
        existing["metabolism_sites"] = merged_sites
        existing["som"] = label_som_indices(smiles, merged_sites)
        existing["all_cyps"] = sorted(set(existing.get("all_cyps", [])) | set(entry.get("all_cyps", [])))
        existing["reactions"] = sorted(set(existing.get("reactions", [])) | set(entry.get("reactions", [])))

        if self._priority(entry) > self._priority(existing):
            preserved_sites = existing["site_atoms"]
            self.by_smiles[smiles] = entry
            self.by_smiles[smiles]["site_atoms"] = preserved_sites
            self.by_smiles[smiles]["metabolism_sites"] = preserved_sites
            self.by_smiles[smiles]["som"] = label_som_indices(smiles, preserved_sites)
            self.by_smiles[smiles]["all_cyps"] = sorted(set(existing.get("all_cyps", [])) | set(entry.get("all_cyps", [])))
            self.by_smiles[smiles]["reactions"] = sorted(set(existing.get("reactions", [])) | set(entry.get("reactions", [])))

    def add_existing_site_labeled(self, path: str, *, source_label: str) -> None:
        dataset_path = Path(path)
        if not dataset_path.exists():
            return
        for drug in load_json_drugs(dataset_path):
            normalized = normalize_entry(
                drug,
                source=str(drug.get("source") or source_label),
                site_source=source_label,
                confidence="high" if extract_site_indices(drug) else "low",
            )
            if normalized is not None:
                self._merge(normalized)

    def expand_drugbank(self, drugbank_path: str) -> None:
        path = Path(drugbank_path)
        if not path.exists():
            raise FileNotFoundError(f"DrugBank JSON not found: {path}")
        added_existing = 0
        added_inferred = 0
        for drug in load_json_drugs(path):
            existing_sites = extract_site_indices(drug)
            if existing_sites:
                normalized = normalize_entry(
                    drug,
                    source=str(drug.get("source") or "DrugBank"),
                    site_source="drugbank_existing",
                    confidence="high",
                    site_indices=existing_sites,
                )
                if normalized is not None:
                    self._merge(normalized)
                    added_existing += 1
                continue

            inferred: List[int] = []
            for reaction in drug.get("reactions") or []:
                for atom_idx, _bond_class in identify_som_from_reaction_type(
                    str(drug.get("smiles", "")),
                    str(reaction),
                    str(drug.get("primary_cyp") or ""),
                ):
                    inferred.append(int(atom_idx))
            inferred = sorted(set(inferred))
            if inferred:
                normalized = normalize_entry(
                    drug,
                    source=str(drug.get("source") or "DrugBank"),
                    site_source="drugbank_reaction_inference",
                    confidence="low",
                    site_indices=inferred,
                )
                if normalized is not None:
                    self._merge(normalized)
                    added_inferred += 1
        print(f"DrugBank existing site-labeled merged: {added_existing}", flush=True)
        print(f"DrugBank reaction-inferred additions: {added_inferred}", flush=True)

    def infer_sites_from_metabolites(self, rows: Iterable[Dict[str, object]]) -> None:
        added = 0
        for drug in rows:
            if extract_site_indices(drug):
                continue
            smiles = str(drug.get("smiles", "")).strip()
            if not smiles:
                continue
            metabolites = drug.get("metabolites")
            if not isinstance(metabolites, list):
                continue
            inferred: List[int] = []
            for metabolite in metabolites:
                if isinstance(metabolite, dict):
                    met_smiles = str(metabolite.get("smiles", "")).strip()
                else:
                    met_smiles = ""
                if not met_smiles:
                    continue
                inferred.extend(identify_som_from_metabolite(smiles, met_smiles))
            inferred = sorted(set(inferred))
            if inferred:
                normalized = normalize_entry(
                    drug,
                    source=str(drug.get("source") or "unknown"),
                    site_source="metabolite_inference",
                    confidence="medium",
                    site_indices=inferred,
                )
                if normalized is not None:
                    self._merge(normalized)
                    added += 1
        print(f"Metabolite-inference additions: {added}", flush=True)

    def load_xenosite_dataset(self, path: str) -> None:
        dataset_path = Path(path)
        if not dataset_path.exists():
            print(f"XenoSite dataset not found: {dataset_path}", flush=True)
            return
        added = 0
        with dataset_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                smiles = str(row.get("SMILES") or row.get("smiles") or "").strip()
                if not smiles:
                    continue
                raw_sites = str(row.get("sites") or row.get("SOM") or "").strip().strip("[]")
                indices: List[int] = []
                for token in raw_sites.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        indices.append(int(token))
                    except ValueError:
                        continue
                normalized = normalize_entry(
                    {
                        "name": row.get("name"),
                        "smiles": smiles,
                        "primary_cyp": row.get("CYP") or row.get("cyp"),
                        "all_cyps": [row.get("CYP") or row.get("cyp")] if (row.get("CYP") or row.get("cyp")) else [],
                    },
                    source="xenosite",
                    site_source="xenosite",
                    confidence="medium",
                    site_indices=indices,
                )
                if normalized is not None:
                    self._merge(normalized)
                    added += 1
        print(f"Loaded XenoSite additions: {added}", flush=True)

    def load_metxbiodb_csv(self, path: str) -> None:
        dataset_path = Path(path)
        if not dataset_path.exists():
            print(f"MetXBioDB export not found: {dataset_path}", flush=True)
            return
        cid_to_smiles: Dict[str, str] = {}
        substances_path = dataset_path.with_name("MetXBioDB_substances.csv")
        if substances_path.exists():
            for row in _read_csv_dict_rows(substances_path):
                cid = str(row.get("PubChem_CID") or "").strip()
                smiles = str(row.get("SMILES") or "").strip()
                if cid and smiles:
                    cid_to_smiles[cid] = smiles
        dialect = csv.excel_tab if dataset_path.suffix.lower() in {".tsv", ".txt"} else csv.excel
        added = 0
        for row in _read_csv_dict_rows(dataset_path, dialect=dialect):
            substrate_cid = str(row.get("substrate_cid") or "").strip()
            product_cid = str(row.get("prod_cid") or row.get("product_cid") or "").strip()
            smiles = str(
                row.get("substrate_smiles")
                or row.get("smiles")
                or cid_to_smiles.get(substrate_cid, "")
                or ""
            ).strip()
            product_smiles = str(
                row.get("product_smiles")
                or row.get("metabolite_smiles")
                or cid_to_smiles.get(product_cid, "")
                or ""
            ).strip()
            if not smiles or not product_smiles:
                continue
            inferred = identify_som_from_metabolite(smiles, product_smiles)
            normalized = normalize_entry(
                {
                    "id": row.get("id") or row.get("reaction_id"),
                    "name": row.get("substrate_name") or row.get("name"),
                    "smiles": smiles,
                    "primary_cyp": row.get("cyp_isoform") or row.get("enzyme") or row.get("cyp"),
                    "all_cyps": [row.get("cyp_isoform") or row.get("enzyme") or row.get("cyp")] if (row.get("cyp_isoform") or row.get("enzyme") or row.get("cyp")) else [],
                    "reactions": [row.get("reaction_type")] if row.get("reaction_type") else [],
                },
                source="metxbiodb",
                site_source="metxbiodb",
                confidence="medium",
                site_indices=inferred,
            )
            if normalized is not None:
                self._merge(normalized)
                added += 1
        print(f"Loaded MetXBioDB additions: {added}", flush=True)

    def add_literature_compounds(self) -> None:
        literature_compounds = [
            {"name": "Caffeine", "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "primary_cyp": "CYP1A2", "site_atoms": [0, 7, 12], "site_type": "N-demethylation"},
            {"name": "Tolbutamide", "smiles": "Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCC", "primary_cyp": "CYP2C9", "site_atoms": [0], "site_type": "methyl hydroxylation"},
            {"name": "S-Mephenytoin", "smiles": "CCC1(NC(=O)N(C1=O)c2ccccc2)C", "primary_cyp": "CYP2C19", "site_atoms": [11, 12, 13], "site_type": "aromatic hydroxylation"},
            {"name": "Dextromethorphan", "smiles": "COc1ccc2c(c1)C3CC4C(C2)N(C)CC3C4", "primary_cyp": "CYP2D6", "site_atoms": [0], "site_type": "O-demethylation"},
            {"name": "Midazolam", "smiles": "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2", "primary_cyp": "CYP3A4", "site_atoms": [0], "site_type": "1-hydroxylation"},
            {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "primary_cyp": "CYP2C9", "site_atoms": [7], "site_type": "benzylic hydroxylation"},
            {"name": "Diclofenac", "smiles": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "primary_cyp": "CYP2C9", "site_atoms": [10, 11], "site_type": "aromatic hydroxylation"},
            {"name": "Omeprazole", "smiles": "COc1ccc2nc(Cc3ncc(C)c(OC)c3C)[nH]c2c1S(=O)C", "primary_cyp": "CYP2C19", "site_atoms": [19], "site_type": "sulfoxidation"},
            {"name": "Codeine", "smiles": "COc1ccc2C3CC4=CC=C(O)C5Oc1c2C35CCN4C", "primary_cyp": "CYP2D6", "site_atoms": [0], "site_type": "O-demethylation"},
            {"name": "Nifedipine", "smiles": "COC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC", "primary_cyp": "CYP3A4", "site_atoms": [7], "site_type": "oxidation to pyridine"},
        ]
        for drug in literature_compounds:
            normalized = normalize_entry(
                drug,
                source="literature",
                site_source="literature",
                confidence="high",
                site_indices=list(drug["site_atoms"]),
            )
            if normalized is not None:
                normalized["site_type"] = drug["site_type"]
                self._merge(normalized)

        for key, record in DRUG_DATABASE.items():
            inferred: List[int] = []
            metabolism_type = str(record.get("metabolism_type", ""))
            if "O-demethyl" in metabolism_type or "O-demethyl" in str(record.get("metabolism_site", "")):
                for atom_idx, _ in identify_som_from_reaction_type(str(record["smiles"]), "o_dealkylation", str(record.get("primary_cyp") or "")):
                    inferred.append(atom_idx)
            elif "N-demethyl" in metabolism_type or "N-dealkyl" in metabolism_type:
                for atom_idx, _ in identify_som_from_reaction_type(str(record["smiles"]), "n_dealkylation", str(record.get("primary_cyp") or "")):
                    inferred.append(atom_idx)
            elif "hydroxyl" in metabolism_type.lower():
                for atom_idx, _ in identify_som_from_reaction_type(str(record["smiles"]), "hydroxylation", str(record.get("primary_cyp") or "")):
                    inferred.append(atom_idx)
            if not inferred:
                continue
            normalized = normalize_entry(
                {
                    "id": record.get("drugbank_id"),
                    "name": record.get("name"),
                    "smiles": record.get("smiles"),
                    "primary_cyp": record.get("primary_cyp"),
                    "all_cyps": [record.get("primary_cyp")] if record.get("primary_cyp") else [],
                    "reactions": [record.get("metabolism_type")] if record.get("metabolism_type") else [],
                    "expected_bond_class": record.get("expected_bde_class"),
                },
                source="validated",
                site_source="validated",
                confidence="high",
                site_indices=inferred[:3],
            )
            if normalized is not None:
                self._merge(normalized)
        print("Added literature and validated compounds", flush=True)

    def summary(self) -> Dict[str, object]:
        rows = list(self.by_smiles.values())
        source_counts = Counter(str(row.get("source", "unknown")) for row in rows)
        site_source_counts = Counter(str(row.get("site_source", "unknown")) for row in rows)
        cyp_counts = Counter(str(row.get("primary_cyp", "unknown")) for row in rows if row.get("primary_cyp"))
        return {
            "n_drugs": len(rows),
            "n_site_labeled": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "site_source_counts": dict(sorted(site_source_counts.items())),
            "cyp_counts": dict(sorted(cyp_counts.items())),
        }

    def save(self, filename: str = "expanded_site_labeled.json") -> Path:
        rows = sorted(self.by_smiles.values(), key=lambda item: (str(item.get("source", "")), str(item.get("name", ""))))
        output = {"metadata": self.summary(), "drugs": rows}
        out_path = self.output_dir / filename
        out_path.write_text(json.dumps(output, indent=2))
        return out_path

    def run_all(
        self,
        *,
        drugbank_json: str,
        combined_site_labeled_json: Optional[str] = None,
        xenosite_csv: Optional[str] = None,
        metxbiodb_csv: Optional[str] = None,
    ) -> Path:
        print("=" * 60, flush=True)
        print("EXPANDING SITE-LABELED DATASET", flush=True)
        print("=" * 60, flush=True)
        self.add_literature_compounds()
        if combined_site_labeled_json:
            self.add_existing_site_labeled(combined_site_labeled_json, source_label="combined_existing")
        self.expand_drugbank(drugbank_json)
        if xenosite_csv:
            self.load_xenosite_dataset(xenosite_csv)
        if metxbiodb_csv:
            self.load_metxbiodb_csv(metxbiodb_csv)
        out_path = self.save()
        summary = self.summary()
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Output: {out_path}", flush=True)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand site-labeled dataset from local JSON/CSV sources")
    parser.add_argument("--drugbank", default="data/drugbank_standardized.json")
    parser.add_argument("--combined-site-labeled", default="data/combined_drugbank_supercyp_full_xtb_valid_site_labeled.json")
    parser.add_argument("--xenosite", default=None)
    parser.add_argument("--metxbiodb", default=None)
    parser.add_argument("--output-dir", default="data/expanded")
    args = parser.parse_args()

    expander = SiteDatasetExpander(output_dir=args.output_dir)
    expander.run_all(
        drugbank_json=args.drugbank,
        combined_site_labeled_json=args.combined_site_labeled,
        xenosite_csv=args.xenosite,
        metxbiodb_csv=args.metxbiodb,
    )


if __name__ == "__main__":
    main()
