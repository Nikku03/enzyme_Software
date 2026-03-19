#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD_DATASET = ROOT / "scripts" / "build_dataset"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BUILD_DATASET) not in sys.path:
    sys.path.insert(0, str(BUILD_DATASET))

from identify_som import label_som_indices


MAIN5 = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
BOM_TO_CYP = {
    "BOM_1A2": "CYP1A2",
    "BOM_2C9": "CYP2C9",
    "BOM_2C19": "CYP2C19",
    "BOM_2D6": "CYP2D6",
    "BOM_3A4": "CYP3A4",
}
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_mol_with_mapping(mol: Chem.Mol) -> Tuple[Optional[str], Optional[List[int]]]:
    base = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(base, canonical=True)
    rebuilt = Chem.MolFromSmiles(canonical)
    if rebuilt is None:
        return None, None
    match = base.GetSubstructMatch(rebuilt)
    if match:
        return canonical, list(match)
    reverse = rebuilt.GetSubstructMatch(base)
    if reverse:
        inverted: List[int] = [0] * len(reverse)
        for rebuilt_idx, base_idx in enumerate(reverse):
            inverted[int(base_idx)] = int(rebuilt_idx)
        return canonical, inverted
    return canonical, None


def canonicalize_smiles_with_mapping(smiles: str) -> Tuple[Optional[str], Optional[List[int]]]:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return None, None
    return canonicalize_mol_with_mapping(mol)


def remap_indices(indices: List[int], mapping: Optional[List[int]]) -> List[int]:
    if not indices:
        return []
    if not mapping:
        return sorted(set(int(idx) for idx in indices if int(idx) >= 0))
    remapped = []
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(mapping):
            remapped.append(int(mapping[idx]))
    return sorted(set(remapped))


def load_json_drugs(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def pick_confidence(current: Optional[str], incoming: Optional[str]) -> str:
    cur = str(current or "low")
    inc = str(incoming or "low")
    return inc if CONFIDENCE_RANK.get(inc, 0) >= CONFIDENCE_RANK.get(cur, 0) else cur


def extract_site_indices(drug: Dict[str, object]) -> List[int]:
    indices: List[int] = []
    for key in ("site_atoms", "site_atom_indices", "metabolism_sites"):
        raw = drug.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, int):
                    indices.append(int(item))
                elif isinstance(item, dict):
                    atom_idx = item.get("atom_idx", item.get("atom_index", item.get("index")))
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


def normalize_entry(drug: Dict[str, object]) -> Optional[Dict[str, object]]:
    smiles, mapping = canonicalize_smiles_with_mapping(str(drug.get("smiles", "")).strip())
    if smiles is None:
        return None
    primary_cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "").strip()
    if primary_cyp not in MAIN5:
        return None
    all_cyps = [str(c) for c in (drug.get("all_cyps") or [primary_cyp]) if str(c) in MAIN5]
    if primary_cyp not in all_cyps:
        all_cyps.append(primary_cyp)
    site_atoms = remap_indices(extract_site_indices(drug), mapping)
    normalized = {
        "id": drug.get("id") or drug.get("drugbank_id"),
        "name": drug.get("name"),
        "smiles": smiles,
        "primary_cyp": primary_cyp,
        "all_cyps": sorted(set(all_cyps), key=MAIN5.index),
        "reactions": list(drug.get("reactions") or []),
        "site_atoms": site_atoms,
        "metabolism_sites": list(site_atoms),
        "source": drug.get("source", "unknown"),
        "site_source": drug.get("site_source", drug.get("source", "unknown")),
        "confidence": str(drug.get("confidence") or "medium"),
        "full_xtb_status": drug.get("full_xtb_status", "unknown"),
    }
    if site_atoms:
        normalized["som"] = label_som_indices(smiles, site_atoms)
    else:
        normalized["som"] = []
    for key in ("expected_bond_class", "site_type"):
        if drug.get(key) is not None:
            normalized[key] = drug.get(key)
    return normalized


def merge_entries(entries: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for raw in entries:
        entry = normalize_entry(raw)
        if entry is None:
            continue
        key = (str(entry["smiles"]), str(entry["primary_cyp"]))
        existing = by_key.get(key)
        if existing is None:
            entry["source_details"] = [str(entry.get("source", "unknown"))]
            by_key[key] = entry
            continue
        if not existing.get("id") and entry.get("id"):
            existing["id"] = entry["id"]
        if not existing.get("name") and entry.get("name"):
            existing["name"] = entry["name"]
        existing["all_cyps"] = sorted(set(existing.get("all_cyps", []) + entry.get("all_cyps", [])), key=MAIN5.index)
        existing["reactions"] = sorted(set(existing.get("reactions", []) + entry.get("reactions", [])))
        merged_sites = sorted(set(extract_site_indices(existing) + extract_site_indices(entry)))
        existing["site_atoms"] = merged_sites
        existing["metabolism_sites"] = list(merged_sites)
        existing["som"] = label_som_indices(str(existing["smiles"]), merged_sites) if merged_sites else []
        existing["confidence"] = pick_confidence(existing.get("confidence"), entry.get("confidence"))
        existing["source_details"] = sorted(set(existing.get("source_details", []) + [str(entry.get("source", "unknown"))]))
        if existing.get("source") == "unknown" and entry.get("source"):
            existing["source"] = entry["source"]
        if existing.get("site_source") == "unknown" and entry.get("site_source"):
            existing["site_source"] = entry["site_source"]
        if existing.get("full_xtb_status") in {None, "unknown"} and entry.get("full_xtb_status"):
            existing["full_xtb_status"] = entry["full_xtb_status"]
    return sorted(by_key.values(), key=lambda row: (MAIN5.index(str(row["primary_cyp"])), str(row["name"] or row["smiles"])))


def parse_bom_annotation(raw: str) -> List[Dict[str, object]]:
    parsed: List[Dict[str, object]] = []
    for line in str(raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"<([^>]+)>", line)
        if not match:
            continue
        parts = [part.strip() for part in match.group(1).split(";")]
        if len(parts) < 2:
            continue
        locus = parts[0]
        reaction = parts[1]
        locus_parts = [token.strip() for token in locus.split(",") if token.strip()]
        if not locus_parts:
            continue
        if len(locus_parts) >= 2 and locus_parts[1].isdigit():
            atoms = []
            for token in locus_parts[:2]:
                if token.isdigit():
                    atoms.append(int(token) - 1)
            if len(atoms) == 2 and min(atoms) >= 0:
                parsed.append({"kind": "bond", "atoms": atoms, "reaction": reaction})
            continue
        if locus_parts[0].isdigit():
            atom_idx = int(locus_parts[0]) - 1
            if atom_idx >= 0:
                parsed.append(
                    {
                        "kind": "atom",
                        "atom": atom_idx,
                        "reaction": reaction,
                        "token": locus_parts[1] if len(locus_parts) > 1 else None,
                    }
                )
    return parsed


def extract_metxbio_entries(path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    conservative: List[Dict[str, object]] = []
    permissive: List[Dict[str, object]] = []
    summary = Counter()
    for mol in supplier:
        if mol is None:
            summary["invalid_molecules"] += 1
            continue
        smiles, mapping = canonicalize_mol_with_mapping(mol)
        if smiles is None or mapping is None:
            summary["invalid_smiles"] += 1
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        inchikey = mol.GetProp("InChIKey") if mol.HasProp("InChIKey") else None
        pubchem_cid = mol.GetProp("PubChem_CID") if mol.HasProp("PubChem_CID") else None
        for prop, cyp in BOM_TO_CYP.items():
            raw = mol.GetProp(prop) if mol.HasProp(prop) else ""
            parsed = parse_bom_annotation(raw)
            if not parsed:
                continue
            summary["bom_rows"] += 1
            atom_sites = remap_indices(
                sorted({int(item["atom"]) for item in parsed if item["kind"] == "atom"}),
                mapping,
            )
            bond_sites = remap_indices(
                sorted({int(atom) for item in parsed if item["kind"] == "bond" for atom in item["atoms"]}),
                mapping,
            )
            reactions = sorted(set(str(item["reaction"]) for item in parsed))
            base = {
                "id": f"metxbiodb:{inchikey or pubchem_cid or name}:{cyp}",
                "name": name,
                "smiles": smiles,
                "primary_cyp": cyp,
                "all_cyps": [cyp],
                "reactions": reactions,
                "source": "MetXBioDB",
                "confidence": "medium",
                "full_xtb_status": "external_uncomputed",
            }
            if atom_sites:
                conservative.append(
                    {
                        **base,
                        "site_atoms": atom_sites,
                        "site_source": "metxbiodb_atom_only",
                    }
                )
                summary["conservative_rows"] += 1
            permissive_sites = sorted(set(atom_sites + bond_sites))
            if permissive_sites:
                permissive.append(
                    {
                        **base,
                        "site_atoms": permissive_sites,
                        "site_source": "metxbiodb_atom_plus_bond_endpoints",
                    }
                )
                summary["permissive_rows"] += 1
            if atom_sites:
                summary["atom_annotated_rows"] += 1
            elif bond_sites:
                summary["bond_only_rows"] += 1
    return conservative, permissive, dict(summary)


def extract_metpred_entries(doc_path: Path, drugbank_path: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    text = subprocess.check_output(["textutil", "-convert", "txt", "-stdout", str(doc_path)]).decode("utf-8", "replace")
    pairs = re.findall(r"(DB\d{5})\s+\n?\s*(CYP[0-9A-Z]+)", text)
    drugbank_map: Dict[str, Dict[str, object]] = {}
    for drug in load_json_drugs(drugbank_path):
        key = str(drug.get("id") or drug.get("drugbank_id") or "").strip()
        if key.startswith("DB"):
            drugbank_map[key] = drug
    rows: List[Dict[str, object]] = []
    summary = Counter()
    for dbid, cyp in pairs:
        if cyp not in MAIN5:
            continue
        source = drugbank_map.get(dbid)
        if source is None:
            summary["missing_drugbank_match"] += 1
            continue
        smiles = canonicalize_smiles(str(source.get("smiles", "")))
        if smiles is None:
            summary["invalid_smiles"] += 1
            continue
        rows.append(
            {
                "id": dbid,
                "name": source.get("name"),
                "smiles": smiles,
                "primary_cyp": cyp,
                "all_cyps": [cyp],
                "reactions": list(source.get("reactions") or []),
                "site_atoms": [],
                "source": "MetaPred",
                "site_source": "none",
                "confidence": "medium",
                "full_xtb_status": source.get("full_xtb_status", "external_uncomputed"),
            }
        )
        summary["resolved_rows"] += 1
    return rows, dict(summary)


def write_dataset(path: Path, rows: List[Dict[str, object]], metadata: Dict[str, object]) -> None:
    payload = {
        "version": "1.0",
        "count": len(rows),
        "drugs": rows,
        "metadata": metadata,
    }
    path.write_text(json.dumps(payload, indent=2))


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "count": len(rows),
        "site_labeled": sum(1 for row in rows if extract_site_indices(row)),
        "source_counts": dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items())),
        "site_source_counts": dict(sorted(Counter(str(row.get("site_source", "unknown")) for row in rows).items())),
        "cyp_counts": dict(sorted(Counter(str(row.get("primary_cyp", "unknown")) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged training datasets for CYP/site models")
    parser.add_argument("--combined", default="data/combined_drugbank_supercyp_full_xtb_valid.json")
    parser.add_argument("--drugbank", default="data/drugbank_standardized.json")
    parser.add_argument("--metxbio-sdf", default="metxbio data.cleaned.sdf")
    parser.add_argument("--metpred-doc", default="SupplemantryData.doc")
    parser.add_argument("--xenosite-manifest", default="data/xenosite_suppl/manifest.json")
    parser.add_argument("--output-dir", default="data/prepared_training")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = merge_entries(load_json_drugs(ROOT / args.combined))
    metxbio_conservative, metxbio_permissive, metxbio_summary = extract_metxbio_entries(ROOT / args.metxbio_sdf)
    metpred_rows, metpred_summary = extract_metpred_entries(ROOT / args.metpred_doc, ROOT / args.drugbank)

    conservative_merged = merge_entries(combined_rows + metpred_rows + metxbio_conservative)
    permissive_merged = merge_entries(combined_rows + metpred_rows + metxbio_permissive)
    conservative_site_only = [row for row in conservative_merged if extract_site_indices(row)]
    permissive_site_only = [row for row in permissive_merged if extract_site_indices(row)]

    outputs = {
        "main5_all_models_conservative": conservative_merged,
        "main5_all_models_permissive": permissive_merged,
        "main5_site_conservative": conservative_site_only,
        "main5_site_permissive": permissive_site_only,
        "metpred_cyp_only": merge_entries(metpred_rows),
        "metxbio_main5_atom_only": merge_entries(metxbio_conservative),
        "metxbio_main5_with_bond_endpoints": merge_entries(metxbio_permissive),
    }

    metadata = {
        "combined_input": args.combined,
        "drugbank_input": args.drugbank,
        "metxbio_input": args.metxbio_sdf,
        "metpred_input": args.metpred_doc,
        "xenosite_manifest": args.xenosite_manifest,
        "notes": {
            "main_training_sets": [
                "main5_all_models_conservative",
                "main5_all_models_permissive",
            ],
            "recommended_site_training_set": "main5_all_models_conservative",
            "recommended_xtb_training_note": "Use --compute-xtb-if-missing for new external molecules.",
            "xenosite_note": "XenoSite supplementary data in data/xenosite_suppl is UGT-oriented pseudo-label data and is not merged into main CYP supervision.",
            "metxbio_note": "Conservative keeps only atom-level BOM annotations; permissive also maps bond events to endpoint atoms.",
        },
        "source_summaries": {
            "combined": summarize_rows(combined_rows),
            "metxbio": metxbio_summary,
            "metpred": metpred_summary,
        },
        "dataset_summaries": {name: summarize_rows(rows) for name, rows in outputs.items()},
    }

    for name, rows in outputs.items():
        write_dataset(output_dir / f"{name}.json", rows, metadata)

    if (ROOT / args.xenosite_manifest).exists():
        manifest = json.loads((ROOT / args.xenosite_manifest).read_text())
        (output_dir / "xenosite_auxiliary_manifest.json").write_text(json.dumps(manifest, indent=2))

    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
