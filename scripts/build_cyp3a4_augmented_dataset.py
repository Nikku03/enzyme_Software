from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared_training"


def _canon_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def _num_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    return int(mol.GetNumAtoms()) if mol is not None else 0


def _valid_site_atoms(site_atoms: list[int], num_atoms: int) -> list[int]:
    valid = sorted({int(idx) for idx in site_atoms if 0 <= int(idx) < int(num_atoms)})
    return valid


def _key_for(smiles: str, cyp: str) -> str:
    return f"{_canon_smiles(smiles)}::{str(cyp).strip()}"


def _source_key(row: dict) -> str:
    return str(row.get("site_source") or row.get("source") or "unknown").strip()


def _safe_mol_prop(mol: Chem.Mol, name: str) -> str:
    if not mol.HasProp(name):
        return ""
    try:
        return str(mol.GetProp(name))
    except Exception:
        try:
            value = mol.GetPropsAsDict(includePrivate=False, includeComputed=False).get(name, "")
            return str(value)
        except Exception:
            return ""


def _site_atoms_from_row(drug: dict) -> list[int]:
    out: list[int] = []
    if drug.get("som"):
        for som in list(drug.get("som") or []):
            atom_idx = som.get("atom_idx", som) if isinstance(som, dict) else som
            if isinstance(atom_idx, int):
                out.append(int(atom_idx))
    elif drug.get("site_atoms"):
        out.extend(int(v) for v in list(drug.get("site_atoms") or []) if isinstance(v, int))
    elif drug.get("site_atom_indices"):
        out.extend(int(v) for v in list(drug.get("site_atom_indices") or []) if isinstance(v, int))
    elif drug.get("metabolism_sites"):
        out.extend(int(v) for v in list(drug.get("metabolism_sites") or []) if isinstance(v, int))
    return sorted(set(out))


def _normalize_row(drug: dict) -> dict | None:
    smiles = _canon_smiles(str(drug.get("smiles", "") or ""))
    cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "").strip()
    if not smiles or not cyp:
        return None
    n_atoms = _num_atoms(smiles)
    if n_atoms <= 0:
        return None
    site_atoms = _valid_site_atoms(_site_atoms_from_row(drug), n_atoms)
    if not site_atoms:
        return None
    row = dict(drug)
    row["smiles"] = smiles
    row["primary_cyp"] = cyp
    row["all_cyps"] = [cyp]
    row["site_atoms"] = site_atoms
    row["metabolism_sites"] = site_atoms
    row["som"] = [{"atom_idx": idx} for idx in site_atoms]
    row.setdefault("confidence", "medium")
    row.setdefault("full_xtb_status", "unknown")
    row.setdefault("reactions", ["hydroxylation"])
    row.setdefault("site_source", str(row.get("source") or "unknown"))
    row.setdefault("source_details", [str(row.get("source") or "unknown")])
    row.setdefault("symmetry_expanded", False)
    row.setdefault("symmetry_expanded_added_atoms", [])
    row.setdefault("symmetry_expanded_groups", [])
    return row


def load_main8_cyp3a4(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = list(payload.get("drugs", payload))
    out = []
    for row in rows:
        if str(row.get("primary_cyp") or row.get("cyp") or "").strip() != "CYP3A4":
            continue
        normalized = _normalize_row(row)
        if normalized is not None:
            out.append(normalized)
    return out


def _parse_python_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return [str(v) for v in value]
    except Exception:
        pass
    return []


def load_astrazeneca_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = _canon_smiles(row.get("SMILES", ""))
            if not smiles:
                continue
            n_atoms = _num_atoms(smiles)
            som_groups = _parse_python_list(row.get("SoMs grouped (numbers provided are atom indices)"))
            exact_flags = _parse_python_list(row.get("Exact SoM annotation (1) or extended SoM annotation (0) per group"))
            exact_atoms: list[int] = []
            all_atoms: list[int] = []
            for idx, group in enumerate(som_groups):
                group_atoms: list[int] = []
                for token in str(group).replace(",", " ").split():
                    try:
                        group_atoms.append(int(token))
                    except ValueError:
                        continue
                all_atoms.extend(group_atoms)
                if idx < len(exact_flags) and str(exact_flags[idx]).strip() == "1":
                    exact_atoms.extend(group_atoms)
            site_atoms = _valid_site_atoms(exact_atoms or all_atoms, n_atoms)
            if not site_atoms:
                continue
            has_exact = bool(exact_atoms)
            confidence = "high" if has_exact else "medium"
            records.append(
                {
                    "id": f"az120:{row.get('Compound ID', '').strip() or len(records)}",
                    "name": str(row.get("Compound ID", "")).strip() or "AZ120",
                    "smiles": smiles,
                    "primary_cyp": "CYP3A4",
                    "all_cyps": ["CYP3A4"],
                    "reactions": ["hydroxylation"],
                    "site_atoms": site_atoms,
                    "metabolism_sites": site_atoms,
                    "som": [{"atom_idx": idx} for idx in site_atoms],
                    "source": "AZ120",
                    "site_source": "AZ120",
                    "confidence": confidence,
                    "full_xtb_status": "unknown",
                    "source_details": ["AZ120", "AstraZeneca120"],
                    "source_comment": str(row.get("Comment", "")).strip(),
                    "symmetry_expanded": False,
                    "symmetry_expanded_added_atoms": [],
                    "symmetry_expanded_groups": [],
                }
            )
    return records


def load_attnsom_sdf(path: Path, *, source_name: str, cyp_name: str = "CYP3A4") -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
        n_atoms = int(Chem.RemoveHs(mol).GetNumAtoms())
        primary: list[int] = []
        secondary: list[int] = []
        tertiary: list[int] = []
        for prop_name, target in (
            ("PRIMARY_SOM", primary),
            ("SECONDARY_SOM", secondary),
            ("TERTIARY_SOM", tertiary),
        ):
            if not mol.HasProp(prop_name):
                continue
            for token in mol.GetProp(prop_name).replace(",", " ").split():
                try:
                    target.append(int(token) - 1)
                except ValueError:
                    continue
        site_atoms = _valid_site_atoms(primary or secondary or tertiary, n_atoms)
        if not site_atoms:
            continue
        source_details = [source_name]
        if mol.HasProp("Citation"):
            source_details.append("literature_cited")
        records.append(
            {
                "id": f"{source_name.lower()}:{_safe_mol_prop(mol, 'ID') or len(records)}",
                "name": _safe_mol_prop(mol, "ID") or _safe_mol_prop(mol, "_Name") or source_name,
                "smiles": smiles,
                "primary_cyp": cyp_name,
                "all_cyps": [cyp_name],
                "reactions": ["hydroxylation"],
                "site_atoms": site_atoms,
                "metabolism_sites": site_atoms,
                "som": [{"atom_idx": idx} for idx in site_atoms],
                "source": source_name,
                "site_source": source_name,
                "confidence": "medium" if secondary and not primary else "validated",
                "full_xtb_status": "unknown",
                "source_details": source_details,
                "citation": _safe_mol_prop(mol, "Citation"),
                "symmetry_expanded": False,
                "symmetry_expanded_added_atoms": [],
                "symmetry_expanded_groups": [],
            }
        )
    return records


def merge_rows(base_rows: list[dict], imported_rows: list[dict], *, merge_policy: str = "union") -> tuple[list[dict], dict]:
    merged: dict[str, dict] = {}
    stats = {
        "base_rows": len(base_rows),
        "import_rows": len(imported_rows),
        "new_rows": 0,
        "overlap_rows": 0,
        "site_atoms_expanded": 0,
        "source_details_augmented": 0,
    }
    for row in base_rows:
        key = _key_for(row["smiles"], row["primary_cyp"])
        if merge_policy == "keep_sources":
            key = f"{key}::{_source_key(row)}"
        merged[key] = dict(row)
    for row in imported_rows:
        key = _key_for(row["smiles"], row["primary_cyp"])
        if merge_policy == "keep_sources":
            key = f"{key}::{_source_key(row)}"
        if key not in merged:
            merged[key] = dict(row)
            stats["new_rows"] += 1
            continue
        stats["overlap_rows"] += 1
        existing = merged[key]
        if merge_policy == "base_priority":
            details = sorted(
                {
                    str(v)
                    for v in list(existing.get("source_details") or []) + list(row.get("source_details") or [])
                    if str(v).strip()
                }
            )
            if details != list(existing.get("source_details") or []):
                stats["source_details_augmented"] += 1
            existing["source_details"] = details
            continue
        merged_sites = sorted(set(existing.get("site_atoms", [])) | set(row.get("site_atoms", [])))
        if merged_sites != list(existing.get("site_atoms", [])):
            stats["site_atoms_expanded"] += 1
        existing["site_atoms"] = merged_sites
        existing["metabolism_sites"] = merged_sites
        existing["som"] = [{"atom_idx": idx} for idx in merged_sites]
        details = sorted(
            {
                str(v)
                for v in list(existing.get("source_details") or []) + list(row.get("source_details") or [])
                if str(v).strip()
            }
        )
        if details != list(existing.get("source_details") or []):
            stats["source_details_augmented"] += 1
        existing["source_details"] = details
        conf_order = {"medium": 0, "high": 1, "validated": 2, "validated_gold": 3}
        old_conf = str(existing.get("confidence", "medium")).lower()
        new_conf = str(row.get("confidence", "medium")).lower()
        if conf_order.get(new_conf, 0) > conf_order.get(old_conf, 0):
            existing["confidence"] = row.get("confidence", existing.get("confidence"))
            existing["site_source"] = row.get("site_source", existing.get("site_source"))
    merged_rows = sorted(merged.values(), key=lambda r: (str(r.get("name", "")), str(r.get("smiles", ""))))
    return merged_rows, stats


def build_summary(rows: list[dict]) -> dict:
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    confidences = Counter(str(row.get("confidence", "unknown")) for row in rows)
    return {
        "n_rows": len(rows),
        "n_site_labeled": len(rows),
        "sources": dict(sorted(sources.items())),
        "confidences": dict(sorted(confidences.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset",
        default="data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
    )
    parser.add_argument(
        "--astra-csv",
        default="AstraZeneca_SoM_annotations_120_Compounds.csv",
    )
    parser.add_argument(
        "--attnsom-3a4-sdf",
        default="data/ATTNSOM/cyp_dataset/3A4.sdf",
    )
    parser.add_argument(
        "--cyp-dbs-3a4-sdf",
        default="CYP_DBs/3A4.sdf",
    )
    parser.add_argument(
        "--out",
        default="data/prepared_training/main8_cyp3a4_augmented.json",
    )
    parser.add_argument(
        "--merge-policy",
        choices=("union", "base_priority", "keep_sources"),
        default="union",
    )
    args = parser.parse_args()

    base_rows = load_main8_cyp3a4(ROOT / args.base_dataset)
    astra_rows = load_astrazeneca_csv(ROOT / args.astra_csv)

    attnsom_path = ROOT / args.attnsom_3a4_sdf
    cyp_dbs_path = ROOT / args.cyp_dbs_3a4_sdf
    if attnsom_path.exists():
        attn_rows = load_attnsom_sdf(attnsom_path, source_name="ATTNSOM")
        attn_source_used = str(attnsom_path.relative_to(ROOT))
    else:
        attn_rows = load_attnsom_sdf(cyp_dbs_path, source_name="CYP_DBs_external")
        attn_source_used = str(cyp_dbs_path.relative_to(ROOT))

    if attnsom_path.exists() and cyp_dbs_path.exists():
        attn_set = {_key_for(row["smiles"], row["primary_cyp"]) for row in attn_rows}
        cyp_rows = load_attnsom_sdf(cyp_dbs_path, source_name="CYP_DBs_external")
        cyp_set = {_key_for(row["smiles"], row["primary_cyp"]) for row in cyp_rows}
        duplicate_source_match = len(attn_set ^ cyp_set) == 0
    else:
        duplicate_source_match = False

    imported_rows = astra_rows + attn_rows
    merged_rows, merge_stats = merge_rows(base_rows, imported_rows, merge_policy=args.merge_policy)
    summary = build_summary(merged_rows)

    payload = {
        "n_drugs": len(merged_rows),
        "n_site_labeled": len(merged_rows),
        "summary": summary,
        "build_stats": {
            "base_dataset": args.base_dataset,
            "astra_rows": len(astra_rows),
            "attnsom_rows": len(attn_rows),
            "attnsom_source_used": attn_source_used,
            "cyp_dbs_duplicate_of_attnsom": duplicate_source_match,
            "merge_policy": args.merge_policy,
            **merge_stats,
        },
        "drugs": merged_rows,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved augmented CYP3A4 dataset to {out_path}")
    print(json.dumps(payload["build_stats"], indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
