"""
Extract novel compounds from CYP_DBs/ that are NOT in the training data.
Converts SDF → SMILES (via RDKit), maps SoM atom indices (1-based → 0-based),
and writes a JSON file compatible with the pipeline.

Usage:
    conda run -n bondbreak python scripts/extract_cyp_dbs_external.py \
        --sdf-dir CYP_DBs \
        --training data/merged_all_sources.json \
        --out data/cyp_dbs_external_test.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import MolToSmiles
except ImportError:
    sys.exit("RDKit required — run with: conda run -n bondbreak python ...")


# ── SDF parser ────────────────────────────────────────────────────────────────

def _parse_field(record: str, field: str) -> str:
    m = re.search(rf"<{field}>\n(.+?)(?:\n|$)", record)
    return m.group(1).strip() if m else ""


def _parse_som_indices(raw: str) -> list[int]:
    """Parse space-separated 1-based atom indices → 0-based list."""
    out = []
    for tok in raw.strip().split():
        try:
            out.append(int(tok) - 1)  # 1-based → 0-based
        except ValueError:
            pass
    return out


def parse_sdf(path: Path, cyp_name: str) -> list[dict]:
    """Parse one SDF file, return list of compound dicts."""
    content = path.read_text(encoding="latin-1")
    records = [r.strip() for r in content.split("$$$$") if r.strip()]
    compounds = []
    for record in records:
        mol = Chem.MolFromMolBlock(record, removeHs=True, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolBlock(record, removeHs=True, sanitize=False)
        if mol is None:
            continue
        try:
            smiles = MolToSmiles(mol)
        except Exception:
            continue

        compound_id = _parse_field(record, "ID") or record.split("\n")[0].strip()
        primary_som_raw  = _parse_field(record, "PRIMARY_SOM")
        secondary_som_raw = _parse_field(record, "SECONDARY_SOM")
        tertiary_som_raw  = _parse_field(record, "TERTIARY_SOM")
        citation = _parse_field(record, "Citation")
        doi      = _parse_field(record, "DOI")

        primary_sites   = _parse_som_indices(primary_som_raw)
        secondary_sites = _parse_som_indices(secondary_som_raw)
        tertiary_sites  = _parse_som_indices(tertiary_som_raw)

        all_sites = list(dict.fromkeys(primary_sites + secondary_sites + tertiary_sites))

        compounds.append({
            "id":          f"cyp_dbs:{cyp_name}:{compound_id}",
            "name":        compound_id,
            "smiles":      smiles,
            "primary_cyp": f"CYP{cyp_name}",
            "site_atoms":  all_sites,
            "som":         [{"atom_idx": idx, "bond_class": "primary"} for idx in primary_sites]
                         + [{"atom_idx": idx, "bond_class": "secondary"} for idx in secondary_sites]
                         + [{"atom_idx": idx, "bond_class": "tertiary"} for idx in tertiary_sites],
            "source":      "CYP_DBs_external",
            "confidence":  "validated",
            "citation":    citation,
            "doi":         doi,
        })
    return compounds


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-dir",  default="CYP_DBs")
    parser.add_argument("--training", default="data/merged_all_sources.json")
    parser.add_argument("--out",      default="data/cyp_dbs_external_test.json")
    args = parser.parse_args()

    # Load training set for deduplication — use canonical SMILES (primary) + name (fallback)
    train_data = json.loads(Path(args.training).read_text())
    train_drugs = train_data.get("drugs", train_data)
    print(f"Training compounds: {len(train_drugs)}")

    def _canon(smi: str) -> str:
        if not smi:
            return ""
        try:
            mol = Chem.MolFromSmiles(smi.strip())
            return MolToSmiles(mol) if mol else smi.strip()
        except Exception:
            return smi.strip()

    train_smiles = {_canon(d.get("smiles", "")) for d in train_drugs} - {""}
    train_names  = {str(d.get("name", "")).lower().strip() for d in train_drugs}

    # CYP isoforms we care about (skip HLM — no specific CYP label)
    cyp_map = {
        "1A2": "1A2", "2A6": "2A6", "2B6": "2B6", "2C19": "2C19",
        "2C8": "2C8", "2C9": "2C9", "2D6": "2D6", "2E1": "2E1", "3A4": "3A4",
    }

    novel = []
    skipped_overlap = 0
    skipped_no_som  = 0
    skipped_parse   = 0

    sdf_dir = Path(args.sdf_dir)
    for cyp_key, cyp_label in cyp_map.items():
        sdf_path = sdf_dir / f"{cyp_key}.sdf"
        if not sdf_path.exists():
            continue
        compounds = parse_sdf(sdf_path, cyp_label)
        n_total = len(compounds)
        n_novel = 0
        for c in compounds:
            # Deduplicate by canonical SMILES first, then name fallback
            if _canon(c["smiles"]) in train_smiles:
                skipped_overlap += 1
                continue
            if c["name"].lower().strip() in train_names:
                skipped_overlap += 1
                continue
            if not c["site_atoms"]:
                skipped_no_som += 1
                continue
            # Track this SMILES so within-CYP_DBs duplicates are also excluded
            train_smiles.add(_canon(c["smiles"]))
            novel.append(c)
            n_novel += 1
        print(f"  CYP{cyp_key:<5}  {n_total:>4} total  →  {n_novel:>3} novel")

    print(f"\nSkipped (in training): {skipped_overlap}")
    print(f"Skipped (no SoM):      {skipped_no_som}")
    print(f"Novel compounds kept:  {len(novel)}")

    # CYP distribution
    from collections import Counter
    dist = Counter(c["primary_cyp"] for c in novel)
    print("\nCYP distribution of novel set:")
    for cyp, n in sorted(dist.items()):
        print(f"  {cyp}: {n}")

    Path(args.out).write_text(json.dumps({"drugs": novel}, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
