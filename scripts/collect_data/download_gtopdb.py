from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

GTOPDB_URL = "https://www.guidetopharmacology.org/DATA"
MAJOR_CYPS = {"CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"}
CYP_PATTERN = re.compile(r"CYP\d+[A-Z]\d+", re.IGNORECASE)


def extract_cyp(value: object) -> Optional[str]:
    match = CYP_PATTERN.search(str(value or ""))
    return match.group(0).upper() if match else None


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)


def _load_tsv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, sep="\t", skiprows=1, low_memory=False)


def download_gtopdb_cyp_data(output_path: str = "data/gtopdb_cyp_drugs.json") -> List[Dict[str, object]]:
    print("=" * 60)
    print("Downloading GtoPdb Enzyme Interactions")
    print("=" * 60)
    enzyme_df = _load_tsv(f"{GTOPDB_URL}/enzyme_interactions.tsv")
    ligands_df = _load_tsv(f"{GTOPDB_URL}/ligands.tsv")
    print(f"Total interactions: {len(enzyme_df)}")
    target_mask = _safe_col(enzyme_df, "Target").astype(str).str.contains("CYP", case=False, na=False)
    gene_mask = _safe_col(enzyme_df, "Target Gene Symbol").astype(str).str.contains("CYP", case=False, na=False)
    cyp_df = enzyme_df[target_mask | gene_mask].copy()
    print(f"CYP-related interactions: {len(cyp_df)}")
    cyp_df["primary_cyp"] = (
        _safe_col(cyp_df, "Target").map(extract_cyp).fillna(_safe_col(cyp_df, "Target Gene Symbol").map(extract_cyp))
    )
    cyp_df = cyp_df[cyp_df["primary_cyp"].notna()].copy()
    interaction_mask = (
        _safe_col(cyp_df, "Type").astype(str).str.contains("substrate", case=False, na=False)
        | _safe_col(cyp_df, "Action").astype(str).str.contains("substrate|metab", case=False, na=False)
        | _safe_col(cyp_df, "Assay Description").astype(str).str.contains("substrate|metab", case=False, na=False)
    )
    substrate_df = cyp_df[interaction_mask].copy()
    if substrate_df.empty:
        print("No explicit substrate rows found in GtoPdb; writing empty drug set.")
        payload = {"metadata": {"source": "Guide to Pharmacology", "url": "https://www.guidetopharmacology.org/", "total_drugs": 0, "cyp_distribution": {}}, "drugs": []}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Saved to {out}")
        return []
    merge_key = "Ligand ID" if "Ligand ID" in substrate_df.columns and "Ligand ID" in ligands_df.columns else None
    if merge_key is None:
        raise KeyError("Could not identify ligand ID column in GtoPdb files")
    merged = substrate_df.merge(ligands_df[[merge_key, "Name", "SMILES"]], on=merge_key, how="left")
    merged = merged[merged["SMILES"].notna() & (merged["SMILES"].astype(str) != "")].copy()
    seen_smiles = set()
    drugs: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        smiles = str(row["SMILES"])
        cyp = str(row["primary_cyp"])
        if cyp not in MAJOR_CYPS or smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)
        drugs.append(
            {
                "name": str(row.get("Name") or row.get("Ligand") or "Unknown"),
                "smiles": smiles,
                "primary_cyp": cyp,
                "source": "GtoPdb",
                "confidence": "validated",
                "cyp_label_source": "GtoPdb",
                "interaction_type": str(row.get("Type") or ""),
                "interaction_action": str(row.get("Action") or ""),
            }
        )
    by_cyp = Counter(d["primary_cyp"] for d in drugs)
    payload = {
        "metadata": {
            "source": "Guide to Pharmacology",
            "url": "https://www.guidetopharmacology.org/",
            "total_drugs": len(drugs),
            "cyp_distribution": dict(by_cyp),
        },
        "drugs": drugs,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nUnique drugs: {len(drugs)}")
    for cyp, count in sorted(by_cyp.items()):
        print(f"  {cyp}: {count}")
    print(f"\nSaved to {out}")
    return drugs


if __name__ == "__main__":
    download_gtopdb_cyp_data()
