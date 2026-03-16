from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

INTERACTIONS_URL = "https://unmtid-dbs.net/download/DrugCentral/2021_09_01/drug.target.interaction.tsv.gz"
SMILES_URL = "https://unmtid-dbs.net/download/DrugCentral/2021_09_01/structures.smiles.tsv"
MAJOR_CYPS = {"CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"}
CYP_PATTERN = re.compile(r"CYP\d+[A-Z]\d+", re.IGNORECASE)


def extract_cyp(value: object) -> Optional[str]:
    match = CYP_PATTERN.search(str(value or ""))
    return match.group(0).upper() if match else None


def download_drugcentral_cyp_data(output_path: str = "data/drugcentral_cyp_drugs.json") -> List[Dict[str, object]]:
    print("=" * 60)
    print("Downloading DrugCentral Data")
    print("=" * 60)
    df = pd.read_csv(INTERACTIONS_URL, sep="\t", compression="gzip", low_memory=False)
    print(f"Total interactions: {len(df)}")
    cyp_mask = df["TARGET_NAME"].astype(str).str.contains("CYP", case=False, na=False) | df["GENE"].astype(str).str.contains("CYP", case=False, na=False)
    cyp_df = df[cyp_mask].copy()
    print(f"CYP-related interactions: {len(cyp_df)}")
    cyp_df["primary_cyp"] = cyp_df["GENE"].map(extract_cyp).fillna(cyp_df["TARGET_NAME"].map(extract_cyp))
    cyp_df = cyp_df[cyp_df["primary_cyp"].isin(MAJOR_CYPS)].copy()
    substrate_mask = (
        cyp_df["ACTION_TYPE"].astype(str).str.contains("substrate", case=False, na=False)
        | cyp_df["ACT_TYPE"].astype(str).str.fullmatch("Km", case=False, na=False)
        | cyp_df["ACT_COMMENT"].astype(str).str.contains("substrate|metabol", case=False, na=False)
    )
    substrate_df = cyp_df[substrate_mask].copy()
    if substrate_df.empty:
        print("No explicit substrate rows found in DrugCentral; writing empty drug set.")
        payload = {"metadata": {"source": "DrugCentral", "url": "https://drugcentral.org/", "total_drugs": 0, "cyp_distribution": {}}, "drugs": []}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Saved to {out}")
        return []
    smiles_df = pd.read_csv(SMILES_URL, sep="\t", low_memory=False)
    merged = substrate_df.merge(smiles_df, left_on="STRUCT_ID", right_on="ID", how="left")
    merged = merged[merged["SMILES"].notna() & (merged["SMILES"].astype(str) != "")].copy()
    seen_smiles = set()
    drugs: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        smiles = str(row["SMILES"])
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)
        drugs.append(
            {
                "name": str(row.get("INN") or row.get("DRUG_NAME") or "Unknown"),
                "smiles": smiles,
                "primary_cyp": str(row["primary_cyp"]),
                "source": "DrugCentral",
                "confidence": "validated",
                "cyp_label_source": "DrugCentral",
                "interaction_type": str(row.get("ACT_TYPE") or ""),
                "interaction_action": str(row.get("ACTION_TYPE") or ""),
            }
        )
    by_cyp = Counter(d["primary_cyp"] for d in drugs)
    payload = {
        "metadata": {
            "source": "DrugCentral",
            "url": "https://drugcentral.org/",
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
    download_drugcentral_cyp_data()
