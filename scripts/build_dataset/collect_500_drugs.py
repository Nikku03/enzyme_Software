from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

CHEMBL_TARGETS = {
    "CYP1A2": "CHEMBL3356",
    "CYP2C9": "CHEMBL3397",
    "CYP2C19": "CHEMBL3622",
    "CYP2D6": "CHEMBL289",
    "CYP3A4": "CHEMBL340",
}

DEFAULT_PER_CYP_TARGET = {
    "CYP3A4": 150,
    "CYP2D6": 100,
    "CYP2C9": 80,
    "CYP2C19": 80,
    "CYP1A2": 90,
}


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError("requests is required for ChEMBL collection")


def _chembl_get_json(url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    _require_requests()
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_cyp_substrates(target_id: str, cyp_name: str, limit: int = 100) -> List[Dict[str, object]]:
    """Fetch unique substrate-like compounds for a CYP enzyme from ChEMBL."""
    url = f"{CHEMBL_API}/activity.json"
    params = {
        "target_chembl_id": target_id,
        "standard_type__in": "IC50,Ki,Km",
        "standard_value__gte": 10000,
        "limit": max(20, min(limit * 2, 1000)),
    }
    data = _chembl_get_json(url, params=params)
    activities = list(data.get("activities", []))

    compounds: Dict[str, Dict[str, object]] = {}
    for act in activities:
        chembl_id = act.get("molecule_chembl_id")
        if not chembl_id or chembl_id in compounds:
            continue
        compounds[str(chembl_id)] = {
            "chembl_id": str(chembl_id),
            "cyp": cyp_name,
            "activity_type": act.get("standard_type"),
            "activity_value": act.get("standard_value"),
        }
        if len(compounds) >= limit:
            break
    return list(compounds.values())


def get_compound_smiles(chembl_id: str) -> Optional[str]:
    """Fetch canonical SMILES for a ChEMBL compound."""
    url = f"{CHEMBL_API}/molecule/{chembl_id}.json"
    data = _chembl_get_json(url)
    structures = data.get("molecule_structures") or {}
    return structures.get("canonical_smiles")


def collect_500_drugs(output_path: str = "data/500_drugs_raw.json") -> List[Dict[str, object]]:
    """Collect a multi-CYP raw substrate set from ChEMBL."""
    all_drugs: List[Dict[str, object]] = []
    for cyp_name, target_id in CHEMBL_TARGETS.items():
        limit = DEFAULT_PER_CYP_TARGET.get(cyp_name, 50)
        print(f"Fetching {cyp_name} substrates (target: {limit})...")
        substrates = get_cyp_substrates(target_id, cyp_name, limit=max(limit * 2, 50))
        print(f"  Found {len(substrates)} candidates")

        count = 0
        for compound in substrates:
            if count >= limit:
                break
            smiles = get_compound_smiles(str(compound["chembl_id"]))
            if smiles and len(smiles) < 200:
                compound["smiles"] = smiles
                all_drugs.append(compound)
                count += 1
            time.sleep(0.05)
        print(f"  Collected {count} with valid SMILES")

    seen_smiles: Dict[str, Dict[str, object]] = {}
    unique_drugs: List[Dict[str, object]] = []
    for drug in all_drugs:
        smiles = str(drug["smiles"])
        if smiles not in seen_smiles:
            merged = dict(drug)
            merged["all_cyps"] = [merged["cyp"]]
            seen_smiles[smiles] = merged
            unique_drugs.append(merged)
        else:
            existing = seen_smiles[smiles]
            if drug["cyp"] not in existing.get("all_cyps", []):
                existing.setdefault("all_cyps", []).append(drug["cyp"])

    print(f"\nTotal unique drugs: {len(unique_drugs)}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(unique_drugs, indent=2))
    print(f"Saved to {out}")
    return unique_drugs


if __name__ == "__main__":
    collect_500_drugs()
