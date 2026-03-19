#!/usr/bin/env python3
"""Merge all 4 CYP source datasets into a single deduplicated training file.

Sources:
  1. DrugBank        – data/training_dataset_drugbank.json   (~611 drugs, 329 site-labeled)
  2. SuperCYP        – data/supercyp_drugs.json              (~435 drugs, CYP labels only)
  3. MetXBioDB       – data/prepared_training/metxbio_main5_atom_only.json  (~1384 site-labeled)
  4. MetaPred        – data/prepared_training/metpred_cyp_only.json         (~85 CYP-only)

Note: XenoSite supplementary data is UGT-oriented (not CYP) → excluded.

Deduplication key: (canonical_smiles, primary_cyp)
Conflict resolution: union of site atoms, keep highest confidence label.

Output: data/merged_all_sources.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD_DATASET = ROOT / "scripts" / "build_dataset"
for p in (str(SRC), str(BUILD_DATASET)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the existing merge/normalize infrastructure.
from prepare_all_model_training_data import (  # noqa: E402
    MAIN5,
    canonicalize_smiles,
    extract_site_indices,
    load_json_drugs,
    merge_entries,
    normalize_entry,
)

DATA = ROOT / "data"

SOURCES: List[Dict[str, object]] = [
    {
        "label": "DrugBank",
        "path": DATA / "training_dataset_drugbank.json",
        "source_override": "DrugBank",
    },
    {
        "label": "SuperCYP",
        "path": DATA / "supercyp_drugs.json",
        "source_override": None,
    },
    {
        "label": "MetXBioDB",
        "path": DATA / "prepared_training" / "metxbio_main5_atom_only.json",
        "source_override": None,
    },
    {
        "label": "MetaPred",
        "path": DATA / "prepared_training" / "metpred_cyp_only.json",
        "source_override": None,
    },
]


def _patch_source(drug: dict, override: str | None) -> dict:
    """Ensure source field is set correctly for sources with legacy names."""
    if override and str(drug.get("source", "")).lower() in ("manual_curation", "", "unknown"):
        drug = dict(drug)
        drug["source"] = override
    return drug


def main() -> None:
    all_raw: List[dict] = []
    per_source_counts: Dict[str, int] = {}

    for spec in SOURCES:
        path = spec["path"]
        label = spec["label"]
        if not path.exists():
            print(f"  [SKIP] {label}: file not found at {path}")
            continue
        drugs = load_json_drugs(path)
        # Apply source override (e.g. manual_curation → DrugBank)
        drugs = [_patch_source(d, spec["source_override"]) for d in drugs]
        per_source_counts[label] = len(drugs)
        all_raw.extend(drugs)
        print(f"  Loaded {len(drugs):5d} entries from {label}")

    print(f"\nTotal raw entries (before dedup): {len(all_raw)}")

    # Merge + deduplicate
    merged = merge_entries(all_raw)

    # Count stats
    source_counts: Counter = Counter()
    site_labeled = 0
    cyp_counts: Counter = Counter()
    for drug in merged:
        source_counts[str(drug.get("source", "unknown"))] += 1
        if extract_site_indices(drug):
            site_labeled += 1
        cyp_counts[str(drug.get("primary_cyp", "unknown"))] += 1

    print(f"After deduplication: {len(merged)} unique (smiles, primary_cyp) pairs")
    print(f"  Site-labeled : {site_labeled}")
    print(f"  CYP-only     : {len(merged) - site_labeled}")
    print(f"  By source    : {dict(source_counts)}")
    print(f"  By CYP       : {dict(cyp_counts)}")

    out_path = DATA / "merged_all_sources.json"
    payload = {
        "description": "Merged CYP metabolism dataset: DrugBank + SuperCYP + MetXBioDB + MetaPred",
        "total": len(merged),
        "site_labeled": site_labeled,
        "source_counts": dict(source_counts),
        "cyp_counts": dict(cyp_counts),
        "drugs": merged,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
