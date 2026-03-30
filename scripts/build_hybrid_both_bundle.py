#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.train_hybrid_full_xtb import _load_xenosite_aux_entries


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def _dataset_summary(drugs: Iterable[dict]) -> dict:
    rows = list(drugs)
    return {
        "count": len(rows),
        "site_labeled": sum(1 for row in rows if _has_site_labels(row)),
        "primary_cyp": sum(1 for row in rows if str(row.get("primary_cyp", "")).strip()),
        "source_counts": dict(Counter(str(row.get("source", "unknown")) for row in rows)),
        "cyp_counts": dict(Counter(str(row.get("primary_cyp", "")) for row in rows if str(row.get("primary_cyp", "")).strip())),
    }


def _dedupe_union(*datasets: Iterable[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for dataset in datasets:
        for row in dataset:
            smiles = str(row.get("smiles", "")).strip()
            if not smiles:
                continue
            merged.setdefault(smiles, row)
    return list(merged.values())


def _resolve_structure(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither structure SDF exists: {primary} | {fallback}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical hybrid+both bundle manifest.")
    parser.add_argument("--base-dataset", default="data/prepared_training/main5_site_conservative_singlecyp_clean.json")
    parser.add_argument("--xenosite-manifest", default="data/xenosite_suppl/manifest.json")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--fallback-structure-sdf", default="structures.sdf")
    parser.add_argument("--output-dir", default="data/hybrid_both_bundle")
    parser.add_argument("--xenosite-topk", type=int, default=1)
    parser.add_argument("--default-epochs", type=int, default=75)
    parser.add_argument("--default-batch-size", type=int, default=16)
    parser.add_argument("--default-learning-rate", type=float, default=2e-4)
    parser.add_argument("--default-weight-decay", type=float, default=1e-4)
    parser.add_argument("--default-xtb-cache-dir", default="cache/full_xtb_hybrid_both")
    args = parser.parse_args()

    base_dataset = ROOT / args.base_dataset
    xenosite_manifest = ROOT / args.xenosite_manifest
    structure_sdf = _resolve_structure(ROOT / args.structure_sdf, ROOT / args.fallback_structure_sdf)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_dataset.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset}")
    if not xenosite_manifest.exists():
        raise FileNotFoundError(f"XenoSite manifest not found: {xenosite_manifest}")

    base_drugs = _load_drugs(base_dataset)
    xenosite_aux = _load_xenosite_aux_entries(xenosite_manifest, topk=args.xenosite_topk, per_file_limit=0)
    xtb_union = _dedupe_union(base_drugs, xenosite_aux)

    resolved_manifest = {"datasets": []}
    manifest_payload = json.loads(xenosite_manifest.read_text())
    for row in manifest_payload.get("datasets", []):
        rel = str(row.get("file", "")).strip()
        if not rel:
            continue
        source_path = xenosite_manifest.parent / rel
        if not source_path.exists():
            continue
        resolved = dict(row)
        resolved["file"] = os.path.relpath(source_path, output_dir)
        resolved_manifest["datasets"].append(resolved)

    resolved_manifest_path = output_dir / "xenosite_manifest_resolved.json"
    xtb_union_path = output_dir / "xtb_union.json"
    bundle_path = output_dir / "bundle.json"

    resolved_manifest_path.write_text(json.dumps(resolved_manifest, indent=2))
    xtb_union_path.write_text(json.dumps({"drugs": xtb_union}, indent=2))

    bundle = {
        "bundle_name": "hybrid_both_main5_clean_plus_xenosite",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Canonical hybrid+both training bundle: clean single-CYP base set, train-only XenoSite auxiliary, 3D structure SDF, and xTB warmup union.",
        "paths": {
            "base_dataset": str(base_dataset.relative_to(ROOT)),
            "xenosite_manifest": str(resolved_manifest_path.relative_to(ROOT)),
            "xtb_union_dataset": str(xtb_union_path.relative_to(ROOT)),
            "structure_sdf": str(structure_sdf.relative_to(ROOT)),
            "fallback_structure_sdf": str((ROOT / args.fallback_structure_sdf).relative_to(ROOT)),
        },
        "defaults": {
            "epochs": int(args.default_epochs),
            "batch_size": int(args.default_batch_size),
            "learning_rate": float(args.default_learning_rate),
            "weight_decay": float(args.default_weight_decay),
            "site_labeled_only": True,
            "include_xenosite": True,
            "xenosite_topk": int(args.xenosite_topk),
            "compute_xtb_if_missing": False,
            "freeze_nexus_memory": False,
            "rebuild_nexus_memory": True,
            "disable_nexus_bridge": False,
            "early_stopping_patience": 0,
            "xtb_cache_dir": args.default_xtb_cache_dir,
        },
        "summaries": {
            "base_dataset": _dataset_summary(base_drugs),
            "xenosite_auxiliary": {
                "count": len(xenosite_aux),
                "site_labeled": len(xenosite_aux),
                "source_counts": dict(Counter(str(row.get("source", "unknown")) for row in xenosite_aux)),
            },
            "xtb_union": {
                "count": len(xtb_union),
            },
        },
        "notes": [
            "The base dataset remains the supervised split source.",
            "XenoSite entries are weak site-only auxiliary rows and must stay train-only.",
            "xtb_union.json is for xTB cache warming and wave-feature coverage, not direct split generation.",
        ],
    }
    bundle_path.write_text(json.dumps(bundle, indent=2))

    print(f"Wrote bundle: {bundle_path.relative_to(ROOT)}")
    print(f"Wrote resolved XenoSite manifest: {resolved_manifest_path.relative_to(ROOT)}")
    print(f"Wrote xTB union dataset: {xtb_union_path.relative_to(ROOT)}")
    print(json.dumps(bundle["summaries"], indent=2))


if __name__ == "__main__":
    main()
