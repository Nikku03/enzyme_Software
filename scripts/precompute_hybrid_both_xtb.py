#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2.features.xtb_features import _full_cache_path, compute_full_xtb_payload
from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


def _load_bundle(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute xTB cache for the hybrid+both bundle.")
    parser.add_argument("--bundle", default="data/hybrid_both_bundle/bundle.json")
    parser.add_argument("--dataset-key", choices=["xtb_union_dataset", "base_dataset"], default="xtb_union_dataset")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    bundle_path = ROOT / args.bundle
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    bundle = _load_bundle(bundle_path)
    dataset_path = ROOT / bundle["paths"][args.dataset_key]
    cache_dir = Path(args.cache_dir or bundle["defaults"]["xtb_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    drugs = _load_drugs(dataset_path)
    if args.limit > 0:
        drugs = drugs[: int(args.limit)]

    print(f"Bundle: {bundle_path.relative_to(ROOT)}")
    print(f"Dataset: {dataset_path.relative_to(ROOT)}")
    print(f"Compounds to process: {len(drugs)}")
    print(f"Cache dir: {cache_dir}")

    ok = skipped = failed = 0
    for i, drug in enumerate(drugs, start=1):
        smiles = str(drug.get("smiles") or "")
        name = str(drug.get("name") or drug.get("id") or f"row_{i}")
        if not smiles:
            failed += 1
            continue
        prep = prepare_mol(smiles)
        canonical = prep.canonical_smiles or smiles
        cache_path = _full_cache_path(canonical, cache_dir)
        if args.skip_existing and cache_path.exists():
            skipped += 1
            continue
        result = compute_full_xtb_payload(canonical)
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if result.get("xtb_valid"):
            ok += 1
        else:
            failed += 1
            print(f"  WARN {name}: {result.get('status', 'unknown')}")
        if i % 25 == 0 or i == len(drugs):
            print(f"  {i}/{len(drugs)} ok={ok} skipped={skipped} failed={failed}")

    print(f"Done. ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
