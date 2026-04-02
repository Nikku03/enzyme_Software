from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_full_xtb_features, payload_cached_xtb_valid, payload_true_xtb_valid


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute full Module -1 xTB features")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", default="cache/full_xtb")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-labeled-only", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    drugs = _load_drugs(dataset_path)
    print(f"Loaded {len(drugs)} drugs", flush=True)
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]
        print(f"Site-labeled: {len(drugs)}", flush=True)
    random.Random(args.seed).shuffle(drugs)
    if args.limit is not None:
        drugs = drugs[: int(args.limit)]
        print(f"Limited to: {len(drugs)}", flush=True)

    ok = 0
    true_ok = 0
    failed = 0
    statuses: dict[str, int] = {}
    for index, drug in enumerate(drugs, start=1):
        smiles = str(drug.get("smiles", "")).strip()
        if not smiles:
            failed += 1
            statuses["missing_smiles"] = statuses.get("missing_smiles", 0) + 1
            continue
        payload = load_or_compute_full_xtb_features(smiles, cache_dir=cache_dir, compute_if_missing=True)
        status = str(payload.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if payload_cached_xtb_valid(payload):
            ok += 1
        else:
            failed += 1
        if payload_true_xtb_valid(payload):
            true_ok += 1
        if index % 25 == 0 or index == len(drugs):
            print(f"{index}/{len(drugs)} processed | payload_ok={ok} | true_xtb_ok={true_ok} | failed={failed}", flush=True)

    print("\nStatus summary:", flush=True)
    for key in sorted(statuses):
        print(f"  {key}: {statuses[key]}", flush=True)
    print(f"Payload-valid molecules: {ok}/{len(drugs)}", flush=True)
    print(f"True-xTB-valid molecules: {true_ok}/{len(drugs)}", flush=True)
    print(f"Cache dir: {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
