from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.dataset import filter_site_labeled_drugs
from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_xtb_features


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute xTB atom features for micropattern reranking")
    parser.add_argument("--dataset", default="data/drugbank_standardized.json")
    parser.add_argument("--supercyp-dataset", default=None)
    parser.add_argument("--cache-dir", default="cache/micropattern_xtb")
    parser.add_argument("--xtb-path", default="xtb")
    parser.add_argument("--solvent", default="water")
    parser.add_argument("--site-labeled-only", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    drugs = _load_drugs(Path(args.dataset))
    if args.supercyp_dataset:
        drugs.extend(_load_drugs(Path(args.supercyp_dataset)))
    if args.site_labeled_only:
        drugs = filter_site_labeled_drugs(drugs)

    unique_smiles = sorted({str(d.get("smiles", "")).strip() for d in drugs if str(d.get("smiles", "")).strip()})
    ok = failed = 0
    for idx, smiles in enumerate(unique_smiles, start=1):
        payload = load_or_compute_xtb_features(
            smiles,
            cache_dir=cache_dir,
            compute_if_missing=True,
            xtb_path=args.xtb_path,
            solvent=args.solvent,
        )
        if payload.get("xtb_valid"):
            ok += 1
        else:
            failed += 1
        if idx % 25 == 0 or idx == len(unique_smiles):
            print(f"{idx}/{len(unique_smiles)} processed | ok={ok} | failed={failed}", flush=True)


if __name__ == "__main__":
    main()
