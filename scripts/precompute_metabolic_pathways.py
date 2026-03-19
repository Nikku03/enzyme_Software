from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_xtb_features
from enzyme_software.recursive_metabolism import PathwayGenerator
from enzyme_software.recursive_metabolism.utils import filter_site_labeled_drugs, load_drugs


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute recursive metabolic pathways")
    parser.add_argument("--dataset", default="data/training_dataset_drugbank.json")
    parser.add_argument("--output", default="cache/recursive_metabolism/pathways.json")
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--min-heavy-atoms", type=int, default=5)
    parser.add_argument("--pseudo-step-weight", type=float, default=0.35)
    parser.add_argument("--xtb-cache-dir", default=None)
    parser.add_argument("--xtb-path", default="xtb")
    parser.add_argument("--solvent", default="water")
    args = parser.parse_args()

    drugs = load_drugs(args.dataset)
    if args.site_labeled_only:
        drugs = filter_site_labeled_drugs(drugs)
    print(f"Loaded {len(drugs)} drugs", flush=True)

    generator = PathwayGenerator(
        max_steps=args.max_steps,
        min_heavy_atoms=args.min_heavy_atoms,
        pseudo_step_weight=args.pseudo_step_weight,
    )
    pathways = generator.generate_dataset(drugs, output_path=args.output)
    total_steps = sum(pathway.total_steps for pathway in pathways)
    report = {
        "num_drugs": len(pathways),
        "total_steps": int(total_steps),
        "expansion_factor": float(total_steps / max(1, len(pathways))),
    }
    if args.xtb_cache_dir:
        smiles_set = set()
        for pathway in pathways:
            if pathway.drug_smiles:
                smiles_set.add(str(pathway.drug_smiles))
            if pathway.terminal_metabolite:
                smiles_set.add(str(pathway.terminal_metabolite))
            for step in pathway.steps:
                smiles_set.add(str(step.parent_smiles))
                smiles_set.add(str(step.metabolite_smiles))
        ok = failed = 0
        for idx, smiles in enumerate(sorted(smiles_set), start=1):
            payload = load_or_compute_xtb_features(
                smiles,
                cache_dir=args.xtb_cache_dir,
                compute_if_missing=True,
                xtb_path=args.xtb_path,
                solvent=args.solvent,
            )
            if payload.get("xtb_valid"):
                ok += 1
            else:
                failed += 1
            if idx % 25 == 0 or idx == len(smiles_set):
                print(f"xTB {idx}/{len(smiles_set)} | ok={ok} | failed={failed}", flush=True)
        report["xtb_cache"] = {
            "cache_dir": args.xtb_cache_dir,
            "unique_smiles": len(smiles_set),
            "ok": ok,
            "failed": failed,
        }
    report_path = Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
