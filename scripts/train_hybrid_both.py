#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_bundle(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid+both from a canonical bundle manifest.")
    parser.add_argument("--bundle", default="data/hybrid_both_bundle/bundle.json")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--xtb-cache-dir", default="")
    parser.add_argument("--output-dir", default="checkpoints/hybrid_both")
    parser.add_argument("--artifact-dir", default="artifacts/hybrid_both")
    parser.add_argument("--warm-start", default="")
    parser.add_argument("--auto-resume-latest", action="store_true")
    parser.add_argument("--compute-xtb-if-missing", action="store_true")
    parser.add_argument("--disable-nexus-bridge", action="store_true")
    parser.add_argument("--freeze-nexus-memory", action="store_true")
    parser.add_argument("--skip-nexus-memory-rebuild", action="store_true")
    parser.add_argument("--site-labeled-only", action="store_true", default=True)
    parser.add_argument("--no-site-labeled-only", dest="site_labeled_only", action="store_false")
    parser.add_argument("--no-xenosite", action="store_true")
    parser.add_argument("--xenosite-topk", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=-1)
    args = parser.parse_args()

    bundle_path = ROOT / args.bundle
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    bundle = _load_bundle(bundle_path)
    defaults = bundle["defaults"]
    paths = bundle["paths"]

    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.warm_start
    if not checkpoint and args.auto_resume_latest:
        candidate = output_dir / "hybrid_full_xtb_latest.pt"
        if candidate.exists():
            checkpoint = str(candidate)
    if not checkpoint:
        checkpoint = str(output_dir / "hybrid_full_xtb_latest.pt")

    xtb_cache_dir = args.xtb_cache_dir or defaults["xtb_cache_dir"]
    argv = [
        str(ROOT / "scripts" / "train_hybrid_full_xtb.py"),
        "--dataset",
        paths["base_dataset"],
        "--structure-sdf",
        paths["structure_sdf"],
        "--checkpoint",
        checkpoint,
        "--xtb-cache-dir",
        xtb_cache_dir,
        "--epochs",
        str(args.epochs or defaults["epochs"]),
        "--batch-size",
        str(args.batch_size or defaults["batch_size"]),
        "--learning-rate",
        str(args.learning_rate or defaults["learning_rate"]),
        "--weight-decay",
        str(args.weight_decay or defaults["weight_decay"]),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--early-stopping-patience",
        str(args.early_stopping_patience if args.early_stopping_patience >= 0 else defaults["early_stopping_patience"]),
    ]
    if args.device:
        argv.extend(["--device", args.device])
    if args.limit > 0:
        argv.extend(["--limit", str(args.limit)])
    if args.site_labeled_only:
        argv.append("--site-labeled-only")
    if args.compute_xtb_if_missing or bool(defaults["compute_xtb_if_missing"]):
        argv.append("--compute-xtb-if-missing")
    if args.disable_nexus_bridge or bool(defaults["disable_nexus_bridge"]):
        argv.append("--disable-nexus-bridge")
    if args.freeze_nexus_memory or bool(defaults["freeze_nexus_memory"]):
        argv.append("--freeze-nexus-memory")
    if args.skip_nexus_memory_rebuild or not bool(defaults["rebuild_nexus_memory"]):
        argv.append("--skip-nexus-memory-rebuild")
    if not args.no_xenosite and bool(defaults["include_xenosite"]):
        argv.extend(["--xenosite-manifest", paths["xenosite_manifest"]])
        argv.extend(["--xenosite-topk", str(args.xenosite_topk or defaults["xenosite_topk"])])

    print("Training hybrid+both bundle")
    print(f"bundle={bundle_path.relative_to(ROOT)}")
    print(f"dataset={paths['base_dataset']}")
    print(f"xenosite={'off' if args.no_xenosite else paths['xenosite_manifest']}")
    print(f"xtb_cache_dir={xtb_cache_dir}")
    print(f"output_dir={output_dir}")
    print(f"artifact_dir={artifact_dir}")
    print(f"warm_start={checkpoint}")

    os.chdir(ROOT)
    sys.argv = argv
    runpy.run_path(str(ROOT / "scripts" / "train_hybrid_full_xtb.py"), run_name="__main__")


if __name__ == "__main__":
    main()
