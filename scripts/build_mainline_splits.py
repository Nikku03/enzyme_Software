from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.mainline.data.split_builder import build_mainline_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit mainline split artifacts.")
    parser.add_argument("--strict-exact-input", required=True)
    parser.add_argument("--tiered-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-prefix", default="mainline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--split-mode", default="scaffold_source_size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_mainline_splits(
        strict_exact_input=Path(args.strict_exact_input),
        tiered_input=Path(args.tiered_input),
        output_dir=Path(args.output_dir),
        dataset_prefix=str(args.dataset_prefix),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        split_mode=str(args.split_mode),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
