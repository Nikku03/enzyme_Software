from __future__ import annotations

import runpy
import sys
from pathlib import Path


def dispatch_master_builder(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[4]
    target = root / "scripts" / "build_all_family_merged_dataset.py"
    args = [str(target), *(argv or [])]
    old_argv = list(sys.argv)
    try:
        sys.argv = args
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old_argv
