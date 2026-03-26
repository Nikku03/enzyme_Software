"""
Dedicated Colab entrypoint for CYP3A4-focused NEXUS training.

Run from a Colab cell with:

    exec(open("/content/enzyme_Software/scripts/colab_train_cyp3a4.py").read())

This wrapper configures a sensible 3A4-only training setup, keeps the broader
analogical memory bank from the main trainer, and then delegates to
`scripts/colab_train.py`.

Presets
-------
Set `NEXUS_COLAB_RUN_PRESET` before executing this file:

    fast      : quickest useful 3A4 debug run
    balanced  : default; good iteration speed / fidelity balance
    full_3a4  : all 3A4 molecules with a trimmed full-physics budget
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path("/content/enzyme_Software")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _ensure_colab_nexus_assets() -> None:
    target_sdf = REPO_DIR / "data" / "ATTNSOM" / "cyp_dataset" / "3A4.sdf"
    setup_script = REPO_DIR / "scripts" / "setup_colab_nexus.sh"
    if target_sdf.exists():
        return
    if not setup_script.exists():
        raise FileNotFoundError(
            f"Missing setup bootstrap script: {setup_script}"
        )
    print("ATTNSOM CYP SDF assets not found; running Colab bootstrap...")
    subprocess.run(
        ["bash", str(setup_script), str(REPO_DIR)],
        check=True,
    )
    if not target_sdf.exists():
        raise FileNotFoundError(
            f"Dataset bootstrap completed but {target_sdf} is still missing."
        )


PRESETS: dict[str, dict[str, str]] = {
    "fast": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_MAX_SAMPLES": "32",
        "NEXUS_COLAB_EPOCHS": "3",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "128",
        "NEXUS_COLAB_SCAN_N_POINTS": "16",
        "NEXUS_COLAB_SCAN_RADIUS": "1.25",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.40,0.65,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "0",
        "NEXUS_COLAB_NAV_OPT_STEPS": "1",
        "NEXUS_COLAB_NAV_CANDIDATES": "2",
    },
    "balanced": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_MAX_SAMPLES": "64",
        "NEXUS_COLAB_EPOCHS": "4",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "128",
        "NEXUS_COLAB_SCAN_N_POINTS": "12",
        "NEXUS_COLAB_SCAN_RADIUS": "1.20",
        "NEXUS_COLAB_SCAN_CHUNK": "6",
        "NEXUS_COLAB_SCAN_SHELLS": "0.40,0.70,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "0",
        "NEXUS_COLAB_NAV_OPT_STEPS": "1",
        "NEXUS_COLAB_NAV_CANDIDATES": "2",
    },
    "full_3a4": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_MAX_SAMPLES": "0",
        "NEXUS_COLAB_EPOCHS": "5",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "160",
        "NEXUS_COLAB_SCAN_N_POINTS": "16",
        "NEXUS_COLAB_SCAN_RADIUS": "1.35",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.40,0.65,0.85,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "2",
        "NEXUS_COLAB_NAV_CANDIDATES": "3",
    },
}


def main() -> None:
    preset = os.environ.get("NEXUS_COLAB_RUN_PRESET", "balanced").strip().lower() or "balanced"
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown NEXUS_COLAB_RUN_PRESET={preset!r}. Valid presets: {valid}")

    for key, value in PRESETS[preset].items():
        _setdefault_env(key, value)

    print("NEXUS Colab CYP3A4 wrapper")
    print(f"preset={preset}")
    for key in sorted(PRESETS[preset]):
        print(f"{key}={os.environ[key]}")
    print()


main()
_ensure_colab_nexus_assets()
runpy.run_path(str(REPO_DIR / "scripts" / "colab_train.py"), run_name="__main__")
