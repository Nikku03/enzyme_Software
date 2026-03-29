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


def _maybe_enable_physics_cache() -> None:
    raw = os.environ.get("NEXUS_COLAB_USE_PHYSICS_CACHE", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return
    _setdefault_env("NEXUS_COLAB_PHYSICS_CACHE_MODE", "hybrid")
    _setdefault_env(
        "NEXUS_COLAB_PHYSICS_CACHE_PATH",
        "/content/drive/MyDrive/nexus_cyp3a4_physics_cache.pt",
    )


PRESETS: dict[str, dict[str, str]] = {
    # Quick smoke-test / debug run — 5 epochs, 32 molecules.
    # DAG warmup = 50 steps (all 5 epochs worth) so DAG never fully ramps
    # during a short debug run; SoM signal stays dominant throughout.
    "fast": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_ANALOGICAL_BANK_MODE": "continuous",
        "NEXUS_COLAB_MAX_SAMPLES": "32",
        "NEXUS_COLAB_EPOCHS": "5",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "128",
        "NEXUS_COLAB_SCAN_N_POINTS": "20",
        "NEXUS_COLAB_SCAN_RADIUS": "1.35",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.35,0.60,0.85,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "1",
        "NEXUS_COLAB_NAV_CANDIDATES": "2",
        "NEXUS_COLAB_DAG_LOSS_WEIGHT": "0.10",
        "NEXUS_COLAB_DAG_LOSS_CAP": "1.0",
        "NEXUS_COLAB_DAG_WARMUP_STEPS": "50",
        "NEXUS_COLAB_KINETIC_WARMUP_STEPS": "32",
        "NEXUS_COLAB_ANA_LOSS_WEIGHT": "0.25",
    },
    # Default 10-epoch run on any ultra_vram GPU.
    # All 391 CYP3A4 molecules. DAG warmup = 400 steps (~1 epoch).
    # Kinetic warmup = 475 steps (1 epoch): lets SIREN find SoM structure before
    # rate prediction penalty kicks in.
    "balanced": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_ANALOGICAL_BANK_MODE": "continuous",
        "NEXUS_COLAB_MAX_SAMPLES": "0",
        "NEXUS_COLAB_EPOCHS": "10",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "160",
        "NEXUS_COLAB_SCAN_N_POINTS": "24",
        "NEXUS_COLAB_SCAN_RADIUS": "1.40",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.35,0.60,0.80,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "2",
        "NEXUS_COLAB_NAV_CANDIDATES": "3",
        "NEXUS_COLAB_DAG_LOSS_WEIGHT": "0.10",
        "NEXUS_COLAB_DAG_LOSS_CAP": "1.0",
        "NEXUS_COLAB_DAG_WARMUP_STEPS": "400",
        "NEXUS_COLAB_KINETIC_WARMUP_STEPS": "475",
        "NEXUS_COLAB_ANA_LOSS_WEIGHT": "0.25",
    },
    # 15-epoch full run on a Colab A100/H100 (80 GB).
    # All 391 CYP3A4 molecules, 4 scan shells, 1 refine step.
    # DAG warmup = 800 steps (~2 epochs). Kinetic warmup = 475 steps (1 epoch).
    "full_3a4": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_ANALOGICAL_BANK_MODE": "continuous",
        "NEXUS_COLAB_MAX_SAMPLES": "0",
        "NEXUS_COLAB_EPOCHS": "15",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "160",
        "NEXUS_COLAB_SCAN_N_POINTS": "24",
        "NEXUS_COLAB_SCAN_RADIUS": "1.50",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.35,0.60,0.80,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "2",
        "NEXUS_COLAB_NAV_CANDIDATES": "3",
        "NEXUS_COLAB_DAG_LOSS_WEIGHT": "0.10",
        "NEXUS_COLAB_DAG_LOSS_CAP": "1.0",
        "NEXUS_COLAB_DAG_WARMUP_STEPS": "800",
        "NEXUS_COLAB_KINETIC_WARMUP_STEPS": "475",
        "NEXUS_COLAB_ANA_LOSS_WEIGHT": "0.25",
    },
    # 30-epoch cloud run on single A100 80GB or H100 80GB.
    # Full physics: dynamics_steps=2, integration_resolution=12, scan_n_points=32.
    # DAG warmup = 1200 steps (~3 epochs). Kinetic warmup = 475 steps (1 epoch).
    # torch.compile enabled. NOTE: current code is single-GPU only.
    "full_3a4_a100": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_ANALOGICAL_BANK_MODE": "continuous",
        "NEXUS_COLAB_MAX_SAMPLES": "0",
        "NEXUS_COLAB_EPOCHS": "30",
        "NEXUS_COLAB_DYNAMICS_STEPS": "2",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "12",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "192",
        "NEXUS_COLAB_SCAN_N_POINTS": "32",
        "NEXUS_COLAB_SCAN_RADIUS": "1.75",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.30,0.55,0.75,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "6",
        "NEXUS_COLAB_NAV_CANDIDATES": "8",
        "NEXUS_COLAB_DAG_LOSS_WEIGHT": "0.10",
        "NEXUS_COLAB_DAG_LOSS_CAP": "1.0",
        "NEXUS_COLAB_DAG_WARMUP_STEPS": "1200",
        "NEXUS_COLAB_KINETIC_WARMUP_STEPS": "475",
        "NEXUS_COLAB_ANA_LOSS_WEIGHT": "0.25",
        "NEXUS_COLAB_ALLOW_COMPILE": "1",
        "NEXUS_COLAB_NUM_WORKERS": "2",
    },
    # 25 epochs on a single RTX 6000 Ada (95 GB) targeting ~2-3 hours.
    # Key levers vs full_3a4_a100:
    #   1. torch.compile on SIREN field (the hot loop) → 2-4× SIREN speedup
    #   2. TF32 matmuls enabled at startup → 20-30% free
    #   3. Physics curriculum (NEXUS_COLAB_CURRICULUM=1):
    #        epochs 1-9   (lite)   : res=4, scan_pts=6, 1 outer shell
    #        epochs 10-18 (medium) : res=6, scan_pts=8, both shells
    #        epochs 19-25 (full)   : res=8, scan_pts=10, both shells
    #   4. num_workers=2: background data loading overlaps GPU compute
    #   5. integration_chunk=256: large GPU passes for better occupancy
    #   6. dynamics_steps=1 (vs 2 in full_3a4_a100) — halves dynamics cost
    #   7. DAG warmup = 1000 steps (~2.5 epochs). Kinetic warmup = 475 steps (1 epoch).
    "rtx6k_2h": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_ANALOGICAL_BANK_MODE": "continuous",
        "NEXUS_COLAB_MAX_SAMPLES": "0",
        "NEXUS_COLAB_EPOCHS": "25",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "8",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "256",
        "NEXUS_COLAB_SCAN_N_POINTS": "24",
        "NEXUS_COLAB_SCAN_RADIUS": "1.50",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.35,0.60,0.80,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "2",
        "NEXUS_COLAB_NAV_CANDIDATES": "3",
        "NEXUS_COLAB_DAG_LOSS_WEIGHT": "0.10",
        "NEXUS_COLAB_DAG_LOSS_CAP": "1.0",
        "NEXUS_COLAB_DAG_WARMUP_STEPS": "1000",
        "NEXUS_COLAB_KINETIC_WARMUP_STEPS": "475",
        "NEXUS_COLAB_ANA_LOSS_WEIGHT": "0.25",
        "NEXUS_COLAB_ALLOW_COMPILE": "1",
        "NEXUS_COLAB_NUM_WORKERS": "2",
        "NEXUS_COLAB_CURRICULUM": "1",
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
_maybe_enable_physics_cache()
runpy.run_path(str(REPO_DIR / "scripts" / "colab_train.py"), run_name="__main__")
