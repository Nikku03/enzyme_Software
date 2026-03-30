"""
Dedicated Colab entrypoint for the hybrid LNN with both imported NEXUS parts:

- wave / quantum bridge
- analogical memory bank

Run from a Colab cell with:

    exec(open('/content/enzyme_Software/scripts/colab_train_hybrid_lnn.py').read())
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_DIR = Path("/content/enzyme_Software")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


PRESETS: dict[str, dict[str, str]] = {
    "fast": {
        "HYBRID_COLAB_DATASET": "data/training_dataset_580.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "3",
        "HYBRID_COLAB_BATCH_SIZE": "16",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_DISABLE_3D": "0",
        "HYBRID_COLAB_DISABLE_NEXUS_BRIDGE": "0",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_SKIP_MEMORY_REBUILD": "0",
        "HYBRID_COLAB_SEED": "42",
    },
    "balanced": {
        "HYBRID_COLAB_DATASET": "data/training_dataset_580.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "12",
        "HYBRID_COLAB_BATCH_SIZE": "24",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_DISABLE_3D": "0",
        "HYBRID_COLAB_DISABLE_NEXUS_BRIDGE": "0",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_SKIP_MEMORY_REBUILD": "0",
        "HYBRID_COLAB_SEED": "42",
    },
    "full": {
        "HYBRID_COLAB_DATASET": "data/training_dataset_580.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "25",
        "HYBRID_COLAB_BATCH_SIZE": "32",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_DISABLE_3D": "0",
        "HYBRID_COLAB_DISABLE_NEXUS_BRIDGE": "0",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_SKIP_MEMORY_REBUILD": "0",
        "HYBRID_COLAB_SEED": "42",
    },
}


def main() -> None:
    preset = os.environ.get("HYBRID_COLAB_PRESET", "balanced").strip().lower() or "balanced"
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown HYBRID_COLAB_PRESET={preset!r}. Valid presets: {valid}")

    for key, value in PRESETS[preset].items():
        _setdefault_env(key, value)

    output_dir = os.environ.get(
        "HYBRID_COLAB_OUTPUT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints",
    )
    manual_cache_dir = os.environ.get(
        "HYBRID_COLAB_MANUAL_CACHE_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/manual_engine_full",
    )

    argv = [
        str(REPO_DIR / "scripts" / "train_hybrid_lnn.py"),
        "--dataset",
        os.environ["HYBRID_COLAB_DATASET"],
        "--structure-sdf",
        os.environ["HYBRID_COLAB_STRUCTURE_SDF"],
        "--epochs",
        os.environ["HYBRID_COLAB_EPOCHS"],
        "--batch-size",
        os.environ["HYBRID_COLAB_BATCH_SIZE"],
        "--learning-rate",
        os.environ["HYBRID_COLAB_LR"],
        "--weight-decay",
        os.environ["HYBRID_COLAB_WD"],
        "--seed",
        os.environ["HYBRID_COLAB_SEED"],
        "--output-dir",
        output_dir,
        "--manual-feature-cache-dir",
        manual_cache_dir,
        "--auto-resume-latest",
    ]

    if os.environ.get("HYBRID_COLAB_DISABLE_3D", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--disable-3d-branch")
    if os.environ.get("HYBRID_COLAB_DISABLE_NEXUS_BRIDGE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--disable-nexus-bridge")
    if os.environ.get("HYBRID_COLAB_FREEZE_NEXUS_MEMORY", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--freeze-nexus-memory")
    if os.environ.get("HYBRID_COLAB_SKIP_MEMORY_REBUILD", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--skip-nexus-memory-rebuild")

    print("Hybrid LNN Colab wrapper")
    print(f"preset={preset}")
    print(f"output_dir={output_dir}")
    print(f"manual_cache_dir={manual_cache_dir}")
    for key in sorted(PRESETS[preset]):
        print(f"{key}={os.environ[key]}")
    print()

    sys.argv = argv
    runpy.run_path(str(REPO_DIR / "scripts" / "train_hybrid_lnn.py"), run_name="__main__")


main()
