"""
Dedicated Colab entrypoint for the hybrid LNN with both imported NEXUS parts
running on the full-xTB hybrid path.

This is the correct Colab route for:
- analogical memory bank
- wave / quantum bridge

because it feeds real xTB atom features into the bridge instead of starving the
wave side on the plain baseline loader.

Run from a Colab cell with:

    exec(open('/content/enzyme_Software/scripts/colab_train_hybrid_lnn.py').read())
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
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _ensure_rdkit() -> None:
    try:
        from rdkit import Chem  # noqa: F401
        return
    except Exception:
        pass
    print("RDKit not found. Installing rdkit-pypi for this Colab runtime...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"],
        cwd=str(REPO_DIR),
    )
    import importlib

    importlib.invalidate_caches()
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime bootstrap failure
        raise RuntimeError(
            "RDKit installation completed but import still failed in the current runtime."
        ) from exc


PRESETS: dict[str, dict[str, str]] = {
    "fast": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main5_site_conservative_singlecyp_clean.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "3",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_LIMIT": "128",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "0",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "1",
        "HYBRID_COLAB_XENOSITE_TOPK": "1",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_SEED": "42",
    },
    "balanced": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main5_site_conservative_singlecyp_clean.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "12",
        "HYBRID_COLAB_BATCH_SIZE": "16",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "0",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "1",
        "HYBRID_COLAB_XENOSITE_TOPK": "1",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_SEED": "42",
    },
    "full": {
        "HYBRID_COLAB_DATASET": "data/combined_drugbank_supercyp_full_xtb_valid_site_labeled.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "25",
        "HYBRID_COLAB_BATCH_SIZE": "24",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "1",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "0",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "1",
        "HYBRID_COLAB_XENOSITE_TOPK": "1",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_SEED": "42",
    },
}


def main() -> None:
    os.chdir(REPO_DIR)
    os.environ["PYTHONPATH"] = f"{SRC_DIR}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    _ensure_rdkit()
    preset = os.environ.get("HYBRID_COLAB_PRESET", "balanced").strip().lower() or "balanced"
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown HYBRID_COLAB_PRESET={preset!r}. Valid presets: {valid}")

    for key, value in PRESETS[preset].items():
        _setdefault_env(key, value)

    requested_dataset = Path(os.environ["HYBRID_COLAB_DATASET"])
    if not requested_dataset.exists():
        fallback_candidates = [
            REPO_DIR / "data" / "prepared_training" / "main5_site_conservative_singlecyp_clean.json",
            REPO_DIR / "data" / "combined_drugbank_supercyp_full_xtb_valid_site_labeled.json",
            REPO_DIR / "data" / "training_dataset_580.json",
        ]
        fallback = next((path for path in fallback_candidates if path.exists()), None)
        if fallback is None:
            raise FileNotFoundError(
                f"Requested dataset not found: {requested_dataset}. No tracked fallback dataset found in repo."
            )
        print(f"Requested dataset not found: {requested_dataset}")
        print(f"Falling back to tracked dataset: {fallback.relative_to(REPO_DIR)}")
        os.environ["HYBRID_COLAB_DATASET"] = str(fallback.relative_to(REPO_DIR))

    output_dir = os.environ.get(
        "HYBRID_COLAB_OUTPUT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/hybrid_full_xtb",
    )
    artifact_dir = os.environ.get(
        "HYBRID_COLAB_ARTIFACT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/artifacts/hybrid_full_xtb",
    )
    manual_cache_dir = os.environ.get(
        "HYBRID_COLAB_MANUAL_CACHE_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/manual_engine_full",
    )
    xtb_cache_dir = os.environ.get(
        "HYBRID_COLAB_XTB_CACHE_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/full_xtb",
    )
    checkpoint = os.environ.get(
        "HYBRID_COLAB_WARM_START",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/hybrid_full_xtb/hybrid_full_xtb_latest.pt",
    )
    xenosite_manifest = os.environ.get(
        "HYBRID_COLAB_XENOSITE_MANIFEST",
        "data/xenosite_suppl/manifest.json",
    )

    argv = [
        str(REPO_DIR / "scripts" / "train_hybrid_full_xtb.py"),
        "--dataset",
        os.environ["HYBRID_COLAB_DATASET"],
        "--structure-sdf",
        os.environ["HYBRID_COLAB_STRUCTURE_SDF"],
        "--checkpoint",
        checkpoint,
        "--xtb-cache-dir",
        xtb_cache_dir,
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
        "--artifact-dir",
        artifact_dir,
        "--manual-feature-cache-dir",
        manual_cache_dir,
        "--early-stopping-patience",
        os.environ["HYBRID_COLAB_EARLY_STOPPING_PATIENCE"],
    ]

    limit = int(os.environ.get("HYBRID_COLAB_LIMIT", "0") or "0")
    if limit > 0:
        argv.extend(["--limit", str(limit)])
    if os.environ.get("HYBRID_COLAB_SITE_LABELED_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--site-labeled-only")
    if os.environ.get("HYBRID_COLAB_COMPUTE_XTB_IF_MISSING", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--compute-xtb-if-missing")
    if os.environ.get("HYBRID_COLAB_INCLUDE_XENOSITE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.extend(["--xenosite-manifest", xenosite_manifest])
        argv.extend(["--xenosite-topk", os.environ["HYBRID_COLAB_XENOSITE_TOPK"]])
    if os.environ.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--disable-precedent-logbook")
    precedent_logbook = os.environ.get("HYBRID_COLAB_PRECEDENT_LOGBOOK", "").strip()
    if precedent_logbook:
        argv.extend(["--precedent-logbook", precedent_logbook])

    print("Hybrid LNN Colab wrapper")
    print(f"preset={preset}")
    print(f"output_dir={output_dir}")
    print(f"artifact_dir={artifact_dir}")
    print(f"manual_cache_dir={manual_cache_dir}")
    print(f"xtb_cache_dir={xtb_cache_dir}")
    print(f"warm_start={checkpoint}")
    print(f"disable_precedent_logbook={os.environ.get('HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK', '1')}")
    if precedent_logbook:
        print(f"precedent_logbook={precedent_logbook}")
    for key in sorted(PRESETS[preset]):
        print(f"{key}={os.environ[key]}")
    print()

    sys.argv = argv
    runpy.run_path(str(REPO_DIR / "scripts" / "train_hybrid_full_xtb.py"), run_name="__main__")


main()
