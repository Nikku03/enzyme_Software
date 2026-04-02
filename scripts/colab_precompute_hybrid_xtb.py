"""
Dedicated Colab entrypoint for precomputing full xTB cache used by the hybrid
LNN + wave path.

Run from a Colab cell with:

    exec(open('/content/enzyme_Software/scripts/colab_precompute_hybrid_xtb.py').read())
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
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "RDKit installation completed but import still failed in the current runtime."
        ) from exc


PRESETS: dict[str, dict[str, str]] = {
    "balanced": {
        "HYBRID_COLAB_XTB_DATASET": "data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
        "HYBRID_COLAB_XTB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_XTB_LIMIT": "0",
        "HYBRID_COLAB_XTB_SEED": "42",
    },
    "full": {
        "HYBRID_COLAB_XTB_DATASET": "data/combined_drugbank_supercyp_full_xtb_valid_site_labeled.json",
        "HYBRID_COLAB_XTB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_XTB_LIMIT": "0",
        "HYBRID_COLAB_XTB_SEED": "42",
    },
}


def main() -> None:
    os.chdir(REPO_DIR)
    os.environ["PYTHONPATH"] = f"{SRC_DIR}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    _ensure_rdkit()

    preset = os.environ.get("HYBRID_COLAB_XTB_PRESET", "balanced").strip().lower() or "balanced"
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown HYBRID_COLAB_XTB_PRESET={preset!r}. Valid presets: {valid}")
    for key, value in PRESETS[preset].items():
        _setdefault_env(key, value)

    requested_dataset = Path(os.environ["HYBRID_COLAB_XTB_DATASET"])
    if not requested_dataset.exists():
        raise FileNotFoundError(
            "Requested dataset not found: "
            f"{requested_dataset}. Set HYBRID_COLAB_XTB_DATASET to an existing path before launching precompute."
        )

    cache_dir = os.environ.get(
        "HYBRID_COLAB_XTB_CACHE_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/full_xtb",
    )

    argv = [
        str(REPO_DIR / "scripts" / "precompute_full_xtb.py"),
        "--dataset",
        os.environ["HYBRID_COLAB_XTB_DATASET"],
        "--cache-dir",
        cache_dir,
        "--seed",
        os.environ["HYBRID_COLAB_XTB_SEED"],
    ]

    if os.environ.get("HYBRID_COLAB_XTB_SITE_LABELED_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--site-labeled-only")
    limit = int(os.environ.get("HYBRID_COLAB_XTB_LIMIT", "0") or "0")
    if limit > 0:
        argv.extend(["--limit", str(limit)])

    print("Hybrid full-xTB cache precompute wrapper")
    print(f"preset={preset}")
    print(f"dataset={os.environ['HYBRID_COLAB_XTB_DATASET']}")
    print(f"cache_dir={cache_dir}")
    print(f"site_labeled_only={os.environ.get('HYBRID_COLAB_XTB_SITE_LABELED_ONLY', '')}")
    print(f"limit={os.environ.get('HYBRID_COLAB_XTB_LIMIT', '')}")
    print(f"seed={os.environ.get('HYBRID_COLAB_XTB_SEED', '')}")
    print()

    sys.argv = argv
    runpy.run_path(str(REPO_DIR / "scripts" / "precompute_full_xtb.py"), run_name="__main__")


main()
