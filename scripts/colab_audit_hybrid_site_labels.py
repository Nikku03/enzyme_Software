"""
Colab helper for running the hybrid site-label audit in the same runtime used
for training.

Run from a Colab cell with:

    exec(open('/content/enzyme_Software/scripts/colab_audit_hybrid_site_labels.py').read())
"""
from __future__ import annotations

import importlib
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
    importlib.invalidate_caches()
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "RDKit installation completed but import still failed in the current runtime."
        ) from exc


def _clear_repo_python_caches() -> None:
    subprocess.run(
        ["find", str(REPO_DIR / "src"), "-name", "*.pyc", "-delete"],
        check=False,
    )
    subprocess.run(
        [
            "find",
            str(REPO_DIR / "src"),
            "-name",
            "__pycache__",
            "-type",
            "d",
            "-exec",
            "rm",
            "-rf",
            "{}",
            "+",
        ],
        check=False,
    )
    stale_modules = [
        name
        for name in list(sys.modules)
        if name == "enzyme_software"
        or name.startswith("enzyme_software.")
        or name == "audit_hybrid_site_labels"
        or name.startswith("scripts.audit_hybrid_site_labels")
    ]
    for name in stale_modules:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    print("Cleared repo bytecode and module caches.", flush=True)


def main() -> None:
    os.chdir(REPO_DIR)
    _clear_repo_python_caches()
    _ensure_rdkit()

    dataset = os.environ.get(
        "HYBRID_COLAB_LABEL_AUDIT_DATASET",
        os.environ.get(
            "HYBRID_COLAB_DATASET",
            "data/prepared_training/main5_site_conservative_singlecyp_clean.json",
        ),
    )
    artifact_dir = os.environ.get(
        "HYBRID_COLAB_ARTIFACT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/artifacts/hybrid_full_xtb",
    )
    stem = os.environ.get("HYBRID_COLAB_LABEL_AUDIT_STEM", "hybrid_label_audit")
    output_json = os.environ.get(
        "HYBRID_COLAB_LABEL_AUDIT_JSON",
        str(Path(artifact_dir) / f"{stem}.json"),
    )
    output_md = os.environ.get(
        "HYBRID_COLAB_LABEL_AUDIT_MD",
        str(Path(artifact_dir) / f"{stem}.md"),
    )

    print("Hybrid Label Audit Colab helper", flush=True)
    print(f"dataset={dataset}", flush=True)
    print(f"output_json={output_json}", flush=True)
    print(f"output_md={output_md}", flush=True)
    print()

    sys.argv = [
        str(REPO_DIR / "scripts" / "audit_hybrid_site_labels.py"),
        "--dataset",
        dataset,
        "--output-json",
        output_json,
        "--output-md",
        output_md,
    ]
    runpy.run_path(str(REPO_DIR / "scripts" / "audit_hybrid_site_labels.py"), run_name="__main__")


if __name__ == "__main__":
    main()
