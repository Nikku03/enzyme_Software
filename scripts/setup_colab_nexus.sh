#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/content/enzyme_Software}"
ATTNSOM_DIR="${REPO_DIR}/data/ATTNSOM"

echo "[setup] repo dir: ${REPO_DIR}"
cd "${REPO_DIR}"

python -m pip install --upgrade pip setuptools wheel

# Keep Colab's preinstalled torch when available; install only the missing pieces
python - <<'PY'
import importlib.util
import subprocess
import sys

def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

packages = []
if not has("numpy"):
    packages.append("numpy>=1.26")
if not has("scipy"):
    packages.append("scipy>=1.13")
if not has("ot"):
    packages.append("pot>=0.9.6")
if not has("galore_torch"):
    packages.append("galore-torch")

if packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

if not has("rdkit"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
PY

python -m pip install -e .

mkdir -p data
if [[ ! -d "${ATTNSOM_DIR}" ]]; then
  echo "[setup] cloning ATTNSOM dataset repo"
  git clone --depth 1 https://github.com/dmis-lab/ATTNSOM.git "${ATTNSOM_DIR}"
else
  echo "[setup] ATTNSOM dataset repo already present"
fi

echo "[setup] complete"
