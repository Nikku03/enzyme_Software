from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "lab_config.yaml"


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    path = Path(config_path or os.getenv("LAB_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_artifacts_root() -> Path:
    return ROOT / "artifacts"
