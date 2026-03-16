from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from lab import ROOT, load_config
from lab.data.vault import DataVault


def load_dataset(dataset_id: str):
    config = load_config()
    vault = DataVault(root=ROOT / config["data_vault"]["root"], processed_format=config["data_vault"]["processed_format"])
    return vault.load(dataset_id)


def load_metrics(metrics_path: str) -> Dict:
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))
