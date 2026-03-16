from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from enzyme_software.data_acquisition.http import HttpClient
from enzyme_software.data_acquisition.sources.base import Collector

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_to_task(seed: str) -> Dict[str, str]:
    cleaned = seed.strip()
    return {"kind": "molecule", "value": cleaned}


def _task_url(task: Dict[str, str]) -> str:
    value = urllib.parse.quote(task["value"])
    return (
        f"{CHEMBL_BASE}/activity.json?molecule_chembl_id={value}&limit=50"
    )


@dataclass
class ChEMBLCollector(Collector):
    seeds: List[str]
    client: HttpClient

    def list_tasks(self) -> Iterable[Dict[str, str]]:
        return [_seed_to_task(seed) for seed in self.seeds if seed.strip()]

    def fetch(self, task: Dict[str, str]) -> bytes:
        url = _task_url(task)
        return self.client.get(url)

    def parse(self, raw: bytes) -> Dict[str, Any]:
        payload = json.loads(raw.decode("utf-8"))
        activities = payload.get("activities", [])
        if not isinstance(activities, list):
            raise ValueError("ChEMBL response missing activities list.")
        return {"activities": activities}

    def normalize(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        activities = parsed.get("activities", [])
        normalized: List[Dict[str, Any]] = []
        for entry in activities:
            normalized.append(
                {
                    "target_chembl_id": entry.get("target_chembl_id"),
                    "assay_type": entry.get("assay_type"),
                    "standard_value": entry.get("standard_value"),
                    "standard_units": entry.get("standard_units"),
                    "relation": entry.get("relation"),
                    "pchembl_value": entry.get("pchembl_value"),
                    "molecule_smiles": entry.get("canonical_smiles")
                    or entry.get("molecule_smiles")
                    or entry.get("smiles"),
                    "metadata": {
                        "activity_id": entry.get("activity_id"),
                        "assay_description": entry.get("assay_description"),
                        "document_chembl_id": entry.get("document_chembl_id"),
                        "published_type": entry.get("published_type"),
                        "published_relation": entry.get("published_relation"),
                        "published_value": entry.get("published_value"),
                        "published_units": entry.get("published_units"),
                        "source": "chembl",
                        "retrieved_at": _now_iso(),
                    },
                }
            )
        return normalized
