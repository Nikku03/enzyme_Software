from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from enzyme_software.data_acquisition.http import HttpClient
from enzyme_software.data_acquisition.models import NormalizedMolecule
from enzyme_software.data_acquisition.provenance import Provenance
from enzyme_software.data_acquisition.sources.base import Collector

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_to_task(seed: str) -> Dict[str, str]:
    cleaned = seed.strip()
    if cleaned.isdigit():
        return {"kind": "cid", "value": cleaned}
    return {"kind": "name", "value": cleaned}


def _task_url(task: Dict[str, str]) -> str:
    value = urllib.parse.quote(task["value"])
    if task["kind"] == "cid":
        return f"{PUG_REST_BASE}/compound/cid/{value}/property/CanonicalSMILES,InChIKey/JSON"
    return f"{PUG_REST_BASE}/compound/name/{value}/property/CanonicalSMILES,InChIKey/JSON"


@dataclass
class PubChemCollector(Collector):
    seeds: List[str]
    client: HttpClient

    def list_tasks(self) -> Iterable[Dict[str, str]]:
        return [_seed_to_task(seed) for seed in self.seeds if seed.strip()]

    def fetch(self, task: Dict[str, str]) -> bytes:
        url = _task_url(task)
        return self.client.get(url)

    def parse(self, raw: bytes) -> Dict[str, Any]:
        payload = json.loads(raw.decode("utf-8"))
        props = payload.get("PropertyTable", {}).get("Properties", [])
        if not props:
            raise ValueError("No properties found in PubChem response.")
        return props[0]

    def normalize(self, parsed: Dict[str, Any]) -> NormalizedMolecule:
        smiles = parsed.get("CanonicalSMILES")
        inchikey = parsed.get("InChIKey")
        cid = parsed.get("CID")
        provenance = Provenance(
            source="pubchem",
            source_id=str(cid) if cid is not None else None,
            url=None,
            retrieved_at=_now_iso(),
            license_hint="PubChem",
            sha256=None,
        )
        return NormalizedMolecule(
            name=None,
            smiles=smiles,
            role=None,
            provenance=provenance,
        )
