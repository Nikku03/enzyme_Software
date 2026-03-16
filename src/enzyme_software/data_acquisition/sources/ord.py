from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from enzyme_software.data_acquisition.models import NormalizedCondition, NormalizedReaction
from enzyme_software.data_acquisition.sources.base import Collector


def _iter_records(path: Path) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                yield path, idx, json.loads(line)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for idx, entry in enumerate(payload):
                yield path, idx, entry
        elif isinstance(payload, dict):
            for idx, entry in enumerate(payload.get("reactions", [])):
                yield path, idx, entry


def _extract_smiles(items: List[Dict[str, Any]]) -> List[str]:
    smiles_list = []
    for item in items or []:
        value = item.get("smiles") or item.get("canonical_smiles")
        if value:
            smiles_list.append(value)
    return smiles_list


def _parse_conditions(entry: Dict[str, Any]) -> NormalizedCondition:
    conditions = entry.get("conditions") or {}
    return NormalizedCondition(
        pH=conditions.get("ph") or conditions.get("pH"),
        temperature_C=conditions.get("temperature_c") or conditions.get("temperature_C"),
        solvent=conditions.get("solvent"),
        ionic_strength=conditions.get("ionic_strength"),
        cofactors=list(conditions.get("catalysts") or []),
        buffer=conditions.get("buffer"),
    )


@dataclass
class ORDCollector(Collector):
    root: Path

    def list_tasks(self) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
        for path in self.root.rglob("*.json*"):
            yield from _iter_records(path)

    def fetch(self, task: Tuple[Path, int, Dict[str, Any]]) -> Dict[str, Any]:
        return task[2]

    def parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return raw

    def normalize(self, parsed: Dict[str, Any]) -> NormalizedReaction:
        reactants = _extract_smiles(parsed.get("reactants") or [])
        products = _extract_smiles(parsed.get("products") or [])
        conditions = _parse_conditions(parsed)
        yield_value = parsed.get("yield") or parsed.get("yield_percent")
        return NormalizedReaction(
            reaction_id=str(parsed.get("reaction_id") or parsed.get("id")),
            substrates=[_mock_molecule(smiles) for smiles in reactants],
            products=[_mock_molecule(smiles) for smiles in products],
            conditions=conditions,
            yield_percent=yield_value,
            notes=parsed.get("notes"),
        )


def _mock_molecule(smiles: str) -> Any:
    return type("Molecule", (), {"smiles": smiles, "name": None, "role": None})()
