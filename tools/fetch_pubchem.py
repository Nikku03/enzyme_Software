from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List

from enzyme_software.data_acquisition.dataset_store import (
    init_db,
    upsert_molecule,
    write_raw_item,
)
from enzyme_software.data_acquisition.http import HttpClient
from enzyme_software.data_acquisition.sources.pubchem import PubChemCollector, _task_url
from enzyme_software.evidence_store import add_literature_evidence
from enzyme_software.literature_evidence import EvidenceRecord

SEED_PATH = Path("data/seeds_pubchem.txt")


def _load_seeds(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Seed file not found: {path}")
    seeds: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        seeds.append(stripped)
    return seeds


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch PubChem molecules into dataset store.")
    parser.add_argument("--db", required=True, help="Path to dataset SQLite DB.")
    parser.add_argument(
        "--evidence-db",
        default=os.environ.get("EVIDENCE_DB_PATH"),
        help="Optional EvidenceStore SQLite DB to append literature evidence.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of items per run.")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds.")
    args = parser.parse_args(argv)

    try:
        seeds = _load_seeds(SEED_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    init_db(args.db)
    client = HttpClient(tokens_per_second=args.rate, timeout_s=args.timeout)
    collector = PubChemCollector(seeds=seeds, client=client)

    count = 0
    for task in collector.list_tasks():
        if count >= args.limit:
            break
        try:
            raw = collector.fetch(task)
            sha256 = hashlib.sha256(raw).hexdigest()
            parsed = collector.parse(raw)
            normalized = collector.normalize(parsed)
            url = _task_url(task)
            write_raw_item(
                args.db,
                source="pubchem",
                source_id=str(parsed.get("CID") or task.get("value")),
                url=url,
                sha256=sha256,
                raw_json=parsed,
                license_hint="PubChem",
            )
            if args.evidence_db and normalized.smiles:
                evidence = EvidenceRecord(
                    source="pubchem",
                    source_id=str(parsed.get("CID") or task.get("value")),
                    substrate_smiles=normalized.smiles,
                    reaction_family="unknown",
                    conditions={},
                    catalyst_family=None,
                    outcome_label=False,
                    notes="molecule_only_pubchem_no_reaction_outcome",
                    confidence=0.05,
                    provenance={
                        "url": url,
                        "sha256": sha256,
                        "parser": "PubChemCollector",
                    },
                )
                add_literature_evidence(args.evidence_db, evidence)
            if normalized.smiles:
                inchikey = parsed.get("InChIKey")
                metadata = {"name": normalized.name, "source": "pubchem"}
                upsert_molecule(args.db, normalized.smiles, inchikey, metadata)
            count += 1
        except Exception as exc:
            print(f"Warning: {task.get('value')} failed: {exc}", file=sys.stderr)
            continue

    print(f"Fetched {count} PubChem entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
