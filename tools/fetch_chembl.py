from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List

from enzyme_software.data_acquisition.dataset_store import (
    init_db,
    write_bioactivity,
    write_raw_item,
)
from enzyme_software.data_acquisition.http import HttpClient
from enzyme_software.data_acquisition.sources.chembl import ChEMBLCollector, _task_url
from enzyme_software.evidence_store import add_literature_evidence
from enzyme_software.literature_evidence import EvidenceRecord

SEED_PATH = Path("data/seeds_chembl.txt")


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
    parser = argparse.ArgumentParser(description="Fetch ChEMBL activities into dataset store.")
    parser.add_argument("--db", required=True, help="Path to dataset SQLite DB.")
    parser.add_argument(
        "--evidence-db",
        default=os.environ.get("EVIDENCE_DB_PATH"),
        help="Optional EvidenceStore SQLite DB to append literature evidence.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of tasks per run.")
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
    collector = ChEMBLCollector(seeds=seeds, client=client)

    count_tasks = 0
    count_activities = 0
    for task in collector.list_tasks():
        if count_tasks >= args.limit:
            break
        try:
            raw = collector.fetch(task)
            sha256 = hashlib.sha256(raw).hexdigest()
            parsed = collector.parse(raw)
            normalized = collector.normalize(parsed)
            url = _task_url(task)
            write_raw_item(
                args.db,
                source="chembl",
                source_id=task.get("value"),
                url=url,
                sha256=sha256,
                raw_json=parsed,
                license_hint="ChEMBL",
            )
            for entry in normalized:
                write_bioactivity(
                    args.db,
                    source="chembl",
                    target_chembl_id=entry.get("target_chembl_id"),
                    assay_type=entry.get("assay_type"),
                    standard_value=entry.get("standard_value"),
                    standard_units=entry.get("standard_units"),
                    relation=entry.get("relation"),
                    pchembl_value=entry.get("pchembl_value"),
                    molecule_smiles=entry.get("molecule_smiles"),
                    metadata_json=entry.get("metadata"),
                )
                if args.evidence_db and entry.get("molecule_smiles"):
                    evidence = EvidenceRecord(
                        source="chembl",
                        source_id=str(task.get("value") or ""),
                        substrate_smiles=entry.get("molecule_smiles"),
                        reaction_family="bioactivity",
                        conditions={},
                        catalyst_family=None,
                        outcome_label=bool((entry.get("pchembl_value") or 0) >= 5.0),
                        notes="chembl_activity_proxy_no_reaction_context",
                        confidence=0.1,
                        provenance={
                            "url": url,
                            "sha256": sha256,
                            "parser": "ChEMBLCollector",
                        },
                    )
                    add_literature_evidence(args.evidence_db, evidence)
                count_activities += 1
            count_tasks += 1
        except Exception as exc:
            print(f"Warning: {task.get('value')} failed: {exc}", file=sys.stderr)
            continue

    print(f"Fetched {count_activities} activities from {count_tasks} tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
