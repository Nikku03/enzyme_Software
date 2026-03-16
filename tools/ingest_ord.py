from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from enzyme_software.data_acquisition.dataset_store import init_db, write_reaction
from enzyme_software.data_acquisition.sources.ord import ORDCollector
from enzyme_software.evidence_store import add_literature_evidence
from enzyme_software.literature_evidence import EvidenceRecord


def _infer_reaction_family(reactants: List[str], products: List[str]) -> str:
    reactant_str = ".".join(reactants)
    product_str = ".".join(products)
    if "C(=O)O" in reactant_str and "C(=O)O" not in product_str:
        return "hydrolysis"
    if "C(=O)N" in reactant_str and "C(=O)N" not in product_str:
        return "hydrolysis"
    if "CBr" in reactant_str or "CCl" in reactant_str:
        return "dehalogenation"
    return "unknown"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest ORD reactions from local files.")
    parser.add_argument("--db", required=True, help="Path to dataset SQLite DB.")
    parser.add_argument(
        "--path",
        default=os.environ.get("ORD_PATH", "data/ord"),
        help="Path to ORD JSON/JSONL files (or set ORD_PATH).",
    )
    parser.add_argument(
        "--evidence-db",
        default=os.environ.get("EVIDENCE_DB_PATH"),
        help="Optional EvidenceStore SQLite DB to append literature evidence.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of records per run.")
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    init_db(args.db)
    collector = ORDCollector(root=root)

    count = 0
    for task in collector.list_tasks():
        if count >= args.limit:
            break
        parsed = collector.parse(collector.fetch(task))
        normalized = collector.normalize(parsed)
        reactants = [mol.smiles for mol in normalized.substrates if mol.smiles]
        products = [mol.smiles for mol in normalized.products if mol.smiles]
        conditions = {
            "pH": normalized.conditions.pH,
            "temperature_C": normalized.conditions.temperature_C,
            "solvent": normalized.conditions.solvent,
            "ionic_strength": normalized.conditions.ionic_strength,
            "cofactors": normalized.conditions.cofactors,
            "buffer": normalized.conditions.buffer,
        }
        write_reaction(
            args.db,
            source="ord",
            source_reaction_id=normalized.reaction_id,
            reactants_smiles=".".join(reactants) if reactants else None,
            products_smiles=".".join(products) if products else None,
            conditions_json=conditions,
            yield_value=normalized.yield_percent,
            metadata_json={"notes": normalized.notes},
        )
        if args.evidence_db and reactants:
            reaction_family = _infer_reaction_family(reactants, products)
            yield_value = normalized.yield_percent
            confidence = 0.8 if yield_value is not None else 0.1
            outcome_label = bool(yield_value and float(yield_value) >= 0.2)
            evidence = EvidenceRecord(
                source="ord",
                source_id=normalized.reaction_id or "unknown",
                substrate_smiles=reactants[0],
                reaction_family=reaction_family,
                conditions=conditions,
                catalyst_family=None,
                outcome_label=outcome_label,
                notes=f"ord_ingest reaction_family={reaction_family} yield={yield_value}",
                confidence=confidence,
                provenance={
                    "path": str(task[0]),
                    "record_index": task[1],
                    "parser": "ORDCollector",
                },
            )
            add_literature_evidence(args.evidence_db, evidence)
        count += 1

    print(f"Ingested {count} ORD records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
