from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_id TEXT,
                url TEXT,
                retrieved_at TEXT NOT NULL,
                sha256 TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_reaction_id TEXT,
                reactants_smiles TEXT,
                products_smiles TEXT,
                conditions_json TEXT,
                yield_value REAL,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smiles TEXT UNIQUE NOT NULL,
                inchikey TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                task_json TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bioactivities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target_chembl_id TEXT,
                assay_type TEXT,
                standard_value REAL,
                standard_units TEXT,
                relation TEXT,
                pchembl_value REAL,
                molecule_smiles TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_source ON queue(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_items(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_react_source ON reactions(source)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bioactivity_smiles ON bioactivities(molecule_smiles)"
        )
        conn.commit()


def enqueue_tasks(path: str, source: str, tasks: Iterable[Dict[str, Any]]) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        for task in tasks:
            payload = json.dumps(task)
            conn.execute(
                """
                INSERT INTO queue (source, task_json, status, last_error)
                VALUES (?, ?, ?, ?)
                """,
                (source, payload, "queued", None),
            )
        conn.commit()


def claim_task(path: str, source: str) -> Optional[Dict[str, Any]]:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, task_json FROM queue
            WHERE source = ? AND status = 'queued'
            ORDER BY id ASC
            LIMIT 1
            """,
            (source,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE queue SET status = ? WHERE id = ?",
            ("in_progress", row["id"]),
        )
        conn.commit()
        payload = json.loads(row["task_json"])
        payload["_queue_id"] = row["id"]
        return payload


def write_raw_item(
    path: str,
    source: str,
    source_id: Optional[str],
    url: Optional[str],
    sha256: Optional[str],
    raw_json: Dict[str, Any],
    license_hint: Optional[str] = None,
) -> None:
    init_db(path)
    if not url or not sha256:
        raise ValueError("raw_items require url and sha256 for provenance.")
    payload = dict(raw_json)
    provenance = payload.get("_provenance") or {}
    provenance.update(
        {
            "url": url,
            "retrieved_at": _now_iso(),
            "sha256": sha256,
            "license_hint": license_hint,
        }
    )
    payload["_provenance"] = provenance
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO raw_items (source, source_id, url, retrieved_at, sha256, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                source_id,
                url,
                _now_iso(),
                sha256,
                json.dumps(payload),
            ),
        )
        conn.commit()


def write_reaction(
    path: str,
    source: str,
    source_reaction_id: Optional[str],
    reactants_smiles: Optional[str],
    products_smiles: Optional[str],
    conditions_json: Optional[Dict[str, Any]],
    yield_value: Optional[float],
    metadata_json: Optional[Dict[str, Any]],
) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO reactions (
                source,
                source_reaction_id,
                reactants_smiles,
                products_smiles,
                conditions_json,
                yield_value,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                source_reaction_id,
                reactants_smiles,
                products_smiles,
                json.dumps(conditions_json) if conditions_json is not None else None,
                yield_value,
                json.dumps(metadata_json) if metadata_json is not None else None,
            ),
        )
        conn.commit()


def upsert_molecule(
    path: str,
    smiles: str,
    inchikey: Optional[str],
    metadata_json: Optional[Dict[str, Any]],
) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        existing = conn.execute(
            "SELECT id, metadata_json FROM molecules WHERE smiles = ?",
            (smiles,),
        ).fetchone()
        if existing:
            metadata = metadata_json
            if metadata_json is None:
                metadata = json.loads(existing[1]) if existing[1] else None
            conn.execute(
                """
                UPDATE molecules
                SET inchikey = ?, metadata_json = ?
                WHERE smiles = ?
                """,
                (
                    inchikey,
                    json.dumps(metadata) if metadata is not None else None,
                    smiles,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO molecules (smiles, inchikey, metadata_json)
                VALUES (?, ?, ?)
                """,
                (
                    smiles,
                    inchikey,
                    json.dumps(metadata_json) if metadata_json is not None else None,
                ),
            )
        conn.commit()


def write_bioactivity(
    path: str,
    source: str,
    target_chembl_id: Optional[str],
    assay_type: Optional[str],
    standard_value: Optional[float],
    standard_units: Optional[str],
    relation: Optional[str],
    pchembl_value: Optional[float],
    molecule_smiles: Optional[str],
    metadata_json: Optional[Dict[str, Any]],
) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO bioactivities (
                source,
                target_chembl_id,
                assay_type,
                standard_value,
                standard_units,
                relation,
                pchembl_value,
                molecule_smiles,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                target_chembl_id,
                assay_type,
                standard_value,
                standard_units,
                relation,
                pchembl_value,
                molecule_smiles,
                json.dumps(metadata_json) if metadata_json is not None else None,
            ),
        )
        conn.commit()


def mark_task_done(path: str, queue_id: int) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE queue SET status = ?, last_error = NULL WHERE id = ?",
            ("done", queue_id),
        )
        conn.commit()


def mark_task_failed(path: str, queue_id: int, error: str) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE queue SET status = ?, last_error = ? WHERE id = ?",
            ("failed", error, queue_id),
        )
        conn.commit()
