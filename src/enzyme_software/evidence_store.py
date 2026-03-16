from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from enzyme_software.literature_evidence import EvidenceRecord as LiteratureEvidenceRecord
from enzyme_software.unity_schema import (
    BondContext,
    ConditionProfile,
    Module0Out,
    Module1Out,
    Module2Out,
    Module3Out,
    PhysicsAudit,
    UnityRecord,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                arm_id TEXT NOT NULL,
                any_activity INTEGER NOT NULL,
                target_match INTEGER,
                conversion REAL,
                failure_mode TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datapoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                module_id INTEGER NOT NULL,
                item_type TEXT NOT NULL,
                scaffold_id TEXT,
                variant_id TEXT,
                data_json TEXT NOT NULL,
                reason_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS literature_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                substrate_smiles TEXT NOT NULL,
                reaction_family TEXT NOT NULL,
                catalyst_family TEXT,
                conditions_json TEXT,
                outcome_label INTEGER NOT NULL,
                notes TEXT,
                confidence REAL,
                provenance_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_run_id ON outcomes(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_run_id ON datapoints(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_module_id ON datapoints(module_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lit_source ON literature_evidence(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lit_reaction_family ON literature_evidence(reaction_family)"
        )


def save_run(path: str, record: UnityRecord) -> None:
    init_db(path)
    payload = json.dumps(asdict(record))
    created_at = record.created_at or _now_iso()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, record_json, created_at)
            VALUES (?, ?, ?)
            """,
            (record.run_id, payload, created_at),
        )
        conn.commit()


def add_outcome(
    path: str,
    run_id: str,
    arm_id: str,
    any_activity: bool,
    target_match: Optional[bool],
    conversion: Optional[float],
    failure_mode: Optional[str],
    notes: Optional[str],
) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        exists = conn.execute(
            "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()
        if not exists:
            raise ValueError(f"run_id not found: {run_id}")
        conn.execute(
            """
            INSERT INTO outcomes (
                run_id,
                arm_id,
                any_activity,
                target_match,
                conversion,
                failure_mode,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                arm_id,
                1 if any_activity else 0,
                None if target_match is None else (1 if target_match else 0),
                conversion,
                failure_mode,
                notes,
                _now_iso(),
            ),
        )
        conn.commit()


def add_datapoints(path: str, run_id: str, datapoints: List[Dict[str, Any]]) -> int:
    if not datapoints:
        return 0
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        exists = conn.execute(
            "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()
        if not exists:
            raise ValueError(f"run_id not found: {run_id}")
        rows = []
        created_at = _now_iso()
        for point in datapoints:
            module_id = int(point.get("module_id", -1))
            item_type = str(point.get("item_type") or "unknown")
            scaffold_id = point.get("scaffold_id")
            variant_id = point.get("variant_id")
            data_json = json.dumps(point.get("data") or {}, default=str)
            reason_json = None
            if "reasons" in point:
                reason_json = json.dumps(point.get("reasons"), default=str)
            rows.append(
                (
                    run_id,
                    module_id,
                    item_type,
                    scaffold_id,
                    variant_id,
                    data_json,
                    reason_json,
                    created_at,
                )
            )
        conn.executemany(
            """
            INSERT INTO datapoints (
                run_id,
                module_id,
                item_type,
                scaffold_id,
                variant_id,
                data_json,
                reason_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_runs(path: str, limit: Optional[int] = None) -> List[UnityRecord]:
    init_db(path)
    rows: Iterable[sqlite3.Row]
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT record_json FROM runs ORDER BY created_at DESC"
        params: Tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        rows = conn.execute(query, params).fetchall()
    return [_unity_record_from_json(row["record_json"]) for row in rows]


def load_labeled_examples(path: str) -> List[Tuple[UnityRecord, Dict[str, Any]]]:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT r.record_json, o.arm_id, o.any_activity, o.target_match, o.conversion,
                   o.failure_mode, o.notes, o.created_at
            FROM runs r
            JOIN outcomes o ON o.run_id = r.run_id
            ORDER BY o.created_at DESC
            """
        ).fetchall()
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        record = _unity_record_from_json(row["record_json"])
        outcome = {
            "arm_id": row["arm_id"],
            "any_activity": bool(row["any_activity"]),
            "target_match": None
            if row["target_match"] is None
            else bool(row["target_match"]),
            "conversion": row["conversion"],
            "failure_mode": row["failure_mode"],
            "notes": row["notes"],
            "created_at": row["created_at"],
        }
        bundle = grouped.setdefault(record.run_id, {"record": record, "outcomes": []})
        bundle["outcomes"].append(outcome)
    return [(bundle["record"], {"outcomes": bundle["outcomes"]}) for bundle in grouped.values()]


def load_datapoints(
    path: str,
    run_id: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT module_id, item_type, scaffold_id, variant_id, data_json, reason_json, created_at
            FROM datapoints
            WHERE run_id = ?
            ORDER BY id ASC
        """
        params: Tuple[Any, ...] = (run_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (run_id, int(limit))
        rows = conn.execute(query, params).fetchall()
    datapoints: List[Dict[str, Any]] = []
    for row in rows:
        try:
            data = json.loads(row["data_json"]) if row["data_json"] else {}
        except json.JSONDecodeError:
            data = {"raw": row["data_json"]}
        reasons = None
        if row["reason_json"]:
            try:
                reasons = json.loads(row["reason_json"])
            except json.JSONDecodeError:
                reasons = row["reason_json"]
        datapoints.append(
            {
                "module_id": row["module_id"],
                "item_type": row["item_type"],
                "scaffold_id": row["scaffold_id"],
                "variant_id": row["variant_id"],
                "data": data,
                "reasons": reasons,
                "created_at": row["created_at"],
            }
        )
    return datapoints


def add_literature_evidence(path: str, record: LiteratureEvidenceRecord) -> None:
    init_db(path)
    conditions_json = json.dumps(record.conditions or {})
    provenance_json = json.dumps(record.provenance or {})
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO literature_evidence (
                source,
                source_id,
                substrate_smiles,
                reaction_family,
                catalyst_family,
                conditions_json,
                outcome_label,
                notes,
                confidence,
                provenance_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.source,
                record.source_id,
                record.substrate_smiles,
                record.reaction_family,
                record.catalyst_family,
                conditions_json,
                1 if record.outcome_label else 0,
                record.notes,
                float(record.confidence),
                provenance_json,
                _now_iso(),
            ),
        )
        conn.commit()


def load_literature_evidence(path: str) -> List[LiteratureEvidenceRecord]:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT source, source_id, substrate_smiles, reaction_family, catalyst_family,
                   conditions_json, outcome_label, notes, confidence, provenance_json
            FROM literature_evidence
            ORDER BY id DESC
            """
        ).fetchall()
    records: List[LiteratureEvidenceRecord] = []
    for row in rows:
        conditions = json.loads(row["conditions_json"]) if row["conditions_json"] else {}
        provenance = json.loads(row["provenance_json"]) if row["provenance_json"] else {}
        records.append(
            LiteratureEvidenceRecord(
                source=row["source"],
                source_id=row["source_id"],
                substrate_smiles=row["substrate_smiles"],
                reaction_family=row["reaction_family"],
                catalyst_family=row["catalyst_family"],
                conditions=conditions,
                outcome_label=bool(row["outcome_label"]),
                notes=row["notes"],
                confidence=float(row["confidence"]) if row["confidence"] is not None else 0.5,
                provenance=provenance,
            )
        )
    return records


def load_literature_sources(path: str) -> Dict[str, int]:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT source, COUNT(*) AS count
            FROM literature_evidence
            GROUP BY source
            """
        ).fetchall()
    return {row["source"]: int(row["count"]) for row in rows}


def _unity_record_from_json(raw: str) -> UnityRecord:
    payload = json.loads(raw)
    return _unity_record_from_dict(payload)


def _unity_record_from_dict(payload: Dict[str, Any]) -> UnityRecord:
    condition = _condition_from_dict(payload.get("condition_profile") or {})
    bond = _bond_from_dict(payload.get("bond_context") or {})
    physics = payload.get("physics_audit")
    physics_audit = _physics_from_dict(physics) if isinstance(physics, dict) else None
    module0 = _module0_from_dict(payload.get("module0") or {}) if payload.get("module0") else None
    module1 = _module1_from_dict(payload.get("module1") or {}) if payload.get("module1") else None
    module2 = _module2_from_dict(payload.get("module2") or {}) if payload.get("module2") else None
    module3 = _module3_from_dict(payload.get("module3") or {}) if payload.get("module3") else None
    return UnityRecord(
        run_id=payload.get("run_id") or "",
        smiles=payload.get("smiles") or "",
        target_bond=payload.get("target_bond") or "",
        requested_output=payload.get("requested_output"),
        condition_profile=condition,
        bond_context=bond,
        physics_audit=physics_audit,
        module0=module0,
        module1=module1,
        module2=module2,
        module3=module3,
        created_at=payload.get("created_at") or _now_iso(),
        updated_at=payload.get("updated_at") or _now_iso(),
    )


def _condition_from_dict(payload: Dict[str, Any]) -> ConditionProfile:
    return ConditionProfile(
        pH=payload.get("pH"),
        temperature_K=payload.get("temperature_K"),
        temperature_C=payload.get("temperature_C"),
        ionic_strength=payload.get("ionic_strength"),
        solvent=payload.get("solvent"),
        cofactors=list(payload.get("cofactors") or []),
        salts_buffer=payload.get("salts_buffer"),
        constraints=payload.get("constraints"),
    )


def _bond_from_dict(payload: Dict[str, Any]) -> BondContext:
    return BondContext(
        bond_role=payload.get("bond_role"),
        bond_role_confidence=payload.get("bond_role_confidence"),
        bond_class=payload.get("bond_class"),
        polarity=payload.get("polarity"),
        atom_count=payload.get("atom_count"),
        hetero_atoms=payload.get("hetero_atoms"),
        ring_count=payload.get("ring_count"),
    )


def _physics_from_dict(payload: Dict[str, Any]) -> PhysicsAudit:
    return PhysicsAudit(
        deltaG_dagger_kJ_per_mol=payload.get("deltaG_dagger_kJ_per_mol"),
        eyring_k_s_inv=payload.get("eyring_k_s_inv"),
        k_eff_s_inv=payload.get("k_eff_s_inv"),
        temperature_K=payload.get("temperature_K"),
        horizon_s=payload.get("horizon_s"),
        notes=list(payload.get("notes") or []),
    )


def _module0_from_dict(payload: Dict[str, Any]) -> Module0Out:
    return Module0Out(
        decision=payload.get("decision"),
        route_family=payload.get("route_family"),
        route_confidence=payload.get("route_confidence"),
        data_support=payload.get("data_support"),
    )


def _module1_from_dict(payload: Dict[str, Any]) -> Module1Out:
    return Module1Out(
        status=payload.get("status"),
        access_score=payload.get("access_score"),
        reach_score=payload.get("reach_score"),
        retention_score=payload.get("retention_score"),
        top_scaffold=payload.get("top_scaffold"),
    )


def _module2_from_dict(payload: Dict[str, Any]) -> Module2Out:
    return Module2Out(
        status=payload.get("status"),
        selected_scaffold=payload.get("selected_scaffold"),
        best_variant=payload.get("best_variant"),
        deltaG_dagger_kJ_per_mol=payload.get("deltaG_dagger_kJ_per_mol"),
        k_pred_s_inv=payload.get("k_pred_s_inv"),
        route_family=payload.get("route_family"),
    )


def _module3_from_dict(payload: Dict[str, Any]) -> Module3Out:
    return Module3Out(
        status=payload.get("status"),
        plan_score=payload.get("plan_score"),
        qc_status=payload.get("qc_status"),
        batch_id=payload.get("batch_id"),
    )
