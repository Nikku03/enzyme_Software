from __future__ import annotations

import contextlib
import contextvars
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


_PROVENANCE_CONTEXT: contextvars.ContextVar[Dict[str, object]] = contextvars.ContextVar(
    "mol_provenance_context",
    default={},
)


def _log_path() -> Path:
    import os

    override = os.environ.get("LNN_MOL_PROVENANCE_LOG", "").strip()
    path = Path(override) if override else Path("artifacts/mol_runtime_provenance.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def mol_provenance_context(**updates):
    current = dict(_PROVENANCE_CONTEXT.get() or {})
    current.update({k: v for k, v in updates.items() if v is not None})
    token = _PROVENANCE_CONTEXT.set(current)
    try:
        yield current
    finally:
        _PROVENANCE_CONTEXT.reset(token)


def current_mol_provenance() -> Dict[str, object]:
    return dict(_PROVENANCE_CONTEXT.get() or {})


def log_mol_provenance_event(
    *,
    stage: str,
    status: str,
    parsed_smiles: str,
    canonical_smiles: Optional[str] = None,
    error: Optional[str] = None,
    rdkit_message: Optional[str] = None,
    repaired: Optional[bool] = None,
    aggressive_repair: Optional[bool] = None,
    module_triggered: Optional[str] = None,
    source_category: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    ctx = current_mol_provenance()
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module_triggered": module_triggered or ctx.get("module_triggered") or ctx.get("caller_module") or "unknown",
        "source_category": source_category or ctx.get("source_category") or "unknown",
        "parsed_smiles": str(parsed_smiles or ""),
        "original_smiles": str(ctx.get("original_smiles") or parsed_smiles or ""),
        "drug_id": ctx.get("drug_id"),
        "drug_name": ctx.get("drug_name"),
        "canonical_smiles": canonical_smiles,
        "stage": stage,
        "status": status,
        "repaired": repaired,
        "aggressive_repair": aggressive_repair,
        "error": error,
        "rdkit_message": rdkit_message,
    }
    if extra:
        event["extra"] = extra
    with _log_path().open("a") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
