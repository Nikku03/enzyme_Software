from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit import rdBase
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    rdBase = None

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import MolPrepResult, prepare_mol


def safe_prepare_mol(
    smiles: str,
    *,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
) -> Tuple[MolPrepResult, List[str]]:
    block_logs = getattr(rdBase, "BlockLogs", None)
    log_context = block_logs() if callable(block_logs) else nullcontext()
    with log_context:
        prep = prepare_mol(
            smiles,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    warnings: List[str] = []
    if prep.status == "repaired_full_sanitize":
        warnings.append("Used full sanitization repair")
    elif prep.status == "repaired_partial_sanitize":
        warnings.append("Used partial sanitization repair")
    elif prep.mol is None:
        detail = prep.error or prep.status or "unknown_parse_failure"
        warnings.append(f"SMILES parse failed: {detail}")
        warnings.append("QUARANTINED: All parsing strategies failed")
    return prep, warnings


def safe_mol_from_smiles(
    smiles: str,
    *,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
) -> Tuple[Optional["Chem.Mol"], List[str]]:
    prep, warnings = safe_prepare_mol(
        smiles,
        allow_partial_sanitize=allow_partial_sanitize,
        allow_aggressive_repair=allow_aggressive_repair,
    )
    return prep.mol, warnings
