from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    rdmolops = None

from enzyme_software.liquid_nn_v2.utils.mol_provenance import log_mol_provenance_event


@dataclass
class MolPrepResult:
    mol: Optional["Chem.Mol"]
    original_smiles: str
    canonical_smiles: Optional[str]
    status: str
    repaired: bool
    aggressive_repair: bool
    error: Optional[str]


_SANITIZE_PARTIAL = None
if Chem is not None:
    _SANITIZE_PARTIAL = (
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    )


def _canonical_smiles(mol) -> Optional[str]:
    if Chem is None or mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _capture_rdkit(callable_):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            value = callable_()
            return value, None, stderr.getvalue().strip() or None
        except Exception as exc:  # pragma: no cover - defensive
            return None, exc, stderr.getvalue().strip() or None


def _log_stage_error(
    *,
    stage: str,
    parsed_smiles: str,
    canonical_smiles: Optional[str],
    exc,
    stderr: Optional[str],
    status: str = "rdkit_error",
    repaired: Optional[bool] = None,
    aggressive_repair: Optional[bool] = None,
):
    message = stderr or (str(exc) if exc is not None else None) or "RDKit operation failed"
    log_mol_provenance_event(
        stage=stage,
        status=status,
        parsed_smiles=parsed_smiles,
        canonical_smiles=canonical_smiles,
        error=str(exc) if exc is not None else None,
        rdkit_message=message,
        repaired=repaired,
        aggressive_repair=aggressive_repair,
    )


def prepare_mol(
    smiles: str,
    *,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
) -> MolPrepResult:
    """
    Centralized RDKit parsing and sanitization with conservative repair stages.

    Default path:
    - normal parse
    - parse sanitize=False + full sanitize
    - parse sanitize=False + partial sanitize skipping kekulize/aromaticity
    - optional non-destructive KekulizeIfPossible on the partial result
    """
    original = str(smiles or "").strip()
    if Chem is None:
        return MolPrepResult(
            mol=None,
            original_smiles=original,
            canonical_smiles=None,
            status="failed",
            repaired=False,
            aggressive_repair=False,
            error="RDKit unavailable",
        )
    if not original:
        return MolPrepResult(
            mol=None,
            original_smiles=original,
            canonical_smiles=None,
            status="failed",
            repaired=False,
            aggressive_repair=False,
            error="Empty SMILES",
        )

    mol, exc, stderr = _capture_rdkit(lambda: Chem.MolFromSmiles(original))
    if mol is not None:
        return MolPrepResult(
            mol=mol,
            original_smiles=original,
            canonical_smiles=_canonical_smiles(mol),
            status="ok",
            repaired=False,
            aggressive_repair=False,
            error=None,
        )
    _log_stage_error(
        stage="normal_parse",
        parsed_smiles=original,
        canonical_smiles=None,
        exc=exc,
        stderr=stderr or "Chem.MolFromSmiles returned None",
        repaired=False,
        aggressive_repair=False,
    )

    def _base_unsanitized():
        parsed = Chem.MolFromSmiles(original, sanitize=False)
        if parsed is None:
            raise ValueError("MolFromSmiles(..., sanitize=False) returned None")
        return parsed

    base_mol, base_exc, base_stderr = _capture_rdkit(_base_unsanitized)
    if base_mol is None:
        error = str(base_exc) if base_exc is not None else (stderr or base_stderr or "Failed to parse SMILES")
        _log_stage_error(
            stage="parse_sanitize_false",
            parsed_smiles=original,
            canonical_smiles=None,
            exc=base_exc,
            stderr=base_stderr or error,
            repaired=False,
            aggressive_repair=bool(allow_aggressive_repair),
        )
        return MolPrepResult(
            mol=None,
            original_smiles=original,
            canonical_smiles=None,
            status="failed",
            repaired=False,
            aggressive_repair=bool(allow_aggressive_repair),
            error=error,
        )

    def _full_sanitize():
        mol_full = Chem.Mol(base_mol)
        mol_full.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol_full)
        return mol_full

    mol_full, full_exc, full_stderr = _capture_rdkit(_full_sanitize)
    if mol_full is not None:
        canonical = _canonical_smiles(mol_full)
        log_mol_provenance_event(
            stage="final_result",
            status="repaired_full_sanitize",
            parsed_smiles=original,
            canonical_smiles=canonical,
            error=None,
            rdkit_message=None,
            repaired=True,
            aggressive_repair=False,
        )
        return MolPrepResult(
            mol=mol_full,
            original_smiles=original,
            canonical_smiles=canonical,
            status="repaired_full_sanitize",
            repaired=True,
            aggressive_repair=False,
            error=None,
        )
    _log_stage_error(
        stage="full_sanitize",
        parsed_smiles=original,
        canonical_smiles=None,
        exc=full_exc,
        stderr=full_stderr or "Full sanitization failed",
        repaired=True,
        aggressive_repair=False,
    )

    if allow_partial_sanitize:
        def _partial_sanitize():
            mol_partial = Chem.Mol(base_mol)
            mol_partial.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol_partial, sanitizeOps=_SANITIZE_PARTIAL)
            if rdmolops is not None:
                try:
                    rdmolops.KekulizeIfPossible(mol_partial, clearAromaticFlags=False)
                except Exception:
                    pass
            return mol_partial

        mol_partial, partial_exc, partial_stderr = _capture_rdkit(_partial_sanitize)
        if mol_partial is not None:
            canonical = _canonical_smiles(mol_partial)
            log_mol_provenance_event(
                stage="final_result",
                status="repaired_partial_sanitize",
                parsed_smiles=original,
                canonical_smiles=canonical,
                error=None,
                rdkit_message=None,
                repaired=True,
                aggressive_repair=False,
            )
            return MolPrepResult(
                mol=mol_partial,
                original_smiles=original,
                canonical_smiles=canonical,
                status="repaired_partial_sanitize",
                repaired=True,
                aggressive_repair=False,
                error=None,
            )
        _log_stage_error(
            stage="partial_sanitize",
            parsed_smiles=original,
            canonical_smiles=None,
            exc=partial_exc,
            stderr=partial_stderr or "Partial sanitization failed",
            repaired=True,
            aggressive_repair=False,
        )
        error = str(partial_exc) if partial_exc is not None else (partial_stderr or full_stderr or stderr)
    else:
        error = str(full_exc) if full_exc is not None else (full_stderr or stderr)

    log_mol_provenance_event(
        stage="final_result",
        status="failed",
        parsed_smiles=original,
        canonical_smiles=None,
        error=error or "Sanitization failed",
        rdkit_message=None,
        repaired=False,
        aggressive_repair=False,
    )
    return MolPrepResult(
        mol=None,
        original_smiles=original,
        canonical_smiles=None,
        status="failed",
        repaired=False,
        aggressive_repair=False,
        error=error or "Sanitization failed",
    )
