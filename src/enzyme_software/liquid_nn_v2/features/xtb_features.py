from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None

from enzyme_software.liquid_nn_v2.features.graph_builder import MoleculeGraph
from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


XTB_FEATURE_NAMES = (
    "charge",
    "abs_charge",
    "charge_centered",
    "charge_rank_norm",
    "wbo_sum",
    "wbo_max",
)
XTB_FEATURE_DIM = len(XTB_FEATURE_NAMES)

FULL_XTB_FEATURE_NAMES = (
    "bde_corrected_norm",
    "bde_inverse",
    "bde_vertical_norm",
    "bde_adiabatic_norm",
    "relaxation_energy_norm",
    "bde_deviation_norm",
    "xtb_confidence",
    "lookup_bde_norm",
)
FULL_XTB_FEATURE_DIM = len(FULL_XTB_FEATURE_NAMES)
FULL_XTB_CACHE_SCHEMA_VERSION = 2
XTB_STATUS_NAMES = (
    "ok",
    "missing",
    "no_xtb_payload",
    "manual_engine_error",
    "rdkit_unavailable",
    "embedding_failed",
    "xtb_unavailable",
    "xtb_error",
    "xtb_failed",
    "other",
)


def xtb_status_vector(status: Optional[str]) -> np.ndarray:
    key = str(status or "").strip().lower()
    vector = np.zeros((len(XTB_STATUS_NAMES),), dtype=np.float32)
    lookup = {name: idx for idx, name in enumerate(XTB_STATUS_NAMES[:-1])}
    index = lookup.get(key, len(XTB_STATUS_NAMES) - 1)
    vector[index] = 1.0
    return vector


def payload_true_xtb_valid(payload: Optional[Dict[str, object]]) -> bool:
    if not payload:
        return False
    if "true_xtb_valid" in payload:
        return bool(payload.get("true_xtb_valid"))
    return bool(payload.get("xtb_valid"))


def payload_cached_xtb_valid(payload: Optional[Dict[str, object]]) -> bool:
    if not payload:
        return False
    return bool(payload.get("xtb_valid"))


def payload_training_xtb_valid(payload: Optional[Dict[str, object]]) -> bool:
    if not payload:
        return False
    if payload_true_xtb_valid(payload):
        return True
    if not payload_cached_xtb_valid(payload):
        return False
    atom_valid_mask = np.asarray(payload.get("atom_valid_mask") or [], dtype=np.float32)
    if atom_valid_mask.size == 0:
        return False
    return bool(np.any(atom_valid_mask > 0.0))


def payload_xtb_validity_summary(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not payload:
        return {
            "status": "missing",
            "source_kind": "missing",
            "strict_true_xtb_valid": False,
            "cached_xtb_valid": False,
            "training_usable_xtb_valid": False,
        }
    return {
        "status": str(payload.get("status") or "unknown"),
        "source_kind": str(payload.get("xtb_source_kind") or "unknown"),
        "strict_true_xtb_valid": bool(payload_true_xtb_valid(payload)),
        "cached_xtb_valid": bool(payload_cached_xtb_valid(payload)),
        "training_usable_xtb_valid": bool(payload_training_xtb_valid(payload)),
    }


def xtb_available(xtb_path: str = "xtb") -> bool:
    return shutil.which(xtb_path) is not None


def normalize_bde(bde: float, min_bde: float = 250.0, max_bde: float = 500.0) -> float:
    span = max(1.0e-6, float(max_bde) - float(min_bde))
    return float(np.clip((float(bde) - float(min_bde)) / span, 0.0, 1.0))


def _cache_key(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24]


def _cache_path(smiles: str, cache_dir: str | Path) -> Path:
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_cache_key(smiles)}.json"


def _full_cache_key(smiles: str, target_bond: Optional[str]) -> str:
    token = f"{smiles}||{str(target_bond or '').strip()}||schema={FULL_XTB_CACHE_SCHEMA_VERSION}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def _full_cache_path(smiles: str, cache_dir: str | Path, *, target_bond: Optional[str] = None) -> Path:
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_full_cache_key(smiles, target_bond)}_full.json"


def _legacy_full_cache_path(smiles: str, cache_dir: str | Path) -> Path:
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_cache_key(smiles)}_full.json"


def _invalid_xtb_payload(
    canonical_smiles: Optional[str],
    num_atoms: int,
    *,
    status: str,
    error: Optional[str],
) -> Dict[str, object]:
    return {
        "canonical_smiles": canonical_smiles,
        "xtb_valid": False,
        "atom_valid_mask": [[0.0] for _ in range(max(0, num_atoms))],
        "atom_features": [[0.0] * XTB_FEATURE_DIM for _ in range(max(0, num_atoms))],
        "feature_names": list(XTB_FEATURE_NAMES),
        "status": status,
        "error": error,
    }


def _invalid_full_xtb_payload(
    canonical_smiles: Optional[str],
    num_atoms: int,
    *,
    status: str,
    error: Optional[str],
) -> Dict[str, object]:
    return {
        "cache_schema_version": FULL_XTB_CACHE_SCHEMA_VERSION,
        "canonical_smiles": canonical_smiles,
        "xtb_valid": False,
        "true_xtb_valid": False,
        "atom_valid_mask": [[0.0] for _ in range(max(0, num_atoms))],
        "atom_features": [[0.0] * FULL_XTB_FEATURE_DIM for _ in range(max(0, num_atoms))],
        "feature_names": list(FULL_XTB_FEATURE_NAMES),
        "status": status,
        "xtb_source_kind": "missing",
        "true_xtb_used": False,
        "lookup_only": False,
        "manual_fallback_used": False,
        "resolved_target_bond": None,
        "error": error,
    }


def _full_xtb_source_summary(module_minus1_result: Dict[str, Any]) -> Dict[str, object]:
    resolved = module_minus1_result.get("resolved_target") or {}
    cpt_scores = module_minus1_result.get("cpt_scores") or {}
    global_bde = cpt_scores.get("bde") or module_minus1_result.get("bde") or {}
    candidates = (
        module_minus1_result.get("candidate_sites")
        or resolved.get("candidate_bonds")
        or resolved.get("candidate_sites")
        or []
    )
    source_names: set[str] = set()
    status_names: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bde_payload = candidate.get("bde") if isinstance(candidate.get("bde"), dict) else {}
        source = str(bde_payload.get("source") or candidate.get("source") or "").strip().lower()
        xtb_status = str(bde_payload.get("xtb_status") or candidate.get("xtb_status") or "").strip().lower()
        if source:
            source_names.add(source)
        if xtb_status:
            status_names.add(xtb_status)
    global_source = str(global_bde.get("source") or "").strip().lower()
    global_status = str(global_bde.get("xtb_status") or "").strip().lower()
    if global_source:
        source_names.add(global_source)
    if global_status:
        status_names.add(global_status)

    true_xtb_sources = {"xtb_validated", "xtb_only"}
    lookup_sources = {"lookup_safeguard", "lookup_fallback"}
    true_xtb_used = bool(source_names.intersection(true_xtb_sources) or status_names.intersection({"ok", "xtb_validated"}))
    lookup_only = bool(source_names) and source_names.issubset(lookup_sources)
    manual_fallback_used = bool(source_names.intersection(lookup_sources) or status_names.intersection({"lookup_fallback", "lookup_safeguard"}))
    if true_xtb_used and lookup_only:
        source_kind = "mixed"
    elif true_xtb_used:
        source_kind = "true_xtb"
    elif lookup_only:
        source_kind = "lookup_only"
    elif source_names:
        source_kind = "heuristic_only"
    else:
        source_kind = "unknown"
    return {
        "xtb_source_kind": source_kind,
        "true_xtb_used": bool(true_xtb_used),
        "lookup_only": bool(lookup_only),
        "manual_fallback_used": bool(manual_fallback_used),
        "source_names": sorted(source_names),
        "status_names": sorted(status_names),
    }


def _normalize_full_xtb_payload(
    payload: Dict[str, object],
    *,
    canonical_smiles: Optional[str],
    num_atoms: int,
    resolved_target_bond: Optional[str],
) -> Dict[str, object]:
    feature_names = list(payload.get("feature_names") or [])
    atom_features = payload.get("atom_features") or []
    atom_valid_mask = payload.get("atom_valid_mask") or []
    if feature_names and feature_names != list(FULL_XTB_FEATURE_NAMES):
        raise ValueError(f"Unexpected full-xTB feature_names: {feature_names}")
    if atom_features and len(atom_features[0]) != FULL_XTB_FEATURE_DIM:
        raise ValueError(
            f"Unexpected full-xTB feature width: {len(atom_features[0])} != {FULL_XTB_FEATURE_DIM}"
        )
    if atom_features and len(atom_features) != num_atoms:
        raise ValueError(f"full-xTB atom count mismatch: {len(atom_features)} != {num_atoms}")
    if atom_valid_mask and len(atom_valid_mask) != num_atoms:
        raise ValueError(f"full-xTB valid-mask length mismatch: {len(atom_valid_mask)} != {num_atoms}")
    normalized = dict(payload)
    normalized["cache_schema_version"] = int(payload.get("cache_schema_version") or FULL_XTB_CACHE_SCHEMA_VERSION)
    normalized["canonical_smiles"] = canonical_smiles
    normalized["resolved_target_bond"] = resolved_target_bond
    normalized.setdefault("feature_names", list(FULL_XTB_FEATURE_NAMES))
    normalized.setdefault("xtb_source_kind", "unknown" if payload_cached_xtb_valid(payload) else "missing")
    normalized.setdefault("true_xtb_used", False)
    normalized.setdefault("true_xtb_valid", bool(payload.get("true_xtb_used")) and payload_cached_xtb_valid(payload))
    normalized.setdefault("lookup_only", False)
    normalized.setdefault("manual_fallback_used", False)
    return normalized


def _embed_molecule(smiles: str):
    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit unavailable")
    prep = prepare_mol(smiles)
    if prep.mol is None:
        raise RuntimeError(prep.error or "prepare_mol failed")
    mol = Chem.AddHs(Chem.Mol(prep.mol))
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0DE) != 0:
        raise RuntimeError("RDKit embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return prep.canonical_smiles or smiles, mol


def _mol_to_xyz(mol) -> str:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), "xtb micropattern features"]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2} {pos.x: .8f} {pos.y: .8f} {pos.z: .8f}")
    return "\n".join(lines) + "\n"


def _parse_charges(path: Path, heavy_atom_count: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    values: List[float] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            values.append(float(parts[-1]))
        except ValueError:
            continue
    if len(values) < heavy_atom_count:
        return None
    return np.asarray(values[:heavy_atom_count], dtype=np.float32)


def _parse_wbo(path: Path, heavy_atom_count: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    wbo_sum = np.zeros(heavy_atom_count, dtype=np.float32)
    wbo_max = np.zeros(heavy_atom_count, dtype=np.float32)
    any_found = False
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            a = int(parts[0]) - 1
            b = int(parts[1]) - 1
            order = float(parts[2])
        except ValueError:
            continue
        if 0 <= a < heavy_atom_count and 0 <= b < heavy_atom_count:
            any_found = True
            wbo_sum[a] += order
            wbo_sum[b] += order
            wbo_max[a] = max(wbo_max[a], order)
            wbo_max[b] = max(wbo_max[b], order)
    if not any_found:
        return None
    return np.stack([wbo_sum, wbo_max], axis=-1)


def compute_xtb_atom_features(
    smiles: str,
    *,
    xtb_path: str = "xtb",
    solvent: str = "water",
    timeout_s: int = 300,
) -> Dict[str, object]:
    if Chem is None or AllChem is None:
        return _invalid_xtb_payload(None, 0, status="rdkit_unavailable", error="RDKit unavailable")
    if not xtb_available(xtb_path):
        prep = prepare_mol(smiles)
        num_atoms = prep.mol.GetNumAtoms() if prep.mol is not None else 0
        return _invalid_xtb_payload(
            prep.canonical_smiles if prep.mol is not None else None,
            num_atoms,
            status="xtb_unavailable",
            error="xTB not available",
        )

    prep = prepare_mol(smiles)
    num_atoms = prep.mol.GetNumAtoms() if prep.mol is not None else 0
    try:
        canonical_smiles, mol_h = _embed_molecule(smiles)
    except Exception as exc:
        return _invalid_xtb_payload(
            prep.canonical_smiles,
            num_atoms,
            status="embedding_failed",
            error=str(exc),
        )

    heavy_atom_count = Chem.RemoveHs(Chem.Mol(mol_h)).GetNumAtoms()
    xyz = _mol_to_xyz(mol_h)
    charge = int(sum(int(atom.GetFormalCharge()) for atom in mol_h.GetAtoms()))

    with tempfile.TemporaryDirectory(prefix="xtb_atom_features_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        xyz_path = tmp_path / "input.xyz"
        xyz_path.write_text(xyz, encoding="utf-8")
        cmd = [
            xtb_path,
            str(xyz_path),
            "--gfn",
            "2",
            "--chrg",
            str(charge),
            "--uhf",
            "0",
            "--wbo",
            "--pop",
        ]
        if solvent:
            cmd.extend(["--alpb", str(solvent)])
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            return _invalid_xtb_payload(canonical_smiles, heavy_atom_count, status="xtb_error", error=str(exc))

        charges = _parse_charges(tmp_path / "charges", heavy_atom_count)
        wbo = _parse_wbo(tmp_path / "wbo", heavy_atom_count)
        if proc.returncode != 0 or charges is None:
            return _invalid_xtb_payload(
                canonical_smiles,
                heavy_atom_count,
                status="xtb_failed",
                error=proc.stderr.strip() or "charges missing",
            )

    charge_centered = charges - float(charges.mean())
    order = np.argsort(np.argsort(charges))
    charge_rank_norm = order.astype(np.float32) / max(1, heavy_atom_count - 1)
    if wbo is None:
        wbo_sum = np.zeros(heavy_atom_count, dtype=np.float32)
        wbo_max = np.zeros(heavy_atom_count, dtype=np.float32)
    else:
        wbo_sum = np.clip(wbo[:, 0] / 4.0, 0.0, 2.0)
        wbo_max = np.clip(wbo[:, 1] / 2.0, 0.0, 2.0)

    features = np.stack(
        [
            charges,
            np.abs(charges),
            charge_centered,
            charge_rank_norm,
            wbo_sum,
            wbo_max,
        ],
        axis=-1,
    ).astype(np.float32)
    valid_mask = np.ones((heavy_atom_count, 1), dtype=np.float32)
    return {
        "canonical_smiles": canonical_smiles,
        "xtb_valid": True,
        "atom_valid_mask": valid_mask.tolist(),
        "atom_features": features.tolist(),
        "feature_names": list(XTB_FEATURE_NAMES),
        "status": "ok",
        "error": None,
    }


def load_or_compute_xtb_features(
    smiles: str,
    *,
    cache_dir: str | Path,
    compute_if_missing: bool = True,
    xtb_path: str = "xtb",
    solvent: str = "water",
) -> Dict[str, object]:
    prep = prepare_mol(smiles)
    cache_smiles = prep.canonical_smiles or smiles
    path = _cache_path(cache_smiles, cache_dir)
    if path.exists():
        return json.loads(path.read_text())
    if not compute_if_missing:
        num_atoms = prep.mol.GetNumAtoms() if prep.mol is not None else 0
        return _invalid_xtb_payload(
            prep.canonical_smiles,
            num_atoms,
            status="missing",
            error="cache_missing",
        )
    payload = compute_xtb_atom_features(cache_smiles, xtb_path=xtb_path, solvent=solvent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def attach_xtb_features_to_graph(
    graph: MoleculeGraph,
    *,
    cache_dir: str | Path,
    compute_if_missing: bool = False,
    xtb_path: str = "xtb",
    solvent: str = "water",
) -> MoleculeGraph:
    payload = load_or_compute_xtb_features(
        graph.canonical_smiles or graph.smiles,
        cache_dir=cache_dir,
        compute_if_missing=compute_if_missing,
        xtb_path=xtb_path,
        solvent=solvent,
    )
    num_atoms = int(graph.num_atoms)
    raw_features = np.asarray(payload.get("atom_features") or [], dtype=np.float32)
    if raw_features.size == 0:
        raw_features = np.zeros((num_atoms, XTB_FEATURE_DIM), dtype=np.float32)
    raw_valid = np.asarray(payload.get("atom_valid_mask") or [], dtype=np.float32)
    if raw_valid.size == 0:
        raw_valid = np.zeros((num_atoms, 1), dtype=np.float32)
    raw_features = raw_features[:num_atoms]
    raw_valid = raw_valid[:num_atoms]
    if raw_features.shape[0] < num_atoms:
        pad = np.zeros((num_atoms - raw_features.shape[0], XTB_FEATURE_DIM), dtype=np.float32)
        raw_features = np.concatenate([raw_features, pad], axis=0)
    if raw_valid.shape[0] < num_atoms:
        pad = np.zeros((num_atoms - raw_valid.shape[0], 1), dtype=np.float32)
        raw_valid = np.concatenate([raw_valid, pad], axis=0)

    graph.xtb_atom_features = raw_features
    graph.xtb_atom_valid_mask = raw_valid
    graph.xtb_mol_valid = np.asarray([[1.0 if payload_training_xtb_valid(payload) else 0.0]], dtype=np.float32)
    graph.xtb_feature_status = str(payload.get("status") or "missing")
    graph.xtb_status_flags = xtb_status_vector(graph.xtb_feature_status)
    return graph


def _candidate_atom_index(candidate: Dict[str, Any]) -> Optional[int]:
    heavy_atom = candidate.get("heavy_atom_index")
    if isinstance(heavy_atom, int) and heavy_atom >= 0:
        return int(heavy_atom)
    atom_index = candidate.get("atom_index", candidate.get("index"))
    if isinstance(atom_index, int) and atom_index >= 0:
        return int(atom_index)
    atom_indices = candidate.get("atom_indices")
    if isinstance(atom_indices, list):
        for value in atom_indices:
            if isinstance(value, int) and value >= 0:
                return int(value)
    return None


def _first_numeric(*values: object) -> Optional[float]:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _status_to_confidence(source: Optional[str], xtb_status: Optional[str]) -> float:
    key = str(source or xtb_status or "unknown").strip().lower()
    confidence_map = {
        "xtb_validated": 1.0,
        "lookup_safeguard": 0.8,
        "lookup_fallback": 0.6,
        "xtb_only": 0.7,
        "ok": 0.75,
        "disabled": 0.4,
        "unknown": 0.5,
        "embedding_failed": 0.35,
        "xtb_failed": 0.35,
        "xtb_error": 0.35,
    }
    return float(confidence_map.get(key, 0.5))


def extract_full_xtb_features(module_minus1_result: Dict[str, Any], num_atoms: int):
    from enzyme_software.liquid_nn_v2._compat import require_torch, torch

    require_torch()
    features = torch.zeros((int(num_atoms), FULL_XTB_FEATURE_DIM), dtype=torch.float32)
    if not module_minus1_result:
        return features

    resolved = module_minus1_result.get("resolved_target") or {}
    cpt_scores = module_minus1_result.get("cpt_scores") or {}
    global_bde = cpt_scores.get("bde") or module_minus1_result.get("bde") or {}
    candidates = (
        module_minus1_result.get("candidate_sites")
        or resolved.get("candidate_bonds")
        or resolved.get("candidate_sites")
        or []
    )

    populated = False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        atom_idx = _candidate_atom_index(candidate)
        if atom_idx is None or not (0 <= atom_idx < int(num_atoms)):
            continue
        bde_payload = candidate.get("bde") if isinstance(candidate.get("bde"), dict) else {}
        corrected = _first_numeric(
            bde_payload.get("corrected_kj_mol"),
            candidate.get("corrected_kj_mol"),
            candidate.get("bde_kj_mol"),
            global_bde.get("corrected_kj_mol"),
            global_bde.get("bde_kj_mol"),
        )
        if corrected is None:
            continue
        vertical = _first_numeric(
            bde_payload.get("xtb_bde_vertical_kj_mol"),
            candidate.get("xtb_bde_vertical_kj_mol"),
            global_bde.get("xtb_bde_vertical_kj_mol"),
            corrected,
        )
        adiabatic = _first_numeric(
            bde_payload.get("xtb_bde_adiabatic_kj_mol"),
            candidate.get("xtb_bde_adiabatic_kj_mol"),
            global_bde.get("xtb_bde_adiabatic_kj_mol"),
            corrected,
        )
        relaxation = _first_numeric(
            bde_payload.get("xtb_relaxation_energy_kj_mol"),
            candidate.get("xtb_relaxation_energy_kj_mol"),
            global_bde.get("xtb_relaxation_energy_kj_mol"),
            (vertical - adiabatic) if vertical is not None and adiabatic is not None else None,
        )
        lookup_bde = _first_numeric(
            bde_payload.get("lookup_bde_kj_mol"),
            candidate.get("lookup_bde_kj_mol"),
            global_bde.get("lookup_bde_kj_mol"),
            corrected,
        )
        deviation = _first_numeric(
            bde_payload.get("deviation_kj_mol"),
            candidate.get("deviation_kj_mol"),
            global_bde.get("deviation_kj_mol"),
            abs(corrected - lookup_bde) if lookup_bde is not None else 0.0,
        )
        source = bde_payload.get("source") or candidate.get("source") or global_bde.get("source")
        xtb_status = bde_payload.get("xtb_status") or candidate.get("xtb_status") or global_bde.get("xtb_status")
        confidence = _status_to_confidence(source, xtb_status)

        features[atom_idx, 0] = normalize_bde(corrected)
        features[atom_idx, 1] = 1.0 / (float(corrected) + 1.0)
        features[atom_idx, 2] = normalize_bde(vertical if vertical is not None else corrected)
        features[atom_idx, 3] = normalize_bde(adiabatic if adiabatic is not None else corrected)
        features[atom_idx, 4] = float(np.clip((relaxation or 0.0) / 50.0, 0.0, 1.0))
        features[atom_idx, 5] = float(np.clip((deviation or 0.0) / 30.0, 0.0, 1.0))
        features[atom_idx, 6] = confidence
        features[atom_idx, 7] = float(np.clip((lookup_bde or corrected) / 500.0, 0.0, 1.5))
        populated = True

    if populated:
        return features

    selected_bond_indices = resolved.get("bond_indices") or []
    fallback_atom_idx = None
    if isinstance(selected_bond_indices, list):
        for value in selected_bond_indices:
            if isinstance(value, int) and 0 <= value < int(num_atoms):
                fallback_atom_idx = int(value)
                break
    if fallback_atom_idx is None or not global_bde:
        return features

    corrected = _first_numeric(global_bde.get("corrected_kj_mol"), global_bde.get("bde_kj_mol"))
    if corrected is None:
        return features
    vertical = _first_numeric(global_bde.get("xtb_bde_vertical_kj_mol"), corrected)
    adiabatic = _first_numeric(global_bde.get("xtb_bde_adiabatic_kj_mol"), corrected)
    relaxation = _first_numeric(
        global_bde.get("xtb_relaxation_energy_kj_mol"),
        (vertical - adiabatic) if vertical is not None and adiabatic is not None else None,
        0.0,
    )
    lookup_bde = _first_numeric(global_bde.get("lookup_bde_kj_mol"), corrected)
    deviation = _first_numeric(global_bde.get("deviation_kj_mol"), abs(corrected - lookup_bde))
    confidence = _status_to_confidence(global_bde.get("source"), global_bde.get("xtb_status"))
    features[fallback_atom_idx, 0] = normalize_bde(corrected)
    features[fallback_atom_idx, 1] = 1.0 / (float(corrected) + 1.0)
    features[fallback_atom_idx, 2] = normalize_bde(vertical if vertical is not None else corrected)
    features[fallback_atom_idx, 3] = normalize_bde(adiabatic if adiabatic is not None else corrected)
    features[fallback_atom_idx, 4] = float(np.clip((relaxation or 0.0) / 50.0, 0.0, 1.0))
    features[fallback_atom_idx, 5] = float(np.clip((deviation or 0.0) / 30.0, 0.0, 1.0))
    features[fallback_atom_idx, 6] = confidence
    features[fallback_atom_idx, 7] = float(np.clip((lookup_bde or corrected) / 500.0, 0.0, 1.5))
    return features


def compute_full_xtb_payload(
    smiles: str,
    *,
    target_bond: Optional[str] = None,
) -> Dict[str, object]:
    if Chem is None:
        return _invalid_full_xtb_payload(None, 0, status="rdkit_unavailable", error="RDKit unavailable")

    from enzyme_software.liquid_nn_v2.features.manual_engine_features import infer_target_bond
    from enzyme_software.pipeline import run_pipeline

    prep = prepare_mol(smiles)
    if prep.mol is None:
        return _invalid_full_xtb_payload(None, 0, status="prepare_failed", error=prep.error)

    canonical_smiles = prep.canonical_smiles or smiles
    num_atoms = prep.mol.GetNumAtoms()
    resolved_target = str(target_bond or infer_target_bond(canonical_smiles))
    try:
        ctx = run_pipeline(smiles=canonical_smiles, target_bond=resolved_target)
    except Exception as exc:
        return _invalid_full_xtb_payload(canonical_smiles, num_atoms, status="manual_engine_error", error=str(exc))

    module_minus1 = ctx.data.get("module_minus1") or {}
    source_summary = _full_xtb_source_summary(module_minus1)
    tensor = extract_full_xtb_features(module_minus1, num_atoms)
    valid_mask = (tensor.abs().sum(dim=1, keepdim=True) > 0).to(dtype=tensor.dtype)
    xtb_valid = bool(valid_mask.any().item())
    return {
        "cache_schema_version": FULL_XTB_CACHE_SCHEMA_VERSION,
        "canonical_smiles": canonical_smiles,
        "xtb_valid": xtb_valid,
        "true_xtb_valid": bool(source_summary["true_xtb_used"]) and xtb_valid,
        "atom_valid_mask": valid_mask.tolist(),
        "atom_features": tensor.tolist(),
        "feature_names": list(FULL_XTB_FEATURE_NAMES),
        "status": "ok" if xtb_valid else "no_xtb_payload",
        "xtb_source_kind": source_summary["xtb_source_kind"],
        "true_xtb_used": bool(source_summary["true_xtb_used"]),
        "lookup_only": bool(source_summary["lookup_only"]),
        "manual_fallback_used": bool(source_summary["manual_fallback_used"]),
        "resolved_target_bond": resolved_target,
        "source_names": list(source_summary["source_names"]),
        "status_names": list(source_summary["status_names"]),
        "error": None,
    }


def load_or_compute_full_xtb_features(
    smiles: str,
    *,
    cache_dir: str | Path,
    compute_if_missing: bool = True,
    target_bond: Optional[str] = None,
) -> Dict[str, object]:
    prep = prepare_mol(smiles)
    cache_smiles = prep.canonical_smiles or smiles
    resolved_target_bond = str(target_bond).strip() if str(target_bond or "").strip() else None
    path = _full_cache_path(cache_smiles, cache_dir, target_bond=resolved_target_bond)
    legacy_path = _legacy_full_cache_path(cache_smiles, cache_dir)
    existing_path = path if path.exists() else (legacy_path if legacy_path.exists() else None)
    if existing_path is not None:
        payload = _normalize_full_xtb_payload(
            json.loads(existing_path.read_text()),
            canonical_smiles=prep.canonical_smiles,
            num_atoms=prep.mol.GetNumAtoms() if prep.mol is not None else 0,
            resolved_target_bond=resolved_target_bond,
        )
        if existing_path != path:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    if not compute_if_missing:
        num_atoms = prep.mol.GetNumAtoms() if prep.mol is not None else 0
        return _invalid_full_xtb_payload(
            prep.canonical_smiles,
            num_atoms,
            status="missing",
            error="cache_missing",
        )
    payload = compute_full_xtb_payload(cache_smiles, target_bond=resolved_target_bond)
    payload = _normalize_full_xtb_payload(
        payload,
        canonical_smiles=prep.canonical_smiles,
        num_atoms=prep.mol.GetNumAtoms() if prep.mol is not None else 0,
        resolved_target_bond=resolved_target_bond,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
