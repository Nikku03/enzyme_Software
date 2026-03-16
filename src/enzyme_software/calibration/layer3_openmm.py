"""Layer 3 OpenMM integration for Module 2 stability checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from enzyme_software.computational_engines import OpenMMEngine
except Exception:  # pragma: no cover - optional dependency
    OpenMMEngine = None  # type: ignore[assignment]


L3_OPENMM_VERSION = "2026-03-11"
_CACHE_ROOT = Path(__file__).resolve().parents[3] / "cache" / "openmm"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_N_STEPS = 25000
DEFAULT_TEMPERATURE_K = 300.0


def _env_enabled(name: str, default: str) -> bool:
    raw = str(os.environ.get(name, default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_layer3_openmm_available() -> bool:
    if OpenMMEngine is None:
        return False
    try:
        return bool(OpenMMEngine().is_available())
    except Exception:
        return False


def is_layer3_openmm_enabled() -> bool:
    return _env_enabled("LAYER3_OPENMM_ENABLED", "0") and is_layer3_openmm_available()


def compute_binding_stability(
    pdb_path: str,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    temperature_K: float = DEFAULT_TEMPERATURE_K,
    use_cache: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "pending",
        "verdict": None,
        "energy_stable": None,
        "minimization": None,
        "md": None,
        "last_5_pe_mean_kj": None,
        "last_5_pe_std_kj": None,
        "layer3_openmm_version": L3_OPENMM_VERSION,
        "error": None,
    }
    if not _env_enabled("LAYER3_OPENMM_ENABLED", "0"):
        result["status"] = "disabled"
        return result
    if not is_layer3_openmm_available():
        result["status"] = "unavailable"
        result["error"] = "OpenMM engine not installed"
        return result
    if not pdb_path or not os.path.exists(str(pdb_path)):
        result["status"] = "error"
        result["error"] = f"PDB file not found: {pdb_path}"
        return result

    cache_key = _compute_cache_key(pdb_path, n_steps, temperature_K)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    try:
        engine = OpenMMEngine(temperature_K=temperature_K)
        system_data = engine.prepare_system(str(pdb_path))
        if system_data.get("error"):
            result["status"] = "error"
            result["error"] = system_data.get("error")
            return result
        min_result = engine.energy_minimize(system_data)
        if min_result.get("error"):
            result["status"] = "error"
            result["error"] = min_result.get("error")
            return result
        system_data["positions"] = min_result.get("positions", system_data.get("positions"))
        md_result = engine.run_md(system_data, n_steps=n_steps)
        if md_result.get("error"):
            result["status"] = "error"
            result["error"] = md_result.get("error")
            return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"MD simulation failed: {exc}"
        return result

    result.update(
        {
            "status": "ok",
            "verdict": "STABLE" if md_result.get("energy_stable") else "UNSTABLE",
            "energy_stable": md_result.get("energy_stable"),
            "minimization": {
                "energy_before_kj": min_result.get("energy_before_kj"),
                "energy_after_kj": min_result.get("energy_after_kj"),
                "energy_change_kj": min_result.get("energy_change_kj"),
            },
            "md": {
                "total_time_ps": md_result.get("total_time_ps"),
                "energy_stable": md_result.get("energy_stable"),
                "final_pe_kj": md_result.get("final_potential_kj"),
                "n_steps": md_result.get("n_steps"),
                "temperature_K": md_result.get("temperature_K"),
            },
            "last_5_pe_mean_kj": md_result.get("last_5_pe_mean_kj"),
            "last_5_pe_std_kj": md_result.get("last_5_pe_std_kj"),
        }
    )
    if use_cache:
        _save_to_cache(cache_key, result)
    result["cache_hit"] = False
    return result


def stability_for_module2(
    docking_result: Dict[str, Any],
    job_card: Dict[str, Any],
) -> Dict[str, Any]:
    receptor_pdb = docking_result.get("receptor_pdb_path")
    if not receptor_pdb:
        receptor_pdbqt = docking_result.get("receptor_pdbqt_path") or docking_result.get("receptor")
        if receptor_pdbqt and str(receptor_pdbqt).lower().endswith(".pdbqt"):
            guess = str(Path(str(receptor_pdbqt)).with_suffix(".pdb"))
            if os.path.exists(guess):
                receptor_pdb = guess
    if not receptor_pdb:
        return {
            "status": "skipped",
            "error": "No receptor PDB available for MD",
            "layer3_openmm_version": L3_OPENMM_VERSION,
        }
    return compute_binding_stability(str(receptor_pdb))


def _compute_cache_key(pdb_path: str, n_steps: int, temperature_K: float) -> str:
    try:
        content_hash = hashlib.sha256(Path(pdb_path).read_bytes()).hexdigest()[:12]
    except Exception:
        content_hash = hashlib.sha256(str(pdb_path).encode("utf-8")).hexdigest()[:12]
    payload = f"pdb:{content_hash}|steps:{int(n_steps)}|temp:{float(temperature_K)}|version:{L3_OPENMM_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    cache_file = _CACHE_ROOT / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_to_cache(cache_key: str, result: Dict[str, Any]) -> None:
    cache_file = _CACHE_ROOT / f"{cache_key}.json"
    try:
        cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass


__all__ = [
    "L3_OPENMM_VERSION",
    "compute_binding_stability",
    "is_layer3_openmm_available",
    "is_layer3_openmm_enabled",
    "stability_for_module2",
]
