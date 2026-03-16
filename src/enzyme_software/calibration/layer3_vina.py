"""Layer 3 Vina integration for Module 2 docking."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import urllib.request as urllib_request
except Exception:  # pragma: no cover - optional dependency
    urllib_request = None

try:
    from enzyme_software.computational_engines import CYP_ACTIVE_SITE_CENTERS, VinaEngine
except Exception:  # pragma: no cover - optional dependency
    CYP_ACTIVE_SITE_CENTERS = {}
    VinaEngine = None  # type: ignore[assignment]


L3_VINA_VERSION = "2026-03-11"
_CACHE_ROOT = Path(__file__).resolve().parents[3] / "cache" / "vina"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_PDB_CACHE = _CACHE_ROOT / "pdb"
_PDB_CACHE.mkdir(parents=True, exist_ok=True)
_PDBQT_CACHE = _CACHE_ROOT / "pdbqt"
_PDBQT_CACHE.mkdir(parents=True, exist_ok=True)

_CYP_ISOFORM_TO_PDB_RAW: Dict[str, str] = {
    "CYP3A4": "1TQN",
    "CYP2D6": "2F9Q",
    "CYP2C9": "1OG5",
    "CYP2C19": "4GQS",
    "CYP1A2": "2HI4",
    "CYP2E1": "3E6I",
    "CYP2B6": "3IBD",
    "CYP2A6": "1Z10",
    "P450CAM": "2CPP",
    "P450BM3": "1FAG",
    "3A4": "1TQN",
    "2D6": "2F9Q",
    "2C9": "1OG5",
    "2C19": "4GQS",
    "1A2": "2HI4",
}
CYP_ISOFORM_TO_PDB: Dict[str, str] = {
    "".join(ch for ch in key.upper() if ch.isalnum()): value
    for key, value in _CYP_ISOFORM_TO_PDB_RAW.items()
}

DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_N_POSES = 5
DEFAULT_BOX_SIZE = (22.0, 22.0, 22.0)


def _env_enabled(name: str, default: str) -> bool:
    raw = str(os.environ.get(name, default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_layer3_vina_available() -> bool:
    if VinaEngine is None:
        return False
    try:
        return bool(VinaEngine().is_available())
    except Exception:
        return False


def is_layer3_vina_enabled() -> bool:
    return _env_enabled("LAYER3_VINA_ENABLED", "1") and is_layer3_vina_available()


def dock_substrate_in_cyp(
    smiles: str,
    *,
    receptor_pdbqt: Optional[str] = None,
    cyp_isoform: Optional[str] = None,
    pdb_id: Optional[str] = None,
    center: Optional[Tuple[float, float, float]] = None,
    box_size: Optional[Tuple[float, float, float]] = None,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    n_poses: int = DEFAULT_N_POSES,
    use_cache: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "pending",
        "binding_energy_kcal": None,
        "binding_energy_kj": None,
        "n_poses": 0,
        "topogate_scores": None,
        "distance_to_center_A": None,
        "receptor_source": None,
        "receptor_pdb_path": None,
        "receptor_pdbqt_path": None,
        "pdb_id": None,
        "cyp_isoform": cyp_isoform,
        "smiles": smiles,
        "layer3_vina_version": L3_VINA_VERSION,
        "error": None,
    }
    if not _env_enabled("LAYER3_VINA_ENABLED", "1"):
        result["status"] = "disabled"
        return result
    if not is_layer3_vina_available():
        result["status"] = "unavailable"
        result["error"] = "Vina engine not installed"
        return result
    if not isinstance(smiles, str) or not smiles.strip():
        result["status"] = "error"
        result["error"] = "Invalid or missing SMILES"
        return result

    cache_key = _compute_cache_key(smiles, receptor_pdbqt, cyp_isoform, pdb_id)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    receptor_info = _resolve_receptor(
        receptor_pdbqt=receptor_pdbqt,
        cyp_isoform=cyp_isoform,
        pdb_id=pdb_id,
        center=center,
    )
    result["receptor_source"] = receptor_info["source"]
    result["pdb_id"] = receptor_info["pdb_id"]
    result["receptor_pdb_path"] = receptor_info["receptor_pdb_path"]
    result["receptor_pdbqt_path"] = receptor_info["receptor_pdbqt_path"]

    if not receptor_info["receptor_pdbqt_path"]:
        result["status"] = "no_receptor"
        result["error"] = f"Could not resolve receptor: {receptor_info['source']}"
        return result

    resolved_center = receptor_info.get("center")
    if resolved_center is None and receptor_info["pdb_id"]:
        site = CYP_ACTIVE_SITE_CENTERS.get(str(receptor_info["pdb_id"]).upper(), {})
        if site.get("center"):
            resolved_center = tuple(site["center"])
        if box_size is None and site.get("box"):
            box_size = tuple(site["box"])
    if resolved_center is None:
        result["status"] = "error"
        result["error"] = "No active site center defined for receptor"
        return result
    if box_size is None:
        box_size = DEFAULT_BOX_SIZE

    try:
        engine = VinaEngine(exhaustiveness=exhaustiveness, n_poses=n_poses)
        dock_result = engine.dock(
            receptor_pdbqt=receptor_info["receptor_pdbqt_path"],
            ligand_smiles=smiles,
            pdb_id=receptor_info["pdb_id"],
            center=resolved_center,
            box_size=box_size,
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Docking failed: {exc}"
        return result

    if dock_result.get("error"):
        result["status"] = "error"
        result["error"] = dock_result.get("error")
        return result

    result.update(
        {
            "status": "ok",
            "binding_energy_kcal": dock_result.get("binding_energy_kcal"),
            "binding_energy_kj": dock_result.get("binding_energy_kj"),
            "n_poses": dock_result.get("n_poses", 0),
            "topogate_scores": dock_result.get("topogate_scores"),
            "distance_to_center_A": dock_result.get("distance_to_center_A"),
            "all_energies_kcal": dock_result.get("all_energies_kcal"),
            "ligand_centroid": dock_result.get("ligand_centroid"),
        }
    )
    if use_cache:
        _save_to_cache(cache_key, result)
    result["cache_hit"] = False
    return result


def dock_for_module2(
    job_card: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    smiles = _extract_smiles_from_job_card(job_card)
    if not smiles:
        return {
            "status": "error",
            "error": "No SMILES found in job_card",
            "layer3_vina_version": L3_VINA_VERSION,
        }
    constraint_map = constraints or {}
    return dock_substrate_in_cyp(
        smiles=smiles,
        receptor_pdbqt=constraint_map.get("receptor_pdbqt") or constraint_map.get("receptor_path"),
        cyp_isoform=constraint_map.get("cyp_isoform") or _extract_cyp_isoform_from_job_card(job_card),
        pdb_id=constraint_map.get("receptor_pdb_id") or constraint_map.get("pdb_id"),
    )


def _resolve_receptor(
    receptor_pdbqt: Optional[str],
    cyp_isoform: Optional[str],
    pdb_id: Optional[str],
    center: Optional[Tuple[float, float, float]],
) -> Dict[str, Any]:
    if receptor_pdbqt and os.path.exists(str(receptor_pdbqt)):
        pdbqt_path = str(receptor_pdbqt)
        pdb_guess = str(Path(pdbqt_path).with_suffix(".pdb"))
        return {
            "receptor_pdbqt_path": pdbqt_path,
            "receptor_pdb_path": pdb_guess if os.path.exists(pdb_guess) else None,
            "source": "user_provided",
            "pdb_id": str(pdb_id).upper() if pdb_id else None,
            "center": center,
        }

    iso_key = _normalize_isoform(cyp_isoform)
    resolved_pdb = CYP_ISOFORM_TO_PDB.get(iso_key) if iso_key else None
    if resolved_pdb:
        bundle = _get_or_fetch_receptor_bundle(resolved_pdb)
        if bundle["receptor_pdbqt_path"]:
            site = CYP_ACTIVE_SITE_CENTERS.get(resolved_pdb, {})
            return {
                "receptor_pdbqt_path": bundle["receptor_pdbqt_path"],
                "receptor_pdb_path": bundle["receptor_pdb_path"],
                "source": f"cyp_isoform:{cyp_isoform}->{resolved_pdb}",
                "pdb_id": resolved_pdb,
                "center": tuple(site["center"]) if site.get("center") else center,
            }

    if pdb_id:
        pdb_key = str(pdb_id).upper()
        bundle = _get_or_fetch_receptor_bundle(pdb_key)
        if bundle["receptor_pdbqt_path"]:
            site = CYP_ACTIVE_SITE_CENTERS.get(pdb_key, {})
            return {
                "receptor_pdbqt_path": bundle["receptor_pdbqt_path"],
                "receptor_pdb_path": bundle["receptor_pdb_path"],
                "source": f"pdb_id:{pdb_key}",
                "pdb_id": pdb_key,
                "center": tuple(site["center"]) if site.get("center") else center,
            }

    for known_pdb, site in CYP_ACTIVE_SITE_CENTERS.items():
        bundle = _get_or_fetch_receptor_bundle(known_pdb)
        if bundle["receptor_pdbqt_path"]:
            return {
                "receptor_pdbqt_path": bundle["receptor_pdbqt_path"],
                "receptor_pdb_path": bundle["receptor_pdb_path"],
                "source": f"fallback_default:{known_pdb}",
                "pdb_id": known_pdb,
                "center": tuple(site["center"]) if site.get("center") else center,
            }

    return {
        "receptor_pdbqt_path": None,
        "receptor_pdb_path": None,
        "source": "no_receptor_available",
        "pdb_id": None,
        "center": None,
    }


def _get_or_fetch_receptor_bundle(pdb_id: str) -> Dict[str, Optional[str]]:
    pdb_key = str(pdb_id).upper()
    pdb_path = _fetch_pdb(pdb_key)
    pdbqt_path = _PDBQT_CACHE / f"{pdb_key}.pdbqt"
    if pdbqt_path.exists() and _is_valid_receptor_pdbqt(pdbqt_path):
        return {
            "receptor_pdb_path": str(pdb_path) if pdb_path else None,
            "receptor_pdbqt_path": str(pdbqt_path),
        }
    if pdb_path is None or VinaEngine is None:
        return {"receptor_pdb_path": str(pdb_path) if pdb_path else None, "receptor_pdbqt_path": None}
    prep_bin = shutil.which("prepare_receptor4.py") or shutil.which("prepare_receptor")
    if prep_bin is None:
        return {"receptor_pdb_path": str(pdb_path), "receptor_pdbqt_path": None}
    try:
        prepared = VinaEngine().prepare_receptor(str(pdb_path), str(pdbqt_path))
    except Exception:
        prepared = None
    return {
        "receptor_pdb_path": str(pdb_path) if pdb_path else None,
        "receptor_pdbqt_path": (
            prepared
            if prepared and os.path.exists(prepared) and _is_valid_receptor_pdbqt(Path(prepared))
            else None
        ),
    }


def _fetch_pdb(pdb_id: str) -> Optional[Path]:
    pdb_path = _PDB_CACHE / f"{str(pdb_id).upper()}.pdb"
    if pdb_path.exists():
        return pdb_path
    if urllib_request is None:
        return None
    url = f"https://files.rcsb.org/download/{str(pdb_id).upper()}.pdb"
    try:
        urllib_request.urlretrieve(url, str(pdb_path))
    except Exception:
        return None
    if pdb_path.exists() and pdb_path.stat().st_size > 100:
        return pdb_path
    return None


def _extract_smiles_from_job_card(job_card: Dict[str, Any]) -> Optional[str]:
    for item in (
        job_card.get("smiles"),
        job_card.get("substrate_smiles"),
        (job_card.get("substrate_context") or {}).get("smiles"),
        (job_card.get("reaction_identity") or {}).get("smiles"),
        ((job_card.get("shared_io") or {}).get("input") or {}).get("substrate_context", {}).get("smiles"),
    ):
        if isinstance(item, str) and item.strip():
            return item.strip()
    reaction_task = job_card.get("reaction_task") or {}
    substrates = reaction_task.get("substrates") or []
    for item in substrates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _extract_cyp_isoform_from_job_card(job_card: Dict[str, Any]) -> Optional[str]:
    cyp_pred = job_card.get("cyp_prediction") or {}
    if cyp_pred.get("primary_isoform"):
        return str(cyp_pred.get("primary_isoform"))
    metab = job_card.get("metabolism_prediction") or {}
    if metab.get("primary_cyp"):
        return str(metab.get("primary_cyp"))
    route = (job_card.get("mechanism_route") or {}).get("primary") or job_card.get("chosen_route") or ""
    route_upper = str(route).upper()
    for raw_key in _CYP_ISOFORM_TO_PDB_RAW:
        if raw_key in route_upper:
            return raw_key
    if "P450" in route_upper:
        return "CYP3A4"
    return None


def _normalize_isoform(value: Optional[str]) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _compute_cache_key(
    smiles: str,
    receptor_pdbqt: Optional[str],
    cyp_isoform: Optional[str],
    pdb_id: Optional[str],
) -> str:
    payload = "|".join(
        [
            f"smiles:{smiles}",
            f"receptor:{receptor_pdbqt or 'auto'}",
            f"isoform:{cyp_isoform or 'none'}",
            f"pdb:{pdb_id or 'none'}",
            f"version:{L3_VINA_VERSION}",
        ]
    )
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


def _is_valid_receptor_pdbqt(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    lines = text.splitlines()
    if not any(line.startswith(("ATOM", "HETATM")) for line in lines):
        return False
    leading = lines[:10]
    if any(line.startswith(("CRYST1", "HEADER", "TITLE", "COMPND")) for line in leading):
        return False
    return True


__all__ = [
    "CYP_ISOFORM_TO_PDB",
    "L3_VINA_VERSION",
    "dock_for_module2",
    "dock_substrate_in_cyp",
    "is_layer3_vina_available",
    "is_layer3_vina_enabled",
]
