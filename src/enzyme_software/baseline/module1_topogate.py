from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.modules.base import BaseModule

MODULE1_VERSION = "v1.0"
CACHE_PATH = Path(__file__).resolve().parents[3] / "cache" / "module1_cache.json"

FAIL_NO_POCKET = "FAIL_NO_POCKET"
FAIL_NO_TUNNEL_COARSE = "FAIL_NO_TUNNEL_COARSE"
FAIL_TUNNEL_TOO_NARROW = "FAIL_TUNNEL_TOO_NARROW"
FAIL_TUNNEL_COLLAPSE = "FAIL_TUNNEL_COLLAPSE"
FAIL_REACH_FAIL = "FAIL_REACH_FAIL"
FAIL_MECH_COMPAT = "FAIL_MECH_COMPAT"
FAIL_RETENTION_IMPOSSIBLE = "FAIL_RETENTION_IMPOSSIBLE"
FAIL_ALL_REJECTED = "FAIL_ALL_REJECTED"
PASS_TOPK_SELECTED = "PASS_TOPK_SELECTED"
WARN_RETENTION_WEAK_BINDING = "WARN_RETENTION_WEAK_BINDING"
WARN_TUNNEL_BORDERLINE = "WARN_TUNNEL_BORDERLINE"


@dataclass
class Scaffold:
    scaffold_id: str
    pdb_path: str
    track: Optional[str] = None


class Module1TopoGate(BaseModule):
    name = "Module 1 - TopoGate + ReachGate + RetentionGate"

    def __init__(self, scaffold_library: Optional[List[Dict[str, Any]]] = None) -> None:
        self._scaffold_library = scaffold_library

    def run(self, ctx: PipelineContext) -> PipelineContext:
        job_card = ctx.data.get("job_card") or {}
        if not job_card:
            ctx.data["module1_topogate"] = {
                "handoff": {},
                "status": "FAIL",
                "halt_reason": "FAIL_MISSING_JOB_CARD",
            }
            return ctx
        module0_job_card = (ctx.data.get("module0_strategy_router") or {}).get("job_card")
        if module0_job_card and module0_job_card != job_card:
            warnings = job_card.get("warnings") or []
            warnings.append(
                "W_JOB_CARD_MISMATCH: module0_strategy_router.job_card differs from data.job_card; using data.job_card."
            )
            job_card["warnings"] = list(dict.fromkeys(warnings))
            ctx.data["job_card"] = job_card

        result = run_module1(
            smiles=ctx.smiles,
            job_card=job_card,
            constraints=ctx.constraints,
            scaffold_library=self._scaffold_library or ctx.data.get("scaffold_library"),
        )
        module1_confidence = result.get("module1_confidence") or {}
        total_confidence = module1_confidence.get("total")
        if isinstance(total_confidence, (int, float)):
            confidence = job_card.get("confidence") or {}
            feasibility = confidence.get("feasibility_if_specified")
            if not isinstance(feasibility, (int, float)):
                feasibility = confidence.get("route", 0.0)
            completeness = confidence.get("completeness")
            if not isinstance(completeness, (int, float)):
                completeness = 1.0
            adjusted = min(float(feasibility), 0.6 + 0.4 * float(total_confidence))
            confidence["feasibility_if_specified"] = round(adjusted, 3)
            confidence["route"] = round(adjusted * float(completeness), 3)
            confidence["module1_total"] = round(float(total_confidence), 3)
            job_card["confidence"] = confidence
            ctx.data["job_card"] = job_card
        ctx.data["module1_topogate"] = result
        return ctx


def run_module1(
    smiles: str,
    job_card: Dict[str, Any],
    constraints: OperationalConstraints,
    scaffold_library: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    compute_plan = job_card.get("compute_plan") or {}
    route = job_card.get("mechanism_route") or {}
    bond_context = job_card.get("bond_context") or {}
    structure_summary = job_card.get("structure_summary") or {}
    resolved = job_card.get("resolved_target") or {}
    difficulty = job_card.get("difficulty_label") or job_card.get("difficulty") or "MEDIUM"
    size_proxies = job_card.get("substrate_size_proxies") or {}

    if not resolved.get("selected_bond") and not resolved.get("atom_indices"):
        return _fail_handoff("FAIL_MISSING_TARGET")

    scaffold_count = int(compute_plan.get("scaffold_count") or 0)
    if scaffold_count <= 0:
        return _fail_handoff("FAIL_NO_SCAFFOLD_TARGET")

    strictness = compute_plan.get("topogate_strictness") or "standard"
    mode, weights = _determine_mode(bond_context, structure_summary)
    mode_override = job_card.get("module1_mode")
    weights_override = job_card.get("module1_weights")
    if mode_override and _valid_weights(weights_override):
        mode = mode_override
        weights = weights_override
    target_role = bond_context.get("primary_role") or bond_context.get("bond_role") or "unknown"

    scaffolds = _select_scaffolds(
        scaffold_library=scaffold_library,
        scaffold_library_id=job_card.get("scaffold_library_id"),
        route=route,
        scaffold_count=scaffold_count,
        difficulty=difficulty,
    )
    if not scaffolds:
        return _fail_handoff("FAIL_NO_SCAFFOLDS")

    cache = _load_cache()
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0

    pass_a_results: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for scaffold in scaffolds:
        cache_key = _cache_key(
            scaffold_id=scaffold.scaffold_id,
            smiles=smiles,
            target_role=target_role,
            mode=mode,
            strictness=strictness,
        )
        cached = cache.get(cache_key)
        if cached:
            cache_hits += 1
            scaffold_state = dict(cached)
            if not scaffold_state.get("pocket_center_candidates"):
                scaffold_state["pocket_center_candidates"] = _detect_pocket_candidates(
                    scaffold_state
                )
            if not scaffold_state.get("pocket_center_candidates"):
                rejected.append(
                    _reject_scaffold(scaffold, [FAIL_NO_POCKET], scaffold_state)
                )
                continue
            if "access_score" not in scaffold_state or "tunnel_summary" not in scaffold_state:
                coarse = _topogate_coarse(
                    scaffold_state,
                    mode=mode,
                    strictness=strictness,
                    structure_summary=structure_summary,
                )
                if not coarse["pass"]:
                    rejected.append(
                        _reject_scaffold(
                            scaffold,
                            [FAIL_NO_TUNNEL_COARSE],
                            scaffold_state,
                            tunnel_summary=coarse["tunnel_summary"],
                        )
                    )
                    continue
                scaffold_state.update(coarse)
                scaffold_state["candidate_residues"] = _seed_reach_candidates(scaffold_state)
            if "candidate_residues" not in scaffold_state:
                scaffold_state["candidate_residues"] = _seed_reach_candidates(scaffold_state)
                cache[cache_key] = scaffold_state
                cache_writes += 1
        else:
            cache_misses += 1
            scaffold_state = _initialize_scaffold_state(scaffold)
            pocket_candidates = _detect_pocket_candidates(scaffold_state)
            if not pocket_candidates:
                rejected.append(
                    _reject_scaffold(scaffold, [FAIL_NO_POCKET], scaffold_state)
                )
                continue
            scaffold_state["pocket_center_candidates"] = pocket_candidates
            coarse = _topogate_coarse(
                scaffold_state,
                mode=mode,
                strictness=strictness,
                structure_summary=structure_summary,
            )
            if not coarse["pass"]:
                rejected.append(
                    _reject_scaffold(
                        scaffold,
                        [FAIL_NO_TUNNEL_COARSE],
                        scaffold_state,
                        tunnel_summary=coarse["tunnel_summary"],
                    )
                )
                continue
            scaffold_state.update(coarse)
            scaffold_state["candidate_residues"] = _seed_reach_candidates(scaffold_state)
            cache[cache_key] = scaffold_state
            cache_writes += 1

        pass_a_results.append(
            {
                "scaffold": scaffold,
                "state": scaffold_state,
            }
        )

    if not pass_a_results:
        _save_cache(cache)
        return _fail_handoff(
            FAIL_ALL_REJECTED,
            rejected=rejected,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_writes=cache_writes,
            cache_size=len(cache),
        )

    pass_a_results.sort(key=lambda item: item["state"].get("access_score", 0.0), reverse=True)
    pass_b_candidates = pass_a_results[: min(20, len(pass_a_results))]

    final_scaffolds: List[Dict[str, Any]] = []
    for entry in pass_b_candidates:
        scaffold = entry["scaffold"]
        state = entry["state"]
        refined = _topogate_refined(
            state,
            mode=mode,
            strictness=strictness,
            structure_summary=structure_summary,
            size_proxies=size_proxies,
        )
        if refined.get("fail_code"):
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [refined["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue
        warn_codes: List[str] = []
        if refined.get("warn_code"):
            warn_codes.append(refined["warn_code"])

        flexibility = _flexibility_check(state, mode=mode)
        if flexibility["fail_code"]:
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [flexibility["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue

        attack_envelope = _build_attack_envelope(
            state,
            route=route,
            strictness=strictness,
        )
        reach_summary = _reach_gate(
            state,
            attack_envelope=attack_envelope,
            route=route,
        )
        if reach_summary["fail_code"] and reach_summary["fail_code"] != FAIL_MECH_COMPAT:
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [reach_summary["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue

        retention_metrics = _retention_gate(
            state,
            structure_summary=structure_summary,
            route=route,
            mode=mode,
            size_proxies=size_proxies,
        )
        if retention_metrics["fail_code"]:
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [retention_metrics["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue

        access_score = refined["access_score"]
        reach_score = reach_summary["reach_score"]
        retention_score = retention_metrics["retention_score"]
        total_score = (
            weights["access"] * access_score
            + weights["reach"] * reach_score
            + weights["retention"] * retention_score
        )
        retention_multiplier = retention_metrics.get("score_multiplier", 1.0)
        if isinstance(retention_multiplier, (int, float)):
            total_score *= retention_multiplier
        fail_codes = []
        if reach_summary.get("fail_code") == FAIL_MECH_COMPAT:
            fail_codes.append(FAIL_MECH_COMPAT)
        if retention_metrics.get("warning_codes"):
            warn_codes.extend(retention_metrics["warning_codes"])
        if warn_codes:
            fail_codes.extend(warn_codes)

        access_confidence = _access_confidence(
            refined.get("tunnel_summary"), access_score
        )
        reach_confidence = (
            reach_summary.get("reach_geom_score", 0.0)
            * reach_summary.get("mechanism_compat", {}).get("score", 0.0)
        )
        retention_confidence = retention_score
        module1_total = (
            weights["access"] * access_confidence
            + weights["reach"] * reach_confidence
            + weights["retention"] * retention_confidence
        )
        if isinstance(retention_multiplier, (int, float)):
            module1_total *= retention_multiplier

        final_scaffolds.append(
            {
                "scaffold_id": scaffold.scaffold_id,
                "pdb_path": scaffold.pdb_path,
                "pocket_center": refined["pocket_center"],
                "tunnel_summary": refined["tunnel_summary"],
                "reach_summary": reach_summary,
                "attack_envelope": attack_envelope,
                "retention_metrics": retention_metrics,
                "scores": {
                    "access_score": round(access_score, 3),
                    "reach_score": round(reach_score, 3),
                    "retention_score": round(retention_score, 3),
                    "total": round(total_score, 3),
                },
                "module1_confidence": {
                    "access": round(access_confidence, 3),
                    "reach": round(reach_confidence, 3),
                    "retention": round(retention_confidence, 3),
                    "total": round(module1_total, 3),
                },
                "fail_codes": fail_codes,
            }
        )

    _save_cache(cache)

    if not final_scaffolds:
        return _fail_handoff(
            FAIL_ALL_REJECTED,
            rejected=rejected,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_writes=cache_writes,
            cache_size=len(cache),
        )

    top_k = _top_k_from_difficulty(difficulty, scaffold_count)
    final_scaffolds.sort(key=lambda item: item["scores"]["total"], reverse=True)
    ranked = final_scaffolds[:top_k]
    for scaffold in ranked:
        scaffold["fail_codes"] = list(dict.fromkeys(scaffold["fail_codes"] + [PASS_TOPK_SELECTED]))

    module2_candidates = [
        scaffold
        for scaffold in ranked
        if scaffold.get("reach_summary", {}).get("fail_code") != FAIL_MECH_COMPAT
        and (scaffold.get("reach_summary", {}).get("mechanism_compat", {}).get("score") or 0.0)
        > 0.0
    ]

    module2_handoff = {
        "top_scaffolds": [
            {
                "scaffold_id": scaffold["scaffold_id"],
                "pdb_path": scaffold["pdb_path"],
                "attack_envelope": scaffold["attack_envelope"],
                "candidate_residues_by_role": scaffold["reach_summary"].get(
                    "candidate_residues_by_role", {}
                ),
                "tunnel_metrics": scaffold["tunnel_summary"],
                "scores": {"total": scaffold["scores"]["total"]},
                "retention_metrics": {
                    "volume_ratio": scaffold["retention_metrics"].get("volume_ratio"),
                    "retention_risk_flag": scaffold["retention_metrics"].get(
                        "retention_risk_flag"
                    ),
                    "warning_codes": scaffold["retention_metrics"].get("warning_codes", []),
                    "score_multiplier": scaffold["retention_metrics"].get("score_multiplier"),
                },
                "reach_summary": {
                    "mechanism_compat_score": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("score"),
                    "nucleophile_type": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("nucleophile_type"),
                    "mechanism_label": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("mechanism_label"),
                    "required_flags": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("required_flags", []),
                    "nucleophile_geometry": _nucleophile_geometry(
                        scaffold["reach_summary"].get("mechanism_compat", {})
                    ),
                },
                "bond_center_hint": job_card.get("bond_center_hint") or {},
            }
            for scaffold in module2_candidates
        ]
    }

    module1_confidence = {}
    if ranked:
        best_by_conf = max(
            ranked,
            key=lambda item: item.get("module1_confidence", {}).get("total", 0.0),
        )
        module1_confidence = dict(best_by_conf.get("module1_confidence", {}))
        if module1_confidence:
            module1_confidence["scaffold_id"] = best_by_conf.get("scaffold_id")
    handoff = {
        "status": "PASS",
        "halt_reason": None,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries_written": cache_writes,
            "cache_size": len(cache),
        },
        "mode": mode,
        "weights": weights,
        "module1_confidence": module1_confidence,
        "ranked_scaffolds": ranked,
        "rejected_scaffolds": rejected,
        "module2_handoff": module2_handoff,
    }
    return handoff


def _determine_mode(
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    heavy_atoms = structure_summary.get("heavy_atoms") or 0
    rotatable = structure_summary.get("rotatable_bonds") or 0
    gas_flag = bond_context.get("is_gas_like_small_molecule") is True or heavy_atoms <= 6
    if gas_flag:
        return "small_gas", {"access": 0.15, "reach": 0.50, "retention": 0.35}
    if heavy_atoms >= 35 or rotatable >= 12:
        return "bulky_substrate", {"access": 0.45, "reach": 0.40, "retention": 0.15}
    return "standard", {"access": 0.35, "reach": 0.45, "retention": 0.20}


def _select_scaffolds(
    scaffold_library: Optional[List[Dict[str, Any]]],
    scaffold_library_id: Optional[str],
    route: Dict[str, Any],
    scaffold_count: int,
    difficulty: str,
) -> List[Scaffold]:
    if scaffold_library:
        scaffolds = []
        for entry in scaffold_library:
            if not isinstance(entry, dict):
                continue
            scaffold_id = entry.get("scaffold_id") or entry.get("id")
            pdb_path = entry.get("pdb_path") or entry.get("path")
            if scaffold_id and pdb_path:
                scaffolds.append(Scaffold(scaffold_id=scaffold_id, pdb_path=pdb_path))
        return scaffolds[:scaffold_count]

    library_id = scaffold_library_id or "scaffold_lib_generic_v1"
    tracks = [track.get("track") for track in route.get("expert_tracks") or [] if track.get("track")]
    if difficulty == "HARD" and tracks:
        return _generate_scaffolds_with_tracks(library_id, tracks, scaffold_count)
    return _generate_scaffolds(library_id, scaffold_count)


def _generate_scaffolds(library_id: str, count: int) -> List[Scaffold]:
    scaffolds = []
    for idx in range(count):
        scaffold_id = f"{library_id}_scaffold_{idx + 1:03d}"
        pdb_path = f"scaffolds/{library_id}/{scaffold_id}.pdb"
        scaffolds.append(Scaffold(scaffold_id=scaffold_id, pdb_path=pdb_path))
    return scaffolds


def _generate_scaffolds_with_tracks(
    library_id: str,
    tracks: List[str],
    count: int,
) -> List[Scaffold]:
    scaffolds: List[Scaffold] = []
    track_count = max(1, min(len(tracks), 3))
    per_track = max(1, math.ceil(count / track_count))
    for track in tracks[:track_count]:
        for idx in range(per_track):
            if len(scaffolds) >= count:
                break
            scaffold_id = f"{library_id}_{track}_{idx + 1:03d}"
            pdb_path = f"scaffolds/{library_id}/{track}/{scaffold_id}.pdb"
            scaffolds.append(Scaffold(scaffold_id=scaffold_id, pdb_path=pdb_path, track=track))
    return scaffolds


def _cache_key(
    scaffold_id: str,
    smiles: str,
    target_role: str,
    mode: str,
    strictness: str,
) -> str:
    substrate_hash = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    raw = f"{scaffold_id}|{substrate_hash}|{target_role}|{mode}|{strictness}|{MODULE1_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)
    tmp_path.replace(CACHE_PATH)


def _initialize_scaffold_state(scaffold: Scaffold) -> Dict[str, Any]:
    return {
        "scaffold_id": scaffold.scaffold_id,
        "pdb_path": scaffold.pdb_path,
    }


def _detect_pocket_candidates(scaffold_state: Dict[str, Any]) -> List[List[float]]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "pocket")
    if rng.random() < 0.08:
        return []
    count = 1 + int(rng.random() * 2)
    centers = []
    for _ in range(count):
        centers.append(
            [
                round(rng.uniform(-5.0, 5.0), 3),
                round(rng.uniform(-5.0, 5.0), 3),
                round(rng.uniform(-5.0, 5.0), 3),
            ]
        )
    return centers


def _topogate_coarse(
    scaffold_state: Dict[str, Any],
    mode: str,
    strictness: str,
    structure_summary: Dict[str, Any],
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "topogate_coarse")
    base_access = {"small_gas": 0.85, "standard": 0.65, "bulky_substrate": 0.45}[mode]
    strict_penalty = {"lenient": 0.0, "standard": 0.05, "strict": 0.1}.get(strictness, 0.05)
    access_score = max(0.0, min(1.0, base_access - strict_penalty + rng.uniform(-0.1, 0.1)))
    pass_gate = rng.random() < access_score
    bottleneck = max(0.4, 1.0 + rng.uniform(-0.3, 0.5))
    path_length = max(5.0, 15.0 + rng.uniform(-5.0, 12.0))
    entry_point = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(3)]
    tunnel_summary = {
        "bottleneck_radius": round(bottleneck, 3),
        "path_length": round(path_length, 3),
        "entry_point": entry_point,
        "curvature_proxy": round(rng.uniform(0.1, 0.6), 3),
    }
    return {
        "pass": pass_gate,
        "access_score": access_score,
        "tunnel_summary": tunnel_summary,
        "pocket_center": scaffold_state["pocket_center_candidates"][0],
    }


def _topogate_refined(
    scaffold_state: Dict[str, Any],
    mode: str,
    strictness: str,
    structure_summary: Dict[str, Any],
    size_proxies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "topogate_refined")
    probe_radius = _probe_radius(structure_summary, mode, size_proxies=size_proxies)
    clearance_margin = _clearance_margin(strictness)
    required_bottleneck = probe_radius + clearance_margin
    bottleneck = max(0.3, probe_radius * rng.uniform(0.7, 1.6))
    path_length = max(6.0, 12.0 + rng.uniform(-4.0, 10.0))
    curvature = round(rng.uniform(0.1, 0.7), 3)
    entry_point = [round(rng.uniform(-8.0, 8.0), 3) for _ in range(3)]
    access_score = min(1.0, bottleneck / max(0.2, probe_radius))
    access_score *= max(0.4, 1.0 - (path_length / 40.0))
    access_score = max(0.0, min(1.0, access_score))

    tunnel_summary = {
        "bottleneck_radius": round(bottleneck, 3),
        "required_bottleneck_radius": round(required_bottleneck, 3),
        "path_length": round(path_length, 3),
        "curvature_proxy": curvature,
        "entry_point": entry_point,
    }

    if bottleneck < required_bottleneck:
        shortfall = required_bottleneck - bottleneck
        if shortfall <= 0.1:
            tunnel_summary["borderline_clearance"] = True
            return {
                "access_score": round(access_score * 0.9, 3),
                "pocket_center": scaffold_state["pocket_center_candidates"][0],
                "tunnel_summary": tunnel_summary,
                "warn_code": WARN_TUNNEL_BORDERLINE,
            }
        return {
            "fail_code": FAIL_TUNNEL_TOO_NARROW,
            "access_score": access_score,
            "pocket_center": scaffold_state["pocket_center_candidates"][0],
            "tunnel_summary": tunnel_summary,
        }

    return {
        "access_score": round(access_score, 3),
        "pocket_center": scaffold_state["pocket_center_candidates"][0],
        "tunnel_summary": tunnel_summary,
    }


def _flexibility_check(scaffold_state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "flexibility")
    open_fraction = rng.uniform(0.2, 0.9)
    threshold = 0.3 if mode != "small_gas" else 0.2
    if open_fraction < threshold:
        return {"fail_code": FAIL_TUNNEL_COLLAPSE, "open_fraction": round(open_fraction, 3)}
    return {"fail_code": None, "open_fraction": round(open_fraction, 3)}


def _build_attack_envelope(
    scaffold_state: Dict[str, Any],
    route: Dict[str, Any],
    strictness: str,
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "attack_envelope")
    pocket_center = scaffold_state["pocket_center_candidates"][0]
    axis = _normalize_vector([rng.uniform(-1.0, 1.0) for _ in range(3)])
    apex = [round(pocket_center[i] + axis[i] * 1.5, 3) for i in range(3)]
    primary = route.get("primary") or "unknown"
    if primary in {"serine_hydrolase", "amidase", "hydrolase"}:
        distance_band = [2.5, 3.5]
    else:
        distance_band = [3.0, 5.0]
    cone_angle = 15 if strictness == "strict" else 30
    return {
        "apex_point": apex,
        "axis_vector": [round(val, 3) for val in axis],
        "distance_band": distance_band,
        "cone_angle": cone_angle,
        "mechanism_family": primary,
    }


def _reach_gate(
    scaffold_state: Dict[str, Any],
    attack_envelope: Dict[str, Any],
    route: Dict[str, Any],
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "reach_gate")
    candidates = scaffold_state.get("candidate_residues") or []
    if not candidates:
        candidates = _seed_reach_candidates(scaffold_state)

    reach_candidates = []
    for residue in candidates:
        reach_score = max(0.0, min(1.0, rng.uniform(0.3, 0.95)))
        reach_candidates.append(
            {
                "residue": residue,
                "reach_score": round(reach_score, 3),
            }
        )

    if not reach_candidates:
        return {"fail_code": FAIL_REACH_FAIL, "reach_score": 0.0, "reachable_residues": []}

    reach_candidates.sort(key=lambda item: item["reach_score"], reverse=True)
    best = reach_candidates[:3]
    reach_score = sum(item["reach_score"] for item in best) / len(best)
    residue_names = [item["residue"] for item in reach_candidates]
    residues_by_role = _categorize_residues(residue_names)
    mechanism_compat = _mechanism_compatibility(route, residues_by_role)
    reach_geom_score = reach_score
    reach_score *= mechanism_compat["score"]

    fail_code = None
    primary = route.get("primary")
    if primary == "serine_hydrolase":
        if not (
            mechanism_compat["has_nucleophile"]
            and mechanism_compat["has_base"]
            and mechanism_compat["has_acid"]
        ):
            fail_code = FAIL_MECH_COMPAT
    return {
        "fail_code": fail_code,
        "reach_score": round(reach_score, 3),
        "reach_geom_score": round(reach_geom_score, 3),
        "reachable_residues": reach_candidates,
        "best_candidates": best,
        "candidate_residues_by_role": residues_by_role,
        "mechanism_compat": mechanism_compat,
    }


def _retention_gate(
    scaffold_state: Dict[str, Any],
    structure_summary: Dict[str, Any],
    route: Dict[str, Any],
    mode: str,
    size_proxies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "retention")
    heavy_atoms = structure_summary.get("heavy_atoms") or 1
    hetero_atoms = structure_summary.get("hetero_atoms") or 0
    ring_count = structure_summary.get("ring_count") or 0
    substrate_volume = max(50.0, heavy_atoms * 18.0)
    if size_proxies:
        proxy_volume = size_proxies.get("approx_volume")
        if isinstance(proxy_volume, (int, float)) and proxy_volume > 0:
            substrate_volume = float(proxy_volume)
        proxy_radius = size_proxies.get("approx_radius")
        if (
            proxy_volume is None
            and isinstance(proxy_radius, (int, float))
            and proxy_radius > 0
        ):
            substrate_volume = (4.0 / 3.0) * math.pi * (float(proxy_radius) ** 3)
    pocket_volume = max(120.0, 150.0 + rng.uniform(-40.0, 250.0))
    volume_ratio = pocket_volume / substrate_volume
    anchor_score = min(1.0, 0.1 + hetero_atoms * 0.05 + ring_count * 0.04)
    anchor_score = max(0.0, min(1.0, anchor_score))
    volume_penalty = 0.0
    if volume_ratio > 0:
        volume_penalty = _clamp01(
            1.0 - (math.log10(volume_ratio) / math.log10(100.0))
        )
    retention_score = (0.6 * anchor_score) + (0.4 * volume_penalty)

    risk_flag = "LOW"
    if volume_ratio >= 15:
        risk_flag = "MEDIUM"
    if volume_ratio >= 30 and anchor_score < 0.5:
        risk_flag = "HIGH"

    fail_code = None
    requires_alignment = route.get("primary") in {"serine_hydrolase", "amidase", "hydrolase"}
    if mode == "small_gas" and volume_ratio > 20 and anchor_score < 0.15 and requires_alignment:
        fail_code = FAIL_RETENTION_IMPOSSIBLE
    warning_codes = []
    score_multiplier = 1.0
    if volume_ratio >= 40 and anchor_score < 0.45:
        warning_codes.append(WARN_RETENTION_WEAK_BINDING)
        score_multiplier = 0.85

    return {
        "fail_code": fail_code,
        "retention_score": round(max(0.0, min(1.0, retention_score)), 3),
        "retention_risk_flag": risk_flag,
        "pocket_volume_proxy": round(pocket_volume, 2),
        "substrate_volume_proxy": round(substrate_volume, 2),
        "volume_ratio": round(volume_ratio, 2),
        "volume_penalty": round(volume_penalty, 3),
        "anchor_score": round(anchor_score, 3),
        "warning_codes": warning_codes,
        "score_multiplier": score_multiplier,
    }


def _seed_reach_candidates(scaffold_state: Dict[str, Any]) -> List[str]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "reach_seed")
    residues = ["Ser", "His", "Asp", "Glu", "Cys", "Lys", "Tyr", "Met"]
    count = max(1, int(rng.uniform(2, 6)))
    candidates = []
    for _ in range(count):
        residue = rng.choice(residues)
        idx = rng.randint(20, 220)
        candidates.append(f"{residue}{idx}")
    return candidates


def _categorize_residues(residues: List[str]) -> Dict[str, List[str]]:
    roles = {"nucleophile": [], "base": [], "acid": [], "other": []}
    nucleophiles = {"SER", "THR", "CYS"}
    bases = {"HIS"}
    acids = {"ASP", "GLU"}
    for residue in residues:
        match = re.match(r"[A-Za-z]+", residue or "")
        token = match.group(0).upper() if match else ""
        if token in nucleophiles:
            roles["nucleophile"].append(residue)
        elif token in bases:
            roles["base"].append(residue)
        elif token in acids:
            roles["acid"].append(residue)
        else:
            roles["other"].append(residue)
    return roles


def _extract_residue_token(residue: str) -> str:
    match = re.match(r"[A-Za-z]+", residue or "")
    return match.group(0).upper() if match else ""


def _mechanism_compatibility(
    route: Dict[str, Any],
    residues_by_role: Dict[str, List[str]],
) -> Dict[str, Any]:
    primary = route.get("primary") or "unknown"
    mechanism_label = primary
    required_flags: List[str] = []
    nucleophile_res = residues_by_role.get("nucleophile", [])
    base_res = residues_by_role.get("base", [])
    acid_res = residues_by_role.get("acid", [])
    nucleophile_tokens = {_extract_residue_token(res) for res in nucleophile_res}
    has_nucleophile = bool(nucleophile_res)
    has_base = bool(base_res)
    has_acid = bool(acid_res)
    nucleophile_type = "None"
    if nucleophile_tokens.intersection({"SER", "THR"}):
        nucleophile_type = "Ser/Thr"
    elif "CYS" in nucleophile_tokens:
        nucleophile_type = "Cys"
    elif has_nucleophile:
        nucleophile_type = "Other"

    notes: List[str] = []
    score = 0.0
    if primary == "serine_hydrolase":
        nucleophile_score = 1.0 if nucleophile_type == "Ser/Thr" else 0.7 if nucleophile_type == "Cys" else 0.0
        base_score = 1.0 if has_base else 0.0
        acid_score = 1.0 if has_acid else 0.0
        score = (nucleophile_score + base_score + acid_score) / 3.0
        if nucleophile_type == "Cys":
            notes.append("Cys nucleophile; serine-hydrolase track downgraded")
            score = min(score, 0.8)
            mechanism_label = "thiol_hydrolase_like"
            required_flags.append("requires_thiol_nucleophile")
        if not has_nucleophile:
            notes.append("Missing nucleophile residue")
        if not has_base:
            notes.append("Missing His base")
        if not has_acid:
            notes.append("Missing Asp/Glu acid")
    elif primary == "metallo_esterase":
        base_score = 1.0 if has_base else 0.0
        acid_score = 1.0 if has_acid else 0.0
        nucleophile_score = 0.2 if has_nucleophile else 0.1
        score = (0.4 * base_score) + (0.4 * acid_score) + (0.2 * nucleophile_score)
        metal_hint = len(acid_res) >= 2 and len(base_res) >= 1
        if metal_hint:
            score = min(1.0, score + 0.1)
            notes.append("Metal-binding hint present")
        if not has_base:
            notes.append("Missing His base")
        if not has_acid:
            notes.append("Missing acidic residue")
    else:
        score = (
            float(has_nucleophile) + float(has_base) + float(has_acid)
        ) / 3.0
        if not has_nucleophile:
            notes.append("Missing nucleophile residue")
        if not has_base:
            notes.append("Missing base residue")
        if not has_acid:
            notes.append("Missing acid residue")

    return {
        "track": primary,
        "mechanism_label": mechanism_label,
        "required_flags": required_flags,
        "has_nucleophile": has_nucleophile,
        "nucleophile_type": nucleophile_type,
        "has_base": has_base,
        "has_acid": has_acid,
        "score": round(score, 3),
        "notes": notes,
    }


def _probe_radius(
    structure_summary: Dict[str, Any],
    mode: str,
    size_proxies: Optional[Dict[str, Any]] = None,
) -> float:
    if size_proxies:
        min_diameter = size_proxies.get("min_diameter_proxy")
        if isinstance(min_diameter, (int, float)) and min_diameter > 0:
            base = max(1.4, float(min_diameter) / 2.0)
            return max(0.6, min(2.8, base))
    heavy_atoms = structure_summary.get("heavy_atoms") or 1
    base = 0.7 + heavy_atoms * 0.02
    if mode == "bulky_substrate":
        base += 0.4
    return max(0.6, min(2.8, base))


def _valid_weights(weights: Any) -> bool:
    if not isinstance(weights, dict):
        return False
    required = {"access", "reach", "retention"}
    if not required.issubset(weights.keys()):
        return False
    return all(isinstance(weights[key], (int, float)) for key in required)


def _normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(val * val for val in vector)) or 1.0
    return [val / norm for val in vector]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clearance_margin(strictness: str) -> float:
    return {"lenient": 0.2, "standard": 0.4, "strict": 0.6}.get(strictness, 0.4)


def _nucleophile_geometry(mechanism_compat: Dict[str, Any]) -> str:
    nucleophile_type = (mechanism_compat or {}).get("nucleophile_type")
    if nucleophile_type == "Ser/Thr":
        return "serine_oxyanion"
    if nucleophile_type == "Cys":
        return "cysteine_thiol"
    if nucleophile_type in {"None", None}:
        return "none"
    return "generic_nucleophile"


def _access_confidence(tunnel_summary: Optional[Dict[str, Any]], access_score: float) -> float:
    if not tunnel_summary:
        return round(_clamp01(access_score), 3)
    bottleneck = tunnel_summary.get("bottleneck_radius")
    required = tunnel_summary.get("required_bottleneck_radius")
    path_length = tunnel_summary.get("path_length")
    curvature = tunnel_summary.get("curvature_proxy")
    if not isinstance(bottleneck, (int, float)) or not isinstance(required, (int, float)):
        return round(_clamp01(access_score), 3)
    clearance = bottleneck - required
    clearance_score = _clamp01(0.5 + (clearance / max(0.5, required)))
    path_score = _clamp01(1.0 - (float(path_length) / 40.0)) if isinstance(path_length, (int, float)) else 0.5
    curvature_score = _clamp01(1.0 - float(curvature)) if isinstance(curvature, (int, float)) else 0.5
    access_conf = (0.5 * clearance_score) + (0.3 * path_score) + (0.2 * curvature_score)
    return round(_clamp01(access_conf), 3)


def _rng_for_step(scaffold_id: str, step: str) -> random.Random:
    seed = f"{scaffold_id}:{step}"
    value = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    return random.Random(value)


def _top_k_from_difficulty(difficulty: str, scaffold_count: int) -> int:
    if difficulty == "EASY":
        return min(5, scaffold_count)
    if difficulty == "MEDIUM":
        return min(10, scaffold_count)
    if scaffold_count >= 200:
        return min(25, scaffold_count)
    return min(15, scaffold_count)


def _reject_scaffold(
    scaffold: Scaffold,
    fail_codes: List[str],
    state: Dict[str, Any],
    tunnel_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "scaffold_id": scaffold.scaffold_id,
        "pdb_path": scaffold.pdb_path,
        "pocket_center": (state.get("pocket_center_candidates") or [None])[0],
        "tunnel_summary": tunnel_summary,
        "reach_summary": {},
        "attack_envelope": {},
        "retention_metrics": {},
        "scores": {},
        "fail_codes": fail_codes,
    }


def _fail_handoff(
    halt_reason: str,
    rejected: Optional[List[Dict[str, Any]]] = None,
    cache_hits: int = 0,
    cache_misses: int = 0,
    cache_writes: int = 0,
    cache_size: int = 0,
) -> Dict[str, Any]:
    return {
        "status": "FAIL",
        "halt_reason": halt_reason,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries_written": cache_writes,
            "cache_size": cache_size,
        },
        "ranked_scaffolds": [],
        "rejected_scaffolds": rejected or [],
        "module2_handoff": {"top_scaffolds": []},
    }
