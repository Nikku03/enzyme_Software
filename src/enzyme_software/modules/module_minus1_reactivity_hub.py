from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.context import PipelineContext
from enzyme_software.modules.base import BaseModule
from enzyme_software.modules.sre_atr import (
    AtomicTruthRegistry,
    GroupRole,
    detect_groups,
    resolve_bond,
    BondResolutionResult,
)
from enzyme_software.modules.sre_fragment_builder import (
    ChemicallyAwareFragmentBuilder,
    Fragment,
)
from enzyme_software.evidence_store import add_datapoints
from enzyme_software.unity_layer import record_interlink

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

try:
    from enzyme_software.cpt.geometric_cpts import (
        EnvironmentAwareStericsCPT_Level2,
        ElectronicPropertiesAttackValidationCPT,
    )
    from enzyme_software.cpt.engine import GeometricCPTEngine
    from enzyme_software.cpt.level3_env_cpts import (
        EnvContext,
        Level3Orchestrator,
        OxyanionHoleGeometryCPT,
        SolventExposurePolarityCPT,
        TransitionStateChargeStabilizationCPT,
    )

    _CPT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    EnvironmentAwareStericsCPT_Level2 = None
    ElectronicPropertiesAttackValidationCPT = None
    GeometricCPTEngine = None
    EnvContext = None
    Level3Orchestrator = None
    OxyanionHoleGeometryCPT = None
    SolventExposurePolarityCPT = None
    TransitionStateChargeStabilizationCPT = None
    _CPT_AVAILABLE = False

try:
    from enzyme_software.calibration.layer1_empirical import (
        CALIBRATION_VERSION as L1_CALIBRATION_VERSION,
        BDE_TABLE as L1_BDE_TABLE,
        bde_record as l1_bde_record,
        bde_kj_mol as l1_bde_kj_mol,
        radical_stability_index as l1_radical_stability_index,
        all_hat_barriers_kj_mol as l1_all_hat_barriers_kj_mol,
    )
except Exception:  # pragma: no cover - optional import
    L1_CALIBRATION_VERSION = "builtin_fallback"
    L1_BDE_TABLE = {}

    def l1_bde_record(bond_class: str) -> Dict[str, Any]:
        return {}

    def l1_bde_kj_mol(bond_class: str, default: float = 410.0) -> float:
        return float(default)

    def l1_radical_stability_index(bond_class: str, default: float = 0.2) -> float:
        return float(default)

    def l1_all_hat_barriers_kj_mol(
        bond_class: str,
        bde_value_kj_mol: float,
        protein_correction_kj: float = -8.0,
        include_legacy_aliases: bool = True,
    ) -> Dict[str, float]:
        return {}

try:
    from enzyme_software.calibration.layer3_xtb import (
        CALIBRATION_VERSION as L3_CALIBRATION_VERSION,
        compute_substrate_bde_with_safeguard_for_mol as l3_compute_substrate_bde_with_safeguard_for_mol,
        is_layer3_xtb_enabled as l3_is_layer3_xtb_enabled,
    )

    _L3_XTB_AVAILABLE = True
except Exception:  # pragma: no cover - optional import
    L3_CALIBRATION_VERSION = "unavailable"

    def l3_compute_substrate_bde_with_safeguard_for_mol(
        mol: Any,
        bond_atom_indices: Tuple[int, int],
        bond_class: str,
        *,
        solvent: Optional[str] = "water",
        xtb_path: str = "xtb",
        n_cores: int = 1,
        timeout_s: int = 300,
    ) -> Dict[str, Any]:
        return {"status": "unavailable", "source": "layer3_xtb_unavailable", "bde_kj_mol": None}

    def l3_is_layer3_xtb_enabled() -> bool:
        return False

    _L3_XTB_AVAILABLE = False


# Legacy fallbacks retained for robustness when calibration module is unavailable.
BDE_TABLE_KJ_MOL: Dict[str, float] = {
    key: float(value.get("bde_kj_mol"))
    for key, value in (L1_BDE_TABLE or {}).items()
    if isinstance(value, dict) and isinstance(value.get("bde_kj_mol"), (int, float))
}
if not BDE_TABLE_KJ_MOL:
    BDE_TABLE_KJ_MOL = {
        "ch__aliphatic": 410.0,
        "ch__primary": 423.0,
        "ch__secondary": 412.5,
        "ch__tertiary": 403.8,
        "ch__benzylic": 375.5,
        "ch__allylic": 371.5,
        "ch__alpha_hetero": 385.0,
        "ch__fluorinated": 446.4,
        "ch__aryl": 472.2,
        "nh__amine": 386.0,
        "nh__amide": 440.0,
        "oh__alcohol": 435.7,
        "oh__phenol": 362.8,
    }

# Optional neighbor corrections only for unresolved generic classes.
ALPHA_CORRECTION_KJ_MOL: Dict[str, float] = {
    "F": 12.0,
    "Cl": 4.0,
    "O": -8.0,
    "N": -5.0,
}

EVANS_POLANYI_PARAMS: Dict[str, Dict[str, float]] = {
    "Fe_IV_oxo": {"alpha": 0.495, "beta": -139.7},
    "Fe_IV_oxo_heme": {"alpha": 0.495, "beta": -139.7},
    "radical_SAM": {"alpha": 0.35, "beta": -55.0},
    "non_heme_Fe": {"alpha": 0.45, "beta": -125.0},
    "Fe_IV_oxo_nonheme": {"alpha": 0.45, "beta": -125.0},
    "generic_radical": {"alpha": 0.40, "beta": -45.0},
}

CH_AUTOPICK_MIN_DELTA_BDE_KJ = 5.0
SN2_LEAVING_GROUP_SCORE: Dict[str, float] = {
    "F": 0.35,
    "Cl": 0.60,
    "Br": 0.80,
    "I": 1.00,
}
_SYNTHETIC_H_PATTERN = re.compile(r"^.+_H(\d+)$")


@dataclass
class TargetSpec:
    kind: str
    indices: Optional[Tuple[int, int]]
    token: Optional[str]


@dataclass
class ModuleMinus1OutputSchema:
    schema_version: str
    group_type: Optional[str]
    roles_resolved: Dict[str, Optional[int]]
    candidate_attack_sites: List[Dict[str, Any]]
    competition: Dict[str, Any]
    constraint_flags: List[str]
    confidence: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "group_type": self.group_type,
            "roles_resolved": self.roles_resolved,
            "candidate_attack_sites": self.candidate_attack_sites,
            "competition": self.competition,
            "constraint_flags": self.constraint_flags,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


class ModuleMinus1SRE(BaseModule):
    name = "Module -1 - Reactivity Hub"
    module_id = -1

    def run(self, ctx: PipelineContext) -> PipelineContext:
        run_id = _get_run_id(ctx)
        result = run_module_minus1_reactivity_hub(
            smiles=ctx.smiles,
            target_bond=ctx.target_bond,
            requested_output=ctx.requested_output,
            constraints=ctx.constraints.to_dict(),
        )
        ctx.data["module_minus1"] = result
        # Backward-compatible alias used by older integration tests/consumers.
        ctx.data["module_minus1_sre"] = result
        before_shared = json.dumps(ctx.data.get("shared_io") or {}, sort_keys=True, default=str)
        ctx.data["shared_io"] = _merge_shared_io(ctx, result, run_id=run_id)
        after_shared = json.dumps(ctx.data.get("shared_io") or {}, sort_keys=True, default=str)
        print(
            f"[run_id={run_id}] [module-1] token={ctx.target_bond} "
            f"canonical={result.get('resolved_target', {}).get('canonical_token')} "
            f"match_count={result.get('resolved_target', {}).get('match_count')}"
        )
        if before_shared != after_shared:
            print(f"[run_id={run_id}] [module-1] shared_io diff: updated")
        _write_module_minus1_datapoint(ctx, result, run_id=run_id)
        record_interlink(
            ctx,
            -1,
            reads=[
                "input_spec.smiles",
                "input_spec.target_bond",
                "constraints.condition_profile",
            ],
            writes=["chem.sre", "chem.context"],
        )
        return ctx


def run_module_minus1_reactivity_hub(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []

    if Chem is None:
        return _build_output(
            status="FAIL",
            bond360_profile={},
            fragment={},
            cpt_scores={},
            mechanism_eligibility={},
            primary_constraint="NONE",
            confidence_prior=0.0,
            route_bias={},
            resolved_target={
                "requested": target_bond,
                "selection_mode": "token",
                "match_count": 0,
                "candidate_bonds": [],
                "next_input_required": ["target_bond"],
                "resolution_source": "module_minus1",
            },
            reactivity={},
            cache_key=None,
            cache_hit=False,
            warnings=["RDKit unavailable; Module -1 cannot run."],
            errors=["rdkit_unavailable"],
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _build_output(
            status="FAIL",
            bond360_profile={},
            fragment={},
            cpt_scores={},
            mechanism_eligibility={},
            primary_constraint="NONE",
            confidence_prior=0.0,
            route_bias={},
            resolved_target={
                "requested": target_bond,
                "selection_mode": "token",
                "match_count": 0,
                "candidate_bonds": [],
                "next_input_required": ["target_bond"],
                "resolution_source": "module_minus1",
            },
            reactivity={},
            cache_key=None,
            cache_hit=False,
            warnings=[],
            errors=["SMILES parse failed."],
        )

    atr = AtomicTruthRegistry.from_smiles(smiles)
    detection = detect_groups(mol, atr=atr)
    warnings.extend(detection.warnings)

    target_spec = _parse_target_bond(target_bond)
    bond_resolution, target_group_type = _resolve_target_bond(
        mol, atr, detection.groups, target_spec
    )
    warnings.extend(bond_resolution.warnings)
    errors.extend(bond_resolution.errors)

    selected_group = _select_group_for_bond(
        detection.groups, bond_resolution, target_group_type
    )

    fragment = None
    if selected_group is not None:
        fragment = _build_fragment_from_group(atr, mol, selected_group, warnings, errors)
    elif bond_resolution.selected is not None:
        fragment = _build_fragment_from_bond(
            atr, mol, bond_resolution.selected.atom_ids, warnings, errors
        )

    bond_type = _bond_type_from_group(selected_group, target_group_type)
    bond360_profile = _bond360_from_group(selected_group, bond_resolution, bond_type)
    resolution_meta = _resolution_metadata_from_result(bond_resolution)

    if bond_type in {"ch", "nh", "oh"}:
        if bond_resolution.selected is None:
            cpt_scores = {
                "status": "no_target_selection",
                "track": "radical_hat",
                "message": "No unique X-H target selected; explicit atom/bond selection required.",
            }
        else:
            bond_class = _classify_xh_bond(mol, atr, bond_resolution, bond_type)
            cpt_scores = _run_radical_cpts(
                mol=mol,
                bond_type=bond_type,
                bond_class=bond_class,
                atr=atr,
                bond_resolution=bond_resolution,
                warnings=warnings,
            )
    elif bond_type == "alkyl_halide":
        cpt_scores = _run_displacement_sn2_cpts(
            mol=mol,
            atr=atr,
            bond_resolution=bond_resolution,
            warnings=warnings,
        )
    else:
        cpt_scores = _run_cpts(fragment, selected_group, warnings)

    mechanism_eligibility = _mechanism_eligibility(bond_type)
    confidence_prior = _confidence_prior_from_bond(bond_type)
    if isinstance(cpt_scores, dict) and str(cpt_scores.get("track") or "").lower() == "radical_hat":
        comp = cpt_scores.get("composite_score")
        if isinstance(comp, (int, float)):
            confidence_prior = max(0.05, min(float(confidence_prior), float(comp)))
    route_bias = _route_bias_from_bond(bond_type)
    primary_constraint = _derive_primary_constraint(cpt_scores)

    candidate_attack_sites = _candidate_attack_sites(bond_resolution, atr)
    requires_explicit_selection = bool(resolution_meta.get("equivalent_sites_detected"))
    resolved_target = {
        "bond_indices": bond360_profile.get("bond_indices", []),
        "bond_type": bond_type,
        "group_type": target_group_type,
        "attack_sites": bond360_profile.get("attack_sites", {}),
        "selection_mode": target_spec.kind,
        "requested": target_bond,
        "canonical_token": (target_spec.token or "").lower() if target_spec.token else None,
        "candidate_bonds": candidate_attack_sites,
        "match_count": len(candidate_attack_sites),
        "next_input_required": (
            ["target_bond_selection"]
            if requires_explicit_selection
            else ["target_bond"]
            if len(candidate_attack_sites) == 0
            else []
        ),
        "resolution_policy": resolution_meta.get("resolution_policy"),
        "resolution_note": resolution_meta.get("resolution_note"),
        "resolution_confidence": resolution_meta.get("resolution_confidence"),
        "bde_gap_kj_mol": resolution_meta.get("bde_gap_kj_mol"),
        "equivalent_sites_detected": resolution_meta.get("equivalent_sites_detected"),
        "resolution_source": "module_minus1",
    }
    if (
        not resolved_target["bond_indices"]
        and candidate_attack_sites
        and isinstance(candidate_attack_sites[0].get("atom_indices"), list)
    ):
        resolved_target["bond_indices"] = candidate_attack_sites[0].get("atom_indices")

    epav_data = cpt_scores.get("epav", {}).get("data", {}) if isinstance(cpt_scores, dict) else {}
    competition = epav_data.get("competition") if isinstance(epav_data, dict) else {}
    if not isinstance(competition, dict) or competition.get("gap") is None:
        competition = _competition_from_candidates(candidate_attack_sites, bond_type)
    equivalent_sites = bool(competition.get("equivalent_sites_detected"))
    if equivalent_sites and "target_bond_selection" not in resolved_target["next_input_required"]:
        resolved_target["next_input_required"] = ["target_bond_selection"]

    constraint_flags = []
    if primary_constraint and primary_constraint != "NONE":
        constraint_flags.append(primary_constraint)
    if requested_output and str(requested_output).strip().startswith("-"):
        constraint_flags.append("FRAGMENT")
    if not candidate_attack_sites:
        constraint_flags.append("NO_MATCH")

    module_minus1_schema = ModuleMinus1OutputSchema(
        schema_version="module_minus1.v1",
        group_type=target_group_type,
        roles_resolved=bond360_profile.get("attack_sites", {}),
        candidate_attack_sites=candidate_attack_sites,
        competition=competition,
        constraint_flags=constraint_flags,
        confidence=float(confidence_prior),
        reasons=(warnings + errors)[:12],
    )

    reactivity = {
        "epav_score": cpt_scores.get("epav", {}).get("score"),
        "epav_passed": cpt_scores.get("epav", {}).get("passed"),
        "level2_score": cpt_scores.get("level2", {}).get("score"),
        "level3_score": cpt_scores.get("level3", {}).get("score"),
        "competition": competition,
        "equivalent_sites_detected": equivalent_sites,
        "primary_constraint": primary_constraint,
        "confidence_prior": confidence_prior,
    }

    return _build_output(
        status="PASS" if not errors else "FAIL",
        bond360_profile=bond360_profile,
        fragment=_fragment_payload(fragment),
        cpt_scores=cpt_scores,
        mechanism_eligibility=mechanism_eligibility,
        primary_constraint=primary_constraint,
        confidence_prior=confidence_prior,
        route_bias=route_bias,
        resolved_target=resolved_target,
        reactivity=reactivity,
        module_minus1_schema=module_minus1_schema.to_dict(),
        cache_key=None,
        cache_hit=False,
        warnings=warnings,
        errors=errors,
    )


def _parse_target_bond(target_bond: str) -> TargetSpec:
    raw = (target_bond or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return TargetSpec(kind="indices", indices=(int(parts[0]), int(parts[1])), token=None)
    if "," in raw and all(part.strip().isdigit() for part in raw.split(",", 1)):
        parts = [p.strip() for p in raw.split(",", 1)]
        return TargetSpec(kind="indices", indices=(int(parts[0]), int(parts[1])), token=None)
    if "-" in raw and all(part.strip().isdigit() for part in raw.split("-", 1)):
        parts = [p.strip() for p in raw.split("-", 1)]
        return TargetSpec(kind="indices", indices=(int(parts[0]), int(parts[1])), token=None)
    return TargetSpec(kind="token", indices=None, token=raw or None)


def classify_ch_subtype(mol: Chem.Mol, atom_idx: int) -> str:
    """Classify a C-H heavy atom context for C-H resolution policy."""
    atom = mol.GetAtomWithIdx(int(atom_idx))
    if atom.GetIsAromatic():
        return "aromatic"
    if any(nbr.GetSymbol() == "F" for nbr in atom.GetNeighbors()):
        return "fluorinated"
    if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
        return "benzylic"
    for nbr in atom.GetNeighbors():
        for bond in nbr.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(nbr)
                if other.GetIdx() != atom.GetIdx() and other.GetSymbol() == "C":
                    return "allylic"
    return "aliphatic"


def _heavy_uuid_from_candidate(candidate: Any) -> Optional[str]:
    for atom_id in candidate.atom_ids:
        atom_id_s = str(atom_id)
        if "_H" not in atom_id_s:
            return atom_id_s
    return None


def _ch_bond_class_from_atom(mol: Chem.Mol, atom_idx: int) -> str:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    subtype = classify_ch_subtype(mol, int(atom_idx))
    if subtype == "aromatic":
        return "ch__aryl"
    if subtype == "fluorinated":
        f_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "F")
        if f_count >= 3:
            return "ch__trifluoromethyl"
        if f_count == 2:
            return "ch__difluoromethyl"
        if f_count == 1:
            return "ch__fluoromethyl"
        return "ch__fluorinated"
    if subtype == "benzylic":
        return "ch__benzylic"
    if subtype == "allylic":
        return "ch__allylic"
    if any(nbr.GetSymbol() in {"O", "N", "S"} for nbr in atom.GetNeighbors()):
        return "ch__alpha_hetero"

    carbon_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C")
    if carbon_neighbors >= 3:
        return "ch__tertiary"
    if carbon_neighbors == 2:
        return "ch__secondary"
    if carbon_neighbors == 1:
        return "ch__primary"
    return "ch__aliphatic"


def _should_apply_alpha_correction(bond_class: str) -> bool:
    cls = str(bond_class or "").lower()
    explicit = (
        "fluoro",
        "chloro",
        "benzylic",
        "allylic",
        "alpha_",
        "aryl",
        "vinyl",
        "oh__",
        "nh__",
    )
    return not any(token in cls for token in explicit)


def _bde_with_alpha_correction(mol: Chem.Mol, atom_idx: int, bond_class: str) -> Tuple[float, float, float]:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    bde_base = float(
        l1_bde_kj_mol(
            bond_class,
            default=BDE_TABLE_KJ_MOL.get(f"{str(bond_class).split('__', 1)[0]}__aliphatic", 410.0),
        )
    )
    alpha_corr = 0.0
    if _should_apply_alpha_correction(bond_class):
        for nbr in atom.GetNeighbors():
            alpha_corr += float(ALPHA_CORRECTION_KJ_MOL.get(nbr.GetSymbol(), 0.0))
    return bde_base + alpha_corr, bde_base, alpha_corr


def _alpha_hetero_neighbor_count(atom: Chem.Atom) -> int:
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() in {"O", "N", "S", "P"})


def _carbon_degree(atom: Chem.Atom) -> int:
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C")


def _rank_ch_candidates(
    mol: Chem.Mol,
    atr: AtomicTruthRegistry,
    candidates: List[Any],
    delta_bde_threshold_kj: float = CH_AUTOPICK_MIN_DELTA_BDE_KJ,
) -> BondResolutionResult:
    result = BondResolutionResult(method="element_pair")
    enriched: List[Any] = []
    scored: List[Dict[str, Any]] = []

    for candidate in candidates:
        heavy_uuid = _heavy_uuid_from_candidate(candidate)
        if not heavy_uuid:
            continue
        try:
            heavy_idx = int(atr.get_by_uuid(heavy_uuid).parent_index)
        except Exception:
            continue
        subtype = classify_ch_subtype(mol, heavy_idx)
        bond_class = _ch_bond_class_from_atom(mol, heavy_idx)
        bde_corr, bde_base, alpha_corr = _bde_with_alpha_correction(mol, heavy_idx, bond_class)
        radical_stability = float(l1_radical_stability_index(bond_class, default=0.20))
        heavy_atom = mol.GetAtomWithIdx(heavy_idx)
        alpha_hetero_neighbors = _alpha_hetero_neighbor_count(heavy_atom)
        carbon_degree = _carbon_degree(heavy_atom)
        evidence = dict(candidate.evidence or {})
        evidence.update(
            {
                "subclass": subtype,
                "bond_class": bond_class,
                "bde_kj_mol": round(float(bde_corr), 2),
                "bde_base_kj_mol": round(float(bde_base), 2),
                "alpha_correction_kj_mol": round(float(alpha_corr), 2),
                "radical_stability": round(float(radical_stability), 3),
                "heavy_atom_index": heavy_idx,
                "alpha_hetero_neighbors": int(alpha_hetero_neighbors),
                "carbon_degree": int(carbon_degree),
            }
        )
        enriched_candidate = type(candidate)(
            atom_ids=candidate.atom_ids,
            bond_order=candidate.bond_order,
            is_aromatic=candidate.is_aromatic,
            roles=candidate.roles,
            evidence=evidence,
        )
        enriched.append(enriched_candidate)
        scored.append(
            {
                "candidate": enriched_candidate,
                "bde_kj_mol": float(bde_corr),
                "radical_stability": float(radical_stability),
                "alpha_hetero_neighbors": int(alpha_hetero_neighbors),
                "carbon_degree": int(carbon_degree),
            }
        )

    if not scored:
        result.errors.append("no C-H candidates could be scored")
        return result

    scored.sort(
        key=lambda item: (
            item["bde_kj_mol"],
            -item["alpha_hetero_neighbors"],
            -item["carbon_degree"],
            -item["radical_stability"],
        )
    )
    ordered = [item["candidate"] for item in scored]
    for rank, item in enumerate(scored, start=1):
        ev = dict(item["candidate"].evidence or {})
        ev["rank"] = rank
        ev["score"] = round(1.0 / (1.0 + float(item["bde_kj_mol"])), 6)
        item["candidate"] = type(item["candidate"])(
            atom_ids=item["candidate"].atom_ids,
            bond_order=item["candidate"].bond_order,
            is_aromatic=item["candidate"].is_aromatic,
            roles=item["candidate"].roles,
            evidence=ev,
        )
    ordered = [item["candidate"] for item in scored]
    result.candidates = ordered

    best = scored[0]
    best_heavy = (best["candidate"].evidence or {}).get("heavy_atom_index")
    best_bde = float(best["bde_kj_mol"])
    near_best = [
        item
        for item in scored
        if float(item["bde_kj_mol"]) <= (best_bde + float(delta_bde_threshold_kj))
    ]
    near_best_heavy = {
        (item["candidate"].evidence or {}).get("heavy_atom_index")
        for item in near_best
        if (item["candidate"].evidence or {}).get("heavy_atom_index") is not None
    }
    near_best_signatures = {
        (
            (item["candidate"].evidence or {}).get("subclass"),
            (item["candidate"].evidence or {}).get("bond_class"),
        )
        for item in near_best
    }
    unique_site_equivalent = len(near_best_heavy) == 1
    class_equivalent = len(near_best_signatures) == 1

    second_site = next(
        (
            item
            for item in scored[1:]
            if (item["candidate"].evidence or {}).get("heavy_atom_index") != best_heavy
        ),
        None,
    )
    gap = None if second_site is None else float(second_site["bde_kj_mol"] - best_bde)

    if (
        second_site is None
        or unique_site_equivalent
        or class_equivalent
        or (gap is not None and gap >= float(delta_bde_threshold_kj))
    ):
        selected = best["candidate"]
        selected_evidence = dict(selected.evidence or {})
        if unique_site_equivalent or class_equivalent:
            policy = "lowest_BDE_auto_equivalent"
            note = (
                f"Auto-selected equivalent weakest C-H set (representative atom {best_heavy}); "
                f"({selected_evidence.get('subclass')}, BDE={best['bde_kj_mol']:.2f} kJ/mol)."
            )
            conf = 0.95
        else:
            policy = "lowest_BDE_auto"
            note = (
                f"Auto-selected weakest C-H ({selected_evidence.get('subclass')}, "
                f"BDE={best['bde_kj_mol']:.2f} kJ/mol)"
            )
            conf = 0.92 if second_site is not None else 0.98
        selected_evidence.update(
            {
                "resolution_policy": policy,
                "resolution_note": note,
                "resolution_confidence": round(float(conf), 3),
                "bde_gap_kj_mol": round(float(gap), 2) if gap is not None else None,
                "equivalent_sites_detected": False,
            }
        )
        result.selected = type(selected)(
            atom_ids=selected.atom_ids,
            bond_order=selected.bond_order,
            is_aromatic=selected.is_aromatic,
            roles=selected.roles,
            evidence=selected_evidence,
        )
        if result.candidates:
            result.candidates[0] = result.selected
        return result

    # Ambiguous weakest-site selection: require explicit bond disambiguation.
    selected = best["candidate"]
    selected_evidence = dict(selected.evidence or {})
    selected_evidence.update(
        {
            "resolution_policy": "ambiguous_BDE_window",
            "resolution_note": (
                f"Top C-H candidates within {gap:.2f} kJ/mol; requires explicit selection."
            ),
            "resolution_confidence": 0.5,
            "bde_gap_kj_mol": round(float(gap), 2),
            "equivalent_sites_detected": True,
        }
    )
    result.selected = type(selected)(
        atom_ids=selected.atom_ids,
        bond_order=selected.bond_order,
        is_aromatic=selected.is_aromatic,
        roles=selected.roles,
        evidence=selected_evidence,
    )
    if result.candidates:
        result.candidates[0] = result.selected
    result.warnings.append(
        "HALT_NEED_SELECTION: multiple C-H candidates have near-equivalent BDE."
    )
    return result


def _run_displacement_sn2_cpts(
    mol: Any,
    atr: AtomicTruthRegistry,
    bond_resolution: BondResolutionResult,
    warnings: List[str],
) -> Dict[str, Any]:
    """SN2 displacement proxy track for alkyl halides (C-X)."""
    payload: Dict[str, Any] = {"status": "ok", "track": "displacement_sn2"}
    selected = bond_resolution.selected
    if selected is None:
        return {
            "status": "no_target_selection",
            "track": "displacement_sn2",
            "message": "No unique alkyl-halide target selected.",
        }

    c_idx: Optional[int] = None
    x_idx: Optional[int] = None
    x_symbol = "X"
    for atom_id in selected.atom_ids:
        try:
            idx = int(atr.get_by_uuid(str(atom_id)).parent_index)
        except Exception:
            continue
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == "C" and c_idx is None:
            c_idx = idx
        elif sym in {"F", "Cl", "Br", "I"} and x_idx is None:
            x_idx = idx
            x_symbol = sym

    if c_idx is None or x_idx is None:
        return {
            "status": "insufficient_target_atoms",
            "track": "displacement_sn2",
            "message": "Could not map C-X atoms for SN2 CPT.",
        }

    c_atom = mol.GetAtomWithIdx(c_idx)
    x_atom = mol.GetAtomWithIdx(x_idx)
    hyb = c_atom.GetHybridization()
    is_sp3 = bool(hyb == Chem.rdchem.HybridizationType.SP3)
    carbon_neighbors = sum(1 for nbr in c_atom.GetNeighbors() if nbr.GetSymbol() == "C")
    steric_score = 1.0 if carbon_neighbors <= 1 else 0.55 if carbon_neighbors == 2 else 0.20
    lg_score = float(SN2_LEAVING_GROUP_SCORE.get(x_atom.GetSymbol(), 0.3))

    qc = None
    try:
        from rdkit.Chem import rdPartialCharges

        mchg = Chem.Mol(mol)
        rdPartialCharges.ComputeGasteigerCharges(mchg)
        qc_atom = mchg.GetAtomWithIdx(c_idx)
        if qc_atom.HasProp("_GasteigerCharge"):
            qc = float(qc_atom.GetProp("_GasteigerCharge"))
    except Exception as exc:
        warnings.append(f"sn2_cpt:gasteiger_failed:{exc}")
    electrophilicity = 0.5 if qc is None else max(0.0, min(1.0, (float(qc) + 0.25) / 0.55))

    backside_access = 0.80 if is_sp3 else 0.25
    if not is_sp3:
        warnings.append("sn2_not_sp3")

    # Conservative barrier proxy; lower is better.
    barrier_kj = 95.0 - 25.0 * lg_score - 20.0 * steric_score - 12.0 * electrophilicity
    barrier_kj = max(35.0, min(130.0, float(barrier_kj)))
    score = max(0.0, min(1.0, 1.0 - (barrier_kj - 35.0) / 95.0))

    payload.update(
        {
            "intent_type": "sn2_displacement",
            "mechanism_hint": "SN2 covalent intermediate",
            "best_barrier_kj_mol": round(barrier_kj, 2),
            "composite_score": round(score, 4),
            "features": {
                "carbon_idx": c_idx,
                "halide_idx": x_idx,
                "halide": x_symbol,
                "is_sp3": is_sp3,
                "carbon_degree": carbon_neighbors,
                "steric_score": round(steric_score, 3),
                "leaving_group_score": round(lg_score, 3),
                "electrophilicity_score": round(electrophilicity, 3),
                "backside_access_score": round(backside_access, 3),
                "gasteiger_q_carbon": qc,
            },
            "feasibility_assessment": (
                "feasible" if score >= 0.6 else "marginal" if score >= 0.35 else "difficult"
            ),
        }
    )
    return payload


def _resolve_target_bond(
    mol: Chem.Mol,
    atr: AtomicTruthRegistry,
    groups: List[Any],
    target_spec: TargetSpec,
) -> Tuple[BondResolutionResult, Optional[str]]:
    if target_spec.kind == "indices" and target_spec.indices:
        return resolve_bond(mol, atr=atr, target_indices=target_spec.indices), None

    token = (target_spec.token or "").lower()
    group_type, role_pair = _token_to_role_pair(token)
    if role_pair is not None:
        return (
            resolve_bond(
                mol,
                atr=atr,
                target_role_pair=role_pair,
                target_group_type=group_type,
            ),
            group_type,
        )

    # Handle alkyl halides via element pair matching
    if group_type == "alkyl_halide":
        halogen_map = {"c-f": "F", "c-cl": "Cl", "c-br": "Br", "c-i": "I"}
        token_norm = token.replace("_", "-").lower()
        halogen = halogen_map.get(token_norm)
        if halogen:
            return (
                resolve_bond(mol, atr=atr, target_element_pair=("C", halogen)),
                "alkyl_halide",
            )
        # Generic alkyl halide - try all halogens
        for hal in ["F", "Cl", "Br", "I"]:
            result = resolve_bond(mol, atr=atr, target_element_pair=("C", hal))
            if result.selected is not None:
                return result, "alkyl_halide"
        return resolve_bond(mol, atr=atr, target_indices=None), "alkyl_halide"

    # Handle X-H bonds via element pair (requires explicit H addition)
    if group_type == "ch":
        base = resolve_bond(mol, atr=atr, target_element_pair=("C", "H"))
        if len(base.candidates) > 1:
            ranked = _rank_ch_candidates(
                mol=mol,
                atr=atr,
                candidates=base.candidates,
                delta_bde_threshold_kj=CH_AUTOPICK_MIN_DELTA_BDE_KJ,
            )
            ranked.warnings = list(dict.fromkeys((base.warnings or []) + (ranked.warnings or [])))
            ranked.errors = list(dict.fromkeys((base.errors or []) + (ranked.errors or [])))
            return ranked, "ch"
        return base, "ch"
    if group_type == "nh":
        return resolve_bond(mol, atr=atr, target_element_pair=("N", "H")), "nh"
    if group_type == "oh":
        return resolve_bond(mol, atr=atr, target_element_pair=("O", "H")), "oh"

    # fallback: if token matches a group type, try to resolve with default roles
    if token in {"ester", "amide"}:
        role_pair = (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)
        return (
            resolve_bond(
                mol,
                atr=atr,
                target_role_pair=role_pair,
                target_group_type=token,
            ),
            token,
        )
    if token in {"aryl_halide", "aryl_c-br", "aryl_c-cl"}:
        role_pair = (GroupRole.ARYL_C, GroupRole.HALOGEN)
        return (
            resolve_bond(
                mol,
                atr=atr,
                target_role_pair=role_pair,
                target_group_type="aryl_halide",
            ),
            "aryl_halide",
        )

    return resolve_bond(mol, atr=atr, target_indices=None), None


def _token_to_role_pair(token: str) -> Tuple[Optional[str], Optional[Tuple[GroupRole, GroupRole]]]:
    _norm = lambda s: s.replace(" ", "").replace("_", "-").lower()
    token = _norm(token)
    # Keys are written in readable form and normalized at build time so
    # that any variant (underscores, hyphens, double-underscores) will match.
    _raw_mapping = {
        "ester_c-o": ("ester", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "ester-c-o": ("ester", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "acetyl_ester_c-o": ("ester", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "acetyl-ester-c-o": ("ester", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "ester__acyl_o": ("ester", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "ester--acyl-o": ("ester", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "amide_c-n": ("amide", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "amide-c-n": ("amide", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "amide__c_n": ("amide", (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH)),
        "aryl_c-br": ("aryl_halide", (GroupRole.ARYL_C, GroupRole.HALOGEN)),
        "aryl-c-br": ("aryl_halide", (GroupRole.ARYL_C, GroupRole.HALOGEN)),
        "aryl_c-cl": ("aryl_halide", (GroupRole.ARYL_C, GroupRole.HALOGEN)),
        "aryl-c-cl": ("aryl_halide", (GroupRole.ARYL_C, GroupRole.HALOGEN)),
        "aryl_halide": ("aryl_halide", (GroupRole.ARYL_C, GroupRole.HALOGEN)),
        "aryl-halide": ("aryl_halide", (GroupRole.ARYL_C, GroupRole.HALOGEN)),
        # Alkyl halides (C-X bonds)
        "c-f": ("alkyl_halide", None),
        "c-cl": ("alkyl_halide", None),
        "c-br": ("alkyl_halide", None),
        "c-i": ("alkyl_halide", None),
        "alkyl-halide": ("alkyl_halide", None),
        "alkyl_halide": ("alkyl_halide", None),
        "haloalkane": ("alkyl_halide", None),
        # C-H bonds (requires explicit hydrogens)
        "c-h": ("ch", None),
        "ch": ("ch", None),
        "c_h": ("ch", None),
        # Other X-H bonds
        "n-h": ("nh", None),
        "nh": ("nh", None),
        "o-h": ("oh", None),
        "oh": ("oh", None),
    }
    mapping = {_norm(k): v for k, v in _raw_mapping.items()}
    return mapping.get(token, (None, None))


def _select_group_for_bond(
    groups: List[Any],
    resolution: BondResolutionResult,
    group_type: Optional[str],
) -> Optional[Any]:
    if resolution.selected is None:
        return None
    if not groups:
        return None
    a_id, b_id = resolution.selected.atom_ids
    for group in groups:
        if group_type and group.group_type != group_type:
            continue
        group_ids = {atom.atom_id for atom in group.atoms}
        if a_id in group_ids and b_id in group_ids:
            return group
    return groups[0] if groups else None


def _build_fragment_from_group(
    atr: AtomicTruthRegistry,
    mol: Chem.Mol,
    group: Any,
    warnings: List[str],
    errors: List[str],
) -> Optional[Fragment]:
    try:
        builder = ChemicallyAwareFragmentBuilder()
        return builder.build_from_group(atr, mol, group)
    except Exception as exc:
        warnings.append(f"fragment build failed: {exc}")
        return None


def _build_fragment_from_bond(
    atr: AtomicTruthRegistry,
    mol: Chem.Mol,
    atom_ids: Tuple[str, str],
    warnings: List[str],
    errors: List[str],
) -> Optional[Fragment]:
    try:
        builder = ChemicallyAwareFragmentBuilder()
        return builder.build_from_bond(atr, mol, atom_ids)
    except Exception as exc:
        warnings.append(f"fragment build failed: {exc}")
        return None


def _bond_type_from_group(group: Optional[Any], fallback: Optional[str]) -> Optional[str]:
    if group is not None:
        return group.group_type
    return fallback


def _bond360_from_group(
    group: Optional[Any],
    resolution: BondResolutionResult,
    bond_type: Optional[str],
) -> Dict[str, Any]:
    bond_indices: List[int] = []
    attack_sites: Dict[str, Optional[int]] = {}
    if resolution.selected is not None:
        a_id, b_id = resolution.selected.atom_ids
        if group is not None:
            a = group.atoms[0].atr.get_by_uuid(a_id)
            b = group.atoms[0].atr.get_by_uuid(b_id)
            bond_indices = [a.parent_index, b.parent_index]
        attack_sites = _attack_sites_from_group(group)
    return {
        "bond_type": bond_type or "unknown",
        "primary_role": bond_type or "unknown",
        "bond_indices": bond_indices,
        "attack_sites": attack_sites,
    }


def _attack_sites_from_group(group: Optional[Any]) -> Dict[str, Optional[int]]:
    if group is None:
        return {}
    roles = group.roles
    if GroupRole.CARBONYL_C in roles:
        return {
            "electrophile": roles[GroupRole.CARBONYL_C].original_index,
            "stabilizable": roles.get(GroupRole.CARBONYL_O).original_index
            if GroupRole.CARBONYL_O in roles
            else None,
            "leaving": roles.get(GroupRole.HETERO_ATTACH).original_index
            if GroupRole.HETERO_ATTACH in roles
            else None,
        }
    if GroupRole.ARYL_C in roles and GroupRole.HALOGEN in roles:
        return {
            "electrophile": roles[GroupRole.ARYL_C].original_index,
            "leaving": roles[GroupRole.HALOGEN].original_index,
        }
    if GroupRole.EPOXIDE_O in roles:
        return {"epoxide_o": roles[GroupRole.EPOXIDE_O].original_index}
    return {}


def _classify_xh_bond(
    mol: Any,
    atr: AtomicTruthRegistry,
    bond_resolution: BondResolutionResult,
    bond_type: str,
) -> str:
    """Classify X-H bonds into coarse classes for radical/HAT priors."""
    bt = (bond_type or "").lower()
    if bt not in {"ch", "nh", "oh"}:
        return f"{bt}__unknown" if bt else "unknown"
    if bond_resolution.selected is None:
        return f"{bt}__aliphatic"

    heavy_id = None
    for aid in bond_resolution.selected.atom_ids:
        if "_H" not in str(aid):
            heavy_id = aid
            break
    if heavy_id is None:
        return f"{bt}__aliphatic"

    try:
        heavy_idx = int(atr.get_by_uuid(heavy_id).parent_index)
        heavy_atom = mol.GetAtomWithIdx(heavy_idx)
    except Exception:
        return f"{bt}__aliphatic"

    if bt == "nh":
        for nbr in heavy_atom.GetNeighbors():
            if nbr.GetSymbol() == "C":
                for b in nbr.GetBonds():
                    if b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(nbr).GetSymbol() == "O":
                        return "nh__amide"
        return "nh__amine"
    if bt == "oh":
        for nbr in heavy_atom.GetNeighbors():
            if nbr.GetIsAromatic():
                return "oh__phenol"
        return "oh__alcohol"

    # C-H classes
    if heavy_atom.GetIsAromatic():
        return "ch__aryl"
    nbr_symbols = [nbr.GetSymbol() for nbr in heavy_atom.GetNeighbors()]
    f_count = nbr_symbols.count("F")
    if f_count >= 3:
        return "ch__trifluoromethyl"
    if f_count == 2:
        return "ch__difluoromethyl"
    if f_count == 1:
        return "ch__fluoromethyl"
    if any(nbr.GetIsAromatic() for nbr in heavy_atom.GetNeighbors()):
        return "ch__benzylic"
    for nbr in heavy_atom.GetNeighbors():
        for bond in nbr.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(nbr)
                if other.GetIdx() != heavy_idx and other.GetSymbol() == "C":
                    return "ch__allylic"
    if any(sym in {"O", "N", "S"} for sym in nbr_symbols):
        return "ch__alpha_hetero"

    carbon_neighbors = sum(1 for nbr in heavy_atom.GetNeighbors() if nbr.GetSymbol() == "C")
    if carbon_neighbors >= 3:
        return "ch__tertiary"
    if carbon_neighbors == 2:
        return "ch__secondary"
    if carbon_neighbors == 1:
        return "ch__primary"
    return "ch__aliphatic"


def _selected_heavy_and_h_indices(
    atr: AtomicTruthRegistry,
    bond_resolution: BondResolutionResult,
) -> Tuple[Optional[int], Optional[int]]:
    if bond_resolution.selected is None:
        return None, None
    heavy_idx: Optional[int] = None
    h_idx: Optional[int] = None
    for atom_id in bond_resolution.selected.atom_ids:
        atom_id_s = str(atom_id)
        match = _SYNTHETIC_H_PATTERN.match(atom_id_s)
        if match:
            try:
                h_idx = int(match.group(1))
            except Exception:
                h_idx = None
            continue
        try:
            heavy_idx = int(atr.get_by_uuid(atom_id_s).parent_index)
        except Exception:
            continue
    return heavy_idx, h_idx


def _run_radical_cpts(
    mol: Any,
    bond_type: str,
    bond_class: str,
    atr: AtomicTruthRegistry,
    bond_resolution: BondResolutionResult,
    warnings: List[str],
) -> Dict[str, Any]:
    """Radical/HAT CPT track for C-H, N-H, O-H activation."""
    cpt_payload: Dict[str, Any] = {"status": "ok", "track": "radical_hat"}

    bde_rec = l1_bde_record(bond_class)
    bde_base = bde_rec.get("bde_kj_mol")
    if not isinstance(bde_base, (int, float)):
        bde_base = l1_bde_kj_mol(
            f"{bond_type}__aliphatic", default=BDE_TABLE_KJ_MOL.get(f"{bond_type}__aliphatic", 410.0)
        )
        warnings.append(f"BDE missing for {bond_class}; using fallback {float(bde_base):.1f} kJ/mol")

    alpha_correction = 0.0
    alpha_neighbors: Dict[str, int] = {}
    heavy_idx, h_idx = _selected_heavy_and_h_indices(atr, bond_resolution)
    if heavy_idx is not None:
        try:
            atom = mol.GetAtomWithIdx(heavy_idx)
            if _should_apply_alpha_correction(bond_class):
                for nbr in atom.GetNeighbors():
                    sym = nbr.GetSymbol()
                    corr = float(ALPHA_CORRECTION_KJ_MOL.get(sym, 0.0))
                    if corr != 0.0:
                        alpha_correction += corr
                        alpha_neighbors[sym] = alpha_neighbors.get(sym, 0) + 1
        except Exception as exc:
            warnings.append(f"radical CPT alpha correction failed: {exc}")

    bde_corrected = float(bde_base) + float(alpha_correction)
    layer3_bde: Dict[str, Any] = {
        "status": "disabled",
        "source": "layer3_xtb_disabled",
        "bde_kj_mol": None,
        "calibration_version": L3_CALIBRATION_VERSION,
    }
    if _L3_XTB_AVAILABLE and l3_is_layer3_xtb_enabled() and heavy_idx is not None and h_idx is not None:
        layer3_bde = l3_compute_substrate_bde_with_safeguard_for_mol(
            mol,
            (int(heavy_idx), int(h_idx)),
            bond_class=bond_class,
        )
        selected_bde = layer3_bde.get("bde_kj_mol")
        if isinstance(selected_bde, (int, float)):
            bde_corrected = float(selected_bde)
    radical_stability = float(l1_radical_stability_index(bond_class, default=0.20))

    # Emit both new and legacy mechanism keys so existing Module 0 routing keeps working.
    hat_barriers = l1_all_hat_barriers_kj_mol(
        bond_class=bond_class,
        bde_value_kj_mol=bde_corrected,
        protein_correction_kj=-8.0,
        include_legacy_aliases=True,
    )
    if not hat_barriers:
        hat_barriers = {}
        for mechanism, params in EVANS_POLANYI_PARAMS.items():
            barrier = float(params["alpha"]) * bde_corrected + float(params["beta"])
            hat_barriers[mechanism] = round(max(5.0, barrier), 1)
    else:
        hat_barriers = {k: round(max(5.0, float(v)), 1) for k, v in hat_barriers.items()}

    ranked_mechanisms = sorted(
        hat_barriers.keys(),
        key=lambda key: (float(hat_barriers[key]), 0 if key in {"Fe_IV_oxo", "Fe_IV_oxo_heme"} else 1),
    )
    best_mechanism = ranked_mechanisms[0]
    best_barrier = hat_barriers[best_mechanism]

    # Bond activation index: lower BDE relative to molecule -> easier activation.
    bai = 0.5
    try:
        mol_h = Chem.AddHs(mol) if Chem is not None else mol
        pair_bdes = {
            ("C", "F"): 485.0,
            ("C", "Cl"): 339.0,
            ("C", "Br"): 285.0,
            ("C", "H"): bde_corrected,
            ("C", "C"): 346.0,
            ("C", "N"): 305.0,
            ("C", "O"): 358.0,
            ("C", "S"): 272.0,
            ("N", "H"): 386.0,
            ("O", "H"): 436.0,
        }
        all_bdes: List[float] = []
        for bond in mol_h.GetBonds():
            a_sym = mol_h.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
            b_sym = mol_h.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
            pair = tuple(sorted([a_sym, b_sym]))
            all_bdes.append(float(pair_bdes.get(pair, 350.0)))
        if all_bdes:
            min_bde = min(all_bdes)
            max_bde = max(all_bdes)
            if max_bde > min_bde:
                bai = (max_bde - bde_corrected) / (max_bde - min_bde)
                bai = max(0.0, min(1.0, bai))
    except Exception as exc:
        warnings.append(f"radical CPT BAI failed: {exc}")

    bde_score = max(0.0, min(1.0, 1.0 - (bde_corrected - 350.0) / 150.0))
    composite = max(0.0, min(1.0, 0.4 * bde_score + 0.3 * radical_stability + 0.3 * bai))

    cpt_payload["bde"] = {
        "base_kj_mol": round(float(bde_base), 1),
        "alpha_correction_kj_mol": round(float(alpha_correction), 1),
        "corrected_kj_mol": round(float(bde_corrected), 1),
        "bond_class": bond_class,
        "alpha_neighbors": alpha_neighbors,
        "source": (
            str(layer3_bde.get("source"))
            if str(layer3_bde.get("source")) in {"xtb_validated", "lookup_safeguard", "lookup_fallback", "xtb_only"}
            else bde_rec.get("source") or "layer1_empirical"
        ),
        "uncertainty_kj_mol": (
            layer3_bde.get("uncertainty_kj_mol")
            if isinstance(layer3_bde.get("bde_kj_mol"), (int, float))
            else bde_rec.get("uncertainty_kj_mol")
        ),
        "lookup_bde_kj_mol": (
            round(float(layer3_bde.get("lookup_bde_kj_mol")), 1)
            if isinstance(layer3_bde.get("lookup_bde_kj_mol"), (int, float))
            else round(float(bde_base), 1)
        ),
        "xtb_bde_kj_mol": (
            round(float(layer3_bde.get("xtb_bde_kj_mol")), 1)
            if isinstance(layer3_bde.get("xtb_bde_kj_mol"), (int, float))
            else None
        ),
        "xtb_status": layer3_bde.get("status"),
        "xtb_error": layer3_bde.get("xtb_error") or layer3_bde.get("error"),
        "safeguard_threshold_kj_mol": layer3_bde.get("safeguard_threshold_kj_mol"),
        "deviation_kj_mol": layer3_bde.get("deviation_kj_mol"),
    }
    if layer3_bde.get("warning"):
        warnings.append(str(layer3_bde.get("warning")))
    cpt_payload["radical_stability"] = {
        "index": round(float(radical_stability), 3),
        "bond_class": bond_class,
        "interpretation": "high"
        if radical_stability > 0.6
        else "moderate"
        if radical_stability > 0.35
        else "low",
    }
    cpt_payload["hat_barriers"] = hat_barriers
    cpt_payload["best_hat"] = {
        "mechanism": best_mechanism,
        "barrier_kj_mol": best_barrier,
        "note": f"Lowest barrier via {best_mechanism}",
    }
    cpt_payload["calibration"] = {
        "layer": "layer1_empirical",
        "version": L1_CALIBRATION_VERSION,
        "layer3_xtb": {
            "enabled": bool(_L3_XTB_AVAILABLE and l3_is_layer3_xtb_enabled()),
            "status": layer3_bde.get("status"),
            "version": L3_CALIBRATION_VERSION,
        },
    }
    cpt_payload["bond_activation_index"] = {
        "bai": round(float(bai), 3),
        "interpretation": "favorable" if bai > 0.6 else "moderate" if bai > 0.3 else "unfavorable",
    }
    cpt_payload["composite_score"] = round(float(composite), 4)
    cpt_payload["feasibility_assessment"] = (
        "feasible" if composite > 0.5 else "marginal" if composite > 0.25 else "very_difficult"
    )
    if heavy_idx is not None:
        cpt_payload["target_heavy_atom_index"] = heavy_idx
    if h_idx is not None:
        cpt_payload["target_h_atom_index"] = h_idx
    return cpt_payload


def _run_cpts(
    fragment: Optional[Fragment],
    group: Optional[Any],
    warnings: List[str],
) -> Dict[str, Any]:
    if fragment is None or fragment.fragment_3d is None:
        return {"status": "no_fragment"}
    if fragment.fragment_3d.mol_3d is None:
        return {"status": "no_3d"}

    mol3d = fragment.fragment_3d.mol_3d
    role_to_idx = fragment.fragment_3d.role_to_frag_idx or {}
    group_type = group.group_type if group is not None else None
    cpt_payload: Dict[str, Any] = {"status": "ok"}

    if _CPT_AVAILABLE and role_to_idx:
        level1 = {}
        if GeometricCPTEngine is not None:
            try:
                engine = GeometricCPTEngine()
                profiles = engine.evaluate(mol3d=mol3d, role_to_idx=role_to_idx)
                level1 = {
                    "profiles": [asdict(p) for p in profiles],
                }
            except Exception as exc:
                warnings.append(f"level1 cpts failed: {exc}")

        level2 = {}
        if EnvironmentAwareStericsCPT_Level2 is not None:
            try:
                l2 = EnvironmentAwareStericsCPT_Level2()
                level2 = l2.run(mol3d=mol3d, role_to_idx=role_to_idx).__dict__
            except Exception as exc:
                warnings.append(f"level2 cpt failed: {exc}")

        epav = {}
        if ElectronicPropertiesAttackValidationCPT is not None:
            try:
                epav = ElectronicPropertiesAttackValidationCPT().run(
                    mol3d, role_to_idx, group_type=group_type
                ).__dict__
            except Exception as exc:
                warnings.append(f"epav failed: {exc}")

        # Build l2_best context for Level 3 from Level 2 results
        l2_best: Dict[str, Any] = {}
        if level2:
            l2_best = {
                "best_face": level2.get("best_face"),
                "best_wobble_deg": level2.get("best_wobble_deg"),
                "min_clearance_A": level2.get("min_clearance_A"),
                "attack_dir": level2.get("attack_dir"),
            }

        level3 = {}
        if Level3Orchestrator is not None:
            try:
                env = None
                if EnvContext is not None:
                    env = EnvContext.pseudo_oxyanion_hole(mol3d, role_to_idx)
                orchestrator = Level3Orchestrator(
                    cpts=[
                        OxyanionHoleGeometryCPT(),
                        TransitionStateChargeStabilizationCPT(),
                        SolventExposurePolarityCPT(),
                    ],
                    weights={
                        "oxyanion_hole_geometry": 0.45,
                        "ts_charge_stabilization": 0.30,
                        "solvent_exposure_polarity": 0.25,
                    },
                )
                level3 = orchestrator.run(mol3d, role_to_idx, l2_best, env).__dict__
            except Exception as exc:
                warnings.append(f"level3 cpt failed: {exc}")

        cpt_payload.update(
            {
                "level1": level1,
                "level2": level2,
                "epav": epav,
                "level3": level3,
            }
        )
    else:
        cpt_payload["status"] = "no_cpt"

    return cpt_payload


def _fragment_payload(fragment: Optional[Fragment]) -> Dict[str, Any]:
    if fragment is None:
        return {}
    return {
        "fragment_smiles": fragment.frag_smiles,
        "parent_smiles": fragment.parent_smiles,
        "mapped_smiles": fragment.mapped_smiles,
        "cut_bonds": [cb.parent_atom_indices for cb in fragment.cut_bonds],
        "cap_records": [cr.cap_atom_symbol for cr in fragment.cap_records],
        "warnings": fragment.warnings,
        "build_notes": fragment.build_notes,
        "context_metrics": fragment.context_metrics,
    }


def _mechanism_eligibility(bond_type: Optional[str]) -> Dict[str, str]:
    bond_type = (bond_type or "unknown").lower()
    if bond_type == "ester":
        return {
            "serine_hydrolase": "APPROVED",
            "metallo_esterase": "APPROVED",
            "cysteine_hydrolase": "REQUIRE_QUORUM",
            "acid_base": "APPROVED",
        }
    if bond_type == "amide":
        return {
            "serine_hydrolase": "REQUIRE_QUORUM",
            "metallo_esterase": "APPROVED",
            "cysteine_hydrolase": "REQUIRE_QUORUM",
            "acid_base": "REJECTED",
        }
    if bond_type == "aryl_halide":
        return {
            "serine_hydrolase": "REJECTED",
            "metallo_esterase": "REQUIRE_QUORUM",
            "radical_transfer": "REQUIRE_QUORUM",
        }
    if bond_type == "alkyl_halide":
        return {
            "haloalkane_dehalogenase": "APPROVED",
            "reductive_dehalogenase": "REQUIRE_QUORUM",
            "serine_hydrolase": "REJECTED",
            "radical_transfer": "REQUIRE_QUORUM",
        }
    if bond_type == "ch":
        return {
            "serine_hydrolase": "REJECTED",
            "radical_transfer": "REQUIRE_QUORUM",
            "p450_oxidation": "APPROVED",
            "monooxygenase": "APPROVED",
        }
    if bond_type in {"nh", "oh"}:
        return {
            "serine_hydrolase": "REJECTED",
            "acid_base": "APPROVED",
            "oxidoreductase": "REQUIRE_QUORUM",
        }
    return {"serine_hydrolase": "REQUIRE_QUORUM"}


def _confidence_prior_from_bond(bond_type: Optional[str]) -> float:
    bond_type = (bond_type or "unknown").lower()
    if bond_type == "ester":
        return 0.7
    if bond_type == "amide":
        return 0.5
    if bond_type == "aryl_halide":
        return 0.3
    if bond_type == "alkyl_halide":
        return 0.5
    if bond_type == "ch":
        return 0.3
    if bond_type in {"nh", "oh"}:
        return 0.4
    return 0.4


def _route_bias_from_bond(bond_type: Optional[str]) -> Dict[str, Any]:
    bond_type = (bond_type or "unknown").lower()
    if bond_type == "ester":
        return {
            "prefer": ["serine_hydrolase", "metallo_esterase"],
            "discourage": [],
            "strength": 0.2,
        }
    if bond_type == "amide":
        return {
            "prefer": ["metallo_esterase"],
            "discourage": ["serine_hydrolase"],
            "strength": 0.15,
        }
    if bond_type == "aryl_halide":
        return {
            "prefer": ["radical_transfer"],
            "discourage": ["serine_hydrolase"],
            "strength": 0.2,
        }
    if bond_type == "alkyl_halide":
        return {
            "prefer": ["haloalkane_dehalogenase", "reductive_dehalogenase"],
            "discourage": ["serine_hydrolase"],
            "strength": 0.25,
        }
    if bond_type == "ch":
        return {
            "prefer": ["p450_oxidation", "monooxygenase", "radical_transfer"],
            "discourage": ["serine_hydrolase"],
            "strength": 0.2,
        }
    if bond_type in {"nh", "oh"}:
        return {
            "prefer": ["acid_base", "oxidoreductase"],
            "discourage": [],
            "strength": 0.15,
        }
    return {"prefer": [], "discourage": [], "strength": 0.0}


def _derive_primary_constraint(cpt_scores: Dict[str, Any]) -> str:
    if (cpt_scores.get("track") or "").lower() == "radical_hat":
        comp = cpt_scores.get("composite_score")
        if isinstance(comp, (int, float)) and float(comp) < 0.35:
            return "BDE"
        return "RADICAL"
    if (cpt_scores.get("track") or "").lower() == "displacement_sn2":
        comp = cpt_scores.get("composite_score")
        if isinstance(comp, (int, float)) and float(comp) < 0.4:
            return "STERIC"
        return "LEAVING_GROUP"
    level2 = cpt_scores.get("level2") or {}
    level3 = cpt_scores.get("level3") or {}
    if isinstance(level2, dict):
        score = level2.get("score")
        if isinstance(score, (int, float)) and score < 0.55:
            return "GEOMETRIC"
    if isinstance(level3, dict):
        score = level3.get("score")
        if isinstance(score, (int, float)) and score < 0.55:
            return "ELECTROSTATIC"
    return "NONE"


def _build_output(
    status: str,
    bond360_profile: Dict[str, Any],
    fragment: Dict[str, Any],
    cpt_scores: Dict[str, Any],
    mechanism_eligibility: Dict[str, str],
    primary_constraint: str,
    confidence_prior: float,
    route_bias: Dict[str, Any],
    resolved_target: Optional[Dict[str, Any]] = None,
    reactivity: Optional[Dict[str, Any]] = None,
    module_minus1_schema: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    cache_hit: bool = False,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "bond360_profile": bond360_profile,
        "fragment": fragment,
        "cpt_scores": cpt_scores,
        "mechanism_eligibility": mechanism_eligibility,
        "primary_constraint": primary_constraint,
        "confidence_prior": round(float(confidence_prior), 3),
        "route_bias": route_bias,
        "resolved_target": resolved_target or {},
        "reactivity": reactivity or {},
        "module_minus1_schema": module_minus1_schema or {},
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "warnings": warnings or [],
        "errors": errors or [],
    }


def _candidate_attack_sites(
    resolution: BondResolutionResult, atr: Optional[AtomicTruthRegistry] = None
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for idx, cand in enumerate(resolution.candidates):
        atom_indices: Optional[List[int]] = None
        if atr is not None:
            try:
                a = atr.get_by_uuid(cand.atom_ids[0])
                b = atr.get_by_uuid(cand.atom_ids[1])
                atom_indices = [int(a.parent_index), int(b.parent_index)]
            except Exception:
                atom_indices = None
        ranked.append(
            {
                "rank": idx + 1,
                "atom_ids": list(cand.atom_ids),
                "atom_indices": atom_indices,
                "roles": [r.value for r in cand.roles] if cand.roles else None,
                "score": round(float((cand.evidence or {}).get("score", 1.0)), 4),
                "bond_order": cand.bond_order,
                "is_aromatic": cand.is_aromatic,
                "subclass": (cand.evidence or {}).get("subclass"),
                "bond_class": (cand.evidence or {}).get("bond_class"),
                "bde_kj_mol": (cand.evidence or {}).get("bde_kj_mol"),
                "radical_stability": (cand.evidence or {}).get("radical_stability"),
                "heavy_atom_index": (cand.evidence or {}).get("heavy_atom_index"),
                "resolution_policy": (cand.evidence or {}).get("resolution_policy"),
                "resolution_note": (cand.evidence or {}).get("resolution_note"),
                "resolution_confidence": (cand.evidence or {}).get("resolution_confidence"),
                "bde_gap_kj_mol": (cand.evidence or {}).get("bde_gap_kj_mol"),
            }
        )
    return ranked


def _resolution_metadata_from_result(resolution: BondResolutionResult) -> Dict[str, Any]:
    selected_evidence = (resolution.selected.evidence or {}) if resolution.selected is not None else {}
    return {
        "resolution_policy": selected_evidence.get("resolution_policy"),
        "resolution_note": selected_evidence.get("resolution_note"),
        "resolution_confidence": selected_evidence.get("resolution_confidence"),
        "bde_gap_kj_mol": selected_evidence.get("bde_gap_kj_mol"),
        "equivalent_sites_detected": bool(selected_evidence.get("equivalent_sites_detected")),
    }


def _competition_from_candidates(
    candidate_attack_sites: List[Dict[str, Any]],
    bond_type: Optional[str],
) -> Dict[str, Any]:
    if (bond_type or "").lower() != "ch" or len(candidate_attack_sites) <= 1:
        return {"gap": None, "best_other_idx": None, "best_other_score": None}

    if len(candidate_attack_sites) < 2:
        return {"gap": None, "best_other_idx": None, "best_other_score": None}

    top_policy = str(candidate_attack_sites[0].get("resolution_policy") or "")
    if top_policy.startswith("lowest_BDE_auto"):
        return {
            "gap": None,
            "best_other_idx": None,
            "best_other_score": None,
            "equivalent_sites_detected": False,
            "metric": "delta_BDE_kj_mol",
            "threshold_kj_mol": CH_AUTOPICK_MIN_DELTA_BDE_KJ,
        }

    best = candidate_attack_sites[0]
    best_bde = best.get("bde_kj_mol")
    if not isinstance(best_bde, (int, float)):
        return {"gap": None, "best_other_idx": None, "best_other_score": None}
    best_heavy = best.get("heavy_atom_index")

    second_site = next(
        (
            cand
            for cand in candidate_attack_sites[1:]
            if cand.get("heavy_atom_index") != best_heavy
            and isinstance(cand.get("bde_kj_mol"), (int, float))
        ),
        None,
    )
    if second_site is None:
        return {
            "gap": None,
            "best_other_idx": None,
            "best_other_score": None,
            "equivalent_sites_detected": False,
            "metric": "delta_BDE_kj_mol",
            "threshold_kj_mol": CH_AUTOPICK_MIN_DELTA_BDE_KJ,
        }

    second_bde = float(second_site["bde_kj_mol"])
    gap = second_bde - float(best_bde)
    near_best = [
        cand
        for cand in candidate_attack_sites
        if isinstance(cand.get("bde_kj_mol"), (int, float))
        and float(cand["bde_kj_mol"]) <= float(best_bde) + CH_AUTOPICK_MIN_DELTA_BDE_KJ
    ]
    unique_heavy = {
        cand.get("heavy_atom_index")
        for cand in near_best
        if cand.get("heavy_atom_index") is not None
    }
    return {
        "gap": round(float(gap), 3),
        "best_other_idx": second_site.get("rank"),
        "best_other_score": round(float(second_bde), 3),
        "equivalent_sites_detected": bool(
            len(unique_heavy) > 1 and abs(float(gap)) <= CH_AUTOPICK_MIN_DELTA_BDE_KJ
        ),
        "metric": "delta_BDE_kj_mol",
        "threshold_kj_mol": CH_AUTOPICK_MIN_DELTA_BDE_KJ,
    }


def _get_run_id(ctx: PipelineContext) -> str:
    shared = ctx.data.get("shared_io") or {}
    telemetry = (shared.get("input") or {}).get("telemetry") or {}
    run_id = telemetry.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    return str(uuid.uuid4())


def _merge_shared_io(ctx: PipelineContext, module_minus1_result: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    shared = ctx.data.get("shared_io") or {}
    shared_input = shared.get("input") or {}
    condition_profile = shared_input.get("condition_profile") or {
        "pH": ctx.constraints.ph_min if ctx.constraints.ph_min is not None else ctx.constraints.ph_max,
        "temperature_K": round(((ctx.constraints.temperature_c or 25.0) + 273.15), 2),
        "temperature_C": ctx.constraints.temperature_c,
        "solvent": None,
        "ionic_strength": None,
        "cofactors": [],
    }
    resolved_target = module_minus1_result.get("resolved_target") or {}
    shared_input.update(
        {
            "schema_version": "shared_io.v2",
            "substrate_context": {
                "smiles": ctx.smiles,
                "mol_block": None,
            },
            "bond_spec": {
                "target_bond": ctx.target_bond,
                "target_bond_indices": resolved_target.get("bond_indices") or None,
                "selection_mode": resolved_target.get("selection_mode"),
                "resolved_target": resolved_target,
                "context": module_minus1_result.get("bond360_profile") or {},
            },
            "condition_profile": condition_profile,
            "telemetry": {
                "run_id": run_id,
                "trace": list(dict.fromkeys(((shared_input.get("telemetry") or {}).get("trace") or []) + ["module-1"])),
                "warnings": list(dict.fromkeys(((shared_input.get("telemetry") or {}).get("warnings") or []) + (module_minus1_result.get("warnings") or []))),
            },
        }
    )
    outputs = shared.get("outputs") or {}
    outputs["module_minus1"] = {
        "result": {
            "status": module_minus1_result.get("status"),
            "resolved_target": resolved_target,
            "reactivity": module_minus1_result.get("reactivity"),
            "primary_constraint": module_minus1_result.get("primary_constraint"),
            "warnings": module_minus1_result.get("warnings", []),
            "errors": module_minus1_result.get("errors", []),
        },
        "sre_atr": {
            "bond360_profile": module_minus1_result.get("bond360_profile"),
            "module_minus1_schema": module_minus1_result.get("module_minus1_schema"),
        },
        "fragment_builder": module_minus1_result.get("fragment") or {},
        "cpt": module_minus1_result.get("cpt_scores") or {},
        "level1": (module_minus1_result.get("cpt_scores") or {}).get("level1", {}),
        "level2": (module_minus1_result.get("cpt_scores") or {}).get("level2", {}),
        "level3": (module_minus1_result.get("cpt_scores") or {}).get("level3", {}),
        "ep_av": (module_minus1_result.get("cpt_scores") or {}).get("epav", {}),
        "evidence_record": {
            "mechanism_eligibility": module_minus1_result.get("mechanism_eligibility"),
            "route_bias": module_minus1_result.get("route_bias"),
            "confidence_prior": module_minus1_result.get("confidence_prior"),
        },
    }
    shared["input"] = shared_input
    shared["outputs"] = outputs
    return shared


def _write_module_minus1_datapoint(ctx: PipelineContext, module_minus1_result: Dict[str, Any], run_id: str) -> None:
    db_path = os.environ.get("EVIDENCE_DB_PATH")
    if not db_path:
        return
    datapoints = [
        {
            "module_id": -1,
            "item_type": "module_minus1_summary",
            "data": {
                "status": module_minus1_result.get("status"),
                "resolved_target": module_minus1_result.get("resolved_target"),
                "module_minus1_schema": module_minus1_result.get("module_minus1_schema"),
                "reactivity": module_minus1_result.get("reactivity"),
            },
            "reasons": (module_minus1_result.get("warnings") or []) + (module_minus1_result.get("errors") or []),
        }
    ]
    try:
        add_datapoints(db_path, run_id, datapoints)
    except Exception:
        # run_id may not exist yet in evidence DB; ignore to keep pipeline robust.
        return
