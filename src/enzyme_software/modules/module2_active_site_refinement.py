from __future__ import annotations

# Contract Notes (output contract freeze):
# - ctx.data["module2_active_site_refinement"] must preserve keys: status, halt_reason, selected_scaffold,
#   selection_explain, objective, variant_set, module3_handoff, predicted_under_given_conditions,
#   optimum_conditions_estimate, delta_from_optimum, confidence_calibrated, evidence_record,
#   module2_physics_audit, warnings/errors, and report sections (candidate_reports, final_report, etc.).
# - shared_io updates via _merge_shared_io must keep shared_io["input"/"outputs"] intact.
# - New physics fields should be added under module2_physics_audit or math_contract, not rename existing keys.
# - Variant boosts/heuristic adjustments are applied in _rank_scaffolds and _apply_variant_energy_scoring.

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

from enzyme_software.context import PipelineContext
from enzyme_software.biocore import (
    nucleophile_change_penalty,
    residue_state_fraction,
)
from enzyme_software.chemcore import (
    PKA_CATALYTIC_GROUPS,
    fraction_deprotonated,
    fraction_protonated,
    solvent_penalty,
)
from enzyme_software.domain import (
    BondSpec,
    ConditionProfile,
    EvidenceRecord,
    FeatureVector,
    SharedInput,
    SharedIO,
    SharedOutput,
    SubstrateContext,
    TelemetryContext,
)
from enzyme_software.mathcore import (
    delta_g_with_conditions,
    estimate_delta_g,
    DistributionEstimate,
    ProbabilityEstimate,
    QCReport,
    rate_constant,
    sample_k_pred,
    percentile,
    record_event,
    search_optimum_conditions,
    validate_math_contract,
)
from enzyme_software.physicscore import (
    J_PER_KCAL,
    R_J_per_molK,
    barrier_gate,
    c_to_k,
    compute_route_prior,
    eyring_rate_constant,
    estimate_barrier_shift_kJ,
    kj_to_j,
)
from enzyme_software.mechanism_registry import resolve_mechanism
from enzyme_software.unity_schema import MechanismSpec
from enzyme_software.scorecard import (
    ScoreCard,
    ScoreCardMetric,
    contributors_from_features,
    metric_status,
)
from enzyme_software.score_ledger import ScoreLedger, ScoreTerm
from enzyme_software.modules.base import BaseModule
try:
    from enzyme_software.calibration.layer2_structure_db import resolve_variant_targets
except Exception:  # pragma: no cover - optional integration
    def resolve_variant_targets(
        enzyme_family: Optional[str],
        pdb_id: Optional[str],
        variant_label: Optional[str],
    ) -> Dict[str, Any]:
        return {}

try:
    from enzyme_software.calibration.layer3_xtb import (
        compute_substrate_strain as l3_compute_substrate_strain,
        is_layer3_xtb_enabled as l3_is_layer3_xtb_enabled,
    )
except Exception:  # pragma: no cover - optional integration
    def l3_compute_substrate_strain(
        smiles: str,
        *,
        bound_conformation_xyz: Optional[str] = None,
        solvent: Optional[str] = "water",
        xtb_path: str = "xtb",
        n_cores: int = 1,
        timeout_s: int = 300,
        default_kj_mol: float = 1.5,
    ) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "source": "layer3_xtb_unavailable",
            "strain_kj_mol": float(default_kj_mol),
        }

    def l3_is_layer3_xtb_enabled() -> bool:
        return False

try:
    from enzyme_software.calibration.layer3_vina import (
        dock_for_module2 as l3_dock_for_module2,
        is_layer3_vina_enabled as l3_is_layer3_vina_enabled,
    )
except Exception:  # pragma: no cover - optional integration
    def l3_dock_for_module2(job_card: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"status": "unavailable", "source": "layer3_vina_unavailable"}

    def l3_is_layer3_vina_enabled() -> bool:
        return False

try:
    from enzyme_software.calibration.layer3_openmm import (
        stability_for_module2 as l3_stability_for_module2,
        is_layer3_openmm_enabled as l3_is_layer3_openmm_enabled,
    )
except Exception:  # pragma: no cover - optional integration
    def l3_stability_for_module2(docking_result: Dict[str, Any], job_card: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "unavailable", "source": "layer3_openmm_unavailable"}

    def l3_is_layer3_openmm_enabled() -> bool:
        return False

FAIL_NO_TOP_SCAFFOLDS = "FAIL_NO_TOP_SCAFFOLDS"
FAIL_NO_TARGET = "FAIL_NO_TARGET"
FAIL_DECISION_NOT_GO = "FAIL_DECISION_NOT_GO"
STATUS_DEGRADED_OK = "DEGRADED_OK"
CONDITION_MEMORY_PATH = Path(__file__).resolve().parents[3] / "cache" / "condition_performance.json"
PHYSICS_GATE_SCORE_MULTIPLIER = 0.3  # Reduce rank when barrier exceeds thermal limit.
KCAL_TO_KJ = 4.184
MECHANISM_MISMATCH_PENALTY_KJ = 2.0
MECHANISM_UNVERIFIED_PENALTY_KJ = 1.5
MECHANISM_UNVERIFIED_CONFIDENCE_PENALTY = 0.1
ALT_NUCLEOPHILE_POSTERIOR_THRESHOLD = 0.35
STRAIN_K_MODEL_KJ_PER_MOL_A2 = 2.0  # Conservative spring constant proxy.
STRAIN_THRESHOLD_KJ_PER_MOL = 8.0  # Above this, strain cancels beneficial ΔΔG‡.
STRAIN_FLOOR_KJ_PER_MOL = 1.5  # Default penalty when localization is unknown.
DDG_BIND_KCAL_BY_CATEGORY = {
    "retention_clamp": -0.7,
    "polar_anchor": -0.8,
    "oxyanion_hole": -0.5,
    "access_preserving_clamp": -0.4,
    "mechanism_alignment": -0.2,
    "pH_tuning": -0.2,
    "radical_metal_opt": -1.0,
    "radical_tunnel": -0.8,
    "radical_rebound": -0.6,
    "radical_electron_transfer": -0.5,
    "baseline": 0.0,
}
DDG_STRAIN_KCAL_BY_CATEGORY = {
    "retention_clamp": 0.4,
    "polar_anchor": 0.3,
    "oxyanion_hole": 0.2,
    "access_preserving_clamp": 0.3,
    "mechanism_alignment": 0.1,
    "pH_tuning": 0.2,
    "radical_metal_opt": 0.25,
    "radical_tunnel": 0.35,
    "radical_rebound": 0.3,
    "radical_electron_transfer": 0.2,
    "baseline": 0.0,
}

# Variant-specific barrier/retention model for radical and hydrolytic designs.
VARIANT_DDG_MODELS: Dict[str, Dict[str, Any]] = {
    "metal_first_shell": {
        "base_ddg_kj": -8.0,
        "distance_shell": 1,
        "confidence": 0.5,
        "retention_boost": 0.0,
        "mechanism": "metal redox tuning",
    },
    "substrate_channel": {
        "base_ddg_kj": -0.5,
        "distance_shell": 3,
        "confidence": 0.4,
        "retention_boost": 0.12,
        "mechanism": "substrate tunnel alignment",
    },
    "substrate_gate": {
        "base_ddg_kj": -0.2,
        "distance_shell": 3,
        "confidence": 0.35,
        "retention_boost": 0.08,
        "mechanism": "substrate trapping gate",
    },
    "radical_cage": {
        "base_ddg_kj": -3.0,
        "distance_shell": 2,
        "confidence": 0.4,
        "retention_boost": 0.04,
        "mechanism": "radical rebound cage",
    },
    "second_shell_electrostatics": {
        "base_ddg_kj": -2.0,
        "distance_shell": 2,
        "confidence": 0.35,
        "retention_boost": 0.0,
        "mechanism": "second-shell electrostatics",
    },
    "polar_anchor": {
        "base_ddg_kj": -1.5,
        "distance_shell": 2,
        "confidence": 0.4,
        "retention_boost": 0.02,
        "mechanism": "polar anchoring",
    },
    "oxyanion_hole": {
        "base_ddg_kj": -2.5,
        "distance_shell": 1,
        "confidence": 0.5,
        "retention_boost": 0.0,
        "mechanism": "oxyanion stabilization",
    },
}
SHELL_ATTENUATION: Dict[int, float] = {1: 1.0, 2: 0.6, 3: 0.25}


def _substrate_smiles_from_job_card(job_card: Dict[str, Any]) -> Optional[str]:
    candidates = [
        job_card.get("smiles"),
        job_card.get("substrate_smiles"),
        (job_card.get("substrate_context") or {}).get("smiles"),
        ((job_card.get("shared_io") or {}).get("input") or {}).get("substrate_context", {}).get("smiles"),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _layer3_strain_floor(job_card: Dict[str, Any], default_kj: float = STRAIN_FLOOR_KJ_PER_MOL) -> Dict[str, Any]:
    if not l3_is_layer3_xtb_enabled():
        return {"strain_kj_mol": float(default_kj), "source": "default_floor", "status": "disabled"}
    smiles = _substrate_smiles_from_job_card(job_card)
    if not smiles:
        return {"strain_kj_mol": float(default_kj), "source": "default_floor", "status": "missing_smiles"}
    payload = l3_compute_substrate_strain(smiles, default_kj_mol=float(default_kj))
    strain_val = payload.get("strain_kj_mol")
    if not isinstance(strain_val, (int, float)):
        return {
            "strain_kj_mol": float(default_kj),
            "source": "default_floor",
            "status": payload.get("status") or "fallback",
            "error": payload.get("error"),
        }
    return {
        "strain_kj_mol": float(strain_val),
        "source": payload.get("source") or "layer3_xtb",
        "status": payload.get("status") or "ok",
        "error": payload.get("error"),
    }


def _compute_composite_binding_score(
    docking: Dict[str, Any],
    md_stability: Dict[str, Any],
) -> Optional[float]:
    topogate = docking.get("topogate_scores") or {}
    vina_score = topogate.get("composite")
    md_verdict = md_stability.get("verdict")
    md_score = None
    if md_verdict == "STABLE":
        md_score = 1.0
    elif md_verdict == "UNSTABLE":
        md_score = 0.3
    if isinstance(vina_score, (int, float)) and isinstance(md_score, (int, float)):
        return round(0.6 * float(vina_score) + 0.4 * float(md_score), 3)
    if isinstance(vina_score, (int, float)):
        return round(float(vina_score), 3)
    if isinstance(md_score, (int, float)):
        return round(float(md_score), 3)
    return None


def _layer3_docking_and_stability(
    job_card: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraint_map = constraints or {}
    vina_override = constraint_map.get("enable_vina")
    openmm_override = constraint_map.get("enable_openmm")
    vina_enabled = bool(vina_override) if isinstance(vina_override, bool) else l3_is_layer3_vina_enabled()
    openmm_enabled = bool(openmm_override) if isinstance(openmm_override, bool) else l3_is_layer3_openmm_enabled()

    result = {
        "docking": {"status": "disabled" if not vina_enabled else "pending"},
        "md_stability": {"status": "disabled" if not openmm_enabled else "pending"},
        "composite_binding_score": None,
        "binding_score_adjustment": None,
        "engines_used": [],
    }
    if vina_enabled:
        result["docking"] = l3_dock_for_module2(job_card, constraints=constraint_map)
        if result["docking"].get("status") == "ok":
            result["engines_used"].append("vina")
    if openmm_enabled and result["docking"].get("status") == "ok":
        result["md_stability"] = l3_stability_for_module2(
            docking_result=result["docking"],
            job_card=job_card,
        )
        if result["md_stability"].get("status") == "ok":
            result["engines_used"].append("openmm")

    composite = _compute_composite_binding_score(
        docking=result["docking"],
        md_stability=result["md_stability"],
    )
    result["composite_binding_score"] = composite
    if isinstance(composite, (int, float)):
        result["binding_score_adjustment"] = round((float(composite) - 0.5) * 0.4, 3)
    return result


def _expected_nucleophile_for_route(route_label: str) -> str:
    label = str(route_label or "").lower()
    if "serine" in label:
        return "Ser"
    if "cys" in label or "cysteine" in label or "thiol" in label:
        return "Cys"
    if "metallo" in label or "metal" in label:
        return "Either"
    return "Either"


def _expected_motif_for_route(route_label: str) -> str:
    label = str(route_label or "").lower()
    if "serine" in label:
        return "Ser-His-Asp triad"
    if "cys" in label or "cysteine" in label or "thiol" in label:
        return "Cys-His-Asp triad"
    if "metallo" in label or "metal" in label:
        return "metal-assisted active site"
    return "generic catalytic motif"


def _detect_nucleophile(candidate_residues_by_role: Dict[str, Any]) -> Optional[str]:
    residues = candidate_residues_by_role.get("nucleophile") or []
    for residue in residues:
        token = str(residue).strip()
        if token[:3].lower() == "ser":
            return "Ser"
        if token[:3].lower() == "cys":
            return "Cys"
        if token[:3].lower() == "thr":
            return "Ser"
    return None


def _scan_pdb_for_residues(pdb_path: Optional[str], residue_names: List[str]) -> List[str]:
    if not pdb_path:
        return []
    path = Path(pdb_path)
    if not path.exists():
        return []
    found: List[str] = []
    names = {name.upper() for name in residue_names}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                resname = line[17:20].strip().upper()
                if resname in names:
                    found.append(resname)
    except OSError:
        return []
    return sorted(set(found))


def _resolve_mechanism_evidence(
    contract: Dict[str, Any],
    candidate_residues_by_role: Dict[str, Any],
    pdb_path: Optional[str],
) -> Dict[str, Any]:
    expected = contract.get("expected_nucleophile")
    observed = _detect_nucleophile(candidate_residues_by_role)
    evidence_residues = candidate_residues_by_role.get("nucleophile") or []
    if observed:
        return {
            "status": "VERIFIED",
            "expected": expected,
            "observed": observed,
            "evidence_residues": list(evidence_residues),
            "source": "candidate_residues_by_role",
            "explanation": "Detected nucleophile residue from reach candidates.",
        }
    residue_map = {"Ser": ["SER", "THR"], "Cys": ["CYS"], "MetalWater": []}
    pdb_hits = _scan_pdb_for_residues(pdb_path, residue_map.get(expected, []))
    if pdb_hits:
        return {
            "status": "VERIFIED",
            "expected": expected,
            "observed": None,
            "evidence_residues": pdb_hits,
            "source": "pdb_scan",
            "explanation": "Expected nucleophile residue present in PDB (site not localized).",
        }
    return {
        "status": "UNVERIFIED",
        "expected": expected,
        "observed": None,
        "evidence_residues": [],
        "source": "none",
        "explanation": "No nucleophile evidence from scaffold or PDB.",
    }


def _mechanism_policy_action(
    policy: str,
    expected: str,
    detected: Optional[str],
) -> str:
    if expected == "Either":
        return "KEEP_WITH_PENALTY"
    if detected is None or detected == expected:
        return "KEEP_WITH_PENALTY"
    if policy == "strict":
        return "SWITCH_ROUTE"
    return "REQUEST_DISAMBIGUATION"


def _compatibility_score(expected: str, detected: Optional[str]) -> Tuple[float, Optional[str]]:
    if detected is None:
        if expected == "Either":
            return 0.6, None
        return 0.4, f"Expected {expected} nucleophile but none detected."
    if expected == "Either":
        return 0.8, None
    if expected == detected:
        return 0.95, None
    return 0.25, f"Expected {expected} nucleophile but detected {detected}."


def _build_mechanism_spec(
    reaction_family: str,
    route_label: str,
    candidate_residues_by_role: Dict[str, Any],
    mechanism_policy: str,
    expected_override: Optional[str] = None,
) -> Dict[str, Any]:
    expected = expected_override or _expected_nucleophile_for_route(route_label)
    motif = _expected_motif_for_route(route_label)
    detected = _detect_nucleophile(candidate_residues_by_role)
    score, reason = _compatibility_score(expected, detected)
    policy_action = _mechanism_policy_action(mechanism_policy, expected, detected)
    spec = MechanismSpec(
        reaction_family=reaction_family or "unknown",
        route_label=route_label or "unknown",
        expected_nucleophile=expected,
        expected_motif=motif,
        detected_nucleophile=detected,
        detected_motif_residues={
            role: list(value) for role, value in candidate_residues_by_role.items()
        },
        compatibility_score=round(float(score), 3),
        mismatch_reason=reason,
        policy_action=policy_action,
    )
    return asdict(spec)


def _geometry_from_contract(contract: Dict[str, Any]) -> Optional[str]:
    allowed = contract.get("allowed_nucleophile_geometries") or []
    if isinstance(allowed, list) and allowed:
        return str(allowed[0])
    return None


def _mechanism_mismatch_from_contract(
    contract: Dict[str, Any],
    observed_nucleophile: Optional[str],
) -> Dict[str, Any]:
    expected = contract.get("expected_nucleophile")
    policy = contract.get("mismatch_policy_default") or "KEEP_WITH_PENALTY"
    if expected in {None, "Either"}:
        return {
            "status": "OK",
            "expected": expected,
            "observed": observed_nucleophile,
            "penalty_kj_mol": 0.0,
            "policy": policy,
            "explanation": "Either nucleophile allowed by contract.",
        }
    if observed_nucleophile is None:
        return {
            "status": "UNVERIFIED",
            "expected": expected,
            "observed": None,
            "penalty_kj_mol": 0.0,
            "confidence_penalty": MECHANISM_UNVERIFIED_CONFIDENCE_PENALTY,
            "policy": policy,
            "explanation": "No detected nucleophile from scaffold residues.",
        }
    if str(expected).lower() == str(observed_nucleophile).lower():
        return {
            "status": "OK",
            "expected": expected,
            "observed": observed_nucleophile,
            "penalty_kj_mol": 0.0,
            "policy": policy,
            "explanation": "Observed nucleophile matches contract.",
        }
    return {
        "status": "MISMATCH",
        "expected": expected,
        "observed": observed_nucleophile,
        "penalty_kj_mol": MECHANISM_MISMATCH_PENALTY_KJ,
        "policy": policy,
        "explanation": f"Expected {expected}, observed {observed_nucleophile}.",
    }


def _route_posterior(job_card: Dict[str, Any], route_name: str) -> Optional[float]:
    posteriors = job_card.get("route_posteriors") or []
    for entry in posteriors:
        if str(entry.get("route") or "").lower() == str(route_name).lower():
            value = entry.get("posterior")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _build_scorecard_module2(
    selected: Dict[str, Any],
    confidence_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    evidence_count = confidence_calibration.get("evidence_count")
    features = {
        "k_pred_mean": selected.get("k_pred_mean") or selected.get("k_pred"),
        "delta_g_mean": selected.get("delta_g_mean") or selected.get("delta_g"),
        "model_risk": selected.get("model_risk"),
    }
    contributors = contributors_from_features(features, limit=5)
    k_ci90 = selected.get("k_pred_ci90")
    ci90 = None
    if isinstance(k_ci90, (list, tuple)) and len(k_ci90) >= 2:
        ci90 = (float(k_ci90[0]), float(k_ci90[1]))
    metrics = [
        ScoreCardMetric(
            name="k_pred",
            raw=float(selected.get("k_pred_mean") or selected.get("k_pred") or 0.0),
            calibrated=float(selected.get("k_pred_mean") or selected.get("k_pred") or 0.0),
            ci90=ci90,
            n_eff=float(evidence_count) if isinstance(evidence_count, (int, float)) else None,
            status=metric_status(selected.get("k_pred_mean") or selected.get("k_pred"), evidence_count),
            definition="Predicted rate proxy (Eyring-based) for top scaffold.",
            contributors=contributors,
        ),
        ScoreCardMetric(
            name="confidence",
            raw=float(confidence_calibration.get("calibrated_confidence") or 0.0),
            calibrated=float(confidence_calibration.get("calibrated_confidence") or 0.0),
            ci90=None,
            n_eff=float(evidence_count) if isinstance(evidence_count, (int, float)) else None,
            status=metric_status(confidence_calibration.get("calibrated_confidence"), evidence_count),
            definition="Calibrated probability of wet-lab success.",
            contributors=contributors,
        ),
    ]
    calibration_status = "weakly_calibrated" if (evidence_count or 0) > 0 else "heuristic"
    return ScoreCard(module_id=2, metrics=metrics, calibration_status=calibration_status).to_dict()


def _build_score_ledger_module2(
    selected: Dict[str, Any],
    module2_physics_audit: Dict[str, Any],
    mechanism_mismatch: Dict[str, Any],
) -> Dict[str, Any]:
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    mismatch_penalty = mechanism_mismatch.get("penalty_kj_mol")
    terms = [
        ScoreTerm(
            name="deltaG_base_kJ_per_mol",
            value=_as_float(module2_physics_audit.get("deltaG_dagger_baseline_kJ_per_mol")),
            unit="kJ/mol",
            formula="baseline barrier estimate",
            inputs={},
            notes="Baseline barrier prior before variant shifts.",
        ),
        ScoreTerm(
            name="deltaDeltaG_kJ_per_mol",
            value=_as_float(module2_physics_audit.get("delta_deltaG_dagger_kJ_per_mol")),
            unit="kJ/mol",
            formula="physicscore.estimate_barrier_shift_kJ",
            inputs={"variant_id": (selected or {}).get("best_variant", {}).get("variant_id")},
            notes="Variant barrier shift prior (negative lowers barrier).",
        ),
        ScoreTerm(
            name="strain_kJ_per_mol",
            value=_as_float(module2_physics_audit.get("strain_energy_kJ_per_mol")),
            unit="kJ/mol",
            formula="0.5 * k_model * displacement^2",
            inputs={},
            notes="Strain penalty applied to barrier when localization is uncertain.",
        ),
        ScoreTerm(
            name="mechanism_mismatch_penalty_kJ_per_mol",
            value=_as_float(mismatch_penalty),
            unit="kJ/mol",
            formula="mechanism_registry penalty",
            inputs={
                "expected": mechanism_mismatch.get("expected"),
                "observed": mechanism_mismatch.get("observed"),
                "policy": mechanism_mismatch.get("policy"),
            },
            notes=mechanism_mismatch.get("explanation") or "Mechanism mismatch penalty.",
        ),
        ScoreTerm(
            name="deltaG_variant_kJ_per_mol",
            value=_as_float(module2_physics_audit.get("deltaG_dagger_variant_kJ_per_mol")),
            unit="kJ/mol",
            formula="deltaG_base + deltaDeltaG + strain + mismatch_penalty",
            inputs={},
            notes="Final activation barrier used for k_variant.",
        ),
        ScoreTerm(
            name="k_variant_s_inv",
            value=_as_float(module2_physics_audit.get("eyring_k_variant_s_inv")),
            unit="s^-1",
            formula="Eyring equation",
            inputs={"temperature_K": module2_physics_audit.get("temperature_K")},
            notes="Variant rate estimate from Eyring.",
        ),
    ]
    return ScoreLedger(module_id=2, terms=terms).to_dict()
R_KCAL_PER_MOLK = R_J_per_molK / J_PER_KCAL
DEFAULT_DIFFUSION_COEFF_M2_S = 5.0e-10


@dataclass
class Variant:
    variant_id: str
    label: str
    description: str
    mutations: List[Dict[str, Any]]
    rationale: str
    estimated_effects: Dict[str, float]
    score: float
    category: str
    requires_structural_localization: bool
    delta_deltaG_dagger_kJ_per_mol: float = 0.0
    strain_displacement_A: float = 0.0
    strain_note: str = "unknown -> no strain penalty applied"


class Module2ActiveSiteRefinement(BaseModule):
    name = "Module 2 - Active-Site Refinement + Variant Proposal"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        job_card = ctx.data.get("job_card") or {}
        shared = ctx.data.get("shared_io") or {}
        if _reaction_hash_mismatch(shared, job_card):
            job_card["pipeline_halt_reason"] = "HASH_MISMATCH"
            warnings = job_card.get("warnings") or []
            warnings.append("W_HASH_MISMATCH: reaction identity mismatch; halting module2.")
            job_card["warnings"] = list(dict.fromkeys(warnings))
            ctx.data["job_card"] = job_card
            result = {
                "status": "FAIL",
                "halt_reason": "FAIL_HASH_MISMATCH",
                "warnings": warnings,
                "errors": [],
            }
            ctx.data["module2_active_site_refinement"] = result
            ctx.data["shared_io"] = _merge_shared_io(ctx, result)
            _update_unity_record_parts(ctx, result)
            return ctx
        module0_job_card = (ctx.data.get("module0_strategy_router") or {}).get("job_card")
        warnings: List[str] = []
        if module0_job_card and module0_job_card != job_card:
            warnings.append(
                "W_JOB_CARD_MISMATCH: module0_strategy_router.job_card differs from data.job_card; using data.job_card."
            )

        module1 = ctx.data.get("module1_topogate") or {}
        module1_handoff = module1.get("module2_handoff") or {}
        top_scaffolds = module1_handoff.get("top_scaffolds") or []

        unity_state = ctx.data.get("unity_state")
        result = run_module2(job_card, top_scaffolds, unity_state=unity_state)
        energy_ledger = (ctx.data.get("shared_io") or {}).get("energy_ledger") or {}
        energy_update = _energy_ledger_update(result, energy_ledger)
        result["energy_ledger_update"] = energy_update
        module2_physics = result.get("module2_physics_audit")
        if isinstance(module2_physics, dict):
            module2_physics["energy_ledger_update"] = energy_update
            result["module2_physics_audit"] = module2_physics
        if warnings:
            result["warnings"] = list(dict.fromkeys(result.get("warnings", []) + warnings))

        mechanism_spec = result.get("mechanism_spec") or {}
        policy_action = mechanism_spec.get("policy_action")
        if policy_action == "REQUEST_DISAMBIGUATION":
            penalty_factor = 0.6
            confidence = job_card.get("confidence") or {}
            for key in ("route", "feasibility_if_specified"):
                value = confidence.get(key)
                if isinstance(value, (int, float)):
                    confidence[key] = round(float(value) * penalty_factor, 3)
            confidence["mechanism_penalty_factor"] = penalty_factor
            job_card["confidence"] = confidence
            warnings = job_card.get("warnings") or []
            warnings.append(
                "W_MECH_DISAMBIGUATION: mechanism mismatch requires disambiguation arm."
            )
            job_card["warnings"] = list(dict.fromkeys(warnings))
        if policy_action == "SWITCH_ROUTE":
            route_override = result.get("mechanism_route_override")
            if route_override:
                job_card["chosen_route"] = route_override
                mechanism_route = job_card.get("mechanism_route")
                if isinstance(mechanism_route, dict):
                    mechanism_route = dict(mechanism_route)
                    mechanism_route["primary"] = route_override
                    job_card["mechanism_route"] = mechanism_route
                warnings = job_card.get("warnings") or []
                warnings.append(
                    f"W_MECH_SWITCH: route switched to {route_override} due to nucleophile mismatch."
                )
                job_card["warnings"] = list(dict.fromkeys(warnings))
        if mechanism_spec:
            job_card["mechanism_spec"] = mechanism_spec
            job_card["mechanism_policy"] = result.get("mechanism_policy")
        if result.get("mechanism_mismatch") is not None:
            job_card["mechanism_mismatch"] = result.get("mechanism_mismatch")
        if result.get("mechanism_contract") is not None:
            job_card["mechanism_contract"] = result.get("mechanism_contract")

        feedback = _module2_bidirectional_feedback(job_card, result)
        if feedback:
            job_card.setdefault("bidirectional_feedback", {})["module2"] = feedback
            penalty = feedback.get("penalty") or 0.0
            if isinstance(penalty, (int, float)) and penalty > 0.0:
                confidence = job_card.get("confidence") or {}
                for key in ("route", "feasibility_if_specified"):
                    value = confidence.get(key)
                    if isinstance(value, (int, float)):
                        adjusted = max(0.0, min(1.0, float(value) * (1.0 - float(penalty))))
                        confidence[key] = round(adjusted, 3)
                confidence["bidirectional_penalty"] = round(float(penalty), 3)
                job_card["confidence"] = confidence
            if feedback.get("hard_veto"):
                warnings = job_card.get("warnings") or []
                warnings.append(
                    "W_MODULE2_VETO: downstream kinetics/geometry indicates reroute or review."
                )
                job_card["warnings"] = list(dict.fromkeys(warnings))
            ctx.data["job_card"] = job_card

        ctx.data["module2_active_site_refinement"] = result
        ctx.data["shared_io"] = _merge_shared_io(ctx, result)
        _update_unity_record_parts(ctx, result)
        return ctx


def _update_unity_record_parts(ctx: PipelineContext, module_output: Dict[str, Any]) -> None:
    parts = ctx.data.setdefault("unity_record_parts", {})
    parts["module2"] = {"module_output": module_output}


def _module2_bidirectional_feedback(
    job_card: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    reasons: List[str] = []
    penalty = 0.0
    hard_veto = False
    status = result.get("status")
    if status and status not in {"PASS", "DEGRADED_OK"}:
        hard_veto = True
        reasons.append(f"module2_status_{status.lower()}")
        penalty += 0.2
    physics_audit = result.get("module2_physics_audit") or {}
    k_eff = physics_audit.get("k_eff_s_inv")
    if isinstance(k_eff, (int, float)) and float(k_eff) < 1e-4:
        penalty += 0.2
        reasons.append("k_eff_below_floor")
    selected = result.get("selected_scaffold") or {}
    retention_flag = (selected.get("retention_metrics") or {}).get("retention_risk_flag")
    if retention_flag == "HIGH":
        penalty += 0.1
        reasons.append("high_retention_risk")
    model_risk = selected.get("model_risk")
    if isinstance(model_risk, (int, float)) and float(model_risk) > 1.2:
        penalty += 0.1
        reasons.append("high_model_risk")
    penalty = min(0.5, float(penalty))
    return {
        "hard_veto": hard_veto,
        "penalty": round(float(penalty), 3),
        "reasons": reasons,
        "recommendation": "review_required" if hard_veto else "monitor",
    }


def run_module2(
    job_card: Dict[str, Any],
    top_scaffolds: List[Dict[str, Any]],
    unity_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not top_scaffolds:
        return _fail_module2(FAIL_NO_TOP_SCAFFOLDS, errors, warnings, job_card, top_scaffolds)

    status = "PASS"
    halt_reason = None
    decision = job_card.get("decision")
    if decision not in {"GO", "LOW_CONF_GO"}:
        status = STATUS_DEGRADED_OK
        halt_reason = FAIL_DECISION_NOT_GO
        warnings.append("W_DECISION_NOT_GO: generating fallback variants.")
    if decision == "LOW_CONF_GO":
        warnings.append("W_DECISION_LOW_CONF_GO: proceeding with caution.")

    if not unity_state:
        status = STATUS_DEGRADED_OK
        halt_reason = halt_reason or "FAIL_SHARED_STATE_MISSING"
        warnings.append("W_SHARED_STATE_MISSING: unity_state not provided; defaults applied.")

    resolved = job_card.get("resolved_target") or {}
    selected_bond = resolved.get("selected_bond") or {}
    if not selected_bond:
        status = STATUS_DEGRADED_OK
        halt_reason = halt_reason or FAIL_NO_TARGET
        warnings.append("W_MISSING_TARGET: proceeding with best-effort variants.")

    reaction_intent = job_card.get("reaction_intent") or {}
    intent_type = reaction_intent.get("intent_type")
    if intent_type not in {"hydrolysis", "deprotection", "fragment_generation", "reagent_generation"}:
        warnings.append("W_INTENT_UNRECOGNIZED: intent type not explicitly supported.")

    scaffold_rankings = _rank_scaffolds(top_scaffolds, job_card)
    selected = scaffold_rankings[0]
    selection_explain = _selection_explain(scaffold_rankings)
    route_primary = (job_card.get("mechanism_route") or {}).get("primary") or "unknown"
    unity_mechanism = (unity_state or {}).get("mechanism") or {}
    mechanism_contract = (
        unity_mechanism.get("contract")
        or job_card.get("mechanism_contract")
        or resolve_mechanism(route_primary).to_dict()
    )
    contract_geometry = _geometry_from_contract(mechanism_contract)
    expected_nucleophile = mechanism_contract.get("expected_nucleophile")
    candidate_residues_by_role = (
        selected.get("reach_summary") or {}
    ).get("candidate_residues_by_role") or {}
    observed_nucleophile = _detect_nucleophile(candidate_residues_by_role)
    mechanism_mismatch = _mechanism_mismatch_from_contract(
        mechanism_contract, observed_nucleophile
    )
    mechanism_evidence = _resolve_mechanism_evidence(
        mechanism_contract,
        candidate_residues_by_role,
        selected.get("pdb_path"),
    )
    if mechanism_evidence.get("status") == "UNVERIFIED" and mechanism_mismatch.get("status") == "OK":
        mechanism_mismatch["status"] = "UNVERIFIED"
        mechanism_mismatch["penalty_kj_mol"] = max(
            float(mechanism_mismatch.get("penalty_kj_mol") or 0.0),
            MECHANISM_UNVERIFIED_PENALTY_KJ,
        )
        mechanism_mismatch["confidence_penalty"] = MECHANISM_UNVERIFIED_CONFIDENCE_PENALTY
        mechanism_mismatch["explanation"] = (
            mechanism_evidence.get("explanation") or mechanism_mismatch.get("explanation")
        )
    allow_alt_nucleophiles = bool(
        job_card.get("hypothesis_exploration")
        or job_card.get("allow_alt_nucleophiles")
    )
    if not allow_alt_nucleophiles:
        cysteine_posterior = _route_posterior(job_card, "cysteine_hydrolase")
        if isinstance(cysteine_posterior, (int, float)) and cysteine_posterior >= ALT_NUCLEOPHILE_POSTERIOR_THRESHOLD:
            allow_alt_nucleophiles = True

    alternative_hypothesis = False
    if allow_alt_nucleophiles and observed_nucleophile and expected_nucleophile not in {None, "Either"}:
        if str(expected_nucleophile).lower() != str(observed_nucleophile).lower():
            alternative_hypothesis = True
            mechanism_mismatch["status"] = "ALTERNATIVE"
            mechanism_mismatch["penalty_kj_mol"] = 0.0
            mechanism_mismatch["policy"] = "REQUEST_DISAMBIGUATION"
            mechanism_mismatch["explanation"] = (
                f"Alternative hypothesis enabled: expected {expected_nucleophile}, observed {observed_nucleophile}."
            )

    mismatch_status = mechanism_mismatch.get("status")
    mismatch_penalty_kj = float(mechanism_mismatch.get("penalty_kj_mol") or 0.0)
    if mismatch_status == "MISMATCH":
        warnings.append(
            f"W_MECH_MISMATCH: {mechanism_mismatch.get('explanation')} (penalty {mismatch_penalty_kj} kJ/mol)."
        )
    if mismatch_status == "ALTERNATIVE":
        warnings.append(
            f"W_MECH_ALT_HYPOTHESIS: {mechanism_mismatch.get('explanation')}"
        )
    if mismatch_status == "UNVERIFIED":
        warnings.append(
            f"W_MECH_UNVERIFIED: {mechanism_mismatch.get('explanation')} (confidence penalty {MECHANISM_UNVERIFIED_CONFIDENCE_PENALTY:.2f})."
        )
    nucleophile_geometry = contract_geometry
    if isinstance(selected.get("reach_summary"), dict):
        reach_summary = dict(selected.get("reach_summary") or {})
        reach_summary["nucleophile_geometry"] = nucleophile_geometry
        selected["reach_summary"] = reach_summary

    evidence_status = mechanism_evidence.get("status")
    expected_label = {
        "Ser": "serine",
        "Cys": "cysteine",
        "MetalWater": "metal-activated water",
    }.get(str(expected_nucleophile), str(expected_nucleophile or "unknown"))
    primary_line = f"Primary hypothesis: {expected_label} nucleophile"
    alt_line = None
    if alternative_hypothesis:
        alt_label = {
            "Ser": "serine",
            "Cys": "cysteine",
            "MetalWater": "metal-activated water",
        }.get(str(observed_nucleophile), str(observed_nucleophile or "unknown"))
        alt_line = f"Alternative hypothesis tested: {alt_label} nucleophile (low weight)"

    mechanism_consistency_line = "Mechanism match: MISMATCH"
    if mismatch_status == "OK" and evidence_status == "VERIFIED":
        mechanism_consistency_line = "Mechanism match: OK"
    elif mismatch_status == "ALTERNATIVE":
        mechanism_consistency_line = "Mechanism match: ALTERNATIVE (hypothesis tracking)"
    elif mismatch_status == "UNVERIFIED" or evidence_status == "UNVERIFIED":
        mechanism_consistency_line = (
            f"Mechanism match: UNVERIFIED (penalty applied: {mismatch_penalty_kj} kJ/mol)"
            if mismatch_penalty_kj > 0.0
            else "Mechanism match: UNVERIFIED"
        )
    elif mismatch_penalty_kj > 0.0:
        mechanism_consistency_line = (
            f"Mechanism match: MISMATCH (penalty applied: {mismatch_penalty_kj} kJ/mol)"
        )

    if alt_line:
        mechanism_consistency_line = f"{primary_line}\n{alt_line}"
    else:
        mechanism_consistency_line = f"{primary_line}\n{mechanism_consistency_line}"

    mechanism_policy = str(job_card.get("mechanism_policy") or "exploratory").strip().lower()
    reaction_family = (job_card.get("chemistry_contract") or {}).get("reaction_family") or "unknown"
    mechanism_spec = _build_mechanism_spec(
        reaction_family=reaction_family,
        route_label=route_primary,
        candidate_residues_by_role=candidate_residues_by_role,
        mechanism_policy=mechanism_policy,
        expected_override=expected_nucleophile,
    )
    if alternative_hypothesis:
        mechanism_spec["policy_action"] = "REQUEST_DISAMBIGUATION"
    route_override = None
    route_prior_switched = None
    if mechanism_spec.get("policy_action") == "SWITCH_ROUTE":
        detected = mechanism_spec.get("detected_nucleophile")
        if detected == "Cys":
            route_override = "cysteine_hydrolase"
        elif detected == "Ser":
            route_override = "serine_hydrolase"
        if route_override and route_override != route_primary:
            mechanism_spec["route_label"] = route_override
            bond_context = job_card.get("bond_context") or {}
            bond_class = bond_context.get("bond_class") or bond_context.get("bond_type") or "unknown"
            condition_profile = job_card.get("condition_profile") or {}
            temp_k = condition_profile.get("temperature_K") or (
                float(condition_profile.get("temperature_C")) + 273.15
                if isinstance(condition_profile.get("temperature_C"), (int, float))
                else None
            )
            physics_audit = job_card.get("physics_audit") or {}
            horizon_s = physics_audit.get("horizon_s") or 3600.0
            if not isinstance(temp_k, (int, float)):
                temp_k = 298.15
            pH = condition_profile.get("pH")
            route_prior_switched = compute_route_prior(
                route_name=route_override,
                bond_class=str(bond_class),
                temperature_K=float(temp_k),
                horizon_s=float(horizon_s),
                pH=pH if isinstance(pH, (int, float)) else None,
                ionic_strength=condition_profile.get("ionic_strength"),
            )

    pocket_check = _pocket_reality_check(selected.get("pdb_path"))

    objective = "Improve retention without breaking access or reach."
    variant_set = _build_variants(job_card, selected, objective, pocket_check)
    if any(v.get("variant_policy") == "unknown_family_minimal" for v in variant_set):
        warnings.append("W_VARIANT_POLICY_UNKNOWN_FAMILY: no family mapping; using baseline-only variants.")
        warnings.append("W_NO_FAMILY_MAPPING_NO_MECHANISTIC_VARIANTS")
    variant_set = _apply_variant_energy_scoring(
        variant_set,
        selected,
        job_card,
        unity_state=unity_state,
        nucleophile_geometry=nucleophile_geometry,
        mismatch_penalty_kj=mismatch_penalty_kj,
        mismatch_policy=mechanism_mismatch.get("policy"),
    )
    best_variant_policy = "ranked_best_variant"
    best_variant = None
    if variant_set:
        best_variant = min(variant_set, key=lambda item: item.get("rank", 999))

    module3_handoff = _module3_handoff(
        job_card, selected, best_variant, variant_set, best_variant_policy, nucleophile_geometry
    )
    fork_enabled = os.environ.get("MECHANISM_FORK", "").strip().lower() in {"1", "true", "yes"}
    mechanism_tracks = None
    if mismatch_status == "MISMATCH" and mechanism_mismatch.get("policy") == "FORK_HYPOTHESES":
        mechanism_tracks = {
            "primary": {
                "route": route_primary,
                "variants": variant_set,
            }
        }
        if fork_enabled:
            alt_route = "cysteine_hydrolase" if observed_nucleophile == "Cys" else route_primary
            alt_job_card = dict(job_card)
            alt_mechanism = dict(job_card.get("mechanism_route") or {})
            alt_mechanism["primary"] = alt_route
            alt_job_card["mechanism_route"] = alt_mechanism
            alt_variants = _build_variants(alt_job_card, selected, objective, pocket_check)
            alt_contract = resolve_mechanism(alt_route).to_dict()
            alt_geometry = _geometry_from_contract(alt_contract)
            alt_variants = _apply_variant_energy_scoring(
                alt_variants,
                selected,
                alt_job_card,
                unity_state=unity_state,
                nucleophile_geometry=alt_geometry,
                mismatch_penalty_kj=0.0,
                mismatch_policy=alt_contract.get("mismatch_policy_default"),
            )
            mechanism_tracks["alternate"] = {
                "route": alt_route,
                "variants": alt_variants,
            }
    if alternative_hypothesis and mechanism_tracks is None:
        alt_route = "cysteine_hydrolase" if observed_nucleophile == "Cys" else route_primary
        mechanism_tracks = {
            "primary": {
                "route": route_primary,
                "variants": variant_set,
            },
            "alternate": {
                "route": alt_route,
                "note": "Alternative nucleophile hypothesis (low weight).",
            },
        }
    fallback_plan = (
        _fallback_plan(job_card, top_scaffolds, halt_reason) if status != "PASS" else None
    )
    design_loop = _design_loop(job_card, selected, variant_set, status)
    condition_assessment = _condition_assessment(job_card, selected)
    condition_search_summary = _condition_search_summary(job_card, condition_assessment)
    ph_locked_mutation_plan = _ph_locked_mutation_plan(job_card)
    confidence_calibration = _confidence_calibration(job_card, condition_assessment)
    mechanism_penalty_factor = 1.0
    if mechanism_spec.get("policy_action") == "REQUEST_DISAMBIGUATION":
        mechanism_penalty_factor = 0.6
        base_conf = float(confidence_calibration.get("calibrated_confidence") or 0.0)
        confidence_calibration["calibrated_confidence"] = round(
            max(0.0, min(1.0, base_conf * mechanism_penalty_factor)), 3
        )
        confidence_calibration["mechanism_penalty_factor"] = mechanism_penalty_factor
        confidence_calibration["mechanism_penalty_reason"] = "Mechanism mismatch; exploratory disambiguation required."
    learning_trace = _learning_trace(job_card, condition_assessment)
    candidate_reports = _candidate_reports(
        scaffold_rankings,
        job_card,
        condition_assessment,
        confidence_calibration,
        variant_set,
    )
    final_report = _final_report(job_card, candidate_reports, condition_assessment)

    predicted_under_given_conditions = _predicted_under_given_conditions(
        selected, condition_assessment, confidence_calibration
    )
    optimum_conditions_estimate = _optimum_conditions_estimate(job_card)
    delta_from_optimum = _delta_from_optimum(condition_assessment, optimum_conditions_estimate)
    evidence_record = EvidenceRecord(
        module_id=2,
        inputs={
            "conditions": condition_assessment.get("given_conditions"),
            "selected_scaffold": selected.get("scaffold_id") if isinstance(selected, dict) else None,
        },
        features_used=FeatureVector(
            values={
                "k_pred_mean": float(selected.get("k_pred_mean") or selected.get("k_pred") or 0.0),
                "delta_g_mean": float(selected.get("delta_g_mean") or selected.get("delta_g") or 0.0),
                "delta_g_std": float(selected.get("delta_g_std") or 0.0),
                "reliability": float(selected.get("reliability") or 0.0),
            },
            missing=[],
            source="module2",
        ),
        score=float(selected.get("adjusted_score") or 0.0),
        confidence=float(confidence_calibration.get("calibrated_confidence") or 0.0),
        uncertainty={"uncertainty": confidence_calibration.get("uncertainty")},
        optimum_conditions=optimum_conditions_estimate,
        explanations=[],
        diagnostics={"model": "eyring_v1"},
    ).to_dict()

    confidence_value = float(confidence_calibration.get("calibrated_confidence") or 0.0)
    uncertainty = float(confidence_calibration.get("uncertainty") or 0.0)
    confidence_estimate = ProbabilityEstimate(
        p_raw=confidence_value,
        p_cal=confidence_value,
        ci90=(max(0.0, confidence_value - uncertainty), min(1.0, confidence_value + uncertainty)),
        n_eff=2.0 + 20.0 * confidence_value,
    ).to_dict()
    prediction_estimates = {
        "k_pred": DistributionEstimate(
            mean=float(selected.get("k_pred_mean") or selected.get("k_pred") or 0.0),
            std=float(selected.get("k_pred_std") or 0.0),
            ci90=tuple(selected.get("k_pred_ci90") or [0.0, 0.0]),
        ).to_dict()
    }
    math_contract = {
        "confidence": confidence_estimate,
        "predictions": prediction_estimates,
        "qc": QCReport(status="N/A", reasons=[], metrics={}).to_dict(),
    }
    scorecard = _build_scorecard_module2(selected, confidence_calibration)

    module2_physics_audit = _module2_variant_physics_audit(
        selected,
        best_variant,
        job_card,
        condition_assessment,
        nucleophile_geometry=nucleophile_geometry,
        mismatch_penalty_kj=mismatch_penalty_kj,
        mismatch_policy=mechanism_mismatch.get("policy"),
    )
    docking_stability = _layer3_docking_and_stability(
        job_card,
        constraints=job_card.get("constraints") or {},
    )
    module2_physics_audit["docking"] = docking_stability.get("docking")
    module2_physics_audit["md_stability"] = docking_stability.get("md_stability")
    module2_physics_audit["composite_binding_score"] = docking_stability.get("composite_binding_score")
    module2_physics_audit["binding_score_adjustment"] = docking_stability.get("binding_score_adjustment")
    score_ledger = _build_score_ledger_module2(
        selected=selected,
        module2_physics_audit=module2_physics_audit,
        mechanism_mismatch=mechanism_mismatch,
    )

    payload = {
        "status": status,
        "halt_reason": halt_reason,
        "selected_scaffold": selected,
        "selection_explain": selection_explain,
        "objective": objective,
        "mechanism_policy": mechanism_policy,
        "mechanism_spec": mechanism_spec,
        "mechanism_contract": mechanism_contract,
        "mechanism_mismatch": mechanism_mismatch,
        "mechanism_evidence": mechanism_evidence,
        "mechanism_consistency": mechanism_consistency_line,
        "mechanism_tracks": mechanism_tracks,
        "mechanism_route_override": route_override,
        "mechanism_route_prior": route_prior_switched,
        "pocket_check": pocket_check,
        "variant_set": variant_set,
        "module3_handoff": module3_handoff,
        "best_variant_policy": best_variant_policy,
        "best_variant": best_variant,
        "fallback_plan": fallback_plan,
        "design_loop": design_loop,
        "condition_assessment": condition_assessment,
        "condition_search_summary": condition_search_summary,
        "ph_locked_mutation_plan": ph_locked_mutation_plan,
        "confidence_calibration": confidence_calibration,
        "learning_trace": learning_trace,
        "candidate_reports": candidate_reports,
        "final_report": final_report,
        "predicted_under_given_conditions": predicted_under_given_conditions,
        "optimum_conditions_estimate": optimum_conditions_estimate,
        "delta_from_optimum": delta_from_optimum,
        "confidence_calibrated": confidence_calibration.get("calibrated_confidence"),
        "module2_physics_audit": module2_physics_audit,
        "docking": docking_stability.get("docking"),
        "md_stability": docking_stability.get("md_stability"),
        "composite_binding_score": docking_stability.get("composite_binding_score"),
        "binding_score_adjustment": docking_stability.get("binding_score_adjustment"),
        "computational_engines_used": docking_stability.get("engines_used", []),
        "evidence_record": evidence_record,
        "scaffold_rankings": scaffold_rankings,
        "math_contract": math_contract,
        "scorecard": scorecard,
        "score_ledger": score_ledger,
        "warnings": warnings,
        "errors": errors,
    }
    contract_violations = validate_math_contract(payload)
    if contract_violations:
        payload["warnings"] = list(
            dict.fromkeys((payload.get("warnings") or []) + contract_violations)
        )
    record_event(
        {
            "module": "module2",
            "status": status,
            "selected_scaffold": selected.get("scaffold_id") if isinstance(selected, dict) else None,
            "k_pred_mean": selected.get("k_pred_mean") if isinstance(selected, dict) else None,
            "delta_g_mean": selected.get("delta_g_mean") if isinstance(selected, dict) else None,
            "given_conditions": condition_assessment.get("given_conditions"),
            "optimum_conditions": optimum_conditions_estimate,
        }
    )
    return payload


def _rank_scaffolds(
    scaffolds: List[Dict[str, Any]],
    job_card: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ranked = []
    difficulty = job_card.get("difficulty_label") or "MEDIUM"
    base_dg = {"EASY": 17.0, "MEDIUM": 21.0, "HARD": 27.0}.get(difficulty, 21.0)
    rcf = job_card.get("reaction_condition_field") or {}
    optimum_hint = rcf.get("optimum_conditions_hint") or {}
    given = dict(rcf.get("given_conditions") or {})
    condition_profile = job_card.get("condition_profile") or {}
    if given.get("pH") is None and condition_profile.get("pH") is not None:
        given["pH"] = condition_profile.get("pH")
    if given.get("temperature_c") is None and condition_profile.get("temperature_K") is not None:
        given["temperature_c"] = round(float(condition_profile["temperature_K"]) - 273.15, 1)
    ph = given.get("pH")
    temp_c = given.get("temperature_c")
    temp_k = c_to_k(float(temp_c)) if temp_c is not None else 298.15
    route_conf = (job_card.get("confidence") or {}).get("route", 0.5)
    if not isinstance(route_conf, (int, float)):
        route_conf = 0.5
    data_support = job_card.get("data_support")
    if not isinstance(data_support, (int, float)):
        data_support = (
            (job_card.get("causal_discovery") or {})
            .get("model_diagnostics", {})
            .get("data_support")
        )
    if not isinstance(data_support, (int, float)):
        data_support = 0.5

    for scaffold in scaffolds:
        scores = scaffold.get("scores") or {}
        retention_metrics = scaffold.get("retention_metrics") or {}
        total = scores.get("total") or 0.0
        penalty, penalty_reasons = _retention_penalty(retention_metrics)
        alignment_error = max(0.0, 1.0 - float(total))
        features = {
            "alignment_error": alignment_error,
            "retention_penalty": penalty,
            "steric_clash": max(0.0, 0.6 - float(total)),
            "electrostatic_mismatch": max(0.0, 0.5 - float(total)),
            "hbond_quality": min(1.0, float(total)),
            "difficulty_label": difficulty,
            "data_support": float(data_support),
        }
        base_dg_mean, base_dg_std = estimate_delta_g(base_dg, features)
        delta_g_mean = delta_g_with_conditions(base_dg_mean, ph, temp_k, optimum_hint)
        physics_gate = barrier_gate(delta_g_mean, temp_k)
        physics_gate["delta_g_kj_mol"] = round(float(delta_g_mean), 3)
        physics_gate["temperature_K"] = round(float(temp_k), 2)
        temp_c_for_sampling = temp_c if temp_c is not None else temp_k - 273.15
        seed = _scaffold_seed(scaffold.get("scaffold_id") or "unknown")
        k_stats = sample_k_pred(
            delta_g_mean,
            base_dg_std,
            temp_c_for_sampling,
            seed=seed,
        )
        k_samples = sample_k_pred(
            delta_g=delta_g_mean,
            temperature=temp_k,
            sigma=base_dg_std,
            n=256,
            rng=seed,
        )
        k_sample_mean = sum(k_samples) / len(k_samples) if k_samples else 0.0
        k_dist_summary = {
            "n": len(k_samples),
            "mean": round(k_sample_mean, 5),
            "p10": round(percentile(k_samples, 0.1), 5),
            "p50": round(percentile(k_samples, 0.5), 5),
            "p90": round(percentile(k_samples, 0.9), 5),
        }
        k_pred_mean = k_stats["mean"]
        k_pred_std = k_stats["std"]
        k_pred_ci90 = k_stats["ci90"]

        feasibility = scaffold.get("feasibility_score") or 0.5
        topology = scaffold.get("topology_feasibility_score") or 0.5
        reliability = (0.6 + 0.4 * float(route_conf)) * (
            0.5 + 0.3 * float(feasibility) + 0.2 * float(topology)
        )
        ranking_score = max(0.0, k_pred_mean) * reliability
        if physics_gate.get("ok") == 0.0:
            ranking_score *= PHYSICS_GATE_SCORE_MULTIPLIER
            penalty_reasons.append("PHYSICS_BARRIER_HIGH")
        model_risk = round(base_dg_std + float(penalty), 3)

        ranked.append(
            {
                "scaffold_id": scaffold.get("scaffold_id"),
                "pdb_path": scaffold.get("pdb_path"),
                "attack_envelope": scaffold.get("attack_envelope"),
                "candidate_residues_by_role": scaffold.get("candidate_residues_by_role"),
                "tunnel_metrics": scaffold.get("tunnel_metrics"),
                "scores": scores,
                "retention_metrics": retention_metrics,
                "reach_summary": scaffold.get("reach_summary") or {},
                "physics_gate": physics_gate,
                "delta_g": round(delta_g_mean, 3),
                "delta_g_mean": round(delta_g_mean, 3),
                "delta_g_std": round(base_dg_std, 3),
                "k_pred": round(k_pred_mean, 5),
                "k_pred_mean": round(k_pred_mean, 5),
                "k_pred_std": round(k_pred_std, 5),
                "k_pred_ci90": [round(k_pred_ci90[0], 5), round(k_pred_ci90[1], 5)],
                "k_dist_summary": k_dist_summary,
                "reliability": round(reliability, 3),
                "adjusted_score": round(ranking_score, 5),
                "retention_penalty": round(penalty, 3),
                "retention_penalty_reasons": penalty_reasons,
                "model_risk": model_risk,
            }
        )
    ranked.sort(key=lambda item: item.get("adjusted_score", 0.0), reverse=True)
    return ranked


def _retention_penalty(retention_metrics: Dict[str, Any]) -> tuple[float, List[str]]:
    risk = retention_metrics.get("retention_risk_flag") or "LOW"
    warning_codes = retention_metrics.get("warning_codes") or []
    penalty = 0.0
    reasons = []
    if risk == "HIGH":
        penalty += 0.08
        reasons.append("HIGH retention risk")
    elif risk == "MEDIUM":
        penalty += 0.04
        reasons.append("MEDIUM retention risk")
    if "WARN_RETENTION_WEAK_BINDING" in warning_codes:
        penalty += 0.05
        reasons.append("Weak-binding warning")
    return penalty, reasons


def _selection_explain(rankings: List[Dict[str, Any]]) -> str:
    if not rankings:
        return "No scaffold rankings available."
    top = rankings[0]
    explanation = [
        f"Selected {top.get('scaffold_id')} with ranking score {top.get('adjusted_score')}."
    ]
    if top.get("k_pred_mean") is not None:
        ci = top.get("k_pred_ci90") or []
        explanation.append(
            "Predicted k_mean={:.5f} (CI90 {:.5f}-{:.5f}) using Eyring ranking.".format(
                float(top.get("k_pred_mean") or 0.0),
                float(ci[0]) if len(ci) == 2 else 0.0,
                float(ci[1]) if len(ci) == 2 else 0.0,
            )
        )
    if top.get("retention_penalty_reasons"):
        explanation.append(
            f"Retention penalty applied ({', '.join(top['retention_penalty_reasons'])})."
        )
    return " ".join(explanation)


def _pocket_reality_check(pdb_path: Optional[str]) -> Dict[str, Any]:
    if not pdb_path:
        return {"status": "skipped", "reason": "No pdb_path provided."}
    path = Path(pdb_path)
    if not path.is_file():
        return {"status": "skipped", "reason": "PDB file not available in runtime."}
    return {"status": "available", "note": "PDB file present; geometry scan deferred in v1."}


def _annotate_variants_with_layer2_targets(
    variants: List[Dict[str, Any]],
    enzyme_family: Optional[str],
    pdb_id: Optional[str],
) -> List[Dict[str, Any]]:
    if not variants:
        return variants
    out: List[Dict[str, Any]] = []
    for variant in variants:
        item = dict(variant)
        targets = resolve_variant_targets(
            enzyme_family=enzyme_family,
            pdb_id=pdb_id,
            variant_label=item.get("label"),
        )
        if targets:
            item["residue_targets"] = targets
            item["residue_targets_source"] = "layer2_structure_db"
        out.append(item)
    return out


def _build_variants(
    job_card: Dict[str, Any],
    selected: Dict[str, Any],
    objective: str,
    pocket_check: Dict[str, Any],
) -> List[Dict[str, Any]]:
    variants: List[Variant] = []
    base_score = float(selected.get("adjusted_score") or 0.0)
    residues_by_role = selected.get("candidate_residues_by_role") or {}
    reach_summary = selected.get("reach_summary") or {}
    nucleophile_type = reach_summary.get("nucleophile_type")
    nucleophile_geometry = reach_summary.get("nucleophile_geometry")
    route = job_card.get("mechanism_route") or {}
    primary_route = route.get("primary")
    biology_audit = job_card.get("biology_audit") or {}
    biology_contract = job_card.get("biology_contract") or {}
    enzyme_family = (
        biology_audit.get("enzyme_family")
        or ((biology_contract.get("enzyme_family_prior") or {}).get("family"))
        or "unknown"
    )
    selected_pdb_id = (
        selected.get("pdb_id")
        or (selected.get("scaffold_metadata") or {}).get("pdb_id")
    )
    def _finalize(serialized_variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _annotate_variants_with_layer2_targets(
            serialized_variants,
            enzyme_family=enzyme_family,
            pdb_id=selected_pdb_id,
        )
    retention_metrics = selected.get("retention_metrics") or {}
    retention_flag = retention_metrics.get("retention_risk_flag")
    warning_codes = retention_metrics.get("warning_codes") or []
    tunnel_metrics = selected.get("tunnel_metrics") or {}
    long_tunnel = (tunnel_metrics.get("path_length") or 0.0) >= 18.0
    retention_weak = retention_flag in {"HIGH", "MEDIUM"} or "WARN_RETENTION_WEAK_BINDING" in warning_codes
    can_localize = pocket_check.get("status") == "available"
    causal = job_card.get("causal_discovery") or {}
    interventions = [item.get("action") for item in causal.get("interventions_topK") or []]

    variants.append(
        Variant(
            variant_id="V0",
            label="Baseline",
            description="No mutations; baseline control.",
            mutations=[],
            rationale="Used as control for downstream comparisons.",
            estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
            score=base_score,
            category="baseline",
            requires_structural_localization=False,
            delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ("baseline"),
            strain_displacement_A=0.0,
            strain_note="unknown -> no strain penalty applied",
        )
    )

    unknown_family_context = str(primary_route or "").lower() in {"unknown", "manual_review"} or str(
        enzyme_family or ""
    ).lower() in {"unknown", "manual_review", "n/a"}
    if unknown_family_context:
        if can_localize:
            variants.append(
                Variant(
                    variant_id="V_generic_pocket",
                    label="Generic pocket shaping",
                    description="Conservative pocket-lining mutation for broad compatibility.",
                    mutations=[
                        {
                            "target_role": "pocket_lining",
                            "preferred_residues": ["Leu", "Val", "Ala"],
                            "note": "generic packing adjustment without mechanism assumptions",
                        }
                    ],
                    rationale="No reliable family mapping; only propose low-risk generic adjustment.",
                    estimated_effects={"reach": 0.0, "retention": 0.04, "access": 0.0},
                    score=base_score,
                    category="baseline",
                    requires_structural_localization=False,
                    delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ("baseline"),
                    strain_displacement_A=0.0,
                    strain_note="unknown family fallback",
                )
            )
        serialized = [variant.__dict__ for variant in variants]
        for variant in serialized:
            variant["objective"] = objective
            variant["nucleophile_geometry"] = nucleophile_geometry
            variant["variant_policy"] = "unknown_family_minimal"
        return _finalize(serialized)

    if _is_radical_hat_context(job_card, str(primary_route or "")):
        variants.extend(_build_radical_hat_variants(base_score=base_score, can_localize=can_localize))
        serialized = [variant.__dict__ for variant in variants]
        for variant in serialized:
            variant["objective"] = objective
            variant["nucleophile_geometry"] = nucleophile_geometry
        return _finalize(serialized)

    if str(primary_route or "").lower() in {
        "haloalkane_dehalogenase",
        "reductive_dehalogenase",
        "glutathione_transferase",
    }:
        variants.extend(_build_dehalogenase_variants(base_score=base_score, can_localize=can_localize))
        serialized = [variant.__dict__ for variant in variants]
        for variant in serialized:
            variant["objective"] = objective
            variant["nucleophile_geometry"] = nucleophile_geometry
        return _finalize(serialized)

    if primary_route == "serine_hydrolase" and nucleophile_type == "Cys":
        cys_residue = _pick_residue(residues_by_role.get("nucleophile", []), ["Cys"])
        if cys_residue or not can_localize:
            mutations = (
                [{"residue": cys_residue, "to": "Ser"}]
                if cys_residue and can_localize
                else [
                    {
                        "target_role": "nucleophile",
                        "preferred_residues": ["Ser"],
                        "note": "align nucleophile with serine-hydrolase geometry",
                    }
                ]
            )
            variants.append(
                Variant(
                    variant_id="V1",
                    label="Mechanism alignment",
                    description="Align nucleophile identity with serine-hydrolase geometry.",
                    mutations=mutations,
                    rationale="Cys nucleophile detected in a serine-hydrolase route; align to Ser.",
                    estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
                    score=base_score,
                    category="mechanism_alignment",
                    requires_structural_localization=not can_localize,
                    delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ(
                        "mechanism_alignment"
                    ),
                    strain_displacement_A=0.0,
                    strain_note="unknown -> no strain penalty applied",
                )
            )

    if retention_weak:
        clamp_residue = _pick_residue(residues_by_role.get("other", []), [])
        clamp_mutations = (
            [{"residue": clamp_residue, "to": "Phe"}]
            if clamp_residue and can_localize
            else [
                {
                    "target_role": "pocket_lining",
                    "preferred_residues": ["Phe", "Tyr", "Leu", "Ile"],
                    "note": "hydrophobic clamp near aromatic ring",
                }
            ]
        )
        variants.append(
            Variant(
                variant_id="V2",
                label="Hydrophobic clamp",
                description="Add aromatic/hydrophobic packing to stabilize the aromatic ring.",
                mutations=clamp_mutations,
                rationale="Retention weakness; add hydrophobic wall for ring packing.",
                estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
                score=base_score,
                category="retention_clamp",
                requires_structural_localization=not can_localize,
                delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ("retention_clamp"),
                strain_displacement_A=0.0,
                strain_note="unknown -> no strain penalty applied",
            )
        )

        anchor_residue = _pick_residue(residues_by_role.get("other", []), [])
        anchor_mutations = (
            [{"residue": anchor_residue, "to": "Arg"}]
            if anchor_residue and can_localize
            else [
                {
                    "target_role": "polar_anchor",
                    "preferred_residues": ["Arg", "Lys", "His"],
                    "note": "anchor deprotonated carboxylate",
                }
            ]
        )
        variants.append(
            Variant(
                variant_id="V3",
                label="Polar anchor (carboxylate)",
                description="Introduce a positive anchor for the carboxylate.",
                mutations=anchor_mutations,
                rationale="Carboxylate is negative at pH 6.5–8; add a cationic anchor.",
                estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
                score=base_score,
                category="polar_anchor",
                requires_structural_localization=not can_localize,
                delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ("polar_anchor"),
                strain_displacement_A=0.0,
                strain_note="unknown -> no strain penalty applied",
            )
        )

        oxyanion_residue = _pick_residue(residues_by_role.get("other", []), [])
        oxyanion_mutations = (
            [{"residue": oxyanion_residue, "to": "Asn"}]
            if oxyanion_residue and can_localize
            else [
                {
                    "target_role": "oxyanion_hole",
                    "preferred_residues": ["Ser", "Thr", "Asn", "Gln"],
                    "note": "strengthen tetrahedral intermediate stabilization",
                }
            ]
        )
        variants.append(
            Variant(
                variant_id="V4",
                label="Oxyanion hole strengthening",
                description="Increase donor strength near the carbonyl oxygen.",
                mutations=oxyanion_mutations,
                rationale="Improve stabilization of the tetrahedral intermediate.",
                estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
                score=base_score,
                category="oxyanion_hole",
                requires_structural_localization=not can_localize,
                delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ("oxyanion_hole"),
                strain_displacement_A=0.0,
                strain_note="unknown -> no strain penalty applied",
            )
        )

    if retention_weak and long_tunnel:
        deep_residue = _pick_residue(residues_by_role.get("other", []), [])
        deep_mutations = (
            [{"residue": deep_residue, "to": "Tyr", "location": "deep_pocket"}]
            if deep_residue and can_localize
            else [
                {
                    "target_role": "deep_pocket_lining",
                    "preferred_residues": ["Tyr", "Phe"],
                    "note": "clamp deeper in pocket to preserve access",
                }
            ]
        )
        variants.append(
            Variant(
                variant_id="V5",
                label="Access-preserving clamp",
                description="Clamp deeper in pocket to avoid tunnel bottleneck.",
                mutations=deep_mutations,
                rationale="Long tunnel; improve retention without blocking access.",
                estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
                score=base_score,
                category="access_preserving_clamp",
                requires_structural_localization=not can_localize,
                delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ(
                    "access_preserving_clamp"
                ),
                strain_displacement_A=0.0,
                strain_note="unknown -> no strain penalty applied",
            )
        )

    ph_locked = _ph_locked_mutation_plan(job_card)
    if ph_locked.get("status") == "locked" or "optimize_pH" in interventions:
        variants.append(
            Variant(
                variant_id="V6",
                label="pH tuning bundle",
                description="Shift catalytic pKa via dyad/triad tuning.",
                mutations=[
                    {
                        "target_role": "general_base",
                        "preferred_residues": ["His", "Asp"],
                        "note": "add His-Asp dyad to tune pKa",
                    }
                ],
                rationale="pH tuning to activate catalytic residues under fixed conditions.",
                estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
                score=base_score,
                category="pH_tuning",
                requires_structural_localization=not can_localize,
                delta_deltaG_dagger_kJ_per_mol=estimate_barrier_shift_kJ("pH_tuning"),
                strain_displacement_A=0.0,
                strain_note="unknown -> no strain penalty applied",
            )
        )

    serialized = [variant.__dict__ for variant in variants]
    for variant in serialized:
        variant["objective"] = objective
        variant["nucleophile_geometry"] = nucleophile_geometry
    return _finalize(serialized)


def _pick_residue(residues: List[str], preferred_prefixes: List[str]) -> Optional[str]:
    if not residues:
        return None
    for prefix in preferred_prefixes:
        for residue in residues:
            if residue.lower().startswith(prefix.lower()):
                return residue
    return residues[0]


def _variant_rank_key(variant: Dict[str, Any], retention_weak: bool) -> float:
    return float(variant.get("score") or 0.0)


def _ddg_bind_kcal_for_variant(category: str) -> float:
    return float(DDG_BIND_KCAL_BY_CATEGORY.get(category, -0.2))


def _ddg_strain_kcal_for_variant(category: str) -> float:
    return float(DDG_STRAIN_KCAL_BY_CATEGORY.get(category, 0.2))


def _effects_from_ddg(
    ddg_total_kcal: float,
    ddg_strain_kcal: float,
) -> Dict[str, float]:
    retention_delta = max(-0.06, min(0.08, -0.04 * ddg_total_kcal))
    strain_penalty = max(0.0, ddg_strain_kcal)
    reach_delta = -min(0.05, 0.02 * strain_penalty)
    access_delta = -min(0.03, 0.01 * strain_penalty)
    return {
        "reach": round(reach_delta, 3),
        "retention": round(retention_delta, 3),
        "access": round(access_delta, 3),
    }


def _compute_variant_ddg(variant: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate variant-specific ΔΔG‡ from mutation roles with shell attenuation."""
    mutations = variant.get("mutations") or []
    if not mutations:
        return {
            "ddg_kj_mol": 0.0,
            "retention_boost": 0.0,
            "confidence": 1.0,
            "components": [],
            "dominant_mechanism": "baseline_control",
        }

    components: List[Dict[str, Any]] = []
    total_ddg = 0.0
    total_retention = 0.0
    confidences: List[float] = []
    diminishing = 1.0
    for mutation in mutations:
        role = str(mutation.get("target_role") or "unknown")
        model = VARIANT_DDG_MODELS.get(
            role,
            {
                "base_ddg_kj": -1.0,
                "distance_shell": 2,
                "confidence": 0.2,
                "retention_boost": 0.0,
                "mechanism": "fallback mutation prior",
            },
        )
        shell = int(model.get("distance_shell") or 2)
        attenuation = float(SHELL_ATTENUATION.get(shell, 0.25))
        ddg_component = float(model.get("base_ddg_kj") or 0.0) * attenuation * diminishing
        retention_component = (
            float(model.get("retention_boost") or 0.0) * diminishing
        )
        components.append(
            {
                "role": role,
                "shell": shell,
                "attenuation": round(attenuation, 3),
                "diminishing_factor": round(diminishing, 3),
                "ddg_contribution_kj": round(ddg_component, 3),
                "retention_contribution": round(retention_component, 3),
                "mechanism": model.get("mechanism"),
            }
        )
        total_ddg += ddg_component
        total_retention += retention_component
        confidences.append(float(model.get("confidence") or 0.2))
        diminishing *= 0.7

    geom_conf = 0.2
    if confidences:
        geom_conf = math.exp(sum(math.log(max(1e-6, c)) for c in confidences) / len(confidences))
    dominant = max(components, key=lambda item: abs(float(item.get("ddg_contribution_kj") or 0.0)))
    return {
        "ddg_kj_mol": round(total_ddg, 3),
        "retention_boost": round(total_retention, 3),
        "confidence": round(float(geom_conf), 3),
        "components": components,
        "dominant_mechanism": dominant.get("mechanism"),
    }


def _diffusion_cap_from_path(path_length_A: Optional[float]) -> Dict[str, Optional[float]]:
    if not isinstance(path_length_A, (int, float)) or path_length_A <= 0.0:
        return {
            "k_diff_cap_s_inv": None,
            "D_m2_s": DEFAULT_DIFFUSION_COEFF_M2_S,
            "L_A": None,
            "tau_s": None,
            "assumptions": ["path length unavailable; diffusion cap not applied"],
        }
    length_m = float(path_length_A) * 1e-10
    D = DEFAULT_DIFFUSION_COEFF_M2_S
    tau = (length_m**2) / max(2.0 * D, 1e-30)
    tau = max(float(tau), 1e-12)
    k_diff = 1.0 / tau
    k_diff = min(k_diff, 1e9)
    return {
        "k_diff_cap_s_inv": float(k_diff),
        "D_m2_s": D,
        "L_A": float(path_length_A),
        "tau_s": float(tau),
        "assumptions": ["diffusion cap from tunnel path length"],
    }


def _protonation_factor_for_route(
    route_name: str,
    nucleophile_geometry: Optional[str],
    pH: Optional[float],
) -> Dict[str, Any]:
    if pH is None:
        return {
            "factor": 0.5,
            "residue": None,
            "notes": ["pH unknown; default protonation factor 0.5"],
            "uncertain": True,
        }
    route_label = str(route_name or "unknown").lower()
    geom_label = str(nucleophile_geometry or "").lower()
    residue = "His"
    mode = "base"
    if "cysteine" in geom_label or "thiol" in geom_label or "cys" in route_label:
        residue = "Cys"
        mode = "base"
    elif "acid" in route_label or "aspart" in route_label or "glu" in route_label:
        residue = "Asp"
        mode = "acid"
    pka = PKA_CATALYTIC_GROUPS.get(residue)
    if pka is None:
        return {
            "factor": 0.5,
            "residue": residue,
            "notes": [f"pKa unknown for {residue}; default 0.5"],
            "uncertain": True,
        }
    if mode == "acid":
        factor = fraction_protonated(float(pH), float(pka))
    else:
        factor = fraction_deprotonated(float(pH), float(pka))
    return {
        "factor": max(0.0, min(1.0, float(factor))),
        "residue": residue,
        "notes": [f"{residue} {mode} state from pH/pKa"],
        "uncertain": False,
    }


def _solvent_penalty_from_profile(condition_profile: Dict[str, Any]) -> Dict[str, Any]:
    penalty_info = solvent_penalty(condition_profile.get("solvent"))
    return {
        "penalty": penalty_info.get("penalty", 0.7),
        "solvent_unknown": penalty_info.get("solvent_unknown", True),
        "note": penalty_info.get("note"),
    }


def _is_radical_hat_context(job_card: Dict[str, Any], primary_route: str) -> bool:
    route_l = str(primary_route or "").lower()
    reaction_family = str(
        (job_card.get("chemistry_contract") or {}).get("reaction_family")
        or (job_card.get("reaction_intent") or {}).get("reaction_family")
        or ""
    ).lower()
    mechanisms = (job_card.get("mechanism_route") or {}).get("mechanisms") or []
    mechanisms_l = " ".join(str(m).lower() for m in mechanisms)
    markers = [
        "c-h activation",
        "radical_rebound",
        "metal_radical_transfer",
        "hydrogen_atom_abstraction",
        "metallo_transfer_cf3",
        "p450",
        "radical",
    ]
    text = f"{route_l} {reaction_family} {mechanisms_l}"
    return any(marker in text for marker in markers)


def _build_radical_hat_variants(
    base_score: float,
    can_localize: bool,
) -> List[Variant]:
    return [
        Variant(
            variant_id="V_metal_opt",
            label="Metal center optimization",
            description="Tune first-shell ligands to favor high-valent metal-oxo species.",
            mutations=[
                {
                    "target_role": "metal_first_shell",
                    "preferred_residues": ["His", "Glu", "Asp"],
                    "note": "2-His-1-carboxylate style motif for non-heme Fe chemistry",
                }
            ],
            rationale="C-H activation requires high-valent oxidants; first-shell ligands tune redox potential.",
            estimated_effects={"reach": 0.0, "retention": 0.0, "access": 0.0},
            score=base_score,
            category="radical_metal_opt",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-8.0,
            strain_displacement_A=0.0,
            strain_note="radical route prior",
        ),
        Variant(
            variant_id="V_tunnel",
            label="Substrate positioning tunnel",
            description="Engineer a tunnel and gate to orient target C-H toward the reactive center.",
            mutations=[
                {
                    "target_role": "substrate_channel",
                    "preferred_residues": ["Phe", "Trp", "Leu"],
                    "note": "hydrophobic channel lining for small gas-like substrates",
                },
                {
                    "target_role": "substrate_gate",
                    "preferred_residues": ["Phe", "Tyr"],
                    "note": "gate residue to increase retention near the catalyst",
                },
            ],
            rationale="Positioning dominates for small substrates; channel engineering reduces entropic penalty.",
            estimated_effects={"reach": 0.05, "retention": 0.15, "access": -0.02},
            score=base_score,
            category="radical_tunnel",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-4.0,
            strain_displacement_A=0.0,
            strain_note="radical route prior",
        ),
        Variant(
            variant_id="V_rebound",
            label="Radical rebound cage",
            description="Constrain radical escape and favor rebound before side reactions.",
            mutations=[
                {
                    "target_role": "radical_cage",
                    "preferred_residues": ["Val", "Ile", "Leu", "Ala"],
                    "note": "tight hydrophobic cage around radical intermediate",
                }
            ],
            rationale="A compact cage reduces radical diffusion and improves rebound probability.",
            estimated_effects={"reach": 0.0, "retention": 0.08, "access": -0.01},
            score=base_score,
            category="radical_rebound",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-3.0,
            strain_displacement_A=0.0,
            strain_note="radical route prior",
        ),
        Variant(
            variant_id="V_electron_transfer",
            label="Electron transfer tuning",
            description="Tune second-shell electrostatics to stabilize high-valent intermediates.",
            mutations=[
                {
                    "target_role": "second_shell_electrostatics",
                    "preferred_residues": ["Arg", "Lys", "Thr"],
                    "note": "electrostatic stabilization of reactive oxidant",
                }
            ],
            rationale="Second-shell charges can stabilize reactive species and extend catalytic window.",
            estimated_effects={"reach": 0.0, "retention": 0.02, "access": 0.0},
            score=base_score,
            category="radical_electron_transfer",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-2.0,
            strain_displacement_A=0.0,
            strain_note="radical route prior",
        ),
    ]


def _build_dehalogenase_variants(
    base_score: float,
    can_localize: bool,
) -> List[Variant]:
    return [
        Variant(
            variant_id="V_sn2_nucleophile_align",
            label="SN2 nucleophile alignment",
            description="Align catalytic nucleophile (Asp/Glu) for backside attack on alkyl halide.",
            mutations=[
                {
                    "target_role": "nucleophile",
                    "preferred_residues": ["Asp", "Glu"],
                    "note": "position carboxylate for SN2 displacement trajectory",
                }
            ],
            rationale="Dehalogenases require nucleophile alignment for productive substitution.",
            estimated_effects={"reach": 0.04, "retention": 0.04, "access": 0.0},
            score=base_score,
            category="mechanism_alignment",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-4.0,
            strain_displacement_A=0.0,
            strain_note="dehalogenase route prior",
        ),
        Variant(
            variant_id="V_halide_stabilization",
            label="Halide stabilization pocket",
            description="Add halide-stabilizing residues near leaving-group trajectory.",
            mutations=[
                {
                    "target_role": "halide_stabilization",
                    "preferred_residues": ["Trp", "Asn", "His"],
                    "note": "stabilize leaving halide and reduce rebound penalty",
                }
            ],
            rationale="Leaving-group stabilization improves displacement efficiency.",
            estimated_effects={"reach": 0.0, "retention": 0.05, "access": 0.0},
            score=base_score,
            category="polar_anchor",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-2.5,
            strain_displacement_A=0.0,
            strain_note="dehalogenase route prior",
        ),
        Variant(
            variant_id="V_cap_tunnel",
            label="Cap-domain tunnel tuning",
            description="Tune cap-domain/tunnel residues to retain small halogenated substrates.",
            mutations=[
                {
                    "target_role": "substrate_channel",
                    "preferred_residues": ["Phe", "Leu", "Val"],
                    "note": "improve access-retention balance for volatile alkyl halides",
                }
            ],
            rationale="Access/retention tradeoff is critical for haloalkane substrates.",
            estimated_effects={"reach": 0.03, "retention": 0.1, "access": -0.01},
            score=base_score,
            category="access_preserving_clamp",
            requires_structural_localization=not can_localize,
            delta_deltaG_dagger_kJ_per_mol=-1.2,
            strain_displacement_A=0.0,
            strain_note="dehalogenase route prior",
        ),
    ]


def _apply_variant_energy_scoring(
    variants: List[Dict[str, Any]],
    selected: Dict[str, Any],
    job_card: Dict[str, Any],
    unity_state: Optional[Dict[str, Any]] = None,
    nucleophile_geometry: Optional[str] = None,
    mismatch_penalty_kj: float = 0.0,
    mismatch_policy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not variants:
        return variants
    baseline_kcal = float(selected.get("delta_g_mean") or selected.get("delta_g") or 0.0)
    baseline_kj = baseline_kcal * KCAL_TO_KJ
    rcf = job_card.get("reaction_condition_field") or {}
    given = dict(rcf.get("given_conditions") or {})
    condition_profile = job_card.get("condition_profile") or {}
    route_primary = (job_card.get("mechanism_route") or {}).get("primary") or "unknown"
    temp_k = None
    profile_temp_k = condition_profile.get("temperature_K")
    if isinstance(profile_temp_k, (int, float)):
        temp_k = float(profile_temp_k)
    else:
        temp_c = given.get("temperature_c")
        if isinstance(temp_c, (int, float)):
            temp_k = c_to_k(float(temp_c))
    if temp_k is None:
        temp_k = 298.15
    pH = given.get("pH")
    if pH is None:
        pH = condition_profile.get("pH")
    if nucleophile_geometry is None:
        nucleophile_geometry = (selected.get("reach_summary", {}) or {}).get(
            "nucleophile_geometry"
        )
    diffusion_info = _diffusion_cap_from_path(
        (selected.get("tunnel_metrics") or {}).get("path_length")
        or (selected.get("tunnel_summary") or {}).get("path_length")
    )
    solvent_info = _solvent_penalty_from_profile(condition_profile)
    protonation_info = _protonation_factor_for_route(
        route_primary,
        nucleophile_geometry,
        pH,
    )
    protonation_factor = float(protonation_info.get("factor") or 0.5)
    unity_bio = (unity_state or {}).get("bio") or {}
    unity_prot = unity_bio.get("protonation") or {}
    unity_factor = unity_prot.get("factor") if isinstance(unity_prot, dict) else None
    if isinstance(unity_factor, (int, float)):
        protonation_factor = float(unity_factor)
        protonation_info["factor"] = protonation_factor
        protonation_info["notes"] = (protonation_info.get("notes") or []) + [
            "unity_protonation_factor"
        ]
    base_residue = None
    if isinstance(unity_prot, dict) and unity_prot.get("residue"):
        base_residue = str(unity_prot.get("residue")).strip().title()
    if base_residue is None:
        candidate_roles = (selected.get("reach_summary") or {}).get(
            "candidate_residues_by_role"
        ) or {}
        base_candidates = candidate_roles.get("base") or []
        if base_candidates:
            base_residue = str(base_candidates[0])[:3].title()
    if base_residue:
        base_fraction = residue_state_fraction(pH, base_residue)
        protonation_factor = max(0.05, min(1.0, protonation_factor * base_fraction))
        protonation_info["factor"] = protonation_factor
        protonation_info["residue"] = base_residue
        protonation_info["notes"] = (protonation_info.get("notes") or []) + [
            f"base_residue_fraction {base_fraction:.2f}"
        ]
    bio_contract = job_card.get("biology_contract") or {}
    residue_fraction_info = unity_bio.get("residue_protonation_fraction")
    if residue_fraction_info is None:
        residue_fraction_info = bio_contract.get("residue_protonation_fraction")
    residue_fraction = None
    if isinstance(residue_fraction_info, dict):
        residue_fraction = residue_fraction_info.get("fraction")
    elif isinstance(residue_fraction_info, (int, float)):
        residue_fraction = residue_fraction_info
    if isinstance(residue_fraction, (int, float)):
        protonation_factor = max(0.05, min(1.0, protonation_factor * float(residue_fraction)))
        protonation_info["factor"] = protonation_factor
        protonation_info["notes"] = (protonation_info.get("notes") or []) + [
            "unity_residue_protonation_fraction"
        ]
    solvent_pen = float(solvent_info.get("penalty") or 0.7)
    k_diff_cap = diffusion_info.get("k_diff_cap_s_inv")
    mismatch_penalty = float(mismatch_penalty_kj or 0.0)
    strain_floor_meta = _layer3_strain_floor(job_card, default_kj=STRAIN_FLOOR_KJ_PER_MOL)
    strain_floor_value = float(strain_floor_meta.get("strain_kj_mol") or STRAIN_FLOOR_KJ_PER_MOL)

    for variant in variants:
        assumptions: List[str] = []
        ddg_prediction = _compute_variant_ddg(variant)
        variant["physics_prediction"] = ddg_prediction
        explicit_ddg_raw = variant.get("delta_deltaG_dagger_kJ_per_mol")
        ddg_model_value = ddg_prediction.get("ddg_kj_mol")
        if (
            isinstance(explicit_ddg_raw, (int, float))
            and not (variant.get("mutations") or [])
        ):
            delta_delta_raw = float(explicit_ddg_raw)
        else:
            delta_delta_raw = (
                float(ddg_model_value)
                if isinstance(ddg_model_value, (int, float))
                else float(explicit_ddg_raw or 0.0)
            )
        delta_delta_clamped = max(-10.0, min(6.0, float(delta_delta_raw)))
        clamp_applied = not math.isclose(delta_delta_raw, delta_delta_clamped, rel_tol=0.0, abs_tol=1e-9)
        delta_delta = float(delta_delta_clamped)
        displacement = variant.get("strain_displacement_A")
        explicit_strain = isinstance(displacement, (int, float)) and float(displacement) != 0.0
        if not explicit_strain:
            displacement = 0.0
        strain_energy = 0.5 * STRAIN_K_MODEL_KJ_PER_MOL_A2 * float(displacement) ** 2
        strain_energy = max(0.0, float(strain_energy))
        delta_delta_effective = delta_delta
        strain_note = variant.get("strain_note") or "unknown -> no strain penalty applied"
        if variant.get("requires_structural_localization") and not explicit_strain and strain_energy <= 0.0:
            strain_energy = max(0.0, float(strain_floor_value))
            source = str(strain_floor_meta.get("source") or "default_floor")
            status = str(strain_floor_meta.get("status") or "unknown")
            strain_note = (
                "structural localization required; "
                f"strain floor applied from {source} ({status})"
            )
            assumptions.append(f"strain floor applied for structural localization ({source}:{status})")
        if clamp_applied:
            assumptions.append("delta_deltaG_dagger clamped to [-10, 6] kJ/mol")
        if strain_energy >= STRAIN_THRESHOLD_KJ_PER_MOL and delta_delta < 0.0:
            delta_delta_effective = 0.0
            strain_note = "strain cancels benefit"
            assumptions.append("strain cancels barrier benefit")
        deltaG_variant_kj = baseline_kj + delta_delta_effective + strain_energy
        if mismatch_penalty > 0.0 and str(mismatch_policy or "KEEP_WITH_PENALTY").upper() == "KEEP_WITH_PENALTY":
            deltaG_variant_kj += mismatch_penalty
            assumptions.append("mechanism mismatch penalty applied to barrier")
        k_variant = eyring_rate_constant(kj_to_j(deltaG_variant_kj), temp_k)
        k_eff = float(k_variant)
        if isinstance(k_diff_cap, (int, float)):
            k_eff = min(k_eff, float(k_diff_cap))
        k_eff *= max(0.0, protonation_factor)
        k_eff *= max(0.0, solvent_pen)

        target_residue = None
        for mutation in variant.get("mutations") or []:
            if mutation.get("target_role") == "nucleophile":
                preferred = mutation.get("preferred_residues") or []
                if preferred:
                    target_residue = preferred[0]
                    break
        change_gate = nucleophile_change_penalty(nucleophile_geometry, target_residue)
        change_penalty = float(change_gate.get("penalty") or 0.0)
        if change_penalty > 0.0:
            k_eff *= max(0.0, 1.0 - change_penalty)
            assumptions.append("nucleophile change gate applied")

        category = variant.get("category") or "baseline"
        ddg_bind_kcal = _ddg_bind_kcal_for_variant(category)
        ddg_strain_kcal = (
            float(strain_energy) / KCAL_TO_KJ
            if strain_energy > 0
            else _ddg_strain_kcal_for_variant(category)
        )
        if strain_energy > 0:
            assumptions.append("strain from displacement proxy")
        ddg_total_kcal = ddg_bind_kcal + ddg_strain_kcal
        rate_ratio = math.exp(-ddg_total_kcal / max(1e-9, R_KCAL_PER_MOLK * temp_k))
        if ddg_bind_kcal != 0.0:
            assumptions.append("ddG_bind from coarse category prior")
        if ddg_strain_kcal != 0.0:
            assumptions.append("ddG_strain from coarse category prior")
        variant["physics"] = {
            "ddG_bind_kcal_mol": round(ddg_bind_kcal, 3),
            "ddG_strain_kcal_mol": round(ddg_strain_kcal, 3),
            "ddG_total_kcal_mol": round(ddg_total_kcal, 3),
            "rate_ratio": round(float(rate_ratio), 3),
            "assumptions": list(dict.fromkeys(assumptions)),
            "variant_ddg_model": ddg_prediction,
        }
        estimated_effects = _effects_from_ddg(ddg_total_kcal, ddg_strain_kcal)
        retention_boost = float(ddg_prediction.get("retention_boost") or 0.0)
        if retention_boost:
            estimated_effects["retention"] = round(
                max(-0.5, min(0.5, float(estimated_effects.get("retention", 0.0)) + retention_boost)),
                3,
            )
        variant["estimated_effects"] = estimated_effects

        estimated_effects = variant.get("estimated_effects") or {}
        variant["strain_energy_kJ_per_mol"] = round(float(strain_energy), 3)
        variant["strain_kj_mol"] = round(float(strain_energy), 3)
        variant["strain_floor_source"] = strain_floor_meta.get("source")
        variant["strain_floor_status"] = strain_floor_meta.get("status")
        variant["delta_deltaG_dagger_effective_kJ_per_mol"] = round(
            float(delta_delta_effective), 3
        )
        variant["delta_deltaG_dagger_raw_kJ_per_mol"] = round(float(delta_delta_raw), 3)
        variant["delta_deltaG_dagger_clamped"] = bool(clamp_applied)
        variant["delta_deltaG_category_used"] = variant.get("category") or "unknown"
        variant["delta_deltaG_dagger_kj_mol"] = round(float(delta_delta_effective), 3)
        variant["deltaG_dagger_variant_kJ_per_mol"] = round(float(deltaG_variant_kj), 3)
        variant["deltaG_dagger_variant_kj_mol"] = round(float(deltaG_variant_kj), 3)
        variant["k_variant_s_inv"] = round(float(k_variant), 6)
        variant["k_eyring_s_inv"] = round(float(k_variant), 6)
        variant["k_diff_cap_s_inv"] = (
            round(float(k_diff_cap), 6) if isinstance(k_diff_cap, (int, float)) else None
        )
        variant["f_protonation"] = round(float(protonation_factor), 3)
        variant["solvent_penalty"] = round(float(solvent_pen), 3)
        variant["k_eff_s_inv"] = round(float(k_eff), 6)
        variant["mechanism_mismatch_penalty_kJ_per_mol"] = round(float(mismatch_penalty), 3)
        variant["nucleophile_change_penalty"] = round(float(change_penalty), 3)
        variant["nucleophile_change_reason"] = change_gate.get("reason")
        variant["protonation_residue"] = protonation_info.get("residue")
        variant["protonation_notes"] = protonation_info.get("notes")
        variant["protonation_uncertain"] = protonation_info.get("uncertain")
        variant["solvent_penalty_note"] = solvent_info.get("note")
        variant["strain_note"] = strain_note
        variant["access_penalty"] = round(max(0.0, -float(estimated_effects.get("access", 0.0))), 3)
        variant["reach_penalty"] = round(max(0.0, -float(estimated_effects.get("reach", 0.0))), 3)
        variant["retention_penalty"] = round(
            max(0.0, -float(estimated_effects.get("retention", 0.0))), 3
        )

    log_k_values = [
        math.log10(
            max(
                float(variant.get("k_eff_s_inv") or variant.get("k_variant_s_inv") or 0.0),
                1e-300,
            )
        )
        for variant in variants
    ]
    if log_k_values:
        min_log_k = min(log_k_values)
        max_log_k = max(log_k_values)
    else:
        min_log_k = 0.0
        max_log_k = 0.0
    score_range = max(1e-9, max_log_k - min_log_k)
    for variant in variants:
        log_k = math.log10(
            max(
                float(variant.get("k_eff_s_inv") or variant.get("k_variant_s_inv") or 0.0),
                1e-300,
            )
        )
        score = (log_k - min_log_k) / score_range
        score = max(0.0, min(1.0, float(score)))
        variant["score"] = round(float(score), 3)
        variant["score_audit"] = {
            "method": "minmax_log10_k",
            "log_k_min": round(float(min_log_k), 3),
            "log_k_max": round(float(max_log_k), 3),
        }

    def _tie_breaker_score(variant: Dict[str, Any]) -> float:
        effects = variant.get("estimated_effects") or {}
        return (
            float(effects.get("access", 0.0))
            + float(effects.get("reach", 0.0))
            + float(effects.get("retention", 0.0))
        )

    variants_sorted = sorted(
        variants,
        key=lambda item: float(item.get("k_eff_s_inv") or item.get("k_variant_s_inv") or 0.0),
        reverse=True,
    )
    ranked: List[Dict[str, Any]] = []
    group: List[Dict[str, Any]] = []
    group_max = None
    for variant in variants_sorted:
        k_val = float(variant.get("k_eff_s_inv") or variant.get("k_variant_s_inv") or 0.0)
        if group_max is None:
            group_max = k_val
            group = [variant]
            continue
        if k_val >= 0.95 * float(group_max):
            group.append(variant)
            continue
        group.sort(key=_tie_breaker_score, reverse=True)
        ranked.extend(group)
        group = [variant]
        group_max = k_val
    if group:
        group.sort(key=_tie_breaker_score, reverse=True)
        ranked.extend(group)

    for idx, variant in enumerate(ranked, start=1):
        variant["rank"] = idx
    return ranked


def _module2_variant_physics_audit(
    selected: Dict[str, Any],
    best_variant: Optional[Dict[str, Any]],
    job_card: Dict[str, Any],
    condition_assessment: Dict[str, Any],
    nucleophile_geometry: Optional[str] = None,
    mismatch_penalty_kj: float = 0.0,
    mismatch_policy: Optional[str] = None,
) -> Dict[str, Any]:
    energy_ledger = job_card.get("energy_ledger") or {}
    baseline_source = "scaffold_estimate"
    baseline_kj = None
    if isinstance(energy_ledger.get("deltaG_dagger_kJ"), (int, float)):
        baseline_kj = float(energy_ledger.get("deltaG_dagger_kJ"))
        baseline_source = "energy_ledger"
    if baseline_kj is None:
        baseline_kcal = float(selected.get("delta_g_mean") or selected.get("delta_g") or 0.0)
        baseline_kj = baseline_kcal * KCAL_TO_KJ
    given = (condition_assessment.get("given_conditions") or {})
    temp_k = None
    profile = job_card.get("condition_profile") or {}
    profile_temp_k = profile.get("temperature_K")
    if isinstance(profile_temp_k, (int, float)):
        temp_k = float(profile_temp_k)
    else:
        temp_c = given.get("temperature_c")
        if temp_c is None:
            temp_c = profile.get("temperature_C")
        if isinstance(temp_c, (int, float)):
            temp_k = c_to_k(float(temp_c))
    if temp_k is None:
        temp_k = 298.15
    route_primary = (job_card.get("mechanism_route") or {}).get("primary") or "unknown"
    pH = given.get("pH")
    if pH is None:
        pH = profile.get("pH")
    if nucleophile_geometry is None:
        nucleophile_geometry = (selected.get("reach_summary", {}) or {}).get("nucleophile_geometry")
    diffusion_info = _diffusion_cap_from_path(
        (selected.get("tunnel_metrics") or {}).get("path_length")
        or (selected.get("tunnel_summary") or {}).get("path_length")
    )
    solvent_info = _solvent_penalty_from_profile(profile)
    protonation_info = _protonation_factor_for_route(route_primary, nucleophile_geometry, pH)
    protonation_factor = float(protonation_info.get("factor") or 0.5)
    bio_contract = job_card.get("biology_contract") or {}
    residue_fraction_info = bio_contract.get("residue_protonation_fraction")
    residue_fraction = None
    if isinstance(residue_fraction_info, dict):
        residue_fraction = residue_fraction_info.get("fraction")
    elif isinstance(residue_fraction_info, (int, float)):
        residue_fraction = residue_fraction_info
    if isinstance(residue_fraction, (int, float)):
        protonation_factor = max(0.05, min(1.0, protonation_factor * float(residue_fraction)))
        protonation_info["factor"] = protonation_factor
        protonation_info["notes"] = (protonation_info.get("notes") or []) + [
            "biology_contract_residue_fraction"
        ]
    solvent_pen = float(solvent_info.get("penalty") or 0.7)
    k_diff_cap = diffusion_info.get("k_diff_cap_s_inv")
    delta_delta = 0.0
    strain_energy = 0.0
    strain_floor_meta = _layer3_strain_floor(job_card, default_kj=STRAIN_FLOOR_KJ_PER_MOL)
    strain_floor_value = float(strain_floor_meta.get("strain_kj_mol") or STRAIN_FLOOR_KJ_PER_MOL)
    delta_delta_category = None
    delta_delta_clamped = None
    if best_variant:
        delta_delta = float(
            best_variant.get("delta_deltaG_dagger_effective_kJ_per_mol")
            or best_variant.get("delta_deltaG_dagger_kJ_per_mol")
            or 0.0
        )
        delta_delta_category = best_variant.get("delta_deltaG_category_used") or best_variant.get(
            "category"
        )
        delta_delta_clamped = best_variant.get("delta_deltaG_dagger_clamped")
        strain_energy = max(0.0, float(best_variant.get("strain_energy_kJ_per_mol") or 0.0))
        if best_variant.get("requires_structural_localization") and strain_energy <= 0.0:
            strain_energy = max(0.0, float(strain_floor_value))
    deltaG_variant_kj = baseline_kj + delta_delta + strain_energy
    if mismatch_penalty_kj > 0.0 and str(mismatch_policy or "KEEP_WITH_PENALTY").upper() == "KEEP_WITH_PENALTY":
        deltaG_variant_kj += mismatch_penalty_kj
    k_variant = eyring_rate_constant(kj_to_j(deltaG_variant_kj), float(temp_k))
    k_eff = float(k_variant)
    if isinstance(k_diff_cap, (int, float)):
        k_eff = min(k_eff, float(k_diff_cap))
    k_eff *= max(0.0, protonation_factor)
    k_eff *= max(0.0, solvent_pen)
    return {
        "temperature_K": round(float(temp_k), 2),
        "baseline_source": baseline_source,
        "deltaG_dagger_baseline_kJ_per_mol": round(float(baseline_kj), 3),
        "delta_deltaG_dagger_kJ_per_mol": round(float(delta_delta), 3),
        "strain_energy_kJ_per_mol": round(float(strain_energy), 3),
        "strain_floor_source": strain_floor_meta.get("source"),
        "strain_floor_status": strain_floor_meta.get("status"),
        "deltaG_dagger_variant_kJ_per_mol": round(float(deltaG_variant_kj), 3),
        "delta_deltaG_category_used": delta_delta_category,
        "delta_deltaG_clamped": delta_delta_clamped,
        "eyring_k_variant_s_inv": round(float(k_variant), 6),
        "k_variant_s_inv": round(float(k_variant), 6),
        "k_eyring_s_inv": round(float(k_variant), 6),
        "k_diff_cap_s_inv": round(float(k_diff_cap), 6)
        if isinstance(k_diff_cap, (int, float))
        else None,
        "f_protonation": round(float(protonation_factor), 3),
        "residue_protonation_fraction": round(float(residue_fraction), 3)
        if isinstance(residue_fraction, (int, float))
        else None,
        "solvent_penalty": round(float(solvent_pen), 3),
        "k_eff_s_inv": round(float(k_eff), 6),
        "mechanism_mismatch_penalty_kJ_per_mol": round(float(mismatch_penalty_kj), 3),
        "mechanism_mismatch_policy": mismatch_policy,
        "protonation_residue": protonation_info.get("residue"),
        "protonation_notes": protonation_info.get("notes"),
        "protonation_uncertain": protonation_info.get("uncertain"),
        "solvent_penalty_note": solvent_info.get("note"),
        "diffusion_D_m2_s": diffusion_info.get("D_m2_s"),
        "diffusion_L_A": diffusion_info.get("L_A"),
        "diffusion_tau_s": diffusion_info.get("tau_s"),
    }

def _module3_handoff(
    job_card: Dict[str, Any],
    selected: Dict[str, Any],
    best_variant: Optional[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    best_variant_policy: str,
    nucleophile_geometry: Optional[str] = None,
) -> Dict[str, Any]:
    selection_reason = (
        f"Selected highest-ranked variant using policy {best_variant_policy}."
        if best_variant
        else "No variant selected."
    )
    return {
        "scaffold_id": selected.get("scaffold_id"),
        "pdb_path": selected.get("pdb_path"),
        "attack_envelope": selected.get("attack_envelope"),
        "candidate_residues_by_role": selected.get("candidate_residues_by_role"),
        "bond_center_hint": job_card.get("bond_center_hint") or {},
        "nucleophile_geometry": nucleophile_geometry
        or (selected.get("reach_summary", {}) or {}).get("nucleophile_geometry"),
        "best_variant": best_variant,
        "best_variant_policy": best_variant_policy,
        "best_variant_reason": selection_reason,
        "variant_set": variants,
        "recommended_computations": [
            "Dock substrate poses and score attack geometry vs envelope.",
            "Evaluate retention proxies (H-bonds, aromatic contacts).",
            "Compare baseline vs top variants for access/reach penalties.",
        ],
    }


def _design_loop(
    job_card: Dict[str, Any],
    selected: Dict[str, Any],
    variants: List[Dict[str, Any]],
    status: str,
) -> Dict[str, Any]:
    return {
        "status": "planned" if status == "PASS" else "degraded",
        "steps": [
            {
                "name": "motif_constraints",
                "status": "planned",
                "inputs": ["attack_envelope", "nucleophile_geometry"],
            },
            {
                "name": "sequence_design",
                "status": "planned",
                "tools": ["ProteinMPNN", "LigandMPNN"],
            },
            {
                "name": "structure_validation",
                "status": "planned",
                "tools": ["AlphaFold3_class"],
            },
            {
                "name": "scoring",
                "status": "planned",
                "metrics": ["binding", "geometry", "retention"],
            },
            {
                "name": "iterate",
                "status": "planned",
                "policy": "active_learning",
            },
        ],
        "context": {
            "decision": job_card.get("decision"),
            "selected_scaffold_id": selected.get("scaffold_id"),
            "variant_count": len(variants),
        },
    }


def _fallback_plan(
    job_card: Dict[str, Any],
    top_scaffolds: List[Dict[str, Any]],
    reason: Optional[str],
) -> Dict[str, Any]:
    route = job_card.get("mechanism_route") or {}
    return {
        "status": "degraded_ok",
        "reason": reason or "unspecified",
        "actions": [
            "Broaden variant generation (retain baseline + retention fixes).",
            "Relax Module 1 strictness or expand scaffold library.",
            "Consider alternative mechanism tracks.",
            "Request more specific trap target if reagent generation is intended.",
        ],
        "alternate_tracks": route.get("secondary") or [],
        "requested_next_input": job_card.get("required_next_input") or [],
        "candidate_set": top_scaffolds,
    }


def _condition_assessment(
    job_card: Dict[str, Any],
    selected: Dict[str, Any],
) -> Dict[str, Any]:
    rcf = job_card.get("reaction_condition_field") or {}
    given = dict(rcf.get("given_conditions") or {})
    optimum_hint = rcf.get("optimum_conditions_hint") or {}
    condition_profile = job_card.get("condition_profile") or {}
    if given.get("pH") is None and condition_profile.get("pH") is not None:
        given["pH"] = condition_profile.get("pH")
    if given.get("temperature_c") is None and condition_profile.get("temperature_K") is not None:
        given["temperature_c"] = round(float(condition_profile["temperature_K"]) - 273.15, 1)
    given_ph = given.get("pH")
    given_temp = given.get("temperature_c")
    score = rcf.get("condition_feasibility", {}).get("given_conditions_score")
    if not isinstance(score, (int, float)):
        score = _condition_score(given, optimum_hint)

    limiting = list(rcf.get("condition_feasibility", {}).get("notes") or [])
    retention_risk = (selected.get("retention_metrics") or {}).get("retention_risk_flag")
    if retention_risk == "HIGH":
        limiting.append("Retention risk high under current conditions.")

    suggested = {}
    if optimum_hint.get("pH_range"):
        ph_range = optimum_hint["pH_range"]
        suggested["pH"] = round(sum(ph_range) / 2.0, 2)
    if optimum_hint.get("temperature_c"):
        temp_range = optimum_hint["temperature_c"]
        suggested["temperature_c"] = round(sum(temp_range) / 2.0, 1)

    status = "FEASIBLE"
    if score < 0.5:
        status = "CONDITIONALLY_WORKING"
    if score < 0.35:
        status = "LOW_CONFIDENCE"

    expected = {
        "likelihood": "LOW_TO_MODERATE" if score < 0.6 else "MODERATE",
        "type": "detectable conversion under optimized conditions",
    }

    return {
        "status": status,
        "given_conditions": {"pH": given_ph, "temperature_c": given_temp},
        "given_conditions_score": round(float(score), 3),
        "limiting_factors": limiting,
        "suggested_adjustments": suggested,
        "optimum_conditions": optimum_hint,
        "expected_outcome": expected,
    }


def _condition_search_summary(
    job_card: Dict[str, Any],
    assessment: Dict[str, Any],
) -> List[Dict[str, Any]]:
    given = assessment.get("given_conditions") or {}
    optimum = assessment.get("optimum_conditions") or {}
    ph_base = given.get("pH")
    temp_base = given.get("temperature_c")
    if ph_base is None or temp_base is None:
        return []

    ph_offsets = [-0.6, 0.0, 0.6]
    temp_offsets = [-10.0, 0.0, 10.0]
    candidates = []
    for ph_offset in ph_offsets:
        for temp_offset in temp_offsets:
            ph_val = round(ph_base + ph_offset, 2)
            temp_val = round(temp_base + temp_offset, 1)
            score = _condition_score({"pH": ph_val, "temperature_c": temp_val}, optimum)
            candidates.append({"pH": ph_val, "temperature_c": temp_val, "score": score})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:6]


def _ph_locked_mutation_plan(job_card: Dict[str, Any]) -> Dict[str, Any]:
    constraints = job_card.get("constraints") or {}
    ph_min = constraints.get("ph_min")
    ph_max = constraints.get("ph_max")
    locked = (
        ph_min is not None
        and ph_max is not None
        and abs(ph_max - ph_min) <= 0.3
    )
    if not locked:
        return {"status": "not_locked", "mutation_bundles": []}

    ph_value = round((ph_min + ph_max) / 2.0, 2)
    bundles = [
        {
            "goal": "activate general base at neutral pH",
            "mutations": [
                "Introduce Asp to form His-Asp dyad",
                "Add Arg to stabilize developing anion",
            ],
            "expected_effect": "optimum pH shifts down ~0.5-1.0",
            "risks": ["stability tradeoff", "more hydrolysis side-reaction"],
        }
    ]
    return {"status": "locked", "if_pH_must_stay": ph_value, "mutation_bundles": bundles}


def _confidence_calibration(
    job_card: Dict[str, Any],
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    confidence = job_card.get("confidence") or {}
    mechanistic = confidence.get("route") or 0.0
    memory = _query_condition_memory(job_card)
    empirical = memory.get("empirical_reliability", 0.5)
    n = memory.get("n", 0)
    if n > 0:
        calibrated = 0.5 * mechanistic + 0.5 * empirical
        uncertainty = 0.12
    else:
        calibrated = 0.8 * mechanistic + 0.2 * empirical
        uncertainty = 0.2
    return {
        "calibrated_confidence": round(calibrated, 3),
        "uncertainty": round(uncertainty, 3),
        "mechanistic_score": round(mechanistic, 3),
        "empirical_reliability": round(empirical, 3),
        "evidence_count": n,
        "main_uncertainty_source": memory.get("main_uncertainty_source", "limited prior data"),
    }


def _learning_trace(job_card: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
    memory = _query_condition_memory(job_card)
    return {
        "evidence_used": memory.get("evidence_used", "no prior runs"),
        "known_failure_modes": memory.get("failure_modes", []),
        "model_update": memory.get("model_update", "no update"),
    }


def _condition_score(
    given: Dict[str, Any],
    optimum: Dict[str, Any],
) -> float:
    ph = given.get("pH")
    temp = given.get("temperature_c")
    ph_range = optimum.get("pH_range") or [6.5, 8.0]
    temp_range = optimum.get("temperature_c") or [25.0, 45.0]
    score = 0.8
    if ph is not None:
        if ph < ph_range[0] - 0.5 or ph > ph_range[1] + 0.5:
            score -= 0.15
        if ph < ph_range[0] - 1.0 or ph > ph_range[1] + 1.0:
            score -= 0.15
    if temp is not None:
        if temp < temp_range[0] - 5 or temp > temp_range[1] + 5:
            score -= 0.1
    return round(max(0.0, min(1.0, score)), 3)


def _delta_g_from_conditions(
    base_dg: float,
    ph: Optional[float],
    temp_k: Optional[float],
    optimum: Dict[str, Any],
) -> float:
    return delta_g_with_conditions(base_dg, ph, temp_k, optimum)


def _rate_from_delta_g(delta_g: float, temp_k: float) -> float:
    return rate_constant(delta_g, temp_k)


def _scaffold_seed(scaffold_id: str) -> int:
    digest = hashlib.sha1(scaffold_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _candidate_reports(
    scaffold_rankings: List[Dict[str, Any]],
    job_card: Dict[str, Any],
    condition_assessment: Dict[str, Any],
    confidence_calibration: Dict[str, Any],
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    difficulty = job_card.get("difficulty_label") or "MEDIUM"
    base_dg = {"EASY": 17.0, "MEDIUM": 21.0, "HARD": 27.0}.get(difficulty, 21.0)
    optimum_hint = (job_card.get("reaction_condition_field") or {}).get(
        "optimum_conditions_hint", {}
    )
    given = condition_assessment.get("given_conditions") or {}
    ph = given.get("pH")
    temp_c = given.get("temperature_c")
    temp_k = float(temp_c + 273.15) if temp_c is not None else 298.15

    causal = job_card.get("causal_discovery") or {}
    causal_edges = causal.get("causal_edges_topK") or []
    causal_factors = [edge.get("to") for edge in causal_edges][:3]

    top_variants = [
        {
            "variant_id": variant.get("variant_id"),
            "label": variant.get("label"),
            "rationale": variant.get("rationale"),
            "expected_effects": variant.get("estimated_effects"),
        }
        for variant in variants[:3]
    ]

    optimum = search_optimum_conditions(base_dg, optimum_hint)
    for scaffold in scaffold_rankings[:3]:
        delta_g_mean = scaffold.get("delta_g_mean") or scaffold.get("delta_g")
        delta_g_std = scaffold.get("delta_g_std")
        k_pred_mean = scaffold.get("k_pred_mean") or scaffold.get("k_pred")
        k_pred_ci90 = scaffold.get("k_pred_ci90") or []
        k_dist_summary = scaffold.get("k_dist_summary")
        if delta_g_mean is None or k_pred_mean is None:
            base_dg_candidate, base_dg_std = estimate_delta_g(
                base_dg, {"difficulty_label": difficulty}
            )
            delta_g_mean = delta_g_with_conditions(
                base_dg_candidate, ph, temp_k, optimum_hint
            )
            delta_g_std = base_dg_std
            temp_c_for_sampling = temp_c if temp_c is not None else temp_k - 273.15
            k_stats = sample_k_pred(
                delta_g_mean,
                base_dg_std,
                temp_c_for_sampling,
                seed=_scaffold_seed(scaffold.get("scaffold_id") or "unknown"),
            )
            k_pred_mean = k_stats["mean"]
            k_pred_ci90 = k_stats["ci90"]
        if k_dist_summary is None and delta_g_mean is not None and delta_g_std is not None:
            seed = _scaffold_seed(scaffold.get("scaffold_id") or "unknown")
            k_samples = sample_k_pred(
                delta_g=delta_g_mean,
                temperature=temp_k,
                sigma=float(delta_g_std),
                n=256,
                rng=seed,
            )
            k_sample_mean = sum(k_samples) / len(k_samples) if k_samples else 0.0
            k_dist_summary = {
                "n": len(k_samples),
                "mean": round(k_sample_mean, 5),
                "p10": round(percentile(k_samples, 0.1), 5),
                "p50": round(percentile(k_samples, 0.5), 5),
                "p90": round(percentile(k_samples, 0.9), 5),
            }
        sensitivities = _condition_sensitivity(base_dg, ph, temp_k, optimum_hint)
        adjusted_score = scaffold.get("adjusted_score") or 0.0
        success_prob = confidence_calibration.get("calibrated_confidence", 0.5) * (
            0.7 + 0.6 * adjusted_score
        )
        success_prob = max(0.0, min(1.0, success_prob))
        confidence_level = "Medium"
        if success_prob >= 0.7:
            confidence_level = "High"
        elif success_prob < 0.4:
            confidence_level = "Low"

        gain = 0.0
        if k_pred_mean and k_pred_mean > 0:
            gain = (optimum["k_pred"] / k_pred_mean) - 1.0

        reports.append(
            {
                "candidate_id": scaffold.get("scaffold_id"),
                "predicted_mechanism": (job_card.get("mechanism_route") or {}).get("primary"),
                "predicted_activity_at_given_conditions": {
                    "k_pred_mean": round(float(k_pred_mean or 0.0), 5),
                    "k_pred_ci90": [
                        round(float(k_pred_ci90[0]), 5),
                        round(float(k_pred_ci90[1]), 5),
                    ]
                    if len(k_pred_ci90) == 2
                    else [],
                    "k_dist_summary": k_dist_summary,
                    "delta_g_mean": round(float(delta_g_mean or 0.0), 3),
                    "delta_g_std": round(float(delta_g_std or 0.0), 3),
                    "success_prob_calibrated": round(success_prob, 3),
                    "confidence_level": confidence_level,
                    "key_condition_sensitivities": sensitivities,
                },
                "optimum_conditions": {
                    "pH_opt_pred": optimum["pH_opt"],
                    "T_opt_pred_K": optimum["T_opt_K"],
                    "T_opt_pred_C": optimum["T_opt_C"],
                    "predicted_gain_vs_given_conditions": round(gain, 3),
                },
                "key_causal_drivers": causal_factors,
                "topology_feasibility": {
                    "feasibility_flag": scaffold.get("feasibility_flag"),
                    "reason_tags": scaffold.get("required_topology_constraints"),
                },
                "mutation_plan": top_variants,
            }
        )
    return reports


def _optimum_conditions_for_candidate(
    base_dg: float,
    optimum_hint: Dict[str, Any],
) -> Dict[str, Any]:
    optimum = search_optimum_conditions(base_dg, optimum_hint)
    return optimum


def _condition_sensitivity(
    base_dg: float,
    ph: Optional[float],
    temp_k: Optional[float],
    optimum_hint: Dict[str, Any],
) -> List[str]:
    if ph is None or temp_k is None:
        return []
    base_k = rate_constant(delta_g_with_conditions(base_dg, ph, temp_k, optimum_hint), temp_k)
    ph_up = rate_constant(
        delta_g_with_conditions(base_dg, ph + 0.5, temp_k, optimum_hint), temp_k
    )
    ph_down = rate_constant(
        delta_g_with_conditions(base_dg, ph - 0.5, temp_k, optimum_hint), temp_k
    )
    temp_up = rate_constant(
        delta_g_with_conditions(base_dg, ph, temp_k + 5.0, optimum_hint), temp_k + 5.0
    )
    temp_down = rate_constant(
        delta_g_with_conditions(base_dg, ph, temp_k - 5.0, optimum_hint), temp_k - 5.0
    )
    sensitivity = []
    if ph_up > base_k and ph_up > ph_down:
        sensitivity.append("activity increases with higher pH")
    elif ph_down > base_k:
        sensitivity.append("activity increases with lower pH")
    if temp_up > base_k and temp_up > temp_down:
        sensitivity.append("activity increases with higher temperature")
    elif temp_down > base_k:
        sensitivity.append("activity increases with lower temperature")
    return sensitivity


def _final_report(
    job_card: Dict[str, Any],
    candidate_reports: List[Dict[str, Any]],
    condition_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    constraints = job_card.get("constraints") or {}
    reaction_task = job_card.get("reaction_task") or {}
    given_conditions = condition_assessment.get("given_conditions") or {}
    retry = {
        "retry_with_better_conditions": condition_assessment.get("suggested_adjustments"),
        "retry_with_better_scaffold": "Consider alternate scaffold family"
        if not candidate_reports
        else None,
    }
    audit = {
        "assumptions_used": job_card.get("assumptions_used") or [],
        "extrapolation": "condition search used heuristic ranges",
        "uncertainty_sources": condition_assessment.get("limiting_factors") or [],
    }
    return {
        "section_a_input_summary": {
            "reaction_task": reaction_task,
            "given_conditions": given_conditions,
            "constraints": constraints,
        },
        "section_b_top_candidates": candidate_reports,
        "section_c_retry_recommendations": retry,
        "section_d_audit_trail": audit,
    }


def _merge_shared_io(ctx: PipelineContext, module2_result: Dict[str, Any]) -> Dict[str, Any]:
    shared = ctx.data.get("shared_io")
    job_card = ctx.data.get("job_card") or {}
    if shared:
        shared_input = shared.get("input", {})
    else:
        bond_spec = BondSpec(
            target_bond=ctx.target_bond,
            selection_mode=(job_card.get("resolved_target") or {}).get("selection_mode"),
            resolved_target=job_card.get("resolved_target"),
            context=job_card.get("bond_context"),
        )
        substrate_context = SubstrateContext(
            smiles=ctx.smiles,
            structure_summary=job_card.get("structure_summary"),
        )
        telemetry = TelemetryContext(run_id="run_unknown", trace=["module2"])
        shared_input = SharedInput(
            bond_spec=bond_spec,
            condition_profile=ConditionProfile(**(job_card.get("condition_profile") or {})),
            substrate_context=substrate_context,
            telemetry=telemetry,
        ).to_dict()
    telemetry = shared_input.get("telemetry") or {}
    trace = telemetry.get("trace") or []
    if "module2" not in trace:
        trace.append("module2")
    telemetry["trace"] = trace
    shared_input["telemetry"] = telemetry

    condition_assessment = module2_result.get("condition_assessment") or {}
    retry_suggestion = _retry_loop_suggestion(
        condition_assessment.get("given_conditions") or {},
        module2_result.get("optimum_conditions_estimate") or {},
    )
    output = SharedOutput(
        result={
            "status": module2_result.get("status"),
            "selected_scaffold": (module2_result.get("selected_scaffold") or {}).get(
                "scaffold_id"
            ),
            "best_variant": (module2_result.get("best_variant") or {}).get("variant_id"),
        },
        given_conditions_effect=module2_result.get("predicted_under_given_conditions") or {},
        optimum_conditions=module2_result.get("optimum_conditions_estimate") or {},
        confidence={
            "calibrated_probability": module2_result.get("confidence_calibrated"),
            "uncertainty": (module2_result.get("confidence_calibration") or {}).get(
                "uncertainty"
            ),
            "main_risks": condition_assessment.get("limiting_factors") or [],
        },
        retry_loop_suggestion=retry_suggestion,
    )
    outputs = dict(shared.get("outputs", {})) if shared else {}
    outputs["module2"] = output.to_dict()
    payload = dict(shared) if shared else {}
    payload["input"] = shared_input
    payload["outputs"] = outputs
    energy_ledger = payload.get("energy_ledger") or {}
    energy_update = module2_result.get("energy_ledger_update")
    if isinstance(energy_update, dict):
        energy_ledger = dict(energy_ledger)
        energy_ledger.update({key: value for key, value in energy_update.items() if value is not None})
    payload["energy_ledger"] = energy_ledger
    return payload


def _reaction_hash_mismatch(shared_io: Dict[str, Any], job_card: Dict[str, Any]) -> bool:
    shared_hash = (shared_io.get("input") or {}).get("reaction_identity_hash")
    job_hash = (job_card.get("reaction_identity") or {}).get("hash")
    return bool(shared_hash and job_hash and shared_hash != job_hash)


def _energy_ledger_update(
    module2_result: Dict[str, Any],
    shared_energy_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    physics = module2_result.get("module2_physics_audit") or {}
    delta_g = physics.get("deltaG_dagger_variant_kJ_per_mol")
    k_eff = physics.get("k_eff_s_inv")
    horizon_s = None
    condition = (module2_result.get("condition_assessment") or {}).get("given_conditions") or {}
    if isinstance(condition.get("temperature_c"), (int, float)):
        horizon_s = module2_result.get("module3_handoff", {}).get("horizon_s")
    horizon_s = horizon_s or shared_energy_ledger.get("horizon_s") or 3600.0
    p_success = None
    if isinstance(k_eff, (int, float)):
        p_success = 1.0 - math.exp(-float(k_eff) * float(horizon_s))
    merged = dict(shared_energy_ledger or {})
    if isinstance(delta_g, (int, float)):
        merged["deltaG_dagger_variant_kJ"] = round(float(delta_g), 3)
    if isinstance(k_eff, (int, float)):
        merged["k_variant_s_inv"] = round(float(k_eff), 6)
    if isinstance(p_success, (int, float)):
        merged["p_success_variant_horizon"] = round(float(p_success), 6)
    merged["horizon_s"] = float(horizon_s)
    notes = list(merged.get("notes") or [])
    notes.append("module2 variant update (variant fields)")
    merged["notes"] = list(dict.fromkeys(notes))
    return merged


def _retry_loop_suggestion(
    given: Dict[str, Any],
    optimum: Dict[str, Any],
    threshold_ph: float = 0.6,
    threshold_temp: float = 8.0,
) -> Dict[str, Any]:
    ph = given.get("pH")
    temp = given.get("temperature_c")
    if ph is None or temp is None:
        return {"status": "insufficient_data"}
    opt_ph = None
    opt_temp = None
    if optimum.get("pH_range"):
        opt_ph = sum(optimum["pH_range"]) / 2.0
    if optimum.get("temperature_c"):
        opt_temp = sum(optimum["temperature_c"]) / 2.0
    if optimum.get("pH_opt") is not None:
        opt_ph = float(optimum.get("pH_opt"))
    if optimum.get("T_opt_C") is not None:
        opt_temp = float(optimum.get("T_opt_C"))
    if opt_ph is None or opt_temp is None:
        return {"status": "unknown_optimum"}
    delta_ph = opt_ph - ph
    delta_temp = opt_temp - temp
    if abs(delta_ph) <= threshold_ph and abs(delta_temp) <= threshold_temp:
        return {"status": "close_enough"}
    return {
        "status": "suggest_retry",
        "proposed_conditions": {
            "pH": round(ph + (delta_ph * 0.5), 2),
            "temperature_c": round(temp + (delta_temp * 0.5), 1),
        },
        "expected_improvement": "moderate",
    }


def _predicted_under_given_conditions(
    selected: Dict[str, Any],
    condition_assessment: Dict[str, Any],
    confidence_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    given = condition_assessment.get("given_conditions") or {}
    return {
        "conditions": given,
        "k_pred_mean": selected.get("k_pred_mean") or selected.get("k_pred"),
        "k_pred_ci90": selected.get("k_pred_ci90"),
        "k_dist_summary": selected.get("k_dist_summary"),
        "delta_g_mean": selected.get("delta_g_mean") or selected.get("delta_g"),
        "delta_g_std": selected.get("delta_g_std"),
        "physics_gate": selected.get("physics_gate"),
        "model_risk": selected.get("model_risk"),
        "confidence_calibrated": confidence_calibration.get("calibrated_confidence"),
    }


def _optimum_conditions_estimate(job_card: Dict[str, Any]) -> Dict[str, Any]:
    difficulty = job_card.get("difficulty_label") or "MEDIUM"
    base_dg = {"EASY": 17.0, "MEDIUM": 21.0, "HARD": 27.0}.get(difficulty, 21.0)
    optimum_hint = (job_card.get("reaction_condition_field") or {}).get(
        "optimum_conditions_hint", {}
    )
    return search_optimum_conditions(base_dg, optimum_hint)


def _delta_from_optimum(
    condition_assessment: Dict[str, Any],
    optimum: Dict[str, Any],
) -> Dict[str, Any]:
    given = condition_assessment.get("given_conditions") or {}
    delta_ph = None
    delta_t = None
    if given.get("pH") is not None:
        delta_ph = round(float(given["pH"]) - float(optimum.get("pH_opt", 0.0)), 2)
    if given.get("temperature_c") is not None:
        delta_t = round(float(given["temperature_c"]) - float(optimum.get("T_opt_C", 0.0)), 2)
    return {"delta_pH": delta_ph, "delta_T_C": delta_t}


def _load_condition_memory() -> Dict[str, Any]:
    if not CONDITION_MEMORY_PATH.exists():
        return {"entries": []}
    try:
        with CONDITION_MEMORY_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"entries": []}


def _query_condition_memory(job_card: Dict[str, Any]) -> Dict[str, Any]:
    route = job_card.get("mechanism_route") or {}
    primary = route.get("primary") or "unknown"
    rcf = job_card.get("reaction_condition_field") or {}
    given = rcf.get("given_conditions") or {}
    ph = given.get("pH")
    temp = given.get("temperature_c")
    memory = _load_condition_memory()
    entries = memory.get("entries", [])
    matches = []
    for entry in entries:
        if entry.get("reaction_family") != primary:
            continue
        ph_range = entry.get("pH_range")
        temp_range = entry.get("temperature_c")
        if ph is not None and ph_range:
            if not (ph_range[0] <= ph <= ph_range[1]):
                continue
        if temp is not None and temp_range:
            if not (temp_range[0] <= temp <= temp_range[1]):
                continue
        matches.append(entry)

    if not matches:
        return {
            "empirical_reliability": 0.5,
            "n": 0,
            "evidence_used": "no prior runs",
            "failure_modes": [],
            "model_update": "no update",
            "main_uncertainty_source": "limited prior data",
        }

    success_rates = [entry.get("success_rate", 0.5) for entry in matches]
    empirical = sum(success_rates) / len(success_rates)
    failure_modes = []
    for entry in matches:
        failure_modes.extend(entry.get("failure_modes", []))
    return {
        "empirical_reliability": empirical,
        "n": len(matches),
        "evidence_used": f"{len(matches)} related runs",
        "failure_modes": list(dict.fromkeys(failure_modes)),
        "model_update": "recommended condition window adjusted based on prior data",
        "main_uncertainty_source": "substrate-specific variability",
    }


def _fail_module2(
    reason: str,
    errors: List[str],
    warnings: List[str],
    job_card: Dict[str, Any],
    top_scaffolds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback_plan = _fallback_plan(job_card, top_scaffolds, reason)
    condition_assessment = _condition_assessment(job_card, {})
    condition_search_summary = _condition_search_summary(job_card, condition_assessment)
    return {
        "status": STATUS_DEGRADED_OK,
        "halt_reason": reason,
        "selected_scaffold": None,
        "selection_explain": None,
        "variant_set": [],
        "module3_handoff": {},
        "fallback_plan": fallback_plan,
        "design_loop": _design_loop(job_card, {}, [], STATUS_DEGRADED_OK),
        "condition_assessment": condition_assessment,
        "condition_search_summary": condition_search_summary,
        "ph_locked_mutation_plan": _ph_locked_mutation_plan(job_card),
        "confidence_calibration": _confidence_calibration(job_card, condition_assessment),
        "learning_trace": _learning_trace(job_card, condition_assessment),
        "docking": {"status": "disabled"},
        "md_stability": {"status": "disabled"},
        "composite_binding_score": None,
        "binding_score_adjustment": None,
        "computational_engines_used": [],
        "warnings": warnings,
        "errors": errors,
    }
