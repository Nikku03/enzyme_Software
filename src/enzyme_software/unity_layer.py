from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, List, Optional

from enzyme_software.chemcore import (
    PKA_CATALYTIC_GROUPS,
    SUBSTRATE_CARBOXYLIC_PKA_RANGE,
    protonation_fractions,
    screening_factor,
    solvent_penalty,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_temperature_k(temp_c: Optional[float]) -> Optional[float]:
    if temp_c is None:
        return None
    try:
        return float(temp_c) + 273.15
    except (TypeError, ValueError):
        return None


@dataclass
class SubState:
    measured: Dict[str, Any] = field(default_factory=dict)
    derived: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured": self.measured,
            "derived": self.derived,
            "audit": self.audit,
        }


@dataclass
class SharedState:
    input_spec: Dict[str, Any] = field(default_factory=dict)
    condition_profile: Dict[str, Any] = field(default_factory=dict)
    physics: SubState = field(default_factory=SubState)
    chemistry: SubState = field(default_factory=SubState)
    biology: SubState = field(default_factory=SubState)
    mechanism: SubState = field(default_factory=SubState)
    scoring: SubState = field(default_factory=SubState)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_spec": self.input_spec,
            "condition_profile": self.condition_profile,
            "physics": self.physics.to_dict(),
            "chem": self.chemistry.to_dict(),
            "bio": self.biology.to_dict(),
            "mechanism": self.mechanism.to_dict(),
            "scoring": self.scoring.to_dict(),
            "provenance": self.provenance,
            "audit": self.audit,
        }


def build_shared_state(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    trap_target: Optional[str],
    constraints: Dict[str, Any],
) -> SharedState:
    temp_c = constraints.get("temperature_c")
    temp_k = _to_temperature_k(temp_c)
    if temp_k is None:
        temp_k = 298.15
    pH = constraints.get("ph_min")
    if pH is None:
        pH = constraints.get("ph_max")
    input_spec = {
        "smiles": smiles,
        "target_bond": target_bond,
        "requested_output": requested_output,
        "trap_target": trap_target,
    }
    condition_profile = {
        "pH": pH,
        "temperature_K": round(float(temp_k), 2),
        "temperature_C": round(float(temp_c), 2) if isinstance(temp_c, (int, float)) else None,
        "solvent": None,
        "ionic_strength": None,
        "cofactors": [],
    }
    physics = SubState(measured={"temperature_K": condition_profile["temperature_K"], "pH": pH})
    chemistry = SubState(
        measured={
            "pH": pH,
            "solvent": condition_profile.get("solvent"),
            "ionic_strength": condition_profile.get("ionic_strength"),
        }
    )
    solvent = condition_profile.get("solvent")
    ionic_strength = condition_profile.get("ionic_strength")
    screening = screening_factor(ionic_strength)
    solvent_info = solvent_penalty(solvent)
    residues = list(PKA_CATALYTIC_GROUPS.keys())
    chemistry.derived["protonation_fractions"] = protonation_fractions(pH, residues)
    chemistry.derived["protonation_residues"] = residues
    chemistry.derived["substrate_carboxylic_acid_pka_range"] = list(
        SUBSTRATE_CARBOXYLIC_PKA_RANGE
    )
    chemistry.derived["functional_group_map"] = {}
    chemistry.derived["reaction_family"] = None
    chemistry.derived["leaving_group_score"] = None
    chemistry.derived["ionic_screening"] = screening
    chemistry.derived["solvent_penalty"] = solvent_info
    chemistry.derived["context"] = {}
    uncertainty_flags: List[str] = []
    if pH is None:
        uncertainty_flags.append("pH_unknown")
    if screening.get("uncertain"):
        uncertainty_flags.append("ionic_strength_unknown")
    if solvent_info.get("solvent_unknown"):
        uncertainty_flags.append("solvent_unknown")
    chemistry.audit["uncertainty_flags"] = uncertainty_flags
    biology = SubState()
    biology.derived["protonation"] = {}
    biology.derived["enzyme_family_prior"] = {}
    biology.derived["residue_protonation_fraction"] = None
    biology.derived["cofactor_requirements"] = {}
    biology.derived["mechanism_spec"] = None
    biology.derived["mechanism_contract"] = None
    mechanism = SubState()
    mechanism.derived["contract"] = None
    mechanism.derived["mismatch"] = {
        "status": "unknown",
        "expected": None,
        "observed": None,
        "penalty": 0.0,
        "policy": None,
        "explanation": "mechanism contract not yet applied",
    }
    physics.derived["energy_ledger"] = {}
    scoring = SubState()
    scoring.derived["ledger"] = {}
    return SharedState(
        input_spec=input_spec,
        condition_profile=condition_profile,
        physics=physics,
        chemistry=chemistry,
        biology=biology,
        mechanism=mechanism,
        scoring=scoring,
    )


def merge_module_output(
    module_id: int,
    shared_state: SharedState,
    module_output: Dict[str, Any],
) -> SharedState:
    source = f"module{module_id}"
    shared_state.provenance.append(
        {
            "source_module": source,
            "timestamp": _utc_timestamp(),
        }
    )
    job_card = module_output.get("job_card") or {}
    if module_id == -1:
        shared_state.chemistry.derived["sre"] = module_output
        shared_state.chemistry.audit.setdefault("module_minus1_keys", list(module_output.keys()))
        rt = module_output.get("resolved_target") or {}
        if rt.get("bond_type"):
            shared_state.chemistry.derived["sre_bond_type"] = rt["bond_type"]
        reactivity = module_output.get("reactivity") or {}
        conf_prior = reactivity.get("confidence_prior")
        if conf_prior is not None:
            shared_state.scoring.derived.setdefault(
                "module_minus1_confidence", conf_prior
            )
        return shared_state
    if module_id == 0:
        if job_card:
            shared_state.chemistry.derived["job_type"] = job_card.get("job_type")
            shared_state.chemistry.derived["reaction_intent"] = job_card.get("reaction_intent")
            shared_state.biology.derived["mechanism_route"] = job_card.get("mechanism_route")
            shared_state.physics.derived["module0_physics_audit"] = job_card.get("physics_audit")
            shared_state.scoring.derived["confidence"] = job_card.get("confidence") or {}
            set_energy_ledger(
                shared_state,
                job_card.get("energy_ledger"),
                audit={"source": "module0"},
                source="module0",
            )
            chem_contract = job_card.get("chemistry_contract") or {}
            shared_state.chemistry.derived["functional_group_map"] = (
                chem_contract.get("functional_group_map")
                or (job_card.get("structure_summary") or {}).get("functional_groups")
                or {}
            )
            shared_state.chemistry.derived["reaction_family"] = chem_contract.get(
                "reaction_family"
            )
            shared_state.chemistry.derived["leaving_group_score"] = chem_contract.get(
                "leaving_group_score"
            )
            token_audit = job_card.get("token_resolution_audit")
            if token_audit:
                shared_state.chemistry.audit["token_resolution"] = token_audit
            bio_contract = job_card.get("biology_contract") or {}
            shared_state.biology.derived["enzyme_family_prior"] = bio_contract.get(
                "enzyme_family_prior"
            ) or {}
            shared_state.biology.derived["residue_protonation_fraction"] = bio_contract.get(
                "residue_protonation_fraction"
            )
            shared_state.biology.derived["cofactor_requirements"] = bio_contract.get(
                "cofactor_requirements"
            ) or {}
            shared_state.biology.derived["mechanism_spec"] = job_card.get("mechanism_spec")
            shared_state.biology.derived["mechanism_contract"] = job_card.get(
                "mechanism_contract"
            )
            set_mechanism_contract(
                shared_state,
                job_card.get("mechanism_contract"),
                job_card.get("mechanism_mismatch"),
                source="module0",
            )
            set_chem_context(shared_state, job_card.get("chem_context"), source="module0")
            set_physics_prior(
                shared_state,
                priors=(job_card.get("physics") or {}).get("routes"),
                audit=job_card.get("physics_audit"),
                source="module0",
            )
            set_bio_gates(
                shared_state,
                protonation=job_card.get("bio_protonation"),
                audit={"source": "module0"},
                source="module0",
            )
            route = job_card.get("mechanism_route") or {}
            residues = _route_key_residues(route)
            pH = shared_state.condition_profile.get("pH")
            shared_state.chemistry.derived["protonation_fractions_route"] = (
                protonation_fractions(pH, residues)
            )
            shared_state.chemistry.derived["protonation_residues_route"] = residues
        _update_scoring_ledger(shared_state, "module0", module_output, job_card)
        shared_state.chemistry.audit.setdefault("module0_keys", list(module_output.keys()))
    elif module_id == 1:
        shared_state.physics.derived["module1_physics_audit"] = module_output.get(
            "module1_physics_audit"
        )
        set_energy_ledger(
            shared_state,
            module_output.get("energy_ledger_update"),
            audit={"source": "module1"},
            source="module1",
        )
        _update_scoring_ledger(shared_state, "module1", module_output, None)
        shared_state.biology.audit.setdefault("module1_keys", list(module_output.keys()))
    elif module_id == 2:
        shared_state.physics.derived["module2_physics_audit"] = module_output.get(
            "module2_physics_audit"
        )
        shared_state.biology.derived["best_variant"] = module_output.get("best_variant")
        if module_output.get("mechanism_spec") is not None:
            shared_state.biology.derived["mechanism_spec"] = module_output.get("mechanism_spec")
        if module_output.get("mechanism_mismatch") is not None:
            set_mechanism_contract(
                shared_state,
                shared_state.mechanism.derived.get("contract"),
                module_output.get("mechanism_mismatch"),
                source="module2",
            )
        if module_output.get("mechanism_evidence") is not None:
            shared_state.mechanism.derived["evidence"] = module_output.get("mechanism_evidence")
        set_energy_ledger(
            shared_state,
            module_output.get("energy_ledger_update"),
            audit={"source": "module2"},
            source="module2",
        )
        set_bio_gates(
            shared_state,
            protonation=module_output.get("protonation_gate"),
            audit={"source": "module2"},
            source="module2",
        )
        _update_scoring_ledger(shared_state, "module2", module_output, None)
        shared_state.biology.audit.setdefault("module2_keys", list(module_output.keys()))
    elif module_id == 3:
        shared_state.physics.derived["module3_physics_audit"] = module_output.get(
            "module3_physics_audit"
        )
        shared_state.biology.derived["protocol_card"] = module_output.get("protocol_card")
        set_energy_ledger(
            shared_state,
            module_output.get("energy_ledger_update"),
            audit={"source": "module3"},
            source="module3",
        )
        _update_scoring_ledger(shared_state, "module3", module_output, None)
        shared_state.biology.audit.setdefault("module3_keys", list(module_output.keys()))
    return shared_state


def record_interlink(
    ctx: Any,
    module_id: int,
    reads: List[str],
    writes: List[str],
) -> None:
    audit = ctx.data.setdefault("interlink_audit", {})
    key = f"module{module_id}"
    audit[key] = {
        "reads": sorted(set(reads)),
        "writes": sorted(set(writes)),
    }


def set_chem_context(
    shared_state: SharedState,
    context: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    if not isinstance(context, dict):
        return
    shared_state.chemistry.derived["context"] = context
    if audit is not None:
        shared_state.chemistry.audit["context_audit"] = audit
    if source:
        shared_state.chemistry.audit["context_source"] = source


def set_physics_prior(
    shared_state: SharedState,
    priors: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    if isinstance(priors, dict):
        shared_state.physics.derived["priors"] = priors
    if audit is not None:
        shared_state.physics.audit["audit"] = audit
    if source:
        shared_state.physics.audit["prior_source"] = source


def set_bio_gates(
    shared_state: SharedState,
    protonation: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    if isinstance(protonation, dict):
        shared_state.biology.derived["protonation"] = protonation
    if audit is not None:
        shared_state.biology.audit["protonation_audit"] = audit
    if source:
        shared_state.biology.audit["protonation_source"] = source


def set_mechanism_contract(
    shared_state: SharedState,
    contract: Optional[Dict[str, Any]],
    mismatch: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    if contract is not None:
        shared_state.mechanism.derived["contract"] = contract
        shared_state.biology.derived["mechanism_contract"] = contract
    if mismatch is not None:
        shared_state.mechanism.derived["mismatch"] = mismatch
    if source:
        shared_state.mechanism.audit["source"] = source


def _update_scoring_ledger(
    shared_state: SharedState,
    module_key: str,
    module_output: Dict[str, Any],
    job_card: Optional[Dict[str, Any]],
) -> None:
    ledger = shared_state.scoring.derived.get("ledger") or {}
    if job_card and job_card.get("score_ledger"):
        ledger[module_key] = job_card.get("score_ledger")
    elif module_output.get("score_ledger"):
        ledger[module_key] = module_output.get("score_ledger")
    shared_state.scoring.derived["ledger"] = ledger


def set_energy_ledger(
    shared_state: SharedState,
    ledger_update: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    if not isinstance(ledger_update, dict):
        return
    current = shared_state.physics.derived.get("energy_ledger") or {}
    merged = dict(current)
    for key, value in ledger_update.items():
        if value is not None:
            merged[key] = value
    shared_state.physics.derived["energy_ledger"] = merged
    if audit is not None:
        shared_state.physics.audit["energy_ledger_audit"] = audit
    if source:
        shared_state.physics.audit["energy_ledger_source"] = source


def export_shared_io_patch(shared_state: SharedState) -> Dict[str, Any]:
    return {
        "sre": shared_state.chemistry.derived.get("sre") or {},
        "physics": {
            "priors": shared_state.physics.derived.get("priors") or {},
            "audit": shared_state.physics.audit.get("audit")
            or shared_state.physics.derived.get("module0_physics_audit")
            or {},
            "energy_ledger": shared_state.physics.derived.get("energy_ledger") or {},
        },
        "chem": {
            "context": shared_state.chemistry.derived.get("context") or {},
            "functional_group_map": shared_state.chemistry.derived.get("functional_group_map")
            or {},
            "reaction_family": shared_state.chemistry.derived.get("reaction_family"),
            "leaving_group_score": shared_state.chemistry.derived.get("leaving_group_score"),
            "audit": shared_state.chemistry.audit,
        },
        "bio": {
            "protonation": shared_state.biology.derived.get("protonation") or {},
            "enzyme_family_prior": shared_state.biology.derived.get("enzyme_family_prior")
            or {},
            "residue_protonation_fraction": shared_state.biology.derived.get(
                "residue_protonation_fraction"
            ),
            "cofactor_requirements": shared_state.biology.derived.get("cofactor_requirements")
            or {},
            "mechanism_spec": shared_state.biology.derived.get("mechanism_spec"),
            "mechanism_contract": shared_state.biology.derived.get("mechanism_contract"),
            "audit": shared_state.biology.audit,
        },
        "mechanism": {
            "contract": shared_state.mechanism.derived.get("contract"),
            "mismatch": shared_state.mechanism.derived.get("mismatch"),
            "evidence": shared_state.mechanism.derived.get("evidence"),
            "audit": shared_state.mechanism.audit,
        },
        "constraints": {
            "condition_profile": shared_state.condition_profile,
        },
        "scoring": {
            "ledger": shared_state.scoring.derived.get("ledger") or {},
            "confidence": shared_state.scoring.derived.get("confidence") or {},
            "audit": shared_state.scoring.audit,
        },
    }


def consistency_market(
    shared_state: SharedState,
    module_outputs: Optional[Dict[str, Any]] = None,
    job_card: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outputs = module_outputs or {}
    module3 = outputs.get("module3_experiment_designer") or {}
    ledger = shared_state.physics.derived.get("energy_ledger") or {}
    p_success = ledger.get("p_success_horizon")
    penalty = 0.0
    reasons: List[str] = []
    route_conf = None
    if isinstance(job_card, dict):
        route_conf = (job_card.get("confidence") or {}).get("route")
    if isinstance(p_success, (int, float)) and isinstance(route_conf, (int, float)):
        if route_conf > p_success + 0.3:
            penalty += 0.2
            reasons.append("route_confidence_gt_physics")
    plan_phys = (module3.get("module3_physics_audit") or {}).get("plan_phys")
    if isinstance(p_success, (int, float)) and isinstance(plan_phys, (int, float)):
        if plan_phys > p_success + 0.2:
            penalty += 0.1
            reasons.append("plan_phys_gt_physics")
    mismatch_pen = (
        shared_state.physics.derived.get("module0_physics_audit") or {}
    ).get("mechanism_mismatch_penalty")
    if isinstance(mismatch_pen, (int, float)) and float(mismatch_pen) > 0.1:
        penalty += 0.1
        reasons.append("mechanism_mismatch_penalty")
    mechanism_evidence = shared_state.mechanism.derived.get("evidence") or {}
    if mechanism_evidence.get("status") == "UNVERIFIED":
        penalty += 0.1
        reasons.append("mechanism_unverified")
    penalty = min(0.5, float(penalty))
    return {
        "penalty": round(penalty, 3),
        "reasons": reasons,
        "p_success_horizon": p_success,
        "route_confidence": route_conf,
        "plan_phys": plan_phys,
    }


def arbitrate_shared_state(
    shared_state: SharedState,
    module_outputs: Optional[Dict[str, Any]] = None,
    job_card: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    outputs = module_outputs or {}
    module0 = outputs.get("module0_strategy_router") or {}
    module1 = outputs.get("module1_topogate") or {}
    module2 = outputs.get("module2_active_site_refinement") or {}
    module3 = outputs.get("module3_experiment_designer") or {}

    module0_phys = shared_state.physics.derived.get("module0_physics_audit") or (
        job_card.get("physics_audit") if isinstance(job_card, dict) else {}
    ) or {}
    if not isinstance(module0_phys, dict):
        module0_phys = {}
    module1_phys = shared_state.physics.derived.get("module1_physics_audit") or module1.get(
        "module1_physics_audit",
        {},
    )
    module2_phys = shared_state.physics.derived.get("module2_physics_audit") or module2.get(
        "module2_physics_audit",
        {},
    )

    condition_profile = shared_state.condition_profile or {}
    solvent = condition_profile.get("solvent")
    ionic_strength = condition_profile.get("ionic_strength")
    pH = condition_profile.get("pH")

    weights = {
        "physics": 0.35,
        "chem": 0.25,
        "bio": 0.25,
        "math": 0.15,
    }
    if solvent is None:
        weights["chem"] *= 0.7
    if ionic_strength is None:
        weights["chem"] *= 0.8
    if pH is None:
        weights["chem"] *= 0.8
    if not module2_phys:
        weights["physics"] *= 0.7
    job_type = shared_state.chemistry.derived.get("job_type")
    if job_type == "REAGENT_GENERATION":
        weights["chem"] += 0.05
        weights["bio"] -= 0.05
    total_weight = sum(max(0.0, val) for val in weights.values()) or 1.0
    weights = {key: round(float(val) / total_weight, 3) for key, val in weights.items()}

    conflicts: List[Dict[str, Any]] = []
    penalty = 0.0
    hard_veto = False

    k_eff = module2_phys.get("k_eff_s_inv")
    if isinstance(k_eff, (int, float)) and float(k_eff) < 1e-4:
        conflicts.append(
            {"type": "kinetics_low", "value": k_eff, "threshold": 1e-4, "severity": "high"}
        )
        penalty += 0.2
        hard_veto = True

    module1_conf = module1.get("module1_confidence") or {}
    total_conf = module1_conf.get("total")
    if isinstance(total_conf, (int, float)) and float(total_conf) < 0.25:
        conflicts.append(
            {
                "type": "low_topogate_confidence",
                "value": total_conf,
                "threshold": 0.25,
                "severity": "medium",
            }
        )
        penalty += 0.15

    retention_mean = (module1_conf.get("ensemble") or {}).get("retention_mean")
    if isinstance(retention_mean, (int, float)) and float(retention_mean) < 0.35:
        conflicts.append(
            {
                "type": "retention_weak",
                "value": retention_mean,
                "threshold": 0.35,
                "severity": "medium",
            }
        )
        penalty += 0.1

    mismatch_pen = module0_phys.get("mechanism_mismatch_penalty")
    if isinstance(mismatch_pen, (int, float)) and float(mismatch_pen) > 0.1:
        conflicts.append(
            {
                "type": "mechanism_mismatch",
                "value": mismatch_pen,
                "threshold": 0.1,
                "severity": "medium",
            }
        )
        penalty += 0.15

    route_conf = None
    if isinstance(job_card, dict):
        route_conf = (job_card.get("confidence") or {}).get("route")
    prior = module0_phys.get("prior_success_probability")
    if (
        isinstance(prior, (int, float))
        and isinstance(route_conf, (int, float))
        and prior < 0.2
        and route_conf > 0.7
    ):
        conflicts.append(
            {
                "type": "overconfident_route",
                "value": route_conf,
                "threshold": 0.7,
                "severity": "low",
            }
        )
        penalty += 0.1

    penalty = min(0.5, float(penalty))
    recommendation = "monitor"
    if hard_veto or penalty >= 0.3:
        recommendation = "review_required"
    if hard_veto and penalty >= 0.4:
        recommendation = "halt_or_reroute"

    arbitration = {
        "stage": stage or "pipeline_end",
        "weights": weights,
        "conflicts": conflicts,
        "penalty": round(float(penalty), 3),
        "recommendation": recommendation,
        "hard_veto": bool(hard_veto),
    }
    shared_state.audit["arbitration"] = arbitration

    if isinstance(job_card, dict):
        job_card["unity_arbitration"] = arbitration
        job_card.setdefault("bidirectional_feedback", {})["unity"] = {
            "penalty": arbitration["penalty"],
            "reasons": [conflict["type"] for conflict in conflicts],
            "recommendation": recommendation,
            "hard_veto": bool(hard_veto),
        }
        if penalty > 0.0:
            confidence = job_card.get("confidence") or {}
            for key in ("route", "feasibility_if_specified"):
                value = confidence.get(key)
                if isinstance(value, (int, float)):
                    adjusted = max(0.0, min(1.0, float(value) * (1.0 - penalty)))
                    confidence[key] = round(adjusted, 3)
            confidence["unity_arbitration_penalty"] = round(float(penalty), 3)
            job_card["confidence"] = confidence
        if hard_veto:
            warnings = job_card.get("warnings") or []
            warnings.append("W_UNITY_ARBITRATION_VETO: cross-domain conflict detected.")
            job_card["warnings"] = list(dict.fromkeys(warnings))

    return arbitration


def _route_key_residues(route: Dict[str, Any]) -> List[str]:
    primary = str((route or {}).get("primary") or "").lower()
    if "serine" in primary or "hydrolase" in primary or "esterase" in primary:
        return ["Ser", "His", "Asp", "Glu"]
    if "metallo" in primary or "metal" in primary:
        return ["His", "Asp", "Glu", "Cys"]
    if "radical" in primary or "sam" in primary or "p450" in primary:
        return ["Cys", "His", "Asp", "Glu"]
    return list(PKA_CATALYTIC_GROUPS.keys())


def validate_contract(shared_state: SharedState, job_type: Optional[str] = None) -> List[str]:
    violations: List[str] = []
    input_spec = shared_state.input_spec or {}
    if not input_spec.get("smiles") and not input_spec.get("molblock"):
        violations.append("missing input_spec.smiles or input_spec.molblock")
    if not input_spec.get("target_bond"):
        violations.append("missing input_spec.target_bond")

    condition_profile = shared_state.condition_profile or {}
    temp_k = condition_profile.get("temperature_K")
    if not isinstance(temp_k, (int, float)):
        violations.append("condition_profile.temperature_K must be a float")
    elif float(temp_k) <= 0.0:
        violations.append("condition_profile.temperature_K must be > 0")
    ph_val = condition_profile.get("pH")
    if ph_val is not None and not isinstance(ph_val, (int, float)):
        violations.append("condition_profile.pH must be a float when provided")

    resolved_job_type = job_type or shared_state.chemistry.derived.get("job_type")
    if resolved_job_type == "REAGENT_GENERATION" and not input_spec.get("trap_target"):
        violations.append("trap_target required for REAGENT_GENERATION")

    def _check_energy_units(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if "kj" in str(key).lower():
                    if value is not None and not isinstance(value, (int, float)):
                        violations.append(f"{name} must be numeric kJ/mol")
                _check_energy_units(value, name)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                _check_energy_units(value, f"{prefix}[{idx}]")

    _check_energy_units(shared_state.physics.to_dict(), "physics")
    _check_energy_units(shared_state.chemistry.to_dict(), "chem")
    _check_energy_units(shared_state.biology.to_dict(), "bio")
    return violations
