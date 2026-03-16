from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConditionProfile:
    pH: Optional[float] = None
    temperature_K: Optional[float] = None
    temperature_C: Optional[float] = None
    ionic_strength: Optional[float] = None
    solvent: Optional[str] = None
    cofactors: List[str] = field(default_factory=list)
    salts_buffer: Optional[str] = None
    constraints: Optional[Dict[str, float]] = None


@dataclass
class BondContext:
    bond_role: Optional[str] = None
    bond_role_confidence: Optional[float] = None
    bond_class: Optional[str] = None
    polarity: Optional[str] = None
    atom_count: Optional[int] = None
    hetero_atoms: Optional[int] = None
    ring_count: Optional[int] = None


@dataclass
class PhysicsAudit:
    deltaG_dagger_kJ_per_mol: Optional[float] = None
    eyring_k_s_inv: Optional[float] = None
    k_eff_s_inv: Optional[float] = None
    temperature_K: Optional[float] = None
    horizon_s: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class EnergyLedger:
    """Cross-module ledger of unit-consistent physical quantities."""
    deltaG_dagger_kJ: Optional[float] = None
    deltaG_dagger_variant_kJ: Optional[float] = None
    deltaG_bind_kJ: Optional[float] = None
    eyring_k_s_inv: Optional[float] = None
    k_diff_cap_s_inv: Optional[float] = None
    k_eff_s_inv: Optional[float] = None
    k_variant_s_inv: Optional[float] = None
    p_success_horizon: Optional[float] = None
    p_success_variant_horizon: Optional[float] = None
    horizon_s: Optional[float] = None
    ci90: Optional[tuple[float, float]] = None
    n_eff: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class UnityPhysicsState:
    priors: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    energy_ledger: Optional[EnergyLedger] = None


@dataclass
class UnityChemState:
    context: Dict[str, Any] = field(default_factory=dict)
    functional_group_map: Dict[str, Any] = field(default_factory=dict)
    reaction_family: Optional[str] = None
    leaving_group_score: Optional[float] = None
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnityBioState:
    protonation: Dict[str, Any] = field(default_factory=dict)
    enzyme_family_prior: Dict[str, Any] = field(default_factory=dict)
    residue_protonation_fraction: Optional[float] = None
    cofactor_requirements: Dict[str, Any] = field(default_factory=dict)
    mechanism_spec: Optional["MechanismSpec"] = None
    mechanism_contract: Optional[Dict[str, Any]] = None
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnityConstraintsState:
    condition_profile: ConditionProfile = field(default_factory=ConditionProfile)


@dataclass
class UnityScoringState:
    ledger: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnityContext:
    physics: UnityPhysicsState = field(default_factory=UnityPhysicsState)
    chem: UnityChemState = field(default_factory=UnityChemState)
    bio: UnityBioState = field(default_factory=UnityBioState)
    constraints: UnityConstraintsState = field(default_factory=UnityConstraintsState)
    scoring: UnityScoringState = field(default_factory=UnityScoringState)


@dataclass
class Module0Out:
    decision: Optional[str] = None
    route_family: Optional[str] = None
    route_confidence: Optional[float] = None
    data_support: Optional[float] = None


@dataclass
class Module1Out:
    status: Optional[str] = None
    access_score: Optional[float] = None
    reach_score: Optional[float] = None
    retention_score: Optional[float] = None
    top_scaffold: Optional[str] = None


@dataclass
class Module2Out:
    status: Optional[str] = None
    selected_scaffold: Optional[str] = None
    best_variant: Optional[str] = None
    deltaG_dagger_kJ_per_mol: Optional[float] = None
    k_pred_s_inv: Optional[float] = None
    route_family: Optional[str] = None


@dataclass
class Module3Out:
    status: Optional[str] = None
    plan_score: Optional[float] = None
    qc_status: Optional[str] = None
    batch_id: Optional[str] = None


@dataclass
class MechanismSpec:
    reaction_family: str
    route_label: str
    expected_nucleophile: Literal["Ser", "Cys", "Either"]
    expected_motif: str
    detected_nucleophile: Optional[str] = None
    detected_motif_residues: Dict[str, List[str]] = field(default_factory=dict)
    compatibility_score: float = 0.0
    mismatch_reason: Optional[str] = None
    policy_action: Literal[
        "KEEP_WITH_PENALTY", "SWITCH_ROUTE", "REQUEST_DISAMBIGUATION"
    ] = "KEEP_WITH_PENALTY"


@dataclass
class UnityRecord:
    run_id: str
    smiles: str
    target_bond: str
    requested_output: Optional[str] = None
    condition_profile: ConditionProfile = field(default_factory=ConditionProfile)
    bond_context: BondContext = field(default_factory=BondContext)
    physics_audit: Optional[PhysicsAudit] = None
    energy_ledger: Optional[EnergyLedger] = None
    chemistry: Optional[UnityChemState] = None
    biology: Optional[UnityBioState] = None
    module0: Optional[Module0Out] = None
    module1: Optional[Module1Out] = None
    module2: Optional[Module2Out] = None
    module3: Optional[Module3Out] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


def build_features(record: UnityRecord) -> Dict[str, float]:
    """Build a stable numeric feature vector (no strings)."""
    condition = record.condition_profile or ConditionProfile()
    bond = record.bond_context or BondContext()
    physics = record.physics_audit or PhysicsAudit()

    temperature_k = float(condition.temperature_K) if condition.temperature_K is not None else 0.0
    ph_value = float(condition.pH) if condition.pH is not None else 0.0
    delta_g = (
        float(physics.deltaG_dagger_kJ_per_mol)
        if physics.deltaG_dagger_kJ_per_mol is not None
        else 0.0
    )
    eyring_k = float(physics.eyring_k_s_inv) if physics.eyring_k_s_inv is not None else 0.0
    bond_role_conf = (
        float(bond.bond_role_confidence) if bond.bond_role_confidence is not None else 0.0
    )
    atom_count = float(bond.atom_count) if bond.atom_count is not None else 0.0
    hetero_atoms = float(bond.hetero_atoms) if bond.hetero_atoms is not None else 0.0
    ring_count = float(bond.ring_count) if bond.ring_count is not None else 0.0

    polarity_text = (bond.polarity or "").lower()
    polarity_encoded = (
        1.0
        if "polar" in polarity_text and "nonpolar" not in polarity_text and "non-polar" not in polarity_text
        else 0.0
    )

    route_family = None
    if record.module0 and record.module0.route_family:
        route_family = record.module0.route_family
    elif record.module2 and record.module2.route_family:
        route_family = record.module2.route_family
    route_family = route_family or "other"
    route_family_lower = route_family.lower()
    serine = 1.0 if "serine_hydrolase" in route_family_lower else 0.0
    metallo = 1.0 if "metallo_esterase" in route_family_lower else 0.0
    other = 1.0 if (serine + metallo) == 0.0 else 0.0

    return {
        "temperature_K": temperature_k,
        "pH": ph_value,
        "deltaG_dagger_kJ": delta_g,
        "eyring_k_s_inv": eyring_k,
        "bond_role_confidence": bond_role_conf,
        "substrate_atom_count": atom_count,
        "hetero_atoms": hetero_atoms,
        "ring_count": ring_count,
        "polarity_encoded": polarity_encoded,
        "route_family_serine_hydrolase": serine,
        "route_family_metallo_esterase": metallo,
        "route_family_other": other,
    }
