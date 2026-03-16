from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

from enzyme_software.physicscore import c_to_k, k_to_c


@dataclass
class ConditionProfile:
    pH: Optional[float] = None
    temperature_K: Optional[float] = None
    temperature_C: Optional[float] = None
    temperature_defaulted: bool = False
    ionic_strength: Optional[float] = None
    solvent: Optional[str] = None
    cofactors: List[str] = field(default_factory=list)
    salts_buffer: Optional[str] = None
    constraints: Optional[str] = None

    def normalize(self) -> "ConditionProfile":
        if self.pH is not None:
            self.pH = round(float(self.pH), 3)
        if self.temperature_K is None and self.temperature_C is None:
            self.temperature_K = 298.15
            self.temperature_C = k_to_c(self.temperature_K)
            self.temperature_defaulted = True
        elif self.temperature_K is None and self.temperature_C is not None:
            self.temperature_K = c_to_k(self.temperature_C)
        elif self.temperature_K is not None and self.temperature_C is None:
            self.temperature_C = k_to_c(self.temperature_K)
        if self.temperature_K is not None:
            self.temperature_K = round(float(self.temperature_K), 2)
        if self.temperature_C is not None:
            self.temperature_C = round(float(self.temperature_C), 2)
        if self.ionic_strength is not None:
            self.ionic_strength = round(float(self.ionic_strength), 3)
        self.cofactors = [str(item).strip() for item in self.cofactors if str(item).strip()]
        return self

    def distance_to(self, other: "ConditionProfile") -> float:
        if other is None:
            return 1.0
        components = []
        if self.pH is not None and other.pH is not None:
            components.append(((self.pH - other.pH) / 14.0) ** 2)
        temp_self = self.temperature_K
        temp_other = other.temperature_K
        if temp_self is None and self.temperature_C is not None:
            temp_self = float(self.temperature_C) + 273.15
        if temp_other is None and other.temperature_C is not None:
            temp_other = float(other.temperature_C) + 273.15
        if temp_self is not None and temp_other is not None:
            components.append(((temp_self - temp_other) / 100.0) ** 2)
        if self.ionic_strength is not None and other.ionic_strength is not None:
            components.append(((self.ionic_strength - other.ionic_strength) / 1.0) ** 2)
        if not components:
            return 1.0
        return math.sqrt(sum(components))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pH": self.pH,
            "temperature_K": self.temperature_K,
            "temperature_C": self.temperature_C,
            "temperature_defaulted": self.temperature_defaulted,
            "ionic_strength": self.ionic_strength,
            "solvent": self.solvent,
            "cofactors": self.cofactors,
            "salts_buffer": self.salts_buffer,
            "constraints": self.constraints,
        }


@dataclass
class ReactionTask:
    bond_to_break_or_form: str
    substrates: List[str]
    products: Optional[List[str]] = None
    mechanism_hint: Optional[str] = None
    required_selectivity: Optional[str] = None
    allowed_scaffold_types: List[str] = field(default_factory=list)
    safety_constraints: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bond_to_break_or_form": self.bond_to_break_or_form,
            "substrates": self.substrates,
            "products": self.products,
            "mechanism_hint": self.mechanism_hint,
            "required_selectivity": self.required_selectivity,
            "allowed_scaffold_types": self.allowed_scaffold_types,
            "safety_constraints": self.safety_constraints,
        }


@dataclass
class Candidate:
    scaffold_id: str
    active_site_definition: Dict[str, Any]
    variant_definition: Dict[str, Any]
    predicted_mechanism: str
    features: Dict[str, Any]
    topology_signature: Dict[str, Any]
    causal_rationale: List[str]
    kinetic_estimates: Dict[str, Any]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scaffold_id": self.scaffold_id,
            "active_site_definition": self.active_site_definition,
            "variant_definition": self.variant_definition,
            "predicted_mechanism": self.predicted_mechanism,
            "features": self.features,
            "topology_signature": self.topology_signature,
            "causal_rationale": self.causal_rationale,
            "kinetic_estimates": self.kinetic_estimates,
            "confidence": self.confidence,
        }


@dataclass
class ExperimentRecord:
    reaction_task_fingerprint: str
    condition_profile: ConditionProfile
    candidate_fingerprint: str
    observed_success: float
    observed_rate_or_yield: Optional[float] = None
    notes: Optional[str] = None
    source_quality: float = 0.5
    route: Optional[str] = None
    substrate_bin: Optional[str] = None
    catalyst_family: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    weight: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reaction_task_fingerprint": self.reaction_task_fingerprint,
            "condition_profile": self.condition_profile.to_dict(),
            "candidate_fingerprint": self.candidate_fingerprint,
            "observed_success": self.observed_success,
            "observed_rate_or_yield": self.observed_rate_or_yield,
            "notes": self.notes,
            "source_quality": self.source_quality,
            "route": self.route,
            "substrate_bin": self.substrate_bin,
            "catalyst_family": self.catalyst_family,
            "metadata": self.metadata,
            "weight": self.weight,
        }


@dataclass
class BondSpec:
    target_bond: str
    target_bond_indices: Optional[List[int]] = None
    selection_mode: Optional[str] = None
    resolved_target: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_bond": self.target_bond,
            "target_bond_indices": self.target_bond_indices,
            "selection_mode": self.selection_mode,
            "resolved_target": self.resolved_target,
            "context": self.context,
        }


@dataclass
class SubstrateContext:
    smiles: str
    structure_summary: Optional[Dict[str, Any]] = None
    descriptors: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "structure_summary": self.structure_summary,
            "descriptors": self.descriptors,
        }


@dataclass
class TelemetryContext:
    run_id: str
    trace: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace": self.trace,
            "warnings": self.warnings,
        }


@dataclass
class SharedInput:
    bond_spec: BondSpec
    condition_profile: ConditionProfile
    substrate_context: SubstrateContext
    telemetry: TelemetryContext
    unity_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "bond_spec": self.bond_spec.to_dict(),
            "condition_profile": self.condition_profile.to_dict(),
            "substrate_context": self.substrate_context.to_dict(),
            "telemetry": self.telemetry.to_dict(),
        }
        if self.unity_state is not None:
            payload["unity_state"] = self.unity_state
        return payload


@dataclass
class SharedOutput:
    result: Dict[str, Any]
    given_conditions_effect: Dict[str, Any]
    optimum_conditions: Dict[str, Any]
    confidence: Dict[str, Any]
    retry_loop_suggestion: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "given_conditions_effect": self.given_conditions_effect,
            "optimum_conditions": self.optimum_conditions,
            "confidence": self.confidence,
            "retry_loop_suggestion": self.retry_loop_suggestion,
        }


@dataclass
class SharedIO:
    input: SharedInput
    outputs: Dict[str, SharedOutput] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "outputs": {key: value.to_dict() for key, value in self.outputs.items()},
        }


@dataclass
class FeatureVector:
    values: Dict[str, float]
    missing: List[str] = field(default_factory=list)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "missing": self.missing,
            "source": self.source,
        }


@dataclass
class EvidenceRecord:
    module_id: int
    inputs: Dict[str, Any]
    features_used: FeatureVector
    score: float
    confidence: float
    uncertainty: Dict[str, Any]
    optimum_conditions: Optional[Dict[str, Any]] = None
    explanations: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "inputs": self.inputs,
            "features_used": self.features_used.to_dict(),
            "score": self.score,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "optimum_conditions": self.optimum_conditions,
            "explanations": self.explanations,
            "diagnostics": self.diagnostics,
        }
