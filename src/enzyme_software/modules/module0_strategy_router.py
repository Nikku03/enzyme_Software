from __future__ import annotations

# Contract Notes (output contract freeze):
# - ctx.data["module0_strategy_router"] must preserve keys: status, job_card_ref, shared_io.
# - ctx.data["job_card"] is authoritative and consumed by UI/tests. Preserve keys including:
#   decision, confidence, difficulty_label/score, job_type, requested_output, structure_summary,
#   bond_context, resolved_target, mechanism_route, compute_plan, reaction_intent,
#   candidate_bonds/candidate_meta, module1_mode/module1_weights, scaffold_library_id,
#   route_posteriors, chosen_route, physics_audit, physics_layer,
#   predicted_under_given_conditions, optimum_conditions_estimate, delta_from_optimum,
#   evidence_record, math_contract, warnings/errors.
# - shared_io shape: {"input": {bond_spec, condition_profile, substrate_context, telemetry},
#   "outputs": {"module0": {result, given_conditions_effect, optimum_conditions,
#   confidence, retry_loop_suggestion}}}. Do not rename.
# - New physics fields must be added under job_card["physics_*"] or shared_io.outputs.module0["physics_*"].
# - Route priors/confidence computed in BayesianDAGRouter.predict plus fallback in
#   _heuristic_route_confidence and _physics_route_audit (see route_posteriors flow).

from collections import Counter
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.config import (
    ROUTE_CONFIDENCE_LOW_THRESHOLD,
    TARGET_RESOLUTION_LOW_THRESHOLD,
)
from enzyme_software.domain import (
    BondSpec,
    ConditionProfile,
    EvidenceRecord,
    FeatureVector,
    ReactionTask,
    SharedInput,
    SharedIO,
    SharedOutput,
    SubstrateContext,
    TelemetryContext,
)
from enzyme_software.mechanism_registry import resolve_mechanism
from enzyme_software.unity_schema import MechanismSpec
from enzyme_software.scorecard import (
    ScoreCard,
    ScoreCardMetric,
    calibration_status_from_signals,
    contributors_from_features,
    metric_status,
)
from enzyme_software.score_ledger import ScoreLedger, ScoreTerm
from enzyme_software.mathcore import (
    BayesianDAGRouter,
    ProbabilityEstimate,
    QCReport,
    beta_credible_interval,
    distribution_from_ci,
    extract_features,
    record_event,
    validate_math_contract,
)
from enzyme_software.mathcore.uncertainty import sigmoid
from enzyme_software.biocore import (
    cofactor_compatibility_penalty,
    enzyme_family_prior,
    mechanism_mismatch_penalty,
    residue_state_fraction,
)
from enzyme_software.chemcore import (
    PKA_CATALYTIC_GROUPS,
    chem_context_from_bond,
    fraction_deprotonated,
    fraction_protonated,
    solvent_penalty,
)
from enzyme_software.physicscore import (
    adjust_barrier_for_temperature,
    coulomb_energy_kj_mol,
    compute_route_prior,
    c_to_k,
    eyring_rate_constant,
    estimate_deltaG_dagger_for_bond,
    estimate_bond_length_A,
    format_rate,
    get_baseline_barrier,
    kinetics_event_probability,
    kinetics_from_context,
    physics_prior_success_probability,
    thermal_energy_kj_per_mol,
)
from enzyme_software.unity_schema import (
    BondContext as UnityBondContext,
    ConditionProfile as UnityConditionProfile,
    Module0Out as UnityModule0Out,
    PhysicsAudit as UnityPhysicsAudit,
    UnityRecord,
    build_features,
)
from enzyme_software.unity_layer import record_interlink
from enzyme_software.modules.base import BaseModule

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None
    rdMolDescriptors = None


DIFFICULTY_ORDER = ["EASY", "MEDIUM", "HARD"]

_CALIBRATION_CACHE: Dict[str, Any] = {"model": None, "source": None}
JOB_TYPE_STANDARD = "STANDARD_TRANSFORMATION"
JOB_TYPE_REAGENT_GENERATION = "REAGENT_GENERATION"
JOB_TYPE_ANALYSIS_ONLY = "MECHANISM_PROBE"
ROUTE_VERSION = "v1"
PHYSICS_POLARIZATION_RATIO_STRONG = 6.0  # ~6 kT indicates strong local polarization
DETECTION_THRESHOLD_CONVERSION = 0.10
DETECTION_NOISE_FLOOR = 0.02
DETECTABILITY_SIGMOID_WIDTH = 0.05
R_KJ_PER_MOL_K = 8.314e-3

# Module -1 radical/HAT-informed route prior wiring.
ROUTE_TO_HAT_MECHANISM: Dict[str, str] = {
    "p450": "Fe_IV_oxo_heme",
    "metallo_transfer_cf3": "Fe_IV_oxo_heme",
    "non_heme_iron": "Fe_IV_oxo_nonheme",
    "radical_sam": "radical_SAM",
    "monooxygenase": "Fe_IV_oxo_heme",
    "copper_radical_oxidase": "generic_radical",
    "amine_oxidase": "generic_radical",
}
ROUTE_TO_ELIGIBILITY_KEY: Dict[str, str] = {
    "p450": "p450_oxidation",
    "metallo_transfer_cf3": "p450_oxidation",
    "non_heme_iron": "monooxygenase",
    "radical_sam": "radical_transfer",
    "monooxygenase": "monooxygenase",
    "copper_radical_oxidase": "oxidoreductase",
    "amine_oxidase": "oxidoreductase",
}
ROUTE_COMPATIBILITY_PRIORS: Dict[str, Dict[str, float]] = {
    "hydrophobic_organic": {
        "P450": 0.3,
        "NHI": 1.5,
        "default": 1.0,
    },
    "polar_amino_acid_deriv": {
        "P450": 1.2,
        "NHI": 0.5,
        "default": 1.0,
    },
    "default": {
        "P450": 1.0,
        "NHI": 1.0,
        "default": 1.0,
    },
}
NHI_REFERENCE_BDE_BY_CLASS: Dict[str, float] = {
    "alpha_amino_CH": 385.0,
    "benzylic_CH": 370.0,
    "allylic_CH": 375.0,
    "aliphatic_CH": 400.0,
    "default": 400.0,
}

# BRENDA/Bar-Even anchored family medians used to map chemical-step physics to realistic
# whole-cycle turnover scales.
FAMILY_MEDIAN_KCAT_S_INV: Dict[str, float] = {
    "cytochrome_p450": 3.0,
    "non_heme_iron_oxygenase": 5.0,
    "serine_hydrolase": 50.0,
    "metallo_esterase": 20.0,
    "haloalkane_dehalogenase": 1.0,
    "unknown": 10.0,
}
FAMILY_REFERENCE_BDE_KJ_MOL: Dict[str, float] = {
    "cytochrome_p450": 400.0,
    "non_heme_iron_oxygenase": 400.0,
    "metalloenzyme_radical": 400.0,
}
EP_ALPHA_BY_FAMILY: Dict[str, float] = {
    "cytochrome_p450": 0.495,
    "non_heme_iron_oxygenase": 0.45,
    "metalloenzyme_radical": 0.45,
}


def _route_debug_enabled() -> bool:
    return str(os.environ.get("ENZYME_ROUTE_DEBUG", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

ROUTE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "ester": {
        "primary": "serine_hydrolase",
        "secondary": ["metallo_esterase"],
        "mechanisms": ["acyl-oxygen hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "serine_hydrolase", "weight": 0.9, "required": False},
            {"track": "metallo_esterase", "weight": 0.4, "required": False},
        ],
        "required_flags": [],
    },
    "amide": {
        "primary": "amidase",
        "secondary": ["serine_hydrolase", "metalloprotease"],
        "mechanisms": ["amide hydrolysis"],
        "cofactors": ["Zn2+ (optional)"],
        "expert_tracks": [
            {"track": "amidase", "weight": 0.8, "required": False},
            {"track": "serine_hydrolase", "weight": 0.6, "required": False},
            {"track": "metalloprotease", "weight": 0.5, "required": False},
        ],
        "required_flags": [],
    },
    "phosphate": {
        "primary": "phosphatase",
        "secondary": ["metallophosphatase"],
        "mechanisms": ["phosphoryl transfer"],
        "cofactors": ["Mg2+"],
        "expert_tracks": [
            {"track": "phosphatase", "weight": 0.8, "required": False},
            {"track": "metallophosphatase", "weight": 0.5, "required": False},
        ],
        "required_flags": [],
    },
    "thioester": {
        "primary": "thioesterase",
        "secondary": ["serine_hydrolase"],
        "mechanisms": ["thioester hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "thioesterase", "weight": 0.8, "required": False},
            {"track": "serine_hydrolase", "weight": 0.5, "required": False},
        ],
        "required_flags": [],
    },
    "C-H": {
        "primary": "P450",
        "secondary": ["radical_SAM", "non_heme_iron"],
        "mechanisms": ["C-H activation"],
        "cofactors": ["heme", "Fe-S", "Fe(II)/2OG"],
        "expert_tracks": [
            {
                "track": "oxidative_activation",
                "weight": 0.9,
                "required": True,
                "requires": ["metals", "oxidation"],
            },
            {"track": "radical_SAM", "weight": 0.7, "required": False},
        ],
        "required_flags": ["metals", "oxidation"],
    },
    "O-H": {
        "primary": "copper_radical_oxidase",
        "secondary": ["P450", "monooxygenase"],
        "mechanisms": ["alcohol oxidation", "radical relay"],
        "cofactors": ["Cu", "heme (optional)"],
        "expert_tracks": [
            {"track": "oxidoreductase", "weight": 0.8, "required": False},
            {"track": "monooxygenase", "weight": 0.5, "required": False},
        ],
        "required_flags": ["oxidation"],
    },
    "N-H": {
        "primary": "amine_oxidase",
        "secondary": ["monooxygenase"],
        "mechanisms": ["N-H oxidation", "dehydrogenation"],
        "cofactors": ["FAD", "Cu"],
        "expert_tracks": [
            {"track": "oxidoreductase", "weight": 0.7, "required": False},
            {"track": "monooxygenase", "weight": 0.4, "required": False},
        ],
        "required_flags": ["oxidation"],
    },
    "aromatic": {
        "primary": "monooxygenase",
        "secondary": ["dioxygenase"],
        "mechanisms": ["aromatic oxidation/cleavage"],
        "cofactors": ["heme", "Fe2+"],
        "expert_tracks": [
            {"track": "monooxygenase", "weight": 0.8, "required": True, "requires": ["metals", "oxidation"]},
            {"track": "dioxygenase", "weight": 0.6, "required": False},
        ],
        "required_flags": ["metals", "oxidation"],
    },
    "C-C": {
        "primary": "lyase",
        "secondary": ["oxygenase"],
        "mechanisms": ["C-C cleavage"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "lyase", "weight": 0.6, "required": False},
            {"track": "oxygenase", "weight": 0.4, "required": False},
        ],
        "required_flags": [],
    },
    "C-N": {
        "primary": "amidase",
        "secondary": ["deaminase"],
        "mechanisms": ["C-N hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "amidase", "weight": 0.6, "required": False},
            {"track": "deaminase", "weight": 0.4, "required": False},
        ],
        "required_flags": [],
    },
    "C-O": {
        "primary": "hydrolase",
        "secondary": [],
        "mechanisms": ["C-O hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "hydrolase", "weight": 0.6, "required": False},
        ],
        "required_flags": [],
    },
    "C-S": {
        "primary": "thioetherase",
        "secondary": [],
        "mechanisms": ["C-S hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "thioetherase", "weight": 0.5, "required": False},
        ],
        "required_flags": [],
    },
    "alkyl_halide": {
        "primary": "haloalkane_dehalogenase",
        "secondary": [],
        "mechanisms": ["SN2-like substitution", "dehalogenation"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "haloalkane_dehalogenase", "weight": 0.7, "required": False},
        ],
        "required_flags": [],
    },
    "aryl_halide": {
        "primary": "reductive_dehalogenase",
        "secondary": ["radical_transferase"],
        "mechanisms": ["reductive dehalogenation", "radical transfer"],
        "cofactors": ["Fe-S"],
        "expert_tracks": [
            {"track": "reductive_dehalogenase", "weight": 0.8, "required": True, "requires": ["metals"]},
            {"track": "radical_transferase", "weight": 0.5, "required": False},
        ],
        "required_flags": ["metals"],
    },
    "sulfonamide": {
        "primary": "sulfonamidase",
        "secondary": [],
        "mechanisms": ["sulfonamide hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "sulfonamidase", "weight": 0.6, "required": False},
        ],
        "required_flags": [],
    },
    "sulfate": {
        "primary": "sulfatase",
        "secondary": [],
        "mechanisms": ["sulfate ester hydrolysis"],
        "cofactors": ["Ca2+"],
        "expert_tracks": [
            {"track": "sulfatase", "weight": 0.7, "required": False},
        ],
        "required_flags": [],
    },
    "imine": {
        "primary": "imine_hydrolase",
        "secondary": [],
        "mechanisms": ["imine hydrolysis"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "imine_hydrolase", "weight": 0.6, "required": False},
        ],
        "required_flags": [],
    },
    "nitrile": {
        "primary": "nitrilase",
        "secondary": [],
        "mechanisms": ["nitrile hydration"],
        "cofactors": [],
        "expert_tracks": [
            {"track": "nitrilase", "weight": 0.6, "required": False},
        ],
        "required_flags": [],
    },
    "azo": {
        "primary": "azoreductase",
        "secondary": [],
        "mechanisms": ["azo reduction"],
        "cofactors": ["FAD"],
        "expert_tracks": [
            {"track": "azoreductase", "weight": 0.6, "required": False},
        ],
        "required_flags": [],
    },
    "disulfide": {
        "primary": "disulfide_reductase",
        "secondary": [],
        "mechanisms": ["disulfide reduction"],
        "cofactors": ["FAD"],
        "expert_tracks": [
            {"track": "disulfide_reductase", "weight": 0.6, "required": False},
        ],
        "required_flags": [],
    },
    "other": {
        "mechanisms": ["manual review required"],
        "primary": "manual_review",
        "secondary": [],
        "cofactors": [],
        "expert_tracks": [],
        "required_flags": [],
    },
}

CANONICAL_TOKENS: set[str] = {
    "ester__acyl_o",
    "acid__c_o",
    "ether__c_o",
    "amide__c_n",
    "lactam__c_n",
    "beta_lactam__c_n",
    "carbamate__c_n",
    "urea__c_n",
    "anhydride__acyl_o",
    "carbonate__acyl_o",
    "thioester__acyl_s",
    "thioether__c_s",
    "sulfonamide__s_n",
    "sulfate_ester__s_o",
    "alkyl_halide__c_x",
    "aryl_halide__c_x",
    "epoxide__c_o",
    "acetal__c_o",
    "imine__c_n",
    "nitrile__c_n",
    "azo__n_n",
    "disulfide__s_s",
    "phosphate_ester__p_o",
    "glycosidic__acetal_o",
    "ch__aliphatic",
    "ch__benzylic",
    "ch__allylic",
    "ch__alpha_hetero",
    "ch__fluorinated",
    "ch__aryl",
    "oh__alcohol",
    "oh__phenol",
    "nh__amine",
    "nh__amide",
    "cc__single",
    "cc__aryl_alkyl",
    "cc__aryl_aryl",
}

TOKEN_ALIASES: Dict[str, str] = {
    "acetyl_ester_c-o": "ester__acyl_o__acetyl",
    "ester_c-o": "ester__acyl_o",
    "amide_c-n": "amide__c_n",
    "aryl_c-br": "aryl_halide__c_x__br",
    "c-h_fluorinated": "ch__fluorinated",
    "o-h": "oh__alcohol",
    "oh": "oh__alcohol",
    "n-h": "nh__amine",
    "nh": "nh__amine",
}

TOKEN_SMARTS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ester__acyl_o": {"smarts": "[C:1](=O)[O:2]"},
    "ester__acyl_o__acetyl": {"smarts": "C[C:1](=O)[O:2]", "context": "acetyl"},
    "amide__c_n": {"smarts": "[C:1](=O)[N:2]"},
    "thioester__acyl_s": {"smarts": "[C:1](=O)[S:2]"},
    "aryl_halide__c_x": {"smarts": "[c:1][F,Cl,Br,I:2]"},
    "aryl_halide__c_x__br": {"smarts": "[c:1][Br:2]", "context": "br"},
    "alkyl_halide__c_x": {"smarts": "[C:1][F,Cl,Br,I:2]"},
}

FUNCTIONAL_GROUP_SMARTS: List[Tuple[str, str]] = [
    ("carboxylic_acid", "C(=O)[OX2H1]"),
    ("ester", "C(=O)O[CX4]"),
    ("amide", "C(=O)N"),
    ("carbamate", "O[C](=O)N"),
    ("urea", "N[C](=O)N"),
    ("anhydride", "C(=O)O[C](=O)"),
    ("carbonate", "O=C(O)O"),
    ("thioester", "C(=O)S"),
    ("thiol", "[SX2H]"),
    ("thioether", "[#6][S][#6]"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("sulfate_ester", "S(=O)(=O)O"),
    ("phosphate", "P(=O)(O)(O)"),
    ("alcohol", "[CX4][OX2H]"),
    ("phenol", "c[OX2H]"),
    ("ether", "[CX4]O[CX4]"),
    ("epoxide", "C1OC1"),
    ("acetal", "C(O)(O)"),
    ("imine", "C=N"),
    ("nitrile", "C#N"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("aryl_ring", "a1aaaaa1"),
    ("halide", "[F,Cl,Br,I]"),
]

PROTONATION_SMARTS: List[Dict[str, Any]] = [
    {
        "group": "carboxylic_acid",
        "smarts": "C(=O)[OX2H1]",
        "pka_range": (3.0, 5.0),
        "note": "Carboxylic acids deprotonate near pH 4-5.",
    },
    {
        "group": "amine",
        "smarts": "[NX3;H2,H1;!$(NC=O)]",
        "pka_range": (8.0, 10.5),
        "note": "Amines protonate below pH 8-10.",
    },
    {
        "group": "phenol",
        "smarts": "c[OX2H]",
        "pka_range": (9.0, 11.0),
        "note": "Phenols deprotonate at basic pH.",
    },
    {
        "group": "imidazole",
        "smarts": "c1ncc[nH]1",
        "pka_range": (6.0, 7.5),
        "note": "Imidazole tautomerization near neutral pH.",
    },
    {
        "group": "thiol",
        "smarts": "[SX2H]",
        "pka_range": (8.0, 10.0),
        "note": "Thiols deprotonate near neutral-basic pH.",
    },
]

ROLE_PRIORITY: Dict[str, int] = {
    "beta_lactam__c_n": 100,
    "lactam__c_n": 95,
    "carbamate__c_n": 92,
    "urea__c_n": 92,
    "amide__c_n": 90,
    "thioester__acyl_s": 90,
    "anhydride__acyl_o": 88,
    "carbonate__acyl_o": 88,
    "ester__acyl_o": 85,
    "acid__c_o": 85,
    "sulfonamide__s_n": 80,
    "sulfate_ester__s_o": 80,
    "phosphate_ester__p_o": 80,
    "epoxide__c_o": 70,
    "glycosidic__acetal_o": 70,
    "acetal__c_o": 65,
    "ether__c_o": 60,
    "thioether__c_s": 60,
    "alkyl_halide__c_x": 60,
    "aryl_halide__c_x": 65,
    "imine__c_n": 60,
    "nitrile__c_n": 70,
    "azo__n_n": 60,
    "disulfide__s_s": 60,
    "ch__fluorinated": 70,
    "ch__aryl": 65,
    "ch__benzylic": 60,
    "ch__allylic": 60,
    "ch__alpha_hetero": 60,
    "ch__aliphatic": 50,
    "oh__phenol": 62,
    "oh__alcohol": 58,
    "nh__amide": 60,
    "nh__amine": 55,
    "cc__aryl_aryl": 60,
    "cc__aryl_alkyl": 55,
    "cc__single": 50,
}

ROLE_FUNCTIONAL_GROUP_MAP: Dict[str, str] = {
    "acid__c_o": "acid",
    "ester__acyl_o": "ester",
    "anhydride__acyl_o": "anhydride",
    "carbonate__acyl_o": "carbonate",
    "amide__c_n": "amide",
    "lactam__c_n": "lactam",
    "beta_lactam__c_n": "beta_lactam",
    "carbamate__c_n": "carbamate",
    "urea__c_n": "urea",
    "thioester__acyl_s": "thioester",
    "thioether__c_s": "thioether",
    "sulfonamide__s_n": "sulfonamide",
    "sulfate_ester__s_o": "sulfate_ester",
    "phosphate_ester__p_o": "phosphate",
    "alkyl_halide__c_x": "alkyl_halide",
    "aryl_halide__c_x": "aryl_halide",
    "epoxide__c_o": "epoxide",
    "acetal__c_o": "acetal",
    "glycosidic__acetal_o": "glycosidic_acetal",
    "imine__c_n": "imine",
    "nitrile__c_n": "nitrile",
    "azo__n_n": "azo",
    "disulfide__s_s": "disulfide",
    "oh__alcohol": "alcohol",
    "oh__phenol": "phenol",
    "nh__amine": "amine",
    "nh__amide": "amide_nh",
}


@dataclass
class TargetBondSpec:
    kind: str
    raw: str
    indices: Optional[Tuple[int, int]] = None
    elements: Optional[Tuple[str, str]] = None
    token: Optional[str] = None
    token_base: Optional[str] = None
    token_context: Optional[str] = None
    smarts: Optional[str] = None
    bond_map: Optional[Tuple[int, int]] = None
    warnings: List[str] = field(default_factory=list)


class Module0StrategyRouter(BaseModule):
    name = "Module 0 - Strategy Router (RDKit)"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not hasattr(ctx, "bayes_router"):
            ctx.bayes_router = BayesianDAGRouter()
        shared_io = ctx.data.get("shared_io") or {}
        module_minus1 = (
            (shared_io.get("outputs") or {}).get("module_minus1")
            or ctx.data.get("module_minus1")
        )
        result = route_job(
            smiles=ctx.smiles,
            target_bond=ctx.target_bond,
            requested_output=ctx.requested_output,
            trap_target=ctx.trap_target,
            constraints=ctx.constraints,
            experiment_records=ctx.data.get("experiment_records"),
            unity_state=ctx.data.get("unity_state"),
            strict_validation=bool(ctx.data.get("strict_validation", True)),
            module_minus1=module_minus1,
        )
        job_card = result.get("job_card")
        ctx.data["job_card"] = job_card
        if result.get("shared_io"):
            ctx.data["shared_io"] = result["shared_io"]
        module0_payload = {key: value for key, value in result.items() if key != "job_card"}
        if job_card is not None:
            module0_payload["job_card_ref"] = "job_card"
            if job_card.get("scorecard") is not None:
                module0_payload["scorecard"] = job_card.get("scorecard")
            if job_card.get("score_ledger") is not None:
                module0_payload["score_ledger"] = job_card.get("score_ledger")
        ctx.data["module0_strategy_router"] = module0_payload
        if job_card and job_card.get("constraints"):
            ctx.constraints = OperationalConstraints(**job_card["constraints"])
        record_interlink(
            ctx,
            0,
            reads=[
                "constraints.condition_profile",
                "chem.context",
            ],
            writes=[
                "physics.energy_ledger",
                "chem.reaction_family",
                "bio.mechanism_contract",
                "scoring.ledger",
            ],
        )
        _update_unity_record_parts(ctx, module0_payload, job_card)
        return ctx


def _update_unity_record_parts(
    ctx: PipelineContext, module_output: Dict[str, Any], job_card: Optional[Dict[str, Any]]
) -> None:
    parts = ctx.data.setdefault("unity_record_parts", {})
    parts["module0"] = {
        "module_output": module_output,
        "job_card": job_card or {},
    }


def route_job(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    trap_target: Optional[str],
    constraints: OperationalConstraints,
    experiment_records: Optional[List[Any]] = None,
    unity_state: Optional[Dict[str, Any]] = None,
    strict_validation: bool = True,
    module_minus1: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    reasons: List[str] = []

    target_spec = _parse_target_bond(target_bond)
    if target_spec.warnings:
        warnings.extend(target_spec.warnings)
    if target_spec.kind == "unknown":
        errors.append("Target bond format not recognized.")

    mol, mol_warnings, mol_errors = _load_molecule(smiles)
    warnings.extend(mol_warnings)
    errors.extend(mol_errors)

    rdkit_available = mol is not None

    job_type, job_type_notes = _infer_job_type(requested_output)
    if job_type_notes:
        warnings.extend(job_type_notes)

    effective_constraints, assumptions_used = _apply_constraint_defaults(constraints)
    success_definition = _success_definition(job_type, None)

    trap_target_spec = _normalize_trap_target(trap_target)
    selection_mode = "atom_indices" if target_spec.kind == "indices" else target_spec.kind
    resolved = {
        "selection_mode": selection_mode,
        "requested": target_bond,
        "canonical_token": target_spec.token,
        "token_context": target_spec.token_context,
        "index_base": None,
        "element_pair": None,
    }
    candidate_bonds: List[Any] = []
    candidate_atoms: List[int] = []
    candidate_bond_options: List[Dict[str, Any]] = []
    candidate_meta: Dict[str, Any] = {
        "mode": target_spec.kind,
        "scanned_bonds": None,
        "matches": None,
    }
    equivalence_groups: List[Dict[str, Any]] = []
    bond_context: Dict[str, Any] = {}
    ambiguous_groups = 0
    descriptor_status = {"required": False, "complete": True, "missing": []}
    small_gas_flag = False
    token_info: Optional[Dict[str, Any]] = None
    token_resolution_audit: Dict[str, Any] = {}
    target_resolution_confidence = 0.0
    target_resolution_audit: Dict[str, Any] = {}
    selected_entry: Optional[Dict[str, Any]] = None
    structure_summary: Dict[str, Any] = {}
    substrate_protonation_flags: List[Dict[str, Any]] = []
    module1_mode = "standard"
    module1_weights = {"access": 0.35, "reach": 0.45, "retention": 0.20}
    substrate_size_proxies: Dict[str, Any] = {}
    bond_center_hint: Dict[str, Any] = {}
    token_context_matched = False
    requested_output_check: Dict[str, Any] = _requested_output_check(
        requested_output=requested_output,
        reaction_intent={},
        bond_context=bond_context,
        token_context=target_spec.token_context,
        token_context_matched=False,
    )

    pipeline_halt_reason = None

    if errors:
        decision = "NO_GO"
        confidence = {
            "feasibility_if_specified": 0.0,
            "completeness": 0.0,
            "route": 0.0,
            "target_resolution": 0.0,
            "wetlab_prior": 0.0,
            "wetlab_prior_target_spec": 0.0,
            "wetlab_prior_any_activity": 0.0,
        }
        reaction_condition_field = _reaction_condition_field(
            job_type=job_type,
            reaction_intent={},
            route={},
            constraints=effective_constraints,
        )
        condition_profile = _condition_profile_from_constraints(effective_constraints).to_dict()
        reaction_task = _reaction_task_from_inputs(
            smiles=smiles,
            target_bond=target_bond,
            requested_output=requested_output,
            reaction_intent={},
            route={},
        )
        causal_discovery = _causal_discovery_summary({}, {}, reaction_condition_field)
        job_card = _build_job_card(
            decision=decision,
            confidence=confidence,
            difficulty="HARD",
            difficulty_score=1.0,
            job_type=job_type,
            requested_output=requested_output,
            trap_target=trap_target_spec,
            trap_target_raw=trap_target,
            resolved=resolved,
            structure_summary=structure_summary,
            reaction_intent={},
            bond_context=bond_context,
            equivalence_groups=equivalence_groups,
            candidate_bond_options=candidate_bond_options,
            candidate_meta=candidate_meta,
            constraints=effective_constraints,
            constraints_assumed_defaults=False,
            assumptions_used=assumptions_used,
            descriptor_status=descriptor_status,
            substrate_protonation_flags=substrate_protonation_flags,
            success_definition=success_definition,
            module1_mode=module1_mode,
            module1_weights=module1_weights,
            substrate_size_proxies=substrate_size_proxies,
            bond_center_hint=bond_center_hint,
            requested_output_check=requested_output_check,
            reaction_condition_field=reaction_condition_field,
            condition_profile=condition_profile,
            reaction_task=reaction_task,
            causal_discovery=causal_discovery,
            pipeline_halt_reason=pipeline_halt_reason,
            required_next_input=[],
            reasons=reasons,
            warnings=warnings,
            errors=errors,
            route_version=ROUTE_VERSION,
            scaffold_library_id=None,
            target_resolution_audit={},
            token_resolution_audit={},
        )
        return {"status": "no_go", "job_card": job_card}

    bond_roles: List[Dict[str, Any]] = []
    if rdkit_available:
        bond_roles = _assign_bond_roles(mol)
    structure_summary = (
        _compute_structure_summary(mol, bond_roles) if rdkit_available else {}
    )

    if rdkit_available:
        resolution = _resolve_target_bonds(
            mol,
            target_spec,
            bond_roles,
            requested_output,
            sre_output=module_minus1,
        )
        warnings.extend(resolution["warnings"])
        errors.extend(resolution["errors"])
        candidate_bonds = resolution["bonds"]
        candidate_atoms = resolution["candidate_atoms"]
        candidate_bond_options = resolution["candidate_bonds"]
        candidate_meta = resolution.get("candidate_meta", candidate_meta)
        token_info = resolution.get("token_info")
        token_resolution_audit = resolution.get("token_resolution_audit") or {}
        selected_entry = resolution.get("selected_entry")
        target_resolution_confidence = resolution.get("target_resolution_confidence", 0.0)
        equivalence_groups = resolution["equivalence_groups"]
        resolved.update(resolution["resolved"])
        if selected_entry is None and candidate_bond_options:
            top_candidate = candidate_bond_options[0]
            atom_indices = top_candidate.get("atom_indices") or []
            matched_bond = None
            if atom_indices and candidate_bonds:
                wanted = set(atom_indices)
                for bond in candidate_bonds:
                    bond_atoms = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
                    if bond_atoms == wanted:
                        matched_bond = bond
                        break
            if matched_bond is not None:
                selected_entry = _entry_from_bond(bond_roles, matched_bond)
                if selected_entry is None:
                    selected_entry = _fallback_entry_from_bond(matched_bond)
        if selected_entry is None and candidate_bonds:
            selected_entry = _entry_from_bond(bond_roles, candidate_bonds[0])
            if selected_entry is None:
                selected_entry = _fallback_entry_from_bond(candidate_bonds[0])
        if selected_entry and not resolved.get("match_count"):
            resolved["match_count"] = resolution.get("match_count") or len(candidate_bonds)
        if target_spec.kind == "indices":
            target_resolution_confidence = 0.99
        if selected_entry and target_resolution_confidence <= 0.0:
            target_resolution_confidence = _compute_target_resolution_confidence(
                target_spec.kind,
                int(resolved.get("match_count") or 1),
                selected_entry,
            )
        if "match_count" not in resolved:
            resolved["match_count"] = resolution.get("match_count")
        resolved["index_base"] = resolution["index_base"]
        resolved["element_pair"] = resolution["element_pair"]
        ambiguous_groups = len(equivalence_groups)

        if errors:
            # Check for semantic validation before early return
            early_validation = validate_request(
                {
                    "requested_output": requested_output,
                    "target_bond": target_bond,
                    "target_spec": target_spec,
                    "bond_context": bond_context,
                    "structure_summary": structure_summary,
                    "reaction_intent": {},
                    "job_type": job_type,
                },
                strict_validation=True,
            )
            if early_validation.get("decision") == "HALT":
                decision = "HALT"
                pipeline_halt_reason = early_validation.get("halt_reason")
                warnings.extend(early_validation.get("warnings") or [])
            else:
                decision = "NO_GO"
            confidence = {
                "feasibility_if_specified": 0.0,
                "completeness": 0.0,
                "route": 0.0,
                "target_resolution": 0.0,
                "wetlab_prior": 0.0,
                "wetlab_prior_target_spec": 0.0,
                "wetlab_prior_any_activity": 0.0,
            }
            required_next_input: List[str] = []
            if decision != "HALT" and any(
                err
                in {
                    "No bonds matched the requested token.",
                    "No bonds match the requested element pair.",
                    "No matching implicit hydrogen bonds found.",
                    "SMARTS pattern did not match the molecule.",
                }
                for err in errors
            ):
                pipeline_halt_reason = pipeline_halt_reason or "M0_NO_MATCH"
                required_next_input.append("target_bond")
            reaction_condition_field = _reaction_condition_field(
                job_type=job_type,
                reaction_intent={},
                route={},
                constraints=effective_constraints,
            )
            condition_profile = _condition_profile_from_constraints(effective_constraints).to_dict()
            reaction_task = _reaction_task_from_inputs(
                smiles=smiles,
                target_bond=target_bond,
                requested_output=requested_output,
                reaction_intent={},
                route={},
            )
            causal_discovery = _causal_discovery_summary({}, {}, reaction_condition_field)
            job_card = _build_job_card(
                decision=decision,
                confidence=confidence,
                difficulty="HARD",
                difficulty_score=1.0,
                job_type=job_type,
                requested_output=requested_output,
                trap_target=trap_target_spec,
                trap_target_raw=trap_target,
                resolved=resolved,
                structure_summary=structure_summary,
                reaction_intent={},
                bond_context=bond_context,
                equivalence_groups=equivalence_groups,
                candidate_bond_options=candidate_bond_options,
                candidate_meta=candidate_meta,
                constraints=effective_constraints,
                constraints_assumed_defaults=False,
                assumptions_used=assumptions_used,
                descriptor_status=descriptor_status,
                substrate_protonation_flags=substrate_protonation_flags,
                success_definition=success_definition,
                module1_mode=module1_mode,
                module1_weights=module1_weights,
                substrate_size_proxies=substrate_size_proxies,
                bond_center_hint=bond_center_hint,
                requested_output_check=requested_output_check,
                reaction_condition_field=reaction_condition_field,
                condition_profile=condition_profile,
                reaction_task=reaction_task,
                causal_discovery=causal_discovery,
                pipeline_halt_reason=pipeline_halt_reason,
                required_next_input=required_next_input,
                reasons=reasons,
                warnings=warnings,
                errors=errors,
                route_version=ROUTE_VERSION,
                scaffold_library_id=None,
                target_resolution_audit={},
                token_resolution_audit=token_resolution_audit,
            )
            return {"status": "no_go", "job_card": job_card}

        if selected_entry:
            bond_context = _bond_context_from_entry(mol, selected_entry)
        elif candidate_bonds:
            bond_context = _bond_context_from_bond(mol, candidate_bonds[0])
        elif candidate_bond_options:
            bond_context = _bond_context_from_candidate(candidate_bond_options[0])
        else:
            bond_context = _bond_context_from_elements(target_spec)
        if (
            target_spec.kind == "token"
            and target_spec.token_context
            and selected_entry is not None
        ):
            token_context_matched = _entry_has_context_tag(
                selected_entry, target_spec.token_context
            )
    else:
        bond_context = _bond_context_from_elements(target_spec)
        warnings.append("RDKit unavailable; bond validity not verified.")

    if selected_entry:
        resolved["selected_bond"] = _entry_to_candidate(selected_entry, mol)
    elif candidate_bond_options:
        resolved["selected_bond"] = candidate_bond_options[0]

    bond_context = _apply_role_overrides(bond_context)
    match_count = resolved.get("match_count")
    target_resolution_audit = _target_resolution_audit(
        match_count,
        candidate_bond_options,
        selected_entry,
        selection_mode=resolved.get("selection_mode"),
    )
    target_resolution_confidence = float(target_resolution_audit.get("confidence", 0.0) or 0.0)
    resolved["target_resolution_confidence"] = target_resolution_confidence
    resolved["ambiguous"] = bool(target_resolution_audit.get("ambiguous"))
    if target_resolution_audit.get("match_count") is not None:
        resolved["match_count"] = target_resolution_audit.get("match_count")

    descriptor_info, descriptor_warnings, descriptor_status, small_gas_flag = (
        _compute_polarity_descriptors(
            mol=mol,
            candidate_bonds=candidate_bonds,
            candidate_atoms=candidate_atoms,
            bond_context=bond_context,
        )
    )
    bond_context.update(descriptor_info)
    warnings.extend(descriptor_warnings)

    bond_type = bond_context.get("bond_type", "other")
    bond_class = bond_context.get("bond_class", bond_type)
    difficulty_label, difficulty_score = _assess_difficulty(
        bond_context,
        structure_summary,
    )
    reaction_intent = _infer_reaction_intent(
        requested_output=requested_output,
        bond_context=bond_context,
        job_type=job_type,
    )
    reaction_intent = _boost_reaction_intent_confidence(
        reaction_intent=reaction_intent,
        bond_context=bond_context,
        match_count=resolved.get("match_count"),
        token_context_matched=token_context_matched,
    )
    success_definition = _success_definition(job_type, reaction_intent)
    requested_output_check = _requested_output_check(
        requested_output=requested_output,
        reaction_intent=reaction_intent,
        bond_context=bond_context,
        token_context=target_spec.token_context,
        token_context_matched=token_context_matched,
    )
    validation_result = validate_request(
        {
            "requested_output": requested_output,
            "target_bond": target_bond,
            "target_spec": target_spec,
            "bond_context": bond_context,
            "structure_summary": structure_summary,
            "reaction_intent": reaction_intent,
            "job_type": job_type,
        },
        strict_validation=strict_validation,
    )
    warnings.extend(validation_result.get("warnings") or [])
    validation_halt_reason = None
    validation_message = None
    if validation_result.get("clear_requested_output"):
        requested_output = None
        job_type, job_type_notes = _infer_job_type(requested_output)
        if job_type_notes:
            warnings.extend(job_type_notes)
        reaction_intent = _infer_reaction_intent(
            requested_output=requested_output,
            bond_context=bond_context,
            job_type=job_type,
        )
        reaction_intent = _boost_reaction_intent_confidence(
            reaction_intent=reaction_intent,
            bond_context=bond_context,
            match_count=resolved.get("match_count"),
            token_context_matched=token_context_matched,
        )
        success_definition = _success_definition(job_type, reaction_intent)
        requested_output_check = _requested_output_check(
            requested_output=requested_output,
            reaction_intent=reaction_intent,
            bond_context=bond_context,
            token_context=target_spec.token_context,
            token_context_matched=token_context_matched,
        )
        warnings.append("Requested output cleared due to strict_validation=False.")
    if validation_result.get("decision") == "HALT":
        validation_halt_reason = validation_result.get("halt_reason")
        validation_message = validation_result.get("message")
    if ambiguous_groups > 1:
        difficulty_score = min(1.0, difficulty_score + 0.05)
    if job_type == JOB_TYPE_REAGENT_GENERATION and not trap_target_spec:
        difficulty_score = min(1.0, difficulty_score + 0.05)
    if job_type == JOB_TYPE_REAGENT_GENERATION and difficulty_score < 0.33:
        difficulty_score = 0.33
    difficulty_label = _difficulty_label_from_score(difficulty_score)
    route = _select_route(bond_context)
    route = _augment_route_for_reagent_generation(
        route,
        job_type,
        bond_class,
        requested_output,
    )
    route = _augment_route_with_token_info(route, token_info)
    reaction_task = _reaction_task_from_inputs(
        smiles=smiles,
        target_bond=target_bond,
        requested_output=requested_output,
        reaction_intent=reaction_intent,
        route=route,
    )

    constraints_assumed_defaults = False
    effective_constraints, assumptions_used, constraints_assumed_defaults = (
        _apply_route_constraint_defaults(effective_constraints, route, assumptions_used)
    )
    reaction_condition_field = _reaction_condition_field(
        job_type=job_type,
        reaction_intent=reaction_intent,
        route=route,
        constraints=effective_constraints,
    )
    condition_profile_obj = _condition_profile_from_constraints(effective_constraints)
    condition_profile = condition_profile_obj.to_dict()
    causal_discovery = _causal_discovery_summary(reaction_intent, route, reaction_condition_field)
    module1_mode, module1_weights = _module1_mode_from_context(
        bond_context,
        structure_summary,
    )
    substrate_size_proxies = _substrate_size_proxies(structure_summary)
    bond_center_hint = _bond_center_hint(bond_context, resolved)
    substrate_protonation_flags = _estimate_protonation_flags(
        mol,
        effective_constraints,
    )

    violations, constraint_warnings = _check_constraints(effective_constraints, route)
    warnings.extend(constraint_warnings)
    if violations:
        errors.extend(violations)

    required_next_input: List[str] = []
    force_review = False
    fatal_halt = False
    match_count = resolved.get("match_count")
    if validation_halt_reason:
        pipeline_halt_reason = validation_halt_reason
        reasons.append(validation_message or "Requested output incompatible with target bond.")
        fatal_halt = True
    if descriptor_status["required"] and not descriptor_status["complete"]:
        force_review = True
        warnings.append(
            "W_DESCRIPTOR_INCOMPLETE: Polarity proxies unavailable; review required."
        )
    if small_gas_flag:
        force_review = True
        warnings.append(
            "W_SMALL_GAS_SUBSTRATE: binding/orientation control is difficult; success depends on trap/transfer mechanism."
        )
    if reaction_intent.get("requires_trap_target") and not trap_target_spec:
        pipeline_halt_reason = "M0_MISSING_TRAP_TARGET"
        reasons.append("Trap/acceptor target missing for reagent generation.")
        required_next_input.append("trap_target")
        fatal_halt = True
    if requested_output_check.get("match") is False:
        warnings.append(
            "WARN_REQUESTED_OUTPUT_MISMATCH: requested output may not align with predicted transformation."
        )
        reasons.append("Requested output may not align with predicted transformation.")
        force_review = True
    unity_state = unity_state or {}
    unity_chem = unity_state.get("chem") or {}
    unity_sre = unity_state.get("sre") or {}
    sre_payload = _normalize_sre_payload_for_physics(module_minus1, unity_sre if isinstance(unity_sre, dict) else None)
    sre_reactivity = unity_sre.get("reactivity") if isinstance(unity_sre, dict) else {}
    sre_competition = (
        sre_reactivity.get("competition") if isinstance(sre_reactivity, dict) else {}
    )
    sre_gap = sre_competition.get("gap") if isinstance(sre_competition, dict) else None
    sre_equivalent = bool(
        isinstance(sre_competition, dict)
        and sre_competition.get("equivalent_sites_detected")
    )
    sre_disambiguation = bool(
        sre_equivalent
        or (
            isinstance(sre_gap, (int, float))
            and abs(float(sre_gap)) <= 1e-9
        )
    )

    if match_count == 0 and target_spec.kind in {"token", "elements", "smarts"}:
        reasons.append("No bonds matched the requested target.")
        required_next_input.append("target_bond")
        if str(job_type).upper() == "REAGENT_GENERATION":
            pipeline_halt_reason = pipeline_halt_reason or "M0_TARGET_CLARIFICATION_REQUIRED"
            force_review = True
            warnings.append(
                "W_TARGET_UNRESOLVED_REAGENT_MODE: waiting for explicit target bond selection."
            )
            suggestions = _find_matching_tokens_for_molecule(smiles)
            if suggestions:
                warnings.append(
                    f"W_TARGET_ALTERNATIVES: consider {', '.join(suggestions)}"
                )
        else:
            pipeline_halt_reason = pipeline_halt_reason or "M0_NO_MATCH"
            fatal_halt = True
    ambiguous_target = bool(resolved.get("ambiguous"))

    selected_bond_exists = bool(resolved.get("selected_bond"))
    if selected_bond_exists and match_count is None:
        match_count = 1
        resolved["match_count"] = 1
    if selected_bond_exists and target_resolution_confidence < TARGET_RESOLUTION_LOW_THRESHOLD.value:
        ambiguous_target = True
    if ambiguous_target and selected_bond_exists:
        warnings.append(
            "W_TARGET_AMBIGUOUS: Target bond resolution is low confidence; proceeding with top-ranked candidate."
        )
    if sre_disambiguation:
        ambiguous_target = True
        force_review = True
        reasons.append("Module -1 found equivalent best attack sites; explicit disambiguation required.")
        warnings.append("W_SRE_EQUIVALENT_SITES: explicit atom_indices recommended.")
    resolved["ambiguous"] = bool(ambiguous_target)

    condition_score = reaction_condition_field.get("condition_feasibility", {}).get(
        "given_conditions_score"
    )
    feature_context = {
        "bond_context": bond_context,
        "route": route,
        "reaction_intent": reaction_intent,
        "descriptor_status": descriptor_status,
        "warnings": warnings,
        "target_resolution_confidence": target_resolution_confidence,
        "match_count": match_count,
        "condition_score": condition_score,
        "job_type": job_type,
        "data_support": 0.7 if route.get("primary") else 0.5,
        "novelty_penalty": 0.1 if difficulty_label == "HARD" else 0.0,
        "source": "module0",
    }
    route_candidates = []
    if route.get("primary"):
        route_candidates.append(route["primary"])
    route_candidates.extend(route.get("secondary") or [])
    route_candidates = list(dict.fromkeys(route_candidates))
    if isinstance(reaction_task, dict):
        allowed_scaffolds = reaction_task.get("allowed_scaffold_types")
    else:
        allowed_scaffolds = getattr(reaction_task, "allowed_scaffold_types", None)
    if allowed_scaffolds is None:
        allowed_scaffolds = []
    known_scaffold = bool(route.get("scaffold_libraries") or allowed_scaffolds)
    unity_state = unity_state or {}
    unity_chem = unity_state.get("chem") or {}
    unity_sre = unity_state.get("sre") or {}
    # Annotate resolved dict with Module -1 findings if available
    sre_resolved = unity_sre.get("resolved_target") or {} if isinstance(unity_sre, dict) else {}
    sre_bond_indices = sre_resolved.get("bond_indices") if isinstance(sre_resolved, dict) else None
    if isinstance(sre_bond_indices, list) and len(sre_bond_indices) == 2:
        resolved["sre_resolved"] = True
        resolved["sre_bond_indices"] = sre_bond_indices
        resolved["sre_bond_type"] = sre_resolved.get("bond_type")
        resolved["sre_attack_sites"] = sre_resolved.get("attack_sites")
    chem_context = unity_chem.get("context")
    if not isinstance(chem_context, dict) or not chem_context:
        chem_context = chem_context_from_bond(
            bond_context, {"structure_summary": structure_summary}
        )
    physics_route_priors = _physics_route_priors(
        route_candidates,
        smiles,
        target_bond,
        bond_context,
        structure_summary,
        condition_profile_obj,
        chem_context=chem_context,
        known_scaffold=known_scaffold,
        metals_allowed=effective_constraints.metals_allowed,
        nucleophile_geometry=bond_context.get("nucleophile_geometry"),
        sre_payload=sre_payload,
    )

    router = BayesianDAGRouter()
    if experiment_records:
        router.update_from_records(experiment_records)
    router_prediction = router.predict(
        task=reaction_task,
        candidates=[],
        conditions=condition_profile_obj,
        routes=route_candidates,
    )
    chosen_route = router_prediction.get("chosen_route")
    if chosen_route and chosen_route in route_candidates:
        route["primary"] = chosen_route
        route["secondary"] = [r for r in route_candidates if r != chosen_route]

    if router_prediction.get("router_empty", True) and physics_route_priors:
        best_route = max(
            physics_route_priors.items(),
            key=lambda item: item[1].get("prior_feasibility") or 0.0,
        )[0]
        if best_route in route_candidates:
            chosen_route = best_route
            route["primary"] = best_route
            route["secondary"] = [r for r in route_candidates if r != best_route]

    physics_audit = _physics_route_audit(
        smiles,
        target_bond,
        bond_context,
        structure_summary,
        reaction_intent,
        condition_profile_obj,
        mechanism_family=chosen_route or route.get("primary"),
        chem_context=chem_context,
        known_scaffold=known_scaffold,
        metals_allowed=effective_constraints.metals_allowed,
        nucleophile_geometry=bond_context.get("nucleophile_geometry"),
        sre_payload=sre_payload,
    )
    physics_audit["route_priors"] = {
        name: payload.get("prior_feasibility") for name, payload in physics_route_priors.items()
    }
    router_empty = router_prediction.get("router_empty", True)
    fallback_used = bool(router_empty)
    route_p_raw = router_prediction.get("route_p_raw")
    route_p_cal = router_prediction.get("route_p_cal")
    route_ci90 = router_prediction.get("route_ci90")
    n_eff = router_prediction.get("n_eff")
    evidence_strength = router_prediction.get("evidence_strength")
    physics_shrink_factor = None
    if router_empty:
        heuristic_confidence = _heuristic_route_confidence(
            bond_context=bond_context,
            route=route,
            warnings=warnings,
            descriptor_incomplete=descriptor_status["required"] and not descriptor_status["complete"],
        )
        selected_physics = None
        if chosen_route and chosen_route in physics_route_priors:
            selected_physics = physics_route_priors[chosen_route].get(
                "prior_success_probability"
            )
        route_confidence = _blend_route_confidence(
            heuristic_confidence,
            selected_physics,
        )
        confidence_label = "Low"
        support = 0.0
        uncertainty = 0.25
        explanation = ["Router has no historical data; fallback to heuristic routing."]
        if isinstance(selected_physics, (int, float)):
            explanation.append("Physics prior blended with heuristic confidence.")
        route_p_raw = round(route_confidence, 3)
        route_p_cal = round(route_confidence, 3)
        evidence_strength = 0.0
        n_eff = 2.0
        route_ci90 = beta_credible_interval(route_confidence, n_eff)
        physics_shrink_factor = _physics_shrink_factor(support, evidence_strength)
        physics_route_priors = _apply_physics_shrink(physics_route_priors, physics_shrink_factor)
        if route_candidates:
            matched_bins = router_prediction.get("matched_bins") or {}
            condition_bin = matched_bins.get("condition_bin", "unknown")
            substrate_bin = matched_bins.get("substrate_bin", "unknown")
            catalyst_bin = matched_bins.get("catalyst_family_bin", "unknown")
            drivers = explanation
            route_weights: Dict[str, float] = {}
            total_weight = 0.0
            for route_name in route_candidates:
                route_phys = physics_route_priors.get(route_name, {})
                weight = route_phys.get("prior_feasibility")
                if not isinstance(weight, (int, float)):
                    weight = 0.5
                weight = max(0.0, float(weight))
                route_weights[route_name] = weight
                total_weight += weight
            if total_weight <= 0.0:
                total_weight = float(len(route_candidates)) or 1.0
                route_weights = {route_name: 1.0 for route_name in route_candidates}
            route_posteriors = []
            for route_name in route_candidates:
                route_phys = physics_route_priors.get(route_name, {})
                phys_weight = route_weights.get(route_name, 0.0) / total_weight
                posterior = _blend_route_confidence(heuristic_confidence, phys_weight)
                route_posteriors.append(
                    {
                        "route": route_name,
                        "posterior": round(posterior, 3),
                        "p_raw": round(posterior, 3),
                        "p_cal": round(posterior, 3),
                        "support": 0.0,
                        "confidence": confidence_label,
                        "uncertainty": round(uncertainty, 4),
                        "bucket": [route_name, condition_bin, substrate_bin, catalyst_bin],
                        "drivers": drivers,
                        "ci90": beta_credible_interval(posterior, n_eff),
                        "n_eff": round(float(n_eff), 2),
                        "evidence_strength": evidence_strength,
                        "prior_feasibility": route_phys.get("prior_feasibility"),
                    }
                )
            router_prediction["route_posteriors"] = route_posteriors
            router_prediction["route_p_raw"] = route_p_raw
            router_prediction["route_p_cal"] = route_p_cal
            router_prediction["route_ci90"] = route_ci90
            router_prediction["n_eff"] = n_eff
            router_prediction["evidence_strength"] = evidence_strength
    else:
        route_posteriors = router_prediction.get("route_posteriors") or []
        selected_entry = None
        if chosen_route:
            selected_entry = next(
                (entry for entry in route_posteriors if entry.get("route") == chosen_route),
                None,
            )
        if selected_entry is None and route_posteriors:
            selected_entry = route_posteriors[0]
        route_confidence = float(
            (selected_entry or {}).get("posterior", 0.5)
        )
        confidence_label = router_prediction.get("confidence_label", "Low")
        support = router_prediction.get("data_support", 0.0)
        uncertainty = router_prediction.get("uncertainty", 0.25)
        explanation = router_prediction.get("explanation", [])
        route_p_raw = route_p_raw if route_p_raw is not None else route_confidence
        route_p_cal = route_p_cal if route_p_cal is not None else route_confidence
        route_ci90 = route_ci90 or beta_credible_interval(route_confidence, n_eff or 2.0)
        physics_shrink_factor = _physics_shrink_factor(support, evidence_strength)
        physics_route_priors = _apply_physics_shrink(physics_route_priors, physics_shrink_factor)

    normalized_route_posteriors = _normalize_route_posteriors(
        route_candidates=route_candidates,
        route_posteriors=router_prediction.get("route_posteriors"),
        physics_route_priors=physics_route_priors,
    )
    router_prediction["route_posteriors"] = normalized_route_posteriors
    if not chosen_route and normalized_route_posteriors:
        chosen_route = normalized_route_posteriors[0].get("route")
        if chosen_route and chosen_route in route_candidates:
            route["primary"] = chosen_route
            route["secondary"] = [r for r in route_candidates if r != chosen_route]
    route_debug = score_all_routes(
        route_candidates=route_candidates,
        physics_route_priors=physics_route_priors,
        router_prediction=router_prediction,
        chosen_route=chosen_route or route.get("primary"),
        support=support,
        evidence_strength=evidence_strength,
        fallback_used=fallback_used,
    )
    route_confidence = float(
        (route_debug.get("confidence_components") or {}).get("confidence", route_confidence)
    )
    confidence_label = _confidence_label_from_score(route_confidence)
    explanation = list(dict.fromkeys((explanation or []) + ((route_debug.get("calibration") or {}).get("reasons") or [])))
    uncertainty = max(
        float(uncertainty or 0.0),
        min(0.95, 0.15 + 0.6 * float((route_debug.get("confidence_components") or {}).get("ambiguity_score") or 0.0)),
    )
    route_gap = route_debug.get("route_gap")
    ambiguity_flag = bool(route_debug.get("ambiguity_flag"))
    evidence_conflicts = list(route_debug.get("evidence_conflicts") or [])

    if physics_shrink_factor is None:
        physics_shrink_factor = _physics_shrink_factor(support, evidence_strength)
        physics_route_priors = _apply_physics_shrink(physics_route_priors, physics_shrink_factor)

    physics_prior_success_raw = physics_audit.get("prior_success_probability")
    physics_prior_success = _shrink_probability(physics_prior_success_raw, physics_shrink_factor)
    physics_audit["support"] = round(float(support or 0.0), 3)
    physics_audit["evidence_strength"] = round(float(evidence_strength or 0.0), 3)
    physics_audit["shrink_factor"] = round(float(physics_shrink_factor), 3)
    physics_audit["prior_success_probability_raw"] = physics_prior_success_raw
    if physics_audit.get("prior_before_damping") is None:
        physics_audit["prior_before_damping"] = physics_prior_success_raw
    physics_audit["prior_success_probability"] = (
        round(float(physics_prior_success), 4) if physics_prior_success is not None else None
    )
    physics_audit["prior_after_damping"] = physics_audit.get("prior_success_probability")
    physics_audit["prior_success_probability_final"] = physics_audit.get(
        "prior_success_probability"
    )

    if (
        isinstance(physics_prior_success, (int, float))
        and physics_prior_success < 0.1
        and confidence_label == "High"
    ):
        confidence_label = "Low"

    physics_prior = _physics_prior(bond_context, condition_profile_obj)
    physics_prior = _fill_physics_layer_xh(
        physics_prior, bond_context, sre_payload=sre_payload, mol=mol
    )
    route_confidence_raw = route_confidence
    confidence_adjustments: List[str] = []
    if physics_prior.get("prior_score") is not None:
        multiplier = 0.85 + 0.15 * float(physics_prior["prior_score"])
        route_confidence = max(0.0, min(1.0, route_confidence * multiplier))
        physics_prior["multiplier"] = round(float(multiplier), 3)
        confidence_adjustments.append(
            f"physics_prior_multiplier={round(float(multiplier), 3)}"
        )
    physics_prior["route_confidence_raw"] = round(float(route_confidence_raw), 3)
    physics_prior["route_confidence_physics"] = round(float(route_confidence), 3)
    if route_ci90:
        route_ci90 = [min(1.0, val * physics_prior["multiplier"]) for val in route_ci90]

    if isinstance(condition_score, (int, float)) and condition_score < 0.45:
        warnings.append(
            "W_CONDITION_MISMATCH: given pH/temperature may limit feasibility."
        )

    evidence_features = extract_features(feature_context)
    evidence_record = EvidenceRecord(
        module_id=0,
        inputs={
            "smiles": smiles,
            "target_bond": target_bond,
            "requested_output": requested_output,
            "job_type": job_type,
            "condition_profile": condition_profile,
        },
        features_used=FeatureVector(
            values=evidence_features.values,
            missing=evidence_features.missing,
            source="module0",
        ),
        score=round(route_confidence, 3),
        confidence=round(route_confidence, 3),
        uncertainty={"uncertainty_90ci": route_ci90},
        optimum_conditions=reaction_condition_field.get("optimum_conditions_hint"),
        explanations=explanation,
        diagnostics={
            "confidence_label": confidence_label,
            "data_support": support,
            "uncertainty_proxy": uncertainty,
            "evidence_strength": evidence_strength,
            "n_eff": n_eff,
            "matched_bins": router_prediction.get("matched_bins"),
        },
    ).to_dict()
    bayes_result = {
        "probability": route_confidence,
        "uncertainty_90ci": route_ci90,
        "drivers": explanation,
        "diagnostics": {
            "confidence_label": confidence_label,
            "data_support": support,
            "uncertainty_proxy": uncertainty,
            "evidence_strength": evidence_strength,
            "n_eff": n_eff,
        },
    }
    causal_discovery = _augment_causal_discovery(causal_discovery, bayes_result)
    wetlab_prior_target, wetlab_prior_any = _estimate_wetlab_priors(
        difficulty_score,
        job_type,
    )
    target_resolution = target_resolution_confidence
    if ambiguous_target:
        ambiguity_penalty = 0.6 + 0.4 * max(0.0, min(1.0, target_resolution))
        route_confidence = max(0.0, min(1.0, route_confidence * ambiguity_penalty))
        reasons.append("Target bond ambiguity penalized route confidence.")
        confidence_adjustments.append(
            f"target_ambiguity_penalty={round(float(ambiguity_penalty), 3)}"
        )
    if isinstance(unity_sre, dict):
        sre_status = unity_sre.get("status")
        sre_conf = unity_sre.get("confidence_prior")
        if sre_status == "PASS" and isinstance(sre_conf, (int, float)) and float(sre_conf) >= 0.75:
            route_confidence = max(0.0, min(1.0, route_confidence + 0.05))
            reasons.append("Module -1 high-confidence pass provided routing uplift.")
            confidence_adjustments.append("module_minus1_uplift=+0.05")
    if sre_disambiguation:
        route_confidence = max(0.0, min(1.0, route_confidence * 0.8))
        confidence_adjustments.append("sre_disambiguation_penalty=0.8")

    route_debug["confidence_components"]["confidence_before_context_adjustments"] = round(
        float(route_confidence_raw),
        6,
    )
    route_debug["confidence_components"]["confidence_after_context_adjustments"] = round(
        float(route_confidence),
        6,
    )
    route_debug["confidence_components"]["audit_score"] = route_debug["confidence_components"].get(
        "audit_score"
    )
    route_debug["confidence_components"]["ambiguity_score"] = route_debug[
        "confidence_components"
    ].get("ambiguity_score")
    route_debug["calibration"]["context_adjustments"] = confidence_adjustments
    route_debug["calibration"]["reasons"] = list(
        dict.fromkeys(
            list(route_debug["calibration"].get("reasons") or [])
            + confidence_adjustments
            + (["ambiguous_target"] if ambiguous_target else [])
        )
    )
    route_debug["evidence_conflicts"] = list(
        dict.fromkeys(
            list(route_debug.get("evidence_conflicts") or [])
            + (["target_bond_ambiguity"] if ambiguous_target else [])
            + (["module_minus1_requires_disambiguation"] if sre_disambiguation else [])
        )
    )
    confidence_label = _confidence_label_from_score(route_confidence)
    route_debug["confidence_components"]["confidence"] = round(float(route_confidence), 6)
    route_debug["confidence_components"]["confidence_label"] = confidence_label

    completeness = 1.0
    if reaction_intent.get("requires_trap_target") and not trap_target_spec:
        completeness = 0.0
    confidence = {
        "feasibility_if_specified": round(route_confidence, 3),
        "completeness": round(completeness, 3),
        "route": round(route_confidence * completeness, 3),
        "target_resolution": round(target_resolution, 3),
        "wetlab_prior": round(wetlab_prior_target, 3),
        "wetlab_prior_target_spec": round(wetlab_prior_target, 3),
        "wetlab_prior_any_activity": round(wetlab_prior_any, 3),
    }
    overall_conf = (route_confidence * completeness) * target_resolution * wetlab_prior_target

    decision = "GO"
    needs_review = False
    if errors:
        decision = "NO_GO"
        reasons.append("Constraint or validation failure.")
    elif fatal_halt:
        decision = "HALT" if validation_halt_reason else "NO_GO"
    else:
        if not rdkit_available:
            needs_review = True
            reasons.append("RDKit unavailable; unable to confirm bond validity.")

        if sre_disambiguation:
            decision = "HALT_NEED_SELECTION"
            pipeline_halt_reason = pipeline_halt_reason or "M0_NEEDS_DISAMBIGUATION"
            required_next_input.append("target_bond_selection")
        elif match_count == 0 or not selected_bond_exists:
            decision = "HALT_NEED_SELECTION"
            pipeline_halt_reason = pipeline_halt_reason or "M0_TARGET_RESOLUTION_LOW"
            required_next_input.append("target_bond_selection")
            reasons.append("Target bond resolution requires explicit selection.")
        else:
            if route_confidence < 0.4:
                decision = "NO_GO"
            elif route_confidence < 0.8:
                decision = "LOW_CONF_GO"
            else:
                decision = "GO"

            if decision == "GO" and overall_conf < 0.5:
                decision = "LOW_CONF_GO"
                reasons.append("Overall confidence below threshold; proceed with caution.")

            if force_review and decision == "GO":
                decision = "LOW_CONF_GO"

            if ambiguous_target:
                needs_review = True
                reasons.append("Target bond ambiguous; review recommended.")

        if force_review:
            needs_review = True
            if "Review required based on routing heuristics." not in reasons:
                reasons.append("Review required based on routing heuristics.")

    if not reasons:
        reasons.append("Bond and constraints appear feasible for routing.")

    if required_next_input:
        required_next_input = list(dict.fromkeys(required_next_input))

    compute_plan = None
    if decision in {"GO", "LOW_CONF_GO"}:
        compute_plan = _compute_plan(
            difficulty_label,
            job_type,
            decision,
            force_review=force_review,
            route_confidence=route_confidence,
            target_resolution=target_resolution,
            reaction_intent=reaction_intent,
        )
        if force_review or pipeline_halt_reason:
            compute_plan["active"] = False
        else:
            compute_plan["active"] = True

    scaffold_library_id = _scaffold_library_id_from_route(route)
    job_card = _build_job_card(
        decision=decision,
        confidence=confidence,
        difficulty=difficulty_label,
        difficulty_score=difficulty_score,
        job_type=job_type,
        requested_output=requested_output,
        trap_target=trap_target_spec,
        trap_target_raw=trap_target,
        resolved=resolved,
        structure_summary=structure_summary,
        reaction_intent=reaction_intent,
        bond_context=bond_context,
        equivalence_groups=equivalence_groups,
        candidate_bond_options=candidate_bond_options,
        candidate_meta=candidate_meta,
        constraints=effective_constraints,
        constraints_assumed_defaults=constraints_assumed_defaults,
        assumptions_used=assumptions_used,
        descriptor_status=descriptor_status,
        substrate_protonation_flags=substrate_protonation_flags,
        success_definition=success_definition,
        module1_mode=module1_mode,
        module1_weights=module1_weights,
        substrate_size_proxies=substrate_size_proxies,
        bond_center_hint=bond_center_hint,
        requested_output_check=requested_output_check,
        reaction_condition_field=reaction_condition_field,
        condition_profile=condition_profile,
        reaction_task=reaction_task,
        causal_discovery=causal_discovery,
        pipeline_halt_reason=pipeline_halt_reason,
        required_next_input=required_next_input,
        reasons=reasons,
        warnings=warnings,
        errors=errors,
        route_version=ROUTE_VERSION,
        scaffold_library_id=scaffold_library_id,
        route=route,
        compute_plan=compute_plan,
        target_resolution_audit=target_resolution_audit,
        token_resolution_audit=token_resolution_audit,
    )
    job_card["physics_layer"] = physics_prior
    job_card["physics_audit"] = physics_audit
    job_card["chem_context"] = chem_context
    reaction_identity = _reaction_identity_payload(
        smiles=smiles,
        target_bond=target_bond,
        requested_output=requested_output,
        trap_target=trap_target,
        resolved=job_card.get("resolved_target") or {},
    )
    job_card["reaction_identity"] = reaction_identity
    reaction_family = _reaction_family_from_context(bond_context, reaction_intent)
    mechanism_policy = os.environ.get("MECHANISM_POLICY", "exploratory").strip().lower()
    chemistry_contract = {
        "functional_group_map": structure_summary.get("functional_groups") or {},
        "reaction_family": reaction_family,
        "leaving_group_score": chem_context.get("leaving_group_quality"),
    }
    job_card["chemistry_contract"] = chemistry_contract
    job_card["chemistry_audit"] = {
        "reaction_family": reaction_family,
        "leaving_group_score": chem_context.get("leaving_group_quality"),
        "functional_group_map_keys": list(
            (structure_summary.get("functional_groups") or {}).keys()
        ),
    }
    job_card["mechanism_policy"] = mechanism_policy
    job_card["mechanism_spec"] = _build_mechanism_spec(
        chosen_route or route.get("primary") or "unknown",
        reaction_family,
    )
    mechanism_contract = _build_mechanism_contract(
        chosen_route or route.get("primary") or "unknown",
        reaction_family,
    )
    job_card["mechanism_contract"] = mechanism_contract
    job_card["mechanism_mismatch"] = _default_mechanism_mismatch(mechanism_contract)
    enzyme_prior = enzyme_family_prior(chosen_route or route.get("primary") or "unknown")
    residue_info = _residue_fraction_for_route(
        chosen_route or route.get("primary") or "unknown",
        condition_profile.get("pH"),
        physics_audit.get("protonation_residue"),
    )
    cofactor_requirements = {
        "requires_metals": enzyme_prior.get("profile", {}).get("requires_metals"),
        "note": "v1 cofactor requirements from enzyme family profile",
    }
    job_card["bio_protonation"] = {
        "route": chosen_route or route.get("primary"),
        "factor": physics_audit.get("f_protonation"),
        "residue": physics_audit.get("protonation_residue"),
        "notes": physics_audit.get("protonation_notes"),
        "uncertain": physics_audit.get("protonation_uncertain"),
    }
    job_card["energy_ledger"] = _energy_ledger_from_job_card(job_card)
    biology_contract = {
        "enzyme_family_prior": enzyme_prior,
        "residue_protonation_fraction": residue_info,
        "cofactor_requirements": cofactor_requirements,
    }
    job_card["biology_contract"] = biology_contract
    job_card["biology_audit"] = {
        "enzyme_family": enzyme_prior.get("family"),
        "residue_protonation_fraction": residue_info,
        "cofactor_requirements": cofactor_requirements,
    }
    physics_selected = None
    if chosen_route and chosen_route in physics_route_priors:
        physics_selected = physics_route_priors[chosen_route]
    elif route.get("primary") in physics_route_priors:
        physics_selected = physics_route_priors[route.get("primary")]
    mapping_notes = [
        "Physics prior derived from compute_route_prior (ΔG‡ → Eyring k).",
        "Converted Eyring rate to event probability over horizon.",
        "Applied context uncertainty penalties when solvent/cofactors/scaffold are missing.",
        "Physics prior represents feasibility under horizon; not certainty.",
    ]
    if physics_selected and physics_selected.get("notes"):
        mapping_notes.extend(physics_selected.get("notes"))
    if physics_selected and physics_selected.get("prior_any_activity_calibrated") is not None:
        mapping_notes.append("Calibration artifact applied to any-activity prior.")
    job_card["physics"] = {
        "baseline": physics_selected or physics_audit,
        "routes": physics_route_priors,
        "p_event_hour": physics_selected.get("p_event_hour")
        if physics_selected
        else physics_audit.get("p_event_hour"),
        "route_prior_any_activity": physics_selected.get("route_prior_any_activity")
        if physics_selected
        else physics_audit.get("route_prior_any_activity"),
        "route_prior_target_specific": physics_selected.get("route_prior_target_specific")
        if physics_selected
        else physics_audit.get("route_prior_target_specific"),
        "prior_success_probability": physics_selected.get("prior_success_probability")
        if physics_selected
        else physics_audit.get("prior_success_probability"),
        "mapping_notes": list(dict.fromkeys(mapping_notes)),
    }
    job_card["predicted_under_given_conditions"] = {
        "route_success_probability": round(route_confidence, 3),
        "confidence_calibrated": round(route_confidence, 3),
        "uncertainty_90ci": route_ci90,
        "route_p_raw": route_p_raw,
        "route_p_cal": route_p_cal,
        "n_eff": n_eff,
        "evidence_strength": evidence_strength,
    }
    job_card["chosen_route"] = chosen_route or route.get("primary")
    job_card["route_posteriors"] = router_prediction.get("route_posteriors")
    job_card["confidence_label"] = confidence_label
    job_card["data_support"] = support
    job_card["route_uncertainty"] = uncertainty
    job_card["route_explanation"] = explanation
    job_card["matched_bins"] = router_prediction.get("matched_bins")
    job_card["route_p_raw"] = route_p_raw
    job_card["route_p_cal"] = route_p_cal
    job_card["route_ci90"] = route_ci90
    job_card["n_eff"] = n_eff
    job_card["evidence_strength"] = evidence_strength
    job_card["top_routes"] = route_debug.get("top_routes")
    job_card["route_gap"] = route_gap
    job_card["ambiguity_flag"] = ambiguity_flag
    job_card["fallback_used"] = fallback_used
    job_card["evidence_conflicts"] = evidence_conflicts
    job_card["route_confidence_components"] = route_debug.get("confidence_components")
    job_card["calibration"] = route_debug.get("calibration")
    job_card["router_debug"] = route_debug
    route_audit_payload = {
        "selected_route": chosen_route or route.get("primary"),
        "top_routes": route_debug.get("top_routes"),
        "route_gap": route_gap,
        "posterior_gap": route_debug.get("posterior_gap"),
        "ambiguity_flag": ambiguity_flag,
        "fallback_used": fallback_used,
        "evidence_conflicts": evidence_conflicts,
        "confidence_components": route_debug.get("confidence_components"),
        "calibration": route_debug.get("calibration"),
        "evaluated_routes": route_debug.get("evaluated_routes"),
    }
    job_card["route_audit"] = route_audit_payload
    physics_audit["route_debug"] = route_debug
    physics_audit["ambiguity_flag"] = ambiguity_flag
    physics_audit["route_gap"] = route_gap
    physics_audit["fallback_used"] = fallback_used
    physics_audit["evidence_conflicts"] = evidence_conflicts
    job_card["optimum_conditions_estimate"] = (
        reaction_condition_field.get("optimum_conditions_hint") or {}
    )
    job_card["delta_from_optimum"] = _delta_from_optimum(
        reaction_condition_field,
        condition_profile,
    )
    job_card["confidence_calibrated"] = round(route_confidence, 3)
    job_card["evidence_record"] = evidence_record
    confidence_estimate = ProbabilityEstimate(
        p_raw=float(route_p_raw or route_confidence),
        p_cal=float(route_p_cal or route_confidence),
        ci90=(float(route_ci90[0]), float(route_ci90[1])) if route_ci90 else (0.0, 1.0),
        n_eff=float(n_eff or 2.0),
    ).to_dict()
    prediction_estimate = distribution_from_ci(
        mean=float(route_p_cal or route_confidence),
        ci90=(float(route_ci90[0]), float(route_ci90[1])) if route_ci90 else (0.0, 1.0),
    ).to_dict()
    math_contract = {
        "confidence": confidence_estimate,
        "predictions": {"route_success_probability": prediction_estimate},
        "qc": QCReport(status="N/A", reasons=[], metrics={}).to_dict(),
    }
    job_card["math_contract"] = math_contract
    job_card["scorecard"] = _build_scorecard_module0(
        job_card=job_card,
        route_confidence=route_confidence,
        overall_confidence=overall_conf,
        route_p_cal=route_p_cal,
        route_ci90=route_ci90,
        n_eff=n_eff,
        evidence_strength=evidence_strength,
        data_support=support,
    )
    job_card["score_ledger"] = _build_score_ledger_module0(
        job_card=job_card,
        route_confidence=route_confidence,
        overall_confidence=overall_conf,
    )
    given_conditions = reaction_condition_field.get("given_conditions") or {}
    if not given_conditions:
        given_conditions = {
            "pH": condition_profile.get("pH"),
            "temperature_c": condition_profile.get("temperature_C"),
        }
    retry_suggestion = _retry_loop_suggestion(
        given=given_conditions,
        optimum=reaction_condition_field.get("optimum_conditions_hint") or {},
    )
    physics_block = _normalized_physics_block(job_card.get("physics_audit"))
    chemistry_block = _normalized_chemistry_block(job_card)
    biology_block = _normalized_biology_block(job_card)
    mechanism_block = _normalized_mechanism_block(job_card)
    energy_ledger = job_card.get("energy_ledger") or {}
    reaction_identity = job_card.get("reaction_identity") or {}
    shared_output = SharedOutput(
        result={
            "decision": decision,
            "chosen_route": chosen_route or route.get("primary"),
            "route_posteriors": router_prediction.get("route_posteriors"),
            "top_routes": route_debug.get("top_routes"),
            "route_gap": route_gap,
            "ambiguity_flag": ambiguity_flag,
            "fallback_used": fallback_used,
        },
        given_conditions_effect={
            "route_success_probability": round(route_confidence, 3),
            "condition_score": condition_score,
            "route_p_raw": route_p_raw,
            "route_p_cal": route_p_cal,
            "route_ci90": route_ci90,
            "route_gap": route_gap,
        },
        optimum_conditions={
            "strategy_prior": reaction_condition_field.get("optimum_conditions_hint") or {},
            "source": "module0",
        },
        confidence={
            "calibrated_probability": round(route_confidence, 3),
            "label": confidence_label,
            "support": support,
            "uncertainty": uncertainty,
            "ambiguity_flag": ambiguity_flag,
            "audit_score": (route_debug.get("confidence_components") or {}).get("audit_score"),
        },
        retry_loop_suggestion=retry_suggestion,
    )
    telemetry_warnings = []
    if condition_profile_obj.temperature_defaulted:
        telemetry_warnings.append(
            "temperature_K defaulted to 298.15 K (no input temperature provided)."
        )
    shared_io = _build_shared_io(
        smiles=smiles,
        target_bond=target_bond,
        resolved=job_card.get("resolved_target") or {},
        bond_context=bond_context,
        structure_summary=structure_summary,
        condition_profile=condition_profile_obj,
        run_id=None,
        output=shared_output,
        physics_block=physics_block,
        chemistry_block=chemistry_block,
        biology_block=biology_block,
        mechanism_block=mechanism_block,
        energy_ledger=energy_ledger,
        reaction_identity=reaction_identity,
        telemetry_warnings=telemetry_warnings,
    )
    contract_violations = validate_math_contract(job_card)
    if contract_violations:
        job_card["warnings"] = list(
            dict.fromkeys((job_card.get("warnings") or []) + contract_violations)
        )
    record_event(
        {
            "module": "module0",
            "status": decision,
            "route_confidence": route_confidence,
            "chosen_route": chosen_route or route.get("primary"),
            "target_bond": target_bond,
            "job_type": job_type,
            "given_conditions": reaction_condition_field.get("given_conditions"),
        }
    )
    if decision == "NO_GO":
        status = "no_go"
    elif decision == "HALT_NEED_SELECTION":
        status = "halt"
    else:
        status = "ok"
    return {
        "status": status,
        "job_card": job_card,
        "shared_io": shared_io,
        "route_audit": job_card.get("route_audit") if isinstance(job_card, dict) else None,
        "router_debug": job_card.get("router_debug") if isinstance(job_card, dict) else None,
    }


def _parse_target_bond(target_bond: str) -> TargetBondSpec:
    index_match = re.match(r"^\s*\[?\s*(\d+)\s*[,;:-]\s*(\d+)\s*\]?\s*$", target_bond)
    if index_match:
        return TargetBondSpec(
            kind="indices",
            raw=target_bond,
            indices=(int(index_match.group(1)), int(index_match.group(2))),
        )

    smarts_match = re.search(r"smarts\s*[:=]\s*([^;]+)", target_bond, re.IGNORECASE)
    if smarts_match:
        smarts = smarts_match.group(1).strip()
        bond_match = re.search(r"bond\s*[:=]?\s*(\d+)\s*[-:]\s*(\d+)", target_bond)
        bond_map = None
        if bond_match:
            bond_map = (int(bond_match.group(1)), int(bond_match.group(2)))
        else:
            bond_map = _extract_smarts_map_pair(smarts)
        return TargetBondSpec(
            kind="smarts",
            raw=target_bond,
            smarts=smarts,
            bond_map=bond_map,
        )

    token_key = _normalize_token_text(target_bond)
    canonical_token, token_base, token_context, token_warnings = _canonicalize_token(
        token_key
    )
    if canonical_token:
        return TargetBondSpec(
            kind="token",
            raw=target_bond,
            token=canonical_token,
            token_base=token_base,
            token_context=token_context,
            warnings=token_warnings,
        )

    element_match = re.match(r"^\s*([A-Za-z]{1,2})\s*[-:]\s*([A-Za-z]{1,2})\s*$", target_bond)
    if element_match:
        element_a = element_match.group(1).capitalize()
        element_b = element_match.group(2).capitalize()
        return TargetBondSpec(
            kind="elements",
            raw=target_bond,
            elements=(element_a, element_b),
        )

    embedded_match = re.findall(
        r"([A-Za-z]{1,2})\s*(?:\([^)]*\))?\s*[-:–—]\s*([A-Za-z]{1,2})",
        target_bond,
    )
    if embedded_match:
        left, right = embedded_match[-1]
        return TargetBondSpec(
            kind="elements",
            raw=target_bond,
            elements=(left.capitalize(), right.capitalize()),
        )

    relaxed = re.split(r"[-:–—]", target_bond)
    if len(relaxed) == 2:
        left, right = relaxed[0].strip(), relaxed[1].strip()
        left_elem = _first_element_symbol(left)
        right_elem = _first_element_symbol(right)
        if left_elem and right_elem:
            return TargetBondSpec(
                kind="elements",
                raw=target_bond,
                elements=(left_elem, right_elem),
            )

    if "_" in token_key or "__" in token_key:
        warnings = ["Token not recognized; unable to infer element pair."]
        return TargetBondSpec(kind="unknown", raw=target_bond, warnings=warnings)

    return TargetBondSpec(kind="unknown", raw=target_bond)


def _find_matching_tokens_for_molecule(smiles: str) -> List[str]:
    if Chem is None:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    candidates = []
    smarts_map = {
        "ester_c-o": "[CX3](=O)[OX2][#6]",
        "amide_c-n": "[CX3](=O)[NX3]",
        "aryl_c-br": "[c][Br]",
        "aryl_c-cl": "[c][Cl]",
        "aryl_halide": "[c][F,Cl,Br,I]",
    }
    for token, smarts in smarts_map.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if mol.GetSubstructMatches(patt, uniquify=True):
            candidates.append(token)
    # C-H needs explicit Hs
    try:
        mol_h = Chem.AddHs(mol)
        patt_ch = Chem.MolFromSmarts("[CX4]-[H]")
        if patt_ch and mol_h.GetSubstructMatches(patt_ch, uniquify=True):
            candidates.append("ch__aliphatic")
        patt_oh = Chem.MolFromSmarts("[O]-[H]")
        if patt_oh and mol_h.GetSubstructMatches(patt_oh, uniquify=True):
            candidates.append("oh__alcohol")
        patt_nh = Chem.MolFromSmarts("[N]-[H]")
        if patt_nh and mol_h.GetSubstructMatches(patt_nh, uniquify=True):
            candidates.append("nh__amine")
    except Exception:
        pass
    return list(dict.fromkeys(candidates))


def _normalize_token_text(token: str) -> str:
    if not token:
        return ""
    text = token.strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "_", text)
    return text


def _canonicalize_token(
    token_key: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    warnings: List[str] = []
    if not token_key:
        return None, None, None, warnings

    normalized = token_key
    if normalized in TOKEN_ALIASES:
        normalized = TOKEN_ALIASES[normalized]

    parts = normalized.split("__")
    if len(parts) < 2:
        inferred = _heuristic_token_parse(normalized)
        if inferred[0]:
            return inferred
        return None, None, None, warnings

    base = "__".join(parts[:2])
    context = "__".join(parts[2:]) if len(parts) > 2 else None
    if base not in CANONICAL_TOKENS:
        inferred = _heuristic_token_parse(normalized)
        if inferred[0]:
            return inferred
        return None, None, None, warnings

    canonical = base if not context else f"{base}__{context}"
    return canonical, base, context, warnings


def _heuristic_token_parse(
    token_key: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    warnings: List[str] = []
    if not token_key:
        return None, None, None, warnings
    text = token_key.lower()
    context = None
    base = None

    if "acetyl" in text:
        context = "acetyl"
    if "fluorinated" in text:
        base = "ch__fluorinated"
    elif "benzylic" in text:
        base = "ch__benzylic"
    elif "allylic" in text:
        base = "ch__allylic"
    elif "alpha_hetero" in text or "alpha-hetero" in text:
        base = "ch__alpha_hetero"
    elif "ch" in text or "c-h" in text:
        base = "ch__aryl" if "aryl" in text else "ch__aliphatic"
    elif "o-h" in text or text == "oh" or "hydroxyl" in text:
        base = "oh__phenol" if "phenol" in text else "oh__alcohol"
    elif "n-h" in text or text == "nh":
        base = "nh__amide" if "amide" in text else "nh__amine"
    elif "thioester" in text:
        base = "thioester__acyl_s"
    elif "amide" in text:
        base = "amide__c_n"
    elif "ester" in text:
        base = "ester__acyl_o"
    elif "halide" in text or re.search(r"c-[fclbri]", text):
        base = "aryl_halide__c_x" if "aryl" in text else "alkyl_halide__c_x"
        if "br" in text:
            context = "br"
        elif "cl" in text:
            context = "cl"
        elif "f" in text:
            context = "f"
        elif "i" in text:
            context = "i"
    elif "ether" in text or "c-o" in text:
        base = "ether__c_o"

    if base is None or base not in CANONICAL_TOKENS:
        return None, None, None, warnings

    canonical = base if not context else f"{base}__{context}"
    warnings.append("Token inferred heuristically from input.")
    return canonical, base, context, warnings


def _first_element_symbol(text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z]{1,2}", text)
    if not match:
        return None
    return match.group(0).capitalize()


def _extract_smarts_map_pair(smarts: str) -> Optional[Tuple[int, int]]:
    maps = re.findall(r":(\d+)", smarts)
    if len(maps) >= 2:
        return int(maps[0]), int(maps[1])
    return None


def _token_smarts_entry(canonical_token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not canonical_token:
        return None
    if canonical_token in TOKEN_SMARTS_REGISTRY:
        entry = dict(TOKEN_SMARTS_REGISTRY[canonical_token])
        entry["canonical_token"] = canonical_token
        return entry
    parts = canonical_token.split("__")
    if len(parts) >= 2:
        base = "__".join(parts[:2])
        if base in TOKEN_SMARTS_REGISTRY:
            entry = dict(TOKEN_SMARTS_REGISTRY[base])
            entry["canonical_token"] = canonical_token
            return entry
    return None


def _load_molecule(smiles: str) -> Tuple[Optional[Any], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not smiles or not smiles.strip():
        errors.append("Empty molecule definition.")
        return None, warnings, errors

    if Chem is None:
        warnings.append("RDKit not installed.")
        return None, warnings, errors

    mol = None
    if _looks_like_mol_block(smiles):
        mol = Chem.MolFromMolBlock(smiles, sanitize=True)
    else:
        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        errors.append("Failed to parse molecule definition.")
        return None, warnings, errors

    return mol, warnings, errors


def _looks_like_mol_block(payload: str) -> bool:
    markers = ("M  END", "V2000", "V3000", "$$$$")
    return any(marker in payload for marker in markers)


def _compute_structure_summary(
    mol: Optional[Any],
    bond_roles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if mol is None or Chem is None:
        return {}

    summary: Dict[str, Any] = {
        "atom_count": mol.GetNumAtoms(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "ring_count": mol.GetRingInfo().NumRings() if mol.GetRingInfo() else 0,
        "hetero_atoms": sum(
            1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"}
        ),
        "rotatable_bonds": None,
        "functional_groups": _identify_functional_groups(mol),
    }

    if rdMolDescriptors is not None:
        summary["rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)

    if bond_roles is None:
        bond_roles = _assign_bond_roles(mol)
    summary["functional_groups"].update(_role_functional_group_counts(bond_roles))
    summary["ring_strain_proxy"] = _ring_strain_proxy(mol)

    return summary


def _estimate_protonation_flags(
    mol: Optional[Any],
    constraints: OperationalConstraints,
) -> List[Dict[str, Any]]:
    if mol is None or Chem is None:
        return []

    ph_min = constraints.ph_min
    ph_max = constraints.ph_max
    flags: List[Dict[str, Any]] = []
    for entry in PROTONATION_SMARTS:
        pattern = Chem.MolFromSmarts(entry["smarts"])
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            continue
        risk_level = _pka_risk_level(entry["pka_range"], ph_min, ph_max)
        flags.append(
            {
                "group": entry["group"],
                "count": len(matches),
                "estimated_pka_range": list(entry["pka_range"]),
                "risk_level": risk_level,
                "note": entry["note"],
            }
        )
    return flags


def _pka_risk_level(
    pka_range: Tuple[float, float],
    ph_min: Optional[float],
    ph_max: Optional[float],
) -> str:
    if ph_min is None or ph_max is None:
        return "unknown"
    low, high = pka_range
    if ph_min <= high and ph_max >= low:
        return "high"
    if min(abs(ph_min - high), abs(ph_max - low)) <= 1.5:
        return "medium"
    return "low"


def _module1_mode_from_context(
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    heavy_atoms = structure_summary.get("heavy_atoms") or 0
    rotatable = structure_summary.get("rotatable_bonds") or 0
    gas_like = bond_context.get("is_gas_like_small_molecule") is True or heavy_atoms <= 6
    if gas_like:
        return "small_gas", {"access": 0.15, "reach": 0.50, "retention": 0.35}
    if heavy_atoms >= 35 or rotatable >= 12:
        return "bulky_substrate", {"access": 0.45, "reach": 0.40, "retention": 0.15}
    return "standard", {"access": 0.35, "reach": 0.45, "retention": 0.20}


def _substrate_size_proxies(structure_summary: Dict[str, Any]) -> Dict[str, float]:
    heavy_atoms = structure_summary.get("heavy_atoms") or 0
    approx_radius = max(1.2, 0.7 + (0.03 * heavy_atoms))
    approx_volume = (4.0 / 3.0) * math.pi * (approx_radius**3)
    min_diameter_proxy = max(1.5, approx_radius * 2.0)
    return {
        "approx_radius": round(approx_radius, 3),
        "approx_volume": round(approx_volume, 2),
        "min_diameter_proxy": round(min_diameter_proxy, 3),
    }


def _bond_center_hint(
    bond_context: Dict[str, Any],
    resolved: Dict[str, Any],
) -> Dict[str, Any]:
    atom_indices = bond_context.get("atom_indices")
    if not atom_indices or any(idx is None for idx in atom_indices):
        atom_indices = resolved.get("atom_indices")
    if not atom_indices and resolved.get("selected_bond"):
        atom_indices = resolved["selected_bond"].get("atom_indices")
    local_context = {
        "neighbor_hetero_atoms": bond_context.get("neighbor_hetero_atoms"),
        "in_ring": bond_context.get("in_ring"),
        "is_aromatic": bond_context.get("is_aromatic"),
        "bond_type": bond_context.get("bond_type"),
        "bond_class": bond_context.get("bond_class"),
    }
    anchor_smarts_used = resolved.get("smarts") or resolved.get("canonical_token")
    return {
        "atom_indices": atom_indices,
        "local_context": local_context,
        "anchor_smarts_used": anchor_smarts_used,
    }


def _identify_functional_groups(mol: Any) -> Dict[str, int]:
    if Chem is None:
        return {}

    counts: Dict[str, int] = {}
    for name, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            counts[name] = len(matches)
    return counts


def _role_functional_group_counts(bond_roles: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in bond_roles:
        group = ROLE_FUNCTIONAL_GROUP_MAP.get(entry.get("primary_role"))
        if group:
            counts[group] += 1
    return dict(counts)


def _ring_strain_proxy(mol: Any) -> Dict[str, int]:
    if mol is None:
        return {}
    ring_info = mol.GetRingInfo()
    if ring_info is None:
        return {}
    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
    return {
        "three_member_rings": sum(1 for size in ring_sizes if size == 3),
        "four_member_rings": sum(1 for size in ring_sizes if size == 4),
    }


def _assign_bond_roles(mol: Any) -> List[Dict[str, Any]]:
    bond_role_map: Dict[int, Dict[str, Any]] = {}

    for atom in mol.GetAtoms():
        if not _is_carbonyl_carbon_atom(atom):
            continue
        carbonyl_idx = atom.GetIdx()
        o_neighbors = []
        n_neighbors = []
        s_neighbors = []
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() != 1.0:
                continue
            neighbor = bond.GetOtherAtom(atom)
            symbol = neighbor.GetSymbol()
            if symbol == "O":
                o_neighbors.append((bond, neighbor))
            elif symbol == "N":
                n_neighbors.append((bond, neighbor))
            elif symbol == "S":
                s_neighbors.append((bond, neighbor))

        has_two_o = len(o_neighbors) >= 2
        has_n = len(n_neighbors) > 0
        has_two_n = len(n_neighbors) >= 2

        for bond, o_atom in o_neighbors:
            if _is_carboxylate_oxygen(o_atom):
                role = "acid__c_o"
            elif _is_anhydride_oxygen(o_atom, carbonyl_idx):
                role = "anhydride__acyl_o"
            elif has_two_o and not has_n:
                role = "carbonate__acyl_o"
            else:
                role = "ester__acyl_o"
            context_tags = []
            if _is_acetyl_carbonyl(atom, o_atom.GetIdx()):
                context_tags.append("acetyl")
            _set_role_for_bond(
                bond_role_map,
                bond,
                role,
                confidence=0.9,
                tags=context_tags,
                evidence="carbonyl_single_bond",
            )

        for bond, n_atom in n_neighbors:
            if bond.IsInRingSize(4):
                role = "beta_lactam__c_n"
            elif bond.IsInRing():
                role = "lactam__c_n"
            elif has_two_n:
                role = "urea__c_n"
            elif has_n and o_neighbors:
                role = "carbamate__c_n"
            else:
                role = "amide__c_n"
            _set_role_for_bond(
                bond_role_map,
                bond,
                role,
                confidence=0.88,
                tags=[],
                evidence="carbonyl_single_bond",
            )

        for bond, _s_atom in s_neighbors:
            _set_role_for_bond(
                bond_role_map,
                bond,
                "thioester__acyl_s",
                confidence=0.88,
                tags=[],
                evidence="carbonyl_single_bond",
            )

    for bond in mol.GetBonds():
        a_atom = bond.GetBeginAtom()
        b_atom = bond.GetEndAtom()
        a_symbol = a_atom.GetSymbol()
        b_symbol = b_atom.GetSymbol()
        symbols = {a_symbol, b_symbol}
        bond_order = float(bond.GetBondTypeAsDouble())

        if symbols == {"C", "O"}:
            if bond.GetIdx() in bond_role_map:
                continue
            if bond.IsInRingSize(3):
                _set_role_for_bond(
                    bond_role_map,
                    bond,
                    "epoxide__c_o",
                    confidence=0.8,
                    tags=[],
                    evidence="ring_size_3",
                )
                continue
            if _is_acetal_bond(a_atom, b_atom):
                role = "glycosidic__acetal_o" if _is_glycosidic_bond(a_atom, b_atom) else "acetal__c_o"
                _set_role_for_bond(
                    bond_role_map,
                    bond,
                    role,
                    confidence=0.7,
                    tags=["glycosidic"] if role.startswith("glycosidic") else [],
                    evidence="acetal_signature",
                )
                continue
            _set_role_for_bond(
                bond_role_map,
                bond,
                "ether__c_o",
                confidence=0.6,
                tags=[],
                evidence="C-O_single",
            )
            continue

        if symbols == {"C", "N"}:
            if bond.GetIdx() in bond_role_map:
                continue
            if bond_order == 3.0:
                _set_role_for_bond(
                    bond_role_map,
                    bond,
                    "nitrile__c_n",
                    confidence=0.8,
                    tags=[],
                    evidence="triple_bond",
                )
                continue
            if bond_order == 2.0:
                _set_role_for_bond(
                    bond_role_map,
                    bond,
                    "imine__c_n",
                    confidence=0.7,
                    tags=[],
                    evidence="double_bond",
                )
                continue

        if symbols == {"C", "S"}:
            if bond.GetIdx() in bond_role_map:
                continue
            _set_role_for_bond(
                bond_role_map,
                bond,
                "thioether__c_s",
                confidence=0.6,
                tags=[],
                evidence="C-S_single",
            )
            continue

        if symbols == {"S", "N"} and _is_sulfonyl_s(bond):
            _set_role_for_bond(
                bond_role_map,
                bond,
                "sulfonamide__s_n",
                confidence=0.8,
                tags=[],
                evidence="sulfonyl_s",
            )
            continue

        if symbols == {"S", "O"} and _is_sulfonyl_s(bond):
            _set_role_for_bond(
                bond_role_map,
                bond,
                "sulfate_ester__s_o",
                confidence=0.8,
                tags=[],
                evidence="sulfonyl_s",
            )
            continue

        if symbols == {"P", "O"} and _is_phosphoryl_p(bond):
            _set_role_for_bond(
                bond_role_map,
                bond,
                "phosphate_ester__p_o",
                confidence=0.8,
                tags=[],
                evidence="phosphoryl_p",
            )
            continue

        if symbols == {"S", "S"} and bond_order == 1.0:
            _set_role_for_bond(
                bond_role_map,
                bond,
                "disulfide__s_s",
                confidence=0.7,
                tags=[],
                evidence="S-S_single",
            )
            continue

        if symbols == {"N", "N"} and bond_order == 2.0:
            _set_role_for_bond(
                bond_role_map,
                bond,
                "azo__n_n",
                confidence=0.7,
                tags=[],
                evidence="N=N_double",
            )
            continue

        if symbols == {"C", "C"} and bond_order == 1.0:
            if bond.GetIdx() in bond_role_map:
                continue
            role = _classify_cc_role(a_atom, b_atom)
            _set_role_for_bond(
                bond_role_map,
                bond,
                role,
                confidence=0.6,
                tags=[],
                evidence="C-C_single",
            )
            continue

        halogen = _halogen_symbol(a_symbol, b_symbol)
        if halogen and "C" in symbols:
            carbon_atom = a_atom if a_symbol == "C" else b_atom
            role = "aryl_halide__c_x" if carbon_atom.GetIsAromatic() else "alkyl_halide__c_x"
            _set_role_for_bond(
                bond_role_map,
                bond,
                role,
                confidence=0.7,
                tags=[halogen.lower()],
                evidence="C-X_halide",
            )
            continue

    _add_environment_roles(bond_role_map, mol)
    entries = list(bond_role_map.values())
    entries.extend(_assign_ch_roles(mol))
    _finalize_primary_roles(entries)
    return entries


def _set_role_for_bond(
    bond_role_map: Dict[int, Dict[str, Any]],
    bond: Any,
    role: str,
    confidence: float,
    tags: List[str],
    evidence: str,
) -> None:
    bond_idx = bond.GetIdx()
    entry = bond_role_map.get(bond_idx)
    if entry is None:
        entry = {
            "kind": "bond",
            "bond": bond,
            "bond_idx": bond_idx,
            "atom_indices": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
            "element_pair": _bond_element_pair(bond),
            "bond_order": float(bond.GetBondTypeAsDouble()),
            "is_aromatic": bond.GetIsAromatic(),
            "bond_roles": [],
        }
        bond_role_map[bond_idx] = entry
    entry["bond_roles"].append(
        {
            "role": role,
            "confidence": confidence,
            "tags": tags,
            "evidence": evidence,
        }
    )


def _assign_ch_roles(mol: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        if atom.GetTotalNumHs() <= 0:
            continue
        role, confidence, tags = _classify_ch_role(atom)
        entries.append(
            {
                "kind": "implicit_h",
                "carbon_index": atom.GetIdx(),
                "hydrogen_count": atom.GetTotalNumHs(),
                "element_pair": ["C", "H"],
                "bond_roles": [
                    {
                        "role": role,
                        "confidence": confidence,
                        "tags": tags,
                        "evidence": "C-H_environment",
                    }
                ],
            }
        )
    _finalize_primary_roles(entries)
    return entries


def _classify_ch_role(atom: Any) -> Tuple[str, float, List[str]]:
    fluorine_neighbors = _neighbor_element_count(atom, "F")
    if fluorine_neighbors >= 2:
        return "ch__fluorinated", 0.85, ["fluorinated"]

    if atom.GetIsAromatic():
        return "ch__aryl", 0.8, ["aryl"]

    if _is_benzylic_carbon(atom):
        return "ch__benzylic", 0.7, ["benzylic"]

    if _is_allylic_carbon(atom):
        return "ch__allylic", 0.7, ["allylic"]

    if _is_alpha_hetero_carbon(atom):
        return "ch__alpha_hetero", 0.65, ["alpha_hetero"]

    return "ch__aliphatic", 0.6, ["aliphatic"]


def _classify_cc_role(a_atom: Any, b_atom: Any) -> str:
    if a_atom.GetIsAromatic() and b_atom.GetIsAromatic():
        return "cc__aryl_aryl"
    if a_atom.GetIsAromatic() or b_atom.GetIsAromatic():
        return "cc__aryl_alkyl"
    return "cc__single"


def _add_environment_roles(bond_role_map: Dict[int, Dict[str, Any]], mol: Any) -> None:
    for entry in bond_role_map.values():
        bond = entry.get("bond")
        if bond is None:
            continue
        roles = entry.get("bond_roles", [])
        if bond.IsInRing():
            roles.append(
                {
                    "role": "ring_constrained",
                    "confidence": 0.4,
                    "tags": ["ring"],
                    "evidence": "bond_in_ring",
                }
            )
        if bond.GetBeginAtom().GetIsAromatic() or bond.GetEndAtom().GetIsAromatic():
            roles.append(
                {
                    "role": "aromatic_adjacent",
                    "confidence": 0.4,
                    "tags": ["aromatic"],
                    "evidence": "adjacent_aromatic",
                }
            )
        hetero_neighbors = _neighbor_hetero_count(bond.GetBeginAtom(), bond.GetEndAtomIdx()) + _neighbor_hetero_count(
            bond.GetEndAtom(), bond.GetBeginAtomIdx()
        )
        if hetero_neighbors >= 2:
            roles.append(
                {
                    "role": "hbond_rich",
                    "confidence": 0.35,
                    "tags": ["hetero_rich"],
                    "evidence": "neighbor_hetero",
                }
            )
        entry["bond_roles"] = roles


def _finalize_primary_roles(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        roles = entry.get("bond_roles", [])
        primary_role = None
        primary_conf = None
        for role_entry in roles:
            role = role_entry.get("role")
            conf = role_entry.get("confidence")
            if role in ROLE_ROUTE_MAP or role in CANONICAL_TOKENS:
                if primary_conf is None or (conf is not None and conf > primary_conf):
                    primary_role = role
                    primary_conf = conf
        if primary_role is None and roles:
            primary_role = roles[0].get("role")
            primary_conf = roles[0].get("confidence")
        entry["primary_role"] = primary_role
        entry["primary_role_confidence"] = primary_conf


def _halogen_symbol(a_symbol: str, b_symbol: str) -> Optional[str]:
    halogens = {"F", "Cl", "Br", "I"}
    if a_symbol in halogens:
        return a_symbol
    if b_symbol in halogens:
        return b_symbol
    return None


def _is_carbonyl_carbon_atom(atom: Any) -> bool:
    if atom.GetSymbol() != "C":
        return False
    for bond in atom.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == "O":
                return True
    return False


def _is_carboxylate_oxygen(atom: Any) -> bool:
    return atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() < 0


def _is_anhydride_oxygen(atom: Any, carbonyl_idx: int) -> bool:
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == carbonyl_idx:
            continue
        if _is_carbonyl_carbon_atom(neighbor):
            return True
    return False


def _is_acetyl_carbonyl(carbonyl_atom: Any, target_o_idx: int) -> bool:
    for neighbor in carbonyl_atom.GetNeighbors():
        if neighbor.GetIdx() == target_o_idx:
            continue
        if neighbor.GetSymbol() != "C":
            continue
        if neighbor.GetTotalNumHs() >= 3:
            return True
    return False


def _is_acetal_bond(a_atom: Any, b_atom: Any) -> bool:
    carbon = a_atom if a_atom.GetSymbol() == "C" else b_atom
    if carbon.GetSymbol() != "C":
        return False
    o_neighbors = []
    for bond in carbon.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        neighbor = bond.GetOtherAtom(carbon)
        if neighbor.GetSymbol() == "O":
            o_neighbors.append(neighbor)
    return len(o_neighbors) >= 2


def _is_glycosidic_bond(a_atom: Any, b_atom: Any) -> bool:
    carbon = a_atom if a_atom.GetSymbol() == "C" else b_atom
    if not carbon.IsInRing():
        return False
    o_neighbors = [neighbor for neighbor in carbon.GetNeighbors() if neighbor.GetSymbol() == "O"]
    return len(o_neighbors) >= 2


def _is_sulfonyl_s(bond: Any) -> bool:
    s_atom = bond.GetBeginAtom() if bond.GetBeginAtom().GetSymbol() == "S" else bond.GetEndAtom()
    if s_atom.GetSymbol() != "S":
        return False
    double_o = 0
    for s_bond in s_atom.GetBonds():
        if s_bond.GetBondTypeAsDouble() == 2.0 and s_bond.GetOtherAtom(s_atom).GetSymbol() == "O":
            double_o += 1
    return double_o >= 2


def _is_phosphoryl_p(bond: Any) -> bool:
    p_atom = bond.GetBeginAtom() if bond.GetBeginAtom().GetSymbol() == "P" else bond.GetEndAtom()
    if p_atom.GetSymbol() != "P":
        return False
    double_o = 0
    for p_bond in p_atom.GetBonds():
        if p_bond.GetBondTypeAsDouble() == 2.0 and p_bond.GetOtherAtom(p_atom).GetSymbol() == "O":
            double_o += 1
    return double_o >= 1


def _is_benzylic_carbon(atom: Any) -> bool:
    if atom.GetIsAromatic():
        return False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIsAromatic():
            return True
    return False


def _is_allylic_carbon(atom: Any) -> bool:
    if atom.GetIsAromatic():
        return False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() != "C":
            continue
        for bond in neighbor.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(neighbor)
                if other.GetSymbol() == "C":
                    return True
    return False


def _is_alpha_hetero_carbon(atom: Any) -> bool:
    hetero = {"O", "N", "S", "P", "F", "Cl", "Br", "I"}
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() in hetero:
            return True
    return False


def _extract_sre_resolution(sre_output: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sre_output:
        return None
    if "result" in sre_output:
        resolved = (sre_output.get("result") or {}).get("resolved_target") or {}
    else:
        resolved = sre_output.get("resolved_target") or {}
    candidates = resolved.get("candidate_bonds") or resolved.get("candidate_sites") or []
    if not candidates:
        return None
    return {
        "candidate_bonds": candidates,
        "canonical_token": resolved.get("canonical_token"),
        "token_context": resolved.get("token_context"),
        "bond_type": resolved.get("bond_type"),
        "resolution_policy": resolved.get("resolution_policy"),
        "resolution_confidence": resolved.get("resolution_confidence"),
        "equivalent_sites_detected": resolved.get("equivalent_sites_detected"),
        "next_input_required": resolved.get("next_input_required") or [],
    }


def _normalize_sre_payload_for_physics(
    module_minus1: Optional[Dict[str, Any]],
    unity_sre: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize Module -1 payload shape for physics consumers.

    Supports both:
    - ctx.data["module_minus1"] style (top-level cpt_scores/route_bias)
    - shared_io.outputs.module_minus1 style (cpt under `cpt`, route under `reactivity`)
    - unity_state.sre style.
    """
    payload: Dict[str, Any] = {}

    if isinstance(unity_sre, dict):
        payload.update(unity_sre)
    if isinstance(module_minus1, dict):
        payload.update(module_minus1)
        reactivity = module_minus1.get("reactivity")
        if isinstance(reactivity, dict):
            for key in ("route_bias", "mechanism_eligibility", "confidence_prior"):
                if key in reactivity and key not in payload:
                    payload[key] = reactivity.get(key)
        if "cpt_scores" not in payload and isinstance(module_minus1.get("cpt"), dict):
            payload["cpt_scores"] = module_minus1.get("cpt")
        if "cpt_scores" not in payload and isinstance(module_minus1.get("ep_av"), dict):
            payload["cpt_scores"] = {"epav": module_minus1.get("ep_av")}

    return payload


def _radical_barrier_from_sre_payload(sre_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(sre_payload, dict):
        return None
    cpt = sre_payload.get("cpt_scores")
    if not isinstance(cpt, dict):
        cpt = sre_payload.get("cpt") if isinstance(sre_payload.get("cpt"), dict) else None
    if not isinstance(cpt, dict):
        return None
    if str(cpt.get("track") or "").lower() != "radical_hat":
        return None
    best_hat = cpt.get("best_hat") if isinstance(cpt.get("best_hat"), dict) else {}
    barrier_kj = best_hat.get("barrier_kj_mol")
    if not isinstance(barrier_kj, (int, float)):
        return None
    bde = cpt.get("bde") if isinstance(cpt.get("bde"), dict) else {}
    return {
        "barrier_kj_mol": float(barrier_kj),
        "mechanism": best_hat.get("mechanism"),
        "bde_kj_mol": bde.get("corrected_kj_mol"),
        "bond_class": bde.get("bond_class"),
        "track": "radical_hat",
    }


def _sn2_barrier_from_sre_payload(sre_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(sre_payload, dict):
        return None
    cpt = sre_payload.get("cpt_scores")
    if not isinstance(cpt, dict):
        cpt = sre_payload.get("cpt") if isinstance(sre_payload.get("cpt"), dict) else None
    if not isinstance(cpt, dict):
        return None
    if str(cpt.get("track") or "").lower() != "displacement_sn2":
        return None
    barrier_kj = cpt.get("best_barrier_kj_mol")
    if not isinstance(barrier_kj, (int, float)):
        return None
    features = cpt.get("features") if isinstance(cpt.get("features"), dict) else {}
    return {
        "barrier_kj_mol": float(barrier_kj),
        "mechanism": cpt.get("mechanism_hint") or "SN2 displacement",
        "track": "displacement_sn2",
        "leaving_group": features.get("halide"),
    }


def _sre_barrier_override(sre_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _radical_barrier_from_sre_payload(sre_payload) or _sn2_barrier_from_sre_payload(
        sre_payload
    )


def _route_key(route_name: str) -> str:
    return str(route_name or "").strip().lower()


def _route_eligibility_status(
    route_name: str,
    mechanism_eligibility: Dict[str, Any],
) -> Optional[str]:
    if not isinstance(mechanism_eligibility, dict):
        return None
    key = _route_key(route_name)
    direct = mechanism_eligibility.get(route_name)
    if isinstance(direct, str):
        return direct
    mapped_key = ROUTE_TO_ELIGIBILITY_KEY.get(key)
    if mapped_key:
        mapped = mechanism_eligibility.get(mapped_key)
        if isinstance(mapped, str):
            return mapped
    return None


def _compute_hat_informed_priors(
    routes: List[str],
    cpt_scores: Dict[str, Any],
    mechanism_eligibility: Dict[str, Any],
    bond_context: Dict[str, Any],
    temperature_K: float = 298.15,
) -> Dict[str, Dict[str, Any]]:
    """Convert Module -1 radical HAT barriers into route-specific feasibility priors."""
    hat_barriers = cpt_scores.get("hat_barriers") if isinstance(cpt_scores, dict) else None
    if not isinstance(hat_barriers, dict) or not hat_barriers:
        return {}

    kT = 8.314e-3 * max(float(temperature_K or 298.15), 1.0)
    beta = 1.0 / max(1e-9, 10.0 * kT)

    route_barriers: Dict[str, float] = {}
    numeric_barriers = [float(v) for v in hat_barriers.values() if isinstance(v, (int, float))]
    if not numeric_barriers:
        return {}
    fallback_barrier = max(numeric_barriers) + 10.0

    def _hat_value_for_key(key: Optional[str]) -> Optional[float]:
        if not key:
            return None
        aliases = [key]
        if key == "Fe_IV_oxo_heme":
            aliases.append("Fe_IV_oxo")
        elif key == "Fe_IV_oxo_nonheme":
            aliases.append("non_heme_Fe")
        elif key == "Fe_IV_oxo":
            aliases.append("Fe_IV_oxo_heme")
        elif key == "non_heme_Fe":
            aliases.append("Fe_IV_oxo_nonheme")
        for alias in aliases:
            value = hat_barriers.get(alias)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    for route in routes:
        route_norm = _route_key(route)
        mechanism_key = ROUTE_TO_HAT_MECHANISM.get(route_norm)
        mechanism_barrier = _hat_value_for_key(mechanism_key)
        route_barriers[route] = (
            float(mechanism_barrier) if mechanism_barrier is not None else float(fallback_barrier)
        )

    min_barrier = min(route_barriers.values()) if route_barriers else 0.0
    boltzmann_weights: Dict[str, float] = {}
    for route, barrier in route_barriers.items():
        boltzmann_weights[route] = math.exp(-beta * (float(barrier) - min_barrier))

    sum_weights = sum(boltzmann_weights.values()) or 1.0
    priors = {route: weight / sum_weights for route, weight in boltzmann_weights.items()}

    compatibility_multiplier: Dict[str, float] = {}
    for route in routes:
        mult = _route_compatibility_weight(route, bond_context=bond_context)
        priors[route] = float(priors.get(route, 0.0)) * float(mult)
        compatibility_multiplier[route] = float(mult)

    sum_priors = sum(priors.values()) or 1.0
    priors = {route: value / sum_priors for route, value in priors.items()}

    eligibility_status: Dict[str, str] = {}
    for route in routes:
        status = _route_eligibility_status(route, mechanism_eligibility) or "REQUIRE_QUORUM"
        eligibility_status[route] = status
        if status == "REJECTED":
            priors[route] = float(priors.get(route, 0.0)) * 0.05
        elif status == "REQUIRE_QUORUM":
            priors[route] = float(priors.get(route, 0.0)) * 0.6

    sum_priors = sum(priors.values()) or 1.0
    priors = {route: value / sum_priors for route, value in priors.items()}

    max_prior = max(priors.values()) if priors else 1.0
    result: Dict[str, Dict[str, Any]] = {}
    for route in routes:
        scaled = (float(priors.get(route, 0.0)) / max(max_prior, 1e-9)) * 0.85
        result[route] = {
            "prior_feasibility": max(0.05, min(0.95, float(scaled))),
            "hat_barrier_kj_mol": float(route_barriers.get(route, fallback_barrier)),
            "substrate_compatibility_multiplier": float(compatibility_multiplier.get(route, 1.0)),
            "mechanism_eligibility": eligibility_status.get(route),
            "prior_source": "hat_informed",
        }
    return result


def _apply_barrier_override_to_route_prior(
    route_prior: Dict[str, Any],
    barrier_kj: float,
    temp_k: float,
    horizon_s: float,
) -> Dict[str, Any]:
    """Project a custom barrier into route_prior kinetic fields."""
    updated = dict(route_prior)
    k_ey = eyring_rate_constant(float(barrier_kj) * 1000.0, float(temp_k))
    diff_cap = updated.get("diffusion_cap_s_inv")
    f_prot = float(updated.get("f_prot") or 1.0)
    solvent_pen = float(updated.get("solvent_penalty") or 1.0)
    k_eff = float(k_ey) * max(0.0, f_prot) * max(0.0, solvent_pen)
    if isinstance(diff_cap, (int, float)) and math.isfinite(float(diff_cap)):
        k_eff = min(k_eff, max(0.0, float(diff_cap)))
    p_raw = 1.0 - math.exp(-max(0.0, k_eff) * max(1.0, float(horizon_s)))
    p_raw = max(0.0, min(1.0, float(p_raw)))
    updated["deltaG_dagger_kJ_per_mol"] = float(barrier_kj)
    updated["eyring_k_s_inv"] = float(k_ey)
    updated["k_effective_s_inv"] = float(k_eff)
    updated["p_raw"] = float(p_raw)
    updated["p_final"] = max(0.01, min(0.99, float(p_raw)))
    return updated


_SYNTHETIC_H_PATTERN = re.compile(r"^(.+)_H(\d+)$")


def _find_synthetic_h_in_ids(atom_ids: Optional[List[Any]]) -> Optional[Tuple[str, int]]:
    """Find a synthetic H UUID in a list of atom IDs from Module -1.
    Returns (heavy_atom_uuid, h_index_in_addhs_mol) or None.
    """
    for aid in (atom_ids or []):
        m = _SYNTHETIC_H_PATTERN.match(str(aid))
        if m:
            return (m.group(1), int(m.group(2)))
    return None


def _fallback_xh_role_for_atom(
    heavy_atom: Any,
    sre_resolution: Optional[Dict[str, Any]],
    candidate_hint: Optional[Dict[str, Any]],
) -> Tuple[str, float, List[str]]:
    if isinstance(candidate_hint, dict):
        bond_class = candidate_hint.get("bond_class")
        if isinstance(bond_class, str) and bond_class in ROLE_ROUTE_MAP:
            return bond_class, 0.85, []
        subclass = candidate_hint.get("subclass")
        if isinstance(subclass, str):
            maybe_role = f"ch__{subclass.lower()}"
            if maybe_role in ROLE_ROUTE_MAP:
                return maybe_role, 0.8, [subclass.lower()]
    bond_type = str((sre_resolution or {}).get("bond_type") or "").lower()
    sym = str(heavy_atom.GetSymbol())
    if bond_type == "oh" or sym == "O":
        is_phenol = any(nbr.GetIsAromatic() for nbr in heavy_atom.GetNeighbors())
        return ("oh__phenol", 0.8, ["phenol"]) if is_phenol else ("oh__alcohol", 0.75, ["alcohol"])
    if bond_type == "nh" or sym == "N":
        is_amide_n = False
        for nbr in heavy_atom.GetNeighbors():
            if nbr.GetSymbol() != "C":
                continue
            for b in nbr.GetBonds():
                if b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(nbr).GetSymbol() == "O":
                    is_amide_n = True
                    break
            if is_amide_n:
                break
        return ("nh__amide", 0.78, ["amide"]) if is_amide_n else ("nh__amine", 0.72, ["amine"])
    if bond_type == "ch" or sym == "C":
        role, conf, tags = _classify_ch_role(heavy_atom)
        return role, float(conf), list(tags or [])
    return "unknown", 0.4, []


def _synthetic_xh_entry(
    heavy_idx: int,
    h_idx: int,
    heavy_atom: Any,
    role: str,
    confidence: float,
    tags: List[str],
    sre_resolution: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "kind": "bond",
        "bond": None,
        "bond_idx": None,
        "atom_indices": [int(heavy_idx), int(h_idx)],
        "element_pair": [str(heavy_atom.GetSymbol()), "H"],
        "bond_order": 1.0,
        "is_aromatic": bool(heavy_atom.GetIsAromatic()),
        "bond_roles": [
            {
                "role": role,
                "confidence": float(confidence),
                "tags": list(tags or []),
                "evidence": "module_minus1_synthetic_h",
            }
        ],
        "primary_role": role,
        "primary_role_confidence": float(confidence),
        "sre_bond_type": (sre_resolution or {}).get("bond_type"),
        "sre_source": True,
        "is_synthetic_h": True,
    }


def _resolve_target_bonds(
    mol: Any,
    target_spec: TargetBondSpec,
    bond_roles: List[Dict[str, Any]],
    requested_output: Optional[str],
    sre_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    resolved: Dict[str, Any] = {}
    bonds: List[Any] = []
    equivalence_groups: List[Dict[str, Any]] = []
    candidate_bonds: List[Dict[str, Any]] = []
    candidate_meta: Dict[str, Any] = {}
    index_base = None
    element_pair: Optional[List[str]] = None
    candidate_atoms: List[int] = []
    token_info: Optional[Dict[str, Any]] = None
    token_resolution_audit: Optional[Dict[str, Any]] = None
    match_count = 0
    target_resolution_confidence = 0.0
    selected_entry: Optional[Dict[str, Any]] = None

    if target_spec.kind != "indices":
        sre_resolution = _extract_sre_resolution(sre_output)
        if sre_resolution and sre_resolution.get("candidate_bonds"):
            sre_candidates = sre_resolution["candidate_bonds"]
            candidate_entries: List[Dict[str, Any]] = []
            used_synthetic_h = False
            sre_auto_selected = bool(
                str(sre_resolution.get("resolution_policy") or "").startswith("lowest_BDE_auto")
                and not (sre_resolution.get("next_input_required") or [])
            )
            _h_mol = None
            for cand in sre_candidates:
                atom_indices = cand.get("atom_indices") or cand.get("bond_indices")
                # --- X-H fallback: implicit H not in original mol ---
                if (not atom_indices or len(atom_indices) < 2) and Chem is not None:
                    h_match = _find_synthetic_h_in_ids(cand.get("atom_ids"))
                    if h_match is not None:
                        _heavy_uuid, h_idx = h_match
                        if _h_mol is None:
                            _h_mol = Chem.AddHs(mol)
                        if h_idx < _h_mol.GetNumAtoms():
                            h_atom = _h_mol.GetAtomWithIdx(h_idx)
                            if h_atom.GetSymbol() == "H":
                                for nbr in h_atom.GetNeighbors():
                                    role, conf, tags = _fallback_xh_role_for_atom(
                                        nbr, sre_resolution, cand
                                    )
                                    entry = _synthetic_xh_entry(
                                        heavy_idx=nbr.GetIdx(),
                                        h_idx=h_idx,
                                        heavy_atom=nbr,
                                        role=role,
                                        confidence=conf,
                                        tags=tags,
                                        sre_resolution=sre_resolution,
                                    )
                                    candidate_entries.append(entry)
                                    used_synthetic_h = True
                                    break
                        continue
                # --- end X-H fallback ---
                if not atom_indices or len(atom_indices) < 2:
                    continue
                idx_a, idx_b = int(atom_indices[0]), int(atom_indices[1])
                bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
                if bond is None:
                    continue
                bonds.append(bond)
                entry = _entry_from_bond(bond_roles, bond) or _fallback_entry_from_bond(bond)
                if entry:
                    entry["sre_source"] = True
                    candidate_entries.append(entry)
            if bonds or candidate_entries:
                match_count = len(candidate_entries)
                candidate_atoms = sorted(
                    {
                        int(idx)
                        for entry in candidate_entries
                        for idx in (entry.get("atom_indices") or [])
                        if isinstance(idx, int) and 0 <= int(idx) < mol.GetNumAtoms()
                    }
                )
                equivalence_groups = _equivalent_groups_from_entries(mol, candidate_entries)
                selected_entry = _select_best_entry(candidate_entries)
                if selected_entry is None and len(candidate_entries) == 1:
                    selected_entry = candidate_entries[0]
                if selected_entry is None and sre_auto_selected and candidate_entries:
                    selected_entry = candidate_entries[0]
                if candidate_entries:
                    candidate_bonds = _rank_candidates(candidate_entries, mol, target_spec, selected_entry)
                if sre_auto_selected:
                    match_count = 1
                    equivalence_groups = []
                candidate_meta = {
                    "source": "module_minus1",
                    "total_scanned": len(sre_candidates),
                    "matched": match_count,
                    "equivalence_count": len(equivalence_groups),
                    "synthetic_h_used": used_synthetic_h,
                    "module_minus1_resolution_policy": sre_resolution.get("resolution_policy"),
                }
                resolved.update(
                    {
                        "canonical_token": sre_resolution.get("canonical_token") or target_spec.token,
                        "token_context": sre_resolution.get("token_context") or target_spec.token_context,
                        "match_count": match_count,
                        "bond_type": sre_resolution.get("bond_type"),
                        "resolution_source": "module_minus1",
                    }
                )
                token_resolution_audit = {
                    "method": "module_minus1",
                    "match_count": match_count,
                    "canonical_token": resolved.get("canonical_token"),
                }
                target_resolution_confidence = 0.9 if selected_entry is not None else 0.55
                if isinstance(sre_resolution.get("resolution_confidence"), (int, float)):
                    target_resolution_confidence = max(
                        float(target_resolution_confidence),
                        float(sre_resolution.get("resolution_confidence")),
                    )
                if selected_entry and isinstance(selected_entry.get("element_pair"), list):
                    element_pair = list(selected_entry.get("element_pair") or [])
                elif bonds:
                    element_pair = _bond_element_pair(bonds[0]) if bonds else None
                index_base = 0
                return {
                    "warnings": warnings,
                    "errors": errors,
                    "resolved": resolved,
                    "bonds": bonds,
                    "candidate_atoms": candidate_atoms,
                    "candidate_bonds": candidate_bonds,
                    "candidate_meta": candidate_meta,
                    "equivalence_groups": equivalence_groups,
                    "index_base": index_base,
                    "element_pair": element_pair,
                    "token_info": token_info,
                    "token_resolution_audit": token_resolution_audit,
                    "target_resolution_confidence": target_resolution_confidence,
                    "match_count": match_count,
                    "selected_entry": selected_entry,
                }

    role_index: Dict[str, List[Dict[str, Any]]] = {}
    for entry in bond_roles:
        roles = {
            role_entry.get("role")
            for role_entry in entry.get("bond_roles", [])
            if role_entry.get("role") in CANONICAL_TOKENS
        }
        for role in roles:
            role_index.setdefault(role, []).append(entry)

    if target_spec.kind == "indices":
        idx_a, idx_b = target_spec.indices or (None, None)
        bond = mol.GetBondBetweenAtoms(idx_a, idx_b) if idx_a is not None else None
        if bond is None and idx_a is not None and idx_b is not None and idx_a > 0 and idx_b > 0:
            bond = mol.GetBondBetweenAtoms(idx_a - 1, idx_b - 1)
            if bond is not None:
                warnings.append("Target bond indices treated as 1-based; converted to 0-based.")
                idx_a -= 1
                idx_b -= 1
                index_base = 1
        if bond is None:
            errors.append("Target bond indices not found in molecule.")
        else:
            bonds = [bond]
            match_count = 1
            target_resolution_confidence = 0.98
            resolved = {"atom_indices": [idx_a, idx_b]}
            index_base = index_base or 0
            element_pair = _bond_element_pair(bond)
            candidate_atoms = [idx_a, idx_b]
            selected_entry = _entry_from_bond(bond_roles, bond)
            equivalence_groups = _equivalent_bonds_from_bonds(mol, bonds)
            if selected_entry:
                candidate_bonds = _rank_candidates(
                    [selected_entry], mol, target_spec, selected_entry
                )
    elif target_spec.kind == "smarts":
        if Chem is None:
            errors.append("SMARTS resolution requires RDKit.")
        else:
            bonds, candidate_atoms, resolved, token_info, smarts_errors = _resolve_smarts_bonds(
                mol,
                target_spec,
            )
            errors.extend(smarts_errors)
            match_count = len(bonds)
            candidates = [_entry_from_bond(bond_roles, bond) for bond in bonds if bond is not None]
            candidates = [entry for entry in candidates if entry]
            if match_count and not candidates:
                candidates = [_fallback_entry_from_bond(bond) for bond in bonds]
            equivalence_groups = _equivalent_groups_from_entries(mol, candidates)
            selected_entry = _select_best_entry(candidates)
            if selected_entry is None and len(candidates) == 1:
                selected_entry = candidates[0]
            if candidates:
                candidate_bonds = _rank_candidates(
                    candidates, mol, target_spec, selected_entry
                )
            target_resolution_confidence = _compute_target_resolution_confidence(
                target_spec.kind,
                match_count,
                selected_entry,
            )
    elif target_spec.kind == "token":
        if Chem is None:
            errors.append("Token resolution requires RDKit.")
        else:
            token_resolution_audit = {
                "input": target_spec.raw,
                "normalized": _normalize_token_text(target_spec.raw),
                "canonical_token": target_spec.token,
                "token_base": target_spec.token_base,
                "token_context": target_spec.token_context,
                "smarts_used": None,
                "method": None,
                "match_count": 0,
                "context_matched": None,
                "warnings": [],
                "errors": [],
            }
            context_matched = False
            smarts_entry = _token_smarts_entry(target_spec.token)
            if smarts_entry:
                smarts_spec = TargetBondSpec(
                    kind="smarts",
                    raw=target_spec.raw,
                    smarts=smarts_entry.get("smarts"),
                    bond_map=smarts_entry.get("bond_map"),
                )
                bonds, candidate_atoms, resolved_smarts, _, smarts_errors = _resolve_smarts_bonds(
                    mol,
                    smarts_spec,
                )
                if bonds:
                    candidates = [
                        _entry_from_bond(bond_roles, bond) or _fallback_entry_from_bond(bond)
                        for bond in bonds
                        if bond is not None
                    ]
                    candidates = [entry for entry in candidates if entry]
                    match_count = len(candidates)
                    if target_spec.token_context:
                        context_matched = any(
                            _entry_has_context_tag(entry, target_spec.token_context)
                            for entry in candidates
                        )
                    equivalence_groups = _equivalent_groups_from_entries(mol, candidates)
                    selected_entry = _select_best_entry(candidates)
                    if selected_entry is None and len(candidates) == 1:
                        selected_entry = candidates[0]
                    if candidates:
                        candidate_bonds = _rank_candidates(
                            candidates, mol, target_spec, selected_entry
                        )
                    target_resolution_confidence = _compute_target_resolution_confidence(
                        target_spec.kind,
                        match_count,
                        selected_entry,
                    )
                    resolved = {
                        "canonical_token": target_spec.token,
                        "token_base": target_spec.token_base,
                        "token_context": target_spec.token_context,
                        "match_count": match_count,
                        "smarts": smarts_spec.smarts,
                        "bond_map": list(resolved_smarts.get("bond_map") or []),
                    }
                    token_resolution_audit["method"] = "smarts"
                    token_resolution_audit["smarts_used"] = smarts_spec.smarts
                    token_resolution_audit["match_count"] = match_count
                    if target_spec.token_context and smarts_entry.get("context") == target_spec.token_context:
                        context_matched = True
                else:
                    token_resolution_audit["smarts_used"] = smarts_spec.smarts
                    if smarts_errors:
                        warnings.extend(smarts_errors)
                        token_resolution_audit["errors"] = smarts_errors

            if match_count == 0:
                candidates = role_index.get(target_spec.token_base or "", [])
                all_candidates = list(candidates)
                if target_spec.token_context:
                    context_key = target_spec.token_context.lower()
                    context_matches = [
                        entry
                        for entry in candidates
                        if _entry_has_context_tag(entry, context_key)
                    ]
                    if context_matches:
                        candidates = context_matches
                        context_matched = True
                    else:
                        if not context_matched:
                            warnings.append(
                                "Token context did not match any bond; using base role matches."
                            )

                if not candidates:
                    errors.append("No bonds matched the requested token.")
                else:
                    match_count = len(candidates)
                    equivalence_groups = _equivalent_groups_from_entries(mol, candidates)
                    selected_entry = _select_best_entry(candidates)
                    if selected_entry is None and len(candidates) == 1:
                        selected_entry = candidates[0]
                    if all_candidates:
                        candidate_bonds = _rank_candidates(
                            all_candidates, mol, target_spec, selected_entry
                        )
                    target_resolution_confidence = _compute_target_resolution_confidence(
                        target_spec.kind,
                        len(candidates),
                        selected_entry,
                    )
                resolved = {
                    "canonical_token": target_spec.token,
                    "token_base": target_spec.token_base,
                    "token_context": target_spec.token_context,
                    "match_count": match_count,
                }
                if token_resolution_audit:
                    token_resolution_audit["method"] = token_resolution_audit["method"] or "role_index"
                    token_resolution_audit["match_count"] = match_count

            if token_resolution_audit:
                token_resolution_audit["context_matched"] = (
                    context_matched if target_spec.token_context else None
                )
                token_resolution_audit["warnings"] = [
                    msg for msg in warnings if "Token context" in msg
                ]
                token_resolution_audit["errors"] = token_resolution_audit["errors"] or []
                if errors:
                    token_errors = [
                        err for err in errors if "token" in err.lower()
                    ]
                    if token_errors:
                        token_resolution_audit["errors"] = list(
                            dict.fromkeys(token_resolution_audit["errors"] + token_errors)
                        )
    elif target_spec.kind == "elements":
        element_pair = list(target_spec.elements) if target_spec.elements else None
        if element_pair is None:
            errors.append("Target bond element pair missing.")
        elif "H" in element_pair:
            candidates = _candidates_from_element_pair(bond_roles, element_pair)
            match_count = len(candidates)
            equivalence_groups = _equivalent_groups_from_entries(mol, candidates)
            if not candidates:
                errors.append("No matching implicit hydrogen bonds found.")
            else:
                warnings.append("Hydrogen bonds resolved via implicit hydrogens.")
                if len(candidates) == 1:
                    selected_entry = candidates[0]
                if candidates:
                    candidate_bonds = _rank_candidates(
                        candidates, mol, target_spec, selected_entry
                    )
                target_resolution_confidence = _compute_target_resolution_confidence(
                    target_spec.kind,
                    match_count,
                    selected_entry,
                )
            resolved = {"element_pair": list(element_pair), "match_count": match_count}
        else:
            candidates = _candidates_from_element_pair(bond_roles, element_pair)
            match_count = len(candidates)
            if not candidates:
                for bond in mol.GetBonds():
                    pair = _bond_element_pair(bond)
                    if pair and set(pair) == set(element_pair):
                        candidates.append(_fallback_entry_from_bond(bond))
            match_count = len(candidates)
            if not candidates:
                errors.append("No bonds match the requested element pair.")
            else:
                if match_count > 1:
                    warnings.append("Multiple bonds match the requested element pair.")
                resolved = {"element_pair": list(element_pair), "match_count": match_count}
                equivalence_groups = _equivalent_groups_from_entries(mol, candidates)
                if len(candidates) == 1:
                    selected_entry = candidates[0]
                if candidates:
                    candidate_bonds = _rank_candidates(
                        candidates, mol, target_spec, selected_entry
                    )
                target_resolution_confidence = _compute_target_resolution_confidence(
                    target_spec.kind,
                    match_count,
                    selected_entry,
                )
    else:
        errors.append("Target bond specification invalid.")

    if selected_entry:
        if selected_entry.get("kind") == "bond":
            bond = selected_entry["bond"]
            bonds = [bond]
            element_pair = element_pair or _bond_element_pair(bond)
            candidate_atoms = selected_entry.get("atom_indices", [])
        else:
            candidate_atoms = [selected_entry["carbon_index"]]
            element_pair = ["C", "H"]

    scanned_bonds = 1 if target_spec.kind == "indices" else len(bond_roles)
    candidate_meta = {
        "mode": target_spec.kind,
        "scanned_bonds": scanned_bonds,
        "matches": match_count,
    }

    return {
        "bonds": bonds,
        "warnings": warnings,
        "errors": errors,
        "resolved": resolved,
        "index_base": index_base,
        "element_pair": element_pair,
        "equivalence_groups": equivalence_groups,
        "candidate_atoms": candidate_atoms,
        "candidate_bonds": candidate_bonds,
        "candidate_meta": candidate_meta,
        "token_info": token_info,
        "token_resolution_audit": token_resolution_audit,
        "selected_entry": selected_entry,
        "match_count": match_count,
        "target_resolution_confidence": target_resolution_confidence,
    }


def _entry_from_bond(
    bond_roles: List[Dict[str, Any]],
    bond: Any,
) -> Optional[Dict[str, Any]]:
    bond_idx = bond.GetIdx()
    for entry in bond_roles:
        if entry.get("kind") == "bond" and entry.get("bond_idx") == bond_idx:
            return entry
    return None


def _select_best_entry(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ordered = sorted(
        candidates,
        key=lambda entry: entry.get("primary_role_confidence") or 0.0,
        reverse=True,
    )
    best = ordered[0]
    second = ordered[1]
    best_conf = best.get("primary_role_confidence", 0.0) or 0.0
    second_conf = second.get("primary_role_confidence", 0.0) or 0.0
    if best_conf >= 0.85 and (best_conf - second_conf) >= 0.15:
        return best
    return None


def _entry_to_candidate(entry: Dict[str, Any], mol: Optional[Any]) -> Dict[str, Any]:
    primary_role = entry.get("primary_role")
    local_context = _local_context_from_entry(entry, mol)
    if entry.get("kind") == "implicit_h":
        return {
            "carbon_index": entry.get("carbon_index"),
            "hydrogen_count": entry.get("hydrogen_count"),
            "mode": "implicit_H",
            "bond_roles": entry.get("bond_roles", []),
            "primary_role": primary_role,
            "primary_role_confidence": entry.get("primary_role_confidence"),
            "local_context": local_context,
        }
    if entry.get("kind") == "bond":
        return {
            "atom_indices": entry.get("atom_indices"),
            "element_pair": entry.get("element_pair"),
            "bond_order": entry.get("bond_order"),
            "is_aromatic": entry.get("is_aromatic"),
            "bond_roles": entry.get("bond_roles", []),
            "primary_role": primary_role,
            "primary_role_confidence": entry.get("primary_role_confidence"),
            "local_context": local_context,
        }
    return {}


def _rank_candidates(
    candidates: List[Dict[str, Any]],
    mol: Optional[Any],
    target_spec: Optional[TargetBondSpec] = None,
    selected_entry: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, float], Dict[str, Any]]] = []
    for entry in candidates:
        score_total, score_breakdown = _score_candidate(entry, mol, target_spec)
        scored.append((score_total, score_breakdown, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked: List[Dict[str, Any]] = []
    for idx, (score_total, score_breakdown, entry) in enumerate(scored, start=1):
        candidate = _entry_to_candidate(entry, mol)
        candidate["score_total"] = round(score_total, 3)
        candidate["score_breakdown"] = score_breakdown
        candidate["priority_rank"] = idx
        if selected_entry and entry is selected_entry:
            candidate["rejection_reason"] = None
        else:
            candidate["rejection_reason"] = (
                "not_selected" if selected_entry else "ambiguous_selection"
            )
        ranked.append(candidate)
    return ranked


def _local_context_from_entry(entry: Dict[str, Any], mol: Optional[Any]) -> Dict[str, Any]:
    if mol is None:
        return {}
    if entry.get("kind") == "bond":
        bond = entry.get("bond")
        if bond is None:
            return {}
        return {
            "in_ring": bond.IsInRing(),
            "is_aromatic": bond.GetIsAromatic(),
            "bond_order": float(bond.GetBondTypeAsDouble()),
            "neighbor_hetero_atoms": _neighbor_hetero_count(bond.GetBeginAtom(), bond.GetEndAtomIdx())
            + _neighbor_hetero_count(bond.GetEndAtom(), bond.GetBeginAtomIdx()),
        }
    if entry.get("kind") == "implicit_h":
        carbon_idx = entry.get("carbon_index")
        if carbon_idx is None:
            return {}
        atom = mol.GetAtomWithIdx(carbon_idx)
        return {
            "in_ring": atom.IsInRing(),
            "is_aromatic": atom.GetIsAromatic(),
            "neighbor_hetero_atoms": _neighbor_hetero_count(atom, exclude_idx=-1),
            "fluorine_neighbor_count": _neighbor_element_count(atom, "F"),
        }
    return {}


def _entry_has_context_tag(entry: Dict[str, Any], context_key: str) -> bool:
    if not context_key:
        return False
    wanted = context_key.lower()
    for tag in entry.get("context_tags", []) or []:
        if tag.lower() == wanted:
            return True
    for role_entry in entry.get("bond_roles", []):
        for tag in role_entry.get("tags") or []:
            if tag.lower() == wanted:
                return True
    return False


def _score_candidate(
    entry: Dict[str, Any],
    mol: Optional[Any],
    target_spec: Optional[TargetBondSpec],
) -> Tuple[float, Dict[str, float]]:
    score = 0.0
    breakdown: Dict[str, float] = {}

    primary_conf = entry.get("primary_role_confidence") or 0.0
    breakdown["primary_role_confidence"] = round(primary_conf, 3)
    score += primary_conf

    tags = set()
    for role_entry in entry.get("bond_roles", []):
        for tag in role_entry.get("tags") or []:
            tags.add(tag.lower())

    if target_spec is not None:
        if target_spec.kind == "elements":
            breakdown["element_match"] = 0.1
            score += 0.1

        if target_spec.token_context:
            key = f"{target_spec.token_context}_context"
            if target_spec.token_context.lower() in tags:
                breakdown[key] = 0.4
                score += 0.4
            else:
                breakdown[key] = -0.2
                score -= 0.2

        if target_spec.token_base == "ester__acyl_o":
            roles = {role.get("role") for role in entry.get("bond_roles", [])}
            if "acid__c_o" in roles:
                breakdown["acid_exclusion"] = -1.0
                score -= 1.0

        if target_spec.token_base and target_spec.token_base.startswith("ch__"):
            wanted = target_spec.token_base.split("__", 1)[1]
            key = f"ch_context_{wanted}"
            if wanted.lower() in tags:
                breakdown[key] = 0.2
                score += 0.2

    local_context = _local_context_from_entry(entry, mol)
    if local_context.get("is_aromatic"):
        breakdown["aromatic_adjacent"] = 0.05
        score += 0.05

    return score, breakdown


def _fallback_entry_from_bond(bond: Any) -> Dict[str, Any]:
    a_atom = bond.GetBeginAtom()
    b_atom = bond.GetEndAtom()
    a_sym = a_atom.GetSymbol()
    b_sym = b_atom.GetSymbol()
    symbols = {a_sym, b_sym}
    role = "unknown"
    confidence = 0.4
    tags: List[str] = []
    if "C" in symbols and symbols & {"F", "Cl", "Br", "I"}:
        carbon = a_atom if a_sym == "C" else b_atom
        role = "aryl_halide__c_x" if carbon.GetIsAromatic() else "alkyl_halide__c_x"
        confidence = 0.7
        tags = [sym.lower() for sym in symbols if sym in {"F", "Cl", "Br", "I"}]
    elif symbols == {"O", "H"}:
        oxygen = a_atom if a_sym == "O" else b_atom
        role = "oh__phenol" if any(nbr.GetIsAromatic() for nbr in oxygen.GetNeighbors()) else "oh__alcohol"
        confidence = 0.75
    elif symbols == {"N", "H"}:
        nitrogen = a_atom if a_sym == "N" else b_atom
        is_amide_n = False
        for nbr in nitrogen.GetNeighbors():
            if nbr.GetSymbol() != "C":
                continue
            for n_bond in nbr.GetBonds():
                if n_bond.GetBondTypeAsDouble() == 2.0 and n_bond.GetOtherAtom(nbr).GetSymbol() == "O":
                    is_amide_n = True
                    break
            if is_amide_n:
                break
        role = "nh__amide" if is_amide_n else "nh__amine"
        confidence = 0.72
    return {
        "kind": "bond",
        "bond": bond,
        "bond_idx": bond.GetIdx(),
        "atom_indices": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
        "element_pair": _bond_element_pair(bond),
        "bond_order": float(bond.GetBondTypeAsDouble()),
        "is_aromatic": bond.GetIsAromatic(),
        "bond_roles": [
            {
                "role": role,
                "confidence": confidence,
                "tags": tags,
                "evidence": "fallback",
            }
        ],
        "primary_role": role,
        "primary_role_confidence": confidence,
    }


def _candidates_from_element_pair(
    bond_roles: List[Dict[str, Any]],
    element_pair: List[str],
) -> List[Dict[str, Any]]:
    wanted = set(element_pair)
    candidates: List[Dict[str, Any]] = []
    for entry in bond_roles:
        pair = entry.get("element_pair")
        if pair and set(pair) == wanted:
            candidates.append(entry)
    return candidates


def _compute_target_resolution_confidence(
    kind: str,
    match_count: int,
    selected_entry: Optional[Dict[str, Any]],
) -> float:
    if match_count <= 0:
        return 0.0
    primary_conf = 0.0
    if selected_entry is not None:
        primary_conf = selected_entry.get("primary_role_confidence") or 0.0
    if match_count == 1:
        return min(0.95, 0.75 + 0.20 * float(primary_conf))
    confidence = sigmoid(0.0)
    return max(0.40, min(0.90, float(confidence)))


def _target_resolution_audit(
    match_count: Optional[int],
    candidate_bond_options: List[Dict[str, Any]],
    selected_entry: Optional[Dict[str, Any]],
    selection_mode: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_count = match_count if match_count is not None else len(candidate_bond_options)
    if not resolved_count:
        return {
            "match_count": resolved_count or 0,
            "top_score": None,
            "score_gap": None,
            "confidence": 0.0,
            "ambiguous": True,
        }

    primary_conf = 0.0
    if selected_entry is not None:
        primary_conf = selected_entry.get("primary_role_confidence") or 0.0
    if primary_conf == 0.0 and candidate_bond_options:
        primary_conf = candidate_bond_options[0].get("primary_role_confidence") or 0.0

    top_score = None
    score_gap = None
    if candidate_bond_options:
        top_score = candidate_bond_options[0].get("score_total")
        if len(candidate_bond_options) > 1:
            second_score = candidate_bond_options[1].get("score_total") or 0.0
            score_gap = float(top_score or 0.0) - float(second_score)

    if resolved_count == 1:
        if selection_mode in {"indices", "atom_indices"}:
            confidence = 0.99
        else:
            confidence = min(0.95, 0.75 + 0.20 * float(primary_conf))
    else:
        confidence = sigmoid(float(score_gap or 0.0))
        confidence = max(0.40, min(0.90, confidence))

    ambiguous = resolved_count > 1 or confidence < TARGET_RESOLUTION_LOW_THRESHOLD.value
    return {
        "match_count": resolved_count,
        "top_score": round(float(top_score), 3) if isinstance(top_score, (int, float)) else None,
        "score_gap": round(float(score_gap), 3) if isinstance(score_gap, (int, float)) else None,
        "confidence": round(float(confidence), 3),
        "ambiguous": bool(ambiguous),
    }


def _equivalent_groups_from_entries(
    mol: Any,
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    if all(entry.get("kind") == "implicit_h" for entry in entries):
        symm_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
        groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for entry in entries:
            carbon_idx = entry.get("carbon_index")
            if carbon_idx is None:
                continue
            symm = symm_classes[carbon_idx]
            role = entry.get("role")
            signature = (symm, role)
            group = groups.setdefault(
                signature,
                {
                    "signature": {"symmetry_class": symm, "role": role},
                    "members": [],
                },
            )
            group["members"].append(
                {
                    "carbon_index": carbon_idx,
                    "hydrogen_count": entry.get("hydrogen_count"),
                    "mode": "implicit_H",
                }
            )
        return list(groups.values())

    bonds = [entry.get("bond") for entry in entries if entry.get("kind") == "bond"]
    bonds = [bond for bond in bonds if bond is not None]
    return _equivalent_bonds_from_bonds(mol, bonds)


def _resolve_implicit_hydrogen_bonds(
    mol: Any,
    element_pair: Tuple[str, str],
) -> Tuple[List[Any], List[Dict[str, Any]], List[int]]:
    heavy_symbol = element_pair[0] if element_pair[1] == "H" else element_pair[1]
    candidate_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == heavy_symbol and atom.GetTotalNumHs() > 0
    ]
    if not candidate_atoms:
        return [], [], []

    symm_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    groups: Dict[int, Dict[str, Any]] = {}
    for atom in candidate_atoms:
        symm = symm_classes[atom.GetIdx()]
        group = groups.setdefault(
            symm,
            {
                "signature": {"symmetry_class": symm, "element_pair": list(element_pair)},
                "members": [],
            },
        )
        group["members"].append(
            {
                "carbon_index": atom.GetIdx(),
                "hydrogen_count": atom.GetTotalNumHs(),
                "mode": "implicit_H",
            }
        )

    return [], list(groups.values()), [atom.GetIdx() for atom in candidate_atoms]


def _resolve_smarts_bonds(
    mol: Any,
    target_spec: TargetBondSpec,
) -> Tuple[List[Any], List[int], Dict[str, Any], Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    bonds: List[Any] = []
    candidate_atoms: List[int] = []
    resolved: Dict[str, Any] = {}
    token_info = None

    if not target_spec.smarts:
        errors.append("SMARTS pattern missing.")
        return bonds, candidate_atoms, resolved, token_info, errors

    query = Chem.MolFromSmarts(target_spec.smarts)
    if query is None:
        errors.append("SMARTS pattern could not be parsed.")
        return bonds, candidate_atoms, resolved, token_info, errors

    bond_map = target_spec.bond_map or _extract_smarts_map_pair(target_spec.smarts)
    if bond_map is None:
        errors.append("SMARTS bond map missing; use atom map numbers or bond:a-b.")
        return bonds, candidate_atoms, resolved, token_info, errors

    map_to_query_idx: Dict[int, int] = {}
    for atom in query.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            map_to_query_idx[map_num] = atom.GetIdx()

    if bond_map[0] not in map_to_query_idx or bond_map[1] not in map_to_query_idx:
        errors.append("SMARTS bond map numbers do not match mapped atoms.")
        return bonds, candidate_atoms, resolved, token_info, errors

    matches = mol.GetSubstructMatches(query)
    if not matches:
        errors.append("SMARTS pattern did not match the molecule.")
        return bonds, candidate_atoms, resolved, token_info, errors

    for match in matches:
        idx_a = match[map_to_query_idx[bond_map[0]]]
        idx_b = match[map_to_query_idx[bond_map[1]]]
        bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
        if bond is None:
            continue
        bonds.append(bond)

    if bonds:
        candidate_atoms = [bonds[0].GetBeginAtomIdx(), bonds[0].GetEndAtomIdx()]

    resolved = {
        "smarts": target_spec.smarts,
        "bond_map": list(bond_map),
        "match_count": len(matches),
    }
    return bonds, candidate_atoms, resolved, token_info, errors


def _candidate_bonds_from_bonds(mol: Any, bonds: List[Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for bond in bonds:
        candidates.append(
            {
                "atom_indices": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                "element_pair": _bond_element_pair(bond),
                "bond_order": float(bond.GetBondTypeAsDouble()),
                "is_aromatic": bond.GetIsAromatic(),
            }
        )
    return candidates


def _candidate_bonds_from_implicit_h(mol: Any, carbon_indices: List[int]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for idx in carbon_indices:
        atom = mol.GetAtomWithIdx(idx)
        candidates.append(
            {
                "carbon_index": idx,
                "hydrogen_count": atom.GetTotalNumHs(),
                "mode": "implicit_H",
            }
        )
    return candidates


def _equivalent_bonds_from_bonds(mol: Any, bonds: List[Any]) -> List[Dict[str, Any]]:
    if not bonds:
        return []
    symm_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    num_atoms = len(symm_classes)
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for bond in bonds:
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        # Skip bonds with indices out of range (e.g., from mol with explicit Hs)
        if a_idx >= num_atoms or b_idx >= num_atoms:
            continue
        signature = (
            min(symm_classes[a_idx], symm_classes[b_idx]),
            max(symm_classes[a_idx], symm_classes[b_idx]),
            float(bond.GetBondTypeAsDouble()),
            bond.GetIsAromatic(),
        )
        group = groups.setdefault(
            signature,
            {
                "signature": {
                    "symmetry_classes": [signature[0], signature[1]],
                    "bond_order": signature[2],
                    "is_aromatic": signature[3],
                },
                "members": [],
            },
        )
        group["members"].append([a_idx, b_idx])
    return list(groups.values())


def _bond_context_from_bond(mol: Any, bond: Any) -> Dict[str, Any]:
    a_atom = bond.GetBeginAtom()
    b_atom = bond.GetEndAtom()
    a_idx = a_atom.GetIdx()
    b_idx = b_atom.GetIdx()
    element_pair = _bond_element_pair(bond)
    bond_type = _classify_bond_type(mol, bond)

    neighbor_hetero = _neighbor_hetero_count(a_atom, b_idx) + _neighbor_hetero_count(
        b_atom, a_idx
    )
    polarity = "polar" if _is_polar_pair(a_atom, b_atom) else "nonpolar"
    activated = bond_type in {"ester", "amide", "thioester", "phosphate"}
    bond_class = bond_type
    special_activation = False
    fluorine_neighbor_count = None
    if bond_type == "C-H":
        carbon_atom = a_atom if a_atom.GetSymbol() == "C" else b_atom
        fluorine_neighbor_count = _neighbor_element_count(carbon_atom, "F")
        if fluorine_neighbor_count >= 2:
            bond_class = "CH_fluorinated"
            special_activation = True
            activated = True

    return {
        "atom_indices": [a_idx, b_idx],
        "element_pair": element_pair,
        "bond_order": float(bond.GetBondTypeAsDouble()),
        "bond_type": bond_type,
        "bond_class": bond_class,
        "is_aromatic": bond.GetIsAromatic(),
        "in_ring": bond.IsInRing(),
        "activated": activated,
        "special_activation": special_activation,
        "polarity": polarity,
        "neighbor_hetero_atoms": neighbor_hetero,
        "fluorine_neighbor_count": fluorine_neighbor_count,
    }


def _bond_context_from_entry(mol: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
    if entry.get("kind") == "bond":
        if entry.get("bond") is not None:
            context = _bond_context_from_bond(mol, entry["bond"])
        else:
            element_pair = entry.get("element_pair") or [None, None]
            bond_type = (
                _classify_bond_type_from_elements(tuple(element_pair))
                if len(element_pair) == 2
                else "other"
            )
            context = {
                "atom_indices": entry.get("atom_indices") or [None, None],
                "element_pair": element_pair,
                "bond_order": entry.get("bond_order") or 1.0,
                "bond_type": bond_type,
                "bond_class": bond_type,
                "is_aromatic": entry.get("is_aromatic"),
                "in_ring": None,
                "activated": False,
                "special_activation": False,
                "polarity": "unknown",
                "neighbor_hetero_atoms": None,
                "fluorine_neighbor_count": None,
                "target_mode": "synthetic_H",
            }
    else:
        carbon_idx = entry.get("carbon_index")
        context = {
            "atom_indices": [carbon_idx],
            "element_pair": ["C", "H"],
            "bond_order": 1.0,
            "bond_type": "C-H",
            "bond_class": "C-H",
            "is_aromatic": False,
            "in_ring": False,
            "activated": False,
            "special_activation": False,
            "polarity": "unknown",
            "neighbor_hetero_atoms": None,
            "fluorine_neighbor_count": None,
            "target_mode": "implicit_H",
        }
        if carbon_idx is not None:
            carbon_atom = mol.GetAtomWithIdx(carbon_idx)
            context["is_aromatic"] = carbon_atom.GetIsAromatic()
            context["in_ring"] = carbon_atom.IsInRing()
            context["neighbor_hetero_atoms"] = _neighbor_hetero_count(
                carbon_atom, exclude_idx=-1
            )
            fluorine_neighbor_count = _neighbor_element_count(carbon_atom, "F")
            context["fluorine_neighbor_count"] = fluorine_neighbor_count
            if fluorine_neighbor_count >= 2:
                context["bond_class"] = "CH_fluorinated"
                context["special_activation"] = True
                context["activated"] = True

    context["bond_roles"] = entry.get("bond_roles", [])
    context["primary_role"] = entry.get("primary_role")
    context["primary_role_confidence"] = entry.get("primary_role_confidence")
    context["role_confidence"] = context.get("primary_role_confidence")
    if context.get("primary_role"):
        context["bond_role"] = context["primary_role"]
    return context


def _bond_context_from_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    local_context = candidate.get("local_context") or {}
    element_pair = candidate.get("element_pair")
    if not element_pair and candidate.get("mode") == "implicit_H":
        element_pair = ["C", "H"]
    context = {
        "atom_indices": candidate.get("atom_indices") or [candidate.get("carbon_index")],
        "element_pair": element_pair,
        "bond_order": candidate.get("bond_order") or 1.0,
        "bond_type": "other",
        "bond_class": "other",
        "is_aromatic": local_context.get("is_aromatic", False),
        "in_ring": local_context.get("in_ring", False),
        "activated": False,
        "special_activation": False,
        "polarity": "unknown",
        "neighbor_hetero_atoms": local_context.get("neighbor_hetero_atoms"),
        "fluorine_neighbor_count": local_context.get("fluorine_neighbor_count"),
    }
    context["bond_roles"] = candidate.get("bond_roles", [])
    context["primary_role"] = candidate.get("primary_role")
    context["primary_role_confidence"] = candidate.get("primary_role_confidence")
    context["role_confidence"] = context.get("primary_role_confidence")
    if context.get("primary_role"):
        context["bond_role"] = context["primary_role"]
    return context


def _apply_role_overrides(bond_context: Dict[str, Any]) -> Dict[str, Any]:
    role = bond_context.get("primary_role") or bond_context.get("bond_role")
    if not role:
        return bond_context

    if role.startswith("ch__"):
        bond_context["bond_type"] = "C-H"
        if role == "ch__fluorinated":
            bond_context["bond_class"] = "CH_fluorinated"
        else:
            bond_context["bond_class"] = "C-H"
        return bond_context
    if role.startswith("oh__"):
        bond_context["bond_type"] = "O-H"
        bond_context["bond_class"] = "O-H"
        return bond_context
    if role.startswith("nh__"):
        bond_context["bond_type"] = "N-H"
        bond_context["bond_class"] = "N-H"
        return bond_context

    route_key = ROLE_ROUTE_MAP.get(role)
    if route_key:
        bond_context["bond_type"] = route_key
        if bond_context.get("bond_class") in {None, "other"}:
            bond_context["bond_class"] = route_key
    return bond_context


def _bond_context_from_elements(target_spec: TargetBondSpec) -> Dict[str, Any]:
    if target_spec.kind == "token" and target_spec.token_base:
        role = target_spec.token_base
        bond_type = ROLE_ROUTE_MAP.get(role, "other")
        return {
            "atom_indices": [None, None],
            "element_pair": None,
            "bond_order": None,
            "bond_type": bond_type,
            "bond_class": bond_type,
            "bond_roles": [
                {
                    "role": role,
                    "confidence": None,
                    "tags": [],
                    "evidence": "token_hint",
                }
            ],
            "primary_role": role,
            "primary_role_confidence": None,
            "role_confidence": None,
            "is_aromatic": None,
            "in_ring": None,
            "activated": None,
            "special_activation": None,
            "polarity": "unknown",
            "neighbor_hetero_atoms": None,
            "fluorine_neighbor_count": None,
        }
    if target_spec.elements:
        bond_type = _classify_bond_type_from_elements(target_spec.elements)
        polarity = (
            "polar"
            if any(elem not in {"C", "H"} for elem in target_spec.elements)
            else "nonpolar"
        )
        return {
            "atom_indices": [None, None],
            "element_pair": list(target_spec.elements),
            "bond_order": None,
            "bond_type": bond_type,
            "bond_class": bond_type,
            "bond_roles": [],
            "primary_role": None,
            "primary_role_confidence": None,
            "role_confidence": None,
            "is_aromatic": None,
            "in_ring": None,
            "activated": bond_type in {"ester", "amide", "thioester", "phosphate"},
            "special_activation": None,
            "polarity": polarity,
            "neighbor_hetero_atoms": None,
            "fluorine_neighbor_count": None,
        }
    return {
        "atom_indices": [None, None],
        "element_pair": None,
        "bond_order": None,
        "bond_type": "other",
        "bond_class": "other",
        "bond_roles": [],
        "primary_role": None,
        "primary_role_confidence": None,
        "role_confidence": None,
        "is_aromatic": None,
        "in_ring": None,
        "activated": None,
        "special_activation": None,
        "polarity": "unknown",
        "neighbor_hetero_atoms": None,
        "fluorine_neighbor_count": None,
    }


def _bond_element_pair(bond: Any) -> Optional[List[str]]:
    if bond is None:
        return None
    return [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]


def _classify_bond_type(mol: Any, bond: Any) -> str:
    if bond.GetIsAromatic():
        return "aromatic"

    a_atom = bond.GetBeginAtom()
    b_atom = bond.GetEndAtom()
    symbols = {a_atom.GetSymbol(), b_atom.GetSymbol()}

    if "H" in symbols:
        if "C" in symbols:
            return "C-H"
        if "O" in symbols:
            return "O-H"
        if "N" in symbols:
            return "N-H"
        return "H-X"

    if _is_carbonyl_bond(a_atom, b_atom):
        if "O" in symbols:
            return "ester"
        if "N" in symbols:
            return "amide"
        if "S" in symbols:
            return "thioester"

    if "P" in symbols and "O" in symbols:
        return "phosphate"

    if symbols == {"C", "C"}:
        return "C-C"
    if symbols == {"C", "N"}:
        return "C-N"
    if symbols == {"C", "O"}:
        return "C-O"
    if symbols == {"C", "S"}:
        return "C-S"

    return "other"


def _classify_bond_type_from_elements(elements: Tuple[str, str]) -> str:
    pair = set(elements)
    if pair == {"C", "H"}:
        return "C-H"
    if pair == {"O", "H"}:
        return "O-H"
    if pair == {"N", "H"}:
        return "N-H"
    if pair == {"C", "C"}:
        return "C-C"
    if pair == {"C", "N"}:
        return "C-N"
    if pair == {"C", "O"}:
        return "C-O"
    if pair == {"C", "S"}:
        return "C-S"
    if pair == {"P", "O"}:
        return "phosphate"
    if pair in [{"C", "F"}, {"C", "Cl"}, {"C", "Br"}, {"C", "I"}]:
        return "alkyl_halide"
    return "other"


def _is_carbonyl_bond(a_atom: Any, b_atom: Any) -> bool:
    return _is_carbonyl_carbon(a_atom, b_atom) or _is_carbonyl_carbon(b_atom, a_atom)


def _is_carbonyl_carbon(atom: Any, partner: Any) -> bool:
    if atom.GetSymbol() != "C":
        return False
    if partner.GetSymbol() not in {"O", "N", "S"}:
        return False
    for bond in atom.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == "O":
                return True
    return False


def _neighbor_hetero_count(atom: Any, exclude_idx: int) -> int:
    count = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == exclude_idx:
            continue
        if neighbor.GetSymbol() not in {"C", "H"}:
            count += 1
    return count


def _neighbor_element_count(atom: Any, element: str) -> int:
    return sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == element)


def _is_polar_pair(a_atom: Any, b_atom: Any) -> bool:
    return a_atom.GetSymbol() not in {"C", "H"} or b_atom.GetSymbol() not in {"C", "H"}


def _assess_difficulty(
    bond_context: Dict[str, Any],
    structure_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float]:
    bond_role = bond_context.get("primary_role") or bond_context.get("bond_role")
    bond_type = bond_context.get("bond_type", "other")
    base_score = {
        "ester__acyl_o": 0.25,
        "anhydride__acyl_o": 0.3,
        "carbonate__acyl_o": 0.3,
        "acid__c_o": 0.35,
        "amide__c_n": 0.45,
        "lactam__c_n": 0.5,
        "beta_lactam__c_n": 0.55,
        "carbamate__c_n": 0.5,
        "urea__c_n": 0.5,
        "thioester__acyl_s": 0.55,
        "ether__c_o": 0.5,
        "acetal__c_o": 0.55,
        "glycosidic__acetal_o": 0.6,
        "epoxide__c_o": 0.6,
        "thioether__c_s": 0.6,
        "sulfonamide__s_n": 0.6,
        "sulfate_ester__s_o": 0.55,
        "phosphate_ester__p_o": 0.4,
        "alkyl_halide__c_x": 0.6,
        "aryl_halide__c_x": 0.8,
        "imine__c_n": 0.6,
        "nitrile__c_n": 0.7,
        "azo__n_n": 0.7,
        "disulfide__s_s": 0.6,
        "ch__aliphatic": 0.8,
        "ch__alpha_hetero": 0.82,
        "ch__allylic": 0.85,
        "ch__benzylic": 0.85,
        "ch__aryl": 0.9,
        "ch__fluorinated": 0.92,
        "cc__single": 0.75,
        "cc__aryl_alkyl": 0.85,
        "cc__aryl_aryl": 0.95,
    }.get(bond_role, None)

    if base_score is None:
        base_score = {
            "ester": 0.25,
            "phosphate": 0.25,
            "amide": 0.45,
            "thioester": 0.5,
            "C-O": 0.5,
            "C-N": 0.55,
            "C-S": 0.6,
            "C-C": 0.75,
            "C-H": 0.85,
            "aromatic": 0.8,
            "other": 0.6,
        }.get(bond_type, 0.6)

    score = base_score
    if bond_context.get("in_ring") is True:
        score += 0.05
    if bond_context.get("activated") is True and bond_context.get("bond_class") != "CH_fluorinated":
        score -= 0.05
    if bond_context.get("bond_class") == "CH_fluorinated":
        score += 0.1
    if bond_context.get("is_gas_like_small_molecule") is True:
        score += 0.1

    if structure_summary:
        heavy_atoms = structure_summary.get("heavy_atoms") or 0
        ring_count = structure_summary.get("ring_count") or 0
        rotatable = structure_summary.get("rotatable_bonds") or 0
        hetero_atoms = structure_summary.get("hetero_atoms") or 0

        if heavy_atoms >= 40:
            score += 0.1
        elif heavy_atoms >= 20:
            score += 0.05

        if ring_count >= 5:
            score += 0.1
        elif ring_count >= 3:
            score += 0.05

        if rotatable >= 20:
            score += 0.1
        elif rotatable >= 10:
            score += 0.05

        if hetero_atoms >= 8:
            score += 0.05

        if heavy_atoms >= 40 or ring_count >= 5:
            score += 0.05

    score = max(0.0, min(1.0, score))

    if score < 0.33:
        label = "EASY"
    elif score < 0.66:
        label = "MEDIUM"
    else:
        label = "HARD"

    return label, score


def _difficulty_label_from_score(score: float) -> str:
    if score < 0.33:
        return "EASY"
    if score < 0.66:
        return "MEDIUM"
    return "HARD"


def _bump_difficulty(level: str) -> str:
    idx = DIFFICULTY_ORDER.index(level)
    return DIFFICULTY_ORDER[min(idx + 1, len(DIFFICULTY_ORDER) - 1)]


def _lower_difficulty(level: str) -> str:
    idx = DIFFICULTY_ORDER.index(level)
    return DIFFICULTY_ORDER[max(idx - 1, 0)]


ROLE_ROUTE_MAP: Dict[str, str] = {
    "ester__acyl_o": "ester",
    "anhydride__acyl_o": "ester",
    "carbonate__acyl_o": "ester",
    "acid__c_o": "ester",
    "amide__c_n": "amide",
    "lactam__c_n": "amide",
    "beta_lactam__c_n": "amide",
    "carbamate__c_n": "amide",
    "urea__c_n": "amide",
    "thioester__acyl_s": "thioester",
    "ether__c_o": "C-O",
    "acetal__c_o": "C-O",
    "glycosidic__acetal_o": "C-O",
    "epoxide__c_o": "C-O",
    "thioether__c_s": "C-S",
    "sulfonamide__s_n": "sulfonamide",
    "sulfate_ester__s_o": "sulfate",
    "phosphate_ester__p_o": "phosphate",
    "alkyl_halide__c_x": "alkyl_halide",
    "aryl_halide__c_x": "aryl_halide",
    "imine__c_n": "imine",
    "nitrile__c_n": "nitrile",
    "azo__n_n": "azo",
    "disulfide__s_s": "disulfide",
    "ch__fluorinated": "C-H",
    "ch__aryl": "C-H",
    "ch__benzylic": "C-H",
    "ch__allylic": "C-H",
    "ch__alpha_hetero": "C-H",
    "ch__aliphatic": "C-H",
    "oh__alcohol": "O-H",
    "oh__phenol": "O-H",
    "nh__amine": "N-H",
    "nh__amide": "N-H",
    "cc__single": "C-C",
    "cc__aryl_alkyl": "C-C",
    "cc__aryl_aryl": "C-C",
}


def _select_route(bond_context: Dict[str, Any]) -> Dict[str, Any]:
    bond_role = bond_context.get("primary_role") or bond_context.get("bond_role")
    bond_type = ROLE_ROUTE_MAP.get(bond_role) if bond_role else None
    bond_type = bond_type or bond_context.get("bond_type", "other")
    base = ROUTE_LIBRARY.get(bond_type, ROUTE_LIBRARY["other"])
    route: Dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, list):
            route[key] = list(value)
        elif isinstance(value, dict):
            route[key] = dict(value)
        else:
            route[key] = value
    return route


def _scaffold_library_id_from_route(route: Optional[Dict[str, Any]]) -> Optional[str]:
    if not route:
        return None
    libraries = route.get("scaffold_libraries") or []
    if libraries:
        return libraries[0]
    primary = route.get("primary")
    if not primary:
        return "scaffold_lib_generic_v1"
    slug = re.sub(r"[^a-z0-9]+", "_", primary.lower()).strip("_")
    return f"scaffold_lib_{slug}_v1"


def _apply_constraint_defaults(
    constraints: OperationalConstraints,
) -> Tuple[OperationalConstraints, List[str]]:
    assumptions: List[str] = []
    metals_allowed = constraints.metals_allowed
    if metals_allowed is None:
        metals_allowed = True
        assumptions.append("metals_allowed default true")

    oxidation_allowed = constraints.oxidation_allowed
    if oxidation_allowed is None:
        oxidation_allowed = True
        assumptions.append("oxidation_allowed default true")

    effective = OperationalConstraints(
        ph_min=constraints.ph_min,
        ph_max=constraints.ph_max,
        temperature_c=constraints.temperature_c,
        metals_allowed=metals_allowed,
        oxidation_allowed=oxidation_allowed,
        host=constraints.host,
    )
    return effective, assumptions


def _apply_route_constraint_defaults(
    constraints: OperationalConstraints,
    route: Dict[str, Any],
    assumptions: List[str],
) -> Tuple[OperationalConstraints, List[str], bool]:
    defaults_used = False
    primary = route.get("primary")
    required_flags = route.get("required_flags") or []

    ph_min = constraints.ph_min
    ph_max = constraints.ph_max
    temperature_c = constraints.temperature_c

    if primary in {"serine_hydrolase", "thioesterase", "hydrolase", "sulfonamidase"}:
        if ph_min is None:
            ph_min = 6.5
            defaults_used = True
        if ph_max is None:
            ph_max = 8.0
            defaults_used = True
        if temperature_c is None:
            temperature_c = 30.0
            defaults_used = True
    elif primary in {"amidase", "metalloprotease"}:
        if ph_min is None:
            ph_min = 6.5
            defaults_used = True
        if ph_max is None:
            ph_max = 8.5
            defaults_used = True
        if temperature_c is None:
            temperature_c = 32.0
            defaults_used = True
    elif primary in {"P450", "monooxygenase", "dioxygenase"} or "oxidation" in required_flags:
        if ph_min is None:
            ph_min = 7.0
            defaults_used = True
        if ph_max is None:
            ph_max = 8.0
            defaults_used = True
        if temperature_c is None:
            temperature_c = 25.0
            defaults_used = True

    if defaults_used:
        assumptions.append("constraints defaulted from route")

    updated = OperationalConstraints(
        ph_min=ph_min,
        ph_max=ph_max,
        temperature_c=temperature_c,
        metals_allowed=constraints.metals_allowed,
        oxidation_allowed=constraints.oxidation_allowed,
        host=constraints.host,
    )
    return updated, assumptions, defaults_used


def _infer_job_type(requested_output: Optional[str]) -> Tuple[str, List[str]]:
    if not requested_output:
        return JOB_TYPE_STANDARD, []
    text = requested_output.strip().lower()
    if not text:
        return JOB_TYPE_STANDARD, []

    analysis_markers = ("analysis", "feasibility", "probe", "mechanism")
    if any(marker in text for marker in analysis_markers):
        return JOB_TYPE_ANALYSIS_ONLY, [
            "Requested output indicates analysis-only intent."
        ]

    reactive_markers = ("radical", "carbene", "nitrene", "anion", "cation")
    group_markers = ("cf3", "trifluoromethyl", "cn", "no2", "n3", "cho", "so3")
    if text.startswith("-") or text.endswith("-"):
        return JOB_TYPE_REAGENT_GENERATION, [
            "Requested output appears to be a functional group."
        ]
    if any(marker in text for marker in reactive_markers):
        return JOB_TYPE_REAGENT_GENERATION, [
            "Requested output indicates a reactive intermediate."
        ]
    if any(marker in text for marker in group_markers) and "trifluoromethane" not in text:
        return JOB_TYPE_REAGENT_GENERATION, [
            "Requested output implies CF3 transfer/reagent generation."
        ]

    return JOB_TYPE_STANDARD, []


def _is_fragment_output(text: str) -> bool:
    if not text:
        return False
    if text.startswith("-") or text.endswith("-"):
        return True
    reactive_markers = ("radical", "carbene", "nitrene", "anion", "cation")
    group_markers = ("cf3", "trifluoromethyl", "cn", "no2", "n3", "cho", "so3")
    if any(marker in text for marker in reactive_markers):
        return True
    if any(marker in text for marker in group_markers) and "trifluoromethane" not in text:
        return True
    return False


def _infer_reaction_intent(
    requested_output: Optional[str],
    bond_context: Dict[str, Any],
    job_type: str,
) -> Dict[str, Any]:
    intent = {
        "intent_type": "other",
        "intent_confidence": 0.3,
        "why": "Defaulted due to limited intent signal.",
        "requires_trap_target": False,
    }
    if job_type == JOB_TYPE_REAGENT_GENERATION:
        intent.update(
            {
                "intent_type": "reagent_generation",
                "intent_confidence": 0.8,
                "why": "Requested output indicates fragment or reactive group.",
                "requires_trap_target": True,
            }
        )
        return intent

    bond_role = bond_context.get("primary_role") or bond_context.get("bond_role")
    if bond_role in {
        "ester__acyl_o",
        "anhydride__acyl_o",
        "carbonate__acyl_o",
        "amide__c_n",
        "lactam__c_n",
        "beta_lactam__c_n",
        "carbamate__c_n",
        "urea__c_n",
        "thioester__acyl_s",
        "phosphate_ester__p_o",
        "sulfate_ester__s_o",
        "glycosidic__acetal_o",
    }:
        intent.update(
            {
                "intent_type": "hydrolysis",
                "intent_confidence": 0.65,
                "why": "Bond role implies hydrolytic transformation.",
                "requires_trap_target": False,
            }
        )
        return intent

    if requested_output:
        text = requested_output.strip().lower()
        if any(keyword in text for keyword in ("deprotect", "deacetyl", "deprotection")):
            intent.update(
                {
                    "intent_type": "deprotection",
                    "intent_confidence": 0.6,
                    "why": "Requested output indicates deprotection.",
                }
            )
        elif any(keyword in text for keyword in ("hydrolysis", "hydrolyze")):
            intent.update(
                {
                    "intent_type": "hydrolysis",
                    "intent_confidence": 0.6,
                    "why": "Requested output indicates hydrolysis.",
                }
            )
        elif "oxid" in text:
            intent.update(
                {
                    "intent_type": "oxidation",
                    "intent_confidence": 0.55,
                    "why": "Requested output indicates oxidation.",
                }
            )
        elif "reduc" in text:
            intent.update(
                {
                    "intent_type": "reduction",
                    "intent_confidence": 0.55,
                    "why": "Requested output indicates reduction.",
                }
            )
        elif any(keyword in text for keyword in ("fragment", "cleave", "break bond")):
            intent.update(
                {
                    "intent_type": "fragment_generation",
                    "intent_confidence": 0.5,
                    "why": "Requested output indicates fragment generation.",
                }
            )
        elif any(keyword in text for keyword in ("install", "functionalize", "add")):
            intent.update(
                {
                    "intent_type": "functional_group_install",
                    "intent_confidence": 0.45,
                    "why": "Requested output suggests functional group installation.",
                }
            )
        return intent

    return intent


def _requested_output_check(
    requested_output: Optional[str],
    reaction_intent: Dict[str, Any],
    bond_context: Dict[str, Any],
    token_context: Optional[str],
    token_context_matched: bool,
) -> Dict[str, Any]:
    if not requested_output:
        return {
            "requested": None,
            "is_fragment": False,
            "implies_reagent_generation": False,
            "requires_trap_target": False,
            "expected_products": [],
            "match": None,
            "confidence": 0.0,
            "why": "No requested output provided.",
        }

    text = requested_output.strip().lower()
    intent_type = reaction_intent.get("intent_type") or ""
    bond_role = bond_context.get("primary_role") or bond_context.get("bond_role") or ""
    is_fragment = _is_fragment_output(text)
    implies_reagent = intent_type == "reagent_generation" or is_fragment
    requires_trap = bool(reaction_intent.get("requires_trap_target")) or is_fragment
    expected_products: List[str] = []
    keywords: List[str] = []
    why = "Heuristic product check based on bond role and intent."
    match: Optional[bool] = None
    confidence = 0.45

    if implies_reagent:
        expected_products = ["fragment transfer or trapping"]
        confidence = 0.8
        why = (
            "Output is a substituent fragment; must be transferred onto an acceptor or trapped."
        )
        return {
            "requested": requested_output,
            "is_fragment": is_fragment,
            "implies_reagent_generation": True,
            "requires_trap_target": requires_trap,
            "expected_products": expected_products,
            "match": None,
            "confidence": round(confidence, 2),
            "why": why,
        }

    if bond_role.startswith("ester__acyl_o") or bond_role.startswith("anhydride__acyl_o"):
        expected_products = ["carboxylic acid", "alcohol"]
        keywords = ["acid", "carboxy", "salicylic", "benzoic", "phenol", "alcohol"]
    elif bond_role.startswith("amide__c_n") or bond_role.startswith("lactam__c_n") or bond_role.startswith(
        "beta_lactam__c_n"
    ):
        expected_products = ["carboxylic acid", "amine"]
        keywords = ["acid", "carboxy", "amine", "amino"]
    elif bond_role.startswith("glycosidic__") or bond_role.startswith("acetal__"):
        expected_products = ["sugar", "aglycone"]
        keywords = ["sugar", "glycos", "aglycone"]
    elif bond_role.startswith("alkyl_halide__") or bond_role.startswith("aryl_halide__"):
        expected_products = ["dehalogenated product"]
        keywords = ["dehalogen", "debrom", "dechloro", "deiodo"]
    elif intent_type in {"hydrolysis", "deprotection"}:
        expected_products = ["deprotected product", "leaving group"]
        keywords = ["acid", "amine", "alcohol", "phenol"]

    if "salicylic" in text and token_context == "acetyl" and token_context_matched:
        expected_products = ["salicylic acid", "acetic acid"]
        match = True
        confidence = 0.95
        why = "Acetyl ester hydrolysis of aspirin yields salicylic acid."
    elif expected_products:
        keyword_match = any(keyword in text for keyword in keywords)
        match = bool(keyword_match)
        confidence = 0.8 if keyword_match else 0.35
        if intent_type:
            confidence += 0.1
        confidence = min(0.95, confidence)
        why = "Requested output checked against expected products for the bond class."
    else:
        match = None
        confidence = 0.4
        why = "No product heuristics available for this bond class."

    return {
        "requested": requested_output,
        "is_fragment": is_fragment,
        "implies_reagent_generation": implies_reagent,
        "requires_trap_target": requires_trap,
        "expected_products": expected_products,
        "match": match,
        "confidence": round(confidence, 2),
        "why": why,
    }


def validate_request(job: Dict[str, Any], strict_validation: bool = True) -> Dict[str, Any]:
    """Validate requested_output against target bond and substrate context."""
    warnings: List[str] = []
    requested_output = job.get("requested_output")
    target_bond = str(job.get("target_bond") or "")
    target_spec = job.get("target_spec")
    bond_context = job.get("bond_context") or {}
    structure_summary = job.get("structure_summary") or {}
    reaction_intent = job.get("reaction_intent") or {}

    token_text = ""
    if target_spec is not None:
        token_text = (
            getattr(target_spec, "token", None)
            or getattr(target_spec, "token_base", None)
            or getattr(target_spec, "raw", None)
            or ""
        )
    if not token_text:
        token_text = target_bond
    token_norm = _normalize_token_text(str(token_text))
    bond_class = str(bond_context.get("bond_class") or bond_context.get("bond_type") or "")

    functional_groups = structure_summary.get("functional_groups") or {}
    has_amide = any(
        functional_groups.get(key)
        for key in ("amide", "lactam", "beta_lactam", "carbamate", "urea")
    )
    if ("amide" in token_norm) and ("c-n" in token_norm or "c_n" in token_norm):
        if not has_amide:
            warnings.append(
                "REQUEST_VALIDATION: target bond token indicates amide, but no amide group found."
            )
            return {
                "decision": "HALT",
                "halt_reason": "BOND_TOKEN_NOT_PRESENT",
                "warnings": warnings,
                "requested_output": requested_output,
                "clear_requested_output": False,
                "message": "Amide token requested but no amide functional group detected.",
            }

    if isinstance(requested_output, str) and requested_output.strip():
        output_lower = requested_output.lower()
        if ("amide" in token_norm) and ("salicylic acid" in output_lower):
            warnings.append(
                "REQUEST_VALIDATION: requested output 'salicylic acid' incompatible with amide_C-N."
            )
            if strict_validation:
                return {
                    "decision": "HALT",
                    "halt_reason": "REQUEST_OUTPUT_MISMATCH",
                    "warnings": warnings,
                    "requested_output": requested_output,
                    "clear_requested_output": False,
                    "message": "Requested output incompatible with amide C-N targeting.",
                }
            return {
                "decision": "OK",
                "halt_reason": None,
                "warnings": warnings,
                "requested_output": None,
                "clear_requested_output": True,
                "message": "Requested output cleared due to incompatibility.",
            }

        if reaction_intent.get("intent_type") == "hydrolysis":
            allowed_keywords: List[str] = []
            if "ester" in bond_class:
                allowed_keywords = ["acid", "carbox", "alcohol", "phenol", "salicylic"]
            elif "amide" in bond_class:
                allowed_keywords = ["amine", "acid", "carbox"]
            if allowed_keywords:
                if not any(keyword in output_lower for keyword in allowed_keywords):
                    warnings.append(
                        "REQUEST_VALIDATION: requested output not consistent with hydrolysis products."
                    )
                    if strict_validation:
                        return {
                            "decision": "HALT",
                            "halt_reason": "REQUEST_OUTPUT_MISMATCH",
                            "warnings": warnings,
                            "requested_output": requested_output,
                            "clear_requested_output": False,
                            "message": "Requested output incompatible with hydrolysis intent.",
                        }
                    return {
                        "decision": "OK",
                        "halt_reason": None,
                        "warnings": warnings,
                        "requested_output": None,
                        "clear_requested_output": True,
                        "message": "Requested output cleared due to incompatibility.",
                    }

    return {
        "decision": "OK",
        "halt_reason": None,
        "warnings": warnings,
        "requested_output": requested_output,
        "clear_requested_output": False,
        "message": None,
    }


def _boost_reaction_intent_confidence(
    reaction_intent: Dict[str, Any],
    bond_context: Dict[str, Any],
    match_count: Optional[int],
    token_context_matched: bool,
) -> Dict[str, Any]:
    if not reaction_intent:
        return reaction_intent
    base_conf = reaction_intent.get("intent_confidence") or 0.0
    role_confidence = bond_context.get("primary_role_confidence") or 0.0
    if (
        role_confidence >= 0.85
        and token_context_matched
        and match_count == 1
        and reaction_intent.get("intent_type") in {"hydrolysis", "deprotection"}
    ):
        boosted = max(base_conf, 0.8)
        reaction_intent["intent_confidence"] = round(boosted, 3)
        reaction_intent["why"] = "High-confidence role and context match with unambiguous target."
    return reaction_intent


def _success_definition(
    job_type: str,
    reaction_intent: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    intent_type = reaction_intent.get("intent_type") if reaction_intent else None
    any_activity = (
        "Detectable conversion or substrate turnover under plausible conditions."
    )
    target_specific = (
        "Product-specific conversion consistent with requested transformation under "
        "plausible conditions; not optimized for industrial yield."
    )
    if job_type == JOB_TYPE_REAGENT_GENERATION or intent_type == "reagent_generation":
        target_specific = (
            "Detectable formation of the requested fragment or trapped product under "
            "plausible conditions."
        )
    if job_type == JOB_TYPE_ANALYSIS_ONLY:
        any_activity = "Analysis-only request; no wet-lab success metric."
        target_specific = "Analysis-only request; no wet-lab success metric."
    return {
        "any_activity": any_activity,
        "target_specific": target_specific,
    }


def _normalize_trap_target(trap_target: Optional[str]) -> Optional[Dict[str, Any]]:
    if not trap_target:
        return None
    raw = trap_target.strip()
    if not raw:
        return None
    return {
        "smiles": None,
        "attach_site": None,
        "desired_product_smiles": None,
        "raw": raw,
    }


def _reaction_condition_field(
    job_type: str,
    reaction_intent: Dict[str, Any],
    route: Dict[str, Any],
    constraints: OperationalConstraints,
) -> Dict[str, Any]:
    intent_type = reaction_intent.get("intent_type") or ""
    primary = route.get("primary") or ""
    tags: List[str] = []

    if intent_type in {"hydrolysis", "deprotection"}:
        tags.append("proton_dependent")
        tags.append("enzyme_like")
    if primary in {"radical_SAM", "P450", "monooxygenase"}:
        tags.append("radical_dependent")
        tags.append("metal_dependent")
    if "metallo" in primary or "organometallic" in primary:
        tags.append("metal_dependent")
        tags.append("organometallic")
    if job_type == JOB_TYPE_REAGENT_GENERATION:
        tags.append("reagent_generation")

    if not tags:
        tags.append("general")

    family_defaults = {
        "serine_hydrolase": {"pH": (6.5, 8.0), "temp": (25.0, 37.0)},
        "amidase": {"pH": (6.5, 8.5), "temp": (25.0, 37.0)},
        "radical_SAM": {"pH": (6.8, 7.8), "temp": (20.0, 30.0)},
        "P450": {"pH": (7.0, 8.0), "temp": (20.0, 35.0)},
        "metallo_transfer_CF3": {"pH": (7.0, 8.5), "temp": (35.0, 55.0)},
    }
    default_range = family_defaults.get(primary, {"pH": (6.5, 8.2), "temp": (25.0, 45.0)})
    ph_low, ph_high = default_range["pH"]
    t_low, t_high = default_range["temp"]

    ph_min = constraints.ph_min
    ph_max = constraints.ph_max
    temp = constraints.temperature_c
    if ph_min is None and ph_max is None:
        ph_value = (ph_low + ph_high) / 2.0
    elif ph_min is None:
        ph_value = ph_max
    elif ph_max is None:
        ph_value = ph_min
    else:
        ph_value = (ph_min + ph_max) / 2.0
    temp_value = temp if temp is not None else (t_low + t_high) / 2.0

    notes: List[str] = []
    score = 0.75
    if ph_value < ph_low - 0.3 or ph_value > ph_high + 0.3:
        score -= 0.15
        notes.append("pH outside preferred window for this reaction family.")
    if ph_value < ph_low - 0.8 or ph_value > ph_high + 0.8:
        score -= 0.15
        notes.append("Large pH deviation may suppress catalytic residue activity.")

    if temp_value < t_low - 5:
        score -= 0.1
        notes.append("Low temperature may slow kinetics.")
    if temp_value > t_high + 5:
        score -= 0.1
        notes.append("High temperature may reduce binding stability.")

    score = max(0.0, min(1.0, score))

    return {
        "reaction_dependence": tags,
        "selected_reaction_families": [primary] + (route.get("secondary") or []),
        "given_conditions": {"pH": round(ph_value, 2), "temperature_c": round(temp_value, 1)},
        "condition_feasibility": {
            "given_conditions_score": round(score, 3),
            "notes": notes,
        },
        "optimum_conditions_hint": {
            "pH_range": [ph_low, ph_high],
            "temperature_c": [t_low, t_high],
        },
    }


def _condition_profile_from_constraints(
    constraints: OperationalConstraints,
) -> ConditionProfile:
    temp_c = constraints.temperature_c
    temperature_K = c_to_k(temp_c) if temp_c is not None else None
    ph_value = None
    if constraints.ph_min is not None and constraints.ph_max is not None:
        ph_value = (constraints.ph_min + constraints.ph_max) / 2.0
    elif constraints.ph_min is not None:
        ph_value = constraints.ph_min
    elif constraints.ph_max is not None:
        ph_value = constraints.ph_max
    profile = ConditionProfile(
        pH=ph_value,
        temperature_K=temperature_K,
        temperature_C=temp_c,
        ionic_strength=None,
        solvent=None,
        cofactors=[],
        salts_buffer=None,
        constraints="fixed" if constraints.ph_min == constraints.ph_max and temp_c is not None else None,
    )
    return profile.normalize()


def _delta_from_optimum(
    reaction_condition_field: Dict[str, Any],
    condition_profile: Dict[str, Any],
) -> Dict[str, Any]:
    given = dict(reaction_condition_field.get("given_conditions") or {})
    if given.get("pH") is None and condition_profile.get("pH") is not None:
        given["pH"] = condition_profile.get("pH")
    if given.get("temperature_c") is None and condition_profile.get("temperature_K") is not None:
        given["temperature_c"] = round(float(condition_profile["temperature_K"]) - 273.15, 1)
    optimum = reaction_condition_field.get("optimum_conditions_hint") or {}
    delta_ph = None
    delta_t = None
    if given.get("pH") is not None and optimum.get("pH_range"):
        delta_ph = round(
            float(given["pH"]) - float(sum(optimum["pH_range"]) / 2.0), 2
        )
    if given.get("temperature_c") is not None and optimum.get("temperature_c"):
        delta_t = round(
            float(given["temperature_c"]) - float(sum(optimum["temperature_c"]) / 2.0), 2
        )
    return {"delta_pH": delta_ph, "delta_T_C": delta_t}


def _reaction_task_from_inputs(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    reaction_intent: Dict[str, Any],
    route: Dict[str, Any],
) -> Dict[str, Any]:
    products = [requested_output] if requested_output else None
    scaffold_types = []
    primary = route.get("primary")
    if primary:
        scaffold_types.append(primary)
    scaffold_types.extend(route.get("secondary") or [])
    return ReactionTask(
        bond_to_break_or_form=target_bond,
        substrates=[smiles],
        products=products,
        mechanism_hint=reaction_intent.get("intent_type"),
        required_selectivity=None,
        allowed_scaffold_types=scaffold_types,
        safety_constraints=None,
    ).to_dict()


def _build_shared_io(
    smiles: str,
    target_bond: str,
    resolved: Dict[str, Any],
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
    condition_profile: ConditionProfile,
    run_id: Optional[str],
    output: SharedOutput,
    physics_block: Optional[Dict[str, Any]] = None,
    chemistry_block: Optional[Dict[str, Any]] = None,
    biology_block: Optional[Dict[str, Any]] = None,
    mechanism_block: Optional[Dict[str, Any]] = None,
    energy_ledger: Optional[Dict[str, Any]] = None,
    reaction_identity: Optional[Dict[str, Any]] = None,
    telemetry_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    target_bond_indices = None
    selected_bond = resolved.get("selected_bond") or {}
    if resolved.get("selection_mode") == "atom_indices":
        target_bond_indices = selected_bond.get("atom_indices")
    bond_spec = BondSpec(
        target_bond=target_bond,
        target_bond_indices=target_bond_indices,
        selection_mode=resolved.get("selection_mode"),
        resolved_target=resolved,
        context=bond_context,
    )
    substrate_context = SubstrateContext(
        smiles=smiles,
        structure_summary=structure_summary,
        descriptors=None,
    )
    telemetry = TelemetryContext(
        run_id=run_id or str(uuid.uuid4()),
        trace=["module0"],
        warnings=telemetry_warnings or [],
    )
    shared_input = SharedInput(
        bond_spec=bond_spec,
        condition_profile=condition_profile,
        substrate_context=substrate_context,
        telemetry=telemetry,
    )
    shared_io = SharedIO(input=shared_input, outputs={"module0": output})
    payload = shared_io.to_dict()
    payload["physics"] = physics_block or {}
    payload["chemistry"] = chemistry_block or {}
    payload["biology"] = biology_block or {}
    payload["mechanism"] = mechanism_block or {}
    payload["energy_ledger"] = energy_ledger or {}
    if reaction_identity:
        payload_input = payload.get("input") or {}
        payload_input["reaction_identity_hash"] = reaction_identity.get("hash")
        payload_input["reaction_identity_components"] = reaction_identity
        payload["input"] = payload_input
    return payload


def _normalized_physics_block(physics_audit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(physics_audit, dict):
        return {}
    delta_g = physics_audit.get("deltaG_dagger_kJ_per_mol")
    if delta_g is None:
        delta_g = physics_audit.get("deltaG_dagger_kj_mol")
    k_eff = physics_audit.get("k_eff_s_inv")
    if k_eff is None:
        k_eff = physics_audit.get("k_effective_s_inv")
    return {
        "deltaG_dagger_kj_mol": float(delta_g) if isinstance(delta_g, (int, float)) else None,
        "eyring_k_s_inv": physics_audit.get("eyring_k_s_inv"),
        "k_eff_s_inv": k_eff,
        "p_convert_horizon": physics_audit.get("p_convert_horizon"),
        "horizon_s": physics_audit.get("horizon_s"),
    }


def _normalized_chemistry_block(job_card: Dict[str, Any]) -> Dict[str, Any]:
    contract = job_card.get("chemistry_contract") or {}
    return {
        "functional_group_map": contract.get("functional_group_map") or {},
        "reaction_family": contract.get("reaction_family"),
        "leaving_group_score": contract.get("leaving_group_score"),
        "token_resolution_audit": job_card.get("token_resolution_audit") or {},
    }


def _normalized_biology_block(job_card: Dict[str, Any]) -> Dict[str, Any]:
    contract = job_card.get("biology_contract") or {}
    return {
        "enzyme_family_prior": contract.get("enzyme_family_prior") or {},
        "residue_protonation_fraction": contract.get("residue_protonation_fraction"),
        "cofactor_requirements": contract.get("cofactor_requirements") or {},
    }


def _normalized_mechanism_block(job_card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "contract": job_card.get("mechanism_contract") or {},
        "mismatch": job_card.get("mechanism_mismatch") or {},
    }


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


def _causal_discovery_summary(
    reaction_intent: Dict[str, Any],
    route: Dict[str, Any],
    reaction_condition_field: Dict[str, Any],
) -> Dict[str, Any]:
    primary = route.get("primary") or "unknown"
    dependence = reaction_condition_field.get("reaction_dependence") or []
    edges = []
    if "proton_dependent" in dependence:
        edges.append({"from": "pH", "to": "base_activation", "strength": 0.6})
    if "metal_dependent" in dependence:
        edges.append({"from": "metals", "to": "catalytic_activity", "strength": 0.55})
    edges.append({"from": "temperature", "to": "kinetics", "strength": 0.5})

    interventions = [
        {
            "action": "optimize_pH",
            "rationale": "Align residue protonation with mechanism requirements.",
            "expected_effect": "increase_success_probability",
        },
        {
            "action": "stabilize_retention",
            "rationale": "Improve dwell time in pocket.",
            "expected_effect": "increase_success_probability",
        },
    ]

    priors = {
        primary: {
            "prior_success_prob": 0.55,
            "uncertainty": 0.15,
        }
    }
    return {
        "causal_edges_topK": edges,
        "interventions_topK": interventions,
        "prior_success_prob": priors,
    }


def _augment_causal_discovery(
    causal_discovery: Dict[str, Any],
    bayes_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bayes_result:
        return causal_discovery
    updated = dict(causal_discovery)
    updated["route_success_probability_given_conditions"] = bayes_result.get("probability")
    updated["uncertainty_90ci"] = bayes_result.get("uncertainty_90ci")
    updated["key_causal_drivers"] = bayes_result.get("drivers", [])
    updated["model_diagnostics"] = bayes_result.get("diagnostics", {})
    return updated


def _augment_route_for_reagent_generation(
    route: Dict[str, Any],
    job_type: str,
    bond_class: str,
    requested_output: Optional[str],
) -> Dict[str, Any]:
    updated = dict(route)
    if job_type == JOB_TYPE_REAGENT_GENERATION or bond_class == "CH_fluorinated":
        updated.setdefault("expert_tracks", [])
        updated.setdefault("scaffold_libraries", [])
        mechanisms = list(updated.get("mechanisms", []))
        if bond_class == "CH_fluorinated" or _mentions_cf3(requested_output):
            mechanisms = _prioritize_cf3_mechanisms(mechanisms)
        updated["mechanisms"] = mechanisms
        existing_tracks = {track["track"] for track in updated["expert_tracks"]}
        if "metallo_transfer_CF3" not in existing_tracks:
            updated["expert_tracks"].insert(
                0,
                {
                    "track": "metallo_transfer_CF3",
                    "weight": 0.9,
                    "required": True,
                    "requires": ["metals"],
                },
            )
        if "radical_activation" not in existing_tracks:
            updated["expert_tracks"].append(
                {"track": "radical_activation", "weight": 0.6, "required": False}
            )
        updated.setdefault("required_flags", [])
        if "metals" not in updated["required_flags"]:
            updated["required_flags"].append("metals")
        updated["scaffold_libraries"] = list(
            {
                *updated.get("scaffold_libraries", []),
                "metalloenzyme_set",
                "organometallic_transfer_set",
            }
        )
        if "metal-CF3 transfer + trapping" not in updated["mechanisms"]:
            updated["mechanisms"].insert(1, "metal-CF3 transfer + trapping")
        if updated.get("primary") != "metallo_transfer_CF3":
            prior_primary = updated.get("primary")
            updated["primary"] = "metallo_transfer_CF3"
            secondary = list(updated.get("secondary", []))
            if prior_primary and prior_primary not in secondary:
                secondary.insert(0, prior_primary)
            updated["secondary"] = secondary
    return updated


def _augment_route_with_token_info(
    route: Dict[str, Any],
    token_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not token_info:
        return route
    updated = dict(route)
    mechanisms = list(updated.get("mechanisms", []))
    mechanism_hint = token_info.get("mechanism_hint")
    if mechanism_hint and mechanism_hint not in mechanisms:
        mechanisms.insert(0, mechanism_hint)
    updated["mechanisms"] = mechanisms

    secondary = list(updated.get("secondary", []))
    for family in token_info.get("family_hint") or []:
        if family not in secondary and family != updated.get("primary"):
            secondary.append(family)
    updated["secondary"] = secondary
    return updated


def _mentions_cf3(requested_output: Optional[str]) -> bool:
    if not requested_output:
        return False
    text = requested_output.strip().lower()
    return "cf3" in text or "trifluoromethyl" in text


def _prioritize_cf3_mechanisms(mechanisms: List[str]) -> List[str]:
    primary = "CF3_generation_and_trapping"
    ordered = [primary]
    seen = {primary}
    for item in mechanisms:
        if item not in seen and item != "C-H activation":
            ordered.append(item)
            seen.add(item)
    if "C-H activation" not in seen:
        ordered.append("C-H activation")
    return ordered


def _check_constraints(
    constraints: OperationalConstraints,
    route: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    violations: List[str] = []
    warnings: List[str] = []

    if (
        constraints.ph_min is not None
        and constraints.ph_max is not None
        and constraints.ph_min > constraints.ph_max
    ):
        violations.append("pH constraints are invalid (min greater than max).")

    required_flags = route.get("required_flags") or []
    if constraints.metals_allowed is False and "metals" in required_flags:
        violations.append("Metals required but forbidden by constraints.")

    if constraints.oxidation_allowed is False and "oxidation" in required_flags:
        violations.append("Oxidation required but forbidden by constraints.")

    if constraints.ph_min is not None and constraints.ph_min < 3.0:
        warnings.append("Low pH may destabilize catalytic residues.")
    if constraints.ph_max is not None and constraints.ph_max > 10.5:
        warnings.append("High pH may destabilize catalytic residues.")
    if constraints.temperature_c is not None:
        if constraints.temperature_c < 0:
            warnings.append("Sub-zero temperature may be impractical.")
        if constraints.temperature_c > 80:
            warnings.append("High temperature may destabilize proteins.")

    return violations, warnings


def _compute_polarity_descriptors(
    mol: Optional[Any],
    candidate_bonds: List[Any],
    candidate_atoms: List[int],
    bond_context: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], bool]:
    descriptors: Dict[str, Any] = {
        "gasteiger_charge_C": None,
        "gasteiger_charge_H": None,
        "gasteiger_charge_a": None,
        "gasteiger_charge_b": None,
        "bond_atom_symbols": None,
        "dipole_proxy": None,
        "is_gas_like_small_molecule": None,
        "fluorine_neighbor_count": None,
    }
    warnings: List[str] = []
    small_gas_flag = False

    bond_type = bond_context.get("bond_type")
    descriptor_required = bond_type == "C-H"

    if mol is None or Chem is None:
        missing = _missing_descriptor_fields(descriptors, descriptor_required)
        status = {
            "required": descriptor_required,
            "complete": not (descriptor_required and missing),
            "missing": missing,
        }
        return descriptors, warnings, status, small_gas_flag

    heavy_atoms = mol.GetNumHeavyAtoms()
    heavy_symbols = {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"}
    contains_only_chf = heavy_symbols.issubset({"C", "F"})
    is_gas_like = heavy_atoms < 6 and contains_only_chf
    descriptors["is_gas_like_small_molecule"] = is_gas_like
    small_gas_flag = is_gas_like
    descriptors.update(_detect_substrate_shape_features(mol))

    carbon_idx, hydrogen_idx = _resolve_charge_indices(
        mol,
        candidate_bonds,
        candidate_atoms,
    )

    if carbon_idx is not None:
        carbon_atom = mol.GetAtomWithIdx(carbon_idx)
        descriptors["fluorine_neighbor_count"] = _neighbor_element_count(carbon_atom, "F")

        if bond_type == "C-H" and descriptors["fluorine_neighbor_count"] is not None:
            if descriptors["fluorine_neighbor_count"] >= 2:
                descriptors["bond_class"] = "CH_fluorinated"
                descriptors["special_activation"] = True
                descriptors["activated"] = True

    bond_atom_indices = None
    if candidate_bonds:
        bond = candidate_bonds[0]
        a_idx, b_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        num_atoms = mol.GetNumAtoms()
        # Only use indices if they're within range of the original mol
        if a_idx < num_atoms and b_idx < num_atoms:
            bond_atom_indices = (a_idx, b_idx)
            descriptors["bond_atom_symbols"] = [
                mol.GetAtomWithIdx(a_idx).GetSymbol(),
                mol.GetAtomWithIdx(b_idx).GetSymbol(),
            ]

    if AllChem is None:
        warnings.append("RDKit AllChem unavailable; Gasteiger charges not computed.")
    else:
        mol_h = Chem.AddHs(mol)
        try:
            AllChem.ComputeGasteigerCharges(mol_h)
        except Exception:
            warnings.append("Gasteiger charge computation failed.")
        else:
            if carbon_idx is not None:
                descriptors["gasteiger_charge_C"] = _get_gasteiger_charge(
                    mol_h.GetAtomWithIdx(carbon_idx)
                )
                if hydrogen_idx is None:
                    hydrogen_idx = _find_attached_hydrogen_index(mol_h, carbon_idx)
            if hydrogen_idx is not None:
                descriptors["gasteiger_charge_H"] = _get_gasteiger_charge(
                    mol_h.GetAtomWithIdx(hydrogen_idx)
                )
            if bond_atom_indices:
                descriptors["gasteiger_charge_a"] = _get_gasteiger_charge(
                    mol_h.GetAtomWithIdx(bond_atom_indices[0])
                )
                descriptors["gasteiger_charge_b"] = _get_gasteiger_charge(
                    mol_h.GetAtomWithIdx(bond_atom_indices[1])
                )
            if (
                descriptors["gasteiger_charge_C"] is not None
                and descriptors["gasteiger_charge_H"] is not None
            ):
                descriptors["dipole_proxy"] = abs(
                    descriptors["gasteiger_charge_C"] - descriptors["gasteiger_charge_H"]
                )
                if bond_type == "C-H":
                    descriptors["polarity"] = _polarity_from_dipole(
                        descriptors["dipole_proxy"]
                    )

    missing = _missing_descriptor_fields(descriptors, descriptor_required)
    status = {
        "required": descriptor_required,
        "complete": not (descriptor_required and missing),
        "missing": missing,
    }
    return descriptors, warnings, status, small_gas_flag


def _detect_substrate_shape_features(mol: Optional[Any]) -> Dict[str, bool]:
    """Derive high-level substrate shape labels for route-compatibility priors."""
    shape = classify_substrate_shape(mol)
    return {
        "hydrophobic_organic": shape == "hydrophobic_organic",
        "polar_amino_acid_deriv": shape == "polar_amino_acid_deriv",
    }


def classify_substrate_shape(mol: Optional[Any]) -> str:
    """Classify substrate shape for route compatibility priors."""
    if mol is None or Chem is None:
        return "default"
    try:
        from rdkit.Chem import Descriptors
    except Exception:
        Descriptors = None
    try:
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        mw = (
            float(Descriptors.MolWt(mol))
            if Descriptors is not None
            else float(heavy_atoms * 12.0)
        )
        logp = (
            float(Descriptors.MolLogP(mol))
            if Descriptors is not None
            else 0.0
        )

        sulfonate = Chem.MolFromSmarts("S(=O)(=O)[O-,OH]")
        carboxylate = Chem.MolFromSmarts("C(=O)[O-,OH]")
        phosphate = Chem.MolFromSmarts("P(=O)([O-,OH])[O-,OH]")

        has_sulfonate = bool(sulfonate) and bool(mol.HasSubstructMatch(sulfonate))
        has_carboxylate = bool(carboxylate) and bool(mol.HasSubstructMatch(carboxylate))
        has_phosphate = bool(phosphate) and bool(mol.HasSubstructMatch(phosphate))

        if (has_sulfonate or has_carboxylate or has_phosphate) and mw < 200.0:
            return "polar_amino_acid_deriv"

        if logp > 2.0 and not (has_sulfonate or has_carboxylate or has_phosphate):
            return "hydrophobic_organic"

        return "default"
    except Exception:
        return "default"


def _route_prior_bucket(route_name: str) -> str:
    label = str(route_name or "").strip().lower()
    if "p450" in label:
        return "P450"
    if "nhi" in label or "non_heme" in label:
        return "NHI"
    return "default"


def get_route_prior(mol: Optional[Any], route_name: str) -> float:
    """Get compatibility prior for a substrate-route pair. Lower is better."""
    shape = classify_substrate_shape(mol)
    priors = ROUTE_COMPATIBILITY_PRIORS.get(
        shape, ROUTE_COMPATIBILITY_PRIORS["default"]
    )
    return float(
        priors.get(_route_prior_bucket(route_name), priors.get("default", 1.0))
    )


def _route_compatibility_weight(
    route_name: str,
    *,
    mol: Optional[Any] = None,
    bond_context: Optional[Dict[str, Any]] = None,
) -> float:
    if mol is not None:
        prior = get_route_prior(mol, route_name)
        return max(1e-6, 1.0 / max(prior, 1e-6))
    if isinstance(bond_context, dict):
        if bond_context.get("hydrophobic_organic"):
            prior = ROUTE_COMPATIBILITY_PRIORS["hydrophobic_organic"].get(
                _route_prior_bucket(route_name),
                ROUTE_COMPATIBILITY_PRIORS["hydrophobic_organic"]["default"],
            )
            return max(1e-6, 1.0 / max(float(prior), 1e-6))
        if bond_context.get("polar_amino_acid_deriv"):
            prior = ROUTE_COMPATIBILITY_PRIORS["polar_amino_acid_deriv"].get(
                _route_prior_bucket(route_name),
                ROUTE_COMPATIBILITY_PRIORS["polar_amino_acid_deriv"]["default"],
            )
            return max(1e-6, 1.0 / max(float(prior), 1e-6))
    return 1.0


def classify_ch_bond_class(mol: Optional[Any], atom_idx: int) -> str:
    """Classify the C-H bond class at a carbon atom index."""
    if mol is None or Chem is None:
        return "default"
    try:
        atom = mol.GetAtomWithIdx(int(atom_idx))
    except Exception:
        return "default"
    if atom.GetAtomicNum() != 6:
        return "default"

    neighbor_atoms = list(atom.GetNeighbors())
    if any(neighbor.GetAtomicNum() == 7 for neighbor in neighbor_atoms):
        return "alpha_amino_CH"

    for neighbor in neighbor_atoms:
        if neighbor.GetAtomicNum() == 6 and neighbor.GetIsAromatic():
            return "benzylic_CH"

    for neighbor in neighbor_atoms:
        if neighbor.GetAtomicNum() != 6 or neighbor.GetIsAromatic():
            continue
        for bond in neighbor.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            other = bond.GetOtherAtom(neighbor)
            if other.GetAtomicNum() == 6 and not other.GetIsAromatic():
                return "allylic_CH"

    return "aliphatic_CH"


def get_nhi_reference_bde(mol: Optional[Any], atom_idx: int) -> float:
    """Get the non-heme iron reference BDE for a target carbon atom."""
    bond_class = classify_ch_bond_class(mol, atom_idx)
    return float(
        NHI_REFERENCE_BDE_BY_CLASS.get(
            bond_class, NHI_REFERENCE_BDE_BY_CLASS["default"]
        )
    )


def _nhi_reference_bde_from_inputs(
    substrate_smiles: Optional[str],
    bond_class: Optional[str],
) -> Tuple[float, str]:
    if Chem is not None and substrate_smiles:
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(str(substrate_smiles)))
        except Exception:
            mol = None
        if mol is not None:
            class_priority = {
                "alpha_amino_CH": 0,
                "benzylic_CH": 1,
                "allylic_CH": 2,
                "aliphatic_CH": 3,
                "default": 4,
            }
            candidates: List[Tuple[int, str]] = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 6:
                    continue
                if not any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()):
                    continue
                bond_bucket = classify_ch_bond_class(mol, atom.GetIdx())
                candidates.append(
                    (class_priority.get(bond_bucket, class_priority["default"]), bond_bucket)
                )
            if candidates:
                candidates.sort(key=lambda item: item[0])
                selected = candidates[0][1]
                return (
                    float(
                        NHI_REFERENCE_BDE_BY_CLASS.get(
                            selected, NHI_REFERENCE_BDE_BY_CLASS["default"]
                        )
                    ),
                    selected,
                )

    bond_l = str(bond_class or "").lower()
    smiles_l = str(substrate_smiles or "")
    if (
        ("NCC" in smiles_l or "NC[C" in smiles_l or "C(N)C" in smiles_l)
        and ("S(=O)(=O)O" in smiles_l or "C(=O)O" in smiles_l or "C(=O)[O-]" in smiles_l)
    ):
        selected = "alpha_amino_CH"
    elif "alpha_hetero" in bond_l or "alpha_amino" in bond_l:
        selected = "alpha_amino_CH"
    elif "benzylic" in bond_l:
        selected = "benzylic_CH"
    elif "allylic" in bond_l:
        selected = "allylic_CH"
    elif bond_l in {"c-h", "ch", "ch__aliphatic", "ch__primary", "ch__secondary"}:
        selected = "aliphatic_CH"
    else:
        selected = "default"

    return (
        float(NHI_REFERENCE_BDE_BY_CLASS.get(selected, NHI_REFERENCE_BDE_BY_CLASS["default"])),
        selected,
    )


def _missing_descriptor_fields(
    descriptors: Dict[str, Any],
    required: bool,
) -> List[str]:
    if not required:
        return []
    required_fields = [
        "gasteiger_charge_C",
        "gasteiger_charge_H",
        "fluorine_neighbor_count",
        "is_gas_like_small_molecule",
        "dipole_proxy",
    ]
    return [field for field in required_fields if descriptors.get(field) is None]


def _bond_class_for_physics(bond_context: Dict[str, Any]) -> str:
    raw = (
        bond_context.get("bond_class")
        or bond_context.get("bond_type")
        or bond_context.get("primary_role")
        or bond_context.get("bond_role")
        or "unknown"
    )
    value = str(raw).lower()
    if "ester" in value:
        return "ester"
    if "amide" in value:
        return "amide"
    if "aryl" in value and ("br" in value or "c_br" in value or "halide" in value):
        return "aryl_c_br"
    return "unknown"


def _reaction_family_from_context(
    bond_context: Dict[str, Any],
    reaction_intent: Optional[Dict[str, Any]],
) -> str:
    bond_class = _bond_class_for_physics(bond_context)
    intent_type = None
    if isinstance(reaction_intent, dict):
        intent_type = reaction_intent.get("intent_type")
    intent = str(intent_type or "").lower()
    if intent in {"hydrolysis", "deprotection"}:
        return "hydrolysis"
    if "halide" in bond_class or "aryl" in bond_class:
        return "dehalogenation"
    if "ch" in bond_class:
        return "c_h_activation"
    if bond_class in {"ester", "amide"}:
        return "hydrolysis"
    return intent or bond_class or "unknown"


def _residue_fraction_for_route(
    route_name: str,
    pH: Optional[float],
    residue_hint: Optional[str],
) -> Dict[str, Any]:
    residue = residue_hint
    if not residue:
        label = str(route_name or "").lower()
        if "cys" in label or "thiol" in label:
            residue = "Cys"
        elif "serine" in label or "hydrolase" in label:
            residue = "His"
        else:
            residue = "His"
    fraction = residue_state_fraction(pH, residue)
    return {
        "residue": residue,
        "fraction": round(float(fraction), 3),
        "pH": pH,
        "pKa": PKA_CATALYTIC_GROUPS.get(str(residue).title()),
        "notes": ["v1 residue protonation fraction"],
    }


def _mechanism_expectations(route_label: str) -> Tuple[str, str]:
    label = str(route_label or "").lower()
    if "serine" in label:
        return "Ser", "Ser-His-Asp triad"
    if "cys" in label or "thiol" in label or "cysteine" in label:
        return "Cys", "Cys-His-Asp triad"
    if "metallo" in label or "metal" in label:
        return "Either", "metal-assisted active site"
    return "Either", "generic catalytic motif"


def _build_mechanism_spec(route_label: str, reaction_family: str) -> Dict[str, Any]:
    expected_nucleophile, expected_motif = _mechanism_expectations(route_label)
    spec = MechanismSpec(
        reaction_family=reaction_family or "unknown",
        route_label=route_label or "unknown",
        expected_nucleophile=expected_nucleophile,
        expected_motif=expected_motif,
        detected_nucleophile=None,
        detected_motif_residues={},
        compatibility_score=0.5,
        mismatch_reason=None,
        policy_action="KEEP_WITH_PENALTY",
    )
    return asdict(spec)


def _build_mechanism_contract(route_label: str, reaction_family: str) -> Dict[str, Any]:
    contract = resolve_mechanism(route_label)
    payload = contract.to_dict()
    if reaction_family:
        payload["reaction_family"] = reaction_family
    return payload


def _default_mechanism_mismatch(contract: Dict[str, Any]) -> Dict[str, Any]:
    expected = contract.get("expected_nucleophile")
    policy = contract.get("mismatch_policy_default")
    return {
        "status": "pending",
        "expected": expected,
        "observed": None,
        "penalty": 0.0,
        "policy": policy,
        "explanation": "awaiting scaffold nucleophile detection",
    }


def _build_scorecard_module0(
    job_card: Dict[str, Any],
    route_confidence: float,
    overall_confidence: float,
    route_p_cal: Optional[float],
    route_ci90: Optional[List[float]],
    n_eff: Optional[float],
    evidence_strength: Optional[float],
    data_support: Optional[float],
) -> Dict[str, Any]:
    features = (
        ((job_card.get("evidence_record") or {}).get("features_used") or {}).get("values")
        or {}
    )
    contributors = contributors_from_features(features, limit=5)
    calibration_source = (job_card.get("physics_audit") or {}).get("calibration_source")
    calibration_status = calibration_status_from_signals(
        calibration_source, data_support, evidence_strength
    )
    ci90_tuple = (
        (float(route_ci90[0]), float(route_ci90[1]))
        if route_ci90 and len(route_ci90) >= 2
        else None
    )
    metrics = [
        ScoreCardMetric(
            name="route_confidence",
            raw=round(float(route_confidence), 3),
            calibrated=round(float(route_p_cal or route_confidence), 3),
            ci90=ci90_tuple,
            n_eff=float(n_eff) if n_eff is not None else None,
            status=metric_status(route_confidence, n_eff),
            definition="Posterior probability that the selected route is feasible.",
            contributors=contributors,
        ),
        ScoreCardMetric(
            name="overall_signal",
            raw=round(float(overall_confidence), 3),
            calibrated=round(float(overall_confidence), 3),
            ci90=None,
            n_eff=float(n_eff) if n_eff is not None else None,
            status=metric_status(overall_confidence, n_eff),
            definition="Composite signal using route confidence, resolution, and wetlab prior.",
            contributors=contributors,
        ),
    ]
    return ScoreCard(module_id=0, metrics=metrics, calibration_status=calibration_status).to_dict()


def _build_score_ledger_module0(
    job_card: Dict[str, Any],
    route_confidence: float,
    overall_confidence: float,
) -> Dict[str, Any]:
    physics_audit = job_card.get("physics_audit") or {}
    energy_ledger = job_card.get("energy_ledger") or {}
    confidence = job_card.get("confidence") or {}
    target_resolution = confidence.get("target_resolution")
    prior_target = (
        physics_audit.get("prior_success_probability_final")
        or physics_audit.get("route_prior_target_specific")
        or physics_audit.get("p_convert_horizon")
    )
    terms = [
        ScoreTerm(
            name="physics_prior_target_specific",
            value=float(prior_target) if isinstance(prior_target, (int, float)) else None,
            unit="probability",
            formula="detectability_probability(k_eff, horizon)",
            inputs={
                "k_eff_s_inv": energy_ledger.get("k_eff_s_inv")
                or physics_audit.get("k_eff_s_inv"),
                "horizon_s": energy_ledger.get("horizon_s")
                or physics_audit.get("horizon_s"),
                "expected_conversion": energy_ledger.get("p_success_horizon")
                or physics_audit.get("expected_conversion"),
            },
            notes="Physics feasibility under horizon; not a certainty estimate.",
        ),
        ScoreTerm(
            name="route_confidence",
            value=float(route_confidence) if isinstance(route_confidence, (int, float)) else None,
            unit="probability",
            formula="bayes_router.posterior",
            inputs={
                "support": job_card.get("data_support"),
                "evidence_strength": job_card.get("evidence_strength"),
            },
            notes="Posterior route confidence from router + heuristics.",
        ),
        ScoreTerm(
            name="overall_signal",
            value=float(overall_confidence) if isinstance(overall_confidence, (int, float)) else None,
            unit="probability",
            formula="route_confidence * resolution * wetlab_prior",
            inputs={
                "target_resolution": target_resolution,
                "wetlab_prior": confidence.get("wetlab_prior"),
            },
            notes="Composite confidence used for routing.",
        ),
        ScoreTerm(
            name="target_resolution",
            value=float(target_resolution) if isinstance(target_resolution, (int, float)) else None,
            unit="probability",
            formula="bond_resolution_confidence",
            inputs={"match_count": (job_card.get("resolved_target") or {}).get("match_count")},
            notes="Bond resolution certainty from matching logic.",
        ),
        ScoreTerm(
            name="route_confidence_low_threshold",
            value=float(ROUTE_CONFIDENCE_LOW_THRESHOLD.value),
            unit="probability",
            formula="config.ROUTE_CONFIDENCE_LOW_THRESHOLD",
            inputs={},
            notes=ROUTE_CONFIDENCE_LOW_THRESHOLD.rationale,
        ),
        ScoreTerm(
            name="target_resolution_low_threshold",
            value=float(TARGET_RESOLUTION_LOW_THRESHOLD.value),
            unit="probability",
            formula="config.TARGET_RESOLUTION_LOW_THRESHOLD",
            inputs={},
            notes=TARGET_RESOLUTION_LOW_THRESHOLD.rationale,
        ),
    ]
    return ScoreLedger(module_id=0, terms=terms).to_dict()


def _normalized_selected_bond(resolved: Dict[str, Any]) -> Dict[str, Any]:
    selected = resolved.get("selected_bond") or {}
    if isinstance(selected, dict):
        if selected.get("atom_indices"):
            return {"atom_indices": list(selected.get("atom_indices") or [])}
        if selected.get("carbon_index") is not None:
            return {
                "carbon_index": selected.get("carbon_index"),
                "hydrogen_count": selected.get("hydrogen_count"),
                "mode": selected.get("mode"),
            }
    if resolved.get("atom_indices"):
        return {"atom_indices": list(resolved.get("atom_indices") or [])}
    return {}


def _reaction_identity_payload(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    trap_target: Optional[str],
    resolved: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "smiles": smiles,
        "target_bond": target_bond,
        "requested_output": requested_output,
        "trap_target": trap_target,
        "selection_mode": resolved.get("selection_mode"),
        "selected_bond": _normalized_selected_bond(resolved),
    }
    serial = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["hash"] = hashlib.sha256(serial.encode("utf-8")).hexdigest()
    return payload


def _energy_ledger_from_job_card(job_card: Dict[str, Any]) -> Dict[str, Any]:
    physics_audit = job_card.get("physics_audit") or {}
    ledger = physics_audit.get("energy_ledger")
    if isinstance(ledger, dict) and ledger:
        return dict(ledger)
    ci90 = job_card.get("route_ci90")
    p_success = (
        physics_audit.get("prior_success_probability_final")
        or physics_audit.get("route_prior_target_specific")
        or physics_audit.get("p_convert_horizon")
    )
    return {
        "deltaG_dagger_kJ": physics_audit.get("deltaG_dagger_kJ_per_mol"),
        "deltaG_bind_kJ": None,
        "eyring_k_s_inv": physics_audit.get("eyring_k_s_inv"),
        "k_diff_cap_s_inv": physics_audit.get("k_diff_cap_s_inv"),
        "k_eff_s_inv": physics_audit.get("k_eff_s_inv"),
        "p_success_horizon": p_success,
        "horizon_s": physics_audit.get("horizon_s"),
        "ci90": ci90,
        "n_eff": job_card.get("n_eff"),
        "notes": ["module0 baseline energy ledger"],
    }


def _temperature_k_from_profile(condition_profile: ConditionProfile) -> float:
    temp_k = condition_profile.temperature_K
    if temp_k is None and condition_profile.temperature_C is not None:
        temp_k = c_to_k(condition_profile.temperature_C)
    if temp_k is None:
        temp_k = 298.15
    return float(temp_k)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _physics_shrink_factor(
    support: Optional[float],
    evidence_strength: Optional[float],
) -> float:
    support_val = max(0.0, float(support or 0.0))
    evidence_val = max(0.0, float(evidence_strength or 0.0))
    strength = min(1.0, (support_val + evidence_val) / 5.0)
    return 0.6 + 0.4 * _clamp01(strength)


def _protonation_factor_for_route(
    route_name: str,
    pH: Optional[float],
) -> Tuple[float, List[str], Optional[str], bool]:
    notes: List[str] = []
    if pH is None:
        return 0.5, ["pH unknown; default protonation factor 0.5"], None, True
    route_label = str(route_name or "unknown").lower()
    residue = "His"
    mode = "base"
    if "thiol" in route_label or "cys" in route_label:
        residue = "Cys"
        mode = "base"
    elif "acid" in route_label or "aspart" in route_label or "glu" in route_label:
        residue = "Asp"
        mode = "acid"
    pka = PKA_CATALYTIC_GROUPS.get(residue)
    if pka is None:
        return 0.5, [f"pKa unknown for {residue}; default 0.5"], residue, True
    if mode == "acid":
        factor = fraction_protonated(float(pH), float(pka))
    else:
        factor = fraction_deprotonated(float(pH), float(pka))
    notes.append(f"{residue} {mode} state from pH/pKa")
    return max(0.0, min(1.0, float(factor))), notes, residue, False


def _solvent_penalty_from_profile(condition_profile: ConditionProfile) -> Dict[str, Any]:
    penalty_info = solvent_penalty(condition_profile.solvent)
    return {
        "penalty": penalty_info.get("penalty", 0.7),
        "solvent_unknown": penalty_info.get("solvent_unknown", True),
        "note": penalty_info.get("note"),
    }


def _shrink_probability(
    p_value: Optional[float],
    shrink_factor: float,
) -> Optional[float]:
    if p_value is None:
        return None
    try:
        value = float(p_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    shrink = _clamp01(shrink_factor)
    p_shrunk = value * shrink
    return max(0.01, min(0.99, float(p_shrunk)))


def _apply_physics_shrink(
    priors: Dict[str, Dict[str, Any]],
    shrink_factor: float,
) -> Dict[str, Dict[str, Any]]:
    for payload in priors.values():
        raw = payload.get("prior_success_probability")
        shrunk = _shrink_probability(raw, shrink_factor)
        payload["prior_success_probability_raw"] = raw
        payload["prior_success_probability"] = (
            round(float(shrunk), 4) if shrunk is not None else None
        )
        payload["shrink_factor"] = round(float(shrink_factor), 3)
    return priors


def _prior_probability_from_k(
    k_s_inv: Optional[float],
    horizon_s: Optional[float],
    alpha: float = 2.0,
    beta: float = 1.0,
) -> Optional[float]:
    detectability = _detectability_prior_from_k(k_s_inv, horizon_s)
    prior_before = detectability.get("prior_before_damping")
    if prior_before is None:
        return None
    return max(0.01, min(0.99, float(prior_before)))


def _detectability_prior_from_k(
    k_s_inv: Optional[float],
    horizon_s: Optional[float],
) -> Dict[str, Optional[float]]:
    if k_s_inv is None or horizon_s is None:
        return {
            "expected_conversion": None,
            "detection_threshold": DETECTION_THRESHOLD_CONVERSION,
            "noise_floor": DETECTION_NOISE_FLOOR,
            "prior_before_damping": None,
        }
    try:
        rate = float(k_s_inv)
        horizon = float(horizon_s)
    except (TypeError, ValueError):
        return {
            "expected_conversion": None,
            "detection_threshold": DETECTION_THRESHOLD_CONVERSION,
            "noise_floor": DETECTION_NOISE_FLOOR,
            "prior_before_damping": None,
        }
    if not math.isfinite(rate) or not math.isfinite(horizon) or horizon <= 0.0:
        return {
            "expected_conversion": None,
            "detection_threshold": DETECTION_THRESHOLD_CONVERSION,
            "noise_floor": DETECTION_NOISE_FLOOR,
            "prior_before_damping": None,
        }
    expected_conversion = 1.0 - math.exp(-max(0.0, rate) * max(0.0, horizon))
    expected_conversion = max(0.0, min(1.0, float(expected_conversion)))
    z = (expected_conversion - DETECTION_THRESHOLD_CONVERSION) / max(
        1e-6, float(DETECTABILITY_SIGMOID_WIDTH)
    )
    if z >= 50.0:
        prior_before = 1.0
    elif z <= -50.0:
        prior_before = 0.0
    else:
        prior_before = 1.0 / (1.0 + math.exp(-z))
    return {
        "expected_conversion": float(expected_conversion),
        "detection_threshold": DETECTION_THRESHOLD_CONVERSION,
        "noise_floor": DETECTION_NOISE_FLOOR,
        "prior_before_damping": float(prior_before),
    }


def _mechanism_prior_weight(mechanism_family: str) -> float:
    name = str(mechanism_family or "unknown").lower()
    if "radical" in name or "sam" in name or "p450" in name:
        return 0.8
    if "metallo" in name or "metal" in name:
        return 0.9
    if "hydrolase" in name or "esterase" in name or "amidase" in name:
        return 1.0
    return 0.95


def _context_uncertainty_penalty(
    condition_profile: ConditionProfile,
    known_scaffold: bool,
) -> Tuple[float, List[str]]:
    missing_notes: List[str] = []
    missing_solvent = not condition_profile.solvent
    missing_ionic = condition_profile.ionic_strength is None
    missing_cofactors = not condition_profile.cofactors
    missing_scaffold = not known_scaffold
    if missing_solvent:
        missing_notes.append("solvent unknown")
    if missing_ionic:
        missing_notes.append("ionic strength unknown")
    if missing_cofactors:
        missing_notes.append("cofactors unspecified")
    if missing_scaffold:
        missing_notes.append("no known scaffold")
    if missing_solvent and missing_ionic and missing_scaffold:
        return 0.6, missing_notes
    return 1.0, missing_notes


def _target_specific_factor(bond_class: Optional[str]) -> float:
    name = (bond_class or "unknown").lower()
    if "ester" in name:
        return 0.85
    if "amide" in name:
        return 0.35
    if "aryl" in name and "halide" in name:
        return 0.45
    if "c-h" in name or "c_h" in name or name.startswith("ch"):
        return 0.25
    return 0.60


def _load_latest_calibration_model() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cached = _CALIBRATION_CACHE.get("model")
    if cached is not None:
        return cached, _CALIBRATION_CACHE.get("source")
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_dir = repo_root / "artifacts"
    latest_path = artifacts_dir / "latest.json"
    candidate_path: Optional[Path] = None
    if latest_path.is_file():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        pack_path = payload.get("path") or payload.get("pack_path") or payload.get("artifacts_path")
        if pack_path:
            candidate = Path(pack_path)
            if candidate.is_dir():
                candidate_path = candidate / "calibration_module0_v1.json"
            elif candidate.is_file():
                candidate_path = candidate
    if candidate_path is None or not candidate_path.is_file():
        fallback = artifacts_dir / "calibration_module0_v1.json"
        candidate_path = fallback if fallback.is_file() else None
    if candidate_path is None:
        _CALIBRATION_CACHE["model"] = None
        _CALIBRATION_CACHE["source"] = None
        return None, None
    try:
        model = json.loads(candidate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        model = None
    _CALIBRATION_CACHE["model"] = model
    _CALIBRATION_CACHE["source"] = str(candidate_path)
    return model, str(candidate_path)


def _calibrator_predict_proba(
    features: Dict[str, float],
    model: Dict[str, Any],
) -> Optional[float]:
    order = model.get("feature_order") or []
    weights = model.get("weights") or []
    means = model.get("feature_mean") or []
    scales = model.get("feature_scale") or []
    bias = float(model.get("bias") or 0.0)
    if not order or not weights:
        return None
    z = bias
    for idx, name in enumerate(order):
        value = float(features.get(name, 0.0))
        mean = float(means[idx]) if idx < len(means) else 0.0
        scale = float(scales[idx]) if idx < len(scales) and scales[idx] else 1.0
        weight = float(weights[idx]) if idx < len(weights) else 0.0
        z += weight * ((value - mean) / scale)
    if z >= 0:
        prob = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        prob = exp_z / (1.0 + exp_z)
    return max(0.0, min(1.0, float(prob)))


def _calibrated_any_activity_probability(
    smiles: str,
    target_bond: str,
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
    physics_payload: Dict[str, Any],
    condition_profile: ConditionProfile,
    route_name: str,
) -> Tuple[Optional[float], Optional[str], Optional[str], Dict[str, Any]]:
    model, source = _load_latest_calibration_model()
    if not model:
        return None, None, None, {}
    unity_condition = UnityConditionProfile(
        pH=condition_profile.pH,
        temperature_K=condition_profile.temperature_K,
        temperature_C=condition_profile.temperature_C,
        ionic_strength=condition_profile.ionic_strength,
        solvent=condition_profile.solvent,
        cofactors=list(condition_profile.cofactors or []),
        salts_buffer=condition_profile.salts_buffer,
        constraints=condition_profile.constraints,
    )
    unity_bond = UnityBondContext(
        bond_role=bond_context.get("primary_role") or bond_context.get("bond_role"),
        bond_role_confidence=bond_context.get("primary_role_confidence")
        or bond_context.get("role_confidence"),
        bond_class=bond_context.get("bond_class"),
        polarity=bond_context.get("polarity"),
        atom_count=structure_summary.get("atom_count"),
        hetero_atoms=structure_summary.get("hetero_atoms"),
        ring_count=structure_summary.get("ring_count"),
    )
    unity_physics = UnityPhysicsAudit(
        deltaG_dagger_kJ_per_mol=physics_payload.get("delta_g_act_kj_mol")
        or physics_payload.get("deltaG_dagger_kJ_per_mol"),
        eyring_k_s_inv=physics_payload.get("k_s_inv") or physics_payload.get("eyring_k_s_inv"),
        k_eff_s_inv=physics_payload.get("k_eff_s_inv"),
        temperature_K=physics_payload.get("temperature_K") or condition_profile.temperature_K,
        horizon_s=physics_payload.get("horizon_s"),
        notes=list(physics_payload.get("notes") or []),
    )
    unity_module0 = UnityModule0Out(route_family=route_name)
    record = UnityRecord(
        run_id="calibration",
        smiles=smiles,
        target_bond=target_bond,
        requested_output=None,
        condition_profile=unity_condition,
        bond_context=unity_bond,
        physics_audit=unity_physics,
        module0=unity_module0,
    )
    features = build_features(record)
    p_cal = _calibrator_predict_proba(features, model)
    meta = {
        "metrics": model.get("metrics") or {},
        "model_version": model.get("model_version"),
    }
    return p_cal, source, model.get("model_version"), meta


def _event_probability_from_k(
    k_s_inv: Optional[float],
    horizon_s: Optional[float],
    diffusion_cap_k_s_inv: Optional[float],
    protonation_factor: float = 1.0,
    solvent_penalty: float = 1.0,
) -> Dict[str, Optional[float]]:
    if k_s_inv is None or horizon_s is None:
        return {
            "k_eff_s_inv": None,
            "p_event": None,
            "protonation_factor": protonation_factor,
            "solvent_penalty": solvent_penalty,
        }
    try:
        rate = float(k_s_inv)
        horizon = float(horizon_s)
    except (TypeError, ValueError):
        return {
            "k_eff_s_inv": None,
            "p_event": None,
            "protonation_factor": protonation_factor,
        }
    if not math.isfinite(rate) or not math.isfinite(horizon) or horizon <= 0.0:
        return {
            "k_eff_s_inv": None,
            "p_event": None,
            "protonation_factor": protonation_factor,
            "solvent_penalty": solvent_penalty,
        }
    k_eff = max(0.0, rate)
    if isinstance(diffusion_cap_k_s_inv, (int, float)) and math.isfinite(
        diffusion_cap_k_s_inv
    ):
        k_eff = min(k_eff, max(0.0, float(diffusion_cap_k_s_inv)))
    k_eff *= max(0.0, float(protonation_factor))
    k_eff *= max(0.0, float(solvent_penalty))
    p_event = 1.0 - math.exp(-k_eff * horizon)
    return {
        "k_eff_s_inv": float(k_eff),
        "p_event": max(0.0, min(1.0, float(p_event))),
        "protonation_factor": float(protonation_factor),
        "solvent_penalty": float(solvent_penalty),
    }


def _detectability_probability(
    turnovers: float,
    n_required: float = 3.0,
    sharpness: float = 3.0,
) -> float:
    try:
        turns = max(0.0, float(turnovers))
        required = max(1e-6, float(n_required))
        slope = max(1e-6, float(sharpness))
    except (TypeError, ValueError):
        return 0.0
    log_turns = math.log10(turns + 1e-9)
    log_req = math.log10(required)
    z = slope * (log_turns - log_req)
    if z >= 50.0:
        return 1.0
    if z <= -50.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z))))


def _normalize_family_key(value: Optional[str]) -> str:
    return str(value or "unknown").strip().lower()


def _infer_track_for_kcat(
    *,
    route_name: str,
    bond_class: str,
    barrier_override: Optional[Dict[str, Any]],
) -> Optional[str]:
    if isinstance(barrier_override, dict):
        track = str(barrier_override.get("track") or "").strip().lower()
        if track:
            return track
    route_l = str(route_name or "").lower()
    bond_l = str(bond_class or "").lower()
    if "halide" in route_l or "dehalogenase" in route_l or "c_cl" in bond_l or "c_br" in bond_l:
        return "displacement_sn2"
    if any(tok in route_l for tok in ("p450", "radical", "non_heme", "monooxygenase", "oxidase")):
        return "radical_hat"
    if any(tok in route_l for tok in ("hydrolase", "amidase", "protease")):
        return "nucleophilic"
    return None


def _predict_kcat_brenda_anchored(
    *,
    route_name: str,
    bond_class: str,
    track: Optional[str],
    bde_kj_mol: Optional[float],
    substrate_smiles: Optional[str],
    temperature_K: float,
) -> Dict[str, Any]:
    family_payload = enzyme_family_prior(route_name) or {}
    family = _normalize_family_key(family_payload.get("family"))
    median_kcat = float(FAMILY_MEDIAN_KCAT_S_INV.get(family, FAMILY_MEDIAN_KCAT_S_INV["unknown"]))
    track_l = str(track or "").lower()
    bond_l = str(bond_class or "").lower()
    smiles = str(substrate_smiles or "")

    if track_l == "radical_hat" and isinstance(bde_kj_mol, (int, float)):
        ref_bde = float(FAMILY_REFERENCE_BDE_KJ_MOL.get(family, 400.0))
        reference_bde_class = "family_default"
        if family == "non_heme_iron_oxygenase":
            ref_bde, reference_bde_class = _nhi_reference_bde_from_inputs(
                substrate_smiles=substrate_smiles,
                bond_class=bond_class,
            )
        alpha = float(EP_ALPHA_BY_FAMILY.get(family, 0.45))
        if any(tok in bond_l for tok in ("benzylic", "allylic", "aryl", "vinyl")):
            alpha = min(alpha, 0.30)
        denom = max(1e-6, R_KJ_PER_MOL_K * float(temperature_K))
        delta_bde = float(bde_kj_mol) - ref_bde
        difficulty_factor = math.exp((-delta_bde * alpha) / denom)
        predicted = max(1e-12, median_kcat * difficulty_factor)
        if _route_debug_enabled():
            print(
                "NHI ref_bde used:"
                f" family={family} route={route_name} class={reference_bde_class}"
                f" ref_bde={ref_bde} substrate_bde={float(bde_kj_mol)}"
                f" delta_bde={float(delta_bde)} predicted_kcat={float(predicted)}"
            )
        return {
            "predicted_kcat_s_inv": float(predicted),
            "source": "brenda_anchored_radical_hat",
            "family": family,
            "track": "radical_hat",
            "components": {
                "family_median_kcat_s_inv": median_kcat,
                "reference_bde_kj_mol": ref_bde,
                "reference_bde_class": reference_bde_class,
                "substrate_bde_kj_mol": float(bde_kj_mol),
                "delta_bde_kj_mol": float(delta_bde),
                "ep_alpha": alpha,
                "difficulty_factor": float(difficulty_factor),
            },
        }

    if track_l == "displacement_sn2" or family == "haloalkane_dehalogenase":
        modifier = 1.0
        notes: List[str] = []
        if "c_br" in bond_l or "br" in bond_l:
            modifier *= 3.0
            notes.append("bromide_bonus_x3")
        elif "c_f" in bond_l:
            modifier *= 0.001
            notes.append("fluoride_penalty_x0.001")
        elif "secondary" in bond_l:
            modifier *= 0.1
            notes.append("secondary_center_penalty_x0.1")
        predicted = max(1e-12, median_kcat * modifier)
        return {
            "predicted_kcat_s_inv": float(predicted),
            "source": "brenda_anchored_sn2",
            "family": family,
            "track": "displacement_sn2",
            "components": {
                "family_median_kcat_s_inv": median_kcat,
                "modifier": float(modifier),
                "notes": notes,
            },
        }

    if track_l in {"nucleophilic", ""} and family in {"serine_hydrolase", "metallo_esterase"}:
        modifier = 1.0
        notes: List[str] = []
        if "amide" in bond_l:
            modifier *= 0.01
            notes.append("amide_penalty_x0.01")
        elif "thioester" in bond_l:
            modifier *= 5.0
            notes.append("thioester_bonus_x5")
        smiles_l = smiles.lower()
        if "oc1" in smiles_l and "ester" in bond_l:
            modifier *= 2.0
            notes.append("aryl_ester_bonus_x2")
        if "f" in smiles and "oc" in smiles_l:
            modifier *= 0.5
            notes.append("fluorinated_ester_penalty_x0.5")
        predicted = max(1e-12, median_kcat * modifier)
        return {
            "predicted_kcat_s_inv": float(predicted),
            "source": "brenda_anchored_hydrolase",
            "family": family,
            "track": "nucleophilic",
            "components": {
                "family_median_kcat_s_inv": median_kcat,
                "modifier": float(modifier),
                "notes": notes,
            },
        }

    return {
        "predicted_kcat_s_inv": float(median_kcat),
        "source": "brenda_family_median_fallback",
        "family": family,
        "track": track_l or None,
        "components": {"family_median_kcat_s_inv": median_kcat},
    }


def _apply_brenda_kcat_to_route_prior(
    route_prior: Dict[str, Any],
    *,
    route_name: str,
    bond_class: str,
    barrier_override: Optional[Dict[str, Any]],
    substrate_smiles: Optional[str],
    temperature_K: float,
    horizon_s: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    updated = dict(route_prior or {})
    track = _infer_track_for_kcat(
        route_name=route_name,
        bond_class=bond_class,
        barrier_override=barrier_override,
    )
    bde_val = None
    if isinstance(barrier_override, dict) and isinstance(barrier_override.get("bde_kj_mol"), (int, float)):
        bde_val = float(barrier_override.get("bde_kj_mol"))

    model_bond_class = (
        str(barrier_override.get("bond_class"))
        if isinstance(barrier_override, dict) and isinstance(barrier_override.get("bond_class"), str)
        else str(bond_class or "")
    )
    kcat_payload = _predict_kcat_brenda_anchored(
        route_name=route_name,
        bond_class=model_bond_class,
        track=track,
        bde_kj_mol=bde_val,
        substrate_smiles=substrate_smiles,
        temperature_K=temperature_K,
    )
    k_model = float(kcat_payload.get("predicted_kcat_s_inv") or 0.0)
    diff_cap = updated.get("diffusion_cap_s_inv")
    if isinstance(diff_cap, (int, float)) and float(diff_cap) > 0.0:
        k_eff = min(k_model, float(diff_cap))
    else:
        k_eff = max(0.0, k_model)

    horizon = max(0.0, float(horizon_s))
    p_convert = 0.0
    if horizon > 0.0 and k_eff > 0.0:
        p_convert = 1.0 - math.exp(-k_eff * horizon)
        p_convert = max(0.0, min(1.0, float(p_convert)))
    turnovers = max(0.0, k_eff * horizon)
    n_required = float(updated.get("detectability_n_required") or 3.0)
    p_raw = _detectability_probability(turnovers, n_required=n_required)
    p_final = 0.5 + 0.6 * (float(p_raw) - 0.5)
    p_final = max(0.05, min(0.85, float(p_final)))

    energy_ledger = dict(updated.get("energy_ledger") or {})
    energy_ledger["k_model_s_inv"] = float(k_model)
    energy_ledger["k_eff_s_inv"] = float(k_eff)
    energy_ledger["p_success_horizon"] = float(p_convert)
    energy_ledger["horizon_s"] = float(horizon)
    energy_ledger["k_model_source"] = str(kcat_payload.get("source"))

    updated["k_effective_s_inv"] = float(k_eff)
    updated["p_convert_horizon"] = float(p_convert)
    updated["turnovers"] = float(turnovers)
    updated["p_raw"] = round(float(p_raw), 6)
    updated["p_final"] = round(float(p_final), 6)
    updated["energy_ledger"] = energy_ledger
    return updated, kcat_payload


def _physics_route_priors(
    route_candidates: List[str],
    smiles: str,
    target_bond: str,
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
    condition_profile: ConditionProfile,
    chem_context: Optional[Dict[str, Any]] = None,
    horizon_s: float = 3600.0,
    known_scaffold: bool = False,
    metals_allowed: Optional[bool] = None,
    nucleophile_geometry: Optional[str] = None,
    sre_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    bond_class = _bond_class_for_physics(bond_context)
    temp_k = _temperature_k_from_profile(condition_profile)
    pH = condition_profile.pH
    solvent_info = _solvent_penalty_from_profile(condition_profile)
    solvent_pen = float(solvent_info.get("penalty") or 0.7)
    context_penalty, context_notes = _context_uncertainty_penalty(
        condition_profile, known_scaffold
    )
    sre_payload = sre_payload or {}
    route_bias = sre_payload.get("route_bias") if isinstance(sre_payload, dict) else {}
    mechanism_eligibility = (
        sre_payload.get("mechanism_eligibility") if isinstance(sre_payload, dict) else {}
    )
    if not isinstance(route_bias, dict):
        route_bias = {}
    if not isinstance(mechanism_eligibility, dict):
        mechanism_eligibility = {}
    cpt_scores = (
        sre_payload.get("cpt_scores")
        if isinstance(sre_payload.get("cpt_scores"), dict)
        else sre_payload.get("cpt")
        if isinstance(sre_payload.get("cpt"), dict)
        else {}
    )
    radical_hat_priors = {}
    if str((cpt_scores or {}).get("track") or "").lower() == "radical_hat":
        radical_hat_priors = _compute_hat_informed_priors(
            routes=[route_name for route_name in route_candidates if route_name],
            cpt_scores=cpt_scores,
            mechanism_eligibility=mechanism_eligibility,
            bond_context=bond_context,
            temperature_K=temp_k,
    )
    prefer = set(route_bias.get("prefer") or [])
    discourage = set(route_bias.get("discourage") or [])
    bias_strength = float(route_bias.get("strength") or 0.0)
    mol = Chem.MolFromSmiles(smiles) if Chem is not None and smiles else None
    priors: Dict[str, Dict[str, Any]] = {}
    for route_name in route_candidates:
        if not route_name:
            continue
        route_prior = compute_route_prior(
            route_name=route_name,
            bond_class=bond_class,
            temperature_K=temp_k,
            horizon_s=horizon_s,
            pH=pH,
            ionic_strength=condition_profile.ionic_strength,
            chem_context=chem_context,
        )
        barrier = get_baseline_barrier(bond_class, route_name)
        barrier_kj = adjust_barrier_for_temperature(barrier.deltaG_dagger_kJ, temp_k)
        barrier_override = _sre_barrier_override(sre_payload)
        baseline_source = barrier.source
        baseline_bond_class = barrier.bond_class
        baseline_mechanism_family = barrier.mechanism_family
        if barrier_override is not None:
            barrier_kj = float(barrier_override["barrier_kj_mol"])
            track = str(barrier_override.get("track") or "")
            baseline_source = (
                "module_minus1_radical_cpt"
                if track == "radical_hat"
                else "module_minus1_displacement_cpt"
            )
            baseline_bond_class = bond_class
            baseline_mechanism_family = str(
                barrier_override.get("mechanism") or route_name
            )
            route_prior = _apply_barrier_override_to_route_prior(
                route_prior, barrier_kj, temp_k, horizon_s
            )
        route_prior, kcat_payload = _apply_brenda_kcat_to_route_prior(
            route_prior,
            route_name=route_name,
            bond_class=bond_class,
            barrier_override=barrier_override,
            substrate_smiles=smiles,
            temperature_K=temp_k,
            horizon_s=horizon_s,
        )
        hat_prior_payload = radical_hat_priors.get(route_name) if radical_hat_priors else None
        hat_prior_value = (
            float(hat_prior_payload.get("prior_feasibility"))
            if isinstance(hat_prior_payload, dict)
            and isinstance(hat_prior_payload.get("prior_feasibility"), (int, float))
            else None
        )
        protonation_factor, protonation_notes, protonation_residue, protonation_uncertain = (
            _protonation_factor_for_route(route_name, pH)
        )
        mismatch = mechanism_mismatch_penalty(route_name, nucleophile_geometry)
        cofactor = cofactor_compatibility_penalty(route_name, metals_allowed)
        phys = kinetics_from_context(bond_class, route_name, temp_k, pH)
        base_prior_raw = (
            float(hat_prior_value)
            if hat_prior_value is not None
            else float(route_prior.get("p_final") or 0.0)
        )
        compatibility_multiplier = _route_compatibility_weight(
            route_name, mol=mol, bond_context=bond_context
        )
        route_bias_multiplier = 1.0
        if route_name in prefer:
            route_bias_multiplier += bias_strength
        if route_name in discourage:
            route_bias_multiplier -= bias_strength
        route_bias_multiplier = max(0.1, route_bias_multiplier)
        eligibility = _route_eligibility_status(route_name, mechanism_eligibility)
        if isinstance(eligibility, str):
            if eligibility == "REJECTED":
                route_bias_multiplier = min(route_bias_multiplier, 0.2)
            elif eligibility == "REQUIRE_QUORUM":
                route_bias_multiplier = min(route_bias_multiplier, 0.7)
        prior_scale = compatibility_multiplier * route_bias_multiplier
        base_prior = max(0.01, min(0.99, base_prior_raw * prior_scale))
        if _route_debug_enabled():
            route_score = None
            if isinstance(barrier_kj, (int, float)):
                route_score = float(barrier_kj) * float(get_route_prior(mol, route_name))
            print(
                "Route debug:"
                f" route={route_name}"
                f" barrier={round(float(barrier_kj), 3)}"
                f" prior={round(float(get_route_prior(mol, route_name)), 3) if mol is not None else 'n/a'}"
                f" score={round(float(route_score), 3) if route_score is not None else 'n/a'}"
                f" compatibility_multiplier={round(float(compatibility_multiplier), 3)}"
                f" bias_multiplier={round(float(route_bias_multiplier), 3)}"
                f" prior_feasibility={round(float(base_prior), 4)}"
            )
        p_event = route_prior.get("p_raw")
        if base_prior is None:
            prior_any = None
            prior_target = None
        else:
            prior_any = max(0.01, min(0.99, float(base_prior) * context_penalty))
            cofactor_pen = float(cofactor.get("penalty") or 0.0)
            mismatch_pen = float(mismatch.get("penalty") or 0.0)
            prior_any = max(0.01, min(0.99, prior_any * (1.0 - cofactor_pen)))
            prior_target = max(0.01, min(0.99, prior_any * 0.85 * (1.0 - mismatch_pen)))
        prior_any_heuristic = prior_any
        prior_any_calibrated = None
        prior_any_final = prior_any
        calibration_source = None
        calibration_version = None
        calibration_meta: Dict[str, Any] = {}
        if prior_any is not None:
            phys_payload = dict(phys)
            phys_payload["delta_g_act_kj_mol"] = route_prior.get("deltaG_dagger_kJ_per_mol")
            phys_payload["k_s_inv"] = route_prior.get("eyring_k_s_inv")
            phys_payload["k_eff_s_inv"] = route_prior.get("k_effective_s_inv")
            phys_payload["temperature_K"] = temp_k
            phys_payload["horizon_s"] = horizon_s
            p_cal, calibration_source, calibration_version, calibration_meta = (
                _calibrated_any_activity_probability(
                smiles=smiles,
                target_bond=target_bond,
                bond_context=bond_context,
                structure_summary=structure_summary,
                physics_payload=phys_payload,
                condition_profile=condition_profile,
                route_name=route_name,
            )
            )
            if isinstance(p_cal, (int, float)):
                prior_any_calibrated = float(p_cal)
                prior_any_final = 0.15 * float(prior_any) + 0.85 * float(prior_any_calibrated)
                if base_prior is not None:
                    prior_any_final = min(float(prior_any_final), float(base_prior))
                prior_any_final = max(0.01, min(0.99, float(prior_any_final)))
                prior_target = max(
                    0.01,
                    min(0.99, float(prior_any_final) * 0.85 * (1.0 - mismatch_pen)),
                )
        priors[route_name] = {
            "bond_class": bond_class,
            "baseline_barrier_source": baseline_source,
            "baseline_barrier_kj_mol": round(float(barrier_kj), 2),
            "baseline_barrier_bond_class": baseline_bond_class,
            "baseline_barrier_mechanism_family": baseline_mechanism_family,
            "mechanism_family": route_name,
            "temperature_K": round(temp_k, 2),
            "pH": pH,
            "delta_g_act_kj_mol": route_prior.get("deltaG_dagger_kJ_per_mol"),
            "delta_g_act_kcal_mol": phys.get("delta_g_act_kcal_mol"),
            "chem_context": route_prior.get("chem_context"),
            "deltaG_components_kJ_per_mol": route_prior.get("deltaG_components_kJ_per_mol"),
            "k_s_inv": route_prior.get("eyring_k_s_inv"),
            "k_eyring_s_inv": route_prior.get("eyring_k_s_inv"),
            "k_eff_s_inv": route_prior.get("k_effective_s_inv"),
            "k_model_source": kcat_payload.get("source"),
            "k_model_family": kcat_payload.get("family"),
            "k_model_track": kcat_payload.get("track"),
            "k_model_components": kcat_payload.get("components"),
            "protonation_factor": route_prior.get("f_prot"),
            "solvent_penalty": solvent_pen,
            "half_life_s": phys.get("half_life_s"),
            "diffusion_cap_k_s_inv": route_prior.get("diffusion_cap_s_inv"),
            "notes": phys.get("notes", []),
            "expected_conversion": p_event,
            "detection_threshold": DETECTION_THRESHOLD_CONVERSION,
            "noise_floor": DETECTION_NOISE_FLOOR,
            "p_event_hour": p_event,
            "prior_feasibility_raw": round(float(hat_prior_value), 4)
            if hat_prior_value is not None
            else route_prior.get("p_raw"),
            "prior_feasibility": round(float(base_prior), 4),
            "prior_note": route_prior.get("prior_note"),
            "route_prior_any_activity": round(float(prior_any_final), 4)
            if prior_any_final is not None
            else None,
            "route_prior_target_specific": round(float(prior_target), 4)
            if prior_target is not None
            else None,
            "context_uncertainty_penalty": round(float(context_penalty), 3),
            "context_missing": context_notes,
            "mechanism_mismatch_penalty": round(float(mismatch.get("penalty") or 0.0), 3),
            "mechanism_mismatch_reason": mismatch.get("reason"),
            "cofactor_penalty": round(float(cofactor.get("penalty") or 0.0), 3),
            "cofactor_penalty_reason": cofactor.get("reason"),
            "protonation_residue": protonation_residue,
            "protonation_notes": protonation_notes,
            "protonation_uncertain": protonation_uncertain,
            "solvent_penalty_note": solvent_info.get("note"),
            "solvent_unknown": solvent_info.get("solvent_unknown", True),
            "prior_before_damping": round(float(prior_target), 4)
            if prior_target is not None
            else None,
            "prior_success_probability": round(float(prior_target), 4)
            if prior_target is not None
            else None,
            "prior_any_activity_heuristic": round(float(prior_any_heuristic), 4)
            if prior_any_heuristic is not None
            else None,
            "prior_any_activity_calibrated": round(float(prior_any_calibrated), 4)
            if prior_any_calibrated is not None
            else None,
            "prior_any_activity_final": round(float(prior_any_final), 4)
            if prior_any_final is not None
            else None,
            "calibration_source": calibration_source,
            "calibration_version": calibration_version,
            "calibration_sources": (calibration_meta.get("metrics") or {}).get("sources"),
            "calibration_samples": (calibration_meta.get("metrics") or {}).get("n_samples"),
            "mechanism_weight": 1.0,
            "prior_available": prior_target is not None,
            "route_bias_multiplier": round(float(route_bias_multiplier), 3),
            "substrate_compatibility_multiplier": round(
                float(
                    hat_prior_payload.get("substrate_compatibility_multiplier")
                    if isinstance(hat_prior_payload, dict)
                    and isinstance(
                        hat_prior_payload.get("substrate_compatibility_multiplier"),
                        (int, float),
                    )
                    else compatibility_multiplier
                ),
                3,
            ),
            "mechanism_eligibility": eligibility,
            "module_minus1_track": barrier_override.get("track") if barrier_override else None,
            "module_minus1_bde_kj_mol": barrier_override.get("bde_kj_mol") if barrier_override else None,
            "hat_barrier_kj_mol": hat_prior_payload.get("hat_barrier_kj_mol")
            if isinstance(hat_prior_payload, dict)
            else None,
            "prior_source": hat_prior_payload.get("prior_source")
            if isinstance(hat_prior_payload, dict)
            else "heuristic_flat",
        }
    return priors


def _blend_route_confidence(
    heuristic_confidence: float,
    physics_confidence: Optional[float],
    physics_weight: float = 0.4,
) -> float:
    if physics_confidence is None:
        return float(heuristic_confidence)
    weight = max(0.0, min(1.0, float(physics_weight)))
    blended = (1.0 - weight) * float(heuristic_confidence) + weight * float(physics_confidence)
    return max(0.0, min(1.0, blended))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _clamp_unit_interval(value: Any, default: float = 0.0) -> float:
    converted = _safe_float(value, default)
    if converted is None:
        converted = default
    return max(0.0, min(1.0, float(converted)))


def _confidence_label_from_score(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _route_eligibility_score(status: Optional[str]) -> float:
    normalized = str(status or "").strip().upper()
    if normalized == "REJECTED":
        return 0.0
    if normalized == "REQUIRE_QUORUM":
        return 0.45
    if normalized == "SUPPORTED":
        return 1.0
    return 0.75


def _normalize_route_posteriors(
    route_candidates: List[str],
    route_posteriors: Optional[List[Dict[str, Any]]],
    physics_route_priors: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return route posterior rows with a normalized `posterior` field.

    The router and heuristic fallback both emit route-like scores, but the fallback
    path historically produced per-route posterior-like values that did not sum to 1.
    This helper preserves the existing `p_raw` / `p_cal` fields while normalizing the
    externally consumed `posterior` field into a coherent pseudo-probability.
    """
    physics_route_priors = physics_route_priors or {}
    raw_entries = route_posteriors or []
    keyed_entries: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _register(route_name: Optional[str], entry: Optional[Dict[str, Any]] = None) -> None:
        if not route_name:
            return
        if route_name not in keyed_entries:
            keyed_entries[route_name] = dict(entry or {})
            order.append(route_name)
        elif entry:
            keyed_entries[route_name].update(entry)

    for route_name in route_candidates:
        _register(route_name)
    for entry in raw_entries:
        _register(entry.get("route"), entry)

    weights: Dict[str, float] = {}
    total_weight = 0.0
    for route_name in order:
        entry = keyed_entries[route_name]
        phys = physics_route_priors.get(route_name, {})
        raw_score = None
        for candidate_value in (
            entry.get("p_cal"),
            entry.get("posterior"),
            entry.get("p_raw"),
            entry.get("ranking_score"),
            phys.get("prior_success_probability"),
            phys.get("route_prior_target_specific"),
            phys.get("prior_feasibility"),
        ):
            converted = _safe_float(candidate_value)
            if converted is not None:
                raw_score = converted
                break
        if raw_score is None:
            raw_score = 0.0
        raw_score = max(0.0, float(raw_score))
        weights[route_name] = raw_score
        total_weight += raw_score

    if total_weight <= 0.0:
        total_weight = float(len(order) or 1.0)
        for route_name in order:
            weights[route_name] = 1.0

    normalized_entries: List[Dict[str, Any]] = []
    for route_name in order:
        entry = dict(keyed_entries[route_name])
        raw_score = weights[route_name]
        posterior = raw_score / total_weight
        entry.setdefault("route", route_name)
        entry.setdefault("p_raw", round(raw_score, 6))
        entry.setdefault("p_cal", round(raw_score, 6))
        entry["posterior_raw"] = round(raw_score, 6)
        entry["posterior"] = round(float(posterior), 6)
        normalized_entries.append(entry)

    normalized_entries.sort(
        key=lambda item: (
            _safe_float(item.get("posterior"), 0.0) or 0.0,
            _safe_float(item.get("posterior_raw"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    return normalized_entries


def _build_route_evidence_decomposition(
    route_candidates: List[str],
    route_posteriors: Optional[List[Dict[str, Any]]],
    physics_route_priors: Optional[Dict[str, Dict[str, Any]]],
    chosen_route: Optional[str] = None,
    fallback_used: bool = False,
) -> List[Dict[str, Any]]:
    """Build a route-level evidence table with decomposed scoring terms."""
    normalized_posteriors = _normalize_route_posteriors(
        route_candidates=route_candidates,
        route_posteriors=route_posteriors,
        physics_route_priors=physics_route_priors,
    )
    physics_route_priors = physics_route_priors or {}
    evidence_rows: List[Dict[str, Any]] = []

    for entry in normalized_posteriors:
        route_name = str(entry.get("route") or "")
        phys = dict(physics_route_priors.get(route_name) or {})
        chemistry_score = _clamp_unit_interval(
            phys.get("prior_feasibility")
            if phys.get("prior_feasibility") is not None
            else phys.get("prior_feasibility_raw"),
            default=0.0,
        )
        physics_score = _clamp_unit_interval(
            phys.get("prior_success_probability")
            if phys.get("prior_success_probability") is not None
            else phys.get("route_prior_target_specific")
            if phys.get("route_prior_target_specific") is not None
            else phys.get("expected_conversion"),
            default=0.0,
        )
        cofactor_score = 1.0 - _clamp_unit_interval(phys.get("cofactor_penalty"), default=0.0)
        protonation_factor = _clamp_unit_interval(phys.get("protonation_factor"), default=0.5)
        biology_score = max(0.0, min(1.0, 0.5 * cofactor_score + 0.5 * protonation_factor))
        compatibility_score = max(
            0.0,
            min(
                1.0,
                0.5 * _safe_float(phys.get("substrate_compatibility_multiplier"), 1.0)
                + 0.25 * _safe_float(phys.get("route_bias_multiplier"), 1.0)
                + 0.25 * cofactor_score,
            ),
        )
        mismatch_penalty = _clamp_unit_interval(phys.get("mechanism_mismatch_penalty"), default=0.0)
        eligibility_status = phys.get("mechanism_eligibility")
        eligibility_score = _route_eligibility_score(eligibility_status)
        posterior = _clamp_unit_interval(entry.get("posterior"), default=0.0)
        ranking_score = max(
            0.0,
            min(
                1.0,
                (
                    0.40 * posterior
                    + 0.20 * chemistry_score
                    + 0.15 * physics_score
                    + 0.10 * biology_score
                    + 0.10 * compatibility_score
                    + 0.05 * cofactor_score
                )
                * (0.60 + 0.40 * eligibility_score)
                - 0.20 * mismatch_penalty,
            ),
        )
        evidence_notes: List[str] = []
        evidence_conflicts: List[str] = []
        if fallback_used:
            evidence_notes.append("Route score includes fallback heuristic weighting.")
            evidence_conflicts.append("fallback_override_used")
        if mismatch_penalty >= 0.2:
            evidence_notes.append("Mechanism mismatch penalty reduced route plausibility.")
            evidence_conflicts.append("compatibility_penalty_high")
        if _clamp_unit_interval(phys.get("cofactor_penalty"), default=0.0) >= 0.2:
            evidence_notes.append("Cofactor compatibility penalty reduced route plausibility.")
            evidence_conflicts.append("cofactor_penalty_high")
        if str(eligibility_status or "").upper() == "REJECTED":
            evidence_notes.append("Module -1 mechanism eligibility rejected this route.")
            evidence_conflicts.append("eligibility_rejected")
        if physics_score < 0.1 and chemistry_score >= 0.2:
            evidence_conflicts.append("chemistry_vs_physics")
        if phys.get("context_missing"):
            evidence_notes.append("Context evidence incomplete; route priors were damped.")
        evidence_rows.append(
            {
                "route_id": route_name,
                "route_family": route_name,
                "selected": route_name == chosen_route,
                "ranking_score": round(float(ranking_score), 6),
                "posterior": round(float(posterior), 6),
                "posterior_raw": _safe_float(entry.get("posterior_raw"), 0.0),
                "chemistry_score": round(float(chemistry_score), 6),
                "physics_score": round(float(physics_score), 6),
                "biology_score": round(float(biology_score), 6),
                "compatibility_score": round(float(compatibility_score), 6),
                "cofactor_score": round(float(cofactor_score), 6),
                "eligibility_score": round(float(eligibility_score), 6),
                "mismatch_penalty": round(float(mismatch_penalty), 6),
                "final_score_pre_calibration": round(float(ranking_score), 6),
                "final_score_post_calibration": round(float(ranking_score), 6),
                "evidence_notes": list(dict.fromkeys(evidence_notes)),
                "evidence_conflicts": list(dict.fromkeys(evidence_conflicts)),
                "raw_router_entry": entry,
                "physics_prior": phys,
            }
        )

    evidence_rows.sort(
        key=lambda item: (
            _safe_float(item.get("final_score_post_calibration"), 0.0) or 0.0,
            _safe_float(item.get("posterior"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    return evidence_rows


def _compute_route_ambiguity_metrics(route_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not route_evidence:
        return {
            "route_gap": None,
            "posterior_gap": None,
            "ambiguity_flag": True,
            "top_routes": [],
            "reasons": ["no_routes_scored"],
        }
    top_routes = route_evidence[: min(3, len(route_evidence))]
    top_1 = top_routes[0]
    top_2 = top_routes[1] if len(top_routes) > 1 else None
    route_gap = None
    posterior_gap = None
    reasons: List[str] = []
    if top_2:
        route_gap = max(
            0.0,
            (_safe_float(top_1.get("final_score_post_calibration"), 0.0) or 0.0)
            - (_safe_float(top_2.get("final_score_post_calibration"), 0.0) or 0.0),
        )
        posterior_gap = max(
            0.0,
            (_safe_float(top_1.get("posterior"), 0.0) or 0.0)
            - (_safe_float(top_2.get("posterior"), 0.0) or 0.0),
        )
    ambiguity_flag = False
    if top_2 is None:
        ambiguity_flag = False
    else:
        if (route_gap or 0.0) < 0.10:
            ambiguity_flag = True
            reasons.append("top_routes_close_by_score")
        if (posterior_gap or 0.0) < 0.08:
            ambiguity_flag = True
            reasons.append("top_routes_close_by_posterior")
    return {
        "route_gap": round(float(route_gap), 6) if route_gap is not None else None,
        "posterior_gap": round(float(posterior_gap), 6) if posterior_gap is not None else None,
        "ambiguity_flag": bool(ambiguity_flag),
        "top_routes": [
            {
                "route_id": row.get("route_id"),
                "route_family": row.get("route_family"),
                "score": row.get("final_score_post_calibration"),
                "posterior": row.get("posterior"),
            }
            for row in top_routes
        ],
        "reasons": reasons,
    }


def _compute_route_confidence_details(
    route_evidence: List[Dict[str, Any]],
    chosen_route: Optional[str],
    *,
    support: Optional[float],
    evidence_strength: Optional[float],
    fallback_used: bool = False,
) -> Dict[str, Any]:
    if not route_evidence:
        return {
            "confidence": 0.1,
            "confidence_label": "Low",
            "audit_score": 0.0,
            "ambiguity_score": 1.0,
            "evidence_agreement": 0.0,
            "selected_route": chosen_route,
            "selected_posterior": None,
            "selected_ranking_score": None,
            "ambiguity": _compute_route_ambiguity_metrics(route_evidence),
            "calibration": {
                "mode": "fallback",
                "top_gap": None,
                "evidence_agreement": 0.0,
                "confidence_adjustment": 0.0,
                "reasons": ["no_route_evidence"],
            },
            "evidence_conflicts": ["no_route_evidence"],
        }

    selected = next(
        (row for row in route_evidence if row.get("route_id") == chosen_route),
        route_evidence[0],
    )
    ambiguity = _compute_route_ambiguity_metrics(route_evidence)
    support_score = _clamp_unit_interval(support, default=0.0)
    evidence_strength_score = _clamp_unit_interval(evidence_strength, default=0.0)
    chemistry_winner = max(route_evidence, key=lambda row: _safe_float(row.get("chemistry_score"), 0.0) or 0.0)
    physics_winner = max(route_evidence, key=lambda row: _safe_float(row.get("physics_score"), 0.0) or 0.0)
    biology_winner = max(route_evidence, key=lambda row: _safe_float(row.get("biology_score"), 0.0) or 0.0)
    agreement_votes = [
        selected.get("route_id") == chemistry_winner.get("route_id"),
        selected.get("route_id") == physics_winner.get("route_id"),
        selected.get("route_id") == biology_winner.get("route_id"),
    ]
    evidence_agreement = sum(1 for vote in agreement_votes if vote) / float(len(agreement_votes))

    conflicts: List[str] = list(selected.get("evidence_conflicts") or [])
    if chemistry_winner.get("route_id") != physics_winner.get("route_id"):
        conflicts.append("chemistry_vs_physics")
    if selected.get("route_id") != chemistry_winner.get("route_id"):
        conflicts.append("selected_route_differs_from_chemistry")
    if selected.get("route_id") != physics_winner.get("route_id"):
        conflicts.append("selected_route_differs_from_physics")
    if fallback_used:
        conflicts.append("fallback_override_used")
    conflicts = list(dict.fromkeys(conflicts))

    missing_evidence = 0.0
    for field_name in ("chemistry_score", "physics_score", "biology_score"):
        if selected.get(field_name) is None:
            missing_evidence += 1.0 / 3.0

    base_confidence = (
        0.35 * _clamp_unit_interval(selected.get("posterior"), default=0.0)
        + 0.25 * _clamp_unit_interval(selected.get("final_score_post_calibration"), default=0.0)
        + 0.15 * evidence_agreement
        + 0.125 * support_score
        + 0.125 * evidence_strength_score
    )
    penalty = 0.0
    if fallback_used:
        penalty += 0.12
    if ambiguity.get("ambiguity_flag"):
        penalty += 0.15
    penalty += 0.12 * _clamp_unit_interval(selected.get("mismatch_penalty"), default=0.0)
    penalty += 0.06 * min(3, len(conflicts))
    penalty += 0.10 * missing_evidence

    confidence = max(0.02, min(0.98, base_confidence - penalty))
    audit_score = max(
        0.0,
        min(
            1.0,
            0.50 * evidence_agreement
            + 0.25 * (1.0 - _clamp_unit_interval(selected.get("mismatch_penalty"), default=0.0))
            + 0.25 * (1.0 - min(1.0, len(conflicts) / 4.0)),
        ),
    )
    ambiguity_score = 1.0 - min(1.0, (_safe_float(ambiguity.get("route_gap"), 0.0) or 0.0) / 0.20)
    calibration_mode = "moderate_evidence"
    calibration_reasons: List[str] = []
    if fallback_used and support_score <= 0.05:
        calibration_mode = "fallback"
        calibration_reasons.append("selected route came from fallback logic")
    elif ambiguity.get("ambiguity_flag"):
        calibration_mode = "ambiguous"
        calibration_reasons.extend(ambiguity.get("reasons") or [])
    elif conflicts:
        calibration_mode = "conflicted"
        calibration_reasons.extend(conflicts)
    elif evidence_agreement >= 2.0 / 3.0 and support_score >= 0.4:
        calibration_mode = "strong_evidence"
        calibration_reasons.append("chemistry, physics, and biology agree on selected route")
    else:
        calibration_reasons.append("mixed evidence; selection preserved with conservative confidence")

    return {
        "confidence": round(float(confidence), 6),
        "confidence_label": _confidence_label_from_score(confidence),
        "audit_score": round(float(audit_score), 6),
        "ambiguity_score": round(float(ambiguity_score), 6),
        "evidence_agreement": round(float(evidence_agreement), 6),
        "selected_route": selected.get("route_id"),
        "selected_posterior": selected.get("posterior"),
        "selected_ranking_score": selected.get("final_score_post_calibration"),
        "ambiguity": ambiguity,
        "calibration": {
            "mode": calibration_mode,
            "top_gap": ambiguity.get("route_gap"),
            "posterior_gap": ambiguity.get("posterior_gap"),
            "evidence_agreement": round(float(evidence_agreement), 6),
            "confidence_adjustment": round(float(-penalty), 6),
            "reasons": list(dict.fromkeys(calibration_reasons)),
        },
        "evidence_conflicts": conflicts,
    }


def _build_route_debug_report(
    route_evidence: List[Dict[str, Any]],
    confidence_details: Dict[str, Any],
    *,
    support: Optional[float],
    evidence_strength: Optional[float],
    fallback_used: bool,
) -> Dict[str, Any]:
    ambiguity = confidence_details.get("ambiguity") or {}
    return {
        "evaluated_routes": route_evidence,
        "top_routes": ambiguity.get("top_routes") or [],
        "route_gap": ambiguity.get("route_gap"),
        "posterior_gap": ambiguity.get("posterior_gap"),
        "ambiguity_flag": bool(ambiguity.get("ambiguity_flag")),
        "fallback_used": bool(fallback_used),
        "confidence_components": {
            "ranking_score": confidence_details.get("selected_ranking_score"),
            "posterior": confidence_details.get("selected_posterior"),
            "confidence": confidence_details.get("confidence"),
            "audit_score": confidence_details.get("audit_score"),
            "ambiguity_score": confidence_details.get("ambiguity_score"),
            "evidence_agreement": confidence_details.get("evidence_agreement"),
            "support": round(float(_safe_float(support, 0.0) or 0.0), 6),
            "evidence_strength": round(
                float(_safe_float(evidence_strength, 0.0) or 0.0),
                6,
            ),
        },
        "calibration": confidence_details.get("calibration") or {},
        "evidence_conflicts": confidence_details.get("evidence_conflicts") or [],
    }


def score_all_routes(
    route_candidates: List[str],
    physics_route_priors: Optional[Dict[str, Dict[str, Any]]] = None,
    router_prediction: Optional[Dict[str, Any]] = None,
    *,
    chosen_route: Optional[str] = None,
    support: Optional[float] = None,
    evidence_strength: Optional[float] = None,
    fallback_used: Optional[bool] = None,
) -> Dict[str, Any]:
    """Public helper for tests and manual-engine diagnostics.

    It mirrors the Module 0 route decomposition logic without requiring the full
    pipeline, making it easier to inspect route candidates on a single case.
    """
    router_prediction = router_prediction or {}
    route_evidence = _build_route_evidence_decomposition(
        route_candidates=route_candidates,
        route_posteriors=router_prediction.get("route_posteriors"),
        physics_route_priors=physics_route_priors,
        chosen_route=chosen_route or router_prediction.get("chosen_route"),
        fallback_used=bool(
            router_prediction.get("router_empty", False)
            if fallback_used is None
            else fallback_used
        ),
    )
    confidence_details = _compute_route_confidence_details(
        route_evidence,
        chosen_route=chosen_route or router_prediction.get("chosen_route"),
        support=support if support is not None else router_prediction.get("data_support"),
        evidence_strength=(
            evidence_strength
            if evidence_strength is not None
            else router_prediction.get("evidence_strength")
        ),
        fallback_used=bool(
            router_prediction.get("router_empty", False)
            if fallback_used is None
            else fallback_used
        ),
    )
    return _build_route_debug_report(
        route_evidence,
        confidence_details,
        support=support if support is not None else router_prediction.get("data_support"),
        evidence_strength=(
            evidence_strength
            if evidence_strength is not None
            else router_prediction.get("evidence_strength")
        ),
        fallback_used=bool(
            router_prediction.get("router_empty", False)
            if fallback_used is None
            else fallback_used
        ),
    )


def _physics_route_audit(
    smiles: str,
    target_bond: str,
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
    reaction_intent: Dict[str, Any],
    condition_profile: ConditionProfile,
    mechanism_family: Optional[str] = None,
    chem_context: Optional[Dict[str, Any]] = None,
    horizon_s: float = 3600.0,
    known_scaffold: bool = False,
    metals_allowed: Optional[bool] = None,
    nucleophile_geometry: Optional[str] = None,
    sre_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bond_class = _bond_class_for_physics(bond_context)
    mol = Chem.MolFromSmiles(smiles) if Chem is not None and smiles else None
    mechanism_hint = mechanism_family or (reaction_intent.get("intent_type") if reaction_intent else None)
    temp_k = _temperature_k_from_profile(condition_profile)
    mechanism_label = mechanism_hint or "unknown"
    pH = condition_profile.pH
    solvent_info = _solvent_penalty_from_profile(condition_profile)
    solvent_pen = float(solvent_info.get("penalty") or 0.7)
    barrier = get_baseline_barrier(bond_class, mechanism_label)
    barrier_kj = adjust_barrier_for_temperature(barrier.deltaG_dagger_kJ, temp_k)
    barrier_override = _sre_barrier_override(sre_payload)
    baseline_source = barrier.source
    baseline_bond_class = barrier.bond_class
    baseline_mechanism_family = barrier.mechanism_family
    if barrier_override is not None:
        barrier_kj = float(barrier_override["barrier_kj_mol"])
        track = str(barrier_override.get("track") or "")
        baseline_source = (
            "module_minus1_radical_cpt"
            if track == "radical_hat"
            else "module_minus1_displacement_cpt"
        )
        baseline_bond_class = bond_class
        baseline_mechanism_family = str(
            barrier_override.get("mechanism") or mechanism_label
        )
    protonation_factor, protonation_notes, protonation_residue, protonation_uncertain = (
        _protonation_factor_for_route(mechanism_label, pH)
    )
    mismatch = mechanism_mismatch_penalty(mechanism_label, nucleophile_geometry)
    cofactor = cofactor_compatibility_penalty(mechanism_label, metals_allowed)
    route_prior = compute_route_prior(
        route_name=mechanism_label,
        bond_class=bond_class,
        temperature_K=temp_k,
        horizon_s=horizon_s,
        pH=pH,
        ionic_strength=condition_profile.ionic_strength,
        chem_context=chem_context,
    )
    if barrier_override is not None:
        route_prior = _apply_barrier_override_to_route_prior(
            route_prior, barrier_kj, temp_k, horizon_s
        )
    route_prior, kcat_payload = _apply_brenda_kcat_to_route_prior(
        route_prior,
        route_name=mechanism_label,
        bond_class=bond_class,
        barrier_override=barrier_override,
        substrate_smiles=smiles,
        temperature_K=temp_k,
        horizon_s=horizon_s,
    )
    energy_ledger = route_prior.get("energy_ledger") or {}
    if barrier_override is not None:
        energy_ledger = dict(energy_ledger)
        energy_ledger["deltaG_dagger_kJ"] = float(barrier_kj)
        energy_ledger["eyring_k_s_inv"] = float(route_prior.get("eyring_k_s_inv") or 0.0)
        energy_ledger["k_eff_s_inv"] = float(route_prior.get("k_effective_s_inv") or 0.0)
        energy_ledger["p_success_horizon"] = float(route_prior.get("p_convert_horizon") or 0.0)
    phys = kinetics_from_context(bond_class, mechanism_label, temp_k, condition_profile.pH)
    k_s_inv = energy_ledger.get("eyring_k_s_inv")
    if not isinstance(k_s_inv, (int, float)):
        k_s_inv = route_prior.get("eyring_k_s_inv", phys.get("k_s_inv", 0.0))
    p_event = route_prior.get("p_raw")
    turnovers = route_prior.get("turnovers")
    detectability_threshold = route_prior.get("detectability_n_required")
    k_eff = energy_ledger.get("k_eff_s_inv")
    p_convert = None
    try:
        if isinstance(k_eff, (int, float)) and math.isfinite(float(k_eff)):
            p_convert = 1.0 - math.exp(-float(k_eff) * float(horizon_s))
            p_convert = max(0.0, min(1.0, float(p_convert)))
    except (TypeError, ValueError):
        p_convert = None
    context_penalty, context_notes = _context_uncertainty_penalty(
        condition_profile, known_scaffold
    )
    compatibility_multiplier = _route_compatibility_weight(
        mechanism_label, mol=mol, bond_context=bond_context
    )
    base_prior_raw = float(route_prior.get("p_final") or 0.0)
    base_prior = max(0.01, min(0.99, base_prior_raw * compatibility_multiplier))
    if base_prior is None:
        prior_any = None
        prior_target = None
    else:
        prior_any = max(0.01, min(0.99, float(base_prior) * context_penalty))
        cofactor_pen = float(cofactor.get("penalty") or 0.0)
        mismatch_pen = float(mismatch.get("penalty") or 0.0)
        prior_any = max(0.01, min(0.99, prior_any * (1.0 - cofactor_pen)))
        prior_target = max(0.01, min(0.99, prior_any * 0.85 * (1.0 - mismatch_pen)))
    prior_any_heuristic = prior_any
    prior_any_calibrated = None
    prior_any_final = prior_any
    calibration_source = None
    calibration_version = None
    calibration_meta: Dict[str, Any] = {}
    if prior_any is not None:
        phys_payload = dict(phys)
        phys_payload["delta_g_act_kj_mol"] = route_prior.get("deltaG_dagger_kJ_per_mol")
        phys_payload["k_s_inv"] = route_prior.get("eyring_k_s_inv")
        phys_payload["k_eff_s_inv"] = route_prior.get("k_effective_s_inv")
        phys_payload["temperature_K"] = temp_k
        phys_payload["horizon_s"] = horizon_s
        p_cal, calibration_source, calibration_version, calibration_meta = (
            _calibrated_any_activity_probability(
            smiles=smiles,
            target_bond=target_bond,
            bond_context=bond_context,
            structure_summary=structure_summary,
            physics_payload=phys_payload,
            condition_profile=condition_profile,
            route_name=mechanism_label,
        )
        )
        if isinstance(p_cal, (int, float)):
            prior_any_calibrated = float(p_cal)
            prior_any_final = 0.15 * float(prior_any) + 0.85 * float(prior_any_calibrated)
            if base_prior is not None:
                prior_any_final = min(float(prior_any_final), float(base_prior))
            prior_any_final = max(0.01, min(0.99, float(prior_any_final)))
            prior_target = max(
                0.01,
                min(0.99, float(prior_any_final) * 0.85 * (1.0 - mismatch_pen)),
            )
    return {
        "bond_class": bond_class,
        "baseline_barrier_source": baseline_source,
        "baseline_barrier_kj_mol": round(float(barrier_kj), 2),
        "baseline_barrier_bond_class": baseline_bond_class,
        "baseline_barrier_mechanism_family": baseline_mechanism_family,
        "deltaG_dagger_kJ_per_mol": round(
            float(energy_ledger.get("deltaG_dagger_kJ") or 0.0), 3
        ),
        "deltaG_dagger_kj_mol": round(
            float(energy_ledger.get("deltaG_dagger_kJ") or 0.0), 3
        ),
        "deltaG_dagger_kcal_per_mol": round(float(phys.get("delta_g_act_kcal_mol") or 0.0), 3),
        "temperature_K": round(float(temp_k), 2),
        "eyring_k_s_inv": float(k_s_inv),
        "eyring_k_display": format_rate(k_s_inv),
        "k_eyring_s_inv": float(k_s_inv),
        "k_eff_s_inv": energy_ledger.get("k_eff_s_inv"),
        "k_model_source": kcat_payload.get("source"),
        "k_model_family": kcat_payload.get("family"),
        "k_model_track": kcat_payload.get("track"),
        "k_model_components": kcat_payload.get("components"),
        "p_convert_horizon": energy_ledger.get("p_success_horizon"),
        "f_protonation": round(float(route_prior.get("f_prot") or protonation_factor), 3),
        "solvent_penalty": round(float(solvent_pen), 3),
        "k_diff_cap_s_inv": energy_ledger.get("k_diff_cap_s_inv")
        or route_prior.get("diffusion_cap_s_inv"),
        "chem_context": route_prior.get("chem_context"),
        "deltaG_components_kJ_per_mol": route_prior.get("deltaG_components_kJ_per_mol"),
        "protonation_factor": protonation_factor,
        "horizon_s": round(float(horizon_s), 1),
        "p_event_hour": p_event,
        "turnovers": turnovers,
        "detectability_threshold": detectability_threshold,
        "prior_feasibility_raw": route_prior.get("p_raw"),
        "prior_feasibility": round(float(base_prior), 4),
        "prior_note": route_prior.get("prior_note"),
        "route_prior_any_activity": round(float(prior_any_final), 4)
        if prior_any_final is not None
        else None,
        "route_prior_target_specific": round(float(prior_target), 4)
        if prior_target is not None
        else None,
        "prior_success_probability": round(float(prior_target), 4)
        if prior_target is not None
        else None,
        "mechanism_family": mechanism_label,
        "expected_conversion": p_event,
        "detection_threshold": DETECTION_THRESHOLD_CONVERSION,
        "noise_floor": DETECTION_NOISE_FLOOR,
        "context_uncertainty_penalty": round(float(context_penalty), 3),
        "context_missing": context_notes,
        "mechanism_mismatch_penalty": round(float(mismatch.get("penalty") or 0.0), 3),
        "module_minus1_track": barrier_override.get("track") if barrier_override else None,
        "module_minus1_bde_kj_mol": barrier_override.get("bde_kj_mol") if barrier_override else None,
        "mechanism_mismatch_reason": mismatch.get("reason"),
        "cofactor_penalty": round(float(cofactor.get("penalty") or 0.0), 3),
        "cofactor_penalty_reason": cofactor.get("reason"),
        "protonation_residue": protonation_residue,
        "protonation_notes": protonation_notes,
        "protonation_uncertain": protonation_uncertain,
        "solvent_penalty_note": solvent_info.get("note"),
        "solvent_unknown": solvent_info.get("solvent_unknown", True),
        "prior_before_damping": round(float(prior_target), 4)
        if prior_target is not None
        else None,
        "substrate_compatibility_multiplier": round(float(compatibility_multiplier), 3),
        "prior_note": "Converted Eyring rate to event probability; applied context uncertainty penalties (missing solvent/cofactors/known scaffold).",
        "prior_any_activity_heuristic": round(float(prior_any_heuristic), 4)
        if prior_any_heuristic is not None
        else None,
        "prior_any_activity_calibrated": round(float(prior_any_calibrated), 4)
        if prior_any_calibrated is not None
        else None,
        "prior_any_activity_final": round(float(prior_any_final), 4)
        if prior_any_final is not None
        else None,
        "calibration_source": calibration_source,
        "calibration_version": calibration_version,
        "calibration_sources": (calibration_meta.get("metrics") or {}).get("sources"),
        "calibration_samples": (calibration_meta.get("metrics") or {}).get("n_samples"),
        "calibration_note": (
            f"Evidence-calibrated prior used ({calibration_version}); "
            f"sources={ (calibration_meta.get('metrics') or {}).get('sources') }"
            if calibration_source
            else None
        ),
        "energy_ledger": energy_ledger,
    }


def _physics_prior(
    bond_context: Dict[str, Any],
    condition_profile: ConditionProfile,
) -> Dict[str, Any]:
    temperature_K = condition_profile.temperature_K
    if temperature_K is None and condition_profile.temperature_C is not None:
        temperature_K = c_to_k(condition_profile.temperature_C)
    if temperature_K is None:
        temperature_K = 298.15
    kT_kj_mol = thermal_energy_kj_per_mol(temperature_K)
    physics = {
        "temperature_K": round(float(temperature_K), 2),
        "kT_kj_per_mol": round(float(kT_kj_mol), 4),
        "bond_length_A": None,
        "coulomb_energy_kj_mol": None,
        "polarization_ratio": None,
        "prior_score": None,
        "route_confidence_raw": None,
        "route_confidence_physics": None,
        "multiplier": 1.0,
    }

    charge_a = bond_context.get("gasteiger_charge_a")
    charge_b = bond_context.get("gasteiger_charge_b")
    element_pair = bond_context.get("element_pair") or []
    bond_order = bond_context.get("bond_order") or 1
    if (
        isinstance(charge_a, (int, float))
        and isinstance(charge_b, (int, float))
        and len(element_pair) == 2
        and kT_kj_mol > 0.0
    ):
        bond_length_A = estimate_bond_length_A(
            str(element_pair[0]), str(element_pair[1]), int(bond_order)
        )
        energy_kj_mol = coulomb_energy_kj_mol(charge_a, charge_b, bond_length_A)
        polarization_ratio = abs(energy_kj_mol) / kT_kj_mol
        prior_score = min(1.0, polarization_ratio / PHYSICS_POLARIZATION_RATIO_STRONG)
        physics.update(
            {
                "bond_length_A": round(bond_length_A, 3),
                "coulomb_energy_kj_mol": round(energy_kj_mol, 3),
                "polarization_ratio": round(polarization_ratio, 3),
                "prior_score": round(prior_score, 3),
            }
        )
    return physics


def _fill_physics_layer_xh(
    physics: Dict[str, Any],
    bond_context: Dict[str, Any],
    sre_payload: Optional[Dict[str, Any]] = None,
    mol: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fill missing physics-layer fields for X-H bonds using charge + bond-length proxies."""
    updated = dict(physics or {})
    if updated.get("bond_length_A") is not None:
        return updated

    symbols = bond_context.get("bond_atom_symbols") or bond_context.get("element_pair") or []
    if len(symbols) != 2:
        return updated

    sym_a, sym_b = str(symbols[0]), str(symbols[1])
    pair = {sym_a.upper(), sym_b.upper()}
    if "H" not in pair:
        return updated

    bond_order = int(bond_context.get("bond_order") or 1)
    bond_length_A = estimate_bond_length_A(sym_a, sym_b, bond_order)
    updated["bond_length_A"] = round(float(bond_length_A), 3)

    charge_a = bond_context.get("gasteiger_charge_a")
    charge_b = bond_context.get("gasteiger_charge_b")
    if not isinstance(charge_a, (int, float)) or not isinstance(charge_b, (int, float)):
        c_q = bond_context.get("gasteiger_charge_C")
        h_q = bond_context.get("gasteiger_charge_H")
        if isinstance(c_q, (int, float)) and isinstance(h_q, (int, float)):
            charge_a, charge_b = float(c_q), float(h_q)

    # Generic fallback: derive charges from an AddHs view using stored atom indices.
    if (
        (not isinstance(charge_a, (int, float)) or not isinstance(charge_b, (int, float)))
        and mol is not None
        and Chem is not None
    ):
        try:
            from rdkit.Chem import rdPartialCharges

            mol_h = Chem.AddHs(mol)
            rdPartialCharges.ComputeGasteigerCharges(mol_h)
            atom_indices = bond_context.get("atom_indices") or []
            idx_a = None
            idx_b = None
            if len(atom_indices) >= 2:
                idx_a = int(atom_indices[0]) if atom_indices[0] is not None else None
                idx_b = int(atom_indices[1]) if atom_indices[1] is not None else None
            if idx_a is not None and idx_b is not None:
                if 0 <= idx_a < mol_h.GetNumAtoms() and 0 <= idx_b < mol_h.GetNumAtoms():
                    atom_a = mol_h.GetAtomWithIdx(idx_a)
                    atom_b = mol_h.GetAtomWithIdx(idx_b)
                    if atom_a.HasProp("_GasteigerCharge") and atom_b.HasProp("_GasteigerCharge"):
                        q_a = float(atom_a.GetProp("_GasteigerCharge"))
                        q_b = float(atom_b.GetProp("_GasteigerCharge"))
                        if math.isfinite(q_a) and math.isfinite(q_b):
                            charge_a, charge_b = q_a, q_b
        except Exception:
            pass

    if isinstance(charge_a, (int, float)) and isinstance(charge_b, (int, float)):
        coulomb_kj = coulomb_energy_kj_mol(float(charge_a), float(charge_b), float(bond_length_A))
        updated["coulomb_energy_kj_mol"] = round(float(coulomb_kj), 3)
        polarization = abs(float(charge_a) - float(charge_b)) / max(1e-6, float(bond_length_A))
        updated["polarization_ratio"] = round(float(polarization), 4)
        updated["prior_score"] = round(
            float(max(0.0, min(1.0, polarization / 0.25))),
            3,
        )

    cpt = {}
    if isinstance(sre_payload, dict):
        cpt = sre_payload.get("cpt_scores") if isinstance(sre_payload.get("cpt_scores"), dict) else {}
    track = str(cpt.get("track") or "").lower()
    if track == "radical_hat":
        updated["module_minus1_track"] = "radical_hat"
        bde = cpt.get("bde") if isinstance(cpt.get("bde"), dict) else {}
        if isinstance(bde.get("corrected_kj_mol"), (int, float)):
            updated["module_minus1_bde_kj_mol"] = float(bde["corrected_kj_mol"])
    elif track == "displacement_sn2":
        updated["module_minus1_track"] = "displacement_sn2"
        if isinstance(cpt.get("best_barrier_kj_mol"), (int, float)):
            updated["module_minus1_sn2_barrier_kj_mol"] = float(cpt["best_barrier_kj_mol"])
    return updated


def _resolve_charge_indices(
    mol: Any,
    candidate_bonds: List[Any],
    candidate_atoms: List[int],
) -> Tuple[Optional[int], Optional[int]]:
    carbon_idx = None
    hydrogen_idx = None
    num_atoms = mol.GetNumAtoms() if mol is not None else 0

    if candidate_bonds:
        bond = candidate_bonds[0]
        a_atom = bond.GetBeginAtom()
        b_atom = bond.GetEndAtom()
        a_idx = a_atom.GetIdx()
        b_idx = b_atom.GetIdx()
        # Only use indices that are within range of the original mol
        if a_atom.GetSymbol() == "C" and a_idx < num_atoms:
            carbon_idx = a_idx
        if b_atom.GetSymbol() == "C" and b_idx < num_atoms:
            carbon_idx = carbon_idx if carbon_idx is not None else b_idx
        if a_atom.GetSymbol() == "H" and a_idx < num_atoms:
            hydrogen_idx = a_idx
        if b_atom.GetSymbol() == "H" and b_idx < num_atoms:
            hydrogen_idx = b_idx
        return carbon_idx, hydrogen_idx

    for idx in candidate_atoms:
        if idx >= num_atoms:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            carbon_idx = idx
            break

    return carbon_idx, hydrogen_idx


def _find_attached_hydrogen_index(mol_h: Any, carbon_idx: int) -> Optional[int]:
    carbon_atom = mol_h.GetAtomWithIdx(carbon_idx)
    for neighbor in carbon_atom.GetNeighbors():
        if neighbor.GetSymbol() == "H":
            return neighbor.GetIdx()
    return None


def _get_gasteiger_charge(atom: Any) -> Optional[float]:
    try:
        value = float(atom.GetProp("_GasteigerCharge"))
    except (ValueError, KeyError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _polarity_from_dipole(dipole_proxy: float) -> str:
    if dipole_proxy < 0.05:
        return "nonpolar"
    if dipole_proxy < 0.15:
        return "moderately_polarized"
    return "highly_polarized_CH"


def _heuristic_route_confidence(
    bond_context: Dict[str, Any],
    route: Dict[str, Any],
    warnings: List[str],
    descriptor_incomplete: bool,
) -> float:
    confidence = 0.3
    role_confidence = bond_context.get("primary_role_confidence")
    if role_confidence is not None:
        confidence += 0.4 * max(0.0, min(1.0, role_confidence))
    if bond_context.get("primary_role") or bond_context.get("bond_role"):
        confidence += 0.15
    if route.get("primary"):
        confidence += 0.1
    if descriptor_incomplete:
        confidence -= 0.1
    if "manual review required" in (route.get("mechanisms") or []):
        confidence -= 0.2
    confidence -= min(0.15, 0.03 * len(warnings))
    return max(0.0, min(1.0, confidence))


def _estimate_wetlab_priors(
    difficulty_score: float,
    job_type: str,
) -> Tuple[float, float]:
    score = max(0.0, min(1.0, difficulty_score))
    base_target = 0.7 - (0.6 * score)
    base_any = min(0.9, base_target + 0.15)
    if score < 0.33:
        base_target += 0.05
        base_any += 0.05
    if job_type == JOB_TYPE_REAGENT_GENERATION:
        base_target -= 0.1
        base_any -= 0.1
    base_target = max(0.05, min(0.85, base_target))
    base_any = max(0.1, min(0.9, base_any))
    return base_target, base_any


def _compute_plan(
    difficulty: str,
    job_type: str,
    decision: str,
    force_review: bool = False,
    route_confidence: float = 0.0,
    target_resolution: float = 0.0,
    reaction_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if job_type == JOB_TYPE_REAGENT_GENERATION and difficulty == "EASY":
        difficulty = "MEDIUM"

    base = {
        "EASY": {
            "scaffold_count": 40,
            "topogate_strictness": "standard",
            "qm_level": "none",
            "md_budget": "none",
        },
        "MEDIUM": {
            "scaffold_count": 90,
            "topogate_strictness": "standard",
            "qm_level": "none",
            "md_budget": "low",
            "md_policy": {
                "stage": "post_topogate",
                "top_k": 5,
            },
        },
        "HARD": {
            "scaffold_count": 200,
            "topogate_strictness": "strict",
            "qm_level": "qm-lite",
            "md_budget": "staged",
            "md_policy": {
                "stage": "post_topogate",
                "top_k": 3,
            },
        },
    }[difficulty]

    plan = dict(base)
    if difficulty == "EASY":
        intent_confidence = 0.0
        if reaction_intent:
            intent_confidence = reaction_intent.get("intent_confidence") or 0.0
        if route_confidence >= 0.85 and target_resolution >= 0.9:
            plan["scaffold_count"] = 16
        elif route_confidence < 0.8 or intent_confidence < 0.5:
            plan["scaffold_count"] = max(plan["scaffold_count"], 60)
    plan["active"] = True
    review_required = difficulty == "HARD"
    if decision == "LOW_CONF_GO" or force_review:
        review_required = True
    plan["review_required"] = review_required
    return plan


def _build_job_card(
    decision: str,
    confidence: Dict[str, float],
    difficulty: str,
    difficulty_score: float,
    job_type: str,
    requested_output: Optional[str],
    trap_target: Optional[Dict[str, Any]],
    trap_target_raw: Optional[str],
    resolved: Dict[str, Any],
    structure_summary: Dict[str, Any],
    reaction_intent: Dict[str, Any],
    bond_context: Dict[str, Any],
    equivalence_groups: List[Dict[str, Any]],
    candidate_bond_options: List[Dict[str, Any]],
    candidate_meta: Dict[str, Any],
    constraints: OperationalConstraints,
    constraints_assumed_defaults: bool,
    assumptions_used: List[str],
    descriptor_status: Dict[str, Any],
    substrate_protonation_flags: List[Dict[str, Any]],
    success_definition: Dict[str, Any],
    module1_mode: str,
    module1_weights: Dict[str, float],
    substrate_size_proxies: Dict[str, Any],
    bond_center_hint: Dict[str, Any],
    requested_output_check: Dict[str, Any],
    reaction_condition_field: Dict[str, Any],
    condition_profile: Dict[str, Any],
    reaction_task: Dict[str, Any],
    causal_discovery: Dict[str, Any],
    pipeline_halt_reason: Optional[str],
    required_next_input: List[str],
    reasons: List[str],
    warnings: List[str],
    errors: List[str],
    route_version: str,
    scaffold_library_id: Optional[str],
    route: Optional[Dict[str, Any]] = None,
    compute_plan: Optional[Dict[str, Any]] = None,
    target_resolution_audit: Optional[Dict[str, Any]] = None,
    token_resolution_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "decision": decision,
        "confidence": confidence,
        "difficulty": difficulty,
        "difficulty_label": difficulty,
        "difficulty_score": round(difficulty_score, 3),
        "job_type": job_type,
        "requested_output": requested_output,
        "trap_target": trap_target,
        "trap_target_raw": trap_target_raw,
        "structure_summary": structure_summary,
        "reaction_intent": reaction_intent,
        "bond_context": bond_context,
        "equivalent_bonds": equivalence_groups,
        "candidate_bonds": candidate_bond_options,
        "candidate_meta": candidate_meta,
        "mechanism_route": route or {},
        "route_version": route_version,
        "scaffold_library_id": scaffold_library_id,
        "compute_plan": compute_plan,
        "success_definition": success_definition,
        "module1_mode": module1_mode,
        "module1_weights": module1_weights,
        "substrate_size_proxies": substrate_size_proxies,
        "bond_center_hint": bond_center_hint,
        "requested_output_check": requested_output_check,
        "reaction_condition_field": reaction_condition_field,
        "condition_profile": condition_profile,
        "reaction_task": reaction_task,
        "causal_discovery": causal_discovery,
        "constraints": constraints.to_dict(),
        "constraints_assumed_defaults": constraints_assumed_defaults,
        "assumptions_used": assumptions_used,
        "descriptor_status": descriptor_status,
        "substrate_protonation_flags": substrate_protonation_flags,
        "pipeline_halt_reason": pipeline_halt_reason,
        "resolved_target": resolved,
        "required_next_input": required_next_input,
        "reasons": reasons,
        "warnings": warnings,
        "errors": errors,
        "target_resolution_audit": target_resolution_audit or {},
        "token_resolution_audit": token_resolution_audit or {},
    }
