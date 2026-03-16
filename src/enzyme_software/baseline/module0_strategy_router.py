from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.modules.base import BaseModule

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None
    rdMolDescriptors = None


DIFFICULTY_ORDER = ["EASY", "MEDIUM", "HARD"]
JOB_TYPE_STANDARD = "STANDARD_TRANSFORMATION"
JOB_TYPE_REAGENT_GENERATION = "REAGENT_GENERATION"
JOB_TYPE_ANALYSIS_ONLY = "MECHANISM_PROBE"
ROUTE_VERSION = "v1"

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
        result = route_job(
            smiles=ctx.smiles,
            target_bond=ctx.target_bond,
            requested_output=ctx.requested_output,
            trap_target=ctx.trap_target,
            constraints=ctx.constraints,
        )
        job_card = result.get("job_card")
        ctx.data["job_card"] = job_card
        module0_payload = {key: value for key, value in result.items() if key != "job_card"}
        if job_card is not None:
            module0_payload["job_card_ref"] = "job_card"
        ctx.data["module0_strategy_router"] = module0_payload
        if job_card and job_card.get("constraints"):
            ctx.constraints = OperationalConstraints(**job_card["constraints"])
        return ctx


def route_job(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    trap_target: Optional[str],
    constraints: OperationalConstraints,
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
    resolved = {
        "selection_mode": target_spec.kind,
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
    target_resolution_confidence = 0.0
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
            pipeline_halt_reason=pipeline_halt_reason,
            required_next_input=[],
            reasons=reasons,
            warnings=warnings,
            errors=errors,
            route_version=ROUTE_VERSION,
            scaffold_library_id=None,
        )
        return {"status": "no_go", "job_card": job_card}

    bond_roles: List[Dict[str, Any]] = []
    if rdkit_available:
        bond_roles = _assign_bond_roles(mol)
    structure_summary = (
        _compute_structure_summary(mol, bond_roles) if rdkit_available else {}
    )

    if rdkit_available:
        resolution = _resolve_target_bonds(mol, target_spec, bond_roles, requested_output)
        warnings.extend(resolution["warnings"])
        errors.extend(resolution["errors"])
        candidate_bonds = resolution["bonds"]
        candidate_atoms = resolution["candidate_atoms"]
        candidate_bond_options = resolution["candidate_bonds"]
        candidate_meta = resolution.get("candidate_meta", candidate_meta)
        token_info = resolution.get("token_info")
        selected_entry = resolution.get("selected_entry")
        target_resolution_confidence = resolution.get("target_resolution_confidence", 0.0)
        equivalence_groups = resolution["equivalence_groups"]
        resolved.update(resolution["resolved"])
        if "match_count" not in resolved:
            resolved["match_count"] = resolution.get("match_count")
        resolved["index_base"] = resolution["index_base"]
        resolved["element_pair"] = resolution["element_pair"]
        ambiguous_groups = len(equivalence_groups)

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
            required_next_input: List[str] = []
            if any(
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
                pipeline_halt_reason=pipeline_halt_reason,
                required_next_input=required_next_input,
                reasons=reasons,
                warnings=warnings,
                errors=errors,
                route_version=ROUTE_VERSION,
                scaffold_library_id=None,
            )
            return {"status": "no_go", "job_card": job_card}

        if selected_entry:
            bond_context = _bond_context_from_entry(mol, selected_entry)
        elif candidate_bonds:
            bond_context = _bond_context_from_bond(mol, candidate_bonds[0])
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

    bond_context = _apply_role_overrides(bond_context)

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

    constraints_assumed_defaults = False
    effective_constraints, assumptions_used, constraints_assumed_defaults = (
        _apply_route_constraint_defaults(effective_constraints, route, assumptions_used)
    )
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
    if match_count == 0 and target_spec.kind in {"token", "elements", "smarts"}:
        pipeline_halt_reason = pipeline_halt_reason or "M0_NO_MATCH"
        reasons.append("No bonds matched the requested target.")
        required_next_input.append("target_bond")
        fatal_halt = True
    if match_count and match_count > 1:
        pipeline_halt_reason = pipeline_halt_reason or "M0_AMBIGUOUS_TARGET"
        reasons.append("Multiple bond candidates require user selection.")
        required_next_input.append("target_bond_selection")

    route_confidence = _estimate_route_confidence(
        bond_context=bond_context,
        route=route,
        warnings=warnings,
        descriptor_incomplete=descriptor_status["required"] and not descriptor_status["complete"],
    )
    wetlab_prior_target, wetlab_prior_any = _estimate_wetlab_priors(
        difficulty_score,
        job_type,
    )
    target_resolution = target_resolution_confidence

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
        decision = "NO_GO"
    else:
        if not rdkit_available:
            needs_review = True
            reasons.append("RDKit unavailable; unable to confirm bond validity.")

        if target_resolution < 0.85 or (match_count is not None and match_count != 1):
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
        pipeline_halt_reason=pipeline_halt_reason,
        required_next_input=required_next_input,
        reasons=reasons,
        warnings=warnings,
        errors=errors,
        route_version=ROUTE_VERSION,
        scaffold_library_id=scaffold_library_id,
        route=route,
        compute_plan=compute_plan,
    )
    if decision == "NO_GO":
        status = "no_go"
    elif decision == "HALT_NEED_SELECTION":
        status = "halt"
    else:
        status = "ok"
    return {"status": status, "job_card": job_card}


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
        return None, None, None, warnings

    base = "__".join(parts[:2])
    context = "__".join(parts[2:]) if len(parts) > 2 else None
    if base not in CANONICAL_TOKENS:
        return None, None, None, warnings

    canonical = base if not context else f"{base}__{context}"
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


def _resolve_target_bonds(
    mol: Any,
    target_spec: TargetBondSpec,
    bond_roles: List[Dict[str, Any]],
    requested_output: Optional[str],
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    resolved: Dict[str, Any] = {}
    bonds: List[Any] = []
    equivalence_groups: List[Dict[str, Any]] = []
    candidate_bonds: List[Dict[str, Any]] = []
    index_base = None
    element_pair: Optional[List[str]] = None
    candidate_atoms: List[int] = []
    token_info: Optional[Dict[str, Any]] = None
    match_count = 0
    target_resolution_confidence = 0.0
    selected_entry: Optional[Dict[str, Any]] = None

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
                else:
                    warnings.append("Token context did not match any bond; using base role matches.")

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
                "match_count": match_count,
            }
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
                "role": "unknown",
                "confidence": 0.4,
                "tags": [],
                "evidence": "fallback",
            }
        ],
        "primary_role": "unknown",
        "primary_role_confidence": 0.4,
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
    if kind == "indices":
        return 0.98 if selected_entry or match_count else 0.0
    if kind == "smarts":
        if match_count == 1:
            return 0.9
        return 0.85 if selected_entry else 0.4
    if kind == "token":
        if match_count == 1:
            return 0.9
        return 0.85 if selected_entry else 0.4
    if kind == "elements":
        if match_count == 1:
            return 0.86
        return 0.85 if selected_entry else 0.35
    return 0.0


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
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for bond in bonds:
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
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
        context = _bond_context_from_bond(mol, entry["bond"])
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
        return "C-H" if "C" in symbols else "H-X"

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


def _resolve_charge_indices(
    mol: Any,
    candidate_bonds: List[Any],
    candidate_atoms: List[int],
) -> Tuple[Optional[int], Optional[int]]:
    carbon_idx = None
    hydrogen_idx = None

    if candidate_bonds:
        bond = candidate_bonds[0]
        a_atom = bond.GetBeginAtom()
        b_atom = bond.GetEndAtom()
        if a_atom.GetSymbol() == "C":
            carbon_idx = a_atom.GetIdx()
        if b_atom.GetSymbol() == "C":
            carbon_idx = carbon_idx if carbon_idx is not None else b_atom.GetIdx()
        if a_atom.GetSymbol() == "H":
            hydrogen_idx = a_atom.GetIdx()
        if b_atom.GetSymbol() == "H":
            hydrogen_idx = b_atom.GetIdx()
        return carbon_idx, hydrogen_idx

    for idx in candidate_atoms:
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


def _estimate_route_confidence(
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
    pipeline_halt_reason: Optional[str],
    required_next_input: List[str],
    reasons: List[str],
    warnings: List[str],
    errors: List[str],
    route_version: str,
    scaffold_library_id: Optional[str],
    route: Optional[Dict[str, Any]] = None,
    compute_plan: Optional[Dict[str, Any]] = None,
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
    }
