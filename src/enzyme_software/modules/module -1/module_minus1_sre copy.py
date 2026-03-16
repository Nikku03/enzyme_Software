from __future__ import annotations

# Contract Notes (output contract freeze):
# - Adds ctx.data["module_minus1_sre"] with keys:
#   status, bond360_profile, fragment, cpt_scores, mechanism_eligibility,
#   primary_constraint, confidence_prior, route_bias, cache_key, cache_hit,
#   warnings, errors.
# - Does NOT select a mechanism; only constrains/biases downstream routing.

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from enzyme_software.context import PipelineContext
from enzyme_software.modules.base import BaseModule
from enzyme_software.unity_layer import record_interlink

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms
    from rdkit.Geometry import rdGeometry

    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None
    rdMolTransforms = None
    _RDKIT_AVAILABLE = False

MODULE_MINUS1_VERSION = "v1.0"
SCHEMA_VERSION = 3
CPT_IMPL_VERSION = "2026-01-25"
CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "sre"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
QM_CACHE_DIR = CACHE_DIR / "qm"
QM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MM_TRIGGER_THRESHOLD_KCAL = 3.0
DEFAULT_STERIC_ACCESSIBILITY = 0.5
QM_TRIGGER_KCAL = 3.0
XTB_PATH = shutil.which("xtb")
XTB_MAX_ATOMS = 35
XTB_TIMEOUT = 25
EH_TO_KCAL_MOL = 627.5095
GEO_THRESH = 5.0
ELECTRO_THRESH = 3.0

TOKEN_SMARTS = {
    "ester__acyl_o__acetyl": {
        "smarts": "[CH3][C](=O)O[*]",
        "bond_map": (1, 3),
    },
    "ester__acyl_o": {
        "smarts": "[C](=O)O[*]",
        "bond_map": (0, 2),
    },
    "amide__c_n": {
        "smarts": "[C](=O)N[*]",
        "bond_map": (0, 2),
    },
    "aryl_halide__c_x": {
        "smarts": "[c]([F,Cl,Br,I])",
        "bond_map": (0, 1),
    },
    "ch__aliphatic": {
        "smarts": "[CX4]-[H]",
        "bond_map": (0, 1),
    },
}

TOKEN_ALIAS = {
    "acetyl_ester_c-o": "ester__acyl_o__acetyl",
    "acetyl_ester_c-o__acetyl": "ester__acyl_o__acetyl",
    "ester_c-o": "ester__acyl_o",
    "amide_c-n": "amide__c_n",
    "aryl_c-br": "aryl_halide__c_x",
}

BOND_ROLE_MAP = {
    "ester__acyl_o__acetyl": "ester__acyl_o",
    "ester__acyl_o": "ester__acyl_o",
    "amide__c_n": "amide__c_n",
    "aryl_halide__c_x": "aryl_halide__c_x",
    "ch__aliphatic": "ch__aliphatic",
}

METAL_ATOMIC_NUMBERS = {
    3,
    4,
    11,
    12,
    13,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    50,
    56,
    57,
    58,
    59,
    60,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
}


class CatalyticPerturbationTest:
    """Base class for mechanism-specific CPTs."""

    def __init__(self, name: str, mechanism_family: str) -> None:
        self.name = name
        self.mechanism_family = mechanism_family

    def run(
        self,
        work_mol: "Chem.Mol",
        bond360: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class NucleophileApproachCPT(CatalyticPerturbationTest):
    """Serine hydrolase: nucleophile geometry cost."""

    def __init__(self, config: Dict[str, Any], mechanism_id: str) -> None:
        super().__init__(config.get("id", "nucleophile_approach"), mechanism_id)
        self._config = config

    def run(
        self,
        fragment_mol: "Chem.Mol",
        bond360: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Test energy cost to place nucleophile at attack geometry."""
        if fragment_mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(fragment_mol, randomSeed=0xBEEF)
        fragment_mol = _prepare_uff_mol(fragment_mol, warnings)
        conf = fragment_mol.GetConformer()

        attack_sites = bond360.get("attack_sites", {})
        targets = attack_sites or {}
        carbon_idx = targets.get("electrophile")
        oxygen_idx = targets.get("stabilizable")
        if carbon_idx is None:
            return {"error": "No carbon found in bond", "energy_penalty_kcal": None}

        work_mol = Chem.RWMol(fragment_mol)
        conf = work_mol.GetConformer()
        c_pos = conf.GetAtomPosition(carbon_idx)

        probe_idx = work_mol.AddAtom(Chem.Atom("O"))
        direction = [1.0, 0.0, 0.0]
        if oxygen_idx is not None:
            o_pos = conf.GetAtomPosition(oxygen_idx)
            direction = _attack_direction_from_carbonyl(c_pos, o_pos)
        else:
            warnings.append("Carbonyl oxygen not found; using fallback attack vector.")

        probe_pos = rdGeometry.Point3D(
            c_pos.x + direction[0] * 3.2,
            c_pos.y + direction[1] * 3.2,
            c_pos.z + direction[2] * 3.2,
        )
        conf.SetAtomPosition(probe_idx, probe_pos)
        _add_probe_oh(work_mol, probe_idx, probe_pos, direction)

        work_mol = _finalize_probe_mol(work_mol, warnings)
        ff = _get_uff_forcefield(work_mol, warnings, "nucleophile_probe")
        if ff is None:
            return {
                "error": "UFF failed for nucleophile_probe",
                "energy_penalty_kcal": None,
            }
        ff.AddDistanceConstraint(carbon_idx, probe_idx, 3.1, 3.3, 100.0)
        ff.Minimize(maxIts=200)
        constrained_energy = float(ff.CalcEnergy())

        ff_ref = _get_uff_forcefield(fragment_mol, warnings, "nucleophile_base")
        if ff_ref is None:
            return {
                "error": "UFF failed for nucleophile_base",
                "energy_penalty_kcal": None,
            }
        ff_ref.Minimize(maxIts=200)
        reference_energy = float(ff_ref.CalcEnergy())

        penalty = constrained_energy - reference_energy
        return {
            "energy_penalty_kcal": round(penalty, 3),
            "method": "UFF_nucleophile_probe",
            "geometry_feasible": penalty < 10.0,
            "target_atoms": {
                "electrophile": carbon_idx,
                "stabilizable": oxygen_idx,
            },
            "_qm_probe_xyz": _mol_to_xyz(work_mol),
            "_qm_baseline_xyz": _probe_baseline_xyz(work_mol, probe_idx, c_pos, direction),
        }


class OxyanionHoleCPT(CatalyticPerturbationTest):
    """Oxyanion hole stabilization proxy."""

    def __init__(self, config: Dict[str, Any], mechanism_id: str) -> None:
        super().__init__(config.get("id", "oxyanion_hole"), mechanism_id)
        self._config = config

    def run(
        self,
        fragment_mol: "Chem.Mol",
        bond360: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Test H-bond stabilization of carbonyl oxygen."""
        if fragment_mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(fragment_mol, randomSeed=0xBEEF)
        fragment_mol = _prepare_uff_mol(fragment_mol, warnings)
        conf = fragment_mol.GetConformer()
        attack_sites = bond360.get("attack_sites", {})
        targets = attack_sites or {}
        oxygen_idx = targets.get("stabilizable")
        if oxygen_idx is None:
            return {"error": "No carbonyl oxygen found", "energy_gain_kcal": None}

        work_mol = Chem.RWMol(fragment_mol)
        conf = work_mol.GetConformer()
        o_pos = conf.GetAtomPosition(oxygen_idx)

        probe1_idx = work_mol.AddAtom(Chem.Atom("N"))
        probe2_idx = work_mol.AddAtom(Chem.Atom("N"))

        import random

        random.seed(42)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        pos1 = rdGeometry.Point3D(
            o_pos.x + vec1[0] * 2.0,
            o_pos.y + vec1[1] * 2.0,
            o_pos.z + vec1[2] * 2.0,
        )
        pos2 = rdGeometry.Point3D(
            o_pos.x + vec2[0] * 2.0,
            o_pos.y + vec2[1] * 2.0,
            o_pos.z + vec2[2] * 2.0,
        )

        conf.SetAtomPosition(probe1_idx, pos1)
        conf.SetAtomPosition(probe2_idx, pos2)
        _add_probe_nh3(work_mol, probe1_idx, pos1)
        _add_probe_nh3(work_mol, probe2_idx, pos2)

        work_mol = _finalize_probe_mol(work_mol, warnings)
        ff = _get_uff_forcefield(work_mol, warnings, "oxyanion_probe")
        if ff is None:
            return {
                "error": "UFF failed for oxyanion_probe",
                "energy_gain_kcal": None,
            }
        ff.AddDistanceConstraint(oxygen_idx, probe1_idx, 1.8, 2.2, 100.0)
        ff.AddDistanceConstraint(oxygen_idx, probe2_idx, 1.8, 2.2, 100.0)
        ff.Minimize(maxIts=200)
        stabilized_energy = float(ff.CalcEnergy())

        ff_ref = _get_uff_forcefield(fragment_mol, warnings, "oxyanion_base")
        if ff_ref is None:
            return {
                "error": "UFF failed for oxyanion_base",
                "energy_gain_kcal": None,
            }
        ff_ref.Minimize(maxIts=200)
        reference_energy = float(ff_ref.CalcEnergy())

        gain = reference_energy - stabilized_energy
        return {
            "energy_gain_kcal": round(gain, 3),
            "method": "UFF_Hbond_probes",
            "stabilization_significant": gain > 2.0,
            "target_atoms": {"stabilizable": oxygen_idx},
            "_qm_probe_xyz": _mol_to_xyz(work_mol),
            "_qm_baseline_xyz": _probe_pair_baseline_xyz(
                work_mol, (probe1_idx, probe2_idx), o_pos
            ),
        }


_MECHANISM_LIBRARY_CACHE: Optional[Dict[str, Any]] = None
_MECHANISM_LIBRARY_WARNINGS: List[str] = []



@dataclass
class TargetSpec:
    kind: str
    indices: Optional[Tuple[int, int]]
    token: Optional[str]
    token_context: Optional[str] = None
    canonical_token: Optional[str] = None


class ModuleMinus1SRE(BaseModule):
    name = "Module -1 - Substrate Reality Engine"
    module_id = -1

    def run(self, ctx: PipelineContext) -> PipelineContext:
        result = run_module_minus1(
            smiles=ctx.smiles,
            target_bond=ctx.target_bond,
            requested_output=ctx.requested_output,
            constraints=ctx.constraints.to_dict(),
        )
        ctx.data["module_minus1_sre"] = result
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


def run_module_minus1(
    smiles: str,
    target_bond: str,
    requested_output: Optional[str],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    mechanism_library, lib_warnings = _get_mechanism_library()
    warnings.extend(lib_warnings)
    mechanism_hash = _mechanism_library_hash(mechanism_library)
    target_spec = _parse_target_bond(target_bond)

    if not _RDKIT_AVAILABLE:
        warnings.append("RDKit unavailable; Module -1 using minimal analysis.")
        bond360 = _empty_bond360(target_spec)
        result = _build_output(
            bond360=bond360,
            fragment=_empty_fragment(),
            cpt_scores=_empty_cpt("rdkit_unavailable"),
            mechanism_eligibility=_mechanism_eligibility(bond360.get("bond_type")),
            primary_constraint="NONE",
            confidence_prior=_confidence_prior_from_bond(bond360.get("bond_type")),
            route_bias=_route_bias_from_bond(bond360.get("bond_type")),
            cache_key=None,
            cache_hit=False,
            warnings=warnings,
            errors=errors,
        )
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        errors.append("SMILES parse failed.")
        bond360 = _empty_bond360(target_spec)
        return _build_output(
            bond360=bond360,
            fragment=_empty_fragment(),
            cpt_scores=_empty_cpt("smiles_parse_failed"),
            mechanism_eligibility=_mechanism_eligibility(bond360.get("bond_type")),
            primary_constraint="NONE",
            confidence_prior=0.2,
            route_bias=_route_bias_from_bond(bond360.get("bond_type")),
            cache_key=None,
            cache_hit=False,
            warnings=warnings,
            errors=errors,
        )

    bond_indices, match_count, bond_class = _resolve_target_bond(mol, target_spec)
    if bond_indices is None:
        warnings.append("Target bond could not be resolved; using first bond.")
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            bond_indices = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    if bond_class:
        bond_type = bond_class
    else:
        bond_type = _bond_type_from_token(target_spec.token) or "unknown"

    bond360 = _bond360_profile(mol, bond_indices, bond_type, target_spec)
    bond360["match_count"] = match_count
    bond360["target_bond"] = target_bond
    bond360["canonical_token"] = target_spec.canonical_token

    fragment = _deterministic_fragment(mol, bond_indices, bond360, warnings)
    cache_key = _cache_key(fragment, bond360, constraints, mechanism_hash)
    if cache_key:
        cache_path = CACHE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                if _validate_cache(
                    cached,
                    expected_meta={
                        "schema_version": SCHEMA_VERSION,
                        "cpt_impl_version": CPT_IMPL_VERSION,
                        "mechanism_hash": mechanism_hash,
                    },
                ):
                    return _build_output(
                        bond360=cached["bond360_profile"],
                        fragment=cached["fragment"],
                        cpt_scores=cached["cpt_scores"],
                        mechanism_eligibility=cached["mechanism_eligibility"],
                        primary_constraint=cached["primary_constraint"],
                        confidence_prior=cached["confidence_prior"],
                        route_bias=cached["route_bias"],
                        cache_key=cache_key,
                        cache_hit=True,
                        warnings=warnings,
                        errors=errors,
                    )
            except Exception:
                pass
    bond360_fragment = _remap_bond360(bond360, fragment.get("atom_map_parent_to_fragment"))
    fragment_mol = _build_fragment_mol(fragment, warnings)
    fragment_bond_indices = fragment.get("bond_indices_fragment")
    mm_results, qm_results = _run_mechanism_cpts(
        fragment_mol,
        bond360_fragment,
        fragment,
        warnings,
        mechanism_library,
        constraints,
    )
    cpt_scores = {
        "status": "ok"
        if mm_results
        else ("no_fragment" if not fragment.get("fragment_smiles") else "no_cpts"),
        "mm_results": mm_results,
        "qm_results": qm_results,
        "mm": _format_mm_results(mm_results),
        "qm": _format_qm_results(qm_results),
        "conformer_count": 1,
        "variance_estimate": 0.0,
        "triggered_qm": bool(qm_results),
    }

    mechanism_eligibility = _derive_mechanism_eligibility(
        bond360, mm_results, mechanism_library
    )
    primary_constraint = _derive_primary_constraint(mm_results)
    confidence_prior = _confidence_prior_from_bond(bond_type)
    route_bias = _route_bias_from_bond(bond_type)

    cache_path = CACHE_DIR / f"{cache_key}.json" if cache_key else None
    cache_hit = False
    if cache_path and cache_path.exists():
        cache_hit = True
    if cache_path:
        payload = {
            "bond360_profile": bond360,
            "fragment": fragment,
            "cpt_scores": cpt_scores,
            "mechanism_eligibility": mechanism_eligibility,
            "primary_constraint": primary_constraint,
            "confidence_prior": confidence_prior,
            "route_bias": route_bias,
            "cache_meta": {
                "schema_version": SCHEMA_VERSION,
                "cpt_impl_version": CPT_IMPL_VERSION,
                "mechanism_hash": mechanism_hash,
            },
        }
        try:
            cache_path.write_text(json.dumps(payload, indent=2))
        except OSError:
            warnings.append("Failed to write SRE cache.")

    return _build_output(
        bond360=bond360,
        fragment=fragment,
        cpt_scores=cpt_scores,
        mechanism_eligibility=mechanism_eligibility,
        primary_constraint=primary_constraint,
        confidence_prior=confidence_prior,
        route_bias=route_bias,
        cache_key=cache_key,
        cache_hit=cache_hit,
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
    token = raw or None
    token_norm = token.lower().replace(" ", "") if token else ""
    canonical = TOKEN_ALIAS.get(token_norm)
    token_context = None
    if canonical and "__" in canonical:
        parts = canonical.split("__", 2)
        if len(parts) == 3:
            token_context = parts[2]
    return TargetSpec(
        kind="token",
        indices=None,
        token=token,
        canonical_token=canonical,
        token_context=token_context,
    )


def _get_mechanism_library() -> Tuple[Dict[str, Any], List[str]]:
    global _MECHANISM_LIBRARY_CACHE, _MECHANISM_LIBRARY_WARNINGS
    if _MECHANISM_LIBRARY_CACHE is not None:
        return _MECHANISM_LIBRARY_CACHE, list(_MECHANISM_LIBRARY_WARNINGS)
    env_dir = os.environ.get("BOND_BREAK_MECH_LIB_DIR")
    candidate_dirs = []
    if env_dir:
        candidate_dirs.append(Path(env_dir))
    candidate_dirs.append(Path(__file__).resolve().parents[4] / "mechanism_library")
    candidate_dirs.append(Path(__file__).resolve().parents[3] / "mechanism_library")
    candidate_dirs.append(Path(__file__).resolve().parent / "mechanism_library")
    mech_dir = next((path for path in candidate_dirs if path.exists()), None)
    if mech_dir is None:
        library, warnings = _default_mechanism_library(), [
            "mechanism_library: directory not found; using embedded default."
        ]
    else:
        library, warnings = _load_mechanism_library(mech_dir)
    _MECHANISM_LIBRARY_CACHE = library
    _MECHANISM_LIBRARY_WARNINGS = warnings
    return library, list(warnings)


def _load_mechanism_library(mech_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    library: Dict[str, Any] = {}
    if mech_dir.exists():
        for path in mech_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                warnings.append(f"mechanism_library: invalid JSON in {path.name}")
                continue
            mechanism_id = payload.get("mechanism_id")
            if not mechanism_id:
                warnings.append(f"mechanism_library: missing mechanism_id in {path.name}")
                continue
            if "cpts" not in payload or "required_bond_types" not in payload:
                warnings.append(f"mechanism_library: missing keys in {path.name}")
                continue
            if "eligibility" not in payload:
                warnings.append(f"mechanism_library: missing eligibility in {path.name}")
                continue
            eligibility = payload.get("eligibility") or {}
            if not isinstance(eligibility.get("required_bond_roles"), list):
                warnings.append(f"mechanism_library: invalid eligibility in {path.name}")
                continue
            if not isinstance(payload.get("cpts"), list):
                warnings.append(f"mechanism_library: cpts must be list in {path.name}")
                continue
            invalid_cpt = False
            for cpt in payload.get("cpts") or []:
                if not isinstance(cpt, dict):
                    invalid_cpt = True
                    break
                if not cpt.get("id") or not cpt.get("cpt_kind") or not cpt.get("score_function"):
                    invalid_cpt = True
                    break
            if invalid_cpt:
                warnings.append(f"mechanism_library: invalid cpt schema in {path.name}")
                continue
            library[str(mechanism_id)] = payload
    if not library:
        warnings.append("mechanism_library: using embedded default.")
        library = _default_mechanism_library()
    return library, warnings


def _default_mechanism_library() -> Dict[str, Any]:
    return {
        "serine_hydrolase": {
            "mechanism_id": "serine_hydrolase",
            "required_bond_types": ["ester", "amide"],
            "eligibility": {
                "required_bond_roles": ["ester__acyl_o", "amide__c_n", "ester", "amide"],
                "forbidden_roles": [],
            },
            "cpts": [
                {
                    "id": "nucleophile_approach",
                    "cpt_kind": "NUCLEOPHILE_APPROACH",
                    "type": "geometric_constraint",
                    "score_function": "energy_penalty",
                    "constraint_type": "GEOMETRIC",
                    "direction": "penalty",
                    "fail_threshold": 5.0,
                    "severity_weight": 1.0,
                    "requires_targets": ["electrophile"],
                    "distance_angstrom": 3.2,
                    "angle_degrees": 107.0,
                    "qm_trigger_threshold_kcal": 3.0,
                },
                {
                    "id": "oxyanion_hole",
                    "cpt_kind": "OXYANION_HOLE",
                    "type": "electrostatic_stabilization",
                    "score_function": "energy_gain",
                    "constraint_type": "ELECTROSTATIC",
                    "direction": "gain",
                    "fail_threshold": 2.0,
                    "severity_weight": 1.0,
                    "requires_targets": ["stabilizable"],
                    "distance_angstrom": 2.0,
                    "qm_trigger_threshold_kcal": 2.0,
                },
            ],
        }
    }


def _resolve_target_bond(
    mol: "Chem.Mol", target_spec: TargetSpec
) -> Tuple[Optional[Tuple[int, int]], int, Optional[str]]:
    if target_spec.kind == "indices" and target_spec.indices:
        bond = mol.GetBondBetweenAtoms(*target_spec.indices)
        if bond is not None:
            return target_spec.indices, 1, _bond_class_from_atoms(mol, bond)
        return None, 0, None
    token = target_spec.canonical_token or target_spec.token
    if not token:
        return None, 0, None
    token_key = token.lower()
    entry = TOKEN_SMARTS.get(token_key)
    if entry is None:
        token_key = TOKEN_ALIAS.get(token_key, token_key)
        entry = TOKEN_SMARTS.get(token_key)
    if entry is None:
        return None, 0, _bond_type_from_token(token)
    smarts = entry["smarts"]
    match_atoms = _smarts_matches(mol, smarts)
    if not match_atoms:
        return None, 0, _bond_type_from_token(token)
    bond_map = entry.get("bond_map") or (0, 1)
    bonds = []
    for match in match_atoms:
        if max(bond_map) >= len(match):
            continue
        bonds.append((match[bond_map[0]], match[bond_map[1]]))
    if not bonds:
        return None, 0, _bond_type_from_token(token)
    bonds = sorted({tuple(sorted(pair)) for pair in bonds})
    return bonds[0], len(bonds), _bond_type_from_token(token)


def _smarts_matches(mol: "Chem.Mol", smarts: str) -> List[Tuple[int, ...]]:
    if not smarts:
        return []
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return []
    return list(mol.GetSubstructMatches(patt, uniquify=True))


def _bond_class_from_atoms(mol: "Chem.Mol", bond: "Chem.Bond") -> str:
    atom_a = bond.GetBeginAtom()
    atom_b = bond.GetEndAtom()
    symbols = {atom_a.GetSymbol(), atom_b.GetSymbol()}
    if bond.GetBondTypeAsDouble() == 1.0 and symbols == {"C", "O"}:
        return "ester"
    if bond.GetBondTypeAsDouble() == 1.0 and symbols == {"C", "N"}:
        return "amide"
    if "H" in symbols:
        return "ch"
    if "Br" in symbols or "Cl" in symbols or "I" in symbols or "F" in symbols:
        return "aryl_halide"
    return "unknown"


def _bond_type_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token_lower = token.lower()
    if "ester" in token_lower:
        return "ester"
    if "amide" in token_lower:
        return "amide"
    if "halide" in token_lower or "c-br" in token_lower or "c-cl" in token_lower:
        return "aryl_halide"
    if "c-h" in token_lower or "ch" in token_lower:
        return "ch"
    return None


def _bond_role_from_token(token: Optional[str]) -> Optional[Dict[str, str]]:
    if not token:
        return None
    token_key = token.lower().replace(" ", "")
    canonical = TOKEN_ALIAS.get(token_key, token_key)
    role = BOND_ROLE_MAP.get(canonical)
    if not role:
        return None
    return {"role": role, "canonical_token": canonical}


def _build_attack_sites(
    mol: "Chem.Mol",
    bond_indices: Tuple[int, int],
    primary_role: Optional[str],
) -> Tuple[Dict[str, Optional[int]], List[int]]:
    attack_sites: Dict[str, Optional[int]] = {}
    fg_atoms: List[int] = []
    if bond_indices is None:
        return attack_sites, fg_atoms
    a_idx, b_idx = bond_indices
    fg_atoms.extend([a_idx, b_idx])
    atom_a = mol.GetAtomWithIdx(a_idx)
    atom_b = mol.GetAtomWithIdx(b_idx)
    role = (primary_role or "").lower()

    if "ester" in role or "amide" in role:
        carbonyl_c, carbonyl_o = _carbonyl_atoms_from_bond(mol, bond_indices)
        electrophile = carbonyl_c
        stabilizable = carbonyl_o
        leaving = None
        if electrophile is not None:
            leaving = b_idx if electrophile == a_idx else a_idx
        if stabilizable is not None:
            fg_atoms.append(stabilizable)
        attack_sites = {
            "electrophile": electrophile,
            "leaving": leaving,
            "stabilizable": stabilizable,
        }
        return attack_sites, fg_atoms

    if "aryl_halide" in role:
        leaving = None
        electrophile = None
        for idx in (a_idx, b_idx):
            symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            if symbol in {"F", "Cl", "Br", "I"}:
                leaving = idx
            else:
                electrophile = idx
        attack_sites = {"electrophile": electrophile, "leaving": leaving}
        return attack_sites, fg_atoms

    if "ch__aliphatic" in role or _bond_type_is_ch(atom_a, atom_b):
        carbon_idx = a_idx if atom_a.GetSymbol() == "C" else b_idx if atom_b.GetSymbol() == "C" else None
        attack_sites = {"abstraction_site": carbon_idx}
        return attack_sites, fg_atoms

    attack_sites = {"electrophile": a_idx, "leaving": b_idx}
    return attack_sites, fg_atoms


def _carbonyl_atoms_from_bond(
    mol: "Chem.Mol", bond_indices: Tuple[int, int]
) -> Tuple[Optional[int], Optional[int]]:
    a_idx, b_idx = bond_indices
    for idx in (a_idx, b_idx):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            continue
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "O":
                    return idx, other.GetIdx()
    return None, None


def _bond_type_is_ch(atom_a: "Chem.Atom", atom_b: "Chem.Atom") -> bool:
    symbols = {atom_a.GetSymbol(), atom_b.GetSymbol()}
    return "H" in symbols


def _pick_oxygen_index(
    mol: "Chem.Mol", idx_a: int, idx_b: int
) -> Optional[int]:
    atom_a = mol.GetAtomWithIdx(idx_a)
    atom_b = mol.GetAtomWithIdx(idx_b)
    if atom_a.GetSymbol() == "O":
        return idx_a
    if atom_b.GetSymbol() == "O":
        return idx_b
    return None


def _bond360_profile(
    mol: "Chem.Mol",
    bond_indices: Optional[Tuple[int, int]],
    bond_type: str,
    target_spec: TargetSpec,
) -> Dict[str, Any]:
    profile = {
        "bond_length": None,
        "bond_order": None,
        "bond_type": bond_type,
        "is_conjugated": False,
        "is_aromatic": False,
        "in_ring": False,
        "bond_indices": list(bond_indices) if bond_indices else [],
        "formal_charge_pair": [0, 0],
        "partial_charges": [None, None],
        "dipole_proxy": None,
        "steric_accessibility": DEFAULT_STERIC_ACCESSIBILITY,
        "topological_depth": None,
        "neighbor_hetero_atoms": 0,
        "rdkit_available": _RDKIT_AVAILABLE,
        "bond_roles": [],
        "primary_role": None,
        "functional_group_atoms": [],
        "attack_sites": {},
    }
    if bond_indices is None or mol is None:
        return profile
    bond = mol.GetBondBetweenAtoms(*bond_indices)
    if bond is None:
        return profile
    profile["bond_order"] = int(round(bond.GetBondTypeAsDouble()))
    profile["is_conjugated"] = bool(bond.GetIsConjugated())
    profile["is_aromatic"] = bool(bond.GetIsAromatic())
    profile["in_ring"] = bool(bond.IsInRing())
    atom_a = bond.GetBeginAtom()
    atom_b = bond.GetEndAtom()
    profile["formal_charge_pair"] = [atom_a.GetFormalCharge(), atom_b.GetFormalCharge()]
    profile["neighbor_hetero_atoms"] = _count_neighbor_hetero_atoms(atom_a) + _count_neighbor_hetero_atoms(
        atom_b
    )
    depth = _topological_depth(mol, bond_indices)
    profile["topological_depth"] = depth
    profile["steric_accessibility"] = _steric_accessibility(atom_a, atom_b, depth)

    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        except Exception:
            pass
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        profile["bond_length"] = round(
            float(rdMolTransforms.GetBondLength(conf, bond_indices[0], bond_indices[1])), 3
        )

    charges = _gasteiger_charges(mol, bond_indices)
    if charges:
        profile["partial_charges"] = list(charges)
        profile["dipole_proxy"] = round(abs(charges[0] - charges[1]), 3)
    role_info = _bond_role_from_token(target_spec.canonical_token or target_spec.token)
    if role_info:
        profile["bond_roles"] = [role_info["role"]]
        profile["primary_role"] = role_info["role"]
    else:
        profile["bond_roles"] = [bond_type]
        profile["primary_role"] = bond_type
    attack_sites, fg_atoms = _build_attack_sites(mol, bond_indices, profile["primary_role"])
    profile["attack_sites"] = attack_sites
    profile["functional_group_atoms"] = fg_atoms
    return profile


def _gasteiger_charges(
    mol: "Chem.Mol", bond_indices: Tuple[int, int]
) -> Optional[Tuple[float, float]]:
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None
    charges = []
    for idx in bond_indices:
        atom = mol.GetAtomWithIdx(idx)
        charge = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else None
        if charge is None:
            return None
        try:
            charges.append(float(charge))
        except ValueError:
            return None
    return (charges[0], charges[1])


def _count_neighbor_hetero_atoms(atom: "Chem.Atom") -> int:
    count = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() not in {"C", "H"}:
            count += 1
    return count


def _topological_depth(mol: "Chem.Mol", bond_indices: Tuple[int, int]) -> int:
    try:
        dist = Chem.GetDistanceMatrix(mol)
        a_idx, b_idx = bond_indices
        max_a = max(dist[a_idx])
        max_b = max(dist[b_idx])
        return int(max(max_a, max_b))
    except Exception:
        return 0


def _steric_accessibility(atom_a: "Chem.Atom", atom_b: "Chem.Atom", depth: int) -> float:
    degree = atom_a.GetDegree() + atom_b.GetDegree()
    penalty = 0.05 * max(0, degree - 2)
    depth_penalty = 0.02 * max(0, depth - 2)
    return max(0.0, min(1.0, 1.0 - penalty - depth_penalty))


def _deterministic_fragment(
    mol: "Chem.Mol",
    bond_indices: Optional[Tuple[int, int]],
    bond360: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    if bond_indices is None:
        return _empty_fragment()
    selected = _expand_fragment_atoms(mol, bond_indices)
    if not selected:
        return _empty_fragment()
    cut_bonds = _cut_bonds(mol, selected)
    ordered_selected = [bond_indices[0], bond_indices[1]]
    ordered_selected.extend(sorted(idx for idx in selected if idx not in bond_indices))
    fragment, cap_map, atom_map = _smart_capped_fragment(mol, ordered_selected, cut_bonds)
    fragment_smiles = None
    fragment_smiles_ordered = None
    if fragment is not None:
        try:
            Chem.SanitizeMol(fragment)
            fragment_smiles = Chem.MolToSmiles(fragment, canonical=True)
            fragment_smiles_ordered = Chem.MolToSmiles(fragment, canonical=False)
        except Exception:
            warnings.append("Fragment sanitization failed; SMILES may be missing.")
    net_charge = _net_charge(fragment) if fragment is not None else 0
    spin_mult = _spin_multiplicity(fragment) if fragment is not None else 1
    fragment_bond_indices = [0, 1]
    return {
        "fragment_smiles": fragment_smiles,
        "fragment_smiles_ordered": fragment_smiles_ordered,
        "net_charge": net_charge,
        "spin_multiplicity": spin_mult,
        "cut_bonds": cut_bonds,
        "cap_map": cap_map,
        "atom_map_parent_to_fragment": {str(k): v for k, v in atom_map.items()},
        "atom_count": len(selected),
        "notes": [
            "Fragment caps applied using H/CH3 based on charge/valence.",
            "Fragment derived from bond-centered BFS + conjugation expansion.",
        ],
        "bond_indices": list(bond_indices),
        "bond_indices_fragment": fragment_bond_indices,
        "bond_type": bond360.get("bond_type"),
    }


def _expand_fragment_atoms(mol: "Chem.Mol", bond_indices: Tuple[int, int]) -> Set[int]:
    selected: Set[int] = set(bond_indices)
    frontier = set(bond_indices)
    for _ in range(2):
        next_frontier: Set[int] = set()
        for idx in frontier:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                next_frontier.add(neighbor.GetIdx())
        selected.update(next_frontier)
        frontier = next_frontier
    selected.update(_conjugation_expand(mol, selected))
    selected.update(_charge_expand(mol, selected))
    return selected


def _conjugation_expand(mol: "Chem.Mol", selected: Set[int]) -> Set[int]:
    expanded = set(selected)
    added = True
    while added:
        added = False
        for bond in mol.GetBonds():
            if not bond.GetIsConjugated():
                continue
            a_idx = bond.GetBeginAtomIdx()
            b_idx = bond.GetEndAtomIdx()
            if a_idx in expanded or b_idx in expanded:
                if a_idx not in expanded or b_idx not in expanded:
                    expanded.update({a_idx, b_idx})
                    added = True
    return expanded


def _charge_expand(mol: "Chem.Mol", selected: Set[int]) -> Set[int]:
    expanded = set(selected)
    for idx in list(expanded):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0 or atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS:
            for neighbor in atom.GetNeighbors():
                expanded.add(neighbor.GetIdx())
    return expanded


def _cut_bonds(mol: "Chem.Mol", selected: Set[int]) -> List[List[int]]:
    cut_bonds: List[List[int]] = []
    for bond in mol.GetBonds():
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        if a_idx in selected and b_idx not in selected and bond.GetBondTypeAsDouble() == 1.0:
            cut_bonds.append([a_idx, b_idx])
        if b_idx in selected and a_idx not in selected and bond.GetBondTypeAsDouble() == 1.0:
            cut_bonds.append([b_idx, a_idx])
    return cut_bonds


def _smart_capped_fragment(
    mol: "Chem.Mol", selected: Set[int], cut_bonds: List[List[int]]
) -> Tuple[Optional["Chem.Mol"], Dict[str, str], Dict[int, int]]:
    """Build fragment with chemically consistent caps."""
    atom_map: Dict[int, int] = {}
    fragment = Chem.RWMol()

    # Copy selected atoms.
    for idx in selected:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        atom_map[idx] = fragment.AddAtom(new_atom)

    # Copy internal bonds.
    for bond in mol.GetBonds():
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        if a_idx in selected and b_idx in selected:
            fragment.AddBond(atom_map[a_idx], atom_map[b_idx], bond.GetBondType())

    # Smart capping.
    cap_map: Dict[str, str] = {}
    for inside_idx, _ in cut_bonds:
        inside_atom = mol.GetAtomWithIdx(inside_idx)
        symbol = inside_atom.GetSymbol()
        charge = inside_atom.GetFormalCharge()

        if inside_atom.GetIsAromatic() or inside_atom.GetIsConjugated():
            cap = Chem.Atom("H")
            cap_type = "H"
        elif symbol != "C":
            cap = Chem.Atom("H")
            cap_type = "H"
            if symbol == "O" and charge == -1:
                fragment.GetAtomWithIdx(atom_map[inside_idx]).SetFormalCharge(0)
        else:
            cap = Chem.Atom("C")
            cap_type = "CH3"

        cap_idx = fragment.AddAtom(cap)
        fragment.AddBond(atom_map[inside_idx], cap_idx, Chem.BondType.SINGLE)
        cap_map[str(inside_idx)] = cap_type

    return fragment.GetMol(), cap_map, atom_map


def _net_charge(mol: Optional["Chem.Mol"]) -> int:
    if mol is None:
        return 0
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _spin_multiplicity(mol: Optional["Chem.Mol"]) -> int:
    if mol is None:
        return 1
    radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    return 1 + int(radicals) if radicals > 0 else 1


def _run_mechanism_cpts(
    fragment_mol: Optional["Chem.Mol"],
    bond360: Dict[str, Any],
    fragment: Dict[str, Any],
    warnings: List[str],
    mechanism_library: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run CPTs relevant to the bond role and return MM/QM results."""
    mm_results: Dict[str, Any] = {}
    qm_results: Dict[str, Any] = {}
    if fragment_mol is None:
        return mm_results, qm_results

    primary_role = bond360.get("primary_role")
    attack_sites = bond360.get("attack_sites") or {}

    pH = constraints.get("ph_min") or constraints.get("ph_max") or 7.0
    temp_c = constraints.get("temperature_c", 25.0)
    for mechanism_id, mech in mechanism_library.items():
        eligibility = mech.get("eligibility") or {}
        required_roles = eligibility.get("required_bond_roles") or []
        forbidden_roles = eligibility.get("forbidden_roles") or []
        if forbidden_roles and primary_role in forbidden_roles:
            continue
        if required_roles and primary_role not in required_roles:
            continue
        for cpt_config in mech.get("cpts") or []:
            cpt_id = str(cpt_config.get("id") or "cpt")
            requires_targets = cpt_config.get("requires_targets") or []
            if any(target not in attack_sites or attack_sites[target] is None for target in requires_targets):
                mm_results[f"{mechanism_id}__{cpt_id}"] = {
                    "skipped": True,
                    "reason": "missing_required_targets",
                    "mechanism": mechanism_id,
                    "cpt_id": cpt_id,
                }
                continue
            cpt = _cpt_from_config(cpt_config, mechanism_id, warnings)
            if cpt is None:
                continue
            key = f"{mechanism_id}__{cpt_id}"
            try:
                result = cpt.run(fragment_mol, bond360, warnings)
                if (
                    result.get("energy_penalty_kcal") is None
                    and result.get("energy_gain_kcal") is None
                    and "error" not in result
                ):
                    warnings.append(f"CPT {key} returned no energy value.")
                probe_xyz = result.pop("_qm_probe_xyz", None)
                baseline_xyz = result.pop("_qm_baseline_xyz", None)
                result["mechanism"] = mechanism_id
                result["cpt_id"] = cpt_id
                result["score_function"] = cpt_config.get("score_function")
                result["constraint_type"] = cpt_config.get("constraint_type")
                result["direction"] = cpt_config.get("direction")
                result["fail_threshold"] = cpt_config.get("fail_threshold")
                result["severity_weight"] = cpt_config.get("severity_weight")
                _attach_mm_delta(result)
                mm_results[key] = result
                qm_result = _maybe_run_qm(
                    fragment,
                    cpt_config,
                    mechanism_id,
                    cpt_id,
                    pH,
                    temp_c,
                    warnings,
                    result,
                    baseline_xyz=baseline_xyz,
                    probe_xyz=probe_xyz,
                )
                if qm_result is not None:
                    qm_results[key] = qm_result
            except Exception as exc:
                mm_results[key] = {
                    "error": str(exc),
                    "energy_penalty_kcal": None,
                    "mechanism": mechanism_id,
                    "cpt_id": cpt_id,
                }
    return mm_results, qm_results


def _cpt_from_config(
    cpt_config: Dict[str, Any], mechanism_id: str, warnings: List[str]
) -> Optional[CatalyticPerturbationTest]:
    cpt_kind = str(cpt_config.get("cpt_kind") or "").upper()
    factory = {
        "NUCLEOPHILE_APPROACH": NucleophileApproachCPT,
        "OXYANION_HOLE": OxyanionHoleCPT,
    }
    cpt_cls = factory.get(cpt_kind)
    if cpt_cls is None:
        warnings.append(f"Unknown cpt_kind '{cpt_kind}' for {mechanism_id}")
        return None
    return cpt_cls(cpt_config, mechanism_id)


def _maybe_run_qm(
    fragment: Dict[str, Any],
    cpt_config: Dict[str, Any],
    mechanism_id: str,
    cpt_id: str,
    pH: float,
    temp_c: float,
    warnings: List[str],
    mm_result: Dict[str, Any],
    *,
    baseline_xyz: Optional[str],
    probe_xyz: Optional[str],
) -> Optional[Dict[str, Any]]:
    if XTB_PATH is None:
        return None
    if not baseline_xyz or not probe_xyz:
        return None
    if _xyz_atom_count(baseline_xyz) != _xyz_atom_count(probe_xyz):
        warnings.append(f"QM skipped for {mechanism_id}__{cpt_id}: geometry mismatch.")
        return None
    if _xyz_atom_count(baseline_xyz) > XTB_MAX_ATOMS:
        return None
    score_fn = cpt_config.get("score_function")
    threshold = float(cpt_config.get("qm_trigger_threshold_kcal") or QM_TRIGGER_KCAL)
    if score_fn == "energy_penalty":
        penalty = mm_result.get("energy_penalty_kcal")
        if not isinstance(penalty, (int, float)) or abs(float(penalty)) < threshold:
            return None
    if score_fn == "energy_gain":
        gain = mm_result.get("energy_gain_kcal")
        if not isinstance(gain, (int, float)) or float(gain) < threshold:
            return None
    fragment_smiles = fragment.get("fragment_smiles")
    if not fragment_smiles:
        return None
    charge = int(fragment.get("net_charge") or 0)
    spin = int(fragment.get("spin_multiplicity") or 1)
    cache_key = _qm_cache_key(
        fragment_smiles,
        charge,
        spin,
        pH,
        temp_c,
        mechanism_id,
        cpt_id,
        baseline_xyz,
        probe_xyz,
    )
    cache_path = QM_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    baseline_kcal = _run_xtb_xyz(baseline_xyz, charge, spin, warnings)
    probe_kcal = _run_xtb_xyz(probe_xyz, charge, spin, warnings)
    if baseline_kcal is None or probe_kcal is None:
        return None
    if score_fn == "energy_gain":
        delta_kcal = baseline_kcal - probe_kcal
    else:
        delta_kcal = probe_kcal - baseline_kcal
    payload = {
        "delta_kcal": round(float(delta_kcal), 4),
        "baseline_kcal": round(float(baseline_kcal), 4),
        "probe_kcal": round(float(probe_kcal), 4),
        "method": "xtb_GFN2",
    }
    try:
        cache_path.write_text(json.dumps(payload, indent=2))
    except OSError:
        warnings.append(f"QM cache write failed for {mechanism_id}__{cpt_id}")
    return payload


def _run_xtb_xyz(
    xyz: str, charge: int, spin: int, warnings: List[str]
) -> Optional[float]:
    if XTB_PATH is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        xyz_path = Path(tmp_dir) / "fragment.xyz"
        try:
            xyz_path.write_text(xyz)
        except Exception:
            warnings.append("xtb: failed to write xyz.")
            return None
        cmd = [
            XTB_PATH,
            str(xyz_path),
            "--chrg",
            str(charge),
            "--uhf",
            str(max(0, spin - 1)),
            "--alpb",
            "water",
            "--gfn",
            "2",
            "--etemp",
            "300",
            "--cycles",
            "250",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=XTB_TIMEOUT,
                check=False,
            )
        except Exception:
            warnings.append("xtb failed to run.")
            return None
        output = proc.stdout + "\n" + proc.stderr
        match = re.search(r"TOTAL ENERGY\\s+(-?\\d+\\.\\d+)", output)
        if not match:
            warnings.append("xtb energy parse failed.")
            return None
        try:
            energy_eh = float(match.group(1))
        except ValueError:
            return None
        return energy_eh * EH_TO_KCAL_MOL


def _qm_cache_key(
    fragment_smiles: str,
    charge: int,
    spin: int,
    pH: float,
    temp_c: float,
    mech_id: str,
    cpt_id: str,
    baseline_xyz: str,
    probe_xyz: str,
) -> str:
    raw = json.dumps(
        {
            "fragment_smiles": fragment_smiles,
            "charge": charge,
            "spin": spin,
            "pH_bucket": math.floor(float(pH) * 2.0) / 2.0,
            "temperature_bucket": int(round(float(temp_c))),
            "mechanism_id": mech_id,
            "cpt_id": cpt_id,
            "baseline_geom": _geometry_hash(baseline_xyz),
            "probe_geom": _geometry_hash(probe_xyz),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    if bond_type == "ch":
        return {
            "serine_hydrolase": "REJECTED",
            "radical_transfer": "REQUIRE_QUORUM",
        }
    return {"serine_hydrolase": "REQUIRE_QUORUM"}


def _primary_constraint(bond360: Dict[str, Any]) -> str:
    steric = bond360.get("steric_accessibility")
    dipole = bond360.get("dipole_proxy")
    if isinstance(steric, (int, float)) and steric < 0.3:
        return "GEOMETRIC"
    if isinstance(dipole, (int, float)) and dipole > 0.3:
        return "ELECTROSTATIC"
    return "NONE"


def _derive_primary_constraint(mm_results: Dict[str, Any]) -> str:
    best = ("NONE", 0.0)
    for result in mm_results.values():
        if not isinstance(result, dict):
            continue
        constraint_type = result.get("constraint_type") or "NONE"
        direction = result.get("direction") or "penalty"
        threshold = float(result.get("fail_threshold") or 0.0)
        weight = float(result.get("severity_weight") or 1.0)
        value = result.get("deltaE_rel")
        if not isinstance(value, (int, float)):
            continue
        if direction == "penalty":
            severity = max(0.0, float(value) - threshold) * weight
        else:
            severity = max(0.0, threshold - float(value)) * weight
        if severity > best[1]:
            best = (constraint_type, severity)
    return best[0] if best[1] > 0.0 else "NONE"


def _build_fragment_mol(
    fragment: Dict[str, Any], warnings: List[str]
) -> Optional["Chem.Mol"]:
    frag_smiles = fragment.get("fragment_smiles_ordered") or fragment.get("fragment_smiles")
    if frag_smiles is None or Chem is None or AllChem is None:
        return None
    fragment_mol = Chem.MolFromSmiles(frag_smiles)
    if fragment_mol is None:
        return None
    try:
        fragment_mol = Chem.AddHs(fragment_mol, addCoords=True)
        fragment_mol = _prepare_uff_mol(fragment_mol, warnings)
        AllChem.EmbedMolecule(fragment_mol, randomSeed=0xBEEF)
        AllChem.UFFOptimizeMolecule(fragment_mol)
    except Exception as exc:
        warnings.append(f"CPT fragment embedding failed; CPTs skipped. ({exc})")
        return None
    return fragment_mol


def _remap_bond360(
    bond360: Dict[str, Any],
    atom_map_parent_to_fragment: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    if not atom_map_parent_to_fragment:
        return dict(bond360)
    mapped = dict(bond360)
    mapper = {int(k): int(v) for k, v in atom_map_parent_to_fragment.items()}
    if "bond_indices" in bond360:
        indices = []
        for idx in bond360.get("bond_indices") or []:
            if isinstance(idx, int) and idx in mapper:
                indices.append(mapper[idx])
        mapped["bond_indices"] = indices
    attack_sites = {}
    for key, idx in (bond360.get("attack_sites") or {}).items():
        attack_sites[key] = mapper.get(int(idx)) if isinstance(idx, int) else None
    mapped["attack_sites"] = attack_sites
    fg_atoms = []
    for idx in bond360.get("functional_group_atoms") or []:
        if isinstance(idx, int) and idx in mapper:
            fg_atoms.append(mapper[idx])
    mapped["functional_group_atoms"] = fg_atoms
    return mapped


def _attach_mm_delta(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict):
        return
    if isinstance(result.get("energy_penalty_kcal"), (int, float)):
        result["deltaE_rel"] = float(result["energy_penalty_kcal"])
        result["energy_penalty_rel"] = float(result["energy_penalty_kcal"])
        result["direction"] = result.get("direction") or "penalty"
        result["units"] = "rel"
        return
    if isinstance(result.get("energy_gain_kcal"), (int, float)):
        result["deltaE_rel"] = float(result["energy_gain_kcal"])
        result["energy_gain_rel"] = float(result["energy_gain_kcal"])
        result["direction"] = result.get("direction") or "gain"
        result["units"] = "rel"


def _format_mm_results(mm_results: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, result in mm_results.items():
        if not isinstance(result, dict):
            continue
        formatted[key] = {
            "deltaE_rel": result.get("deltaE_rel"),
            "units": result.get("units", "rel"),
            "method": result.get("method"),
            "direction": result.get("direction"),
        }
    return formatted


def _format_qm_results(qm_results: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, result in qm_results.items():
        if not isinstance(result, dict):
            continue
        formatted[key] = {
            "deltaE_kcal": result.get("delta_kcal"),
            "method": result.get("method"),
        }
    return formatted


def _get_uff_forcefield(
    mol: "Chem.Mol", warnings: List[str], label: str
) -> Optional["AllChem.ForceField"]:
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
    except Exception as exc:
        warnings.append(f"UFF failed for {label}: {exc}")
        return None
    if ff is None:
        warnings.append(f"UFF failed for {label}: no forcefield")
        return None
    return ff


def _add_probe_oh(
    work_mol: "Chem.RWMol",
    probe_idx: int,
    probe_pos: "rdGeometry.Point3D",
    direction: List[float],
) -> int:
    direction_norm = _normalize(direction) or [1.0, 0.0, 0.0]
    h_pos = rdGeometry.Point3D(
        probe_pos.x + direction_norm[0] * 0.96,
        probe_pos.y + direction_norm[1] * 0.96,
        probe_pos.z + direction_norm[2] * 0.96,
    )
    h_idx = work_mol.AddAtom(Chem.Atom("H"))
    work_mol.AddBond(probe_idx, h_idx, Chem.BondType.SINGLE)
    conf = work_mol.GetConformer()
    conf.SetAtomPosition(h_idx, h_pos)
    return h_idx


def _add_probe_nh3(
    work_mol: "Chem.RWMol",
    probe_idx: int,
    probe_pos: "rdGeometry.Point3D",
) -> List[int]:
    vectors = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ]
    h_indices: List[int] = []
    for vec in vectors:
        norm_vec = _normalize(vec) or [1.0, 0.0, 0.0]
        h_pos = rdGeometry.Point3D(
            probe_pos.x + norm_vec[0] * 1.01,
            probe_pos.y + norm_vec[1] * 1.01,
            probe_pos.z + norm_vec[2] * 1.01,
        )
        h_idx = work_mol.AddAtom(Chem.Atom("H"))
        work_mol.AddBond(probe_idx, h_idx, Chem.BondType.SINGLE)
        conf = work_mol.GetConformer()
        conf.SetAtomPosition(h_idx, h_pos)
        h_indices.append(h_idx)
    return h_indices


def _shift_attached_hydrogens(
    mol: "Chem.Mol",
    conf: "Chem.Conformer",
    atom_idx: int,
    delta: List[float],
) -> None:
    atom = mol.GetAtomWithIdx(atom_idx)
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() != "H":
            continue
        n_idx = neighbor.GetIdx()
        pos = conf.GetAtomPosition(n_idx)
        conf.SetAtomPosition(
            n_idx,
            rdGeometry.Point3D(
                pos.x + delta[0], pos.y + delta[1], pos.z + delta[2]
            ),
        )


def _prepare_uff_mol(mol: "Chem.Mol", warnings: List[str]) -> "Chem.Mol":
    """Ensure property cache is ready for UFF typing."""
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception as exc:
        warnings.append(f"UFF prep failed: {exc}")
    return mol


def _finalize_probe_mol(
    work_mol: "Chem.RWMol", warnings: List[str]
) -> "Chem.Mol":
    mol = work_mol.GetMol()
    return _prepare_uff_mol(mol, warnings)


def _attack_direction_from_carbonyl(
    c_pos: "rdGeometry.Point3D", o_pos: "rdGeometry.Point3D"
) -> List[float]:
    co_vec = [o_pos.x - c_pos.x, o_pos.y - c_pos.y, o_pos.z - c_pos.z]
    co_norm = _normalize(co_vec)
    if co_norm is None:
        return [1.0, 0.0, 0.0]
    perp = _cross(co_norm, [0.0, 1.0, 0.0])
    if _norm(perp) < 1e-6:
        perp = _cross(co_norm, [1.0, 0.0, 0.0])
    perp_norm = _normalize(perp)
    if perp_norm is None:
        return co_norm
    angle = math.radians(107.0)
    return _rodrigues_rotate(co_norm, perp_norm, angle)


def _rodrigues_rotate(vec: List[float], axis: List[float], angle: float) -> List[float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dot = _dot(axis, vec)
    cross = _cross(axis, vec)
    return [
        vec[0] * cos_a + cross[0] * sin_a + axis[0] * dot * (1.0 - cos_a),
        vec[1] * cos_a + cross[1] * sin_a + axis[1] * dot * (1.0 - cos_a),
        vec[2] * cos_a + cross[2] * sin_a + axis[2] * dot * (1.0 - cos_a),
    ]


def _normalize(vec: List[float]) -> Optional[List[float]]:
    norm = _norm(vec)
    if norm < 1e-9:
        return None
    return [component / norm for component in vec]


def _norm(vec: List[float]) -> float:
    return (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5


def _dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _identify_target_atoms(
    fragment_mol: "Chem.Mol", bond_indices: Tuple[int, int]
) -> Dict[str, Optional[int]]:
    """Identify carbonyl carbon and oxygen in fragment space."""
    a_idx, b_idx = bond_indices
    carbon_idx = None
    oxygen_idx = None
    atom_a = fragment_mol.GetAtomWithIdx(a_idx)
    atom_b = fragment_mol.GetAtomWithIdx(b_idx)
    if atom_a.GetSymbol() == "C":
        carbon_idx = a_idx
    if atom_b.GetSymbol() == "C" and carbon_idx is None:
        carbon_idx = b_idx
    if atom_a.GetSymbol() == "O":
        oxygen_idx = a_idx
    if atom_b.GetSymbol() == "O" and oxygen_idx is None:
        oxygen_idx = b_idx
    if carbon_idx is not None and oxygen_idx is None:
        carbon_atom = fragment_mol.GetAtomWithIdx(carbon_idx)
        for bond in carbon_atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(carbon_atom)
                if other.GetSymbol() == "O":
                    oxygen_idx = other.GetIdx()
                    break
    return {"carbon_idx": carbon_idx, "oxygen_idx": oxygen_idx}


def _mol_to_xyz(mol: "Chem.Mol") -> Optional[str]:
    try:
        return Chem.MolToXYZBlock(mol)
    except Exception:
        return None


def _probe_baseline_xyz(
    work_mol: "Chem.Mol",
    probe_idx: int,
    anchor_pos: "rdGeometry.Point3D",
    direction: List[float],
) -> Optional[str]:
    baseline_mol = Chem.RWMol(work_mol)
    conf = baseline_mol.GetConformer()
    old_pos = conf.GetAtomPosition(probe_idx)
    far_pos = rdGeometry.Point3D(
        anchor_pos.x + direction[0] * 10.0,
        anchor_pos.y + direction[1] * 10.0,
        anchor_pos.z + direction[2] * 10.0,
    )
    conf.SetAtomPosition(probe_idx, far_pos)
    delta = [far_pos.x - old_pos.x, far_pos.y - old_pos.y, far_pos.z - old_pos.z]
    _shift_attached_hydrogens(baseline_mol, conf, probe_idx, delta)
    return _mol_to_xyz(baseline_mol)


def _probe_pair_baseline_xyz(
    work_mol: "Chem.Mol",
    probe_indices: Tuple[int, int],
    anchor_pos: "rdGeometry.Point3D",
) -> Optional[str]:
    baseline_mol = Chem.RWMol(work_mol)
    conf = baseline_mol.GetConformer()
    pos1 = rdGeometry.Point3D(anchor_pos.x + 6.0, anchor_pos.y, anchor_pos.z)
    pos2 = rdGeometry.Point3D(anchor_pos.x, anchor_pos.y + 6.0, anchor_pos.z)
    old_pos1 = conf.GetAtomPosition(probe_indices[0])
    old_pos2 = conf.GetAtomPosition(probe_indices[1])
    conf.SetAtomPosition(probe_indices[0], pos1)
    conf.SetAtomPosition(probe_indices[1], pos2)
    delta1 = [pos1.x - old_pos1.x, pos1.y - old_pos1.y, pos1.z - old_pos1.z]
    delta2 = [pos2.x - old_pos2.x, pos2.y - old_pos2.y, pos2.z - old_pos2.z]
    _shift_attached_hydrogens(baseline_mol, conf, probe_indices[0], delta1)
    _shift_attached_hydrogens(baseline_mol, conf, probe_indices[1], delta2)
    return _mol_to_xyz(baseline_mol)


def _geometry_hash(xyz: str) -> str:
    return hashlib.sha256(xyz.encode("utf-8")).hexdigest()[:12]


def _xyz_atom_count(xyz: str) -> int:
    try:
        return int(xyz.splitlines()[0].strip())
    except Exception:
        return 0


def _mechanism_library_hash(library: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(library, sort_keys=True)
    except TypeError:
        payload = str(library)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _derive_mechanism_eligibility(
    bond360: Dict[str, Any],
    mm_results: Dict[str, Any],
    mechanism_library: Dict[str, Any],
) -> Dict[str, str]:
    eligibility: Dict[str, str] = {}
    primary_role = bond360.get("primary_role")
    for mechanism_id, mech in mechanism_library.items():
        mech_elig = mech.get("eligibility") or {}
        required_roles = mech_elig.get("required_bond_roles") or []
        forbidden_roles = mech_elig.get("forbidden_roles") or []
        if forbidden_roles and primary_role in forbidden_roles:
            eligibility[mechanism_id] = "REJECTED"
            continue
        if required_roles and primary_role not in required_roles:
            eligibility[mechanism_id] = "REJECTED"
            continue
        penalties: List[float] = []
        has_any = False
        prefix = f"{mechanism_id}__"
        for key, result in mm_results.items():
            if not key.startswith(prefix):
                continue
            has_any = True
            if isinstance(result, dict) and isinstance(result.get("deltaE_rel"), (int, float)):
                if result.get("direction") == "penalty":
                    penalties.append(float(result["deltaE_rel"]))
        if not has_any:
            eligibility[mechanism_id] = "REQUIRE_QUORUM"
            continue
        max_penalty = max(penalties) if penalties else 0.0
        eligibility[mechanism_id] = "APPROVED" if max_penalty < 10.0 else "REQUIRE_QUORUM"
    return eligibility


def _confidence_prior_from_bond(bond_type: Optional[str]) -> float:
    bond_type = (bond_type or "unknown").lower()
    if bond_type == "ester":
        return 0.7
    if bond_type == "amide":
        return 0.5
    if bond_type == "aryl_halide":
        return 0.3
    if bond_type == "ch":
        return 0.2
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
    return {"prefer": [], "discourage": [], "strength": 0.0}


def _cache_key(
    fragment: Dict[str, Any],
    bond360: Dict[str, Any],
    constraints: Dict[str, Any],
    mechanism_hash: str,
) -> Optional[str]:
    frag_smiles = fragment.get("fragment_smiles")
    if not frag_smiles:
        return None
    pH = constraints.get("ph_min")
    if pH is None:
        pH = constraints.get("ph_max")
    if pH is None:
        pH = 7.0
    pH_bucket = math.floor(float(pH) * 2.0) / 2.0
    temp_c = constraints.get("temperature_c")
    temp_bucket = int(round(float(temp_c) if isinstance(temp_c, (int, float)) else 25.0))
    raw = json.dumps(
        {
            "fragment_smiles": frag_smiles,
            "bond_role": bond360.get("primary_role") or "unknown",
            "pH_bucket": pH_bucket,
            "temperature_bucket": temp_bucket,
            "mechanism_hash": mechanism_hash,
            "schema_version": SCHEMA_VERSION,
            "cpt_impl_version": CPT_IMPL_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key_early(
    smiles: str, target_bond: str, constraints: Dict[str, Any], mechanism_hash: str
) -> Optional[str]:
    """Compute cache key from inputs only (no fragment needed)."""
    if not smiles or not target_bond:
        return None
    pH = constraints.get("ph_min") or constraints.get("ph_max") or 7.0
    pH_bucket = math.floor(float(pH) * 2.0) / 2.0
    temp = constraints.get("temperature_c", 25.0)
    temp_bucket = int(round(float(temp)))
    raw = json.dumps(
        {
            "smiles": smiles,
            "target_bond": target_bond,
            "pH_bucket": pH_bucket,
            "temperature_bucket": temp_bucket,
            "version": MODULE_MINUS1_VERSION,
            "mechanism_hash": mechanism_hash,
            "schema_version": SCHEMA_VERSION,
            "cpt_impl_version": CPT_IMPL_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_cache(payload: Dict[str, Any], expected_meta: Dict[str, Any]) -> bool:
    required = {
        "bond360_profile",
        "fragment",
        "cpt_scores",
        "mechanism_eligibility",
        "primary_constraint",
        "confidence_prior",
        "route_bias",
        "cache_meta",
    }
    if not all(key in payload for key in required):
        return False
    meta = payload.get("cache_meta") or {}
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            return False
    return True


def _build_output(
    bond360: Dict[str, Any],
    fragment: Dict[str, Any],
    cpt_scores: Dict[str, Any],
    mechanism_eligibility: Dict[str, str],
    primary_constraint: str,
    confidence_prior: float,
    route_bias: Dict[str, Any],
    cache_key: Optional[str],
    cache_hit: bool,
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "bond360_profile": bond360,
        "fragment": fragment,
        "cpt_scores": cpt_scores,
        "mechanism_eligibility": mechanism_eligibility,
        "primary_constraint": primary_constraint,
        "confidence_prior": round(float(confidence_prior), 3),
        "route_bias": route_bias,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "warnings": warnings,
        "errors": errors,
    }


def _empty_bond360(target_spec: TargetSpec) -> Dict[str, Any]:
    role_info = _bond_role_from_token(target_spec.canonical_token or target_spec.token)
    return {
        "bond_length": None,
        "bond_order": None,
        "bond_type": _bond_type_from_token(target_spec.token) or "unknown",
        "is_conjugated": False,
        "is_aromatic": False,
        "in_ring": False,
        "bond_indices": [],
        "formal_charge_pair": [0, 0],
        "partial_charges": [None, None],
        "dipole_proxy": None,
        "steric_accessibility": DEFAULT_STERIC_ACCESSIBILITY,
        "topological_depth": None,
        "neighbor_hetero_atoms": 0,
        "rdkit_available": _RDKIT_AVAILABLE,
        "bond_roles": [role_info["role"]] if role_info else [],
        "primary_role": role_info["role"] if role_info else None,
        "functional_group_atoms": [],
        "attack_sites": {},
    }


def _empty_fragment() -> Dict[str, Any]:
    return {
        "fragment_smiles": None,
        "fragment_smiles_ordered": None,
        "net_charge": 0,
        "spin_multiplicity": 1,
        "cut_bonds": [],
        "cap_map": {},
        "atom_map_parent_to_fragment": {},
        "notes": [],
    }


def _empty_cpt(reason: str) -> Dict[str, Any]:
    return {
        "status": "no_cpts" if reason else "skipped",
        "mm_results": {},
        "qm_results": {},
        "mm": {},
        "qm": {},
        "conformer_count": 0,
        "variance_estimate": 0.0,
        "triggered_qm": False,
        "note": reason,
    }
