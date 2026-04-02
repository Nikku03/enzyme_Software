from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

from enzyme_software.pipeline import run_pipeline
from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.features.route_prior import route_posteriors_to_cyp_prior
from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol
from enzyme_software.liquid_nn_v2.utils.mol_provenance import log_mol_provenance_event, mol_provenance_context


DEFAULT_MANUAL_FEATURE_DIM = 32
_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "manual_engine"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION = 3
MANUAL_ENGINE_STATUS_NAMES = ("manual", "fallback", "missing")


def manual_engine_status_vector(status: Optional[str]) -> np.ndarray:
    key = str(status or "").strip().lower()
    vector = np.zeros((len(MANUAL_ENGINE_STATUS_NAMES),), dtype=np.float32)
    index = {
        "manual": 0,
        "fallback": 1,
    }.get(key, 2)
    vector[index] = 1.0
    return vector


def infer_target_bond(smiles: str, *, allow_partial_sanitize: bool = True, allow_aggressive_repair: bool = False) -> str:
    if Chem is None:
        return "C-H"
    with mol_provenance_context(module_triggered="manual-engine bridge", source_category="manual-engine bridge", parsed_smiles=smiles):
        prep = prepare_mol(
            smiles,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    if prep.mol is None:
        return "C-H"
    mol = prep.mol
    ester = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    if ester is not None and mol.HasSubstructMatch(ester):
        return "ester"
    if amide is not None and mol.HasSubstructMatch(amide):
        return "amide"
    return "C-H"


def _cache_path(smiles: str, target_bond: str, cache_dir: Optional[str] = None) -> Path:
    base = Path(cache_dir) if cache_dir else _CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps({"smiles": smiles, "target_bond": target_bond, "cache_version": _CACHE_VERSION}, sort_keys=True).encode()
    ).hexdigest()[:20]
    return base / f"{key}.json"


def _run_manual_engine(smiles: str, target_bond: str) -> Dict[str, object]:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        ctx = run_pipeline(smiles=smiles, target_bond=target_bond)
    stderr_text = stderr.getvalue().strip()
    if stderr_text:
        log_mol_provenance_event(
            stage="manual_engine_internal",
            status="rdkit_error",
            parsed_smiles=smiles,
            canonical_smiles=smiles,
            error=None,
            rdkit_message=stderr_text,
            module_triggered="manual-engine internal module",
            source_category="manual-engine internal module",
            extra={"target_bond": target_bond},
        )
    return {
        "module_minus1": ctx.data.get("module_minus1") or {},
        "module0": ctx.data.get("module0_strategy_router") or {},
        "job_card": ctx.data.get("job_card") or {},
        "pipeline_summary": ctx.data.get("pipeline_summary") or {},
    }


def _fallback_manual_features(
    smiles: str,
    target_bond: str,
    *,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
) -> Optional[Dict[str, object]]:
    if Chem is None:
        return None
    with mol_provenance_context(module_triggered="manual-engine bridge", source_category="manual-engine bridge", parsed_smiles=smiles):
        prep = prepare_mol(
            smiles,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    if prep.mol is None:
        return None
    mol = prep.mol

    benzylic = Chem.MolFromSmarts("[CH3,CH2,CH1][c]")
    allylic = Chem.MolFromSmarts("[CH3,CH2,CH1][C]=[C]")
    alpha_hetero = Chem.MolFromSmarts("[CH3,CH2,CH1][N,O,S]")
    patterns = [
        ("ch__benzylic", benzylic, 360.0, 0.90, 0.80),
        ("ch__allylic", allylic, 365.0, 0.82, 0.72),
        ("ch__alpha_hetero", alpha_hetero, 380.0, 0.74, 0.60),
    ]
    seen = set()
    candidate_sites = []
    for bond_class, pattern, bde, score, radical in patterns:
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            atom_idx = int(match[0])
            if atom_idx in seen:
                continue
            seen.add(atom_idx)
            candidate_sites.append(
                {
                    "heavy_atom_index": atom_idx,
                    "atom_indices": [atom_idx],
                    "bond_class": bond_class,
                    "bde_kj_mol": bde,
                    "score": score,
                    "radical_stability": radical,
                }
            )
    if not candidate_sites:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            atom_idx = int(atom.GetIdx())
            candidate_sites.append(
                {
                    "heavy_atom_index": atom_idx,
                    "atom_indices": [atom_idx],
                    "bond_class": "ch__other",
                    "bde_kj_mol": 410.0,
                    "score": 0.25,
                    "radical_stability": 0.20,
                }
            )

    attack_sites = {}
    if candidate_sites:
        attack_sites["fallback_site"] = int(candidate_sites[0]["heavy_atom_index"])

    route_posteriors = {
        "p450": 0.70,
        "monooxygenase": 0.15,
        "oxidoreductase": 0.10,
        "serine_hydrolase": 0.05 if target_bond in {"ester", "amide"} else 0.05,
    }
    return {
        "smiles": smiles,
        "target_bond": target_bond,
        "candidate_sites": candidate_sites,
        "bond360_profile": {"attack_sites": attack_sites},
        "mechanism_eligibility": {"p450_oxidation": "SUPPORTED"},
        "route_posteriors": route_posteriors,
        "selected_route": "p450",
        "route_confidence": 0.35,
        "top_routes": [{"route_id": "p450", "score": 0.7, "posterior": 0.7}],
        "route_gap": 0.10,
        "ambiguity_flag": True,
        "fallback_used": True,
        "manual_feature_status": "fallback",
    }


def _normalize_manual_result(
    payload: Dict[str, object],
    *,
    smiles: str,
    target: str,
    fallback_used: bool,
    manual_feature_status: str,
) -> Dict[str, object]:
    module_minus1 = payload.get("module_minus1") or {}
    job_card = payload.get("job_card") or {}
    route_posteriors = job_card.get("route_posteriors") or []
    if isinstance(route_posteriors, list):
        route_posteriors = {
            str(item.get("route_id") or item.get("route_family") or item.get("route") or f"route_{idx}"): float(item.get("posterior", 0.0))
            for idx, item in enumerate(route_posteriors)
            if isinstance(item, dict)
        }
    return {
        "smiles": smiles,
        "target_bond": target,
        "candidate_sites": (module_minus1.get("resolved_target") or {}).get("candidate_bonds") or [],
        "bond360_profile": module_minus1.get("bond360_profile") or {},
        "mechanism_eligibility": module_minus1.get("mechanism_eligibility") or {},
        "route_posteriors": route_posteriors,
        "selected_route": job_card.get("chosen_route"),
        "route_confidence": job_card.get("confidence_calibrated") or (job_card.get("confidence") or {}).get("route", 0.0),
        "top_routes": job_card.get("top_routes") or [],
        "route_gap": job_card.get("route_gap", 0.0),
        "ambiguity_flag": bool(job_card.get("ambiguity_flag", False)),
        "fallback_used": bool(job_card.get("fallback_used", False) or fallback_used),
        "manual_feature_status": manual_feature_status,
    }


def extract_module_minus1_features(
    smiles: str,
    target_bond: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
) -> Optional[Dict]:
    with mol_provenance_context(module_triggered="manual-engine bridge", source_category="manual-engine bridge", parsed_smiles=smiles):
        prep = prepare_mol(
            smiles,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    if prep.mol is None:
        return None
    canonical_smiles = prep.canonical_smiles or str(smiles)
    target = str(
        target_bond
        or infer_target_bond(
            canonical_smiles,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    )
    cache_path = _cache_path(canonical_smiles, target, cache_dir=cache_dir)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    try:
        with mol_provenance_context(
            module_triggered="manual-engine bridge",
            source_category="manual-engine bridge",
            parsed_smiles=canonical_smiles,
        ):
            payload = _run_manual_engine(canonical_smiles, target)
        result = _normalize_manual_result(
            payload,
            smiles=canonical_smiles,
            target=target,
            fallback_used=False,
            manual_feature_status="manual",
        )
    except KeyError as exc:
        if "Unknown atom_id" in str(exc):
            log_mol_provenance_event(
                stage="manual_engine_bridge",
                status="manual_engine_error",
                parsed_smiles=canonical_smiles,
                canonical_smiles=canonical_smiles,
                error=str(exc),
                rdkit_message=None,
                module_triggered="manual-engine bridge",
                source_category="manual-engine bridge",
                extra={"target_bond": target, "error_type": "Unknown atom_id"},
            )
            result = _fallback_manual_features(
                canonical_smiles,
                target,
                allow_partial_sanitize=allow_partial_sanitize,
                allow_aggressive_repair=allow_aggressive_repair,
            )
        else:
            print(f"Manual engine error for {canonical_smiles}: {exc}")
            result = None
    except Exception as exc:
        log_mol_provenance_event(
            stage="manual_engine_bridge",
            status="manual_engine_error",
            parsed_smiles=canonical_smiles,
            canonical_smiles=canonical_smiles,
            error=str(exc),
            rdkit_message=None,
            module_triggered="manual-engine bridge",
            source_category="manual-engine bridge",
            extra={"target_bond": target},
        )
        print(f"Manual engine error for {canonical_smiles}: {exc}")
        result = _fallback_manual_features(
            canonical_smiles,
            target,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    if result is None:
        return None
    try:
        cache_path.write_text(json.dumps(result, indent=2))
    except Exception:
        pass
    return result


if TORCH_AVAILABLE:
    def manual_features_to_tensor(
        manual_features: Dict,
        num_atoms: int,
        feature_dim: int = DEFAULT_MANUAL_FEATURE_DIM,
    ) -> torch.Tensor:
        features = torch.zeros(num_atoms, feature_dim, dtype=torch.float32)
        if manual_features is None:
            return features

        candidate_sites = manual_features.get("candidate_sites") or []
        bond360 = manual_features.get("bond360_profile") or {}
        attack_sites = bond360.get("attack_sites") or {}
        mech_elig = manual_features.get("mechanism_eligibility") or {}
        class_to_idx = {
            "benzylic": 2,
            "allylic": 3,
            "alpha_hetero": 4,
            "tertiary": 5,
            "secondary": 6,
            "primary": 7,
            "aryl": 8,
            "vinyl": 9,
        }
        mech_to_idx = {
            "p450_oxidation": 12,
            "monooxygenase": 13,
            "radical_transfer": 14,
            "oxidoreductase": 15,
            "sn2_displacement": 16,
            "serine_hydrolase": 17,
            "metallo_hydrolase": 18,
        }
        for cand in candidate_sites:
            if not isinstance(cand, dict):
                continue
            atom_idx = cand.get("heavy_atom_index")
            if not isinstance(atom_idx, int):
                atom_indices = cand.get("atom_indices") or []
                atom_idx = atom_indices[0] if atom_indices else None
            if not isinstance(atom_idx, int) or not (0 <= atom_idx < num_atoms):
                continue
            bde = cand.get("bde_kj_mol")
            if isinstance(bde, (int, float)):
                bde_norm = max(0.0, min(1.0, (float(bde) - 300.0) / 200.0))
                features[atom_idx, 0] = bde_norm
                features[atom_idx, 1] = 1.0 - bde_norm
            score = cand.get("score")
            if isinstance(score, (int, float)):
                features[atom_idx, 10] = max(0.0, min(1.0, float(score)))
            bond_class = str(cand.get("bond_class") or "").lower()
            for key, index in class_to_idx.items():
                if key in bond_class:
                    features[atom_idx, index] = 1.0
                    break
            radical = cand.get("radical_stability")
            if isinstance(radical, (int, float)):
                features[atom_idx, 11] = float(radical)

        for role, atom_idx in attack_sites.items():
            if isinstance(atom_idx, int) and 0 <= atom_idx < num_atoms:
                features[atom_idx, 19] = 1.0

        for mech, status in mech_elig.items():
            idx = mech_to_idx.get(str(mech))
            if idx is None:
                continue
            value = 1.0 if str(status).upper() in {"ELIGIBLE", "SUPPORTED", "PASS"} else 0.25
            features[:, idx] = torch.maximum(features[:, idx], torch.full((num_atoms,), value))

        features[:, 24] = float(manual_features.get("route_confidence") or 0.0)
        features[:, 25] = float(manual_features.get("route_gap") or 0.0)
        features[:, 26] = float(bool(manual_features.get("ambiguity_flag")))
        features[:, 27] = float(bool(manual_features.get("fallback_used")))
        features[:, 28] = min(1.0, float(len(candidate_sites)) / max(1.0, float(num_atoms)))
        status = str(manual_features.get("manual_feature_status") or "")
        features[:, 29] = float(status == "manual")
        features[:, 30] = float(status == "fallback")
        return features


    def build_manual_engine_bundle(
        smiles: str,
        *,
        num_atoms: Optional[int] = None,
        target_bond: Optional[str] = None,
        cyp_order: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        feature_dim: int = DEFAULT_MANUAL_FEATURE_DIM,
        allow_partial_sanitize: bool = True,
        allow_aggressive_repair: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        manual_features = extract_module_minus1_features(
            smiles,
            target_bond=target_bond,
            cache_dir=cache_dir,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
        if manual_features is None:
            return None
        if num_atoms is None:
            if Chem is None:
                return None
            with mol_provenance_context(module_triggered="manual-engine bridge", source_category="manual-engine bridge", parsed_smiles=smiles):
                prep = prepare_mol(
                    smiles,
                    allow_partial_sanitize=allow_partial_sanitize,
                    allow_aggressive_repair=allow_aggressive_repair,
                )
            if prep.mol is None:
                return None
            num_atoms = prep.mol.GetNumAtoms()

        atom_features = manual_features_to_tensor(manual_features, num_atoms=num_atoms, feature_dim=feature_dim)
        atom_prior_logits = atom_features[:, 10:11] + 0.5 * atom_features[:, 1:2]

        route_prior = route_posteriors_to_cyp_prior(
            manual_features.get("route_posteriors"),
            cyp_order=cyp_order or list(MAJOR_CYP_CLASSES),
        )
        cyp_prior_logits = torch.log(route_prior.clamp(min=1.0e-6))

        mol_features = torch.zeros((1, 8), dtype=torch.float32)
        mol_features[0, 0] = float(manual_features.get("route_confidence") or 0.0)
        mol_features[0, 1] = float(manual_features.get("route_gap") or 0.0)
        mol_features[0, 2] = float(bool(manual_features.get("ambiguity_flag")))
        mol_features[0, 3] = float(bool(manual_features.get("fallback_used")))
        mol_features[0, 4] = float(len(manual_features.get("candidate_sites") or []))
        mol_features[0, 5] = float(route_prior.max().item())
        mol_features[0, 6] = float(-(route_prior * torch.log(route_prior.clamp(min=1.0e-6))).sum().item())
        mol_features[0, 7] = float(len(manual_features.get("top_routes") or []))

        return {
            "manual_engine_atom_features": atom_features.cpu().numpy().astype(np.float32),
            "manual_engine_mol_features": mol_features.cpu().numpy().astype(np.float32),
            "manual_engine_atom_prior_logits": atom_prior_logits.cpu().numpy().astype(np.float32),
            "manual_engine_cyp_prior_logits": cyp_prior_logits.unsqueeze(0).cpu().numpy().astype(np.float32),
            "manual_engine_route_prior": route_prior.unsqueeze(0).cpu().numpy().astype(np.float32),
            "manual_engine_status": manual_engine_status_vector(manual_features.get("manual_feature_status")),
        }


    def attach_manual_engine_features_to_graph(
        graph,
        *,
        target_bond: Optional[str] = None,
        cyp_order: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        allow_partial_sanitize: bool = True,
        allow_aggressive_repair: bool = False,
    ):
        bundle = build_manual_engine_bundle(
            graph.smiles,
            num_atoms=int(graph.num_atoms),
            target_bond=target_bond,
            cyp_order=cyp_order,
            cache_dir=cache_dir,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
        if bundle is None:
            graph.manual_engine_status = manual_engine_status_vector("missing")
            return graph
        for key, value in bundle.items():
            setattr(graph, key, value)
        return graph
else:  # pragma: no cover
    def manual_features_to_tensor(*args, **kwargs):
        require_torch()

    def build_manual_engine_bundle(*args, **kwargs):
        require_torch()

    def attach_manual_engine_features_to_graph(*args, **kwargs):
        require_torch()
