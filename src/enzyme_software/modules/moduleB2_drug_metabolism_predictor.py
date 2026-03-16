from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

from enzyme_software.context import OperationalConstraints
from enzyme_software.pipeline import run_pipeline
from enzyme_software.modules.moduleB2_site_enumeration import enumerate_metabolism_targets


_SITE_BASE_REACTIVITY: Dict[str, float] = {
    "benzylic_ch": 0.95,
    "allylic_ch": 0.92,
    "alpha_hetero_ch": 0.88,
    "aliphatic_tertiary_ch": 0.74,
    "aliphatic_secondary_ch": 0.68,
    "aliphatic_primary_ch": 0.55,
    "aromatic_ch": 0.24,
    "o_demethyl": 0.90,
    "o_dealkyl": 0.82,
    "n_demethyl": 0.84,
    "n_dealkyl": 0.78,
}


def _target_bond_for_site(site: Dict[str, Any]) -> str:
    cls = str(site.get("site_class") or "")
    if cls.startswith("o_"):
        return "C-O"
    if cls.startswith("n_"):
        return "C-N"
    return "C-H"


def _make_constraints(payload: Optional[Dict[str, Any]]) -> OperationalConstraints:
    data = payload or {}
    return OperationalConstraints(
        ph_min=data.get("ph_min"),
        ph_max=data.get("ph_max"),
        temperature_c=data.get("temperature_c"),
        metals_allowed=data.get("metals_allowed"),
        oxidation_allowed=data.get("oxidation_allowed"),
        host=data.get("host"),
    )


def _run_module_m1_m0(smiles: str, target_bond: str, constraints: OperationalConstraints) -> Dict[str, Any]:
    from enzyme_software.modules.module_minus1_reactivity_hub import ModuleMinus1SRE
    from enzyme_software.modules.module0_strategy_router import Module0StrategyRouter

    try:
        ctx = run_pipeline(
            smiles=smiles,
            target_bond=target_bond,
            constraints=constraints,
            modules=[ModuleMinus1SRE(), Module0StrategyRouter()],
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "decision": "ERROR",
            "route": None,
            "enzyme_family": "unknown",
            "k_eff_s_inv": 0.0,
            "p_convert": 0.0,
            "route_confidence": 0.0,
            "ctx": None,
            "error": str(exc),
        }
    jc = ctx.data.get("job_card") or {}
    summary = ctx.data.get("pipeline_summary") or {}
    results = summary.get("results") or {}
    energy = jc.get("energy_ledger") or {}
    confidence = jc.get("confidence") or {}
    return {
        "decision": results.get("decision") or jc.get("decision"),
        "route": results.get("route") or (jc.get("mechanism_route") or {}).get("primary"),
        "enzyme_family": (jc.get("biology_audit") or {}).get("enzyme_family"),
        "k_eff_s_inv": energy.get("k_eff_s_inv"),
        "p_convert": energy.get("p_success_horizon"),
        "route_confidence": confidence.get("route"),
        "ctx": ctx,
    }


def _k_norm(k_eff: Any) -> float:
    if not isinstance(k_eff, (int, float)):
        return 0.0
    k = max(1e-9, float(k_eff))
    # map ~1e-3..1e3 into 0..1 smoothly
    return max(0.0, min(1.0, (math.log10(k) + 3.0) / 6.0))


def _isoform_bias(isoform_hint: Optional[str], site: Dict[str, Any], pred: Dict[str, Any]) -> float:
    if not isoform_hint:
        return 0.0
    iso = str(isoform_hint).upper().strip()
    route = str(pred.get("route") or "").lower()
    family = str(pred.get("enzyme_family") or "").lower()
    rxn = str(site.get("reaction_class") or "")
    bias = 0.0

    if iso.startswith("CYP"):
        if "p450" in route or "cytochrome" in family:
            bias += 0.05
        else:
            bias -= 0.02
    if iso == "CYP2D6" and rxn in {"o_demethylation", "n_demethylation", "o_dealkylation", "n_dealkylation"}:
        bias += 0.04
    if iso == "CYP1A2" and ("aromatic" in rxn or "n_demethylation" == rxn):
        bias += 0.03
    if iso == "CYP2C9" and "benzylic" in rxn:
        bias += 0.02
    return bias


def _site_score(site: Dict[str, Any], pred: Dict[str, Any], isoform_hint: Optional[str]) -> Tuple[float, Dict[str, float]]:
    base = float(_SITE_BASE_REACTIVITY.get(str(site.get("site_class") or ""), 0.45))
    p_convert = float(pred.get("p_convert") or 0.0) if isinstance(pred.get("p_convert"), (int, float)) else 0.0
    route_conf = float(pred.get("route_confidence") or 0.0) if isinstance(pred.get("route_confidence"), (int, float)) else 0.0
    k_norm = _k_norm(pred.get("k_eff_s_inv"))

    score = 0.50 * base + 0.20 * p_convert + 0.20 * k_norm + 0.10 * route_conf
    if str(site.get("site_class") or "") == "aromatic_ch":
        score -= 0.08
    score += _isoform_bias(isoform_hint, site, pred)
    score = max(0.0, min(1.0, score))
    return score, {
        "base": round(base, 4),
        "p_convert": round(p_convert, 4),
        "k_norm": round(k_norm, 4),
        "route_conf": round(route_conf, 4),
    }


def predict_drug_metabolism(
    smiles: str,
    isoform_hint: str | None = None,
    topk: int = 5,
    condition_profile: dict | None = None,
    constraints: dict | None = None,
) -> Dict[str, Any]:
    """Predict ranked metabolism sites for a drug-like substrate."""
    if Chem is None:
        return {
            "ranked_sites": [],
            "summary": {"best_site": None, "best_reaction": None, "confidence": 0.0},
            "error": "rdkit_unavailable",
        }

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {
            "ranked_sites": [],
            "summary": {"best_site": None, "best_reaction": None, "confidence": 0.0},
            "error": "invalid_smiles",
        }

    site_candidates = enumerate_metabolism_targets(mol)
    if not site_candidates or site_candidates[0].get("site_id") == "none":
        return {
            "ranked_sites": [],
            "summary": {"best_site": None, "best_reaction": None, "confidence": 0.0},
            "debug": {"candidate_count": 0},
        }

    op_constraints = _make_constraints(constraints)
    pipeline_cache: Dict[str, Dict[str, Any]] = {}
    ranked: List[Dict[str, Any]] = []

    for site in site_candidates:
        target_bond = _target_bond_for_site(site)
        if target_bond not in pipeline_cache:
            pipeline_cache[target_bond] = _run_module_m1_m0(
                smiles=smiles,
                target_bond=target_bond,
                constraints=op_constraints,
            )
        pred = pipeline_cache[target_bond]
        score, components = _site_score(site, pred, isoform_hint=isoform_hint)

        ranked.append(
            {
                "site_id": site.get("site_id"),
                "site_type": site.get("site_type"),
                "atom_indices": list(site.get("atom_indices") or []),
                "bond_indices": list(site.get("bond_indices") or []),
                "anchor_atom_indices": list(site.get("anchor_atom_indices") or []),
                "site_class": site.get("site_class"),
                "reaction_class": site.get("reaction_class"),
                "score": round(score, 6),
                "score_components": components,
                "pipeline_target_bond": target_bond,
                "predicted_route": pred.get("route"),
                "predicted_family": pred.get("enzyme_family"),
                "predicted_kcat_s_inv": pred.get("k_eff_s_inv"),
                "predicted_p_convert": pred.get("p_convert"),
                "route_confidence": pred.get("route_confidence"),
                "decision": pred.get("decision"),
            }
        )

    ranked.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("site_id") or "")))
    topk_n = max(1, int(topk or 5))
    out = ranked[:topk_n]
    best = out[0] if out else None

    return {
        "ranked_sites": out,
        "summary": {
            "best_site": best.get("site_id") if best else None,
            "best_reaction": best.get("reaction_class") if best else None,
            "confidence": best.get("score") if best else 0.0,
            "isoform_hint": isoform_hint,
        },
        "debug": {
            "candidate_count": len(site_candidates),
            "pipeline_targets_used": sorted(pipeline_cache.keys()),
            "condition_profile": condition_profile or {},
        },
    }
