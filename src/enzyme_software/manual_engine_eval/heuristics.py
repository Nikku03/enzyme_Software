from __future__ import annotations

from typing import Any, Dict, List, Optional


def reactive_site_heuristic(case, module_minus1_output: Dict[str, Any]) -> Dict[str, Any]:
    resolved = module_minus1_output.get("resolved_target") or {}
    candidates = (
        resolved.get("candidate_bonds")
        or resolved.get("candidate_attack_sites")
        or []
    )
    ranked = []
    for candidate in candidates:
        score = 0.0
        bond_class = str(candidate.get("bond_class") or candidate.get("class") or "").lower()
        bde = candidate.get("bde_kj_mol")
        if isinstance(bde, (int, float)):
            score -= float(bde)
        if "benzylic" in bond_class:
            score += 50.0
        if "allylic" in bond_class:
            score += 40.0
        if "alpha" in bond_class:
            score += 25.0
        if "aryl" in bond_class:
            score -= 30.0
        atom_indices = candidate.get("atom_indices") or candidate.get("site_indices") or []
        ranked.append({"site": atom_indices, "score": score, "bond_class": bond_class})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {"top_sites": [item["site"] for item in ranked[:3]], "details": ranked[:5]}


def route_heuristic(case, module_minus1_output: Dict[str, Any]) -> Dict[str, Any]:
    bond = str(case.target_bond).lower()
    if "ester" in bond:
        route = "serine_hydrolase"
    elif "amide" in bond:
        route = "amidase"
    elif any(token in bond for token in ["c-cl", "c-br", "c-i"]):
        route = "haloalkane_dehalogenase"
    elif any(token in bond for token in ["c-h", "o-h", "n-h"]):
        route = "p450"
    else:
        route = "unknown"
    return {"route_family": route}


def scaffold_heuristic(module1_output: Dict[str, Any]) -> Dict[str, Any]:
    ranked = module1_output.get("ranked_scaffolds") or []
    if not ranked:
        return {"top_scaffold": None}
    best = max(
        ranked,
        key=lambda item: float(
            (item.get("module1_confidence") or {}).get("total")
            or item.get("feasibility_score")
            or 0.0
        ),
    )
    return {"top_scaffold": best.get("scaffold_id")}
