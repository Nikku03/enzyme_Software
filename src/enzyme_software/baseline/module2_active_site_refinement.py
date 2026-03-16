from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from enzyme_software.context import PipelineContext
from enzyme_software.modules.base import BaseModule

FAIL_NO_TOP_SCAFFOLDS = "FAIL_NO_TOP_SCAFFOLDS"
FAIL_NO_TARGET = "FAIL_NO_TARGET"
FAIL_DECISION_NOT_GO = "FAIL_DECISION_NOT_GO"


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


class Module2ActiveSiteRefinement(BaseModule):
    name = "Module 2 - Active-Site Refinement + Variant Proposal"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        job_card = ctx.data.get("job_card") or {}
        module0_job_card = (ctx.data.get("module0_strategy_router") or {}).get("job_card")
        warnings: List[str] = []
        if module0_job_card and module0_job_card != job_card:
            warnings.append(
                "W_JOB_CARD_MISMATCH: module0_strategy_router.job_card differs from data.job_card; using data.job_card."
            )

        module1 = ctx.data.get("module1_topogate") or {}
        module1_handoff = module1.get("module2_handoff") or {}
        top_scaffolds = module1_handoff.get("top_scaffolds") or []

        result = run_module2(job_card, top_scaffolds)
        if warnings:
            result["warnings"] = list(dict.fromkeys(result.get("warnings", []) + warnings))
        ctx.data["module2_active_site_refinement"] = result
        return ctx


def run_module2(job_card: Dict[str, Any], top_scaffolds: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not top_scaffolds:
        return _fail_module2(FAIL_NO_TOP_SCAFFOLDS, errors, warnings)

    if job_card.get("decision") != "GO":
        return _fail_module2(FAIL_DECISION_NOT_GO, errors, warnings)

    resolved = job_card.get("resolved_target") or {}
    selected_bond = resolved.get("selected_bond") or {}
    if not selected_bond:
        return _fail_module2(FAIL_NO_TARGET, errors, warnings)

    reaction_intent = job_card.get("reaction_intent") or {}
    intent_type = reaction_intent.get("intent_type")
    if intent_type not in {"hydrolysis", "deprotection", "fragment_generation", "reagent_generation"}:
        warnings.append("W_INTENT_UNRECOGNIZED: intent type not explicitly supported.")

    scaffold_rankings = _rank_scaffolds(top_scaffolds)
    selected = scaffold_rankings[0]
    selection_explain = _selection_explain(scaffold_rankings)

    pocket_check = _pocket_reality_check(selected.get("pdb_path"))

    objective = "Improve retention without breaking access or reach."
    variant_set = _build_variants(job_card, selected, objective, pocket_check)
    best_variant_policy = "ranked_best_variant"
    best_variant = None
    if variant_set:
        best_variant = min(variant_set, key=lambda item: item.get("rank", 999))

    module3_handoff = _module3_handoff(
        job_card, selected, best_variant, variant_set, best_variant_policy
    )

    return {
        "status": "PASS",
        "selected_scaffold": selected,
        "selection_explain": selection_explain,
        "objective": objective,
        "pocket_check": pocket_check,
        "variant_set": variant_set,
        "module3_handoff": module3_handoff,
        "best_variant_policy": best_variant_policy,
        "best_variant": best_variant,
        "scaffold_rankings": scaffold_rankings,
        "warnings": warnings,
        "errors": errors,
    }


def _rank_scaffolds(scaffolds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for scaffold in scaffolds:
        scores = scaffold.get("scores") or {}
        retention_metrics = scaffold.get("retention_metrics") or {}
        total = scores.get("total") or 0.0
        penalty, penalty_reasons = _retention_penalty(retention_metrics)
        adjusted = max(0.0, float(total) - penalty)
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
                "adjusted_score": round(adjusted, 3),
                "retention_penalty": round(penalty, 3),
                "retention_penalty_reasons": penalty_reasons,
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
        f"Selected {top.get('scaffold_id')} with adjusted score {top.get('adjusted_score')}."
    ]
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
    retention_metrics = selected.get("retention_metrics") or {}
    retention_flag = retention_metrics.get("retention_risk_flag")
    warning_codes = retention_metrics.get("warning_codes") or []
    tunnel_metrics = selected.get("tunnel_metrics") or {}
    long_tunnel = (tunnel_metrics.get("path_length") or 0.0) >= 18.0
    retention_weak = retention_flag in {"HIGH", "MEDIUM"} or "WARN_RETENTION_WEAK_BINDING" in warning_codes
    can_localize = pocket_check.get("status") == "available"

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
        )
    )

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
                    estimated_effects={"reach": 0.06, "retention": 0.0, "access": 0.0},
                    score=base_score + 0.06,
                    category="mechanism_alignment",
                    requires_structural_localization=not can_localize,
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
        clamp_boost = 0.08 if retention_flag == "HIGH" else 0.06
        if "WARN_RETENTION_WEAK_BINDING" in warning_codes:
            clamp_boost += 0.03
        variants.append(
            Variant(
                variant_id="V2",
                label="Hydrophobic clamp",
                description="Add aromatic/hydrophobic packing to stabilize the aromatic ring.",
                mutations=clamp_mutations,
                rationale="Retention weakness; add hydrophobic wall for ring packing.",
                estimated_effects={"reach": -0.01, "retention": clamp_boost, "access": -0.01},
                score=base_score + clamp_boost - 0.01,
                category="retention_clamp",
                requires_structural_localization=not can_localize,
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
                estimated_effects={"reach": 0.0, "retention": 0.06, "access": -0.01},
                score=base_score + 0.05,
                category="polar_anchor",
                requires_structural_localization=not can_localize,
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
                estimated_effects={"reach": 0.04, "retention": 0.02, "access": -0.01},
                score=base_score + 0.05,
                category="oxyanion_hole",
                requires_structural_localization=not can_localize,
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
                estimated_effects={"reach": -0.01, "retention": 0.05, "access": -0.005},
                score=base_score + 0.045,
                category="access_preserving_clamp",
                requires_structural_localization=not can_localize,
            )
        )

    serialized = [variant.__dict__ for variant in variants]
    serialized.sort(
        key=lambda item: _variant_rank_key(item, retention_weak),
        reverse=True,
    )
    for idx, variant in enumerate(serialized, start=1):
        variant["rank"] = idx
        variant["objective"] = objective
        variant["nucleophile_geometry"] = nucleophile_geometry
    return serialized


def _pick_residue(residues: List[str], preferred_prefixes: List[str]) -> Optional[str]:
    if not residues:
        return None
    for prefix in preferred_prefixes:
        for residue in residues:
            if residue.lower().startswith(prefix.lower()):
                return residue
    return residues[0]


def _variant_rank_key(variant: Dict[str, Any], retention_weak: bool) -> float:
    score = float(variant.get("score") or 0.0)
    category = variant.get("category") or "other"
    if not retention_weak:
        return score
    if category in {
        "retention_clamp",
        "polar_anchor",
        "oxyanion_hole",
        "access_preserving_clamp",
    }:
        return score + 0.05
    if category == "mechanism_alignment":
        return score - 0.02
    return score - 0.04


def _module3_handoff(
    job_card: Dict[str, Any],
    selected: Dict[str, Any],
    best_variant: Optional[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    best_variant_policy: str,
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
        "nucleophile_geometry": selected.get("reach_summary", {}).get("nucleophile_geometry"),
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


def _fail_module2(reason: str, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    return {
        "status": "FAIL",
        "halt_reason": reason,
        "selected_scaffold": None,
        "selection_explain": None,
        "variant_set": [],
        "module3_handoff": {},
        "warnings": warnings,
        "errors": errors,
    }
