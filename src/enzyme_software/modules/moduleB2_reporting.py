from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_metabolism_prediction_report(
    *,
    drug_key: Optional[str],
    drug_name: Optional[str],
    primary_isoform: Optional[str],
    prediction: Dict[str, Any],
    top_n: int = 3,
) -> Dict[str, Any]:
    ranked = list(prediction.get("ranked_sites") or [])
    top_n_rows = ranked[: max(1, int(top_n))]
    best = top_n_rows[0] if top_n_rows else None

    if best is None:
        kcat_band = "unknown"
    else:
        kcat = best.get("predicted_kcat_s_inv")
        if isinstance(kcat, (int, float)):
            if kcat < 0.01:
                kcat_band = "low"
            elif kcat < 5.0:
                kcat_band = "medium"
            else:
                kcat_band = "high"
        else:
            kcat_band = "unknown"

    explanation: List[str] = []
    if best is not None:
        explanation.append(
            f"Top site {best.get('site_id')} favors {best.get('reaction_class')} with score {best.get('score')}"
        )
        explanation.append(
            f"Route {best.get('predicted_route')} and family {best.get('predicted_family')} support oxidative metabolism"
        )

    return {
        "drug_key": drug_key,
        "drug_name": drug_name,
        "primary_isoform": primary_isoform,
        "best_site": best,
        "top_sites": top_n_rows,
        "kcat_band": kcat_band,
        "summary": prediction.get("summary") or {},
        "explanation": explanation,
    }


def build_metabolism_validation_report(validation_payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(validation_payload.get("metrics") or {})
    top1 = float(metrics.get("top1_acc") or 0.0)
    top3 = float(metrics.get("top3_acc") or 0.0)
    note = "top3_improves_over_top1" if top3 >= top1 else "top3_below_top1_check_inputs"
    return {
        "summary": {
            "total_drugs": validation_payload.get("total_drugs"),
            "evaluable": validation_payload.get("evaluable"),
            "metrics": metrics,
            "note": note,
        },
        "error_buckets": validation_payload.get("error_buckets") or {},
        "per_drug_results": validation_payload.get("per_drug_results") or [],
    }
