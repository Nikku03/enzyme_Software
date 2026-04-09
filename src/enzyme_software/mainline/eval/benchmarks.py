from __future__ import annotations

from typing import Any


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _truth_set(row: dict[str, Any], mode: str) -> set[int]:
    normalized = str(mode or "exact").strip().lower()
    if normalized == "primary_only":
        return {int(v) for v in list(row.get("primary_site_atoms") or [])}
    if normalized == "primary_plus_secondary":
        return {
            int(v)
            for v in (
                list(row.get("primary_site_atoms") or [])
                + list(row.get("secondary_site_atoms") or [])
            )
        }
    if normalized == "any_labeled_site":
        return {
            int(v)
            for v in (
                list(row.get("primary_site_atoms") or [])
                + list(row.get("secondary_site_atoms") or [])
                + list(row.get("tertiary_site_atoms") or [])
                + list(row.get("all_labeled_site_atoms") or [])
            )
        }
    return {int(v) for v in list(row.get("site_atoms") or [])}


def _evaluate_rows(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    evaluated = 0
    shortlist6_hits = 0
    shortlist12_hits = 0
    winner_hits = 0
    winner_hit_pool = 0
    top1_hits = 0
    top3_hits = 0
    for row in rows:
        truth = _truth_set(row, mode)
        if not truth:
            continue
        evaluated += 1
        top1_atom = row.get("predicted_atom_idx")
        top3_atoms = [int(v) for v in list(row.get("winner_top3_atom_indices") or [])]
        shortlist6 = [int(v) for v in list(row.get("shortlist_top6_atom_indices") or [])]
        shortlist12 = [int(v) for v in list(row.get("shortlist_top12_atom_indices") or [])]
        shortlist_hit = any(int(v) in truth for v in shortlist6)
        shortlist_hit12 = any(int(v) in truth for v in shortlist12)
        shortlist6_hits += int(shortlist_hit)
        shortlist12_hits += int(shortlist_hit12)
        if any(int(v) in truth for v in list(row.get("winner_candidate_atom_indices") or [])):
            winner_hit_pool += 1
            winner_hits += int(top1_atom is not None and int(top1_atom) in truth)
        top1_hits += int(top1_atom is not None and int(top1_atom) in truth)
        top3_hits += int(any(int(v) in truth for v in top3_atoms))
    return {
        "evaluated_count": int(evaluated),
        "shortlist_recall_at_6": _fraction(shortlist6_hits, evaluated),
        "shortlist_recall_at_12": _fraction(shortlist12_hits, evaluated),
        "winner_acc_given_hit": _fraction(winner_hits, winner_hit_pool),
        "end_to_end_top1": _fraction(top1_hits, evaluated),
        "end_to_end_top3": _fraction(top3_hits, evaluated),
        "winner_hit_pool_count": int(winner_hit_pool),
    }


def evaluate_strict_exact_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _evaluate_rows(rows, mode="exact")


def evaluate_tiered_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_only = _evaluate_rows(rows, mode="primary_only")
    primary_plus_secondary = _evaluate_rows(rows, mode="primary_plus_secondary")
    any_labeled_site = _evaluate_rows(rows, mode="any_labeled_site")
    gain_ps = int(round(primary_plus_secondary["end_to_end_top1"] * primary_plus_secondary["evaluated_count"])) - int(
        round(primary_only["end_to_end_top1"] * primary_only["evaluated_count"])
    )
    gain_any = int(round(any_labeled_site["end_to_end_top1"] * any_labeled_site["evaluated_count"])) - int(
        round(primary_only["end_to_end_top1"] * primary_only["evaluated_count"])
    )
    return {
        "primary_only": primary_only,
        "primary_plus_secondary": primary_plus_secondary,
        "any_labeled_site": any_labeled_site,
        "relaxed_gain_primary_plus_secondary": int(gain_ps),
        "relaxed_gain_any_labeled_site": int(gain_any),
    }


def evaluate_high_confidence_benchmark(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    total = int(len(rows))
    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        cutoff = float(threshold)
        covered = [row for row in rows if float(row.get("confidence_score", 0.0)) >= cutoff]
        covered_total = int(len(covered))
        correct = sum(
            int(
                row.get("predicted_atom_idx") is not None
                and int(row["predicted_atom_idx"]) in {int(v) for v in list(row.get("site_atoms") or [])}
            )
            for row in covered
        )
        by_threshold[f"{cutoff:.2f}"] = {
            "threshold": float(cutoff),
            "coverage": _fraction(covered_total, total),
            "covered_count": int(covered_total),
            "covered_top1_accuracy": _fraction(correct, covered_total),
        }
    return {
        "total_count": int(total),
        "by_threshold": by_threshold,
    }
