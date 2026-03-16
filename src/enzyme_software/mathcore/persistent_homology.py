from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def compute_signature(
    point_cloud: Optional[List[List[float]]],
    tunnel_summary: Optional[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    tunnel_summary = tunnel_summary or {}
    bottleneck = float(tunnel_summary.get("bottleneck_radius") or 0.0)
    path_length = float(tunnel_summary.get("path_length") or 0.0)
    curvature = float(tunnel_summary.get("curvature_proxy") or 0.0)

    if point_cloud:
        spread = _cloud_spread(point_cloud)
    else:
        spread = max(1.0, bottleneck * 2.0 + (path_length / 12.0))

    h1_max = round(min(3.0, bottleneck * 1.2 + spread * 0.15), 3)
    h2_sum = round(min(3.5, (path_length / 18.0) + bottleneck * 0.4), 3)
    entropy = round(_entropy([h1_max, h2_sum, curvature + 0.01]), 3)

    return {
        "mode": mode,
        "diagram": {
            "h1_max_persistence": h1_max,
            "h2_sum_persistence": h2_sum,
            "entropy": entropy,
            "curvature_proxy": round(curvature, 3),
            "point_cloud_spread": round(spread, 3),
        },
    }


def score_signature(signature: Dict[str, Any]) -> Tuple[float, float]:
    diagram = signature.get("diagram") or {}
    h1 = float(diagram.get("h1_max_persistence") or 0.0)
    h2 = float(diagram.get("h2_sum_persistence") or 0.0)
    curvature = float(diagram.get("curvature_proxy") or 0.0)
    entropy = float(diagram.get("entropy") or 0.0)

    topology_score = 0.35 * _clamp01(h1 / 2.5) + 0.35 * _clamp01(h2 / 2.5)
    topology_score += 0.2 * _clamp01(1.0 - curvature)
    topology_score += 0.1 * _clamp01(1.2 - entropy)

    robustness = 0.5 * _clamp01(h1 / 2.0) + 0.3 * _clamp01(h2 / 2.0)
    robustness += 0.2 * _clamp01(1.0 - entropy)

    return round(_clamp01(topology_score), 3), round(_clamp01(robustness), 3)


def topology_energy_component(topology_score: float, weight: float) -> float:
    return float(weight) * (1.0 - _clamp01(topology_score))


def _cloud_spread(points: List[List[float]]) -> float:
    if not points:
        return 0.0
    min_vals = [min(coord[i] for coord in points) for i in range(3)]
    max_vals = [max(coord[i] for coord in points) for i in range(3)]
    spans = [max_vals[i] - min_vals[i] for i in range(3)]
    return sum(spans) / 3.0


def _entropy(values: List[float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        p = value / total
        entropy -= p * _safe_log(p)
    return entropy


def _safe_log(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(__import__("math").log(value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
