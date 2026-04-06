from __future__ import annotations

from heapq import heappop, heappush
from typing import Dict

import numpy as np


def _safe_coords(atom_coordinates: np.ndarray | None, count: int) -> np.ndarray:
    if atom_coordinates is None:
        return np.zeros((count, 3), dtype=np.float32)
    coords = np.asarray(atom_coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return np.zeros((count, 3), dtype=np.float32)
    if coords.shape[0] != count:
        return np.zeros((count, 3), dtype=np.float32)
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def _real_harmonic_basis(unit: np.ndarray, radius_ratio: np.ndarray) -> np.ndarray:
    x = unit[:, 0]
    y = unit[:, 1]
    z = unit[:, 2]
    rr1 = radius_ratio
    rr2 = radius_ratio * radius_ratio
    basis = np.stack(
        [
            np.ones_like(x),
            rr1 * z,
            rr1 * x,
            rr1 * y,
            rr2 * (0.5 * (3.0 * z * z - 1.0)),
            rr2 * (x * z),
            rr2 * (y * z),
            rr2 * (x * x - y * y),
            rr2 * (x * y),
        ],
        axis=1,
    )
    return basis.astype(np.float32)


def compute_boundary_field_features(
    atom_coordinates: np.ndarray | None,
    *,
    cyp_profile: Dict[str, np.ndarray | float],
    access_proxy: np.ndarray | None = None,
    crowding: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    count = 0
    if atom_coordinates is not None:
        raw_coords = np.asarray(atom_coordinates)
        if raw_coords.ndim >= 1:
            count = int(raw_coords.shape[0])
    coords = _safe_coords(atom_coordinates, count)
    num_atoms = int(coords.shape[0])
    if num_atoms == 0:
        zeros = np.zeros((0, 1), dtype=np.float32)
        return {
            "scalar": zeros,
            "vector": np.zeros((0, 3), dtype=np.float32),
            "heme_center": np.zeros((1, 3), dtype=np.float32),
            "heme_distance": zeros,
            "axis_cosine": zeros,
            "radial_gate": zeros,
        }
    profile = np.asarray(cyp_profile.get("profile"), dtype=np.float32).reshape(1, -1)
    axis = np.asarray(cyp_profile.get("axis"), dtype=np.float32).reshape(3)
    coeffs = np.asarray(cyp_profile.get("boundary_coeffs"), dtype=np.float32).reshape(-1)
    radius = max(2.0, float(cyp_profile.get("boundary_radius", 6.0)))
    heme_offset = float(cyp_profile.get("heme_offset", 1.75))
    centroid = coords.mean(axis=0, keepdims=True)
    access = np.asarray(access_proxy if access_proxy is not None else np.zeros((num_atoms, 1), dtype=np.float32), dtype=np.float32).reshape(num_atoms, -1)
    crowd = np.asarray(crowding if crowding is not None else np.zeros((num_atoms, 1), dtype=np.float32), dtype=np.float32).reshape(num_atoms, -1)
    source_score = access[:, 0] - 0.35 * crowd[:, 0]
    source_idx = int(np.argmax(source_score)) if source_score.size else 0
    heme_center = centroid + (0.18 * (coords[source_idx : source_idx + 1] - centroid)) + (heme_offset * axis.reshape(1, 3))
    rel = coords - heme_center
    r = np.linalg.norm(rel, axis=1, keepdims=True)
    safe_r = np.clip(r, 1.0e-4, None)
    unit = rel / safe_r
    radius_ratio = np.clip((safe_r[:, 0] / radius), 0.0, 1.0)
    basis = _real_harmonic_basis(unit, radius_ratio)
    scalar = (basis @ coeffs.reshape(-1, 1)).astype(np.float32)
    radial_gate = np.exp(-np.square(safe_r / radius)).astype(np.float32)
    scalar = scalar * radial_gate
    vector = (unit * scalar).astype(np.float32)
    axis_cosine = (unit @ axis.reshape(3, 1)).astype(np.float32)
    return {
        "scalar": scalar.astype(np.float32),
        "vector": vector.astype(np.float32),
        "heme_center": heme_center.astype(np.float32),
        "heme_distance": safe_r.astype(np.float32),
        "axis_cosine": axis_cosine.astype(np.float32),
        "radial_gate": radial_gate.astype(np.float32),
        "profile": np.repeat(profile.astype(np.float32), num_atoms, axis=0),
    }


def compute_accessibility_features(
    atom_coordinates: np.ndarray | None,
    edge_index: np.ndarray | None,
    *,
    steric_score: np.ndarray | None = None,
    crowding: np.ndarray | None = None,
    boundary_scalar: np.ndarray | None = None,
    heme_distance: np.ndarray | None = None,
    cyp_profile: Dict[str, np.ndarray | float],
) -> Dict[str, np.ndarray]:
    count = 0
    if atom_coordinates is not None:
        raw_coords = np.asarray(atom_coordinates)
        if raw_coords.ndim >= 1:
            count = int(raw_coords.shape[0])
    coords = _safe_coords(atom_coordinates, count)
    num_atoms = int(coords.shape[0])
    zeros = np.zeros((num_atoms, 1), dtype=np.float32)
    if num_atoms == 0:
        return {"cost": zeros, "score": zeros, "blockage": zeros}
    steric = np.asarray(steric_score if steric_score is not None else zeros, dtype=np.float32).reshape(num_atoms, 1)
    crowd = np.asarray(crowding if crowding is not None else zeros, dtype=np.float32).reshape(num_atoms, 1)
    boundary = np.asarray(boundary_scalar if boundary_scalar is not None else zeros, dtype=np.float32).reshape(num_atoms, 1)
    heme_dist = np.asarray(heme_distance if heme_distance is not None else zeros, dtype=np.float32).reshape(num_atoms, 1)
    lam = max(1.0e-3, float(cyp_profile.get("access_lambda", 0.42)))
    source_idx = int(np.argmin(heme_dist[:, 0])) if heme_dist.size else 0

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_atoms)]
    if edge_index is not None and np.asarray(edge_index).size:
        edge_arr = np.asarray(edge_index, dtype=np.int64)
        for begin, end in edge_arr.T:
            u = int(begin)
            v = int(end)
            if u < 0 or v < 0 or u >= num_atoms or v >= num_atoms or u == v:
                continue
            dist = float(np.linalg.norm(coords[u] - coords[v]))
            block = 0.25 * max(float(steric[u, 0]), 0.0) + 0.35 * max(float(crowd[v, 0]), 0.0) + 0.15 * max(0.0, 1.0 - float(boundary[v, 0]))
            weight = max(1.0e-3, dist) * (1.0 + block)
            adjacency[u].append((v, weight))
    dist_cost = np.full((num_atoms,), np.inf, dtype=np.float32)
    dist_cost[source_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source_idx)]
    while heap:
        current_cost, node = heappop(heap)
        if current_cost > float(dist_cost[node]) + 1.0e-8:
            continue
        for nbr, weight in adjacency[node]:
            new_cost = float(current_cost) + float(weight)
            if new_cost < float(dist_cost[nbr]):
                dist_cost[nbr] = new_cost
                heappush(heap, (new_cost, nbr))
    if not np.isfinite(dist_cost).all():
        fallback = heme_dist[:, 0] + 0.35 * np.maximum(crowd[:, 0], 0.0)
        dist_cost = np.where(np.isfinite(dist_cost), dist_cost, fallback).astype(np.float32)
    blockage = (
        0.45 * np.maximum(crowd[:, 0], 0.0)
        + 0.30 * np.maximum(steric[:, 0], 0.0)
        + 0.25 * np.maximum(0.0, 1.0 - boundary[:, 0])
    ).astype(np.float32).reshape(-1, 1)
    total_cost = (dist_cost.reshape(-1, 1) + 0.35 * blockage).astype(np.float32)
    score = np.exp(-lam * total_cost).astype(np.float32)
    return {
        "cost": total_cost.astype(np.float32),
        "score": score.astype(np.float32),
        "blockage": blockage.astype(np.float32),
    }
