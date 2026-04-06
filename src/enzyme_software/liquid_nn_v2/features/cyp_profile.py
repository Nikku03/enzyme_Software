from __future__ import annotations

from typing import Dict

import numpy as np


CYP_PROFILE_DIM = 8
BOUNDARY_COEFF_DIM = 9


_DEFAULT_PROFILE = {
    "profile": np.asarray([0.55, 0.45, 0.50, 0.50, 0.45, 0.50, 0.50, 0.50], dtype=np.float32),
    "axis": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    "boundary_coeffs": np.asarray([0.30, 0.10, -0.08, 0.06, 0.14, -0.05, 0.07, 0.03, -0.02], dtype=np.float32),
    "boundary_radius": 6.0,
    "access_lambda": 0.42,
    "heme_offset": 1.75,
}


_CYP_PROFILES: Dict[str, Dict[str, np.ndarray | float]] = {
    "CYP3A4": {
        "profile": np.asarray([0.90, 0.80, 0.72, 0.88, 0.78, 0.82, 0.92, 0.70], dtype=np.float32),
        "axis": np.asarray([0.18, -0.10, 0.98], dtype=np.float32),
        "boundary_coeffs": np.asarray([0.46, 0.14, -0.11, 0.09, 0.20, -0.07, 0.10, 0.06, -0.03], dtype=np.float32),
        "boundary_radius": 7.2,
        "access_lambda": 0.34,
        "heme_offset": 2.05,
    },
    "CYP2D6": {
        "profile": np.asarray([0.45, 0.62, 0.78, 0.55, 0.40, 0.66, 0.36, 0.82], dtype=np.float32),
        "axis": np.asarray([-0.08, 0.24, 0.97], dtype=np.float32),
        "boundary_coeffs": np.asarray([0.34, -0.06, 0.12, 0.08, 0.10, 0.11, -0.05, -0.04, 0.06], dtype=np.float32),
        "boundary_radius": 5.8,
        "access_lambda": 0.50,
        "heme_offset": 1.55,
    },
    "CYP2C9": {
        "profile": np.asarray([0.52, 0.58, 0.48, 0.62, 0.44, 0.54, 0.46, 0.60], dtype=np.float32),
        "axis": np.asarray([0.10, 0.16, 0.98], dtype=np.float32),
        "boundary_coeffs": np.asarray([0.30, 0.08, 0.05, -0.03, 0.12, 0.07, -0.05, 0.02, 0.04], dtype=np.float32),
        "boundary_radius": 6.1,
        "access_lambda": 0.46,
        "heme_offset": 1.70,
    },
    "CYP1A2": {
        "profile": np.asarray([0.38, 0.74, 0.82, 0.34, 0.30, 0.68, 0.28, 0.90], dtype=np.float32),
        "axis": np.asarray([-0.20, 0.06, 0.98], dtype=np.float32),
        "boundary_coeffs": np.asarray([0.24, -0.12, 0.06, 0.14, 0.18, -0.03, 0.04, -0.02, 0.08], dtype=np.float32),
        "boundary_radius": 5.4,
        "access_lambda": 0.56,
        "heme_offset": 1.45,
    },
    "CYP2C19": {
        "profile": np.asarray([0.48, 0.56, 0.52, 0.58, 0.42, 0.57, 0.43, 0.66], dtype=np.float32),
        "axis": np.asarray([0.06, -0.18, 0.98], dtype=np.float32),
        "boundary_coeffs": np.asarray([0.28, 0.04, -0.02, 0.10, 0.11, -0.04, 0.06, 0.01, 0.05], dtype=np.float32),
        "boundary_radius": 5.9,
        "access_lambda": 0.48,
        "heme_offset": 1.60,
    },
}


def get_cyp_profile(cyp_label: str | None) -> Dict[str, np.ndarray | float]:
    key = str(cyp_label or "").strip().upper()
    payload = dict(_DEFAULT_PROFILE)
    if key in _CYP_PROFILES:
        payload.update(_CYP_PROFILES[key])
    profile_raw = payload.get("profile", _DEFAULT_PROFILE["profile"])
    axis_raw = payload.get("axis", _DEFAULT_PROFILE["axis"])
    coeffs_raw = payload.get("boundary_coeffs", _DEFAULT_PROFILE["boundary_coeffs"])
    profile = np.asarray(profile_raw, dtype=np.float32).reshape(1, -1)
    if profile.shape[1] != CYP_PROFILE_DIM:
        profile = np.asarray(_DEFAULT_PROFILE["profile"], dtype=np.float32).reshape(1, CYP_PROFILE_DIM)
    axis = np.asarray(axis_raw, dtype=np.float32).reshape(-1)
    if axis.size != 3:
        axis = np.asarray(_DEFAULT_PROFILE["axis"], dtype=np.float32).reshape(3)
    else:
        axis = axis.reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-6:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis = axis / axis_norm
    coeffs = np.asarray(coeffs_raw, dtype=np.float32).reshape(-1)
    if coeffs.size != BOUNDARY_COEFF_DIM:
        coeffs = np.asarray(_DEFAULT_PROFILE["boundary_coeffs"], dtype=np.float32).reshape(BOUNDARY_COEFF_DIM)
    return {
        "name": key or "UNKNOWN",
        "profile": profile,
        "axis": axis.astype(np.float32),
        "boundary_coeffs": coeffs.astype(np.float32),
        "boundary_radius": float(payload.get("boundary_radius", _DEFAULT_PROFILE["boundary_radius"])),
        "access_lambda": float(payload.get("access_lambda", _DEFAULT_PROFILE["access_lambda"])),
        "heme_offset": float(payload.get("heme_offset", _DEFAULT_PROFILE["heme_offset"])),
    }
