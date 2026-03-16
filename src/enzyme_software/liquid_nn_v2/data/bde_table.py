from __future__ import annotations

BDE_TABLE = {
    "benzylic": 375.5,
    "allylic": 371.5,
    "alpha_hetero": 385.0,
    "tertiary_CH": 403.8,
    "secondary_CH": 412.5,
    "primary_CH": 423.0,
    "aryl": 472.2,
    "amine_NH": 386.0,
    "amide_NH": 440.0,
    "alcohol_OH": 435.7,
    "phenol_OH": 362.8,
    "other": 410.0,
}

BDE_MIN = 360.0
BDE_MAX = 480.0


def bde_to_tau_init(bde_value: float, tau_min: float = 0.1, tau_max: float = 1.5) -> float:
    normalized = (float(bde_value) - BDE_MIN) / (BDE_MAX - BDE_MIN)
    normalized = max(0.0, min(1.0, normalized))
    return float(tau_min + (tau_max - tau_min) * normalized)
