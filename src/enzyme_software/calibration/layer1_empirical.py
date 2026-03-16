"""Layer 1 empirical calibration tables for bond/reactivity priors."""

from __future__ import annotations

from typing import Any, Dict


CALIBRATION_VERSION = "layer1_empirical.v1"


BDE_TABLE: Dict[str, Dict[str, Any]] = {
    # C-H (kJ/mol)
    "ch__aliphatic": {"bde_kj_mol": 410.0, "uncertainty_kj_mol": 2.0, "radical_stability_index": 0.20},
    "ch__primary": {"bde_kj_mol": 423.0, "uncertainty_kj_mol": 1.7, "radical_stability_index": 0.20},
    "ch__secondary": {"bde_kj_mol": 412.5, "uncertainty_kj_mol": 1.7, "radical_stability_index": 0.35},
    "ch__tertiary": {"bde_kj_mol": 403.8, "uncertainty_kj_mol": 1.7, "radical_stability_index": 0.45},
    "ch__benzylic": {"bde_kj_mol": 375.5, "uncertainty_kj_mol": 2.5, "radical_stability_index": 0.75},
    "ch__allylic": {"bde_kj_mol": 371.5, "uncertainty_kj_mol": 1.7, "radical_stability_index": 0.80},
    "ch__alpha_hetero": {"bde_kj_mol": 385.0, "uncertainty_kj_mol": 4.0, "radical_stability_index": 0.50},
    "ch__fluoromethyl": {"bde_kj_mol": 423.8, "uncertainty_kj_mol": 4.2, "radical_stability_index": 0.20},
    "ch__difluoromethyl": {"bde_kj_mol": 431.8, "uncertainty_kj_mol": 4.2, "radical_stability_index": 0.15},
    "ch__trifluoromethyl": {"bde_kj_mol": 446.4, "uncertainty_kj_mol": 4.2, "radical_stability_index": 0.05},
    "ch__fluorinated": {"bde_kj_mol": 446.4, "uncertainty_kj_mol": 4.2, "radical_stability_index": 0.05},
    "ch__aryl": {"bde_kj_mol": 472.2, "uncertainty_kj_mol": 2.5, "radical_stability_index": 0.05},
    "ch__vinyl": {"bde_kj_mol": 463.2, "uncertainty_kj_mol": 2.1, "radical_stability_index": 0.05},
    # O-H
    "oh__alcohol": {"bde_kj_mol": 435.7, "uncertainty_kj_mol": 1.7, "radical_stability_index": 0.30},
    "oh__phenol": {"bde_kj_mol": 362.8, "uncertainty_kj_mol": 2.9, "radical_stability_index": 0.85},
    "oh__water": {"bde_kj_mol": 497.1, "uncertainty_kj_mol": 0.3, "radical_stability_index": 0.00},
    # N-H
    "nh__amine": {"bde_kj_mol": 386.0, "uncertainty_kj_mol": 8.0, "radical_stability_index": 0.30},
    "nh__amide": {"bde_kj_mol": 440.0, "uncertainty_kj_mol": 10.0, "radical_stability_index": 0.20},
    # C-X
    "c_cl__primary": {"bde_kj_mol": 350.2, "uncertainty_kj_mol": 1.7, "radical_stability_index": None},
    "c_br__primary": {"bde_kj_mol": 292.9, "uncertainty_kj_mol": 1.7, "radical_stability_index": None},
    "c_f__primary": {"bde_kj_mol": 472.0, "uncertainty_kj_mol": 4.0, "radical_stability_index": None},
}


EVANS_POLANYI_PARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Fe_IV_oxo_heme": {
        "saturated_ch": {"alpha": 0.495, "beta_kj_mol": -139.7},
        "unsaturated_ch": {"alpha": 0.30, "beta_kj_mol": -80.0},
    },
    "Fe_IV_oxo_nonheme": {
        "saturated_ch": {"alpha": 0.45, "beta_kj_mol": -125.0},
        "unsaturated_ch": {"alpha": 0.32, "beta_kj_mol": -95.0},
    },
    "radical_SAM": {
        "saturated_ch": {"alpha": 0.35, "beta_kj_mol": -55.0},
        "unsaturated_ch": {"alpha": 0.33, "beta_kj_mol": -45.0},
    },
    "generic_radical": {
        "saturated_ch": {"alpha": 0.40, "beta_kj_mol": -45.0},
        "unsaturated_ch": {"alpha": 0.32, "beta_kj_mol": -35.0},
    },
}


ENZYME_KINETICS_REFERENCE: Dict[str, Dict[str, Any]] = {
    "all_enzymes": {"median_kcat_s_inv": 10.0},
    "cytochrome_P450": {"median_kcat_s_inv": 3.0, "kcat_range_90pct": [0.01, 100.0]},
    "non_heme_iron_oxygenase": {"median_kcat_s_inv": 5.0, "kcat_range_90pct": [0.1, 50.0]},
    "serine_hydrolase": {"median_kcat_s_inv": 50.0, "kcat_range_90pct": [0.5, 5000.0]},
    "haloalkane_dehalogenase": {"median_kcat_s_inv": 1.0, "kcat_range_90pct": [0.01, 10.0]},
}


OXIDANT_ALIASES: Dict[str, str] = {
    "Fe_IV_oxo": "Fe_IV_oxo_heme",
    "non_heme_Fe": "Fe_IV_oxo_nonheme",
    "radical_SAM": "radical_SAM",
    "generic_radical": "generic_radical",
}


def bde_record(bond_class: str) -> Dict[str, Any]:
    return dict(BDE_TABLE.get(str(bond_class or "").lower(), {}))


def bde_kj_mol(bond_class: str, default: float = 410.0) -> float:
    rec = bde_record(bond_class)
    val = rec.get("bde_kj_mol")
    return float(val) if isinstance(val, (int, float)) else float(default)


def radical_stability_index(bond_class: str, default: float = 0.2) -> float:
    rec = bde_record(bond_class)
    val = rec.get("radical_stability_index")
    return float(val) if isinstance(val, (int, float)) else float(default)


def ep_bucket_for_bond_class(bond_class: str) -> str:
    cls = str(bond_class or "").lower()
    if any(key in cls for key in ("benzylic", "allylic", "aryl", "vinyl")):
        return "unsaturated_ch"
    return "saturated_ch"


def _normalize_oxidant(oxidant: str) -> str:
    raw = str(oxidant or "")
    if raw in EVANS_POLANYI_PARAMS:
        return raw
    return OXIDANT_ALIASES.get(raw, "generic_radical")


def hat_barrier_kj_mol(
    bond_class: str,
    bde_value_kj_mol: float,
    oxidant: str = "Fe_IV_oxo_heme",
    protein_correction_kj: float = -8.0,
) -> Dict[str, float]:
    bucket = ep_bucket_for_bond_class(bond_class)
    oxidant_key = _normalize_oxidant(oxidant)
    oxidant_params = EVANS_POLANYI_PARAMS.get(oxidant_key) or EVANS_POLANYI_PARAMS["generic_radical"]
    pars = oxidant_params.get(bucket) or EVANS_POLANYI_PARAMS["generic_radical"][bucket]
    alpha = float(pars["alpha"])
    beta = float(pars["beta_kj_mol"])
    barrier_raw = alpha * float(bde_value_kj_mol) + beta
    barrier_in_protein = barrier_raw + float(protein_correction_kj)
    return {
        "alpha": alpha,
        "beta_kj_mol": beta,
        "barrier_raw_kj_mol": float(barrier_raw),
        "barrier_in_protein_kj_mol": float(barrier_in_protein),
    }


def all_hat_barriers_kj_mol(
    bond_class: str,
    bde_value_kj_mol: float,
    protein_correction_kj: float = -8.0,
    include_legacy_aliases: bool = True,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for oxidant in EVANS_POLANYI_PARAMS:
        rec = hat_barrier_kj_mol(
            bond_class=bond_class,
            bde_value_kj_mol=bde_value_kj_mol,
            oxidant=oxidant,
            protein_correction_kj=protein_correction_kj,
        )
        out[oxidant] = float(rec["barrier_in_protein_kj_mol"])
    if include_legacy_aliases:
        if "Fe_IV_oxo_heme" in out:
            out["Fe_IV_oxo"] = float(out["Fe_IV_oxo_heme"])
        if "Fe_IV_oxo_nonheme" in out:
            out["non_heme_Fe"] = float(out["Fe_IV_oxo_nonheme"])
    return out
