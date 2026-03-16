from __future__ import annotations

from typing import Any, Dict, Optional


ENZYME_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "serine_hydrolase": {
        "nucleophiles": ["Ser", "Thr"],
        "ph_range": (6.5, 8.5),
        "temp_c_range": (20.0, 40.0),
        "requires_metals": False,
    },
    "metallo_esterase": {
        "nucleophiles": ["metal_bound_water", "Asp", "Glu", "His"],
        "ph_range": (6.0, 8.5),
        "temp_c_range": (20.0, 45.0),
        "requires_metals": True,
    },
    "cysteine_hydrolase": {
        "nucleophiles": ["Cys"],
        "ph_range": (7.0, 9.0),
        "temp_c_range": (20.0, 37.0),
        "requires_metals": False,
    },
    "cytochrome_P450": {
        "nucleophiles": ["Fe_IV_oxo", "Compound_I"],
        "ph_range": (6.5, 8.5),
        "temp_c_range": (20.0, 40.0),
        "requires_metals": True,
        "cofactors": ["heme", "NADPH", "O2"],
        "mechanism": "radical_rebound",
        "protonation_residue": "Cys",
        "protonation_pka": 8.5,
    },
    "radical_SAM_enzyme": {
        "nucleophiles": ["5_deoxyadenosyl_radical"],
        "ph_range": (7.0, 8.5),
        "temp_c_range": (25.0, 45.0),
        "requires_metals": True,
        "cofactors": ["SAM", "Fe4S4_cluster"],
        "mechanism": "hydrogen_atom_abstraction",
        "protonation_residue": "Cys",
        "protonation_pka": 8.3,
    },
    "non_heme_iron_oxygenase": {
        "nucleophiles": ["Fe_IV_oxo"],
        "ph_range": (6.5, 8.0),
        "temp_c_range": (20.0, 45.0),
        "requires_metals": True,
        "cofactors": ["Fe2+", "alpha_ketoglutarate", "O2"],
        "mechanism": "radical_rebound",
        "protonation_residue": "His",
        "protonation_pka": 6.5,
    },
    "metalloenzyme_radical": {
        "nucleophiles": ["metal_CF3", "metal_radical", "Fe_IV_oxo"],
        "ph_range": (6.5, 8.5),
        "temp_c_range": (20.0, 50.0),
        "requires_metals": True,
        "cofactors": ["metal_center"],
        "mechanism": "metal_radical_transfer",
        "protonation_residue": "His",
        "protonation_pka": 6.5,
    },
}

ROUTE_TO_ENZYME_FAMILY: Dict[str, str] = {
    "serine_hydrolase": "serine_hydrolase",
    "metallo_esterase": "metallo_esterase",
    "cysteine_hydrolase": "cysteine_hydrolase",
    "amidase": "metallo_esterase",
    "metalloprotease": "metallo_esterase",
    "p450": "cytochrome_P450",
    "p450_oxidation": "cytochrome_P450",
    "radical_sam": "radical_SAM_enzyme",
    "non_heme_iron": "non_heme_iron_oxygenase",
    "metallo_transfer_cf3": "metalloenzyme_radical",
    "radical_transfer": "metalloenzyme_radical",
    "radical_transferase": "metalloenzyme_radical",
}

NUCLEOPHILE_GEOMETRY_MAP: Dict[str, str] = {
    "cysteine_thiol": "Cys",
    "serine_og": "Ser",
    "threonine_og": "Thr",
    "metal_bound_water": "metal_bound_water",
}

RESIDUE_PKA_DEFAULTS: Dict[str, float] = {
    "Asp": 4.0,
    "Glu": 4.0,
    "His": 6.5,
    "Cys": 8.3,
    "Lys": 10.5,
    "Ser": 13.0,
}


def _match_family(route_name: str) -> Optional[str]:
    label = str(route_name or "").lower()
    mapped = ROUTE_TO_ENZYME_FAMILY.get(label)
    if mapped:
        return mapped
    for key, fam in ROUTE_TO_ENZYME_FAMILY.items():
        if key in label:
            return fam
    if "serine" in label:
        return "serine_hydrolase"
    if "p450" in label:
        return "cytochrome_P450"
    if "radical" in label and "sam" in label:
        return "radical_SAM_enzyme"
    if "non_heme" in label and "iron" in label:
        return "non_heme_iron_oxygenase"
    if "radical" in label or ("metallo" in label and "ester" not in label):
        return "metalloenzyme_radical"
    if "metallo" in label or "metal" in label:
        return "metallo_esterase"
    if "cys" in label or "thiol" in label or "cysteine" in label:
        return "cysteine_hydrolase"
    return None


def enzyme_family_prior(route_name: str) -> Dict[str, Any]:
    """Return a lightweight enzyme-family prior for routing/audit."""
    family = _match_family(route_name) or "unknown"
    profile = ENZYME_FAMILY_PROFILES.get(family)
    if profile is None:
        return {
            "family": family,
            "confidence": 0.3,
            "profile": {},
            "note": "no profile available; default prior",
        }
    return {
        "family": family,
        "confidence": 0.7 if family in ENZYME_FAMILY_PROFILES else 0.3,
        "profile": dict(profile),
        "note": "v2 family prior from route-family mapping",
    }


def nucleophile_from_geometry(geometry: Optional[str]) -> Optional[str]:
    if geometry is None:
        return None
    return NUCLEOPHILE_GEOMETRY_MAP.get(str(geometry).strip().lower())


def protonation_fraction(pH: Optional[float], pKa: float, acid_or_base: str) -> float:
    """Return active fraction (0..1) based on pH for acid/base residues."""
    if pH is None:
        return 0.5
    try:
        ph_val = float(pH)
        pka_val = float(pKa)
    except (TypeError, ValueError):
        return 0.5
    mode = str(acid_or_base or "").lower()
    if mode == "base":
        return 1.0 / (1.0 + 10 ** (pka_val - ph_val))
    if mode == "acid":
        return 1.0 / (1.0 + 10 ** (ph_val - pka_val))
    return 0.5


def residue_state_fraction(
    pH: Optional[float],
    residue_name: Optional[str],
    microenv: Optional[Dict[str, Any]] = None,
) -> float:
    """Return an estimated active fraction for a residue at pH (unitless 0..1)."""
    if residue_name is None:
        return 0.5
    name = str(residue_name).strip().title()
    pka = RESIDUE_PKA_DEFAULTS.get(name)
    if pka is None:
        return 0.5
    mode = "base" if name == "His" else "acid"
    if name in {"Cys"}:
        mode = "base"
    if name in {"Lys"}:
        mode = "base"
    if name in {"Asp", "Glu"}:
        mode = "acid"
    return protonation_fraction(pH, pka, mode)


def mechanism_mismatch_penalty(
    route_name: str,
    nucleophile_geometry: Optional[str],
) -> Dict[str, Any]:
    family = _match_family(route_name)
    observed = nucleophile_from_geometry(nucleophile_geometry)
    if family is None or observed is None:
        return {"penalty": 0.0, "reason": "insufficient data", "expected": None, "observed": observed}
    profile = ENZYME_FAMILY_PROFILES.get(family, {})
    expected = profile.get("nucleophiles") or []
    if observed in expected:
        return {
            "penalty": 0.0,
            "reason": "nucleophile matches family",
            "expected": expected,
            "observed": observed,
        }
    if family == "serine_hydrolase" and observed == "Cys":
        penalty = 0.15
        reason = "serine route with cysteine nucleophile"
    elif family == "cysteine_hydrolase" and observed in {"Ser", "Thr"}:
        penalty = 0.12
        reason = "cysteine route with serine/threonine nucleophile"
    else:
        penalty = 0.08
        reason = "nucleophile mismatch"
    return {
        "penalty": penalty,
        "reason": reason,
        "expected": expected,
        "observed": observed,
    }


def cofactor_compatibility_penalty(
    route_name: str,
    metals_allowed: Optional[bool],
) -> Dict[str, Any]:
    family = _match_family(route_name)
    if family is None:
        return {"penalty": 0.0, "reason": "unknown family"}
    profile = ENZYME_FAMILY_PROFILES.get(family, {})
    requires_metals = bool(profile.get("requires_metals"))
    if not requires_metals:
        return {"penalty": 0.0, "reason": "no metal requirement"}
    if metals_allowed is False:
        return {"penalty": 0.25, "reason": "metals forbidden"}
    if metals_allowed is None:
        return {"penalty": 0.1, "reason": "metals unspecified"}
    return {"penalty": 0.0, "reason": "metals allowed"}


def nucleophile_change_penalty(
    current_geometry: Optional[str],
    target_residue: Optional[str],
) -> Dict[str, Any]:
    current = nucleophile_from_geometry(current_geometry)
    target = target_residue
    if current is None or target is None:
        return {"penalty": 0.0, "reason": "insufficient data", "current": current, "target": target}
    if current == target:
        return {"penalty": 0.0, "reason": "no nucleophile change", "current": current, "target": target}
    return {
        "penalty": 0.12,
        "reason": "nucleophile identity change gate",
        "current": current,
        "target": target,
    }
