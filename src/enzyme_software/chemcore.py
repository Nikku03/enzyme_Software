from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


PKA_CATALYTIC_GROUPS: Dict[str, float] = {
    "Asp": 4.0,
    "Glu": 4.0,
    "His": 6.5,
    "Cys": 8.3,
    "Lys": 10.5,
    "Tyr": 10.1,
    "Ser": 13.0,
}

SUBSTRATE_CARBOXYLIC_PKA_RANGE = (3.0, 5.0)
IONIC_SCREENING_COEFF = 2.0  # unitless, simple v1 attenuation factor
SOLVENT_UNKNOWN_PENALTY = 0.7


def fraction_protonated(pH: float, pKa: float) -> float:
    """Return fraction protonated for a generic acid/base (unitless 0..1)."""
    return 1.0 / (1.0 + 10 ** (pH - pKa))


def fraction_deprotonated(pH: float, pKa: float) -> float:
    """Return fraction deprotonated for a generic acid/base (unitless 0..1)."""
    return 1.0 - fraction_protonated(pH, pKa)


def screening_factor(ionic_strength: Optional[float]) -> Dict[str, Any]:
    """Return a simple ionic-strength screening factor (unitless) with uncertainty."""
    if ionic_strength is None:
        return {
            "value": None,
            "uncertain": True,
            "note": "ionic strength unknown",
        }
    try:
        strength = max(0.0, float(ionic_strength))
    except (TypeError, ValueError):
        return {
            "value": None,
            "uncertain": True,
            "note": "ionic strength invalid",
        }
    factor = 1.0 / (1.0 + IONIC_SCREENING_COEFF * strength)
    return {
        "value": float(factor),
        "uncertain": False,
        "note": "simple ionic strength screening",
    }


def solvent_penalty(solvent: Optional[str]) -> Dict[str, Any]:
    """Return a solvent penalty (unitless) and flags for missing solvent."""
    if solvent is None:
        return {
            "penalty": SOLVENT_UNKNOWN_PENALTY,
            "solvent_unknown": True,
            "note": "solvent unknown; applying conservative penalty",
        }
    text = str(solvent).strip().lower()
    if not text:
        return {
            "penalty": SOLVENT_UNKNOWN_PENALTY,
            "solvent_unknown": True,
            "note": "solvent empty; applying conservative penalty",
        }
    if "water" in text or "buffer" in text or "aqueous" in text:
        return {
            "penalty": 1.0,
            "solvent_unknown": False,
            "note": "aqueous solvent; no penalty",
        }
    return {
        "penalty": 0.85,
        "solvent_unknown": False,
        "note": "non-aqueous solvent; mild penalty",
    }


def protonation_fractions(
    pH: Optional[float],
    residues: Iterable[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return protonation fractions for catalytic residues at a given pH."""
    results: Dict[str, Dict[str, Optional[float]]] = {}
    if pH is None:
        for residue in residues:
            results[str(residue)] = {
                "pKa": None,
                "fraction_protonated": None,
                "fraction_deprotonated": None,
            }
        return results

    for residue in residues:
        key = str(residue)
        pka = PKA_CATALYTIC_GROUPS.get(key)
        if pka is None:
            results[key] = {
                "pKa": None,
                "fraction_protonated": None,
                "fraction_deprotonated": None,
            }
        else:
            frac_prot = fraction_protonated(pH, pka)
            results[key] = {
                "pKa": float(pka),
                "fraction_protonated": float(frac_prot),
                "fraction_deprotonated": float(1.0 - frac_prot),
            }
    return results


def chem_context_from_bond(
    bond_context: Dict[str, Any],
    substrate_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return simple chemistry proxies (0..1) derived from bond + substrate context."""
    notes: list[str] = []
    bond_class = str(bond_context.get("bond_class") or bond_context.get("bond_type") or "").lower()
    is_aromatic = bool(bond_context.get("is_aromatic"))
    in_ring = bool(bond_context.get("in_ring"))
    neighbor_hetero = bond_context.get("neighbor_hetero_atoms")
    charge_c = bond_context.get("gasteiger_charge_C")
    dipole_proxy = bond_context.get("dipole_proxy")
    polarity = str(bond_context.get("polarity") or "").lower()

    leaving_group_quality = 0.2
    if "anhydride" in bond_class or "ester" in bond_class or "carbonate" in bond_class:
        leaving_group_quality = 0.8
        notes.append("leaving_group: acyl-oxygen context")
    elif "thioester" in bond_class:
        leaving_group_quality = 0.7
        notes.append("leaving_group: thioester")
    elif "amide" in bond_class or "urea" in bond_class or "carbamate" in bond_class:
        leaving_group_quality = 0.2
        notes.append("leaving_group: amide-like")
    elif "halide" in bond_class:
        leaving_group_quality = 0.6
        notes.append("leaving_group: halide")
    elif "ether" in bond_class:
        leaving_group_quality = 0.3
        notes.append("leaving_group: ether")
    elif "ch" in bond_class:
        leaving_group_quality = 0.05
        notes.append("leaving_group: C-H")

    electrophilicity_proxy = 0.4
    if isinstance(charge_c, (int, float)):
        electrophilicity_proxy = min(1.0, max(0.0, (float(charge_c) + 0.1) / 0.6))
        notes.append("electrophilicity: Gasteiger charge")
    elif isinstance(dipole_proxy, (int, float)):
        electrophilicity_proxy = min(1.0, max(0.0, float(dipole_proxy) / 0.2))
        notes.append("electrophilicity: dipole proxy")
    else:
        if "highly" in polarity:
            electrophilicity_proxy = 0.7
        elif "moderately" in polarity:
            electrophilicity_proxy = 0.5
        elif "polar" in polarity:
            electrophilicity_proxy = 0.6
        else:
            electrophilicity_proxy = 0.2
        notes.append("electrophilicity: polarity class")

    resonance_stabilization_proxy = 0.2
    if is_aromatic or "aryl" in bond_class:
        resonance_stabilization_proxy = 0.7
        notes.append("resonance: aromatic context")
    elif "amide" in bond_class or "carbamate" in bond_class:
        resonance_stabilization_proxy = 0.6
        notes.append("resonance: amide-like")
    elif in_ring:
        resonance_stabilization_proxy = 0.5
        notes.append("resonance: ring context")

    steric_hindrance_proxy = 0.2
    structure_summary = (substrate_context or {}).get("structure_summary") or {}
    heavy_atoms = structure_summary.get("heavy_atoms")
    if isinstance(heavy_atoms, (int, float)):
        if heavy_atoms >= 30:
            steric_hindrance_proxy = 0.7
        elif heavy_atoms >= 20:
            steric_hindrance_proxy = 0.5
        elif heavy_atoms >= 10:
            steric_hindrance_proxy = 0.3
    if in_ring:
        steric_hindrance_proxy += 0.1
    if isinstance(neighbor_hetero, (int, float)) and neighbor_hetero >= 2:
        steric_hindrance_proxy += 0.05
    steric_hindrance_proxy = min(1.0, max(0.0, steric_hindrance_proxy))

    return {
        "leaving_group_quality": round(float(leaving_group_quality), 3),
        "electrophilicity_proxy": round(float(electrophilicity_proxy), 3),
        "resonance_stabilization_proxy": round(float(resonance_stabilization_proxy), 3),
        "steric_hindrance_proxy": round(float(steric_hindrance_proxy), 3),
        "notes": notes,
    }
