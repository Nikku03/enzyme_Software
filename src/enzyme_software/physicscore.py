"""Physics core: constants and physical estimators with explicit units."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Physical constants (SI)
R_J_per_molK = 8.314462618
kB_J_per_K = 1.380649e-23
h_Js = 6.62607015e-34
NA = 6.02214076e23
E_CHARGE_C = 1.602176634e-19
J_PER_KCAL = 4184.0

# Backwards-compatible aliases
R_J_PER_MOL_K = R_J_per_molK
K_B_J_PER_K = kB_J_per_K
R_KJ_PER_MOL_K = R_J_per_molK / 1000.0

# Coulomb constant in kJ*Å/(mol*e^2)
# Source: standard molecular mechanics conversion of 1/(4*pi*epsilon0)
COULOMB_KJ_MOL_A_PER_E2 = 138.935458

# Covalent radii (Å) - Pyykkö & Atsumi 2009 (approx)
COVALENT_RADII_A = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

# Approximate bond-order contraction (Å)
BondOrderContraction_A = {
    1: 0.0,
    2: 0.1,
    3: 0.2,
}

# Thermal breathing amplitude at 300 K (Å), order-of-magnitude envelope.
THERMAL_BREATHING_A_AT_300K = 0.05  # ~0.05 Å, soft protein fluctuations

# Barrier ratio threshold (dimensionless): delta_g / (R*T)
# 40 kT ~ 100 kJ/mol at 300 K, effectively rare without strong fields.
MAX_BARRIER_RATIO_KT = 40.0

# Diffusion defaults
DEFAULT_DIFFUSION_COEFF_M2_S = 1.0e-9  # Typical small-molecule diffusion in water.
DEFAULT_WATER_VISCOSITY_PA_S = 1.0e-3
DEFAULT_ENCOUNTER_RADIUS_M = 3.5e-10  # 3.5 Å

# Baseline ΔG‡ priors (kJ/mol) - conservative v1 defaults.
BASELINE_DG_KJ_MOL = {
    "ester": 80.0,  # acyl-oxygen hydrolysis
    "amide": 120.0,  # conservative 110-130 kJ/mol midpoint
    "aryl_c_br": 160.0,  # very high, non-catalyzed cleavage
}
BASELINE_DG_RANGE_KJ_MOL = {
    "ester": (70.0, 95.0),
    "amide": (95.0, 130.0),
    "aryl_c_br": (140.0, 180.0),
    "c_h_activation": (140.0, 200.0),
}


@dataclass(frozen=True)
class BaselineBarrier:
    """Baseline activation barrier definition (kJ/mol)."""

    bond_class: str
    mechanism_family: str
    deltaG_dagger_kJ: float
    source: str


_BASELINE_BARRIER_TABLE: Dict[Tuple[str, str], float] = {
    ("ester", "serine_hydrolase"): 90.0,
    ("ester", "metallo_esterase"): 92.0,
    ("ester", "hydrolysis"): 80.0,
    ("ester", "other"): 80.0,
    ("amide", "serine_hydrolase"): 105.0,
    ("amide", "hydrolysis"): 120.0,
    ("amide", "other"): 120.0,
    ("aryl_halide", "other"): 115.0,
    ("c-h", "other"): 125.0,
    ("c_h", "other"): 125.0,
    ("c-h_activation", "other"): 125.0,
    ("unknown", "other"): 110.0,
}


def _normalize_token(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    return (
        str(value)
        .strip()
        .lower()
        .replace("__", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_baseline_barrier(bond_class: str, mechanism_family: str) -> BaselineBarrier:
    """Return a conservative baseline barrier from the v1 table."""
    bond_norm = _normalize_token(bond_class)
    mech_norm = _normalize_token(mechanism_family)
    if bond_norm.startswith("ester"):
        bond_key = "ester"
    elif bond_norm.startswith("amide"):
        bond_key = "amide"
    elif "aryl" in bond_norm and ("br" in bond_norm or "halide" in bond_norm):
        bond_key = "aryl_halide"
    elif bond_norm.startswith("c_h") or bond_norm.startswith("ch") or "c_h" in bond_norm:
        bond_key = "c-h"
    else:
        bond_key = bond_norm or "unknown"

    mech_key = mech_norm or "other"
    key = (bond_key, mech_key)
    if key in _BASELINE_BARRIER_TABLE:
        delta_kj = _BASELINE_BARRIER_TABLE[key]
    else:
        fallback_key = (bond_key, "other")
        delta_kj = _BASELINE_BARRIER_TABLE.get(fallback_key, 110.0)
        mech_key = "other"
    return BaselineBarrier(
        bond_class=bond_key,
        mechanism_family=mech_key,
        deltaG_dagger_kJ=float(delta_kj),
        source="v1_table",
    )


def adjust_barrier_for_temperature(deltaG_kJ: float, temperature_K: float) -> float:
    """Placeholder temperature correction; returns deltaG unchanged."""
    _ = float(temperature_K)
    return float(deltaG_kJ)


def estimate_barrier_shift_kJ(variant_category: str) -> float:
    """Return ΔΔG‡ shift (kJ/mol) based on variant category (v1 conservative defaults)."""
    category = _normalize_token(variant_category)
    if category in ("oxyanion_hole", "oxyanion_hole_strengthening"):
        return -2.0
    if category in ("polar_anchor", "polar_anchor_carboxylate"):
        return -1.5
    if category in ("retention_clamp", "hydrophobic_clamp", "hydrophobic_clamp_retention"):
        return -0.8
    if category in ("mechanism_alignment", "alignment"):
        return 0.5
    if category in ("baseline",):
        return 0.0
    return -0.5


def kj_to_j(kj_per_mol: float) -> float:
    """Convert kJ/mol to J/mol."""
    return float(kj_per_mol) * 1000.0


def j_to_kj(j_per_mol: float) -> float:
    """Convert J/mol to kJ/mol."""
    return float(j_per_mol) / 1000.0


def j_to_kcal(j_per_mol: float) -> float:
    """Convert J/mol to kcal/mol."""
    return float(j_per_mol) / J_PER_KCAL


def kcal_to_j(kcal_per_mol: float) -> float:
    """Convert kcal/mol to J/mol."""
    return float(kcal_per_mol) * J_PER_KCAL


def kcal_to_kj(kcal_per_mol: float) -> float:
    """Convert kcal/mol to kJ/mol."""
    return float(kcal_per_mol) * (J_PER_KCAL / 1000.0)


def kj_to_kcal(kj_per_mol: float) -> float:
    """Convert kJ/mol to kcal/mol."""
    return float(kj_per_mol) / (J_PER_KCAL / 1000.0)


def c_to_k(T_C: float) -> float:
    """Convert Celsius to Kelvin."""
    return float(T_C) + 273.15


def k_to_c(T_K: float) -> float:
    """Convert Kelvin to Celsius."""
    return float(T_K) - 273.15


def eyring_rate_constant(deltaG_dagger_J_per_mol: float, T_K: float) -> float:
    """Eyring rate constant k [s^-1] for given ΔG‡ [J/mol] and temperature [K]."""
    temp = float(T_K)
    if temp <= 0:
        raise ValueError("T_K must be > 0")
    exponent = -float(deltaG_dagger_J_per_mol) / (R_J_per_molK * temp)
    return (kB_J_per_K * temp / h_Js) * math.exp(exponent)


def format_rate(k_s_inv: Optional[float]) -> str:
    """Format rate constants for display without rounding to 0.0000."""
    if k_s_inv is None:
        return "n/a"
    try:
        rate = float(k_s_inv)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(rate):
        return "n/a"
    if abs(rate) < 1.0e-4:
        return f"{rate:.3e}"
    return f"{rate:.4f}"


def eyring_k(delta_g_act_j_per_mol: float, T_K: float) -> float:
    """Alias for Eyring k [s^-1] using ΔG‡ [J/mol] and temperature [K]."""
    return eyring_rate_constant(delta_g_act_j_per_mol, T_K)


def delta_g_from_k(k_s_inv: float, T_K: float) -> float:
    """Invert Eyring equation to ΔG‡ [J/mol] from k [s^-1] and temperature [K]."""
    rate = float(k_s_inv)
    temp = float(T_K)
    if rate <= 0.0 or temp <= 0.0:
        return math.inf
    prefactor = (kB_J_per_K * temp) / h_Js
    return -R_J_per_molK * temp * math.log(rate / prefactor)


def half_life_seconds(k_s_inv: float) -> float:
    """Half-life t1/2 [s] from rate constant k [s^-1]."""
    rate = float(k_s_inv)
    if rate <= 0:
        return math.inf
    return math.log(2.0) / rate


def henderson_hasselbalch_fraction_deprotonated(pH: float, pKa: float) -> float:
    """Return deprotonated fraction fA- for an acid at given pH and pKa."""
    return 1.0 / (1.0 + 10.0 ** (float(pKa) - float(pH)))


def fraction_deprotonated_acid(pH: float, pKa: float) -> float:
    """Fraction deprotonated for an acid (A- form) at given pH."""
    return henderson_hasselbalch_fraction_deprotonated(pH, pKa)


def fraction_protonated_base(pH: float, pKa: float) -> float:
    """Fraction protonated for a base (BH+ form) at given pH."""
    return 1.0 / (1.0 + 10.0 ** (float(pH) - float(pKa)))


def catalytic_fraction(pH: float, pKa: float, mode: str = "acid") -> float:
    """Return catalytic fraction [0..1] for acid/base roles at given pH."""
    if mode == "acid":
        return 1.0 - fraction_deprotonated_acid(pH, pKa)
    if mode == "base":
        return 1.0 - fraction_protonated_base(pH, pKa)
    return 0.0


def stokes_einstein_D(
    T_K: float,
    eta_Pa_s: float,
    radius_m: float,
) -> float:
    """Stokes-Einstein diffusion coefficient D [m^2/s]."""
    temp = float(T_K)
    if temp <= 0:
        raise ValueError("T_K must be > 0")
    viscosity = max(1e-12, float(eta_Pa_s))
    radius = max(1e-12, float(radius_m))
    return (kB_J_per_K * temp) / (6.0 * math.pi * viscosity * radius)


def smoluchowski_kdiff(
    T_K: float,
    eta_Pa_s: float,
    radius_m: float,
    encounter_radius_m: float,
) -> float:
    """Smoluchowski diffusion limit k_diff [M^-1 s^-1]."""
    D = stokes_einstein_D(T_K, eta_Pa_s, radius_m)
    encounter = max(1e-12, float(encounter_radius_m))
    return 4.0 * math.pi * D * encounter * NA * 1.0e3


def diffusion_limit_rate_constant(
    T_K: float,
    viscosity_Pa_s: float,
    encounter_radius_m: float,
    diffusion_coeff_m2_s: Optional[float] = None,
) -> float:
    """Smoluchowski diffusion limit k_diff [M^-1 s^-1]."""
    temp = float(T_K)
    if temp <= 0:
        raise ValueError("T_K must be > 0")
    viscosity = max(1e-12, float(viscosity_Pa_s))
    radius_m = max(1e-12, float(encounter_radius_m))
    if diffusion_coeff_m2_s is None:
        diffusion_coeff_m2_s = (kB_J_per_K * temp) / (6.0 * math.pi * viscosity * radius_m)
    D = max(0.0, float(diffusion_coeff_m2_s))
    return 4.0 * math.pi * D * radius_m * NA * 1.0e3


def diffusion_cap(k_s_inv: float, cap_s_inv: float) -> float:
    """Cap a rate constant by a diffusion limit [s^-1]."""
    return min(max(0.0, float(k_s_inv)), float(cap_s_inv))


def coulomb_energy_J(
    q1_e: float,
    q2_e: float,
    r_m: float,
    eps_r: float = 78.5,
) -> float:
    """Coulomb interaction energy [J] for charges in units of e."""
    distance = max(1e-12, float(r_m))
    permittivity = max(1e-6, float(eps_r))
    coulomb_const = 8.9875517923e9
    return coulomb_const * (q1_e * E_CHARGE_C) * (q2_e * E_CHARGE_C) / (
        permittivity * distance
    )


def coulomb_energy_kj_mol_from_J(energy_J: float) -> float:
    """Convert Coulomb energy in J to kJ/mol."""
    return (float(energy_J) * NA) / 1000.0


def coulomb_energy_kcal_mol_from_J(energy_J: float) -> float:
    """Convert Coulomb energy in J to kcal/mol."""
    return float(energy_J) * NA / J_PER_KCAL


def boltzmann_weight(deltaE_kJ: float, temperature_K: float) -> float:
    """Return unnormalized Boltzmann weight exp(-ΔE/(R*T)) with ΔE in kJ/mol."""
    temp = float(temperature_K)
    if temp <= 0:
        raise ValueError("temperature_K must be > 0")
    delta_j = kj_to_j(float(deltaE_kJ))
    exponent = -delta_j / (R_J_per_molK * temp)
    exponent = max(-80.0, min(80.0, float(exponent)))
    return math.exp(exponent)


def screened_coulomb_energy_kJ(
    q1: float,
    q2: float,
    r_A: float,
    *,
    dielectric: float = 20.0,
) -> float:
    """Return screened Coulomb energy (kJ/mol) using a simple dielectric proxy."""
    distance_A = max(1.5, float(r_A))
    distance_nm = distance_A * 0.1
    denom = max(1e-6, float(dielectric)) * max(1e-6, distance_nm)
    return COULOMB_KJ_MOL_A_PER_E2 * float(q1) * float(q2) / denom


def diffusion_rate_cap_s_inv(
    D_m2_s: Optional[float] = None,
    r_m: Optional[float] = None,
) -> float:
    """Return diffusion-limited pseudo-first-order cap [s^-1] at 1 mM."""
    D = float(D_m2_s) if D_m2_s is not None else 5e-10
    radius_m = float(r_m) if r_m is not None else 5e-10
    k_L_mol_s = 4.0 * math.pi * D * radius_m * NA * 1.0e-3
    c_ref = 1.0e-3
    return max(0.0, float(k_L_mol_s) * c_ref)


def diffusion_cap_rate(
    substrate_context: Optional[Dict[str, Any]],
    condition_profile: Optional[Dict[str, Any]],
) -> float:
    """Return diffusion cap rate [s^-1], defaulting to 1e9 unless solvent/viscosity known."""
    default_cap = 1.0e9
    solvent = None
    viscosity = None
    temp_k = 298.15
    if condition_profile:
        if isinstance(condition_profile, dict):
            solvent = condition_profile.get("solvent")
            viscosity = condition_profile.get("viscosity_Pa_s")
            temp_k = condition_profile.get("temperature_K") or temp_k
        else:
            solvent = getattr(condition_profile, "solvent", None)
            viscosity = getattr(condition_profile, "viscosity_Pa_s", None)
            temp_k = getattr(condition_profile, "temperature_K", temp_k)
    if viscosity is None and not solvent:
        return float(default_cap)

    radius_A = None
    if substrate_context and isinstance(substrate_context, dict):
        radius_A = substrate_context.get("approx_radius")
        if radius_A is None and isinstance(substrate_context.get("substrate_size_proxies"), dict):
            size_proxies = substrate_context.get("substrate_size_proxies") or {}
            radius_A = size_proxies.get("approx_radius")
            if radius_A is None and size_proxies.get("min_diameter_proxy") is not None:
                radius_A = float(size_proxies.get("min_diameter_proxy")) / 2.0
    if not isinstance(radius_A, (int, float)) or radius_A <= 0.0:
        radius_A = 1.5
    radius_m = float(radius_A) * 1e-10
    eta = float(viscosity) if isinstance(viscosity, (int, float)) else DEFAULT_WATER_VISCOSITY_PA_S
    D = stokes_einstein_D(float(temp_k), eta, radius_m)
    return diffusion_rate_cap_s_inv(D_m2_s=D, r_m=radius_m)


def normalized_weights(energies_J: Iterable[float], T_K: float) -> List[float]:
    """Return normalized Boltzmann weights for energies [J]."""
    energies = [float(val) for val in energies_J]
    if not energies:
        return []
    temp = float(T_K)
    if temp <= 0:
        raise ValueError("T_K must be > 0")
    min_energy = min(energies)
    weights = [math.exp(-(val - min_energy) / (kB_J_per_K * temp)) for val in energies]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def estimate_deltaG_dagger_for_bond(
    bond_class: str,
    mechanism: Optional[str] = None,
    context: Optional[Dict[str, float]] = None,
) -> float:
    """Estimate ΔG‡ [J/mol] using conservative v1 bond-class priors."""
    context = context or {}
    if "delta_g_override_kj_mol" in context:
        return kj_to_j(float(context["delta_g_override_kj_mol"]))
    barrier = get_baseline_barrier(bond_class, mechanism or "other")
    return kj_to_j(barrier.deltaG_dagger_kJ)


def physics_prior_success_probability(
    deltaG_dagger_J_per_mol: float,
    T_K: float,
    time_horizon_s: float = 3600.0,
) -> float:
    """Return detectability probability p in [0,1] from ΔG‡ and time horizon."""
    temp = float(T_K)
    if temp <= 0:
        raise ValueError("T_K must be > 0")
    horizon = max(0.0, float(time_horizon_s))
    k_s_inv = eyring_rate_constant(deltaG_dagger_J_per_mol, temp)
    diff_cap = diffusion_cap_rate(
        substrate_context=None,
        condition_profile={"temperature_K": temp},
    )
    if diff_cap > 0.0:
        k_s_inv = min(float(k_s_inv), float(diff_cap))
    turnovers = expected_turnovers(k_s_inv, horizon)
    return detectability_probability(turnovers, n_required=3.0)


def expected_turnovers(k_s_inv: float, horizon_s: float) -> float:
    """Return expected turnovers (k * t)."""
    try:
        rate = float(k_s_inv)
        horizon = float(horizon_s)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(rate) or not math.isfinite(horizon) or rate <= 0.0 or horizon <= 0.0:
        return 0.0
    return max(0.0, rate * horizon)


def detectability_probability(
    turnovers: float,
    n_required: float,
    sharpness: float = 3.0,
) -> float:
    """Map turnovers to detectability probability using a log10 logistic."""
    try:
        turns = float(turnovers)
        required = float(n_required)
        slope = float(sharpness)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(turns) or not math.isfinite(required) or required <= 0.0:
        return 0.0
    turns = max(0.0, turns)
    log_turns = math.log10(turns + 1e-9)
    log_req = math.log10(required)
    z = slope * (log_turns - log_req)
    if z >= 50.0:
        return 1.0
    if z <= -50.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z))))


def compute_energy_ledger(
    deltaG_dagger_kJ: float,
    temperature_K: float,
    horizon_s: float,
    *,
    diffusion_cap_s_inv: Optional[float] = None,
    rate_multiplier: float = 1.0,
) -> Dict[str, float]:
    """Compute a unit-consistent energy ledger for a single ΔG‡ baseline.

    Returns a dict with:
      - deltaG_dagger_kJ
      - eyring_k_s_inv
      - k_diff_cap_s_inv
      - k_eff_s_inv
      - horizon_s
      - p_success_horizon
    """
    temp_k = float(temperature_K)
    horizon = max(0.0, float(horizon_s))
    delta_g_kj = float(deltaG_dagger_kJ)
    k_eyring = eyring_rate_constant(kj_to_j(delta_g_kj), temp_k)
    k_cap = (
        float(diffusion_cap_s_inv)
        if isinstance(diffusion_cap_s_inv, (int, float))
        else None
    )
    if k_cap is not None and k_cap > 0.0:
        k_eff = min(float(k_eyring), float(k_cap))
    else:
        k_eff = float(k_eyring)
        k_cap = None
    multiplier = max(0.0, min(1.0, float(rate_multiplier)))
    k_eff *= multiplier
    p_success = 0.0
    if horizon > 0.0 and k_eff > 0.0:
        p_success = 1.0 - math.exp(-k_eff * horizon)
        p_success = max(0.0, min(1.0, float(p_success)))
    return {
        "deltaG_dagger_kJ": float(delta_g_kj),
        "eyring_k_s_inv": float(k_eyring),
        "k_diff_cap_s_inv": float(k_cap) if k_cap is not None else None,
        "k_eff_s_inv": float(k_eff),
        "horizon_s": float(horizon),
        "p_success_horizon": float(p_success),
    }


def compute_route_prior(
    route_name: str,
    bond_class: str,
    temperature_K: float,
    horizon_s: float,
    pH: Optional[float] = None,
    ionic_strength: Optional[float] = None,
    chem_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute feasibility prior for a route using conservative physics-only mapping."""
    barrier = get_baseline_barrier(bond_class, route_name)
    delta_g_kj = adjust_barrier_for_temperature(barrier.deltaG_dagger_kJ, temperature_K)
    chem_context = chem_context or {}
    leaving_group_quality = float(chem_context.get("leaving_group_quality") or 0.0)
    electrophilicity_proxy = float(chem_context.get("electrophilicity_proxy") or 0.0)
    resonance_proxy = float(chem_context.get("resonance_stabilization_proxy") or 0.0)
    steric_proxy = float(chem_context.get("steric_hindrance_proxy") or 0.0)

    leaving_adj = -3.0 * max(0.0, min(1.0, leaving_group_quality))
    steric_adj = 5.0 * max(0.0, min(1.0, steric_proxy))
    electro_adj = -2.0 * max(0.0, min(1.0, electrophilicity_proxy))
    route_lower = str(route_name or "").lower()
    resonance_sign = -1.0 if any(key in route_lower for key in ("radical", "metal", "metallo")) else 1.0
    resonance_adj = resonance_sign * 1.5 * max(0.0, min(1.0, resonance_proxy))
    chem_adjustment = leaving_adj + steric_adj + electro_adj + resonance_adj
    delta_g_final_kj = max(1.0, float(delta_g_kj) + float(chem_adjustment))
    diff_cap = diffusion_cap_rate(
        substrate_context=chem_context,
        condition_profile={"temperature_K": temperature_K},
    )
    f_prot = 0.5
    if isinstance(chem_context.get("protonation_factor"), (int, float)):
        f_prot = float(chem_context.get("protonation_factor"))
    rate_multiplier = max(0.05, min(1.0, float(f_prot)))
    energy_ledger = compute_energy_ledger(
        deltaG_dagger_kJ=delta_g_final_kj,
        temperature_K=temperature_K,
        horizon_s=horizon_s,
        diffusion_cap_s_inv=diff_cap,
        rate_multiplier=rate_multiplier,
    )
    k_eyring = energy_ledger.get("eyring_k_s_inv")
    k_eff = energy_ledger.get("k_eff_s_inv")
    p_convert = energy_ledger.get("p_success_horizon")
    turns = expected_turnovers(k_eff or 0.0, horizon_s)
    p_raw = detectability_probability(turns, n_required=3.0)
    shrink_factor = 0.6
    p_final = 0.5 + shrink_factor * (float(p_raw) - 0.5)
    p_final = max(0.05, min(0.85, float(p_final)))
    return {
        "route_name": route_name,
        "bond_class": bond_class,
        "deltaG_dagger_kJ_per_mol": round(float(delta_g_final_kj), 3),
        "eyring_k_s_inv": float(k_eyring) if isinstance(k_eyring, (int, float)) else None,
        "diffusion_cap_s_inv": float(diff_cap),
        "k_effective_s_inv": float(k_eff) if isinstance(k_eff, (int, float)) else None,
        "p_convert_horizon": float(p_convert) if isinstance(p_convert, (int, float)) else None,
        "energy_ledger": energy_ledger,
        "turnovers": float(turns),
        "detectability_n_required": 3.0,
        "p_raw": round(float(p_raw), 6),
        "p_final": round(float(p_final), 6),
        "deltaG_components_kJ_per_mol": {
            "baseline": round(float(delta_g_kj), 3),
            "leaving_group": round(float(leaving_adj), 3),
            "steric": round(float(steric_adj), 3),
            "electrophilicity": round(float(electro_adj), 3),
            "resonance": round(float(resonance_adj), 3),
            "total_adjustment": round(float(chem_adjustment), 3),
            "final": round(float(delta_g_final_kj), 3),
        },
        "chem_context": chem_context,
        "f_prot": round(float(f_prot), 3),
        "prior_note": "Physics feasibility prior under horizon; not conversion certainty.",
        "baseline_barrier_source": barrier.source,
        "baseline_barrier_bond_class": barrier.bond_class,
        "baseline_barrier_mechanism_family": barrier.mechanism_family,
        "temperature_K": round(float(temperature_K), 2),
        "horizon_s": round(float(horizon_s), 1),
        "pH": pH,
        "ionic_strength": ionic_strength,
    }


def kinetics_event_probability(
    k_s_inv: float,
    horizon_s: float,
    *,
    p_floor: float = 0.01,
    p_ceil: float = 0.95,
    softening: float = 0.25,
) -> float:
    """Return softened event probability from k [s^-1] over horizon [s]."""
    try:
        rate = float(k_s_inv)
        horizon = float(horizon_s)
    except (TypeError, ValueError):
        return float(p_floor)
    if rate <= 0.0 or horizon <= 0.0:
        return float(p_floor)
    p_raw = 1.0 - math.exp(-rate * horizon)
    p_raw = max(0.0, min(1.0, float(p_raw)))
    softness = max(1e-6, min(1.0, float(softening)))
    p_soft = p_raw ** softness
    p_soft = max(float(p_floor), min(float(p_ceil), float(p_soft)))
    return p_soft


def kinetics_from_context(
    bond_class: str,
    mechanism_family: str,
    T_K: float,
    pH: Optional[float] = None,
) -> Dict[str, Any]:
    """Return kinetics estimate dict for a bond/mechanism at given conditions."""
    notes: List[str] = []
    delta_g_j = estimate_deltaG_dagger_for_bond(bond_class, mechanism_family, {})
    delta_g_kj = j_to_kj(delta_g_j)
    delta_g_kcal = j_to_kcal(delta_g_j)
    k_uncapped = eyring_rate_constant(delta_g_j, T_K)
    diffusion_kdiff = smoluchowski_kdiff(
        T_K,
        DEFAULT_WATER_VISCOSITY_PA_S,
        radius_m=2.0e-10,
        encounter_radius_m=DEFAULT_ENCOUNTER_RADIUS_M,
    )
    cap_s_inv = diffusion_kdiff * 1.0e-3  # assume 1 mM pseudo-first-order
    k_s_inv = k_uncapped if cap_s_inv <= 0.0 else diffusion_cap(k_uncapped, cap_s_inv)
    if k_s_inv < k_uncapped:
        notes.append("Diffusion cap applied (assumes 1 mM encounter).")
    if pH is None:
        notes.append("pH not provided; catalytic fraction not applied.")
    half_life = half_life_seconds(k_s_inv)
    return {
        "delta_g_act_kj_mol": round(float(delta_g_kj), 2),
        "delta_g_act_kcal_mol": round(float(delta_g_kcal), 2),
        "k_s_inv": round(float(k_s_inv), 4),
        "half_life_s": round(float(half_life), 3) if math.isfinite(half_life) else math.inf,
        "notes": notes,
        "diffusion_cap_k_s_inv": round(float(cap_s_inv), 3),
    }


def thermal_energy_kj_per_mol(temp_K: float) -> float:
    """Return kT in kJ/mol at temperature in Kelvin."""
    return R_KJ_PER_MOL_K * float(temp_K)


def coulomb_energy_kj_mol(charge_a_e: float, charge_b_e: float, distance_A: float) -> float:
    """Electrostatic interaction energy (kJ/mol) for point charges in vacuum."""
    dist = max(1e-6, float(distance_A))
    return COULOMB_KJ_MOL_A_PER_E2 * float(charge_a_e) * float(charge_b_e) / dist


def estimate_bond_length_A(
    atom_a: str,
    atom_b: str,
    bond_order: int = 1,
) -> float:
    """Estimate bond length in Å from covalent radii and bond order."""
    radius_a = COVALENT_RADII_A.get(atom_a, 0.77)
    radius_b = COVALENT_RADII_A.get(atom_b, 0.77)
    contraction = BondOrderContraction_A.get(int(bond_order), 0.0)
    return max(0.6, float(radius_a + radius_b - contraction))


def thermal_breathing_buffer_A(temp_K: float) -> float:
    """Thermal breathing buffer in Å scaled from 300 K reference."""
    temp = max(1e-6, float(temp_K))
    return THERMAL_BREATHING_A_AT_300K * math.sqrt(temp / 300.0)


def access_score_from_tunnel(
    substrate_radius_A: float,
    bottleneck_radii_A: Iterable[float],
    temp_K: float,
) -> Dict[str, float]:
    """Estimate access score from tunnel bottlenecks and substrate radius."""
    radii = [float(val) for val in bottleneck_radii_A if val is not None]
    if not radii:
        return {
            "score": 0.0,
            "ok": 0.0,
            "min_bottleneck_A": 0.0,
            "effective_radius_A": float(substrate_radius_A),
            "thermal_buffer_A": 0.0,
            "margin_A": 0.0,
        }
    buffer_A = thermal_breathing_buffer_A(temp_K)
    effective_radius_A = max(0.1, float(substrate_radius_A) - buffer_A)
    min_bottleneck_A = min(radii)
    margin_A = min_bottleneck_A - effective_radius_A
    score = max(0.0, min(1.0, 0.5 + (margin_A / max(0.1, effective_radius_A))))
    ok = 1.0 if margin_A >= 0.0 else 0.0
    return {
        "score": score,
        "ok": ok,
        "min_bottleneck_A": min_bottleneck_A,
        "effective_radius_A": effective_radius_A,
        "thermal_buffer_A": buffer_A,
        "margin_A": margin_A,
    }


def barrier_ratio_kT(delta_g_kj_mol: float, temp_K: float) -> float:
    """Return dimensionless barrier ratio ΔG/(R*T)."""
    return float(delta_g_kj_mol) / max(1e-6, thermal_energy_kj_per_mol(temp_K))


def barrier_gate(delta_g_kj_mol: float, temp_K: float) -> Dict[str, float]:
    """Gate based on barrier ratio to thermal energy."""
    ratio = barrier_ratio_kT(delta_g_kj_mol, temp_K)
    ok = 1.0 if ratio <= MAX_BARRIER_RATIO_KT else 0.0
    return {
        "ratio_kT": ratio,
        "ok": ok,
        "threshold_kT": MAX_BARRIER_RATIO_KT,
        "kT_kj_mol": thermal_energy_kj_per_mol(temp_K),
    }


if __name__ == "__main__":
    low = eyring_rate_constant(kj_to_j(20.0), 298.15)
    high = eyring_rate_constant(kj_to_j(80.0), 298.15)
    assert low > high
    assert abs(henderson_hasselbalch_fraction_deprotonated(7.0, 7.0) - 0.5) < 1e-6
    wts = normalized_weights([0.0, 1.0, 2.0], 300.0)
    assert abs(sum(wts) - 1.0) < 1e-6
