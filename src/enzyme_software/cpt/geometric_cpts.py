from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem

from .cpt_base import CPT
from .types import CPTResult
from .geometry import (
    angle_deg,
    build_attack_geometry,
    get_atom_pos,
    steric_clashes_along_ray,
)


def _score_linear_band(x: float, ideal: float, tol: float, hard: float) -> float:
    """Score 1 at ideal, decreases linearly to 0 by hard limit."""
    d = abs(x - ideal)
    if d <= tol:
        return 1.0
    if d >= hard:
        return 0.0
    return max(0.0, 1.0 - (d - tol) / (hard - tol))


class AttackAngleCPT(CPT):
    """Burgi-Dunitz-like geometry check around a carbonyl."""
    cpt_id = "attack_angle"
    fidelity = "geometric_basic"

    def run(
        self,
        *,
        mechanism_id: str,
        mol3d: Chem.Mol,
        role_to_idx: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> CPTResult:
        warnings = []
        required = ["carbonyl_c", "carbonyl_o", "hetero_attach"]
        for r in required:
            if r not in role_to_idx:
                return CPTResult(
                    cpt_id=self.cpt_id,
                    mechanism_id=mechanism_id,
                    passed=False,
                    score=0.0,
                    confidence=0.2,
                    message=f"Missing required role: {r}",
                    data={"fatal": True, "fidelity": self.fidelity},
                    warnings=[],
                )

        c = role_to_idx["carbonyl_c"]
        o = role_to_idx["carbonyl_o"]
        x = role_to_idx["hetero_attach"]

        ag = build_attack_geometry(mol3d, c, o, x, theta_deg=107.0)

        conf = mol3d.GetConformer()
        C = get_atom_pos(conf, c)
        O = get_atom_pos(conf, o)
        vCO = (O[0] - C[0], O[1] - C[1], O[2] - C[2])
        theta = angle_deg(vCO, ag.ideal_attack_dir)

        atomC = mol3d.GetAtomWithIdx(c)
        neighs = [a.GetIdx() for a in atomC.GetNeighbors() if a.GetIdx() != o]
        planarity_score = 0.75
        if len(neighs) >= 2:
            y = neighs[0] if neighs[0] != x else (neighs[1] if len(neighs) > 1 else x)
            ang1 = rdMolTransforms.GetAngleDeg(conf, o, c, x)
            ang2 = rdMolTransforms.GetAngleDeg(conf, o, c, y)
            s1 = _score_linear_band(ang1, ideal=120.0, tol=10.0, hard=35.0)
            s2 = _score_linear_band(ang2, ideal=120.0, tol=10.0, hard=35.0)
            planarity_score = 0.5 * (s1 + s2)

        angle_score = _score_linear_band(theta, ideal=107.0, tol=8.0, hard=30.0)
        score = 0.70 * angle_score + 0.30 * planarity_score
        passed = score >= 0.60

        confidence = 0.80
        if angle_score < 0.25:
            warnings.append("attack_angle_far_from_ideal")

        return CPTResult(
            cpt_id=self.cpt_id,
            mechanism_id=mechanism_id,
            passed=passed,
            score=float(score),
            confidence=float(confidence),
            message=(
                f"Attack geometry: theta(C->O vs ideal attack dir)={theta:.1f} deg; "
                f"planarity={planarity_score:.2f}"
            ),
            data={
                "theta_deg": float(theta),
                "angle_score": float(angle_score),
                "planarity_score": float(planarity_score),
                "ideal_theta": 107.0,
                "fidelity": self.fidelity,
            },
            warnings=warnings,
            atoms_involved=[c, o, x],
        )


class StericOcclusionCPT(CPT):
    """Approximates whether a probe could approach carbonyl carbon."""
    cpt_id = "steric_occlusion"
    fidelity = "geometric_with_sterics"

    def run(
        self,
        *,
        mechanism_id: str,
        mol3d: Chem.Mol,
        role_to_idx: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> CPTResult:
        warnings = []
        required = ["carbonyl_c", "carbonyl_o", "hetero_attach"]
        for r in required:
            if r not in role_to_idx:
                return CPTResult(
                    cpt_id=self.cpt_id,
                    mechanism_id=mechanism_id,
                    passed=False,
                    score=0.0,
                    confidence=0.25,
                    message=f"Missing required role: {r}",
                    data={"fatal": True, "fidelity": self.fidelity},
                )

        c = role_to_idx["carbonyl_c"]
        o = role_to_idx["carbonyl_o"]
        x = role_to_idx["hetero_attach"]
        ag = build_attack_geometry(mol3d, c, o, x, theta_deg=107.0)

        probe_radius = (extra or {}).get("probe_radius", 1.4)
        ignore = [c, o, x]

        clash_count, min_clear, worst = steric_clashes_along_ray(
            mol3d,
            origin_idx=c,
            direction=ag.ideal_attack_dir,
            start=1.6,
            end=3.2,
            step=0.2,
            probe_radius=probe_radius,
            ignore_indices=ignore,
        )

        if clash_count == 0:
            clear_score = _score_linear_band(min_clear, ideal=0.6, tol=0.4, hard=1.6)
            score = 0.85 + 0.15 * clear_score
        else:
            score = max(0.0, 1.0 - 0.15 * clash_count - max(0.0, (-min_clear)) * 0.6)

        passed = score >= 0.55
        confidence = 0.85

        if clash_count > 0:
            warnings.append("steric_clashes_detected")
        if min_clear < 0:
            warnings.append("negative_clearance_overlap")

        return CPTResult(
            cpt_id=self.cpt_id,
            mechanism_id=mechanism_id,
            passed=passed,
            score=float(score),
            confidence=float(confidence),
            message=(
                f"Steric corridor: clashes={clash_count}, min_clearance={min_clear:.2f} A, "
                f"probe={probe_radius:.2f} A"
            ),
            data={
                "clash_count": int(clash_count),
                "min_clearance_A": float(min_clear),
                "worst_atoms": [
                    {"idx": int(i), "min_clearance_A": float(ca)} for i, ca in worst
                ],
                "probe_radius": float(probe_radius),
                "fidelity": self.fidelity,
            },
            warnings=warnings,
            atoms_involved=[c, o, x],
        )


class LeavingGroupDistanceCPT(CPT):
    """Uses acyl_C-hetero bond length as a proxy for leaving-group readiness."""
    cpt_id = "leaving_group_distance"
    fidelity = "geometric_basic"

    def run(
        self,
        *,
        mechanism_id: str,
        mol3d: Chem.Mol,
        role_to_idx: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> CPTResult:
        required = ["carbonyl_c", "hetero_attach"]
        for r in required:
            if r not in role_to_idx:
                return CPTResult(
                    cpt_id=self.cpt_id,
                    mechanism_id=mechanism_id,
                    passed=False,
                    score=0.0,
                    confidence=0.25,
                    message=f"Missing required role: {r}",
                    data={"fatal": True, "fidelity": self.fidelity},
                )

        c = role_to_idx["carbonyl_c"]
        x = role_to_idx["hetero_attach"]
        conf = mol3d.GetConformer()
        dist = rdMolTransforms.GetBondLength(conf, c, x)

        score = _score_linear_band(dist, ideal=1.36, tol=0.04, hard=0.14)
        passed = score >= 0.55
        confidence = 0.75

        msg = f"Acyl C-X distance={dist:.3f} A (ideal ~1.36 A)."
        return CPTResult(
            cpt_id=self.cpt_id,
            mechanism_id=mechanism_id,
            passed=passed,
            score=float(score),
            confidence=float(confidence),
            message=msg,
            data={
                "acyl_C_hetero_A": float(dist),
                "ideal_A": 1.36,
                "fidelity": self.fidelity,
            },
            atoms_involved=[c, x],
        )


# ----------------------------
# EP-AV CPT (Electronic Properties & Attack Site Validation)
# ----------------------------


@dataclass(frozen=True)
class EPAVResult:
    passed: bool
    score: float
    confidence: float
    dominant_driver: str
    breakdown: Dict[str, float]
    warnings: List[str]
    data: Dict[str, Any]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_float_atomprop(atom: Chem.Atom, prop: str) -> Optional[float]:
    if atom.HasProp(prop):
        try:
            return float(atom.GetProp(prop))
        except Exception:
            return None
    return None


def _compute_gasteiger(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(m)
    except Exception:
        pass
    return m


def _charges_list(mol: Chem.Mol) -> List[Optional[float]]:
    charges: List[Optional[float]] = []
    for atom in mol.GetAtoms():
        qc = _safe_float_atomprop(atom, "_GasteigerCharge")
        charges.append(qc)
    return charges


def _find_carbonyl_carbons(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    patt = Chem.MolFromSmarts("[CX3](=O)[#6,#7,#8,#16]")
    hits = mol.GetSubstructMatches(patt) if patt is not None else ()
    return [(c, o, x) for (c, o, x) in hits]


LG_SCORE_BY_GROUP = {
    "ester": 0.70,
    "thioester": 0.85,
    "carbonate": 0.65,
    "amide": 0.20,
    "urea": 0.10,
    "carbamate": 0.25,
    "anhydride": 0.90,
    "acyl_halide": 1.00,
}


def _score_site(
    mol: Chem.Mol,
    c: int,
    o: int,
    x: int,
    charges: List[Optional[float]],
    group_type: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    qc = charges[c] if charges[c] is not None else 0.0
    qo = charges[o] if charges[o] is not None else 0.0

    charge_score = _clamp01((qc - 0.05) / (0.45 - 0.05))

    x_atom = mol.GetAtomWithIdx(x)
    x_sym = x_atom.GetSymbol()
    if group_type in LG_SCORE_BY_GROUP:
        lg_score = LG_SCORE_BY_GROUP[group_type]
    else:
        if x_sym == "O":
            lg_score = 0.70
        elif x_sym == "N":
            lg_score = 0.20
        elif x_sym == "S":
            lg_score = 0.60
        else:
            lg_score = 0.30

    total = 0.55 * charge_score + 0.45 * lg_score
    return total, {"charge": charge_score, "leaving_group": lg_score, "lg_symbol": x_sym}


def _electrophile_candidates_carbonyl(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    patt = Chem.MolFromSmarts("[CX3](=[OX1])[#8,#7,#16,F,Cl,Br,I]")
    hits = mol.GetSubstructMatches(patt) if patt is not None else ()
    out: List[Tuple[int, int, int]] = []
    for (c_idx, o_idx, x_idx) in hits:
        out.append((c_idx, o_idx, x_idx))
    return out


def _leaving_group_score(attached_atom: Chem.Atom, carbonyl_c: Chem.Atom) -> float:
    z = attached_atom.GetAtomicNum()
    if z in (9, 17, 35, 53):
        return 1.00
    if z == 8:
        for nb in attached_atom.GetNeighbors():
            if nb.GetIdx() == carbonyl_c.GetIdx():
                continue
            if nb.GetAtomicNum() == 6:
                for b in nb.GetBonds():
                    if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8:
                        return 0.85
        return 0.70
    if z == 16:
        return 0.80
    if z == 7:
        return 0.20
    return 0.35


def _charge_electrophilicity_score(mol_with_charges: Chem.Mol, c_idx: int, o_idx: int) -> float:
    c = mol_with_charges.GetAtomWithIdx(c_idx)
    o = mol_with_charges.GetAtomWithIdx(o_idx)
    qc = _safe_float_atomprop(c, "_GasteigerCharge")
    qo = _safe_float_atomprop(o, "_GasteigerCharge")
    if qc is None or qo is None:
        return 0.50
    qc_term = _clamp01(0.5 + (qc / 0.45) * 0.5)
    qo_term = _clamp01(0.5 + ((-qo) / 0.45) * 0.5)
    return _clamp01(0.55 * qc_term + 0.45 * qo_term)


def _competing_site_score_gap(
    site_scores: List[Tuple[int, float]], chosen_c_idx: int
) -> Tuple[float, Optional[int], float]:
    chosen_score = None
    best_other = (-1, -1.0)
    for idx, sc in site_scores:
        if idx == chosen_c_idx:
            chosen_score = sc
        else:
            if sc > best_other[1]:
                best_other = (idx, sc)
    if chosen_score is None:
        return (-1.0, best_other[0] if best_other[0] != -1 else None, best_other[1])
    return (chosen_score - best_other[1], best_other[0] if best_other[0] != -1 else None, best_other[1])


class ElectronicPropertiesAttackValidationCPT:
    """
    Electronic Properties & Attack-site Validation CPT.

    Evaluates whether a carbonyl site is electronically suitable for
    nucleophilic attack, using three sub-scores:

      charge        Gasteiger electrophilicity of the carbonyl carbon.
                    Mapped linearly from qC in [0.05, 0.45] to [0, 1].
      leaving_group How good the heteroatom leaving group is.
                    Lookup by group_type (LG_SCORE_BY_GROUP), else by element.
      competition   Whether a more reactive competing site exists in the
                    same molecule.  Sigmoid around the score gap.

    Pass / fail gating
    ------------------
    ``passed = True`` requires ALL of:
      1. total >= min_total_score_pass  (default 0.55)
      2. charge >= 0.35
      3. leaving_group >= 0.20
      4. competition >= 0.25  OR equivalent_sites_detected

    Interpretation downstream
    -------------------------
    * passed=True  -> proceed to geometric / env CPTs
    * passed=False, score in [0.35, 0.55) -> borderline; may still be
      feasible with a strong catalyst family (metalloenzyme, radical)
    * passed=False, score < 0.35 -> electronically disfavored; route is
      unlikely without covalent activation or extreme conditions

    Symmetry / equivalence
    ----------------------
    competition_mode="allow_equivalents" (default):
      If two sites have nearly identical scores (|gap| < equivalent_gap_eps)
      AND the same leaving-group type, they are treated as symmetry-
      equivalent and competition_score is set to 1.0.
    competition_mode="penalize":
      All competing sites reduce competition_score regardless of symmetry.
    """

    def __init__(
        self,
        w_charge: float = 0.55,
        w_lg: float = 0.25,
        w_competition: float = 0.20,
        min_total_score_pass: float = 0.55,
        competition_margin: float = 0.25,
        competition_scale: float = 0.10,
        competition_mode: str = "allow_equivalents",
        equivalent_gap_eps: float = 0.02,
        debug: bool = False,
    ) -> None:
        self.w_charge = w_charge
        self.w_lg = w_lg
        self.w_competition = w_competition
        self.min_total_score_pass = min_total_score_pass
        self.competition_margin = competition_margin
        self.competition_scale = competition_scale
        self.competition_mode = competition_mode
        self.equivalent_gap_eps = equivalent_gap_eps
        self.debug = debug

    def run(self, mol3d: Chem.Mol, role_to_idx: Dict[str, int], group_type: Optional[str] = None) -> EPAVResult:
        warnings: List[str] = []
        need = {"carbonyl_c", "carbonyl_o", "hetero_attach"}
        if not need.issubset(set(role_to_idx.keys())):
            return EPAVResult(
                passed=False,
                score=0.0,
                confidence=0.45,
                dominant_driver="missing_roles",
                breakdown={"charge": 0.0, "leaving_group": 0.0, "competition": 0.0},
                warnings=["missing_roles"],
                data={"need": sorted(need), "have": sorted(role_to_idx.keys())},
            )

        c_idx = int(role_to_idx["carbonyl_c"])
        o_idx = int(role_to_idx["carbonyl_o"])
        x_idx = int(role_to_idx["hetero_attach"])

        mchg = _compute_gasteiger(mol3d)
        charges = _charges_list(mchg)
        total_score, parts = _score_site(mchg, c_idx, o_idx, x_idx, charges, group_type=group_type)
        charge_score = parts["charge"]
        lg_score = parts["leaving_group"]

        candidates_raw = _find_carbonyl_carbons(mchg)
        # Deduplicate by carbonyl carbon index, keep best score per index.
        best_by_c: Dict[int, Tuple[float, Dict[str, float]]] = {}
        for (ci, oi, xi) in candidates_raw:
            sc, parts = _score_site(mchg, ci, oi, xi, charges, group_type=group_type)
            prev = best_by_c.get(ci)
            if (prev is None) or (sc > prev[0]):
                best_by_c[ci] = (sc, parts)

        site_scores: List[Tuple[int, float, Dict[str, float]]] = [
            (ci, best_by_c[ci][0], best_by_c[ci][1]) for ci in sorted(best_by_c.keys())
        ]

        sel_tuple = next((t for t in site_scores if t[0] == c_idx), None)
        others = [t for t in site_scores if t[0] != c_idx]

        equivalent_sites_detected = False
        if not others:
            competition_score = 1.0
            gap = 1.0
            best_other_idx = None
            best_other_score = 0.0
            warnings.append("no_competing_sites_found")
        else:
            best_other = max(others, key=lambda t: t[1])
            gap = (sel_tuple[1] if sel_tuple is not None else 0.0) - best_other[1]
            # Allow equivalent sites if configured
            if (
                self.competition_mode == "allow_equivalents"
                and abs(gap) < self.equivalent_gap_eps
                and sel_tuple is not None
                and sel_tuple[2].get("leaving_group") == best_other[2].get("leaving_group")
                and sel_tuple[2].get("lg_symbol") == best_other[2].get("lg_symbol")
            ):
                competition_score = 1.0
                equivalent_sites_detected = True
                warnings.append("equivalent_sites_detected")
            else:
                z = (gap - self.competition_margin) / max(1e-6, self.competition_scale)
                competition_score = 1.0 / (1.0 + math.exp(-z))
            best_other_idx = best_other[0]
            best_other_score = best_other[1]
            if gap < 0:
                warnings.append("competing_site_more_electrophilic")
            elif gap < self.competition_margin:
                warnings.append("competing_site_close")

        # Ensure single source of truth for LG/charge from selected tuple (if present)
        if sel_tuple is not None:
            charge_score = sel_tuple[2]["charge"]
            lg_score = sel_tuple[2]["leaving_group"]

        total = _clamp01(
            self.w_charge * charge_score + self.w_lg * lg_score + self.w_competition * competition_score
        )
        passed = (
            total >= self.min_total_score_pass
            and charge_score >= 0.35
            and lg_score >= 0.20
            and (competition_score >= 0.25 or equivalent_sites_detected)
        )
        # Dominant driver = weakest component (the bottleneck holding score back)
        dominant = min(
            [("charge", charge_score), ("leaving_group", lg_score), ("competition", competition_score)],
            key=lambda t: t[1],
        )[0]

        conf = 0.80
        qc = _safe_float_atomprop(mchg.GetAtomWithIdx(c_idx), "_GasteigerCharge")
        qo = _safe_float_atomprop(mchg.GetAtomWithIdx(o_idx), "_GasteigerCharge")
        if qc is None or qo is None:
            conf -= 0.20
            warnings.append("charges_unavailable")
        if "competing_site_more_electrophilic" in warnings:
            conf -= 0.10
        conf = _clamp01(conf)

        if self.debug:
            print("[EP-AV] candidate sites:")
            for (ci, sc, parts) in site_scores:
                qc = _safe_float_atomprop(mchg.GetAtomWithIdx(ci), "_GasteigerCharge")
                # find corresponding carbonyl oxygen for printing if possible
                print(
                    f"  c_idx={ci} qc={qc} "
                    f"charge={parts.get('charge'):.2f} lg={parts.get('leaving_group'):.2f} "
                    f"site_score={sc:.2f}"
                )
            print(
                f"[EP-AV] sel_c={c_idx} charge={charge_score:.2f} lg={lg_score:.2f} "
                f"comp={competition_score:.2f} gap={gap:.3f} total={total:.2f}"
            )

        return EPAVResult(
            passed=passed,
            score=float(total),
            confidence=float(conf),
            dominant_driver=dominant,
            breakdown={
                "charge": float(charge_score),
                "leaving_group": float(lg_score),
                "competition": float(competition_score),
            },
            warnings=warnings,
            data={
                "selected": {"carbonyl_c": c_idx, "carbonyl_o": o_idx, "hetero_attach": x_idx},
                "selected_charges": {"qc": qc, "qo": qo},
                "candidate_site_scores": site_scores,
                "competition": {
                    "gap": gap,
                    "best_other_idx": best_other_idx,
                    "best_other_score": best_other_score,
                    "equivalent_sites_detected": equivalent_sites_detected,
                },
            },
        )


# ----------------------------
# Level 1 Steric CPT (cone + envelope + wobble)
# ----------------------------


def _unit_vec(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return v
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _angle_deg_vec(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
    u = _unit_vec(u)
    v = _unit_vec(v)
    c = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1] + u[2] * v[2]))
    return math.degrees(math.acos(c))


def _rotation_matrix(axis: Tuple[float, float, float], angle_rad: float) -> Tuple[Tuple[float, float, float], ...]:
    axis = _unit_vec(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return (
        (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
    )


def _mat_vec(m: Tuple[Tuple[float, float, float], ...], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _vdw_radius(atom: Chem.Atom) -> float:
    table = {
        1: 1.20,  # H
        6: 1.70,  # C
        7: 1.55,  # N
        8: 1.52,  # O
        9: 1.47,  # F
        15: 1.80, # P
        16: 1.80, # S
        17: 1.75, # Cl
        35: 1.85, # Br
        53: 1.98, # I
    }
    return table.get(atom.GetAtomicNum(), 1.80)


def _find_third_neighbor(mol: Chem.Mol, c_idx: int, exclude: set[int]) -> Optional[int]:
    atom = mol.GetAtomWithIdx(c_idx)
    for nb in atom.GetNeighbors():
        j = nb.GetIdx()
        if j not in exclude:
            return j
    return None


@dataclass
class StericEnvelopeResult:
    passed: bool
    score: float
    confidence: float
    clashes: int
    min_clearance_A: float
    required_extra_clearance_A: float
    best_face: str
    message: str
    warnings: List[str]


class StericOcclusionEnvelopeCPT:
    """
    Level 1: Implicit environment-aware steric CPT.
    - Uses an approach cone (only counts atoms in front of the carbonyl along approach dir).
    - Evaluates both faces (Re/Si) by flipping in-plane component.
    - Adds wobble: rotates approach direction around C=O axis within +/- wobble_deg.
    """

    def __init__(
        self,
        probe_radius_A: float = 1.40,
        cone_half_angle_deg: float = 35.0,
        bd_angle_deg: float = 107.0,
        dmin_A: float = 2.0,
        dmax_A: float = 3.2,
        step_A: float = 0.20,
        wobble_deg: float = 25.0,
        wobble_steps: int = 7,
        clearance_pass_A: float = 0.25,
        clearance_soft_A: float = 0.00,
    ) -> None:
        self.probe_radius_A = probe_radius_A
        self.cone_half_angle_deg = cone_half_angle_deg
        self.bd_angle_deg = bd_angle_deg
        self.dmin_A = dmin_A
        self.dmax_A = dmax_A
        self.step_A = step_A
        self.wobble_deg = wobble_deg
        self.wobble_steps = wobble_steps
        self.clearance_pass_A = clearance_pass_A
        self.clearance_soft_A = clearance_soft_A

    def run(self, mol3d: Chem.Mol, role_to_idx: Dict[str, int]) -> StericEnvelopeResult:
        warnings: List[str] = []

        if (
            "carbonyl_c" not in role_to_idx
            or "carbonyl_o" not in role_to_idx
            or "hetero_attach" not in role_to_idx
        ):
            return StericEnvelopeResult(
                passed=False,
                score=0.0,
                confidence=0.4,
                clashes=0,
                min_clearance_A=-999.0,
                required_extra_clearance_A=999.0,
                best_face="n/a",
                message="Missing required roles: need carbonyl_c, carbonyl_o, hetero_attach.",
                warnings=["missing_roles"],
            )

        c_idx = role_to_idx["carbonyl_c"]
        o_idx = role_to_idx["carbonyl_o"]
        x_idx = role_to_idx["hetero_attach"]

        conf = mol3d.GetConformer()
        C = conf.GetAtomPosition(c_idx)
        O = conf.GetAtomPosition(o_idx)
        X = conf.GetAtomPosition(x_idx)

        vCO = _unit_vec((O.x - C.x, O.y - C.y, O.z - C.z))

        third = _find_third_neighbor(mol3d, c_idx, exclude={o_idx, x_idx})
        if third is None:
            vCX = _unit_vec((X.x - C.x, X.y - C.y, X.z - C.z))
            n = _cross(vCO, vCX)
            if math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) < 1e-8:
                warnings.append("degenerate_plane")
                tmp = (1.0, 0.0, 0.0)
                if abs(vCO[0]) > 0.9:
                    tmp = (0.0, 1.0, 0.0)
                n = _cross(vCO, tmp)
            n = _unit_vec(n)
        else:
            R = conf.GetAtomPosition(third)
            vCR = _unit_vec((R.x - C.x, R.y - C.y, R.z - C.z))
            n = _cross(vCO, vCR)
            if math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) < 1e-8:
                warnings.append("degenerate_plane")
                vCX = _unit_vec((X.x - C.x, X.y - C.y, X.z - C.z))
                n = _cross(vCO, vCX)
            n = _unit_vec(n)

        p = _unit_vec(_cross(n, vCO))
        if math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) < 1e-8:
            warnings.append("degenerate_inplane")
            tmp = (0.0, 1.0, 0.0)
            p = _unit_vec(_cross(tmp, vCO))

        phi_deg = 180.0 - self.bd_angle_deg
        phi = math.radians(phi_deg)
        base1 = (-vCO[0], -vCO[1], -vCO[2])

        faces = [("Re", 1.0), ("Si", -1.0)]

        best = None
        best_details: Optional[Tuple[str, float, int]] = None

        exclude = {c_idx, o_idx, x_idx}
        heavy_atoms = [
            (i, mol3d.GetAtomWithIdx(i))
            for i in range(mol3d.GetNumAtoms())
            if mol3d.GetAtomWithIdx(i).GetAtomicNum() != 1
        ]

        d_samples: List[float] = []
        t = self.dmin_A
        while t <= self.dmax_A + 1e-9:
            d_samples.append(float(t))
            t += self.step_A

        if self.wobble_steps <= 1:
            wobble_angles = [0.0]
        else:
            wobble_angles = [
                -self.wobble_deg + (2 * self.wobble_deg) * i / (self.wobble_steps - 1)
                for i in range(self.wobble_steps)
            ]

        for face_name, sgn in faces:
            a0 = _unit_vec(
                (
                    math.cos(phi) * base1[0] + math.sin(phi) * (sgn * p[0]),
                    math.cos(phi) * base1[1] + math.sin(phi) * (sgn * p[1]),
                    math.cos(phi) * base1[2] + math.sin(phi) * (sgn * p[2]),
                )
            )

            for wob_deg in wobble_angles:
                Rmat = _rotation_matrix(vCO, math.radians(float(wob_deg)))
                a = _unit_vec(_mat_vec(Rmat, a0))

                min_clearance = 1e9
                clash_count = 0

                for d in d_samples:
                    P = (C.x + a[0] * d, C.y + a[1] * d, C.z + a[2] * d)
                    for i, atom in heavy_atoms:
                        if i in exclude:
                            continue
                        Ai = conf.GetAtomPosition(i)
                        vCA = (Ai.x - C.x, Ai.y - C.y, Ai.z - C.z)
                        if math.sqrt(vCA[0] * vCA[0] + vCA[1] * vCA[1] + vCA[2] * vCA[2]) < 1e-8:
                            continue
                        ang = _angle_deg_vec(vCA, a)
                        if ang > self.cone_half_angle_deg:
                            continue
                        dx = P[0] - Ai.x
                        dy = P[1] - Ai.y
                        dz = P[2] - Ai.z
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        clearance = dist - (_vdw_radius(atom) + self.probe_radius_A)
                        if clearance < min_clearance:
                            min_clearance = clearance
                        if clearance < 0.0:
                            clash_count += 1

                if best is None or min_clearance > best:
                    best = min_clearance
                    best_details = (face_name, float(wob_deg), clash_count)

        face_name, wob_deg, clashes = best_details if best_details else ("n/a", 0.0, 0)
        min_clearance = float(best if best is not None else -999.0)
        required_extra = max(0.0, -min_clearance)

        if min_clearance >= self.clearance_pass_A:
            score = 1.0
            passed = True
        elif min_clearance >= self.clearance_soft_A:
            t = (min_clearance - self.clearance_soft_A) / max(1e-9, (self.clearance_pass_A - self.clearance_soft_A))
            score = 0.4 + 0.6 * max(0.0, min(1.0, t))
            passed = True
            warnings.append("borderline_clearance")
        else:
            score = 0.0
            passed = False
            warnings.extend(["steric_clashes_detected", "negative_clearance_overlap"])

        confidence = 0.85
        if "borderline_clearance" in warnings:
            confidence = 0.70

        msg = (
            f"Steric envelope: best_face={face_name}, wobble={wob_deg:+.1f} deg, "
            f"clashes={clashes}, min_clearance={min_clearance:+.2f} A, "
            f"probe={self.probe_radius_A:.2f} A, cone={self.cone_half_angle_deg:.0f} deg."
        )
        if required_extra > 0:
            msg += (
                f" Needs +{required_extra:.2f} A extra clearance (or a wider pocket corridor)."
            )

        return StericEnvelopeResult(
            passed=passed,
            score=float(score),
            confidence=float(confidence),
            clashes=int(clashes),
            min_clearance_A=min_clearance,
            required_extra_clearance_A=required_extra,
            best_face=face_name,
            message=msg,
            warnings=warnings,
        )


# ----------------------------
# Level 2 CPT (soft sterics + SASA + corridor polarity)
# ----------------------------


def _calc_sasa_per_atom(mol3d: Chem.Mol) -> Optional[List[float]]:
    """Use rdFreeSASA if available; return per-atom SASA or None."""
    try:
        from rdkit.Chem import rdFreeSASA
        radii = rdFreeSASA.classifyAtoms(mol3d)
        rdFreeSASA.CalcSASA(mol3d, radii)
        per_atom: List[float] = []
        for a in mol3d.GetAtoms():
            per_atom.append(float(a.GetProp("SASA")) if a.HasProp("SASA") else 0.0)
        return per_atom
    except Exception:
        return None


@dataclass
class Level2EnvCPTResult:
    passed: bool
    score: float
    confidence: float
    best_face: str
    best_wobble_deg: float
    min_clearance_A: float
    soft_steric_energy: float
    sasa_reactive_A2: Optional[float]
    corridor_polarity_score: float
    message: str
    warnings: List[str]
    soft_steric_energy_raw: Optional[float] = None
    soft_steric_energy_capped: Optional[float] = None
    worst_blocker_atom_idx: Optional[int] = None
    worst_blocker_element: Optional[str] = None
    worst_blocker_distance_A: Optional[float] = None
    worst_blocker_clearance_A: Optional[float] = None
    worst_blocker_side: Optional[str] = None
    best_conf_id: Optional[int] = None


class EnvironmentAwareStericsCPT_Level2:
    """
    Level 2: protein-free micro-environment proxies.
    - Soft steric energy along approach corridor (continuous penalty).
    - Local exposure via SASA of reactive atoms (if rdFreeSASA available).
    - Corridor polarity proxy: fraction of polar atoms in corridor (O/N) vs nonpolar.
    """

    def __init__(
        self,
        probe_radius_A: float = 1.40,
        cone_half_angle_deg: float = 35.0,
        bd_angle_deg: float = 107.0,
        dmin_A: float = 2.0,
        dmax_A: float = 3.2,
        step_A: float = 0.20,
        wobble_deg: float = 25.0,
        wobble_steps: int = 7,
        softness_A: float = 0.35,
        repulsion_power: float = 2.0,
        energy_cap: float = 50.0,
        clearance_hard_fail_A: float = -0.75,
        energy_fail: float = 12.0,
        sasa_floor_A2: float = 3.0,
        clearance_pass_A: float = 1.8,
        w_clearance: float = 0.45,
        w_energy: float = 0.35,
        w_sasa: float = 0.10,
        w_polarity: float = 0.10,
        debug: bool = False,
        protein_aware: bool = False,
        n_conformers: int = 1,
    ) -> None:
        self.probe_radius_A = probe_radius_A
        self.cone_half_angle_deg = cone_half_angle_deg
        self.bd_angle_deg = bd_angle_deg
        self.dmin_A = dmin_A
        self.dmax_A = dmax_A
        self.step_A = step_A
        self.wobble_deg = wobble_deg
        self.wobble_steps = wobble_steps

        self.softness_A = softness_A
        self.repulsion_power = repulsion_power
        self.energy_cap = energy_cap

        self.clearance_hard_fail_A = clearance_hard_fail_A
        self.energy_fail = energy_fail
        self.sasa_floor_A2 = sasa_floor_A2
        self.clearance_pass_A = clearance_pass_A

        self.w_clearance = w_clearance
        self.w_energy = w_energy
        self.w_sasa = w_sasa
        self.w_polarity = w_polarity
        self.debug = debug
        self.protein_aware = protein_aware
        self.n_conformers = max(1, int(n_conformers))

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _prepare_conformers(
        self, mol3d: Chem.Mol
    ) -> Tuple[Chem.Mol, List[int]]:
        if self.n_conformers <= 1:
            conf_ids = [mol3d.GetConformer().GetId()]
            return mol3d, conf_ids

        mol_work = Chem.Mol(mol3d)
        mol_work.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        params.useSmallRingTorsions = True
        params.useBasicKnowledge = True
        conf_ids = AllChem.EmbedMultipleConfs(
            mol_work, numConfs=self.n_conformers, params=params
        )
        if not conf_ids:
            self._log("[L2] conformer embed failed; using existing conformer")
            conf_ids = [mol3d.GetConformer().GetId()]
            return mol3d, conf_ids

        try:
            AllChem.MMFFOptimizeMoleculeConfs(
                mol_work, maxIters=200, mmffVariant="MMFF94s"
            )
        except Exception:
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol_work, maxIters=200)
            except Exception:
                self._log("[L2] conformer optimization failed; using embedded coords")

        return mol_work, [int(i) for i in conf_ids]

    def for_mechanism(self, mechanism_id: str) -> "EnvironmentAwareStericsCPT_Level2":
        c = EnvironmentAwareStericsCPT_Level2(
            probe_radius_A=self.probe_radius_A,
            cone_half_angle_deg=self.cone_half_angle_deg,
            bd_angle_deg=self.bd_angle_deg,
            dmin_A=self.dmin_A,
            dmax_A=self.dmax_A,
            step_A=self.step_A,
            wobble_deg=self.wobble_deg,
            wobble_steps=self.wobble_steps,
            softness_A=self.softness_A,
            repulsion_power=self.repulsion_power,
            energy_cap=self.energy_cap,
            clearance_hard_fail_A=self.clearance_hard_fail_A,
            energy_fail=self.energy_fail,
            sasa_floor_A2=self.sasa_floor_A2,
            debug=self.debug,
            protein_aware=self.protein_aware,
            n_conformers=self.n_conformers,
        )
        if mechanism_id == "serine_hydrolase":
            c.w_clearance, c.w_energy, c.w_sasa, c.w_polarity = 0.40, 0.30, 0.10, 0.20
        elif mechanism_id == "metallo_esterase":
            c.w_clearance, c.w_energy, c.w_sasa, c.w_polarity = 0.50, 0.35, 0.10, 0.05
        return c

    def run(self, mol3d: Chem.Mol, role_to_idx: Dict[str, int]) -> Level2EnvCPTResult:
        warnings: List[str] = []
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        need = {"carbonyl_c", "carbonyl_o", "hetero_attach"}
        if not need.issubset(set(role_to_idx.keys())):
            self._log(f"[L2] missing roles: need={sorted(need)} have={sorted(role_to_idx.keys())}")
            return Level2EnvCPTResult(
                passed=False,
                score=0.0,
                confidence=0.4,
                best_face="n/a",
                best_wobble_deg=0.0,
                min_clearance_A=-999.0,
                soft_steric_energy=999.0,
                sasa_reactive_A2=None,
                corridor_polarity_score=0.0,
                message="Missing required roles: carbonyl_c, carbonyl_o, hetero_attach.",
                warnings=["missing_roles"],
            )

        c_idx = role_to_idx["carbonyl_c"]
        o_idx = role_to_idx["carbonyl_o"]
        x_idx = role_to_idx["hetero_attach"]
        self._log(f"[L2] roles: carbonyl_c={c_idx} carbonyl_o={o_idx} hetero_attach={x_idx}")

        mol_use, conf_ids = self._prepare_conformers(mol3d)
        dmat = Chem.GetDistanceMatrix(mol_use)

        acyl_nb = _find_third_neighbor(mol_use, c_idx, exclude={o_idx, x_idx})
        alkoxy_nb = None
        for nb in mol_use.GetAtomWithIdx(x_idx).GetNeighbors():
            j = nb.GetIdx()
            if j != c_idx:
                alkoxy_nb = j
                break

        def _classify_blocker(idx: int) -> str:
            atom = mol_use.GetAtomWithIdx(idx)
            if atom.IsInRing():
                return "ring"
            if acyl_nb is not None and alkoxy_nb is not None:
                if dmat[acyl_nb][idx] < dmat[alkoxy_nb][idx]:
                    return "acyl_substituent"
                if dmat[alkoxy_nb][idx] < dmat[acyl_nb][idx]:
                    return "alkoxy_side"
            if acyl_nb is not None and dmat[acyl_nb][idx] <= 2:
                return "acyl_substituent"
            if alkoxy_nb is not None and dmat[alkoxy_nb][idx] <= 2:
                return "alkoxy_side"
            return "other"

        best = None
        best_pack = None

        for conf_id in conf_ids:
            conf = mol_use.GetConformer(int(conf_id))
            C = conf.GetAtomPosition(c_idx)
            O = conf.GetAtomPosition(o_idx)
            X = conf.GetAtomPosition(x_idx)

            vCO = _unit_vec((O.x - C.x, O.y - C.y, O.z - C.z))
            self._log(f"[L2] conf={conf_id} vCO={vCO}")

            third = _find_third_neighbor(mol_use, c_idx, exclude={o_idx, x_idx})
            if third is None:
                self._log("[L2] third neighbor: none (using fallback plane)")
                vCX = _unit_vec((X.x - C.x, X.y - C.y, X.z - C.z))
                n = _cross(vCO, vCX)
                if math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) < 1e-8:
                    warnings.append("degenerate_plane")
                    tmp = (1.0, 0.0, 0.0)
                    if abs(vCO[0]) > 0.9:
                        tmp = (0.0, 1.0, 0.0)
                    n = _cross(vCO, tmp)
                n = _unit_vec(n)
            else:
                self._log(f"[L2] third neighbor idx={third}")
                R = conf.GetAtomPosition(third)
                vCR = _unit_vec((R.x - C.x, R.y - C.y, R.z - C.z))
                n = _cross(vCO, vCR)
                if math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) < 1e-8:
                    warnings.append("degenerate_plane")
                    vCX = _unit_vec((X.x - C.x, X.y - C.y, X.z - C.z))
                    n = _cross(vCO, vCX)
                n = _unit_vec(n)

            p = _unit_vec(_cross(n, vCO))
            if math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) < 1e-8:
                warnings.append("degenerate_inplane")
                tmp = (0.0, 1.0, 0.0)
                p = _unit_vec(_cross(tmp, vCO))
            self._log(f"[L2] plane normal={n} in-plane vec={p}")

            phi_deg = 180.0 - self.bd_angle_deg
            phi = math.radians(phi_deg)
            base1 = (-vCO[0], -vCO[1], -vCO[2])
            self._log(f"[L2] bd_angle={self.bd_angle_deg} phi_deg={phi_deg}")

            faces = [("Re", +1.0), ("Si", -1.0)]
            if self.wobble_steps <= 1:
                wobble_angles = [0.0]
            else:
                wobble_angles = [
                    -self.wobble_deg + (2 * self.wobble_deg) * i / (self.wobble_steps - 1)
                    for i in range(self.wobble_steps)
                ]
            d_samples: List[float] = []
            t = self.dmin_A
            while t <= self.dmax_A + 1e-9:
                d_samples.append(float(t))
                t += self.step_A
            self._log(f"[L2] wobble_angles={wobble_angles}")
            self._log(f"[L2] d_samples={d_samples}")

            exclude = {c_idx, o_idx, x_idx}
            # Exclude first-shell heavy neighbors of carbonyl carbon (acyl substituent)
            c_atom = mol_use.GetAtomWithIdx(c_idx)
            for nbr in c_atom.GetNeighbors():
                n_idx = nbr.GetIdx()
                if n_idx in (o_idx, x_idx):
                    continue
                if nbr.GetAtomicNum() == 1:
                    continue
                exclude.add(n_idx)
            heavy_atoms = [
                (i, mol_use.GetAtomWithIdx(i))
                for i in range(mol_use.GetNumAtoms())
                if mol_use.GetAtomWithIdx(i).GetAtomicNum() != 1
            ]

            sasa = _calc_sasa_per_atom(mol_use)
            sasa_reactive = None
            if sasa is not None:
                sasa_reactive = float(sasa[c_idx] + sasa[o_idx] + sasa[x_idx])
            self._log(f"[L2] sasa_reactive={sasa_reactive}")

            for face_name, sgn in faces:
                a0 = _unit_vec(
                    (
                        math.cos(phi) * base1[0] + math.sin(phi) * (sgn * p[0]),
                        math.cos(phi) * base1[1] + math.sin(phi) * (sgn * p[1]),
                        math.cos(phi) * base1[2] + math.sin(phi) * (sgn * p[2]),
                    )
                )

                for wob_deg in wobble_angles:
                    Rmat = _rotation_matrix(vCO, math.radians(float(wob_deg)))
                    a = _unit_vec(_mat_vec(Rmat, a0))

                    min_clearance = 1e9
                    soft_energy_raw = 0.0
                    polar_hits = 0
                    nonpolar_hits = 0
                    worst_atom_idx: Optional[int] = None
                    worst_atom_dist: Optional[float] = None

                    for d in d_samples:
                        P = (C.x + a[0] * d, C.y + a[1] * d, C.z + a[2] * d)

                        for i, atom in heavy_atoms:
                            if i in exclude:
                                continue

                            Ai = conf.GetAtomPosition(i)
                            vCA = (Ai.x - C.x, Ai.y - C.y, Ai.z - C.z)
                            if math.sqrt(vCA[0] * vCA[0] + vCA[1] * vCA[1] + vCA[2] * vCA[2]) < 1e-8:
                                continue

                            if _angle_deg_vec(vCA, a) > self.cone_half_angle_deg:
                                continue

                            dx = P[0] - Ai.x
                            dy = P[1] - Ai.y
                            dz = P[2] - Ai.z
                            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                            clearance = dist - (_vdw_radius(atom) + self.probe_radius_A)
                            if clearance < min_clearance:
                                min_clearance = clearance
                                worst_atom_idx = i
                                worst_atom_dist = dist

                            if clearance < 0.0:
                                overlap = -clearance
                                e = (overlap / max(1e-9, self.softness_A)) ** self.repulsion_power
                                soft_energy_raw += float(e)

                            z = atom.GetAtomicNum()
                            if z in (7, 8):
                                polar_hits += 1
                            elif z == 6:
                                nonpolar_hits += 1

                    soft_energy_capped = float(min(self.energy_cap, soft_energy_raw))

                    denom = max(1, polar_hits + nonpolar_hits)
                    polarity = polar_hits / denom

                    clearance_term = _sigmoid((min_clearance - 0.2) / 0.25)
                    energy_term = math.exp(-soft_energy_capped / 8.0)

                    if sasa_reactive is None:
                        sasa_term = 0.5
                    else:
                        sasa_term = max(0.0, min(1.0, (sasa_reactive - self.sasa_floor_A2) / 10.0))

                    composite = (
                        0.55 * float(clearance_term)
                        + 0.35 * float(energy_term)
                        + 0.10 * float(sasa_term)
                    )

                    self._log(
                        f"[L2] conf={conf_id} face={face_name} wobble={wob_deg:+.1f} "
                        f"min_clearance={min_clearance:+.2f} soft_energy={soft_energy_capped:.2f} "
                        f"polarity={polarity:.2f} clearance_term={clearance_term:.2f} "
                        f"energy_term={energy_term:.2f} sasa_term={sasa_term:.2f} "
                        f"composite={composite:.3f}"
                    )

                    if (best is None) or (composite > best):
                        best = float(composite)
                        best_pack = (
                            int(conf_id),
                            face_name,
                            float(wob_deg),
                            min_clearance,
                            soft_energy_capped,
                            soft_energy_raw,
                            polarity,
                            worst_atom_idx,
                            worst_atom_dist,
                            sasa_reactive,
                        )

        (
            best_conf_id,
            face_name,
            wob_deg,
            min_clearance,
            soft_energy,
            soft_energy_raw,
            polarity,
            worst_atom_idx,
            worst_atom_dist,
            sasa_reactive,
        ) = best_pack
        self._log(
            f"[L2] best conf={best_conf_id} face={face_name} wobble={wob_deg:+.1f} "
            f"min_clearance={min_clearance:+.2f} soft_energy={soft_energy:.2f} "
            f"polarity={polarity:.2f} composite={best:.3f}"
        )

        OVERLAP_TOL_A = 0.30
        E_MAX = 15.0

        if min_clearance > 1e8:
            min_clearance = self.clearance_pass_A
            warnings.append("no_blockers_in_cone")

        hard_fail = False
        if min_clearance < -OVERLAP_TOL_A:
            hard_fail = True
            warnings.append("hard_overlap_fail")
        if soft_energy_raw > E_MAX:
            hard_fail = True
            warnings.append("corridor_energy_fail")

        passed = (not hard_fail) if self.protein_aware else True

        score = float(best)
        if self.protein_aware and not passed:
            score = min(score, 0.15)

        confidence = 0.80
        if sasa_reactive is None:
            warnings.append("sasa_unavailable")
            confidence -= 0.05
        if "hard_overlap_fail" in warnings or "corridor_energy_fail" in warnings:
            confidence -= 0.10
        self._log(f"[L2] passed={passed} score={score:.3f} confidence={confidence:.2f} warnings={warnings}")

        worst_element = None
        worst_side = None
        if worst_atom_idx is not None:
            try:
                worst_element = mol_use.GetAtomWithIdx(int(worst_atom_idx)).GetSymbol()
                worst_side = _classify_blocker(int(worst_atom_idx))
            except Exception:
                worst_element = None
                worst_side = None
        if worst_atom_idx is not None:
            self._log(
                f"[L2] worst_blocker idx={worst_atom_idx} element={worst_element} "
                f"side={worst_side} dist={None if worst_atom_dist is None else f'{worst_atom_dist:.2f}'}"
            )

        msg = (
            f"Env sterics (L2): best_face={face_name}, wobble={wob_deg:+.1f} deg, "
            f"min_clearance={min_clearance:+.2f} A, soft_energy={soft_energy:.2f}, "
            f"corridor_polarity={polarity:.2f}"
        )
        if sasa_reactive is not None:
            msg += f", reactive_SASA={sasa_reactive:.1f} A^2"
        if worst_atom_idx is not None:
            msg += (
                f", worst_blocker={worst_element}{worst_atom_idx} "
                f"({worst_side or 'unknown'})"
            )
        msg += "."

        return Level2EnvCPTResult(
            passed=passed,
            score=score,
            confidence=float(max(0.0, min(1.0, confidence))),
            best_face=face_name,
            best_wobble_deg=wob_deg,
            min_clearance_A=float(min_clearance),
            soft_steric_energy=float(soft_energy),
            sasa_reactive_A2=sasa_reactive,
            corridor_polarity_score=float(polarity),
            message=msg,
            warnings=warnings,
            soft_steric_energy_raw=float(soft_energy_raw),
            soft_steric_energy_capped=float(soft_energy),
            worst_blocker_atom_idx=worst_atom_idx,
            worst_blocker_element=worst_element,
            worst_blocker_distance_A=float(worst_atom_dist) if worst_atom_dist is not None else None,
            worst_blocker_clearance_A=float(min_clearance),
            worst_blocker_side=worst_side,
            best_conf_id=best_conf_id,
        )
