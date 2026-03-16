from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def _dist(a: Vec3, b: Vec3) -> float:
    return _norm(_sub(a, b))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _angle_deg(u: Vec3, v: Vec3) -> float:
    cu = _unit(u)
    cv = _unit(v)
    c = _clamp(_dot(cu, cv), -1.0, 1.0)
    return math.degrees(math.acos(c))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnvPoint:
    pos: Vec3
    kind: str  # "donor", "charge", "hydrophobe"
    label: str = ""
    weight: float = 1.0
    q: float = 0.0
    direction: Optional[Vec3] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class EnvContext:
    donors: List[EnvPoint]
    charges: List[EnvPoint]
    hydrophobes: List[EnvPoint]
    meta: Optional[Dict[str, Any]] = None

    def add(self, p: EnvPoint) -> None:
        if p.kind == "donor":
            self.donors.append(p)
        elif p.kind in ("pos", "neg", "charge"):
            self.charges.append(p)
        elif p.kind in ("hydrophobe", "nonpolar"):
            self.hydrophobes.append(p)
        else:
            # Default: treat unknowns as charges/polar points
            self.charges.append(p)

    @staticmethod
    def ideal_oxyanion_hole_from_vectors(
        carbonyl_o_xyz: Vec3,
        attack_vector: Vec3,
        donor_distance_A: float = 2.9,
        perp_weight: float = 0.7,
        axial_weight: float = 0.3,
        label_prefix: str = "ideal_donor",
    ) -> "EnvContext":
        """
        Build an idealized oxyanion hole from an O position and attack vector.
        Produces two donor EnvPoints with direction pointing toward the oxygen.
        """
        o = carbonyl_o_xyz
        attack = _unit(attack_vector)
        # Choose a perpendicular axis
        perp = (
            attack[1] * 0.0 - attack[2] * 1.0,
            attack[2] * 0.0 - attack[0] * 0.0,
            attack[0] * 1.0 - attack[1] * 0.0,
        )
        if _norm(perp) < 0.1:
            perp = (
                attack[1] * 0.0 - attack[2] * 0.0,
                attack[2] * 1.0 - attack[0] * 0.0,
                attack[0] * 0.0 - attack[1] * 1.0,
            )
        perp = _unit(perp)

        donor1_pos = _add(
            o,
            _mul(_add(_mul(attack, -axial_weight), _mul(perp, +perp_weight)), donor_distance_A),
        )
        donor2_pos = _add(
            o,
            _mul(_add(_mul(attack, -axial_weight), _mul(perp, -perp_weight)), donor_distance_A),
        )

        d1 = EnvPoint(
            pos=donor1_pos,
            kind="donor",
            label=f"{label_prefix}_1",
            weight=1.0,
            direction=_sub(o, donor1_pos),
        )
        d2 = EnvPoint(
            pos=donor2_pos,
            kind="donor",
            label=f"{label_prefix}_2",
            weight=1.0,
            direction=_sub(o, donor2_pos),
        )

        ctx = EnvContext(donors=[], charges=[], hydrophobes=[], meta={"ideal_oxyanion": True})
        ctx.add(d1)
        ctx.add(d2)
        return ctx

    @staticmethod
    def ideal_env_from_fragment(
        mol3d,
        role_to_idx: Dict[str, int],
        donor_distance_A: float = 2.9,
        charge_distance_A: float = 3.5,
        perp_weight: float = 0.7,
        axial_weight: float = 0.3,
        label_prefix: str = "ideal",
    ) -> "EnvContext":
        """
        Build a pseudo enzyme env in the fragment frame:
          - 2 donors (oxyanion hole)
          - 1 positive point near the oxyanion direction
        """
        conf = mol3d.GetConformer(0)
        o_idx = role_to_idx["carbonyl_o"]
        c_idx = role_to_idx["carbonyl_c"]
        O = conf.GetAtomPosition(o_idx)
        C = conf.GetAtomPosition(c_idx)

        attack_vec = (C.x - O.x, C.y - O.y, C.z - O.z)
        attack_vec = _unit(_mul(attack_vec, -1.0))

        base_ctx = EnvContext.ideal_oxyanion_hole_from_vectors(
            carbonyl_o_xyz=(O.x, O.y, O.z),
            attack_vector=attack_vec,
            donor_distance_A=donor_distance_A,
            perp_weight=perp_weight,
            axial_weight=axial_weight,
            label_prefix=f"{label_prefix}_donor",
        )

        # Add a positive point roughly along the oxyanion direction.
        pos_xyz = _add((O.x, O.y, O.z), _mul(attack_vec, charge_distance_A))
        base_ctx.add(
            EnvPoint(
                pos=pos_xyz,
                kind="pos",
                label=f"{label_prefix}_pos",
                weight=1.0,
                q=+1.0,
            )
        )

        if base_ctx.meta is None:
            base_ctx.meta = {}
        base_ctx.meta["ideal_env"] = True
        return base_ctx

    @staticmethod
    def pseudo_oxyanion_hole(
        fragment_3d,
        role_to_idx: Dict[str, int],
        donor_distance_A: float = 2.9,
        spread_A: float = 1.0,
    ) -> "EnvContext":
        """
        Test-only: add 2 donor points in an oxyanion-hole-ish region.
        Returns an EnvContext with donors populated.
        """
        conf = fragment_3d.GetConformer(0)
        o_idx = role_to_idx["carbonyl_o"]
        c_idx = role_to_idx["carbonyl_c"]

        O = tuple(conf.GetAtomPosition(o_idx))
        C = tuple(conf.GetAtomPosition(c_idx))

        o_to_c = _unit(_sub(C, O))
        outward = _mul(o_to_c, -1.0)

        ref = (1.0, 0.0, 0.0)
        if abs(_dot(ref, outward)) > 0.9:
            ref = (0.0, 1.0, 0.0)

        px = (
            outward[1] * ref[2] - outward[2] * ref[1],
            outward[2] * ref[0] - outward[0] * ref[2],
            outward[0] * ref[1] - outward[1] * ref[0],
        )
        perp = _unit(px)

        p1 = _add(O, _add(_mul(outward, donor_distance_A), _mul(perp, +spread_A / 2)))
        p2 = _add(O, _add(_mul(outward, donor_distance_A), _mul(perp, -spread_A / 2)))

        donors = [
            EnvPoint(pos=p1, kind="donor", q=0.0, meta={"label": "pseudo_NH_1"}),
            EnvPoint(pos=p2, kind="donor", q=0.0, meta={"label": "pseudo_NH_2"}),
        ]
        return EnvContext(donors=donors, charges=[], hydrophobes=[], meta={"pseudo": True})


@dataclass
class Level3EnvCPTResult:
    passed: bool
    score: float
    confidence: float
    breakdown: Dict[str, float]
    dominant_driver: str
    warnings: List[str]
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseEnvCPT:
    name: str

    def run(self, mol3d, role_to_idx, l2_best, env: EnvContext) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CPT 1 – Oxyanion Hole Geometry
# ---------------------------------------------------------------------------

class OxyanionHoleGeometryCPT(BaseEnvCPT):
    """Oxyanion hole donor geometry via env.donors."""

    name = "oxyanion_hole_geometry"

    def __init__(
        self,
        d_min_A: float = 2.6,
        d_max_A: float = 3.2,
        min_angle_deg: float = 120.0,
        require_two_donors_for_pass: bool = False,
    ) -> None:
        self.d_min_A = d_min_A
        self.d_max_A = d_max_A
        self.min_angle_deg = min_angle_deg
        self.require_two = require_two_donors_for_pass

    def run(self, mol3d, role_to_idx, l2_best, env: EnvContext) -> Dict[str, Any]:
        warnings: List[str] = []
        if env is None or not env.donors:
            return {
                "passed": False,
                "score": 0.0,
                "confidence": 0.25,
                "breakdown": {"donors_found": 0.0},
                "dominant_driver": "no_env_context",
                "warnings": ["no_env_context"],
                "data": {},
            }

        conf = mol3d.GetConformer(0)
        o_idx = role_to_idx["carbonyl_o"]
        c_idx = role_to_idx["carbonyl_c"]

        O = tuple(conf.GetAtomPosition(o_idx))
        C = tuple(conf.GetAtomPosition(c_idx))
        o_to_c = _sub(C, O)
        attack_vec_unit = None
        if isinstance(l2_best, dict):
            attack_vec_unit = l2_best.get("attack_dir")
        if attack_vec_unit is None:
            attack_vec_unit = _mul(_unit(_sub(C, O)), -1.0)
        attack_vec_unit = _unit(attack_vec_unit)

        if env is not None and env.meta and env.meta.get("debug"):
            env_points = []
            for p in env.donors:
                env_points.append({"type": "donor", "xyz": p.pos})
            for p in env.charges:
                env_points.append({"type": "charge", "xyz": p.pos})
            for p in env.hydrophobes:
                env_points.append({"type": "hydrophobe", "xyz": p.pos})
            print("\n[DEBUG] Oxyanion Hole CPT - Input Check")
            print(f"Substrate Carbonyl O position: {O}")
            print(f"Total environment points received: {len(env_points)}")
            for i, point in enumerate(env_points):
                print(f"\n  Point {i}: Type={point['type']}, Pos={point['xyz']}")
                if point["type"] == "donor":
                    dist = _dist(point["xyz"], O)
                    od = _sub(point["xyz"], O)
                    od_unit = _unit(od)
                    angle = _angle_deg(od_unit, attack_vec_unit)
                    print(f"    \u2192 Dist to O: {dist:.3f} A")
                    print(f"    \u2192 Angle vs attack: {angle:.1f} deg")
                    print(f"    \u2192 Passes basic filter? {dist < 3.5}")

        hits = []
        for p in env.donors:
            d = _dist(p.pos, O)
            if not (self.d_min_A <= d <= self.d_max_A):
                continue
            donor_to_O = _sub(O, p.pos)
            ang = _angle_deg(donor_to_O, o_to_c)
            if ang < self.min_angle_deg:
                continue
            label = (p.meta or {}).get("label", "")
            hits.append({"label": label, "dist_A": d, "angle_deg": ang, "w": 1.0})

        hits = sorted(hits, key=lambda x: (-x["angle_deg"], abs(x["dist_A"] - 2.9)))

        n = len(hits)
        if n == 0:
            score = 0.0
        elif n == 1:
            score = 0.65
        else:
            score = 1.0

        passed = (n >= 2) if self.require_two else (n >= 1)
        dominant = "two_donors" if n >= 2 else "one_donor" if n == 1 else "none"

        if n == 0:
            warnings.append("no_oxyanion_donors")

        return {
            "passed": bool(passed),
            "score": float(score),
            "confidence": 0.75,
            "breakdown": {"donors_count": float(n), "score": float(score)},
            "dominant_driver": dominant,
            "warnings": warnings,
            "data": {"hits": hits[:5]},
        }


# ---------------------------------------------------------------------------
# CPT 2 – Solvent Exposure / Corridor Polarity
# ---------------------------------------------------------------------------

class SolventExposurePolarityCPT(BaseEnvCPT):
    """
    Corridor polarity / exposure signal.
    Uses hydrophobes (negative polarity) and charges (positive polarity).
    """

    name = "solvent_exposure_polarity"

    def __init__(
        self,
        cone_half_angle_deg: float = 35.0,
        r_min_A: float = 2.0,
        r_max_A: float = 6.0,
        occ_target: float = 6.0,
        occ_tolerance: float = 6.0,
    ) -> None:
        self.cone_half_angle_deg = cone_half_angle_deg
        self.r_min_A = r_min_A
        self.r_max_A = r_max_A
        self.occ_target = occ_target
        self.occ_tolerance = occ_tolerance

    def run(self, mol3d, role_to_idx, l2_best, env: EnvContext) -> Dict[str, Any]:
        warnings: List[str] = []
        if env is None or (not env.hydrophobes and not env.charges):
            return {
                "passed": False,
                "score": 0.0,
                "confidence": 0.25,
                "breakdown": {"corridor_polarity": 0.0, "occupancy": 0.0},
                "dominant_driver": "no_env_context",
                "warnings": ["no_env_context"],
                "data": {},
            }

        conf = mol3d.GetConformer(0)
        o_idx = role_to_idx["carbonyl_o"]
        c_idx = role_to_idx["carbonyl_c"]
        O = tuple(conf.GetAtomPosition(o_idx))
        C = tuple(conf.GetAtomPosition(c_idx))

        attack_dir = l2_best.get("attack_dir") if isinstance(l2_best, dict) else None
        if attack_dir is None:
            attack_dir = _mul(_unit(_sub(C, O)), -1.0)
        attack_dir = _unit(attack_dir)

        in_cone = []
        pol_sum = 0.0
        w_sum = 0.0

        # Hydrophobes => -1 polarity
        for p in env.hydrophobes:
            v = _sub(p.pos, O)
            r = _norm(v)
            if not (self.r_min_A <= r <= self.r_max_A):
                continue
            ang = _angle_deg(v, attack_dir)
            if ang > self.cone_half_angle_deg:
                continue
            w = float((p.meta or {}).get("weight", 1.0))
            pol_sum += w * -1.0
            w_sum += w
            in_cone.append({"kind": "hydrophobe", "r_A": r, "ang_deg": ang, "w": w})

        # Charges => +1 polarity (pos/neg both make polar environment)
        for p in env.charges:
            v = _sub(p.pos, O)
            r = _norm(v)
            if not (self.r_min_A <= r <= self.r_max_A):
                continue
            ang = _angle_deg(v, attack_dir)
            if ang > self.cone_half_angle_deg:
                continue
            w = float((p.meta or {}).get("weight", 1.0))
            pol_sum += w * 1.0
            w_sum += w
            in_cone.append({"kind": "charge", "r_A": r, "ang_deg": ang, "w": w, "q": p.q})

        occupancy = w_sum
        corridor_polarity = 0.0 if w_sum < 1e-9 else _clamp(pol_sum / w_sum, -1.0, 1.0)

        polarity_score = (1.0 - corridor_polarity) / 2.0
        occ_score = math.exp(-((occupancy - self.occ_target) ** 2) / (2.0 * (self.occ_tolerance ** 2)))
        score = 0.7 * polarity_score + 0.3 * occ_score

        passed = (occupancy >= 1.5) and (corridor_polarity <= 0.5) and (score >= 0.55)
        dominant = "hydrophobic_pocket" if corridor_polarity < 0 else "polar_corridor" if corridor_polarity > 0.3 else "mixed_corridor"

        if occupancy < 1.5:
            warnings.append("low_corridor_occupancy_possible_solvent_exposed")
        if corridor_polarity > 0.7:
            warnings.append("high_corridor_polarity")

        data = {"attack_dir": attack_dir, "in_cone": in_cone}
        breakdown = {
            "corridor_polarity": float(corridor_polarity),
            "occupancy": float(occupancy),
            "polarity_score": float(polarity_score),
            "occ_score": float(occ_score),
        }

        return {
            "passed": bool(passed),
            "score": float(score),
            "confidence": 0.75,
            "breakdown": breakdown,
            "dominant_driver": dominant,
            "warnings": warnings,
            "data": data,
        }


# ---------------------------------------------------------------------------
# CPT 3 – Transition-State Charge Stabilization
# ---------------------------------------------------------------------------

class TransitionStateChargeStabilizationCPT(BaseEnvCPT):
    """Charge-field proxy near oxyanion region."""

    name = "ts_charge_stabilization"

    def __init__(
        self,
        near_min_A: float = 2.5,
        near_max_A: float = 6.0,
        align_bonus_angle_deg: float = 45.0,
    ) -> None:
        self.near_min_A = near_min_A
        self.near_max_A = near_max_A
        self.align_bonus_angle_deg = align_bonus_angle_deg

    def run(self, mol3d, role_to_idx, l2_best, env: EnvContext) -> Dict[str, Any]:
        warnings: List[str] = []
        if env is None or not env.charges:
            return {
                "passed": False,
                "score": 0.0,
                "confidence": 0.25,
                "breakdown": {"pos_support": 0.0, "neg_penalty": 0.0},
                "dominant_driver": "no_env_context",
                "warnings": ["no_env_context"],
                "data": {},
            }

        conf = mol3d.GetConformer(0)
        o_idx = role_to_idx["carbonyl_o"]
        c_idx = role_to_idx["carbonyl_c"]
        O = tuple(conf.GetAtomPosition(o_idx))
        C = tuple(conf.GetAtomPosition(c_idx))

        oxy_dir = _mul(_unit(_sub(C, O)), -1.0)

        pos_hits = []
        neg_hits = []
        pos_score_raw = 0.0
        neg_score_raw = 0.0

        for p in env.charges:
            r = _dist(p.pos, O)
            if not (self.near_min_A <= r <= self.near_max_A):
                continue
            v = _unit(_sub(p.pos, O))
            ang = _angle_deg(v, oxy_dir)
            align_bonus = 1.0 if ang <= self.align_bonus_angle_deg else 0.7
            dist_w = 1.0 / max(r * r, 1e-6)
            w = abs(float(p.q)) * dist_w * align_bonus

            if p.q >= 0.0:
                pos_score_raw += w
                pos_hits.append({"r_A": r, "ang_deg": ang, "w": w, "q": p.q})
            else:
                neg_score_raw += w
                neg_hits.append({"r_A": r, "ang_deg": ang, "w": w, "q": p.q})

        pos_support = 1.0 - math.exp(-pos_score_raw / 0.06)
        neg_penalty = 1.0 - math.exp(-neg_score_raw / 0.06)

        net = _clamp(pos_support - 0.9 * neg_penalty, -1.0, 1.0)
        score = (net + 1.0) / 2.0

        passed = (pos_support >= 0.45) and (neg_penalty <= 0.55) and (score >= 0.55)
        dominant = "pos_stabilization" if pos_support > neg_penalty else "neg_destabilization" if neg_penalty > 0.4 else "weak_field"

        if pos_support < 0.25:
            warnings.append("weak_positive_field_near_oxyanion")
        if neg_penalty > 0.5:
            warnings.append("strong_negative_field_near_oxyanion")

        data = {
            "oxy_dir": oxy_dir,
            "pos_hits": sorted(pos_hits, key=lambda x: -x["w"])[:15],
            "neg_hits": sorted(neg_hits, key=lambda x: -x["w"])[:15],
            "pos_score_raw": pos_score_raw,
            "neg_score_raw": neg_score_raw,
        }
        breakdown = {
            "pos_support": float(pos_support),
            "neg_penalty": float(neg_penalty),
            "net": float(net),
        }

        return {
            "passed": bool(passed),
            "score": float(score),
            "confidence": 0.75,
            "breakdown": breakdown,
            "dominant_driver": dominant,
            "warnings": warnings,
            "data": data,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Level3Orchestrator:
    def __init__(self, cpts: List[BaseEnvCPT], weights: Dict[str, float], pass_threshold: float = 0.60) -> None:
        self.cpts = cpts
        self.weights = weights
        self.pass_threshold = pass_threshold

    def run(self, mol3d, role_to_idx, l2_best, env: EnvContext) -> Level3EnvCPTResult:
        breakdown: Dict[str, float] = {}
        warnings: List[str] = []
        debug: Dict[str, Any] = {}

        for cpt in self.cpts:
            out = cpt.run(mol3d, role_to_idx, l2_best, env)
            breakdown[cpt.name] = float(out["score"])
            warnings += out.get("warnings", [])
            debug[cpt.name] = out.get("data", {})

        score = 0.0
        for k, v in breakdown.items():
            score += self.weights.get(k, 0.0) * v

        dominant = min(breakdown, key=lambda k: breakdown[k]) if breakdown else "none"
        passed = score >= self.pass_threshold

        return Level3EnvCPTResult(
            passed=passed,
            score=float(score),
            confidence=float(0.75),
            breakdown=breakdown,
            dominant_driver=dominant,
            warnings=warnings,
            data=debug,
        )
