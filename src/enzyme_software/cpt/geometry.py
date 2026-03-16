from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def _norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _mul(v, s: float):
    return (v[0]*s, v[1]*s, v[2]*s)


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v, eps=1e-12):
    n = _norm(v)
    if n < eps:
        return (0.0, 0.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def angle_deg(v1, v2) -> float:
    """Angle between vectors."""
    u1 = _unit(v1)
    u2 = _unit(v2)
    c = max(-1.0, min(1.0, _dot(u1, u2)))
    return math.degrees(math.acos(c))


def get_atom_pos(conf: Chem.Conformer, idx: int) -> Tuple[float, float, float]:
    p = conf.GetAtomPosition(idx)
    return (float(p.x), float(p.y), float(p.z))


def vdw_radius(atom: Chem.Atom) -> float:
    """Simple vdw radii (Å) fallback if not in table."""
    z = atom.GetAtomicNum()
    table = {
        1: 1.20,
        6: 1.70,
        7: 1.55,
        8: 1.52,
        9: 1.47,
        15: 1.80,
        16: 1.80,
        17: 1.75,
        35: 1.85,
        53: 1.98,
    }
    return table.get(z, 1.80)


@dataclass(frozen=True)
class AttackGeometry:
    """Derived geometry for nucleophilic attack around a carbonyl-like center."""
    c_idx: int
    o_idx: int
    x_idx: int
    e1_CO: Tuple[float, float, float]
    e2_inplane: Tuple[float, float, float]
    e3_normal: Tuple[float, float, float]
    ideal_attack_dir: Tuple[float, float, float]
    theta_deg: float


def build_attack_geometry(
    mol3d: Chem.Mol,
    c_idx: int,
    o_idx: int,
    x_idx: int,
    theta_deg: float = 107.0
) -> AttackGeometry:
    """
    Construct a local orthonormal frame around carbonyl carbon:
      e1 = unit(C->O)
      e2 = unit(projection of C->X onto plane orthogonal to e1)
      e3 = e1 x e2
    Ideal attack direction: vector making theta with e1, lying in (e1,e2) plane.
    """
    conf = mol3d.GetConformer()
    C = get_atom_pos(conf, c_idx)
    O = get_atom_pos(conf, o_idx)
    X = get_atom_pos(conf, x_idx)

    vCO = _sub(O, C)
    vCX = _sub(X, C)
    e1 = _unit(vCO)

    vCX_para = _mul(e1, _dot(vCX, e1))
    vCX_perp = _sub(vCX, vCX_para)
    e2 = _unit(vCX_perp)

    if _norm(e2) < 1e-8:
        neighs = [a.GetIdx() for a in mol3d.GetAtomWithIdx(c_idx).GetNeighbors()
                  if a.GetIdx() not in (o_idx, x_idx)]
        if neighs:
            Y = get_atom_pos(conf, neighs[0])
            vCY = _sub(Y, C)
            vCY_para = _mul(e1, _dot(vCY, e1))
            vCY_perp = _sub(vCY, vCY_para)
            e2 = _unit(vCY_perp)

    e3 = _unit(_cross(e1, e2))

    th = math.radians(theta_deg)
    d = _unit(_add(_mul(e1, math.cos(th)), _mul(e2, math.sin(th))))

    return AttackGeometry(
        c_idx=c_idx, o_idx=o_idx, x_idx=x_idx,
        e1_CO=e1, e2_inplane=e2, e3_normal=e3,
        ideal_attack_dir=d,
        theta_deg=theta_deg
    )


def steric_clashes_along_ray(
    mol3d: Chem.Mol,
    origin_idx: int,
    direction: Tuple[float, float, float],
    start: float = 1.6,
    end: float = 3.2,
    step: float = 0.2,
    probe_radius: float = 1.4,
    ignore_indices: Optional[List[int]] = None
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Approximate steric occlusion by sampling points along a ray and checking vdw overlaps.
    Returns: (clash_count, min_clearance, per_atom_min_clearance)
    """
    if ignore_indices is None:
        ignore_indices = []
    ignore = set(ignore_indices)

    conf = mol3d.GetConformer()
    O = get_atom_pos(conf, origin_idx)
    d = _unit(direction)

    atoms = [a for a in mol3d.GetAtoms() if a.GetIdx() not in ignore]
    atom_pos = {a.GetIdx(): get_atom_pos(conf, a.GetIdx()) for a in atoms}
    atom_rad = {a.GetIdx(): vdw_radius(a) for a in atoms}

    clash_count = 0
    min_clearance = 1e9
    per_atom_min = {a.GetIdx(): 1e9 for a in atoms}

    t = start
    while t <= end + 1e-9:
        P = _add(O, _mul(d, t))
        for idx, pos in atom_pos.items():
            dx = P[0] - pos[0]
            dy = P[1] - pos[1]
            dz = P[2] - pos[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            clearance = dist - (atom_rad[idx] + probe_radius)
            if clearance < 0.0:
                clash_count += 1
            if clearance < min_clearance:
                min_clearance = clearance
            if clearance < per_atom_min[idx]:
                per_atom_min[idx] = clearance
        t += step

    per_atom_list = sorted([(k, v) for k, v in per_atom_min.items()], key=lambda x: x[1])
    return clash_count, float(min_clearance), per_atom_list[:8]
