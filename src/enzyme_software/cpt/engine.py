from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rdkit import Chem

from .evidence import EvidenceGraph
from .scorer import UnifiedMechanismScorer
from .types import MechanismProfile
from .geometric_cpts import AttackAngleCPT, StericOcclusionCPT, LeavingGroupDistanceCPT


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    cpts: List[str]


DEFAULT_MECHANISMS = [
    MechanismSpec(
        mechanism_id="serine_hydrolase",
        cpts=["attack_angle", "steric_occlusion", "leaving_group_distance"],
    ),
    MechanismSpec(
        mechanism_id="metallo_esterase",
        cpts=["attack_angle", "steric_occlusion", "leaving_group_distance"],
    ),
]


class GeometricCPTEngine:
    def __init__(self) -> None:
        self.registry = {
            "attack_angle": AttackAngleCPT(),
            "steric_occlusion": StericOcclusionCPT(),
            "leaving_group_distance": LeavingGroupDistanceCPT(),
        }
        self.scorer = UnifiedMechanismScorer()

    def evaluate(
        self,
        *,
        mol3d: Chem.Mol,
        role_to_idx: Dict[str, int],
        mechanisms: Optional[List[MechanismSpec]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[MechanismProfile]:
        mechanisms = mechanisms or DEFAULT_MECHANISMS
        graph = EvidenceGraph()

        for ms in mechanisms:
            for cpt_id in ms.cpts:
                cpt = self.registry[cpt_id]
                r = cpt.run(
                    mechanism_id=ms.mechanism_id,
                    mol3d=mol3d,
                    role_to_idx=role_to_idx,
                    extra=extra,
                )
                graph.add(r)

        profiles: List[MechanismProfile] = []
        for ms in mechanisms:
            ev = graph.get(ms.mechanism_id)
            profiles.append(self.scorer.score(ms.mechanism_id, ev))

        profiles.sort(key=lambda p: (p.feasibility_score, p.confidence), reverse=True)
        return profiles
