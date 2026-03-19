from __future__ import annotations

import math
from typing import Dict, List, Tuple

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.cahml.config import PHYSICS_RULES, SITE_SMARTS_PATTERNS


class PhysicsConstraints:
    def __init__(self, *, boost_factor: float = 1.5, penalty_factor: float = 0.3, block_value: float = -10.0):
        require_torch()
        self.boost_factor = float(boost_factor)
        self.penalty_factor = float(penalty_factor)
        self.block_value = float(block_value)
        self.pattern_names = list(SITE_SMARTS_PATTERNS.keys())
        self.pattern_index = {name: idx for idx, name in enumerate(self.pattern_names)}

    def _apply_logit_scale(self, value: torch.Tensor, factor: float) -> torch.Tensor:
        if factor <= 0.0:
            return torch.full_like(value, self.block_value)
        return value + math.log(max(factor, 1.0e-6))

    def apply_constraints(self, atom_features_raw: torch.Tensor, smarts_matches: torch.Tensor, site_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        constrained = site_scores.clone()
        blocked_atoms: List[int] = []
        boosted_atoms: List[int] = []
        penalized_atoms: List[int] = []
        rules: List[Dict[str, object]] = []

        for atom_idx in range(min(constrained.shape[0], atom_features_raw.shape[0])):
            atomic_num = int(round(float(atom_features_raw[atom_idx, 0].item() * 20.0)))
            total_h = float(atom_features_raw[atom_idx, 2].item() * 4.0)
            if atomic_num == 6 and total_h <= 0.0:
                constrained[atom_idx] = self.block_value
                blocked_atoms.append(atom_idx)
                rules.append({"atom": atom_idx, "rule": "no_hydrogen", "action": "block", "reason": "No available hydrogen"})
                continue
            for pattern_name, (action, factor, reason) in PHYSICS_RULES.items():
                pattern_idx = self.pattern_index.get(pattern_name)
                if pattern_idx is None or atom_idx >= smarts_matches.shape[0]:
                    continue
                if smarts_matches[atom_idx, pattern_idx].item() <= 0.5:
                    continue
                if action == "block":
                    constrained[atom_idx] = self.block_value
                    blocked_atoms.append(atom_idx)
                elif action == "boost":
                    constrained[atom_idx] = self._apply_logit_scale(constrained[atom_idx], factor * self.boost_factor / 1.5)
                    boosted_atoms.append(atom_idx)
                elif action == "penalize":
                    constrained[atom_idx] = self._apply_logit_scale(constrained[atom_idx], max(factor * self.penalty_factor / 0.3, 1.0e-3))
                    penalized_atoms.append(atom_idx)
                rules.append({"atom": atom_idx, "rule": pattern_name, "action": action, "reason": reason})

        return constrained, {
            "n_rules_applied": len(rules),
            "blocked_atoms": blocked_atoms,
            "boosted_atoms": boosted_atoms,
            "penalized_atoms": penalized_atoms,
            "rules": rules,
        }
