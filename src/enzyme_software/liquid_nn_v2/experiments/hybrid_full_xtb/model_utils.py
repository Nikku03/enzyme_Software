from __future__ import annotations

from pathlib import Path
from typing import Dict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


MANUAL_ATOM_PROJECTION_KEY = "base_lnn.impl.manual_priors.atom_feature_proj.weight"


if TORCH_AVAILABLE:
    def expand_manual_atom_projection(state_dict: Dict[str, object], new_input_dim: int) -> Dict[str, object]:
        weight = state_dict.get(MANUAL_ATOM_PROJECTION_KEY)
        if weight is None or not hasattr(weight, "shape"):
            return state_dict
        if int(weight.shape[1]) >= int(new_input_dim):
            return state_dict
        expanded = torch.zeros((int(weight.shape[0]), int(new_input_dim)), dtype=weight.dtype)
        expanded[:, : int(weight.shape[1])] = weight
        state_dict = dict(state_dict)
        state_dict[MANUAL_ATOM_PROJECTION_KEY] = expanded
        return state_dict


    def load_full_xtb_warm_start(model, checkpoint_path: str | Path, device, *, new_manual_atom_dim: int) -> None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = payload.get("model_state_dict") or payload
        expanded = expand_manual_atom_projection(state_dict, new_manual_atom_dim)
        model.load_state_dict(expanded, strict=False)
else:  # pragma: no cover
    def expand_manual_atom_projection(*args, **kwargs):
        require_torch()

    def load_full_xtb_warm_start(*args, **kwargs):
        require_torch()
