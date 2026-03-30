from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


MANUAL_ATOM_PROJECTION_KEY = "base_lnn.impl.manual_priors.atom_feature_proj.weight"
INPUT_PROJ_KEY = "base_lnn.impl.shared_encoder.input_proj.0.weight"


if TORCH_AVAILABLE:
    def _compat_types() -> Tuple[object, object]:
        uninitialized_param = getattr(torch.nn.parameter, "UninitializedParameter", ())
        uninitialized_buffer = getattr(torch.nn.parameter, "UninitializedBuffer", ())
        return uninitialized_param, uninitialized_buffer


    def _sanitize_state_dict_for_model(model, state_dict: Dict[str, object]) -> tuple[Dict[str, object], Dict[str, int]]:
        filtered: Dict[str, object] = {}
        stats = {
            "loaded": 0,
            "missing": 0,
            "nonfinite": 0,
            "mismatch": 0,
        }
        current_state = model.state_dict()
        uninitialized_param, uninitialized_buffer = _compat_types()
        for key, value in state_dict.items():
            current = current_state.get(key)
            if current is None:
                stats["missing"] += 1
                continue
            if hasattr(value, "is_floating_point") and bool(value.is_floating_point()):
                if not bool(torch.isfinite(value).all()):
                    stats["nonfinite"] += 1
                    continue
            if isinstance(current, (uninitialized_param, uninitialized_buffer)):
                filtered[key] = value
                stats["loaded"] += 1
                continue
            if hasattr(current, "shape") and hasattr(value, "shape") and tuple(current.shape) != tuple(value.shape):
                stats["mismatch"] += 1
                continue
            filtered[key] = value
            stats["loaded"] += 1
        return filtered, stats


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


    def expand_input_proj(state_dict: Dict[str, object], new_input_dim: int) -> Dict[str, object]:
        """Expand input_proj.0.weight (Linear in_features) for full-XTB atom dim increase."""
        weight = state_dict.get(INPUT_PROJ_KEY)
        if weight is None or not hasattr(weight, "shape"):
            return state_dict
        # weight shape: (out_features, in_features); expand in_features
        if int(weight.shape[1]) >= int(new_input_dim):
            return state_dict
        expanded = torch.zeros((int(weight.shape[0]), int(new_input_dim)), dtype=weight.dtype)
        expanded[:, : int(weight.shape[1])] = weight
        state_dict = dict(state_dict)
        state_dict[INPUT_PROJ_KEY] = expanded
        return state_dict


    def load_full_xtb_warm_start(
        model,
        checkpoint_path: str | Path,
        device,
        *,
        new_manual_atom_dim: int,
        new_atom_input_dim: int | None = None,
    ) -> Dict[str, int]:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = payload.get("model_state_dict") or payload
        expanded = expand_manual_atom_projection(state_dict, new_manual_atom_dim)
        if new_atom_input_dim is not None:
            expanded = expand_input_proj(expanded, new_atom_input_dim)
        sanitized, stats = _sanitize_state_dict_for_model(model, expanded)
        model.load_state_dict(sanitized, strict=False)
        return stats
else:  # pragma: no cover
    def expand_manual_atom_projection(*args, **kwargs):
        require_torch()

    def load_full_xtb_warm_start(*args, **kwargs):
        require_torch()
