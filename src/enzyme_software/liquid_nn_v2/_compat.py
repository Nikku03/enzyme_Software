"""Optional torch compatibility helpers."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for liquid_nn_v2 model/training components")


def to_python(value: Any) -> Any:
    if TORCH_AVAILABLE and hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value
