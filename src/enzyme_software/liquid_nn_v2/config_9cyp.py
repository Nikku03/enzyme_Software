from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES

CYP_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(ALL_CYP_CLASSES)}
IDX_TO_CYP: Dict[int, str] = {idx: name for name, idx in CYP_TO_IDX.items()}


def encode_cyp(cyp_name: str) -> int:
    return CYP_TO_IDX.get(str(cyp_name), -1)


def decode_cyp(idx: int) -> str:
    return IDX_TO_CYP.get(int(idx), "Unknown")


@dataclass
class ModelConfig9CYP(ModelConfig):
    cyp_names: Tuple[str, ...] = tuple(ALL_CYP_CLASSES)


__all__ = ["ModelConfig9CYP", "CYP_TO_IDX", "IDX_TO_CYP", "encode_cyp", "decode_cyp"]
