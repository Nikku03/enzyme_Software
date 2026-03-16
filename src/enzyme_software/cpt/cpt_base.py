from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from rdkit import Chem

from .types import CPTResult


class CPT(ABC):
    """Base class for CPTs."""
    cpt_id: str
    version: str = "v1"
    fidelity: str = "geometric_basic"

    @abstractmethod
    def run(
        self,
        *,
        mechanism_id: str,
        mol3d: Chem.Mol,
        role_to_idx: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None
    ) -> CPTResult:
        raise NotImplementedError
