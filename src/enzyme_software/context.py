from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OperationalConstraints:
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    temperature_c: Optional[float] = None
    metals_allowed: Optional[bool] = None
    oxidation_allowed: Optional[bool] = None
    host: Optional[str] = None
    receptor_pdbqt: Optional[str] = None
    receptor_pdb_id: Optional[str] = None
    cyp_isoform: Optional[str] = None
    enable_vina: Optional[bool] = None
    enable_openmm: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ph_min": self.ph_min,
            "ph_max": self.ph_max,
            "temperature_c": self.temperature_c,
            "metals_allowed": self.metals_allowed,
            "oxidation_allowed": self.oxidation_allowed,
            "host": self.host,
            "receptor_pdbqt": self.receptor_pdbqt,
            "receptor_pdb_id": self.receptor_pdb_id,
            "cyp_isoform": self.cyp_isoform,
            "enable_vina": self.enable_vina,
            "enable_openmm": self.enable_openmm,
        }


@dataclass
class PipelineContext:
    smiles: str
    target_bond: str
    requested_output: Optional[str] = None
    trap_target: Optional[str] = None
    constraints: OperationalConstraints = field(default_factory=OperationalConstraints)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "target_bond": self.target_bond,
            "requested_output": self.requested_output,
            "trap_target": self.trap_target,
            "constraints": self.constraints.to_dict(),
            "data": self.data,
        }
