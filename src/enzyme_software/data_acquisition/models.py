from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from enzyme_software.data_acquisition.provenance import Provenance


@dataclass
class NormalizedCondition:
    pH: Optional[float] = None
    temperature_C: Optional[float] = None
    solvent: Optional[str] = None
    ionic_strength: Optional[float] = None
    cofactors: List[str] = field(default_factory=list)
    buffer: Optional[str] = None


@dataclass
class NormalizedMolecule:
    name: Optional[str]
    smiles: Optional[str]
    role: Optional[str] = None
    provenance: Optional[Provenance] = None


@dataclass
class NormalizedReaction:
    reaction_id: Optional[str]
    substrates: List[NormalizedMolecule] = field(default_factory=list)
    products: List[NormalizedMolecule] = field(default_factory=list)
    conditions: NormalizedCondition = field(default_factory=NormalizedCondition)
    yield_percent: Optional[float] = None
    notes: Optional[str] = None
    provenance: Optional[Provenance] = None
