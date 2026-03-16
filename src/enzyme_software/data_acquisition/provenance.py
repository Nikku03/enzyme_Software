from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Provenance:
    source: str
    source_id: Optional[str]
    url: Optional[str]
    retrieved_at: str
    license_hint: Optional[str]
    sha256: Optional[str]
