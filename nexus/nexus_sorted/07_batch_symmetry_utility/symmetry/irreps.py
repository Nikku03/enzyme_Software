from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence


DEFAULT_IRREPS = "128x0e + 128x0o + 128x1o + 128x1e + 64x2e + 64x2o"


@dataclass(frozen=True)
class O3Irrep:
    multiplicity: int
    degree: int
    parity: str

    @property
    def dim(self) -> int:
        return int(self.multiplicity) * (2 * int(self.degree) + 1)

    @property
    def key(self) -> str:
        return f"{self.degree}{self.parity}"

    @property
    def is_even(self) -> bool:
        return self.parity == "e"


_IRREP_PATTERN = re.compile(r"^\s*(\d+)x(\d+)([eo])\s*$")


def parse_irreps(spec: str | Sequence[O3Irrep]) -> List[O3Irrep]:
    if isinstance(spec, (list, tuple)):
        return [O3Irrep(int(item.multiplicity), int(item.degree), str(item.parity)) for item in spec]
    chunks = [chunk.strip() for chunk in str(spec).split("+") if chunk.strip()]
    irreps: List[O3Irrep] = []
    for chunk in chunks:
        match = _IRREP_PATTERN.match(chunk)
        if not match:
            raise ValueError(f"Invalid irrep chunk: {chunk}")
        multiplicity, degree, parity = match.groups()
        irreps.append(O3Irrep(int(multiplicity), int(degree), parity))
    return irreps
