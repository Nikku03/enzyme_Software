from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Source:
    name: str

    def build_download_url(self, symbol: str, interval: str = "d") -> str:
        raise NotImplementedError

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.lower()


@dataclass(frozen=True)
class StooqSource(Source):
    name: str = "stooq"

    def normalize_symbol(self, symbol: str) -> str:
        base = symbol.strip().lower()
        if "." in base:
            return base
        return f"{base}.us"

    def build_download_url(self, symbol: str, interval: str = "d") -> str:
        norm = self.normalize_symbol(symbol)
        return f"https://stooq.com/q/d/l/?s={norm}&i={interval}"


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: Dict[str, Source] = {
            "stooq": StooqSource(),
        }

    def get(self, name: str) -> Optional[Source]:
        return self._sources.get(name.lower())

    def register(self, source: Source) -> None:
        self._sources[source.name.lower()] = source

    def list_sources(self) -> Dict[str, Source]:
        return dict(self._sources)


REGISTRY = SourceRegistry()
