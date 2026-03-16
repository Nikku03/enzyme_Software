from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class Collector(ABC):
    """Abstract collector interface for data acquisition sources."""

    @abstractmethod
    def list_tasks(self) -> Iterable[Any]:
        """Return tasks/queries to fetch from the source."""

    @abstractmethod
    def fetch(self, task: Any) -> Any:
        """Fetch raw data for a given task."""

    @abstractmethod
    def parse(self, raw: Any) -> Any:
        """Parse raw payload into a structured intermediate form."""

    @abstractmethod
    def normalize(self, parsed: Any) -> Any:
        """Normalize parsed data into canonical dataclasses."""
