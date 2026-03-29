"""
Wave-engine fork for the analogical memory bank.

The wave path must not reuse continuous bank caches built before wave-routed
node multivectors reached retrieval. This subclass therefore invalidates legacy
continuous caches automatically and swaps in the wave-aware PGW transporter.
"""
from __future__ import annotations

from pathlib import Path

import torch

from nexus.reasoning.hyperbolic_memory import HyperbolicMemoryBank as ClassicHyperbolicMemoryBank

from .pgw_transport import PGWTransporter as WavePGWTransporter


class HyperbolicMemoryBank(ClassicHyperbolicMemoryBank):
    WAVE_CONTINUOUS_CACHE_VERSION = "wave_v1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.transport_backend == "pgw":
            self.pgw = WavePGWTransporter(device=self.device)

    def _load_continuous_bank_cache(self, cache_path: Path) -> bool:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  Continuous bank cache load failed: {type(exc).__name__}: {exc}")
            return False

        cache_version = payload.get("version")
        cache_engine = str(payload.get("engine", "classic")).strip().lower()
        if cache_version != self.WAVE_CONTINUOUS_CACHE_VERSION or cache_engine != "wave":
            print(
                "  Continuous bank cache mismatch: wave engine requires a fresh "
                "continuous bank rebuild."
            )
            return False
        return super()._load_continuous_bank_cache(cache_path)

    def _save_continuous_bank_cache(self, cache_path: Path) -> None:
        if self.memory_embeddings is None or self.memory_projected_mask is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        projected_mask_cpu = self.memory_projected_mask.detach().cpu().clone().view(-1)
        invalid_projected = 0
        for idx, (is_projected, mv) in enumerate(zip(projected_mask_cpu.tolist(), self.historical_node_multivectors)):
            if is_projected and mv is None:
                projected_mask_cpu[idx] = False
                invalid_projected += 1
        payload = {
            "version": self.WAVE_CONTINUOUS_CACHE_VERSION,
            "engine": "wave",
            "signatures": self._bank_signatures(),
            "memory_embeddings": self.memory_embeddings.detach().cpu(),
            "memory_projected_mask": projected_mask_cpu,
            "historical_node_multivectors": [
                None if mv is None else mv.detach().cpu()
                for mv in self.historical_node_multivectors
            ],
        }
        torch.save(payload, cache_path)
        if invalid_projected > 0:
            print(
                "  Saved wave continuous bank cache with "
                f"{invalid_projected} downgraded projected entries (missing node multivectors)."
            )
        print(f"  Saved wave continuous bank cache → {cache_path}")


__all__ = ["HyperbolicMemoryBank"]
