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
        self._wave_query_embedding: torch.Tensor | None = None
        if self.transport_backend == "pgw":
            self.pgw = WavePGWTransporter(device=self.device)

    def _dense_projected_similarity(self, query_embedding: torch.Tensor) -> torch.Tensor:
        if self.memory_embeddings is None:
            raise RuntimeError("Call populate_from_mols() before dense projected retrieval.")
        target_dim = int(self.memory_embeddings.size(-1))
        prepared_query = torch.as_tensor(query_embedding, dtype=torch.float32, device=self.device).view(-1)
        if prepared_query.numel() < target_dim:
            prepared_query = torch.nn.functional.pad(prepared_query, (0, target_dim - prepared_query.numel()))
        elif prepared_query.numel() > target_dim:
            prepared_query = prepared_query[:target_dim]
        q_embed_h = self._project_inside_ball(prepared_query.view(1, -1))
        distances = self._poincare_distance(q_embed_h, self.memory_embeddings).view(-1)
        return self._projected_similarity_from_distance(distances)

    def _projected_shortlist_size(self, candidate_count: int) -> int:
        return min(max(self.tanimoto_shortlist_k * 2, 24), min(int(candidate_count), 96))

    def _projected_similarity_from_distance(self, distances: torch.Tensor) -> torch.Tensor:
        dist = torch.as_tensor(distances, dtype=torch.float32, device=self.device).view(-1)
        if dist.numel() == 0:
            return dist
        center = dist.median()
        spread = (dist - center).abs().mean().clamp_min(0.05)
        return torch.sigmoid((center - dist) / spread)

    def _retrieval_confidence_from_shortlist(
        self,
        blended_similarity: torch.Tensor,
        shortlist_scores: torch.Tensor,
        *,
        best_local_index: int,
    ) -> float:
        best_local = int(best_local_index)
        sims = torch.as_tensor(blended_similarity, dtype=torch.float32, device=self.device).view(-1)
        if sims.numel() == 0:
            return 0.0
        best_sim = float(sims[best_local].detach().item())
        if sims.numel() > 1:
            sorted_sim = torch.sort(sims.detach(), descending=True).values
            second_sim = float(sorted_sim[1].item())
        else:
            second_sim = 0.0
        margin = max(best_sim - second_sim, 0.0)
        probs = torch.softmax(torch.as_tensor(shortlist_scores, dtype=torch.float32, device=self.device).view(-1), dim=0)
        top_prob = float(probs[best_local].item()) if probs.numel() > best_local else 0.0
        confidence = 0.45 * best_sim + 0.30 * margin + 0.25 * top_prob
        return float(min(max(confidence, 0.0), 1.0))

    def _tanimoto_similarity(self, q_fp: torch.Tensor) -> torch.Tensor:
        if self._wave_query_embedding is not None and self.memory_embeddings is not None:
            try:
                return self._dense_projected_similarity(self._wave_query_embedding)
            except Exception:
                pass
        return super()._tanimoto_similarity(q_fp)

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

    def retrieve_and_transport(self, *args, query_embedding=None, **kwargs):
        self._wave_query_embedding = None
        if query_embedding is not None:
            try:
                self._wave_query_embedding = torch.as_tensor(
                    query_embedding, dtype=torch.float32, device=self.device
                )
            except Exception:
                self._wave_query_embedding = None
        try:
            return super().retrieve_and_transport(*args, query_embedding=query_embedding, **kwargs)
        finally:
            self._wave_query_embedding = None


__all__ = ["HyperbolicMemoryBank"]
