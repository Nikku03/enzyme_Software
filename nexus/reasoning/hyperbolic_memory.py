"""
nexus/reasoning/hyperbolic_memory.py

Hyperbolic analogue of the ECFP4 memory bank.

Fingerprints are first L2-normalised in Euclidean space, then projected
strictly inside the Poincare ball so nearest-neighbour lookup can use the
hyperbolic geodesic distance.  Transport remains discrete and scaffold-based
via RDKit MCS, matching the existing baseline memory-bank contract.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, List, Mapping

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False

from .baseline_memory import (
    MemoryRetrievalResult,
    _extract_som_idx,
    _morgan_fp_tensor,
    build_morphism_supervision,
    scaffold_key,
)
from .pgw_transport import PGWTransporter


def _canonical_smiles(mol) -> str | None:
    try:
        base = Chem.RemoveHs(Chem.Mol(mol))
        return Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)
    except Exception:
        return None


class HyperbolicMemoryBank:
    """
    Poincare-ball memory bank with MCS-guided SoM transport.

    The external API intentionally matches BaselineMemoryBank so the trainer
    and Colab scripts can swap retrieval geometry without changing call sites.
    """

    def __init__(
        self,
        device: str = "cpu",
        *,
        curvature: float = 1.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        identity_distance_threshold: float = 1.0e-4,
        poincare_radius: float = 0.95,
        tangent_scale: float = 0.35,
        max_tangent_norm: float = 3.0,
        transport_backend: str = "pgw",
        fallback_to_mcs: bool = True,
        mcs_timeout: int = 2,
        tanimoto_prefilter_threshold: float = 0.5,
        tanimoto_shortlist_k: int = 16,
        retrieval_mix_top_k: int = 3,
        retrieval_mix_temperature: float = 0.25,
        projected_retrieval_burn_in_epochs: int = 1,
        projected_retrieval_ramp_epochs: int = 3,
        projected_query_radius_floor: float = 0.15,
    ) -> None:
        if not _RDKIT_OK:
            raise ImportError("RDKit is required for HyperbolicMemoryBank")
        self.device = device
        self.curvature = float(max(curvature, 1.0e-8))
        self.fp_radius = int(fp_radius)
        self.fp_bits = int(fp_bits)
        self.identity_distance_threshold = float(max(identity_distance_threshold, 0.0))
        self.poincare_radius = float(min(max(poincare_radius, 1.0e-3), 1.0 - 1.0e-5))
        self.tangent_scale = float(max(tangent_scale, 1.0e-6))
        self.max_tangent_norm = float(max(max_tangent_norm, 1.0e-3))
        self.transport_backend = str(transport_backend).lower()
        self.fallback_to_mcs = bool(fallback_to_mcs)
        self.mcs_timeout = int(max(mcs_timeout, 1))
        self.tanimoto_prefilter_threshold = float(min(max(tanimoto_prefilter_threshold, 0.0), 1.0))
        self.tanimoto_shortlist_k = int(max(tanimoto_shortlist_k, 1))
        self.retrieval_mix_top_k = int(max(retrieval_mix_top_k, 1))
        self.retrieval_mix_temperature = float(max(retrieval_mix_temperature, 1.0e-3))
        self.projected_retrieval_burn_in_epochs = int(max(projected_retrieval_burn_in_epochs, 0))
        self.projected_retrieval_ramp_epochs = int(max(projected_retrieval_ramp_epochs, 1))
        self.projected_query_radius_floor = float(min(max(projected_query_radius_floor, 1.0e-3), 0.95))

        self.historical_mols: List = []
        self.historical_soms: List[int] = []
        self.historical_smiles: List[str | None] = []
        self.historical_scaffolds: List[str] = []
        self.historical_heavy_atoms: List[int] = []
        self.historical_ring_counts: List[int] = []
        self.historical_shortlist_keys: List[str] = []
        self.historical_key_frequencies: List[float] = []
        self.historical_node_multivectors: List[torch.Tensor | None] = []
        self.memory_embeddings: torch.Tensor | None = None
        self.memory_fingerprints: torch.Tensor | None = None
        self.memory_bit_counts: torch.Tensor | None = None
        self.memory_projected_mask: torch.Tensor | None = None
        self.pgw: PGWTransporter | None = None
        if self.transport_backend == "pgw":
            try:
                self.pgw = PGWTransporter(device=device)
            except ImportError:
                self.transport_backend = "mcs"

    def _reset(self) -> None:
        self.historical_mols = []
        self.historical_soms = []
        self.historical_smiles = []
        self.historical_scaffolds = []
        self.historical_heavy_atoms = []
        self.historical_ring_counts = []
        self.historical_shortlist_keys = []
        self.historical_key_frequencies = []
        self.historical_node_multivectors = []
        self.memory_embeddings = None
        self.memory_fingerprints = None
        self.memory_bit_counts = None
        self.memory_projected_mask = None

    def set_device(self, device: str | torch.device) -> None:
        resolved = torch.device(device)
        self.device = str(resolved)
        if self.memory_embeddings is not None:
            self.memory_embeddings = self.memory_embeddings.to(resolved)
        if self.memory_fingerprints is not None:
            self.memory_fingerprints = self.memory_fingerprints.to(resolved)
        if self.memory_bit_counts is not None:
            self.memory_bit_counts = self.memory_bit_counts.to(resolved)
        if self.memory_projected_mask is not None:
            self.memory_projected_mask = self.memory_projected_mask.to(resolved)
        self.historical_node_multivectors = [
            None if mv is None else mv.to(resolved)
            for mv in self.historical_node_multivectors
        ]
        if self.pgw is not None:
            self.pgw.device = str(resolved)
            self.pgw.neural_approximator = self.pgw.neural_approximator.to(resolved)

    def _bank_signatures(self) -> List[str]:
        signatures: List[str] = []
        for mol, som_idx, smiles in zip(self.historical_mols, self.historical_soms, self.historical_smiles):
            canonical = smiles or ""
            signatures.append(f"{canonical}|som={som_idx}|atoms={mol.GetNumAtoms()}")
        return signatures

    def _load_continuous_bank_cache(self, cache_path: Path) -> bool:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  Continuous bank cache load failed: {type(exc).__name__}: {exc}")
            return False

        signatures = self._bank_signatures()
        cached_signatures = payload.get("signatures")
        if cached_signatures != signatures:
            print("  Continuous bank cache mismatch: signatures changed; rebuilding cache.")
            return False

        try:
            memory_embeddings = torch.as_tensor(
                payload["memory_embeddings"],
                dtype=torch.float32,
                device=self.device,
            )
            projected_mask = torch.as_tensor(
                payload["memory_projected_mask"],
                dtype=torch.bool,
                device=self.device,
            ).view(-1)
            node_mv_payload = list(payload.get("historical_node_multivectors", []))
        except Exception as exc:
            print(f"  Continuous bank cache parse failed: {type(exc).__name__}: {exc}")
            return False

        if memory_embeddings.ndim != 2 or memory_embeddings.size(0) != len(signatures):
            print("  Continuous bank cache mismatch: memory embedding shape changed; rebuilding cache.")
            return False
        if projected_mask.numel() != len(signatures):
            print("  Continuous bank cache mismatch: projected-mask length changed; rebuilding cache.")
            return False
        if len(node_mv_payload) != len(signatures):
            print("  Continuous bank cache mismatch: multivector list length changed; rebuilding cache.")
            return False

        loaded_multivectors: List[torch.Tensor | None] = []
        for item in node_mv_payload:
            if item is None:
                loaded_multivectors.append(None)
                continue
            tensor = torch.as_tensor(item, dtype=torch.float32, device=self.device)
            loaded_multivectors.append(tensor)

        # Fusion requires node-level multivectors for projected entries.
        # Older or partial caches may contain valid projected embeddings but
        # missing multivectors for a small minority of entries. Downgrade those
        # entries in place instead of forcing a full bank rebuild.
        invalid_projected = [
            idx
            for idx, (is_projected, mv) in enumerate(zip(projected_mask.tolist(), loaded_multivectors))
            if is_projected and mv is None
        ]
        if invalid_projected:
            projected_mask = projected_mask.clone()
            projected_mask[torch.tensor(invalid_projected, dtype=torch.long, device=self.device)] = False
            print(
                "  Continuous bank cache warning: "
                f"{len(invalid_projected)} projected entries are missing node "
                "multivectors; downgrading them to fingerprint-only retrieval."
            )

        self.memory_embeddings = memory_embeddings
        self.memory_projected_mask = projected_mask
        self.historical_node_multivectors = loaded_multivectors
        print(
            f"  Loaded continuous bank cache → {cache_path} "
            f"({int(projected_mask.sum().item())}/{len(signatures)} projected)"
        )
        return True

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
        payload: dict[str, Any] = {
            "version": 1,
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
                "  Saved continuous bank cache with "
                f"{invalid_projected} downgraded projected entries (missing node multivectors)."
            )
        print(f"  Saved continuous bank cache → {cache_path}")

    def _ball_radius(self) -> float:
        return self.poincare_radius / math.sqrt(self.curvature)

    def _prepare_tangent(self, x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        # Encode the binary fingerprint in tangent space near the origin instead
        # of blasting it directly onto the Poincare boundary. log1p(||x||) keeps
        # high-bit-count scaffolds from collapsing to nearly identical boundary
        # points, while preserving directional information.
        norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        unit = x / norm
        tangent_norm = torch.log1p(norm) * self.tangent_scale
        tangent_norm = tangent_norm.clamp_max(self.max_tangent_norm)
        return unit * tangent_norm

    def _project_inside_ball(self, x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        radius = x.new_tensor(self._ball_radius())
        norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        max_norm = radius * (1.0 - eps)
        return torch.where(norm > max_norm, x / norm * max_norm, x)

    def _expmap0(self, u: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        c = u.new_tensor(self.curvature)
        sqrt_c = torch.sqrt(c).clamp_min(eps)
        u_norm = u.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.tanh(sqrt_c * u_norm) / (sqrt_c * u_norm)
        return self._project_inside_ball(u * scale, eps=eps)

    def _encode_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        return self._expmap0(self._prepare_tangent(x))

    def _poincare_distance(self, u: torch.Tensor, v: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        c = u.new_tensor(self.curvature)
        sqrt_c = torch.sqrt(c)
        sqdist = (u - v).pow(2).sum(dim=-1)
        squnorm = u.pow(2).sum(dim=-1)
        sqvnorm = v.pow(2).sum(dim=-1)

        denom_u = (1.0 - c * squnorm).clamp_min(eps)
        denom_v = (1.0 - c * sqvnorm).clamp_min(eps)
        arg = 1.0 + 2.0 * c * sqdist / (denom_u * denom_v)
        arg = arg.clamp_min(1.0 + eps)
        return torch.acosh(arg) / sqrt_c.clamp_min(eps)

    def populate_from_mols(self, mols: List, *, continuous_encoder=None, continuous_cache_path: str | Path | None = None) -> None:
        print(f"Populating Hyperbolic Memory Bank from {len(mols)} input molecules...")
        self._reset()
        fps: List[torch.Tensor] = []
        skipped = 0

        for mol in mols:
            som_idx = _extract_som_idx(mol)
            if som_idx is None:
                skipped += 1
                continue
            fp = _morgan_fp_tensor(mol, radius=self.fp_radius, n_bits=self.fp_bits)
            fps.append(fp)
            self.historical_mols.append(mol)
            self.historical_soms.append(som_idx)
            canonical = _canonical_smiles(mol)
            self.historical_smiles.append(canonical)
            scaffold = scaffold_key(mol)
            self.historical_scaffolds.append(scaffold)
            self.historical_shortlist_keys.append(scaffold or canonical or f"idx:{len(self.historical_mols)-1}")
            self.historical_heavy_atoms.append(int(max(mol.GetNumHeavyAtoms(), 1)))
            try:
                self.historical_ring_counts.append(int(mol.GetRingInfo().NumRings()))
            except Exception:
                self.historical_ring_counts.append(0)

        if not fps:
            raise RuntimeError("No labelled molecules found — hyperbolic memory bank is empty.")

        raw = torch.stack(fps).to(self.device)
        key_counts = Counter(self.historical_shortlist_keys)
        total_keys = float(max(len(self.historical_shortlist_keys), 1))
        self.historical_key_frequencies = [
            float(key_counts.get(key, 0)) / total_keys
            for key in self.historical_shortlist_keys
        ]
        self.memory_fingerprints = raw
        self.memory_bit_counts = raw.sum(dim=-1)
        cache_path = Path(continuous_cache_path) if continuous_cache_path is not None else None
        if continuous_encoder is not None and cache_path is not None and cache_path.exists():
            if self._load_continuous_bank_cache(cache_path):
                return

        projected_embeddings: List[torch.Tensor | None] = []
        self.historical_node_multivectors = []
        projection_failures = 0
        projection_failure_examples: List[str] = []
        for mol in self.historical_mols:
            projected: torch.Tensor | None = None
            node_multivectors: torch.Tensor | None = None
            if continuous_encoder is not None:
                try:
                    encoded = continuous_encoder(mol)
                    if isinstance(encoded, Mapping):
                        projected = encoded.get("graph_embedding")
                        node_multivectors = encoded.get("node_multivectors")
                    else:
                        projected = encoded
                    if projected is not None:
                        projected = torch.as_tensor(projected, dtype=torch.float32, device=self.device).view(-1)
                        if projected.numel() == 0 or not bool(torch.isfinite(projected).all().item()):
                            projected = None
                    if node_multivectors is not None:
                        node_multivectors = torch.as_tensor(
                            node_multivectors,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        if node_multivectors.numel() == 0 or not bool(torch.isfinite(node_multivectors).all().item()):
                            node_multivectors = None
                except Exception as exc:
                    projection_failures += 1
                    if len(projection_failure_examples) < 3:
                        projection_failure_examples.append(
                            f"{type(exc).__name__}: {exc}"
                        )
                    projected = None
                    node_multivectors = None
            projected_embeddings.append(projected)
            self.historical_node_multivectors.append(node_multivectors)

        projected_mask = torch.zeros(len(projected_embeddings), dtype=torch.bool, device=self.device)
        if any(embed is not None for embed in projected_embeddings):
            # Determine target dimension from the continuous (HGNN) embeddings.
            projected_dim = next(
                (int(embed.numel()) for embed in projected_embeddings if embed is not None),
                None,
            )
            # Build fallback embeddings in the SAME projected_dim space.
            # We derive them from the ECFP4 fingerprint properly: encode hyperbolic
            # (2048D), then project to projected_dim via a random-but-fixed linear map
            # stored at populate time so retrieval distances remain consistent.
            # This preserves the full fingerprint structure rather than truncating.
            fallback_embeddings_full = self._encode_hyperbolic(raw)  # [N, 2048]
            fp_dim = int(fallback_embeddings_full.size(-1))
            if fp_dim != projected_dim:
                # Use PCA-like projection: multiply by a fixed [fp_dim, projected_dim]
                # matrix derived from the first principal components of the fp space.
                # Simple deterministic approach: stride-average the fp dimensions.
                if projected_dim < fp_dim:
                    # Average-pool fp_dim → projected_dim
                    factor = fp_dim / projected_dim
                    indices = torch.arange(projected_dim, device=self.device)
                    lo = (indices * factor).long().clamp(0, fp_dim - 1)
                    hi = ((indices + 1) * factor).long().clamp(0, fp_dim)
                    fallback_embeddings = torch.stack(
                        [fallback_embeddings_full[:, lo[i]:hi[i]].mean(dim=-1) for i in range(projected_dim)],
                        dim=-1,
                    )
                else:
                    # Pad with zeros if projected_dim > fp_dim (unusual)
                    fallback_embeddings = F.pad(fallback_embeddings_full, (0, projected_dim - fp_dim))
                fallback_embeddings = self._project_inside_ball(fallback_embeddings)
            else:
                fallback_embeddings = fallback_embeddings_full

            final_embeddings: List[torch.Tensor] = []
            for idx, embed in enumerate(projected_embeddings):
                if embed is None:
                    final_embeddings.append(fallback_embeddings[idx])
                else:
                    current = embed.view(-1)
                    if current.numel() < projected_dim:
                        current = F.pad(current, (0, projected_dim - current.numel()))
                    elif current.numel() > projected_dim:
                        current = current[:projected_dim]
                    final_embeddings.append(self._project_inside_ball(current))
                    projected_mask[idx] = True
            self.memory_embeddings = torch.stack(final_embeddings, dim=0)
        else:
            self.memory_embeddings = self._encode_hyperbolic(raw)
        self.memory_projected_mask = projected_mask
        projected_count = int(projected_mask.sum().item())
        print(
            f"Hyperbolic Memory Bank Active: {len(self.historical_mols)} molecules "
            f"({skipped} skipped — no SoM label, {projected_count} projected in continuous space)."
        )
        if continuous_encoder is not None and projection_failures > 0:
            print(
                f"  Continuous projection failures: {projection_failures}"
            )
            for sample in projection_failure_examples:
                print(f"    sample failure: {sample}")
        if continuous_encoder is not None and projected_count == 0 and len(self.historical_mols) > 0:
            print(
                "  WARNING: continuous bank mode requested but zero bank molecules "
                "were projected. Retrieval will fall back to fingerprint/hyperbolic mode."
            )
        if continuous_encoder is not None and cache_path is not None and projected_count > 0:
            self._save_continuous_bank_cache(cache_path)

    def _tanimoto_similarity(self, q_fp: torch.Tensor) -> torch.Tensor:
        if self.memory_fingerprints is None or self.memory_bit_counts is None:
            raise RuntimeError("Call populate_from_mols() before retrieve_and_transport().")
        query = q_fp.to(self.device)
        intersection = torch.matmul(self.memory_fingerprints, query)
        q_count = query.sum()
        union = (self.memory_bit_counts + q_count - intersection).clamp_min(1.0e-6)
        return intersection / union

    @staticmethod
    def _morphism_vector(morph_target: torch.Tensor, morph_mask: torch.Tensor) -> torch.Tensor:
        weighted = (morph_target * morph_mask).sum(dim=0)
        if weighted.ndim != 1:
            weighted = weighted.view(-1)
        if weighted.sum().item() <= 0.0:
            return torch.zeros_like(weighted)
        return weighted / weighted.sum().clamp_min(1.0e-8)

    def _candidate_mechanism_overlap(
        self,
        query_morphism_prior: torch.Tensor | None,
        candidate_idx: int,
    ) -> float:
        if query_morphism_prior is None:
            return 0.0
        cand_target, cand_mask, cand_has, _cand_conf = build_morphism_supervision(
            self.historical_mols[candidate_idx],
            self.historical_soms[candidate_idx],
            device=self.device,
        )
        if not cand_has:
            return 0.0
        cand_vec = self._morphism_vector(cand_target, cand_mask)
        query_vec = torch.as_tensor(query_morphism_prior, dtype=torch.float32, device=self.device).view(-1)
        if query_vec.numel() != cand_vec.numel():
            return 0.0
        if query_vec.sum().item() <= 0.0:
            return 0.0
        query_vec = query_vec / query_vec.sum().clamp_min(1.0e-8)
        return float((query_vec * cand_vec).sum().item())

    def _candidate_structural_bonus(
        self,
        query_mol,
        candidate_idx: int,
    ) -> float:
        query_heavy = int(max(query_mol.GetNumHeavyAtoms(), 1))
        candidate_heavy = (
            self.historical_heavy_atoms[candidate_idx]
            if candidate_idx < len(self.historical_heavy_atoms)
            else int(max(self.historical_mols[candidate_idx].GetNumHeavyAtoms(), 1))
        )
        try:
            query_rings = int(query_mol.GetRingInfo().NumRings())
        except Exception:
            query_rings = 0
        candidate_rings = (
            self.historical_ring_counts[candidate_idx]
            if candidate_idx < len(self.historical_ring_counts)
            else 0
        )
        scaffold = self.historical_scaffolds[candidate_idx] if candidate_idx < len(self.historical_scaffolds) else ""

        size_ratio = min(query_heavy, candidate_heavy) / max(query_heavy, candidate_heavy)
        bonus = 1.15 * size_ratio - 0.25

        if candidate_heavy <= 4:
            bonus -= 1.25
        elif candidate_heavy <= 6:
            bonus -= 0.55

        if query_heavy >= 10 and candidate_heavy <= 6:
            bonus -= 1.10
        elif query_heavy >= 16 and candidate_heavy <= 8:
            bonus -= 0.75

        if abs(query_heavy - candidate_heavy) <= max(2, int(0.15 * query_heavy)):
            bonus += 0.25

        if scaffold:
            bonus += 0.20
        else:
            bonus -= 0.55 if candidate_heavy <= 8 else 0.20

        if query_rings > 0 and candidate_rings == 0 and query_heavy >= 8:
            bonus -= 0.45
        elif query_rings == 0 and candidate_rings > 0 and candidate_heavy >= 8:
            bonus -= 0.15

        return float(0.45 * bonus)

    def _candidate_global_hub_penalty(self, candidate_idx: int) -> float:
        frequency = (
            self.historical_key_frequencies[candidate_idx]
            if candidate_idx < len(self.historical_key_frequencies)
            else 0.0
        )
        # Strongly penalize globally common scaffold/structure keys before
        # shortlist formation so broad aromatic hubs do not dominate the pool.
        return float(0.85 * min(frequency * 20.0, 1.0))

    def _select_diverse_candidates(
        self,
        query_mol,
        candidate_indices: List[int],
        base_scores: torch.Tensor,
        *,
        top_k: int,
        query_morphism_prior: torch.Tensor | None = None,
    ) -> tuple[List[int], torch.Tensor, float, dict[str, float]]:
        if not candidate_indices:
            return [], torch.empty(0, dtype=torch.float32, device=self.device), 0.0, {}
        scores = torch.as_tensor(base_scores, dtype=torch.float32, device=self.device).view(-1).clone()
        candidate_keys = [
            self.historical_shortlist_keys[candidate_idx]
            if candidate_idx < len(self.historical_shortlist_keys)
            else (self.historical_scaffolds[candidate_idx] if candidate_idx < len(self.historical_scaffolds) else "") or (self.historical_smiles[candidate_idx] if candidate_idx < len(self.historical_smiles) else None) or f"idx:{candidate_idx}"
            for candidate_idx in candidate_indices
        ]
        key_counts = Counter(candidate_keys)
        mechanism_overlaps: List[float] = []
        structural_bonuses: List[float] = []
        hub_penalties: List[float] = []
        for local_idx, candidate_idx in enumerate(candidate_indices):
            overlap = self._candidate_mechanism_overlap(query_morphism_prior, candidate_idx)
            mechanism_overlaps.append(overlap)
            structural_bonus = self._candidate_structural_bonus(query_mol, candidate_idx)
            structural_bonuses.append(structural_bonus)
            local_fraction = float(key_counts[candidate_keys[local_idx]]) / float(max(len(candidate_indices), 1))
            hub_penalty = 1.20 * max(local_fraction - (1.0 / max(len(candidate_indices), 1)), 0.0)
            hub_penalties.append(hub_penalty)
            scores[local_idx] = scores[local_idx] + 0.70 * overlap + structural_bonus - hub_penalty
        chosen_locals: List[int] = []
        used_keys: set[str] = set()
        k = min(int(top_k), len(candidate_indices))
        while len(chosen_locals) < k:
            best_local = None
            best_score = None
            for local_idx, candidate_idx in enumerate(candidate_indices):
                if local_idx in chosen_locals:
                    continue
                diversity_key = candidate_keys[local_idx]
                score = float(scores[local_idx].item())
                if diversity_key in used_keys:
                    score -= 0.35
                if best_local is None or score > best_score:
                    best_local = local_idx
                    best_score = score
            if best_local is None:
                break
            chosen_locals.append(best_local)
            diversity_key = candidate_keys[best_local]
            used_keys.add(diversity_key)
        selected_scores = scores.index_select(0, torch.tensor(chosen_locals, dtype=torch.long, device=self.device))
        selected_indices = [candidate_indices[i] for i in chosen_locals]
        diversity_score = float(len(used_keys) / max(len(selected_indices), 1))
        debug = {}
        if chosen_locals:
            best_local = chosen_locals[0]
            debug = {
                "best_mechanism_bonus": float(0.70 * mechanism_overlaps[best_local]),
                "best_structural_bonus": float(structural_bonuses[best_local]),
                "best_hub_penalty": float(hub_penalties[best_local]),
                "shortlist_top_fraction": float(max(key_counts.values()) / max(len(candidate_indices), 1)),
            }
        return selected_indices, selected_scores, diversity_score, debug

    def _shortlist_scores_from_tanimoto(
        self,
        query_mol,
        candidate_indices: torch.Tensor,
        candidate_tanimoto: torch.Tensor,
        *,
        query_morphism_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = candidate_tanimoto.to(dtype=torch.float32).clone()
        for local_idx, candidate_idx_t in enumerate(candidate_indices.tolist()):
            candidate_idx = int(candidate_idx_t)
            overlap = self._candidate_mechanism_overlap(query_morphism_prior, candidate_idx)
            hub_penalty = self._candidate_global_hub_penalty(candidate_idx)
            # Keep shortlist-stage structure weak; shortlist should be broad and
            # mechanism-aware, not dominated by generic scaffold similarity.
            scores[local_idx] = scores[local_idx] + 0.35 * overlap - hub_penalty
        return scores

    def _projected_retrieval_weight(
        self,
        current_epoch: int | None,
        query_embedding: torch.Tensor,
    ) -> float:
        # Epoch ramp only — the previous radius_weight gate was silently killing
        # the hyperbolic path because a freshly initialised HGNNProjection maps
        # near the Poincaré origin (small norm), giving radius_weight ≈ 0 even
        # after burn-in finished.  The encoder needs the path to be active in
        # order to receive gradients, but the radius gate prevented activation
        # until the encoder had already converged — a circular dependency.
        epoch_idx = int(current_epoch or 0)
        if epoch_idx < self.projected_retrieval_burn_in_epochs:
            return 0.0
        ramp_pos = epoch_idx - self.projected_retrieval_burn_in_epochs + 1
        return min(float(ramp_pos) / float(self.projected_retrieval_ramp_epochs), 1.0)

    def _mcs_transport(
        self,
        query_mol,
        retrieved_mol,
        retrieved_som: int,
    ) -> tuple[torch.Tensor, bool, int, torch.Tensor | None, float, str]:
        n_query = query_mol.GetNumAtoms()
        n_retrieved = retrieved_mol.GetNumAtoms()
        analogical_pred = torch.zeros(n_query, dtype=torch.float32, device=self.device)
        mcs_size = 0
        transport_ok = False
        transport_plan: torch.Tensor | None = None
        transported_mass = 0.0
        route_reason = "mcs_unmapped"
        res = rdFMCS.FindMCS(
            [query_mol, retrieved_mol],
            timeout=self.mcs_timeout,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )

        if not res.canceled and res.numAtoms > 0:
            mcs_size = res.numAtoms
            mcs_mol = Chem.MolFromSmarts(res.smartsString)
            match_query = query_mol.GetSubstructMatch(mcs_mol)
            match_retrieved = retrieved_mol.GetSubstructMatch(mcs_mol)

            if match_query and match_retrieved:
                support_cols = list(match_retrieved)
                support_len = max(len(support_cols), 1)
                transport_plan = torch.zeros(
                    (n_query, n_retrieved),
                    dtype=torch.float32,
                    device=self.device,
                )
                fallback_row = torch.zeros(n_retrieved, dtype=torch.float32, device=self.device)
                fallback_row[torch.as_tensor(support_cols, dtype=torch.long, device=self.device)] = 1.0 / float(support_len)

                matched_query_set = set(int(idx) for idx in match_query)
                for q_idx, ret_idx in zip(match_query, match_retrieved):
                    transport_plan[int(q_idx), int(ret_idx)] = 1.0
                for q_idx in range(n_query):
                    if q_idx not in matched_query_set:
                        transport_plan[q_idx] = fallback_row

                mapped_query_idx: int | None = None
                if retrieved_som in match_retrieved:
                    mapped_query_idx = int(match_query[match_retrieved.index(retrieved_som)])
                    transported_mass = 1.0
                    route_reason = "mcs_exact"
                else:
                    try:
                        distance_matrix = Chem.GetDistanceMatrix(retrieved_mol)
                    except Exception:
                        distance_matrix = None
                    if distance_matrix is not None:
                        best_pos = None
                        best_dist = None
                        for pos, ret_idx in enumerate(match_retrieved):
                            dist = float(distance_matrix[int(retrieved_som), int(ret_idx)])
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_pos = pos
                        if best_pos is not None and best_dist is not None and best_dist <= 2.0:
                            mapped_query_idx = int(match_query[int(best_pos)])
                            transported_mass = float(math.exp(-best_dist))
                            route_reason = "mcs_som_neighborhood"

                if mapped_query_idx is not None:
                    analogical_pred[mapped_query_idx] = 1.0
                    transport_ok = True
                else:
                    transport_plan = None
        return analogical_pred, transport_ok, mcs_size, transport_plan, transported_mass, route_reason

    def _transport_candidate(
        self,
        query_mol,
        retrieved_mol,
        retrieved_som: int,
        *,
        query_multivectors: torch.Tensor | None = None,
        retrieved_multivectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool, int, str, float, torch.Tensor | None, bool, float, float, torch.Tensor | None, str | None, str]:
        if self.transport_backend == "pgw" and self.pgw is not None:
            pgw_result = self.pgw.transport_label(
                query_mol,
                retrieved_mol,
                retrieved_som,
                query_multivectors=query_multivectors,
                retrieved_multivectors=retrieved_multivectors,
            )
            analogical_pred = pgw_result.analogical_pred
            transport_ok = pgw_result.transport_succeeded
            support_size = pgw_result.support_size
            transport_backend = pgw_result.transport_backend
            transported_mass = pgw_result.transported_mass
            transport_distill_loss = pgw_result.distill_loss
            neuralgw_used_exact = pgw_result.neuralgw_used_exact
            neuralgw_confidence = pgw_result.neuralgw_confidence
            neuralgw_distill_loss = pgw_result.neuralgw_distill_loss
            transport_plan = pgw_result.coupling_matrix
            transport_error_message = pgw_result.transport_error_message
            neuralgw_route_reason = pgw_result.neuralgw_route_reason
            if (not transport_ok) and self.fallback_to_mcs:
                (
                    analogical_pred,
                    transport_ok,
                    support_size,
                    transport_plan,
                    transported_mass,
                    mcs_route_reason,
                ) = self._mcs_transport(query_mol, retrieved_mol, retrieved_som)
                transport_backend = (
                    "mcs_fallback_som_neighborhood"
                    if mcs_route_reason == "mcs_som_neighborhood"
                    else "mcs_fallback"
                )
                transport_distill_loss = None
                neuralgw_used_exact = False
                neuralgw_confidence = 0.0
                neuralgw_distill_loss = 0.0
                neuralgw_route_reason = (
                    "mcs_fallback_som_neighborhood"
                    if mcs_route_reason == "mcs_som_neighborhood"
                    else "mcs_fallback"
                )
            return (
                analogical_pred,
                transport_ok,
                support_size,
                transport_backend,
                transported_mass,
                transport_distill_loss,
                neuralgw_used_exact,
                neuralgw_confidence,
                neuralgw_distill_loss,
                transport_plan,
                transport_error_message,
                neuralgw_route_reason,
            )

        (
            analogical_pred,
            transport_ok,
            support_size,
            transport_plan,
            transported_mass,
            mcs_route_reason,
        ) = self._mcs_transport(query_mol, retrieved_mol, retrieved_som)
        return (
            analogical_pred,
            transport_ok,
            support_size,
            "mcs_som_neighborhood" if mcs_route_reason == "mcs_som_neighborhood" else "mcs",
            transported_mass,
            None,
            False,
            0.0,
            0.0,
            transport_plan,
            None,
            "mcs_only_som_neighborhood" if mcs_route_reason == "mcs_som_neighborhood" else "mcs_only",
        )

    def retrieve_and_transport(
        self,
        query_mol,
        query_smiles: str | None = None,
        mechanism_encoder=None,
        query_embedding: torch.Tensor | None = None,
        query_multivectors: torch.Tensor | None = None,
        query_morphism_prior: torch.Tensor | None = None,
        current_epoch: int | None = None,
    ) -> MemoryRetrievalResult:
        """
        Args:
            query_mol:           RDKit Mol to retrieve for.
            query_smiles:        Optional canonical SMILES for exact-query masking.
            mechanism_encoder:   Optional MechanismEncoder instance.  When provided,
                                 it re-ranks the Tanimoto shortlist using mechanism-
                                 aware cosine similarity and attaches query_embed /
                                 retrieved_embed_detached to the result for
                                 encoder_supervision_loss in the trainer.
        """
        if self.memory_embeddings is None:
            raise RuntimeError("Call populate_from_mols() before retrieve_and_transport().")

        query_key = query_smiles or _canonical_smiles(query_mol)
        same_query = None
        if query_key is not None and self.historical_smiles:
            same_query = torch.tensor(
                [s == query_key for s in self.historical_smiles],
                dtype=torch.bool,
                device=self.device,
            )

        q_fp = _morgan_fp_tensor(query_mol, radius=self.fp_radius, n_bits=self.fp_bits)
        tanimoto_scores = self._tanimoto_similarity(q_fp)

        projected_ready = (
            query_embedding is not None
            and self.memory_projected_mask is not None
            and bool(self.memory_projected_mask.any().item())
        )
        retrieval_embedding_space = "euclidean"

        query_embed: torch.Tensor | None = None
        retrieved_embed_detached: torch.Tensor | None = None
        mix_candidate_indices: List[int] = []
        mix_weights: List[float] = []
        retrieval_mix_count = 1
        retrieval_mix_entropy = 0.0
        retrieval_mechanism_overlap = 0.0
        retrieval_diversity_score = 0.0
        retrieval_candidate_count = 0
        retrieval_projected_weight = 0.0
        retrieval_best_tanimoto = 0.0
        retrieval_best_projected_similarity = 0.0
        retrieval_best_mechanism_bonus = 0.0
        retrieval_best_structural_bonus = 0.0
        retrieval_best_hub_penalty = 0.0
        retrieval_shortlist_top_fraction = 0.0

        if projected_ready:
            candidate_mask = self.memory_projected_mask.clone()
            if same_query is not None:
                candidate_mask = candidate_mask & (~same_query)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
            if candidate_indices.numel() > 0:
                retrieval_candidate_count = int(candidate_indices.numel())
                projected_weight = self._projected_retrieval_weight(current_epoch, query_embedding)
                retrieval_projected_weight = float(projected_weight)
                retrieval_embedding_space = "hybrid" if projected_weight < 0.999 else "hyperbolic"
                target_dim = int(self.memory_embeddings.size(-1))
                prepared_query = torch.as_tensor(query_embedding, dtype=torch.float32, device=self.device).view(-1)
                if prepared_query.numel() < target_dim:
                    prepared_query = F.pad(prepared_query, (0, target_dim - prepared_query.numel()))
                elif prepared_query.numel() > target_dim:
                    prepared_query = prepared_query[:target_dim]
                q_embed_h = self._project_inside_ball(
                    prepared_query.view(1, -1)
                )
                candidate_tanimoto = tanimoto_scores.index_select(0, candidate_indices)
                shortlist_k = min(max(self.tanimoto_shortlist_k * 3, 48), int(candidate_indices.numel()))
                shortlist_seed_scores = self._shortlist_scores_from_tanimoto(
                    query_mol,
                    candidate_indices,
                    candidate_tanimoto,
                    query_morphism_prior=query_morphism_prior,
                )
                top_shortlist_scores, top_local = torch.topk(shortlist_seed_scores, shortlist_k, largest=True)
                shortlist_candidate_indices = [
                    int(candidate_indices[int(top_local[i].item())].item())
                    for i in range(shortlist_k)
                ]
                shortlist_indices = candidate_indices.index_select(0, top_local)
                top_tanimoto = candidate_tanimoto.index_select(0, top_local)
                shortlist_embeddings = self.memory_embeddings.index_select(0, shortlist_indices)
                distances = self._poincare_distance(q_embed_h, shortlist_embeddings).view(-1)
                projected_similarity = torch.exp(-distances / 5.0).to(dtype=torch.float32)
                blended_similarity = (
                    projected_weight * projected_similarity
                    + (1.0 - projected_weight) * top_tanimoto.to(dtype=torch.float32)
                )
                shortlist_scores = blended_similarity / self.retrieval_mix_temperature
                mix_k = min(max(self.retrieval_mix_top_k, 3), shortlist_k)
                mix_candidate_indices, mix_logits, retrieval_diversity_score, retrieval_debug = self._select_diverse_candidates(
                    query_mol,
                    shortlist_candidate_indices,
                    shortlist_scores,
                    top_k=mix_k,
                    query_morphism_prior=query_morphism_prior,
                )
                if not mix_candidate_indices:
                    mix_candidate_indices = shortlist_candidate_indices[:mix_k]
                    mix_logits = shortlist_scores[:mix_k]
                mix_prob = torch.softmax(mix_logits, dim=0)
                mix_weights = [float(v.item()) for v in mix_prob]
                retrieval_mix_count = len(mix_candidate_indices)
                retrieval_mix_entropy = float(
                    (-(mix_prob * torch.log(mix_prob.clamp_min(1.0e-9))).sum()).item()
                )
                best_idx = int(mix_candidate_indices[0])
                best_rank_local = shortlist_candidate_indices.index(best_idx)
                retrieval_best_tanimoto = float(top_tanimoto[best_rank_local].item())
                retrieval_best_projected_similarity = float(projected_similarity[best_rank_local].item())
                confidence = float(
                    projected_weight * projected_similarity[best_rank_local].item()
                    + (1.0 - projected_weight) * top_tanimoto[best_rank_local].item()
                )
                retrieval_mechanism_overlap = self._candidate_mechanism_overlap(query_morphism_prior, best_idx)
                retrieval_best_mechanism_bonus = float(retrieval_debug.get("best_mechanism_bonus", 0.0))
                retrieval_best_structural_bonus = float(retrieval_debug.get("best_structural_bonus", 0.0))
                retrieval_best_hub_penalty = float(retrieval_debug.get("best_hub_penalty", 0.0))
                retrieval_shortlist_top_fraction = float(retrieval_debug.get("shortlist_top_fraction", 0.0))
                query_embed = q_embed_h.squeeze(0)
                retrieved_embed_detached = self.memory_embeddings[best_idx].detach()
            else:
                projected_ready = False

        if not projected_ready:
            retrieval_embedding_space = "euclidean"
            candidate_mask = torch.ones_like(tanimoto_scores, dtype=torch.bool)
            if same_query is not None and bool(same_query.any().item()):
                candidate_mask = candidate_mask & (~same_query)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
            if candidate_indices.numel() == 0:
                candidate_indices = torch.arange(len(self.historical_mols), device=self.device, dtype=torch.long)
            candidate_scores = tanimoto_scores.index_select(0, candidate_indices)
            shortlist_k = min(max(self.tanimoto_shortlist_k * 3, 48), int(candidate_indices.numel()))
            retrieval_candidate_count = int(candidate_indices.numel())
            shortlist_seed_scores = self._shortlist_scores_from_tanimoto(
                query_mol,
                candidate_indices,
                candidate_scores,
                query_morphism_prior=query_morphism_prior,
            )
            _top_shortlist_scores, top_local = torch.topk(shortlist_seed_scores, shortlist_k, largest=True)
            top_tanimoto = candidate_scores.index_select(0, top_local)
            shortlist_indices = candidate_indices.index_select(0, top_local)
            best_tanimoto = float(top_tanimoto[0].item()) if shortlist_k > 0 else -1.0
            best_shortlist_idx = int(shortlist_indices[0].item()) if shortlist_k > 0 else 0

            if best_tanimoto < self.tanimoto_prefilter_threshold:
                n_query = query_mol.GetNumAtoms()
                _pref_mol = self.historical_mols[best_shortlist_idx]
                _pref_som = self.historical_soms[best_shortlist_idx]
                _pref_morph_target, _pref_morph_mask, _pref_has_morph, _pref_morph_conf = build_morphism_supervision(
                    _pref_mol,
                    _pref_som,
                    device=self.device,
                )
                return MemoryRetrievalResult(
                    analogical_pred=torch.zeros(n_query, dtype=torch.float32, device=self.device),
                    confidence=0.0,
                    retrieved_mol=_pref_mol,
                    retrieved_som_idx=_pref_som,
                    transport_succeeded=False,
                    mcs_size=0,
                    embedding_space=retrieval_embedding_space,
                    transport_backend="prefilter_reject",
                    transport_support_size=0,
                    transported_mass=0.0,
                    retrieved_same_query=(
                        query_key is not None
                        and self.historical_smiles[best_shortlist_idx] == query_key
                    ),
                    transport_plan=None,
                    retrieved_node_multivectors=None,
                    transport_error_message=None,
                    retrieved_morphism_target=_pref_morph_target,
                    retrieved_morphism_loss_mask=_pref_morph_mask,
                    retrieved_has_morphism_label=_pref_has_morph,
                    retrieved_label_confidence=_pref_morph_conf,
                    retrieved_scaffold=self.historical_scaffolds[best_shortlist_idx] if best_shortlist_idx < len(self.historical_scaffolds) else "",
                    retrieval_candidate_count=shortlist_k,
                    retrieval_mechanism_overlap=self._candidate_mechanism_overlap(query_morphism_prior, best_shortlist_idx),
                    retrieval_diversity_score=0.0,
                    retrieval_projected_weight=0.0,
                    retrieval_best_tanimoto=best_tanimoto,
                    retrieval_best_projected_similarity=0.0,
                    retrieval_best_mechanism_bonus=0.0,
                    retrieval_best_structural_bonus=0.0,
                    retrieval_best_hub_penalty=0.0,
                    retrieval_shortlist_top_fraction=0.0,
                    neuralgw_route_reason="prefilter_reject",
                )

            # ── mechanism-aware re-ranking (optional) ─────────────────────
            # If a MechanismEncoder is provided, we encode the query and all
            # shortlist candidates, then pick the closest in mechanism space
            # rather than purely in hyperbolic ECFP4 space.  This prevents
            # retrieval of a scaffold-similar but mechanistically-different
            # molecule (e.g. methyl-blocked analogue).
            target_dim = int(self.memory_embeddings.size(-1))

            if mechanism_encoder is not None:
                try:
                    q_fp_dev = q_fp.to(self.device)
                    query_embed = mechanism_encoder(q_fp_dev.unsqueeze(0)).squeeze(0)  # [embed_dim]

                    # Encode all shortlist fingerprints in one pass.
                    shortlist_fps = self.memory_fingerprints.index_select(0, shortlist_indices)  # [k, fp_bits]
                    shortlist_embeds = mechanism_encoder(shortlist_fps)  # [k, embed_dim]

                    mech_scores = (query_embed.detach().unsqueeze(0) * shortlist_embeds.detach()).sum(dim=-1)  # [k]
                    shortlist_candidate_indices = [int(v) for v in shortlist_indices.tolist()]
                    mix_k = min(max(self.retrieval_mix_top_k, 3), shortlist_k)
                    mix_candidate_indices, mix_logits, retrieval_diversity_score, retrieval_debug = self._select_diverse_candidates(
                        query_mol,
                        shortlist_candidate_indices,
                        mech_scores.to(dtype=torch.float32) / self.retrieval_mix_temperature,
                        top_k=mix_k,
                        query_morphism_prior=query_morphism_prior,
                    )
                    if not mix_candidate_indices:
                        mix_candidate_indices = shortlist_candidate_indices[:mix_k]
                        mix_logits = mech_scores[:mix_k].to(dtype=torch.float32) / self.retrieval_mix_temperature
                    mix_prob = torch.softmax(mix_logits, dim=0)
                    mix_weights = [float(v.item()) for v in mix_prob]
                    retrieval_mix_count = len(mix_candidate_indices)
                    retrieval_mix_entropy = float(
                        (-(mix_prob * torch.log(mix_prob.clamp_min(1.0e-9))).sum()).item()
                    )
                    best_idx = int(mix_candidate_indices[0])
                    best_mech_rank = shortlist_candidate_indices.index(best_idx)
                    retrieved_embed_detached = shortlist_embeds[best_mech_rank].detach()
                    retrieval_mechanism_overlap = self._candidate_mechanism_overlap(query_morphism_prior, best_idx)
                    retrieval_best_tanimoto = float(top_tanimoto[best_mech_rank].item())
                    retrieval_best_projected_similarity = 0.0
                    retrieval_best_mechanism_bonus = float(retrieval_debug.get("best_mechanism_bonus", 0.0))
                    retrieval_best_structural_bonus = float(retrieval_debug.get("best_structural_bonus", 0.0))
                    retrieval_best_hub_penalty = float(retrieval_debug.get("best_hub_penalty", 0.0))
                    retrieval_shortlist_top_fraction = float(retrieval_debug.get("shortlist_top_fraction", 0.0))

                    # Confidence from cosine similarity in the MechanismEncoder
                    # space — the same space used for retrieval ranking.  The
                    # previous code switched to Poincaré distance of fingerprint
                    # embeddings (a different encoder), making the confidence
                    # number incoherent with the actual retrieval criterion.
                    # Both query_embed and retrieved_embed_detached are already
                    # L2-normalised by MechanismEncoder, so dot product = cosine.
                    confidence = float(
                        (query_embed * retrieved_embed_detached).sum().clamp(0.0, 1.0).item()
                    )

                except Exception:
                    # Encoder failed (e.g. fp_bits mismatch during warm-up) — fall back
                    query_embed = None
                    retrieved_embed_detached = None
                    mechanism_encoder = None  # disable for the rest of this call

            if mechanism_encoder is None:
                # Standard hyperbolic retrieval path.
                q_embed_h = self._encode_hyperbolic(q_fp.unsqueeze(0).to(self.device))
                if q_embed_h.size(-1) < target_dim:
                    q_embed_h = F.pad(q_embed_h, (0, target_dim - q_embed_h.size(-1)))
                elif q_embed_h.size(-1) > target_dim:
                    q_embed_h = q_embed_h[..., :target_dim]
                candidate_embeddings = self.memory_embeddings.index_select(0, shortlist_indices)
                distances = self._poincare_distance(q_embed_h, candidate_embeddings).squeeze(0)
                shortlist_candidate_indices = [int(v) for v in shortlist_indices.tolist()]
                shortlist_scores = -distances.to(dtype=torch.float32) / self.retrieval_mix_temperature
                mix_k = min(max(self.retrieval_mix_top_k, 3), shortlist_k)
                mix_candidate_indices, mix_logits, retrieval_diversity_score, retrieval_debug = self._select_diverse_candidates(
                    query_mol,
                    shortlist_candidate_indices,
                    shortlist_scores,
                    top_k=mix_k,
                    query_morphism_prior=query_morphism_prior,
                )
                if not mix_candidate_indices:
                    mix_candidate_indices = shortlist_candidate_indices[:mix_k]
                    mix_logits = shortlist_scores[:mix_k]
                mix_prob = torch.softmax(mix_logits, dim=0)
                mix_weights = [float(v.item()) for v in mix_prob]
                retrieval_mix_count = len(mix_candidate_indices)
                retrieval_mix_entropy = float(
                    (-(mix_prob * torch.log(mix_prob.clamp_min(1.0e-9))).sum()).item()
                )
                best_idx = int(mix_candidate_indices[0])
                tau = 10.0
                best_rank_local = shortlist_candidate_indices.index(best_idx)
                confidence = math.exp(-float(distances[best_rank_local].item()) / tau)
                retrieval_mechanism_overlap = self._candidate_mechanism_overlap(query_morphism_prior, best_idx)
                retrieval_best_tanimoto = float(top_tanimoto[best_rank_local].item())
                retrieval_best_projected_similarity = float(math.exp(-float(distances[best_rank_local].item()) / 5.0))
                retrieval_best_mechanism_bonus = float(retrieval_debug.get("best_mechanism_bonus", 0.0))
                retrieval_best_structural_bonus = float(retrieval_debug.get("best_structural_bonus", 0.0))
                retrieval_best_hub_penalty = float(retrieval_debug.get("best_hub_penalty", 0.0))
                retrieval_shortlist_top_fraction = float(retrieval_debug.get("shortlist_top_fraction", 0.0))

        retrieved_mol = self.historical_mols[best_idx]
        retrieved_som = self.historical_soms[best_idx]
        retrieved_morph_target, retrieved_morph_mask, retrieved_has_morph, retrieved_morph_conf = build_morphism_supervision(
            retrieved_mol,
            retrieved_som,
            device=self.device,
        )
        retrieved_multivectors = None
        if 0 <= best_idx < len(self.historical_node_multivectors):
            retrieved_multivectors = self.historical_node_multivectors[best_idx]
        (
            analogical_pred,
            transport_ok,
            support_size,
            transport_backend,
            transported_mass,
            transport_distill_loss,
            neuralgw_used_exact,
            neuralgw_confidence,
            neuralgw_distill_loss,
            transport_plan,
            transport_error_message,
            neuralgw_route_reason,
        ) = self._transport_candidate(
            query_mol,
            retrieved_mol,
            retrieved_som,
            query_multivectors=query_multivectors,
            retrieved_multivectors=retrieved_multivectors,
        )
        anchor_idx = best_idx

        # Multi-memory mixture for the continuous path: blend a small number of
        # transported analogical priors while still using the best analogue as
        # the anchor for fusion / detailed diagnostics.
        if mix_candidate_indices and len(mix_candidate_indices) > 1:
            mixture = torch.zeros_like(analogical_pred)
            total_weight = 0.0
            current_anchor_has_mv = retrieved_multivectors is not None
            best_anchor_score = float(mix_weights[0]) * (transported_mass + 0.05 * max(support_size, 1))
            if current_anchor_has_mv:
                best_anchor_score += 0.05
            for candidate_idx, mix_weight in zip(mix_candidate_indices, mix_weights):
                cand_mol = self.historical_mols[candidate_idx]
                cand_som = self.historical_soms[candidate_idx]
                cand_mv = None
                if 0 <= candidate_idx < len(self.historical_node_multivectors):
                    cand_mv = self.historical_node_multivectors[candidate_idx]
                cand_pred, cand_ok, _cand_support, _cand_backend, _cand_mass, _cand_distill, _cand_exact, _cand_conf, _cand_distill_value, _cand_plan, _cand_err, _cand_route_reason = self._transport_candidate(
                    query_mol,
                    cand_mol,
                    cand_som,
                    query_multivectors=query_multivectors,
                    retrieved_multivectors=cand_mv,
                )
                if cand_ok:
                    mixture = mixture + float(mix_weight) * cand_pred
                    total_weight += float(mix_weight)
                    cand_score = float(mix_weight) * (float(_cand_mass) + 0.05 * max(int(_cand_support), 1))
                    cand_has_mv = cand_mv is not None
                    if cand_has_mv:
                        cand_score += 0.05
                    should_switch_anchor = (
                        _cand_plan is not None
                        and cand_score > best_anchor_score
                        and (cand_has_mv or not current_anchor_has_mv)
                    )
                    if should_switch_anchor:
                        best_anchor_score = cand_score
                        anchor_idx = candidate_idx
                        current_anchor_has_mv = cand_has_mv
                        analogical_pred = cand_pred
                        transport_ok = cand_ok
                        support_size = int(_cand_support)
                        transport_backend = _cand_backend
                        transported_mass = float(_cand_mass)
                        transport_distill_loss = _cand_distill
                        neuralgw_used_exact = _cand_exact
                        neuralgw_confidence = _cand_conf
                        neuralgw_distill_loss = _cand_distill_value
                        transport_plan = _cand_plan
                        transport_error_message = _cand_err
                        neuralgw_route_reason = _cand_route_reason
                        retrieved_mol = cand_mol
                        retrieved_som = cand_som
                        retrieved_multivectors = cand_mv
                        retrieved_morph_target, retrieved_morph_mask, retrieved_has_morph, retrieved_morph_conf = build_morphism_supervision(
                            retrieved_mol,
                            retrieved_som,
                            device=self.device,
                        )
                        retrieval_mechanism_overlap = self._candidate_mechanism_overlap(query_morphism_prior, anchor_idx)
            if total_weight > 0.0:
                analogical_pred = mixture / mixture.sum().clamp_min(1.0e-8)
                support_size = int((analogical_pred >= self.pgw.support_threshold).sum().item()) if self.pgw is not None else support_size
                transport_ok = True

        return MemoryRetrievalResult(
            analogical_pred=analogical_pred,
            confidence=confidence,
            retrieved_mol=retrieved_mol,
            retrieved_som_idx=retrieved_som,
            transport_succeeded=transport_ok,
            mcs_size=support_size,
            query_embed=query_embed,
            retrieved_embed_detached=retrieved_embed_detached,
            embedding_space=retrieval_embedding_space,
            transport_backend=transport_backend,
            transport_support_size=support_size,
            transported_mass=transported_mass,
            retrieved_same_query=(query_key is not None and self.historical_smiles[anchor_idx] == query_key),
            transport_distill_loss=transport_distill_loss,
            neuralgw_used_exact=neuralgw_used_exact,
            neuralgw_confidence=neuralgw_confidence,
            neuralgw_distill_loss=neuralgw_distill_loss,
            transport_plan=transport_plan,
            retrieved_node_multivectors=retrieved_multivectors,
            transport_error_message=transport_error_message,
            retrieval_mix_count=retrieval_mix_count,
            retrieval_mix_entropy=retrieval_mix_entropy,
            retrieved_morphism_target=retrieved_morph_target,
            retrieved_morphism_loss_mask=retrieved_morph_mask,
            retrieved_has_morphism_label=retrieved_has_morph,
            retrieved_label_confidence=retrieved_morph_conf,
            retrieved_scaffold=self.historical_scaffolds[anchor_idx] if anchor_idx < len(self.historical_scaffolds) else "",
            retrieval_candidate_count=retrieval_candidate_count,
            retrieval_mechanism_overlap=retrieval_mechanism_overlap,
            retrieval_diversity_score=retrieval_diversity_score,
            retrieval_projected_weight=retrieval_projected_weight,
            retrieval_best_tanimoto=retrieval_best_tanimoto,
            retrieval_best_projected_similarity=retrieval_best_projected_similarity,
            retrieval_best_mechanism_bonus=retrieval_best_mechanism_bonus,
            retrieval_best_structural_bonus=retrieval_best_structural_bonus,
            retrieval_best_hub_penalty=retrieval_best_hub_penalty,
            retrieval_shortlist_top_fraction=retrieval_shortlist_top_fraction,
            neuralgw_route_reason=neuralgw_route_reason,
        )

    def batch_stats(self, mols: List, true_soms: List[int]) -> dict:
        n = len(mols)
        successes = top1_hits = 0
        total_conf = total_mcs = 0.0

        for mol, true_som in zip(mols, true_soms):
            result = self.retrieve_and_transport(mol)
            total_conf += result.confidence
            total_mcs += result.mcs_size
            if result.transport_succeeded:
                successes += 1
                pred_atom = int(result.analogical_pred.argmax().item())
                if pred_atom == true_som:
                    top1_hits += 1

        return {
            "n": n,
            "transport_success_rate": successes / max(n, 1),
            "top1_accuracy": top1_hits / max(n, 1),
            "mean_confidence": total_conf / max(n, 1),
            "mean_mcs_size": total_mcs / max(n, 1),
        }
