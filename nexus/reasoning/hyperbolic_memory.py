"""
nexus/reasoning/hyperbolic_memory.py

Hyperbolic analogue of the ECFP4 memory bank.

Fingerprints are first L2-normalised in Euclidean space, then projected
strictly inside the Poincare ball so nearest-neighbour lookup can use the
hyperbolic geodesic distance.  Transport remains discrete and scaffold-based
via RDKit MCS, matching the existing baseline memory-bank contract.
"""
from __future__ import annotations

import math
from typing import List, Mapping

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False

from .baseline_memory import MemoryRetrievalResult, _extract_som_idx, _morgan_fp_tensor
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

        self.historical_mols: List = []
        self.historical_soms: List[int] = []
        self.historical_smiles: List[str | None] = []
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
        self.historical_node_multivectors = []
        self.memory_embeddings = None
        self.memory_fingerprints = None
        self.memory_bit_counts = None
        self.memory_projected_mask = None

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

    def populate_from_mols(self, mols: List, *, continuous_encoder=None) -> None:
        print(f"Populating Hyperbolic Memory Bank from {len(mols)} input molecules...")
        self._reset()
        fps: List[torch.Tensor] = []
        projected_embeddings: List[torch.Tensor | None] = []
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
            projected: torch.Tensor | None = None
            node_multivectors: torch.Tensor | None = None
            if continuous_encoder is not None and canonical:
                try:
                    encoded = continuous_encoder(canonical)
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
                except Exception:
                    projected = None
                    node_multivectors = None
            projected_embeddings.append(projected)
            self.historical_node_multivectors.append(node_multivectors)

        if not fps:
            raise RuntimeError("No labelled molecules found — hyperbolic memory bank is empty.")

        raw = torch.stack(fps).to(self.device)
        self.memory_fingerprints = raw
        self.memory_bit_counts = raw.sum(dim=-1)
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

    def _tanimoto_similarity(self, q_fp: torch.Tensor) -> torch.Tensor:
        if self.memory_fingerprints is None or self.memory_bit_counts is None:
            raise RuntimeError("Call populate_from_mols() before retrieve_and_transport().")
        query = q_fp.to(self.device)
        intersection = torch.matmul(self.memory_fingerprints, query)
        q_count = query.sum()
        union = (self.memory_bit_counts + q_count - intersection).clamp_min(1.0e-6)
        return intersection / union

    def _mcs_transport(self, query_mol, retrieved_mol, retrieved_som: int) -> tuple[torch.Tensor, bool, int]:
        n_query = query_mol.GetNumAtoms()
        analogical_pred = torch.zeros(n_query, dtype=torch.float32, device=self.device)
        mcs_size = 0
        transport_ok = False
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
                try:
                    mcs_pos = match_retrieved.index(retrieved_som)
                    mapped_query_idx = match_query[mcs_pos]
                    analogical_pred[mapped_query_idx] = 1.0
                    transport_ok = True
                except ValueError:
                    pass
        return analogical_pred, transport_ok, mcs_size

    def retrieve_and_transport(
        self,
        query_mol,
        query_smiles: str | None = None,
        mechanism_encoder=None,
        query_embedding: torch.Tensor | None = None,
        query_multivectors: torch.Tensor | None = None,
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

        projected_ready = (
            query_embedding is not None
            and self.memory_projected_mask is not None
            and bool(self.memory_projected_mask.any().item())
        )

        query_embed: torch.Tensor | None = None
        retrieved_embed_detached: torch.Tensor | None = None

        if projected_ready:
            candidate_mask = self.memory_projected_mask.clone()
            if same_query is not None:
                candidate_mask = candidate_mask & (~same_query)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
            if candidate_indices.numel() > 0:
                target_dim = int(self.memory_embeddings.size(-1))
                prepared_query = torch.as_tensor(query_embedding, dtype=torch.float32, device=self.device).view(-1)
                if prepared_query.numel() < target_dim:
                    prepared_query = F.pad(prepared_query, (0, target_dim - prepared_query.numel()))
                elif prepared_query.numel() > target_dim:
                    prepared_query = prepared_query[:target_dim]
                q_embed_h = self._project_inside_ball(
                    prepared_query.view(1, -1)
                )
                candidate_embeddings = self.memory_embeddings.index_select(0, candidate_indices)
                distances = self._poincare_distance(q_embed_h, candidate_embeddings).view(-1)
                shortlist_k = min(self.tanimoto_shortlist_k, int(candidate_indices.numel()))
                top_distances, top_local = torch.topk(distances, shortlist_k, largest=False)
                best_idx = int(candidate_indices[int(top_local[0].item())].item())
                tau = 10.0
                confidence = math.exp(-float(top_distances[0].item()) / tau)
                query_embed = q_embed_h.squeeze(0)
                retrieved_embed_detached = self.memory_embeddings[best_idx].detach()
            else:
                projected_ready = False

        if not projected_ready:
            q_fp = _morgan_fp_tensor(query_mol, radius=self.fp_radius, n_bits=self.fp_bits)
            tanimoto_scores = self._tanimoto_similarity(q_fp)
            if same_query is not None and bool(same_query.any().item()):
                tanimoto_scores = tanimoto_scores.masked_fill(same_query, -1.0)
            shortlist_k = min(self.tanimoto_shortlist_k, len(self.historical_mols))
            top_tanimoto, shortlist_indices = torch.topk(tanimoto_scores, shortlist_k, largest=True)
            best_tanimoto = float(top_tanimoto[0].item()) if shortlist_k > 0 else -1.0
            best_shortlist_idx = int(shortlist_indices[0].item()) if shortlist_k > 0 else 0

            if best_tanimoto < self.tanimoto_prefilter_threshold:
                n_query = query_mol.GetNumAtoms()
                return MemoryRetrievalResult(
                    analogical_pred=torch.zeros(n_query, dtype=torch.float32, device=self.device),
                    confidence=0.0,
                    retrieved_mol=self.historical_mols[best_shortlist_idx],
                    retrieved_som_idx=self.historical_soms[best_shortlist_idx],
                    transport_succeeded=False,
                    mcs_size=0,
                    embedding_space="hyperbolic" if projected_ready else "euclidean",
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

                    # Cosine similarity between query embed and shortlist embeds.
                    mech_scores = (query_embed.detach().unsqueeze(0) * shortlist_embeds.detach()).sum(dim=-1)  # [k]

                    # Combine Poincaré distance ranking with mechanism score:
                    #   final_score = mech_scores (higher = better)
                    # Use mechanism score to override the Poincaré selection.
                    best_mech_rank = int(mech_scores.argmax().item())
                    best_idx = int(shortlist_indices[best_mech_rank].item())

                    retrieved_embed_detached = shortlist_embeds[best_mech_rank].detach()

                    # Compute confidence from hyperbolic distance for the selected candidate.
                    q_h_embed = self._encode_hyperbolic(q_fp.unsqueeze(0).to(self.device))
                    if q_h_embed.size(-1) < target_dim:
                        q_h_embed = F.pad(q_h_embed, (0, target_dim - q_h_embed.size(-1)))
                    elif q_h_embed.size(-1) > target_dim:
                        q_h_embed = q_h_embed[..., :target_dim]
                    r_h_embed = self.memory_embeddings[best_idx].unsqueeze(0)
                    dist = float(self._poincare_distance(q_h_embed, r_h_embed).item())
                    tau = 10.0
                    confidence = math.exp(-dist / tau)

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

                k = min(2, shortlist_k)
                top_distances, top_indices = torch.topk(distances, k, largest=False)

                tau = 10.0  # Aggressive temperature scalar: exp(-d/tau) keeps confidence
                            # high for typical hyperbolic distances (~2.9 → exp(-0.29)=0.748)
                if top_distances[0] < 1e-4 and k > 1:
                    best_idx = int(shortlist_indices[top_indices[1]].item())
                    confidence = math.exp(-float(top_distances[1].item()) / tau)
                else:
                    best_idx = int(shortlist_indices[top_indices[0]].item())
                    confidence = math.exp(-float(top_distances[0].item()) / tau)

        retrieved_mol = self.historical_mols[best_idx]
        retrieved_som = self.historical_soms[best_idx]

        analogical_pred: torch.Tensor
        transport_ok: bool
        support_size: int
        if self.transport_backend == "pgw" and self.pgw is not None:
            retrieved_multivectors = None
            if 0 <= best_idx < len(self.historical_node_multivectors):
                retrieved_multivectors = self.historical_node_multivectors[best_idx]
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
            if (not transport_ok) and self.fallback_to_mcs:
                analogical_pred, transport_ok, support_size = self._mcs_transport(query_mol, retrieved_mol, retrieved_som)
        else:
            analogical_pred, transport_ok, support_size = self._mcs_transport(query_mol, retrieved_mol, retrieved_som)

        return MemoryRetrievalResult(
            analogical_pred=analogical_pred,
            confidence=confidence,
            retrieved_mol=retrieved_mol,
            retrieved_som_idx=retrieved_som,
            transport_succeeded=transport_ok,
            mcs_size=support_size,
            query_embed=query_embed,
            retrieved_embed_detached=retrieved_embed_detached,
            embedding_space="hyperbolic" if projected_ready else "euclidean",
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
