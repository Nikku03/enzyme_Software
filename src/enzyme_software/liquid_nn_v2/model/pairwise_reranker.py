"""Phase 2: Pairwise Reranker for Site-of-Metabolism Prediction.

Uses the trained pairwise head (77% pairwise accuracy) to rerank the top-K
candidates from the proposer via tournament-style voting.

Strategy:
1. Get top-K (e.g., K=6) candidates from proposer scores
2. Run all-pairs comparisons using pairwise head
3. Aggregate via Copeland score (wins - losses) or Bradley-Terry
4. Return reranked candidates

This is an inference-time module that wraps a trained pairwise head.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch

if TORCH_AVAILABLE:
    import torch.nn as nn
    import torch.nn.functional as F

    class PairwiseReranker(nn.Module):
        """Reranks top-K candidates using pairwise comparisons.
        
        Given atom embeddings and initial proposer scores, selects top-K candidates
        and performs all-pairs comparisons using a trained pairwise head to produce
        a refined ranking.
        """

        def __init__(
            self,
            pairwise_head: nn.Module,
            *,
            top_k: int = 6,
            aggregation: str = "copeland",  # "copeland", "bradley_terry", "sum"
            temperature: float = 1.0,
            min_candidates: int = 2,
        ):
            """Initialize the reranker.
            
            Args:
                pairwise_head: Trained PairwiseHead module
                top_k: Number of top candidates to consider for reranking
                aggregation: Method to aggregate pairwise scores
                    - "copeland": Count wins minus losses (robust)
                    - "bradley_terry": Log-likelihood aggregation
                    - "sum": Sum of pairwise probabilities
                temperature: Softmax temperature for final scores
                min_candidates: Minimum candidates needed to rerank
            """
            super().__init__()
            self.pairwise_head = pairwise_head
            self.top_k = int(top_k)
            self.aggregation = str(aggregation).lower()
            self.temperature = float(temperature)
            self.min_candidates = int(min_candidates)
            
            # Freeze pairwise head
            for param in self.pairwise_head.parameters():
                param.requires_grad = False

        def _build_pair_features(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            score_a: torch.Tensor,
            score_b: torch.Tensor,
        ) -> torch.Tensor:
            """Build pairwise feature vector matching training format.
            
            Feature format: [emb_a, emb_b, emb_a - emb_b, emb_a * emb_b, score_gap, |score_gap|]
            """
            score_gap = score_a - score_b
            return torch.cat([
                emb_a,
                emb_b,
                emb_a - emb_b,
                emb_a * emb_b,
                score_gap.view(-1),
                score_gap.abs().view(-1),
            ], dim=-1)

        def _rerank_single_molecule(
            self,
            atom_embeddings: torch.Tensor,  # [num_atoms, D]
            proposer_scores: torch.Tensor,   # [num_atoms]
            candidate_mask: Optional[torch.Tensor] = None,  # [num_atoms]
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
            """Rerank candidates for a single molecule.
            
            Returns:
                reranked_scores: [num_atoms] tensor with updated scores
                reranked_indices: [K] tensor with indices sorted by reranked score
                metadata: Dict with reranking statistics
            """
            num_atoms = atom_embeddings.size(0)
            device = atom_embeddings.device
            
            # Apply candidate mask if provided
            if candidate_mask is not None:
                valid_mask = candidate_mask.view(-1) > 0.5
            else:
                valid_mask = torch.ones(num_atoms, dtype=torch.bool, device=device)
            
            valid_indices = torch.where(valid_mask)[0]
            num_valid = valid_indices.numel()
            
            # Not enough candidates to rerank
            if num_valid < self.min_candidates:
                return proposer_scores, torch.argsort(proposer_scores, descending=True), {
                    "reranked": False,
                    "num_candidates": float(num_valid),
                    "reason": "insufficient_candidates",
                }
            
            # Get top-K candidates
            valid_scores = proposer_scores[valid_indices]
            k = min(self.top_k, num_valid)
            topk_local_indices = torch.argsort(valid_scores, descending=True)[:k]
            topk_global_indices = valid_indices[topk_local_indices]
            topk_embeddings = atom_embeddings[topk_global_indices]  # [K, D]
            topk_scores = proposer_scores[topk_global_indices]  # [K]
            
            # Build all-pairs comparison matrix
            # For K candidates, we have K*(K-1)/2 unique pairs
            pair_features_list = []
            pair_indices = []  # (i, j) pairs where we predict P(i > j)
            
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    pair_feat = self._build_pair_features(
                        topk_embeddings[i],
                        topk_embeddings[j],
                        topk_scores[i],
                        topk_scores[j],
                    )
                    pair_features_list.append(pair_feat)
                    pair_indices.append((i, j))
            
            if not pair_features_list:
                return proposer_scores, torch.argsort(proposer_scores, descending=True), {
                    "reranked": False,
                    "num_candidates": float(k),
                    "reason": "no_pairs",
                }
            
            # Run pairwise head
            pair_features = torch.stack(pair_features_list, dim=0)  # [K*(K-1), D_pair]
            with torch.no_grad():
                pair_logits = self.pairwise_head(pair_features).view(-1)  # [K*(K-1)]
                pair_probs = torch.sigmoid(pair_logits)  # P(first > second)
            
            # Build comparison matrix: M[i,j] = P(i > j)
            comparison_matrix = torch.zeros(k, k, device=device)
            for idx, (i, j) in enumerate(pair_indices):
                comparison_matrix[i, j] = pair_probs[idx]
            
            # Aggregate pairwise scores
            if self.aggregation == "copeland":
                # Copeland score: sum of wins (P > 0.5) minus losses
                wins = (comparison_matrix > 0.5).float().sum(dim=1)
                losses = (comparison_matrix < 0.5).float().sum(dim=1)
                aggregated_scores = wins - losses
            elif self.aggregation == "bradley_terry":
                # Sum of log-odds
                eps = 1e-6
                log_odds = torch.log(comparison_matrix + eps) - torch.log(1 - comparison_matrix + eps)
                aggregated_scores = log_odds.sum(dim=1)
            else:  # "sum"
                # Simple sum of win probabilities
                aggregated_scores = comparison_matrix.sum(dim=1)
            
            # Normalize to [0, 1] range for combining with original scores
            if aggregated_scores.max() > aggregated_scores.min():
                normalized_scores = (aggregated_scores - aggregated_scores.min()) / (
                    aggregated_scores.max() - aggregated_scores.min()
                )
            else:
                normalized_scores = torch.ones_like(aggregated_scores) * 0.5
            
            # Create output scores: keep original for non-top-K, use reranked for top-K
            reranked_scores = proposer_scores.clone()
            
            # Blend reranked scores with original proposer scores
            # Use reranked order but preserve relative magnitudes
            rerank_order = torch.argsort(aggregated_scores, descending=True)
            original_order = torch.argsort(topk_scores, descending=True)
            
            # Assign new scores based on reranked positions
            # Highest reranked gets highest original score, etc.
            sorted_original_scores = topk_scores[original_order]
            for new_rank, orig_rank in enumerate(rerank_order.tolist()):
                global_idx = topk_global_indices[orig_rank]
                reranked_scores[global_idx] = sorted_original_scores[new_rank]
            
            # Compute statistics
            original_top1 = topk_global_indices[0]
            reranked_top1 = topk_global_indices[rerank_order[0]]
            top1_changed = original_top1 != reranked_top1
            
            # Kendall tau correlation between original and reranked
            original_ranks = torch.argsort(torch.argsort(topk_scores, descending=True))
            reranked_ranks = torch.argsort(torch.argsort(aggregated_scores, descending=True))
            concordant = 0
            discordant = 0
            for i in range(k):
                for j in range(i + 1, k):
                    orig_order = (original_ranks[i] - original_ranks[j]).sign()
                    new_order = (reranked_ranks[i] - reranked_ranks[j]).sign()
                    if orig_order == new_order:
                        concordant += 1
                    else:
                        discordant += 1
            total_pairs = concordant + discordant
            kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
            
            metadata = {
                "reranked": True,
                "num_candidates": float(k),
                "top1_changed": float(top1_changed),
                "original_top1_idx": float(original_top1.item()),
                "reranked_top1_idx": float(reranked_top1.item()),
                "kendall_tau": kendall_tau,
                "mean_pairwise_prob": float(pair_probs.mean().item()),
                "pairwise_confidence": float((pair_probs - 0.5).abs().mean().item()),
            }
            
            return reranked_scores, torch.argsort(reranked_scores, descending=True), metadata

        def forward(
            self,
            atom_embeddings: torch.Tensor,  # [total_atoms, D]
            proposer_scores: torch.Tensor,   # [total_atoms]
            batch_index: torch.Tensor,       # [total_atoms] molecule assignment
            candidate_mask: Optional[torch.Tensor] = None,  # [total_atoms]
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Rerank candidates for a batch of molecules.
            
            Args:
                atom_embeddings: Atom feature embeddings from backbone
                proposer_scores: Initial scores from proposer (site_logits)
                batch_index: Molecule assignment for each atom
                candidate_mask: Optional mask for valid candidates
                
            Returns:
                reranked_scores: [total_atoms] tensor with updated scores
                metadata: Aggregated statistics across molecules
            """
            device = atom_embeddings.device
            num_atoms = atom_embeddings.size(0)
            
            if num_atoms == 0:
                return proposer_scores, {"reranked_molecules": 0.0}
            
            num_molecules = int(batch_index.max().item()) + 1
            reranked_scores = proposer_scores.clone()
            
            # Aggregate metadata
            total_reranked = 0
            total_top1_changed = 0
            kendall_sum = 0.0
            confidence_sum = 0.0
            
            for mol_idx in range(num_molecules):
                mol_mask = batch_index == mol_idx
                mol_indices = torch.where(mol_mask)[0]
                
                if mol_indices.numel() == 0:
                    continue
                
                mol_embeddings = atom_embeddings[mol_indices]
                mol_scores = proposer_scores[mol_indices]
                mol_candidate_mask = candidate_mask[mol_indices] if candidate_mask is not None else None
                
                mol_reranked, _, mol_meta = self._rerank_single_molecule(
                    mol_embeddings,
                    mol_scores,
                    mol_candidate_mask,
                )
                
                reranked_scores[mol_indices] = mol_reranked
                
                if mol_meta.get("reranked", False):
                    total_reranked += 1
                    total_top1_changed += int(mol_meta.get("top1_changed", 0))
                    kendall_sum += mol_meta.get("kendall_tau", 0.0)
                    confidence_sum += mol_meta.get("pairwise_confidence", 0.0)
            
            metadata = {
                "reranked_molecules": float(total_reranked),
                "top1_changed_count": float(total_top1_changed),
                "top1_changed_fraction": float(total_top1_changed) / max(1, total_reranked),
                "mean_kendall_tau": kendall_sum / max(1, total_reranked),
                "mean_pairwise_confidence": confidence_sum / max(1, total_reranked),
            }
            
            return reranked_scores, metadata

        def rerank_with_labels(
            self,
            atom_embeddings: torch.Tensor,
            proposer_scores: torch.Tensor,
            batch_index: torch.Tensor,
            site_labels: torch.Tensor,
            candidate_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, float]:
            """Rerank and compute metrics against ground truth labels.
            
            Returns metrics showing impact of reranking on prediction quality.
            """
            device = atom_embeddings.device
            num_atoms = atom_embeddings.size(0)
            
            if num_atoms == 0:
                return {
                    "reranker_corrected_count": 0.0,
                    "reranker_harmed_count": 0.0,
                    "reranker_unchanged_count": 0.0,
                    "original_top1_acc": 0.0,
                    "reranked_top1_acc": 0.0,
                    "reranker_lift": 0.0,
                }
            
            num_molecules = int(batch_index.max().item()) + 1
            labels = site_labels.view(-1) > 0.5
            
            original_correct = 0
            reranked_correct = 0
            corrected = 0  # Was wrong, now correct
            harmed = 0     # Was correct, now wrong
            unchanged = 0  # Same outcome
            total_molecules = 0
            
            for mol_idx in range(num_molecules):
                mol_mask = batch_index == mol_idx
                mol_indices = torch.where(mol_mask)[0]
                
                if mol_indices.numel() == 0:
                    continue
                
                mol_labels = labels[mol_indices]
                if not mol_labels.any():
                    continue  # No true sites
                
                mol_embeddings = atom_embeddings[mol_indices]
                mol_scores = proposer_scores[mol_indices]
                mol_candidate_mask = candidate_mask[mol_indices] if candidate_mask is not None else None
                
                # Original prediction
                if mol_candidate_mask is not None:
                    valid_scores = torch.where(
                        mol_candidate_mask.view(-1) > 0.5, 
                        mol_scores.view(-1), 
                        torch.full_like(mol_scores.view(-1), -1e9)
                    )
                else:
                    valid_scores = mol_scores.view(-1)
                original_top1_local = torch.argmax(valid_scores)
                original_is_correct = mol_labels[original_top1_local].item()
                
                # Reranked prediction
                mol_reranked, _, _ = self._rerank_single_molecule(
                    mol_embeddings,
                    mol_scores,
                    mol_candidate_mask,
                )
                if mol_candidate_mask is not None:
                    valid_reranked = torch.where(
                        mol_candidate_mask.view(-1) > 0.5, 
                        mol_reranked.view(-1), 
                        torch.full_like(mol_reranked.view(-1), -1e9)
                    )
                else:
                    valid_reranked = mol_reranked.view(-1)
                reranked_top1_local = torch.argmax(valid_reranked)
                reranked_is_correct = mol_labels[reranked_top1_local].item()
                
                total_molecules += 1
                original_correct += int(original_is_correct)
                reranked_correct += int(reranked_is_correct)
                
                if not original_is_correct and reranked_is_correct:
                    corrected += 1
                elif original_is_correct and not reranked_is_correct:
                    harmed += 1
                else:
                    unchanged += 1
            
            original_acc = original_correct / max(1, total_molecules)
            reranked_acc = reranked_correct / max(1, total_molecules)
            
            return {
                "reranker_corrected_count": float(corrected),
                "reranker_harmed_count": float(harmed),
                "reranker_unchanged_count": float(unchanged),
                "reranker_corrected_fraction": float(corrected) / max(1, total_molecules),
                "reranker_harmed_fraction": float(harmed) / max(1, total_molecules),
                "original_top1_acc": original_acc,
                "reranked_top1_acc": reranked_acc,
                "reranker_lift": reranked_acc - original_acc,
                "reranker_lift_relative": (reranked_acc - original_acc) / max(0.01, original_acc),
                "total_molecules": float(total_molecules),
            }

else:
    class PairwiseReranker:
        def __init__(self, *args, **kwargs):
            require_torch()
