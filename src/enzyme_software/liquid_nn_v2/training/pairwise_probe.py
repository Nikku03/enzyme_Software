from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    def _empty_pairwise_probe_payload(atom_embeddings, *, embedding_dim: int, expected_dim: int):
        empty_features = atom_embeddings.new_zeros((0, expected_dim))
        empty_labels = atom_embeddings.new_zeros((0,), dtype=torch.float32)
        return empty_features, empty_labels, {
            "pair_count": 0.0,
            "molecules_with_pairs": 0.0,
            "hard_neg_rank_mean": 0.0,
            "score_gap_mean": 0.0,
            "true_beats_fraction_before_pairwise": 0.0,
            "embedding_dim": float(embedding_dim),
            "pair_feature_dim": float(expected_dim),
            "positive_pair_count": 0.0,
            "negative_pair_count": 0.0,
        }


    def apply_candidate_mask_to_site_logits(
        site_logits,
        candidate_mask,
        *,
        mask_mode: str = "hard",
        logit_bias: float = 2.0,
    ):
        if site_logits is None or candidate_mask is None:
            return site_logits
        mask = candidate_mask.to(device=site_logits.device, dtype=site_logits.dtype).view_as(site_logits)
        if mask.numel() != site_logits.numel():
            return site_logits
        normalized_mode = str(mask_mode or "hard").strip().lower()
        if normalized_mode == "off":
            return site_logits
        if normalized_mode == "soft":
            return site_logits - ((1.0 - mask) * float(logit_bias))
        return torch.where(mask > 0.5, site_logits, torch.full_like(site_logits, -20.0))


    def build_pairwise_probe_examples(
        *,
        atom_embeddings,
        site_logits,
        site_labels,
        batch_index,
        supervision_mask=None,
        candidate_mask=None,
        max_pairs: Optional[int] = None,
    ):
        if atom_embeddings.ndim != 2:
            raise ValueError(f"Expected atom embeddings [A, D], got {tuple(atom_embeddings.shape)}")
        embedding_dim = int(atom_embeddings.size(-1))
        expected_dim = (4 * embedding_dim) + 2

        if atom_embeddings.numel() == 0:
            return _empty_pairwise_probe_payload(
                atom_embeddings,
                embedding_dim=embedding_dim,
                expected_dim=expected_dim,
            )

        scores = site_logits.view(-1)
        labels = site_labels.view(-1) > 0.5
        mol_batch = batch_index.view(-1)
        supervision = (
            supervision_mask.view(-1) > 0.5
            if supervision_mask is not None
            else torch.ones_like(labels, dtype=torch.bool)
        )
        ranking = (
            candidate_mask.view(-1) > 0.5
            if candidate_mask is not None
            else torch.ones_like(labels, dtype=torch.bool)
        )
        valid_mask = supervision & ranking

        pair_groups = []
        pair_gaps = []
        pair_ranks = []
        pair_beats = []
        molecules_with_pairs = 0

        num_molecules = int(mol_batch.max().item()) + 1 if mol_batch.numel() else 0
        for mol_idx in range(num_molecules):
            mol_mask = mol_batch == mol_idx
            mol_valid = mol_mask & valid_mask
            if not bool(mol_valid.any()):
                continue
            mol_true = torch.where(mol_valid & labels)[0]
            mol_false = torch.where(mol_valid & (~labels))[0]
            if int(mol_true.numel()) == 0 or int(mol_false.numel()) == 0:
                continue
            false_scores = scores[mol_false]
            hard_idx = mol_false[int(torch.argmax(false_scores).item())]
            ranked_indices = torch.where(mol_valid)[0][torch.argsort(scores[mol_valid], descending=True)]
            hard_rank_tensor = torch.where(ranked_indices == hard_idx)[0]
            hard_rank = int(hard_rank_tensor[0].item()) + 1 if int(hard_rank_tensor.numel()) > 0 else 0
            hard_embedding = atom_embeddings[hard_idx]
            hard_score = scores[hard_idx]

            created_pair = False
            for true_idx in mol_true.tolist():
                true_embedding = atom_embeddings[int(true_idx)]
                true_score = scores[int(true_idx)]
                score_gap = float((true_score - hard_score).detach().item())
                # The Stage 1 probe is directional: both (true, hard) -> 1 and
                # (hard, true) -> 0 are emitted so the probe cannot collapse to
                # the previous degenerate all-positive target.
                pos_feature = torch.cat(
                    [
                        true_embedding,
                        hard_embedding,
                        true_embedding - hard_embedding,
                        true_embedding * hard_embedding,
                        (true_score - hard_score).view(1),
                        (true_score - hard_score).abs().view(1),
                    ],
                    dim=0,
                )
                neg_feature = torch.cat(
                    [
                        hard_embedding,
                        true_embedding,
                        hard_embedding - true_embedding,
                        hard_embedding * true_embedding,
                        (hard_score - true_score).view(1),
                        (hard_score - true_score).abs().view(1),
                    ],
                    dim=0,
                )
                if int(pos_feature.numel()) != int(expected_dim) or int(neg_feature.numel()) != int(expected_dim):
                    raise ValueError(
                        "Expected pair feature dim "
                        f"{expected_dim}, got pos={int(pos_feature.numel())}, neg={int(neg_feature.numel())}"
                    )
                pair_groups.append((pos_feature, neg_feature))
                pair_gaps.append(score_gap)
                pair_ranks.append(float(hard_rank))
                pair_beats.append(float(score_gap > 0.0))
                created_pair = True
            if created_pair:
                molecules_with_pairs += 1

        if not pair_groups:
            return _empty_pairwise_probe_payload(
                atom_embeddings,
                embedding_dim=embedding_dim,
                expected_dim=expected_dim,
            )

        gap_tensor = atom_embeddings.new_tensor(pair_gaps, dtype=torch.float32)
        rank_tensor = atom_embeddings.new_tensor(pair_ranks, dtype=torch.float32)
        beats_tensor = atom_embeddings.new_tensor(pair_beats, dtype=torch.float32)

        if max_pairs is not None and int(max_pairs) > 0:
            max_groups = max(0, int(max_pairs) // 2)
            if max_groups > 0 and int(len(pair_groups)) > max_groups:
                keep = torch.argsort(gap_tensor, descending=False)[:max_groups]
                keep_list = keep.tolist()
                pair_groups = [pair_groups[idx] for idx in keep_list]
                gap_tensor = gap_tensor[keep]
                rank_tensor = rank_tensor[keep]
                beats_tensor = beats_tensor[keep]
            elif max_groups == 0:
                return _empty_pairwise_probe_payload(
                    atom_embeddings,
                    embedding_dim=embedding_dim,
                    expected_dim=expected_dim,
                )

        pair_rows = []
        pair_labels = []
        for pos_feature, neg_feature in pair_groups:
            pair_rows.extend((pos_feature, neg_feature))
            pair_labels.extend((1.0, 0.0))

        pair_features = torch.stack(pair_rows, dim=0)
        pair_targets = atom_embeddings.new_tensor(pair_labels, dtype=torch.float32)

        pair_features = torch.nan_to_num(pair_features, nan=0.0, posinf=0.0, neginf=0.0)
        if not bool(torch.isfinite(pair_features).all()):
            raise FloatingPointError("Non-finite pairwise probe features detected")
        unique_labels = torch.unique(pair_targets)
        if int(pair_targets.numel()) > 0 and int(unique_labels.numel()) != 2:
            raise RuntimeError(
                "Pairwise probe requires both directional classes; "
                f"got labels {unique_labels.detach().cpu().tolist()}"
            )

        return pair_features, pair_targets, {
            "pair_count": float(pair_targets.numel()),
            "molecules_with_pairs": float(molecules_with_pairs),
            "hard_neg_rank_mean": float(rank_tensor.mean().item()) if rank_tensor.numel() else 0.0,
            "score_gap_mean": float(gap_tensor.mean().item()) if gap_tensor.numel() else 0.0,
            "true_beats_fraction_before_pairwise": float(beats_tensor.mean().item()) if beats_tensor.numel() else 0.0,
            "embedding_dim": float(embedding_dim),
            "pair_feature_dim": float(expected_dim),
            "positive_pair_count": float((pair_targets > 0.5).sum().item()),
            "negative_pair_count": float((pair_targets <= 0.5).sum().item()),
        }


    def zero_pairwise_probe_metrics() -> Dict[str, float]:
        return {
            "pairwise_loss": 0.0,
            "pairwise_accuracy": 0.0,
            "pairwise_auc": 0.5,
            "pair_count": 0.0,
            "molecules_with_pairs": 0.0,
            "hard_neg_rank_mean": 0.0,
            "score_gap_mean": 0.0,
            "true_beats_fraction_before_pairwise": 0.0,
            "pairwise_mean_probability": 0.0,
            "pairwise_positive_rate": 0.0,
            "pairwise_mean_probability_pos": 0.0,
            "pairwise_mean_probability_neg": 0.0,
            "pairwise_logit_mean_pos": 0.0,
            "pairwise_logit_mean_neg": 0.0,
            "zero_pair_batches": 0.0,
            "single_class_batches": 0.0,
        }
else:  # pragma: no cover
    def apply_candidate_mask_to_site_logits(*args, **kwargs):
        require_torch()

    def build_pairwise_probe_examples(*args, **kwargs):
        require_torch()

    def zero_pairwise_probe_metrics() -> Dict[str, float]:
        require_torch()
