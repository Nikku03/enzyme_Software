from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    def zero_pairwise_distillation_metrics() -> Dict[str, float]:
        return {
            "distilled_kl_loss": 0.0,
            "distilled_target_entropy_mean": 0.0,
            "distilled_target_max_mean": 0.0,
            "distilled_target_argmax_match_true_fraction": 0.0,
            "distilled_pred_argmax_match_target_fraction": 0.0,
            "distilled_target_true_mass_mean": 0.0,
            "candidate_count_mean": 0.0,
            "skipped_singleton_candidate_molecules": 0.0,
            "distilled_molecule_count": 0.0,
        }


    def _empty_payload():
        return {"molecules": [], "metrics": zero_pairwise_distillation_metrics()}


    def _build_pair_feature_matrix(atom_embeddings, teacher_scores):
        candidate_count = int(atom_embeddings.size(0))
        embedding_dim = int(atom_embeddings.size(-1))
        input_dim = (4 * embedding_dim) + 2
        if candidate_count <= 1:
            return atom_embeddings.new_zeros((0, input_dim)), []
        pair_rows = []
        pair_indices: list[tuple[int, int]] = []
        for left_idx in range(candidate_count):
            for right_idx in range(candidate_count):
                if left_idx == right_idx:
                    continue
                left_embedding = atom_embeddings[left_idx]
                right_embedding = atom_embeddings[right_idx]
                score_delta = teacher_scores[left_idx] - teacher_scores[right_idx]
                pair_rows.append(
                    torch.cat(
                        [
                            left_embedding,
                            right_embedding,
                            left_embedding - right_embedding,
                            left_embedding * right_embedding,
                            score_delta.view(1),
                            score_delta.abs().view(1),
                        ],
                        dim=0,
                    )
                )
                pair_indices.append((left_idx, right_idx))
        pair_features = torch.stack(pair_rows, dim=0)
        if not bool(torch.isfinite(pair_features).all()):
            raise FloatingPointError("Non-finite pairwise distillation teacher features detected")
        return pair_features, pair_indices


    def build_pairwise_distilled_targets(
        *,
        atom_embeddings,
        teacher_site_logits,
        site_labels,
        batch_index,
        pairwise_head,
        supervision_mask=None,
        candidate_mask=None,
        candidate_topk: int = 6,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        restrict_to_candidates: bool = True,
    ):
        if atom_embeddings.ndim != 2:
            raise ValueError(f"Expected atom embeddings [A, D], got {tuple(atom_embeddings.shape)}")

        scores = teacher_site_logits.view(-1)
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
        valid_mask = supervision & (ranking if restrict_to_candidates else torch.ones_like(ranking, dtype=torch.bool))
        num_molecules = int(mol_batch.max().item()) + 1 if mol_batch.numel() else 0
        target_temperature = max(1.0e-6, float(temperature))
        smoothing = min(max(float(label_smoothing), 0.0), 0.5)

        payloads = []
        entropy_values = []
        max_values = []
        argmax_true_hits = []
        true_mass_values = []
        candidate_counts = []
        skipped_singletons = 0

        for mol_idx in range(num_molecules):
            mol_valid = (mol_batch == mol_idx) & valid_mask
            if not bool(mol_valid.any()):
                continue
            mol_indices = torch.nonzero(mol_valid, as_tuple=False).view(-1)
            mol_scores = scores[mol_indices]
            if int(mol_scores.numel()) == 0:
                continue
            if int(candidate_topk) > 0 and int(mol_scores.numel()) > int(candidate_topk):
                order = torch.argsort(mol_scores, descending=True)[: int(candidate_topk)]
                mol_indices = mol_indices[order]
                mol_scores = mol_scores[order]
            candidate_count = int(mol_indices.numel())
            if candidate_count <= 1:
                skipped_singletons += 1
                continue

            mol_embeddings = atom_embeddings[mol_indices].detach()
            mol_scores = mol_scores.detach()
            pair_features, pair_indices = _build_pair_feature_matrix(mol_embeddings, mol_scores)
            if int(pair_features.size(0)) == 0:
                skipped_singletons += 1
                continue
            with torch.no_grad():
                pair_logits = pairwise_head(pair_features).view(-1)
                if not bool(torch.isfinite(pair_logits).all()):
                    raise FloatingPointError("Non-finite pairwise teacher logits detected")
                pair_probabilities = torch.sigmoid(pair_logits)
            win_matrix = mol_embeddings.new_zeros((candidate_count, candidate_count), dtype=pair_probabilities.dtype)
            for pair_offset, (left_idx, right_idx) in enumerate(pair_indices):
                win_matrix[left_idx, right_idx] = pair_probabilities[pair_offset]
            win_strength = win_matrix.sum(dim=1) / float(max(1, candidate_count - 1))
            target_distribution = torch.softmax(win_strength / target_temperature, dim=0)
            if smoothing > 0.0:
                uniform = target_distribution.new_full((candidate_count,), 1.0 / float(candidate_count))
                target_distribution = ((1.0 - smoothing) * target_distribution) + (smoothing * uniform)
                target_distribution = target_distribution / target_distribution.sum().clamp_min(1.0e-8)

            mol_labels = labels[mol_indices].to(dtype=target_distribution.dtype)
            entropy_values.append(float((-(target_distribution * target_distribution.clamp_min(1.0e-8).log())).sum().item()))
            max_values.append(float(target_distribution.max().item()))
            argmax_true_hits.append(float(mol_labels[int(torch.argmax(target_distribution).item())].item() > 0.5))
            true_mass_values.append(float(target_distribution[mol_labels > 0.5].sum().item()) if bool((mol_labels > 0.5).any()) else 0.0)
            candidate_counts.append(float(candidate_count))
            payloads.append(
                {
                    "candidate_indices": mol_indices,
                    "target_distribution": target_distribution,
                    "target_argmax_index": int(torch.argmax(target_distribution).item()),
                    "candidate_labels": mol_labels,
                }
            )

        if not payloads:
            metrics = zero_pairwise_distillation_metrics()
            metrics["skipped_singleton_candidate_molecules"] = float(skipped_singletons)
            return {"molecules": [], "metrics": metrics}

        metrics = {
            "distilled_target_entropy_mean": float(sum(entropy_values) / len(entropy_values)),
            "distilled_target_max_mean": float(sum(max_values) / len(max_values)),
            "distilled_target_argmax_match_true_fraction": float(sum(argmax_true_hits) / len(argmax_true_hits)),
            "distilled_target_true_mass_mean": float(sum(true_mass_values) / len(true_mass_values)),
            "candidate_count_mean": float(sum(candidate_counts) / len(candidate_counts)),
            "skipped_singleton_candidate_molecules": float(skipped_singletons),
            "distilled_molecule_count": float(len(payloads)),
        }
        return {"molecules": payloads, "metrics": metrics}
else:  # pragma: no cover
    def build_pairwise_distilled_targets(*args, **kwargs):
        require_torch()

    def zero_pairwise_distillation_metrics() -> Dict[str, float]:
        require_torch()
