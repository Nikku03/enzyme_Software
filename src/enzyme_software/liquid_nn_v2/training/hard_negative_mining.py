from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


HARD_NEGATIVE_TYPE_NAMES = ("top_score", "graph_local", "3d_local")
_HARD_NEGATIVE_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(HARD_NEGATIVE_TYPE_NAMES)}


if TORCH_AVAILABLE:
    def _zero_scalar(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0


    def _rank_from_scores(scores: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(scores, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(1, int(scores.numel()) + 1, device=scores.device, dtype=order.dtype)
        return ranks


    def _build_local_adjacency(
        mol_atom_indices: torch.Tensor,
        edge_index: Optional[torch.Tensor],
    ) -> list[list[int]]:
        rows = int(mol_atom_indices.numel())
        adjacency: list[list[int]] = [[] for _ in range(rows)]
        if edge_index is None or int(edge_index.numel()) == 0 or rows <= 1:
            return adjacency
        local_lookup = torch.full(
            (int(mol_atom_indices.max().item()) + 1,),
            fill_value=-1,
            dtype=torch.long,
            device=mol_atom_indices.device,
        )
        local_lookup[mol_atom_indices.long()] = torch.arange(rows, device=mol_atom_indices.device, dtype=torch.long)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        valid = (
            (src >= 0)
            & (dst >= 0)
            & (src < int(local_lookup.numel()))
            & (dst < int(local_lookup.numel()))
        )
        if not bool(valid.any()):
            return adjacency
        src_local = local_lookup[src[valid]]
        dst_local = local_lookup[dst[valid]]
        local_valid = (src_local >= 0) & (dst_local >= 0)
        for u, v in zip(src_local[local_valid].tolist(), dst_local[local_valid].tolist()):
            if v not in adjacency[u]:
                adjacency[u].append(v)
        return adjacency


    def _bfs_shortest_paths(adjacency: list[list[int]], start_idx: int, *, device, dtype) -> torch.Tensor:
        rows = len(adjacency)
        distances = torch.full((rows,), float("inf"), device=device, dtype=dtype)
        if start_idx < 0 or start_idx >= rows:
            return distances
        queue = [int(start_idx)]
        distances[start_idx] = 0.0
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            next_distance = float(distances[current].item()) + 1.0
            for neighbor in adjacency[current]:
                if not bool(torch.isfinite(distances[neighbor])):
                    distances[neighbor] = next_distance
                    queue.append(int(neighbor))
        return distances


    def _valid_3d_coordinates(mol_coords: Optional[torch.Tensor]) -> bool:
        if mol_coords is None or int(mol_coords.numel()) == 0 or mol_coords.ndim != 2 or int(mol_coords.size(-1)) != 3:
            return False
        coords = torch.nan_to_num(mol_coords.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if int(coords.size(0)) <= 1:
            return False
        return bool((coords.abs().sum(dim=1) > 1.0e-6).any() and (coords.std(dim=0) > 1.0e-6).any())


    def mine_hard_negative_pairs(
        scores,
        labels,
        batch,
        *,
        supervision_mask=None,
        candidate_mask=None,
        edge_index=None,
        atom_coordinates=None,
        use_top_score: bool = True,
        use_graph_local: bool = True,
        use_3d_local: bool = True,
        max_hard_negs_per_true: int = 3,
    ) -> Dict[str, object]:
        scores_flat = scores.view(-1)
        labels_flat = labels.view(-1)
        batch_flat = batch.view(-1)
        sup_flat = supervision_mask.view(-1) if supervision_mask is not None else None
        cand_flat = candidate_mask.view(-1) if candidate_mask is not None else None
        coords = atom_coordinates if atom_coordinates is not None else None
        num_molecules = int(batch_flat.max().item()) + 1 if batch_flat.numel() else 0

        true_scores = []
        negative_scores = []
        pair_type_indices = []
        true_ranks = []
        negative_ranks_by_type = {name: [] for name in HARD_NEGATIVE_TYPE_NAMES}
        margins_by_type = {name: [] for name in HARD_NEGATIVE_TYPE_NAMES}
        beats_by_type = {name: [] for name in HARD_NEGATIVE_TYPE_NAMES}
        available_pair_counts = {name: 0 for name in HARD_NEGATIVE_TYPE_NAMES}
        available_molecule_counts = {name: 0 for name in HARD_NEGATIVE_TYPE_NAMES}
        recall_hits = {6: 0, 12: 0}
        recall_total = 0

        for mol_idx in range(num_molecules):
            mol_mask = batch_flat == mol_idx
            if not bool(mol_mask.any()):
                continue
            mol_atom_indices = torch.nonzero(mol_mask, as_tuple=False).view(-1)
            mol_scores = scores_flat[mol_mask]
            mol_labels = labels_flat[mol_mask]
            mol_supervised = (
                sup_flat[mol_mask] > 0.5 if sup_flat is not None else torch.ones_like(mol_labels, dtype=torch.bool)
            )
            mol_candidates = (
                cand_flat[mol_mask] > 0.5 if cand_flat is not None else torch.ones_like(mol_labels, dtype=torch.bool)
            )
            eligible_pool = mol_supervised & mol_candidates
            pos_mask = (mol_labels > 0.5) & mol_supervised
            neg_mask = (mol_labels <= 0.5) & eligible_pool
            if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                continue

            recall_total += 1
            candidate_scores = mol_scores[eligible_pool]
            candidate_labels = mol_labels[eligible_pool]
            candidate_ranks = _rank_from_scores(candidate_scores)
            candidate_local_indices = torch.nonzero(eligible_pool, as_tuple=False).view(-1)
            for k in (6, 12):
                topk = min(k, int(candidate_scores.numel()))
                if topk <= 0:
                    continue
                hit = bool((candidate_labels[torch.argsort(candidate_scores, descending=True)[:topk]] > 0.5).any())
                recall_hits[k] += int(hit)

            adjacency = _build_local_adjacency(mol_atom_indices, edge_index)
            mol_coords = coords[mol_mask] if coords is not None and int(coords.numel()) else None
            has_valid_3d = _valid_3d_coordinates(mol_coords)
            per_molecule_available = {name: False for name in HARD_NEGATIVE_TYPE_NAMES}

            positive_local_indices = torch.nonzero(pos_mask, as_tuple=False).view(-1).tolist()
            negative_local_indices = torch.nonzero(neg_mask, as_tuple=False).view(-1)
            negative_scores_pool = mol_scores[neg_mask]
            top_score_local = None
            if use_top_score and int(negative_scores_pool.numel()) > 0:
                top_score_local = int(negative_local_indices[int(torch.argmax(negative_scores_pool).item())].item())

            for true_local in positive_local_indices:
                true_score = mol_scores[true_local]
                eligible_local = candidate_local_indices
                eligible_rank_match = (eligible_local == int(true_local)).nonzero(as_tuple=False).view(-1)
                if int(eligible_rank_match.numel()) > 0:
                    true_ranks.append(candidate_ranks[int(eligible_rank_match[0].item())].to(dtype=torch.float32))
                else:
                    true_rank = 1.0 + float((candidate_scores > true_score).sum().item())
                    true_ranks.append(torch.tensor(true_rank, device=mol_scores.device, dtype=torch.float32))
                chosen_locals: set[int] = set()
                selected: list[tuple[str, int]] = []

                if use_top_score and top_score_local is not None:
                    per_molecule_available["top_score"] = True
                    if int(top_score_local) not in chosen_locals:
                        selected.append(("top_score", int(top_score_local)))
                        chosen_locals.add(int(top_score_local))

                if use_graph_local:
                    graph_distances = _bfs_shortest_paths(
                        adjacency,
                        int(true_local),
                        device=mol_scores.device,
                        dtype=mol_scores.dtype,
                    )
                    neg_graph_dist = graph_distances[negative_local_indices]
                    finite_graph = torch.isfinite(neg_graph_dist)
                    if bool(finite_graph.any()):
                        per_molecule_available["graph_local"] = True
                        finite_neg_local = negative_local_indices[finite_graph]
                        best_graph_local = int(finite_neg_local[int(torch.argmin(neg_graph_dist[finite_graph]).item())].item())
                        if best_graph_local not in chosen_locals:
                            selected.append(("graph_local", best_graph_local))
                            chosen_locals.add(best_graph_local)

                if use_3d_local and has_valid_3d:
                    true_coord = mol_coords[true_local : true_local + 1]
                    neg_coords = mol_coords[negative_local_indices]
                    neg_dist = torch.norm(neg_coords - true_coord, dim=-1)
                    finite_3d = torch.isfinite(neg_dist)
                    if bool(finite_3d.any()):
                        per_molecule_available["3d_local"] = True
                        finite_neg_local = negative_local_indices[finite_3d]
                        best_3d_local = int(finite_neg_local[int(torch.argmin(neg_dist[finite_3d]).item())].item())
                        if best_3d_local not in chosen_locals:
                            selected.append(("3d_local", best_3d_local))
                            chosen_locals.add(best_3d_local)

                if max_hard_negs_per_true > 0 and len(selected) > max_hard_negs_per_true:
                    selected = selected[: max_hard_negs_per_true]

                for type_name, neg_local in selected:
                    negative_score = mol_scores[neg_local]
                    neg_rank_pos = int((eligible_local == int(neg_local)).nonzero(as_tuple=False).view(-1)[0].item())
                    margin = true_score - negative_score
                    true_scores.append(true_score)
                    negative_scores.append(negative_score)
                    pair_type_indices.append(
                        torch.tensor(_HARD_NEGATIVE_TYPE_TO_INDEX[type_name], device=mol_scores.device, dtype=torch.long)
                    )
                    negative_ranks_by_type[type_name].append(candidate_ranks[neg_rank_pos].to(dtype=torch.float32))
                    margins_by_type[type_name].append(margin)
                    beats_by_type[type_name].append((margin > 0.0).to(dtype=torch.float32))
                    available_pair_counts[type_name] += 1

            for type_name, available in per_molecule_available.items():
                available_molecule_counts[type_name] += int(available)

        if true_scores:
            true_scores_tensor = torch.stack(true_scores)
            negative_scores_tensor = torch.stack(negative_scores)
            pair_types_tensor = torch.stack(pair_type_indices)
            true_ranks_tensor = torch.stack(true_ranks) if true_ranks else torch.empty(0, device=scores_flat.device)
        else:
            true_scores_tensor = scores_flat.new_empty((0,))
            negative_scores_tensor = scores_flat.new_empty((0,))
            pair_types_tensor = torch.empty((0,), dtype=torch.long, device=scores_flat.device)
            true_ranks_tensor = scores_flat.new_empty((0,))

        stats: Dict[str, float] = {
            "recall_at_6": float(recall_hits[6]) / float(recall_total) if recall_total > 0 else 0.0,
            "recall_at_12": float(recall_hits[12]) / float(recall_total) if recall_total > 0 else 0.0,
            "true_site_rank_mean": float(torch.stack(true_ranks).mean().item()) if true_ranks else 0.0,
        }
        for type_name in HARD_NEGATIVE_TYPE_NAMES:
            rank_values = negative_ranks_by_type[type_name]
            margin_values = margins_by_type[type_name]
            beat_values = beats_by_type[type_name]
            stats[f"{type_name}_hard_neg_rank_mean"] = (
                float(torch.stack(rank_values).mean().item()) if rank_values else 0.0
            )
            stats[f"{type_name}_margin_mean"] = (
                float(torch.stack(margin_values).mean().item()) if margin_values else 0.0
            )
            stats[f"{type_name}_true_beats_fraction"] = (
                float(torch.stack(beat_values).mean().item()) if beat_values else 0.0
            )
            stats[f"{type_name}_hard_neg_molecule_count"] = float(available_molecule_counts[type_name])
            stats[f"{type_name}_hard_neg_pair_count"] = float(available_pair_counts[type_name])

        return {
            "true_scores": true_scores_tensor,
            "negative_scores": negative_scores_tensor,
            "pair_type_indices": pair_types_tensor,
            "stats": stats,
        }


    def compute_hard_negative_margin_loss(
        scores,
        labels,
        batch,
        *,
        margin: float,
        supervision_mask=None,
        candidate_mask=None,
        edge_index=None,
        atom_coordinates=None,
        use_top_score: bool = True,
        use_graph_local: bool = True,
        use_3d_local: bool = True,
        max_hard_negs_per_true: int = 3,
    ):
        mined = mine_hard_negative_pairs(
            scores,
            labels,
            batch,
            supervision_mask=supervision_mask,
            candidate_mask=candidate_mask,
            edge_index=edge_index,
            atom_coordinates=atom_coordinates,
            use_top_score=use_top_score,
            use_graph_local=use_graph_local,
            use_3d_local=use_3d_local,
            max_hard_negs_per_true=max_hard_negs_per_true,
        )
        true_scores = mined["true_scores"]
        negative_scores = mined["negative_scores"]
        stats = dict(mined["stats"])
        if int(true_scores.numel()) == 0:
            zero = _zero_scalar(scores.view(-1))
            stats["hard_negative_loss"] = 0.0
            stats["hard_negative_active_fraction"] = 0.0
            stats["hard_negative_pair_count"] = 0.0
            return zero, stats
        margin_terms = torch.relu(float(margin) - (true_scores - negative_scores))
        stats["hard_negative_loss"] = float(margin_terms.mean().detach().item())
        stats["hard_negative_active_fraction"] = float((margin_terms > 0.0).float().mean().detach().item())
        stats["hard_negative_pair_count"] = float(int(margin_terms.numel()))
        return margin_terms.mean(), stats
else:  # pragma: no cover
    def mine_hard_negative_pairs(*args, **kwargs):
        require_torch()


    def compute_hard_negative_margin_loss(*args, **kwargs):
        require_torch()
