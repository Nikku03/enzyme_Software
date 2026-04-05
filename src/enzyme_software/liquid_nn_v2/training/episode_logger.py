from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, torch


def _to_serializable(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if TORCH_AVAILABLE and hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return float(value)
    except Exception:
        return str(value)


class EpisodeLogger:
    def __init__(self, path: str | Path, *, run_id: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def _write(self, record: Dict[str, object]) -> None:
        payload = dict(record)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def log_step(
        self,
        *,
        split: str,
        epoch: Optional[int],
        batch_idx: int,
        batch,
        stats: Optional[Dict[str, float]] = None,
        outputs: Optional[Dict[str, object]] = None,
    ) -> None:
        record = {
            "record_type": "step",
            "split": str(split),
            "epoch": None if epoch is None else int(epoch) + 1,
            "batch_idx": int(batch_idx),
            "num_graphs": int(batch.get("num_graphs", 0)),
            "graph_names": list(batch.get("graph_names", [])),
            "graph_smiles": list(batch.get("canonical_smiles", [])),
            "stats": _to_serializable(stats or {}),
        }
        if outputs is not None:
            diagnostics = outputs.get("diagnostics") or {}
            record["diagnostics"] = _to_serializable(diagnostics)
        self._write(record)

    def log_examples(
        self,
        *,
        split: str,
        epoch: Optional[int],
        batch_idx: int,
        batch,
        outputs: Dict[str, object],
        stats: Optional[Dict[str, float]] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            return
        graph_num_atoms = [int(v) for v in list(batch.get("graph_num_atoms", []))]
        if not graph_num_atoms:
            return
        batch_metadata = list(batch.get("graph_metadata", []))
        graph_names = list(batch.get("graph_names", []))
        graph_confidences = list(batch.get("graph_confidences", []))
        canonical_smiles = list(batch.get("canonical_smiles", []))
        parsing_status = list(batch.get("parsing_status", []))
        repaired = list(batch.get("repaired", []))
        aggressive_repair = list(batch.get("aggressive_repair", []))
        xtb_status = list(batch.get("xtb_feature_status", []))

        site_logits = outputs["site_logits"].detach().cpu().view(-1)
        site_logits_base = outputs.get("site_logits_base")
        site_logits_base_cpu = site_logits_base.detach().cpu().view(-1) if site_logits_base is not None else None
        site_logits_proposal = outputs.get("site_logits_proposal")
        site_logits_proposal_cpu = site_logits_proposal.detach().cpu().view(-1) if site_logits_proposal is not None else None
        site_scores = outputs.get("site_scores")
        if site_scores is None:
            site_scores = torch.sigmoid(outputs["site_logits"])
        site_scores = site_scores.detach().cpu().view(-1)
        site_scores_proposal = outputs.get("site_scores_proposal")
        if site_scores_proposal is None and site_logits_proposal is not None:
            site_scores_proposal = torch.sigmoid(site_logits_proposal)
        site_scores_proposal_cpu = site_scores_proposal.detach().cpu().view(-1) if site_scores_proposal is not None else None
        site_labels = batch.get("site_labels")
        site_labels_cpu = site_labels.detach().cpu().view(-1) if site_labels is not None else None
        site_mask = batch.get("site_supervision_mask")
        site_mask_cpu = (
            site_mask.detach().cpu().view(-1)
            if site_mask is not None
            else torch.ones_like(site_scores, dtype=torch.float32)
        )
        candidate_mask = batch.get("candidate_mask")
        candidate_mask_cpu = (
            candidate_mask.detach().cpu().view(-1)
            if candidate_mask is not None
            else torch.ones_like(site_scores, dtype=torch.float32)
        )
        cyp_logits = outputs.get("cyp_logits")
        cyp_logits_cpu = cyp_logits.detach().cpu() if cyp_logits is not None else None
        cyp_probs_cpu = torch.softmax(cyp_logits_cpu, dim=-1) if cyp_logits_cpu is not None else None
        cyp_labels = batch.get("cyp_labels")
        cyp_labels_cpu = cyp_labels.detach().cpu() if cyp_labels is not None else None
        cyp_mask = batch.get("cyp_supervision_mask")
        cyp_mask_cpu = cyp_mask.detach().cpu() if cyp_mask is not None else None

        vote_heads = outputs.get("site_vote_heads") or {}
        vote_heads_cpu = {
            key: value.detach().cpu()
            for key, value in vote_heads.items()
            if value is not None
        }

        bridge = outputs.get("nexus_bridge_outputs") or {}
        atom_multivectors = bridge.get("atom_multivectors")
        atom_multivectors_cpu = atom_multivectors.detach().cpu() if atom_multivectors is not None else None
        wave_predictions = bridge.get("wave_predictions") or {}
        wave_predictions_cpu = {
            key: value.detach().cpu()
            for key, value in wave_predictions.items()
            if value is not None
        }
        wave_field = bridge.get("wave_field") or {}
        wave_field_cpu = {
            key: value.detach().cpu()
            for key, value in wave_field.items()
            if value is not None
        }
        analogical_site_prior = bridge.get("analogical_site_prior")
        analogical_site_prior_cpu = analogical_site_prior.detach().cpu() if analogical_site_prior is not None else None
        analogical_cyp_prior = bridge.get("analogical_cyp_prior")
        analogical_cyp_prior_cpu = analogical_cyp_prior.detach().cpu() if analogical_cyp_prior is not None else None
        analogical_confidence = bridge.get("analogical_confidence")
        analogical_confidence_cpu = analogical_confidence.detach().cpu() if analogical_confidence is not None else None
        analogical_gate = bridge.get("analogical_gate")
        analogical_gate_cpu = analogical_gate.detach().cpu() if analogical_gate is not None else None
        analogical_selectivity = bridge.get("analogical_selectivity")
        analogical_selectivity_cpu = analogical_selectivity.detach().cpu() if analogical_selectivity is not None else None
        analogical_margin = bridge.get("analogical_margin")
        analogical_margin_cpu = analogical_margin.detach().cpu() if analogical_margin is not None else None
        analogical_site_bias = bridge.get("analogical_site_bias")
        analogical_site_bias_cpu = analogical_site_bias.detach().cpu() if analogical_site_bias is not None else None
        analogical_cyp_bias = bridge.get("analogical_cyp_bias")
        analogical_cyp_bias_cpu = analogical_cyp_bias.detach().cpu() if analogical_cyp_bias is not None else None
        continuous_reasoning = bridge.get("continuous_reasoning_features")
        continuous_reasoning_cpu = continuous_reasoning.detach().cpu() if continuous_reasoning is not None else None
        precedent_brief = bridge.get("precedent_brief")
        precedent_brief_cpu = precedent_brief.detach().cpu() if precedent_brief is not None else None
        wave_reliability = bridge.get("wave_reliability")
        wave_reliability_cpu = wave_reliability.detach().cpu() if wave_reliability is not None else None
        bridge_metrics = _to_serializable(bridge.get("metrics") or {})
        local_chem = batch.get("local_chem_features")
        local_chem_cpu = local_chem.detach().cpu() if local_chem is not None else None
        local_charge = batch.get("local_charge_updated")
        local_charge_cpu = local_charge.detach().cpu() if local_charge is not None else None
        local_charge_delta = batch.get("local_charge_delta")
        local_charge_delta_cpu = local_charge_delta.detach().cpu() if local_charge_delta is not None else None
        local_etn = batch.get("local_etn_prior")
        local_etn_cpu = local_etn.detach().cpu() if local_etn is not None else None
        local_etn_features = batch.get("local_etn_features")
        local_etn_features_cpu = local_etn_features.detach().cpu() if local_etn_features is not None else None
        anomaly_score = batch.get("local_anomaly_score")
        anomaly_score_cpu = anomaly_score.detach().cpu() if anomaly_score is not None else None
        anomaly_score_norm = batch.get("local_anomaly_score_normalized")
        anomaly_score_norm_cpu = anomaly_score_norm.detach().cpu() if anomaly_score_norm is not None else None
        anomaly_flag = batch.get("local_anomaly_flag")
        anomaly_flag_cpu = anomaly_flag.detach().cpu() if anomaly_flag is not None else None
        phase2 = outputs.get("phase2_context_outputs") or {}
        phase2_event_strain = phase2.get("event_strain")
        phase2_event_strain_cpu = phase2_event_strain.detach().cpu() if phase2_event_strain is not None else None
        phase2_event_neighbors = phase2.get("event_active_neighbor_count")
        phase2_event_neighbors_cpu = phase2_event_neighbors.detach().cpu() if phase2_event_neighbors is not None else None
        phase2_event_depth = phase2.get("event_depth")
        phase2_event_depth_cpu = phase2_event_depth.detach().cpu() if phase2_event_depth is not None else None
        phase2_access_score = phase2.get("access_score")
        phase2_access_score_cpu = phase2_access_score.detach().cpu() if phase2_access_score is not None else None
        phase2_access_cost = phase2.get("access_cost")
        phase2_access_cost_cpu = phase2_access_cost.detach().cpu() if phase2_access_cost is not None else None
        phase2_barrier = phase2.get("barrier_score")
        phase2_barrier_cpu = phase2_barrier.detach().cpu() if phase2_barrier is not None else None
        topk_reranker = outputs.get("topk_reranker_outputs") or {}
        reranker_selected = topk_reranker.get("selected_mask")
        reranker_selected_cpu = reranker_selected.detach().cpu().view(-1) if reranker_selected is not None else None
        reranker_raw_delta = topk_reranker.get("raw_delta")
        reranker_raw_delta_cpu = reranker_raw_delta.detach().cpu().view(-1) if reranker_raw_delta is not None else None
        reranker_gate = topk_reranker.get("gate")
        reranker_gate_cpu = reranker_gate.detach().cpu().view(-1) if reranker_gate is not None else None
        reranker_applied_delta = topk_reranker.get("applied_delta")
        reranker_applied_delta_cpu = reranker_applied_delta.detach().cpu().view(-1) if reranker_applied_delta is not None else None

        def _slice_vote_head(name: str, *, start_idx: int, end_idx: int):
            value = vote_heads_cpu.get(name)
            if value is None:
                return None
            if hasattr(value, "ndim") and int(value.ndim) <= 1:
                return _to_serializable(value[start_idx:end_idx].view(-1))
            return _to_serializable(value[start_idx:end_idx])

        offset = 0
        for graph_idx, num_atoms in enumerate(graph_num_atoms):
            start = offset
            end = offset + num_atoms
            offset = end

            meta = dict(batch_metadata[graph_idx]) if graph_idx < len(batch_metadata) else {}
            if "site_atoms" not in meta:
                meta["site_atoms"] = []
            predicted_order = torch.argsort(site_scores[start:end], descending=True)
            top5_count = min(5, int(num_atoms))
            top5 = [int(v) for v in predicted_order[:top5_count].tolist()]
            top3 = top5[: min(3, len(top5))]
            top1 = top5[0] if top5 else None
            proposal_order = (
                torch.argsort(site_scores_proposal_cpu[start:end], descending=True)
                if site_scores_proposal_cpu is not None
                else predicted_order
            )
            proposal_top5 = [int(v) for v in proposal_order[:top5_count].tolist()]
            proposal_top3 = proposal_top5[: min(3, len(proposal_top5))]
            proposal_top1 = proposal_top5[0] if proposal_top5 else None
            true_site_atoms = list(meta.get("site_atoms", []))
            if site_labels_cpu is not None:
                true_site_atoms = [
                    int(local_idx)
                    for local_idx in range(num_atoms)
                    if float(site_mask_cpu[start + local_idx].item()) > 0.0
                    and float(site_labels_cpu[start + local_idx].item()) > 0.5
                ]
            true_site_set = set(true_site_atoms)
            top1_hit = bool(top1 in true_site_set) if top1 is not None else False
            top3_hit = bool(true_site_set.intersection(top3))
            top5_hit = bool(true_site_set.intersection(top5))
            proposal_top1_hit = bool(proposal_top1 in true_site_set) if proposal_top1 is not None else False

            record = {
                "record_type": "episode",
                "split": str(split),
                "epoch": None if epoch is None else int(epoch) + 1,
                "batch_idx": int(batch_idx),
                "graph_idx": int(graph_idx),
                "input": {
                    "id": meta.get("id", ""),
                    "name": meta.get("name", graph_names[graph_idx] if graph_idx < len(graph_names) else ""),
                    "smiles": meta.get(
                        "smiles",
                        canonical_smiles[graph_idx] if graph_idx < len(canonical_smiles) else "",
                    ),
                    "canonical_smiles": canonical_smiles[graph_idx] if graph_idx < len(canonical_smiles) else "",
                    "source": meta.get("source", ""),
                    "confidence": meta.get(
                        "confidence",
                        graph_confidences[graph_idx] if graph_idx < len(graph_confidences) else "",
                    ),
                    "primary_cyp": meta.get("primary_cyp", ""),
                    "all_cyps": meta.get("all_cyps", []),
                    "site_atoms": true_site_atoms,
                    "site_source": meta.get("site_source", ""),
                    "auxiliary_site_only": bool(meta.get("auxiliary_site_only", False)),
                    "num_atoms": int(num_atoms),
                    "parsing_status": parsing_status[graph_idx] if graph_idx < len(parsing_status) else "",
                    "repaired": bool(repaired[graph_idx]) if graph_idx < len(repaired) else False,
                    "aggressive_repair": bool(aggressive_repair[graph_idx]) if graph_idx < len(aggressive_repair) else False,
                    "xtb_feature_status": xtb_status[graph_idx] if graph_idx < len(xtb_status) else "",
                    "site_supervision": bool(true_site_atoms),
                    "cyp_supervision": bool(
                        cyp_mask_cpu is None or float(cyp_mask_cpu[graph_idx].item()) > 0.0
                    ),
                },
                "output": {
                    "site_logits": _to_serializable(site_logits[start:end]),
                    "site_logits_base": _to_serializable(site_logits_base_cpu[start:end]) if site_logits_base_cpu is not None else None,
                    "site_logits_proposal": _to_serializable(site_logits_proposal_cpu[start:end]) if site_logits_proposal_cpu is not None else None,
                    "site_scores": _to_serializable(site_scores[start:end]),
                    "site_scores_proposal": _to_serializable(site_scores_proposal_cpu[start:end]) if site_scores_proposal_cpu is not None else None,
                    "candidate_mask": _to_serializable(candidate_mask_cpu[start:end]),
                    "cyp_logits": _to_serializable(cyp_logits_cpu[graph_idx]) if cyp_logits_cpu is not None else None,
                    "cyp_probs": _to_serializable(cyp_probs_cpu[graph_idx]) if cyp_probs_cpu is not None else None,
                    "true_cyp_label": int(cyp_labels_cpu[graph_idx].item()) if cyp_labels_cpu is not None else None,
                },
                "votes": {
                    "lnn_vote": _slice_vote_head("lnn_vote", start_idx=start, end_idx=end) or _to_serializable(site_logits[start:end]),
                    "lnn_conf": _slice_vote_head("lnn_conf", start_idx=start, end_idx=end),
                    "wave_vote": _slice_vote_head("wave_vote", start_idx=start, end_idx=end),
                    "wave_conf": _slice_vote_head("wave_conf", start_idx=start, end_idx=end),
                    "analogical_vote": _slice_vote_head("analogical_vote", start_idx=start, end_idx=end),
                    "analogical_conf": _slice_vote_head("analogical_conf", start_idx=start, end_idx=end),
                    "council_logit": _slice_vote_head("council_logit", start_idx=start, end_idx=end),
                    "arbiter_residual": _slice_vote_head("arbiter_residual", start_idx=start, end_idx=end),
                    "board_weights": _slice_vote_head("board_weights", start_idx=start, end_idx=end),
                },
                "wave": {
                    "atom_multivectors": _to_serializable(atom_multivectors_cpu[start:end]) if atom_multivectors_cpu is not None else None,
                    "predicted_charges": _to_serializable(wave_predictions_cpu.get("predicted_charges")[start:end]) if "predicted_charges" in wave_predictions_cpu else None,
                    "predicted_fukui": _to_serializable(wave_predictions_cpu.get("predicted_fukui")[start:end]) if "predicted_fukui" in wave_predictions_cpu else None,
                    "predicted_gap": _to_serializable(wave_predictions_cpu.get("predicted_gap")[graph_idx]) if "predicted_gap" in wave_predictions_cpu else None,
                    "atom_field_features": _to_serializable(wave_field_cpu.get("atom_field_features")[start:end]) if "atom_field_features" in wave_field_cpu else None,
                    "global_density": _to_serializable(wave_field_cpu.get("global_density")[graph_idx]) if "global_density" in wave_field_cpu else None,
                    "global_gap_proxy": _to_serializable(wave_field_cpu.get("global_gap_proxy")[graph_idx]) if "global_gap_proxy" in wave_field_cpu else None,
                    "reliability": _to_serializable(wave_reliability_cpu[start:end]) if wave_reliability_cpu is not None else None,
                },
                "analogical": {
                    "site_prior": _to_serializable(analogical_site_prior_cpu[start:end]) if analogical_site_prior_cpu is not None else None,
                    "site_bias": _to_serializable(analogical_site_bias_cpu[start:end]) if analogical_site_bias_cpu is not None else None,
                    "confidence": _to_serializable(analogical_confidence_cpu[start:end]) if analogical_confidence_cpu is not None else None,
                    "gate": _to_serializable(analogical_gate_cpu[start:end]) if analogical_gate_cpu is not None else None,
                    "selectivity": _to_serializable(analogical_selectivity_cpu[start:end]) if analogical_selectivity_cpu is not None else None,
                    "margin": _to_serializable(analogical_margin_cpu[start:end]) if analogical_margin_cpu is not None else None,
                    "continuous_reasoning_features": _to_serializable(continuous_reasoning_cpu[start:end]) if continuous_reasoning_cpu is not None else None,
                    "precedent_brief": _to_serializable(precedent_brief_cpu[start:end]) if precedent_brief_cpu is not None else None,
                    "cyp_prior": _to_serializable(analogical_cyp_prior_cpu[graph_idx]) if analogical_cyp_prior_cpu is not None else None,
                    "cyp_bias": _to_serializable(analogical_cyp_bias_cpu[graph_idx]) if analogical_cyp_bias_cpu is not None else None,
                    "bridge_metrics": bridge_metrics,
                },
                "chemistry": {
                    "local_chem_features": _to_serializable(local_chem_cpu[start:end]) if local_chem_cpu is not None else None,
                    "updated_charge": _to_serializable(local_charge_cpu[start:end]) if local_charge_cpu is not None else None,
                    "charge_delta": _to_serializable(local_charge_delta_cpu[start:end]) if local_charge_delta_cpu is not None else None,
                    "etn_prior": _to_serializable(local_etn_cpu[start:end]) if local_etn_cpu is not None else None,
                    "etn_features": _to_serializable(local_etn_features_cpu[start:end]) if local_etn_features_cpu is not None else None,
                    "anomaly_score": _to_serializable(anomaly_score_cpu[graph_idx]) if anomaly_score_cpu is not None else None,
                    "anomaly_score_normalized": _to_serializable(anomaly_score_norm_cpu[graph_idx]) if anomaly_score_norm_cpu is not None else None,
                    "anomaly_flag": _to_serializable(anomaly_flag_cpu[graph_idx]) if anomaly_flag_cpu is not None else None,
                },
                "phase2": {
                    "event_strain": _to_serializable(phase2_event_strain_cpu[start:end]) if phase2_event_strain_cpu is not None else None,
                    "event_active_neighbor_count": _to_serializable(phase2_event_neighbors_cpu[start:end]) if phase2_event_neighbors_cpu is not None else None,
                    "event_depth": _to_serializable(phase2_event_depth_cpu[start:end]) if phase2_event_depth_cpu is not None else None,
                    "access_score": _to_serializable(phase2_access_score_cpu[start:end]) if phase2_access_score_cpu is not None else None,
                    "access_cost": _to_serializable(phase2_access_cost_cpu[start:end]) if phase2_access_cost_cpu is not None else None,
                    "barrier_score": _to_serializable(phase2_barrier_cpu[start:end]) if phase2_barrier_cpu is not None else None,
                },
                "reranker": {
                    "selected_mask": _to_serializable(reranker_selected_cpu[start:end]) if reranker_selected_cpu is not None else None,
                    "raw_delta": _to_serializable(reranker_raw_delta_cpu[start:end]) if reranker_raw_delta_cpu is not None else None,
                    "gate": _to_serializable(reranker_gate_cpu[start:end]) if reranker_gate_cpu is not None else None,
                    "applied_delta": _to_serializable(reranker_applied_delta_cpu[start:end]) if reranker_applied_delta_cpu is not None else None,
                    "selected_atom_indices": [
                        int(local_idx)
                        for local_idx in range(num_atoms)
                        if reranker_selected_cpu is not None and float(reranker_selected_cpu[start + local_idx].item()) > 0.5
                    ],
                    "proposal_top1_atom": proposal_top1,
                    "proposal_top3_atoms": proposal_top3,
                    "proposal_top5_atoms": proposal_top5,
                    "proposal_top1_hit": proposal_top1_hit,
                    "corrected_top1": bool((not proposal_top1_hit) and top1_hit),
                    "harmed_top1": bool(proposal_top1_hit and (not top1_hit)),
                },
                "decision": {
                    "top1_atom": top1,
                    "top3_atoms": top3,
                    "top5_atoms": top5,
                    "top1_score": float(site_scores[start + top1].item()) if top1 is not None else None,
                    "candidate_atoms": int(candidate_mask_cpu[start:end].sum().item()),
                    "candidate_atom_indices": [
                        int(local_idx)
                        for local_idx in range(num_atoms)
                        if float(candidate_mask_cpu[start + local_idx].item()) > 0.5
                    ],
                    "candidate_fraction": float(candidate_mask_cpu[start:end].mean().item()),
                    "predicted_cyp_idx": int(torch.argmax(cyp_logits_cpu[graph_idx]).item()) if cyp_logits_cpu is not None else None,
                    "predicted_cyp_prob": float(torch.max(cyp_probs_cpu[graph_idx]).item()) if cyp_probs_cpu is not None else None,
                },
                "outcome": {
                    "true_site_atoms": true_site_atoms,
                    "top1_hit": top1_hit,
                    "top3_hit": top3_hit,
                    "top5_hit": top5_hit,
                    "positive_site_count": int(len(true_site_atoms)),
                },
                "step_stats": _to_serializable(stats or {}),
            }
            self._write(record)
