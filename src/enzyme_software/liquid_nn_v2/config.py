from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES


@dataclass
class ModelConfig:
    model_variant: str = "baseline"
    atom_input_dim: int = 148  # 140 base + 8 full-xTB features on the hybrid full-xTB path
    hidden_dim: int = 128
    shared_hidden_dim: Optional[int] = None
    som_branch_dim: Optional[int] = None
    cyp_branch_dim: Optional[int] = None
    physics_dim: int = 32
    mol_dim: int = 128
    num_liquid_layers: int = 2
    split_after_layer: int = 1
    shared_encoder_layers: Optional[int] = None
    som_branch_layers: int = 1
    cyp_branch_layers: int = 1
    ode_steps: int = 6
    edge_feature_dim: int = 10
    group_pooling_hidden_dim: Optional[int] = None
    use_manual_engine_priors: bool = True
    manual_atom_feature_dim: int = 8
    manual_mol_feature_dim: int = 8
    manual_prior_fusion_mode: str = "gated_add"
    manual_prior_init_scale: float = 0.65
    use_3d_branch: bool = True
    steric_feature_dim: int = 8
    steric_hidden_dim: int = 32
    use_contextual_tau: bool = True
    tau_min: float = 0.1
    tau_max: float = 1.5
    tau_context_dim: Optional[int] = None
    use_hierarchical_pooling: bool = True
    return_intermediate_stats: bool = True
    use_energy_module: bool = False
    use_tunneling_module: bool = False
    use_graph_tunneling: bool = False
    use_phase_augmented_state: bool = False
    use_higher_order_coupling: bool = False
    use_physics_residual: bool = False
    use_deliberation_loop: bool = False
    use_energy_dynamics: bool = False
    use_tunneling_for_site_scores: bool = True
    use_tunneling_for_messages: bool = True
    use_local_tunneling_bias: bool = False
    local_tunneling_scale: float = 0.1
    local_tunneling_clamp: float = 5.0
    use_output_refinement: bool = False
    output_refinement_scale: float = 0.1
    output_refinement_hidden_dim: int = 64
    energy_hidden_dim: int = 32
    tunneling_hidden_dim: int = 32
    phase_hidden_dim: int = 16
    phase_scale: float = 0.25
    graph_tunneling_dim: int = 30
    higher_order_hidden_dim: int = 32
    higher_order_topk: int = 8
    higher_order_heads: int = 2
    physics_residual_hidden_dim: int = 32
    num_deliberation_steps: int = 0
    deliberation_hidden_dim: int = 24
    max_tunneling_edges_per_node: int = 4
    max_tunneling_path_length: int = 4
    tunneling_alpha_init: float = 1.0
    tunneling_barrier_min: float = 0.0
    tunneling_barrier_max: float = 12.0
    tunneling_probability_floor: float = 1.0e-4
    tunnel_residual_scale: float = 0.05
    tunnel_residual_scale_max: float = 0.25
    energy_loss_weight: float = 0.0
    energy_loss_clip: float = 2.0
    tau_prior_weight: float = 0.01
    deliberation_loss_weight: float = 0.0
    energy_margin: float = 0.15
    energy_value_clip: float = 6.0
    deliberation_step_scale: float = 0.1
    deliberation_max_state_norm: float = 10.0
    enable_finite_checks: bool = True
    instability_hidden_norm_warn: float = 25.0
    instability_energy_warn: float = 8.0
    tau_blend_mode: str = "learnable"
    tau_prior_blend: float = 0.5
    som_competition_heads: int = 1
    som_head_hidden_dim: int = 96
    cyp_head_hidden_dim: int = 128
    dropout: float = 0.1
    # Architecture improvements
    use_cyp_site_conditioning: bool = True   # FiLM-style per-atom modulation from CYP posterior before site head
    cyp_site_condition_scale: float = 0.20
    use_cross_atom_attention: bool = True    # 2-layer self-attention on SoM atom features before site head
    use_bde_prior: bool = True               # learnable BDE→logit residual on top of site head
    bde_feature_index: int = 44             # deprecated; BDE prior now uses named physics_features["bde_values"]
    site_logit_bias_warmup_epochs: int = 8
    site_logit_bias_target: float = -0.10
    site_logit_bias_weight: float = 0.05
    site_ranking_weight: float = 0.5
    site_hard_negative_fraction: float = 0.5
    site_top1_margin_topk: int = 1
    site_top1_margin_decay: float = 1.0
    site_cover_weight: float = 0.0
    site_cover_margin: float = 0.20
    site_cover_topk: int = 5
    site_shortlist_weight: float = 0.0
    site_shortlist_temperature: float = 0.70
    site_shortlist_topk: int = 5
    site_use_rank_weighted_shortlist: bool = False
    site_hard_negative_weight: float = 0.0
    site_hard_negative_margin: float = 0.20
    site_hard_negative_max_per_true: int = 3
    site_use_top_score_hard_neg: bool = True
    site_use_graph_local_hard_neg: bool = True
    site_use_3d_local_hard_neg: bool = True
    site_use_rank_weighted_hard_neg: bool = False
    site_source_weight_default: float = 1.0
    site_source_weight_drugbank: float = 1.0
    site_source_weight_az120: float = 1.0
    site_source_weight_metxbiodb: float = 1.0
    site_source_weight_attnsom: float = 1.0
    site_source_weight_cyp_dbs_external: float = 1.0
    enable_pairwise_probe: bool = False
    pairwise_probe_dropout: float = 0.1
    pairwise_probe_hidden_scale: float = 2.0
    pairwise_probe_max_pairs_per_batch: Optional[int] = None
    pairwise_probe_freeze_backbone: bool = True
    pairwise_probe_freeze_proposer: bool = True
    pairwise_probe_log_every_epoch: bool = True
    enable_pairwise_aux: bool = False
    pairwise_aux_weight: float = 0.1
    pairwise_aux_unfreeze_proposer_head: bool = True
    pairwise_aux_unfreeze_last_backbone_block: bool = False
    pairwise_aux_recompute_hard_neg_online: bool = True
    pairwise_aux_log_every_epoch: bool = True
    pairwise_aux_backbone_lr_scale: float = 0.1
    pairwise_aux_proposer_lr_scale: float = 0.1
    enable_pairwise_distilled_proposer: bool = False
    distilled_proposer_use_frozen_backbone: bool = True
    distilled_proposer_use_frozen_pairwise_head: bool = True
    distilled_proposer_candidate_topk: int = 6
    distilled_proposer_target_temperature: float = 1.0
    distilled_proposer_head_hidden_dim: Optional[int] = None
    distilled_proposer_dropout: float = 0.1
    distilled_proposer_loss_type: str = "kl"
    distilled_proposer_label_smoothing: float = 0.0
    distilled_proposer_lr_scale: float = 1.0
    distilled_proposer_backbone_lr_scale: float = 0.1
    distilled_proposer_restrict_to_candidates: bool = True
    distilled_proposer_log_every_epoch: bool = True
    distilled_proposer_trainable_proposer_head_only: bool = True
    distilled_proposer_unfreeze_last_backbone_block: bool = False
    distilled_proposer_pairwise_teacher_checkpoint_path: str = ""
    enable_pairwise_distilled_proposer_supervised: bool = False
    distilled_proposer_supervised_weight: float = 1.0
    distilled_proposer_distill_weight: float = 0.1
    distilled_proposer_use_main_site_loss_impl: bool = True
    enable_pairwise_distilled_proposer_unfreeze: bool = False
    distilled_proposer_unfreeze_proposer_head: bool = True
    distilled_proposer_student_lr_scale: float = 1.0
    distilled_proposer_unfrozen_head_lr_scale: float = 0.1
    distilled_proposer_unfrozen_backbone_lr_scale: float = 0.05
    enable_two_head_shortlist_winner: bool = False
    shortlist_topk: int = 6
    shortlist_head_hidden_dim: Optional[int] = None
    shortlist_head_dropout: float = 0.1
    winner_head_hidden_dim: Optional[int] = None
    winner_head_dropout: float = 0.1
    shortlist_loss_weight: float = 1.0
    winner_loss_weight: float = 1.0
    train_winner_only_on_hits: bool = True
    shortlist_use_existing_site_loss: bool = True
    shortlist_selection_metric: str = "recall_at_6"
    two_head_log_every_epoch: bool = True
    enable_two_head_shortlist_winner_v2: bool = False
    frozen_shortlist_checkpoint_path: str = ""
    frozen_shortlist_topk: int = 6
    winner_v2_hidden_dim: Optional[int] = None
    winner_v2_dropout: float = 0.1
    winner_v2_use_existing_candidate_features: bool = True
    winner_v2_use_score_gap_features: bool = True
    winner_v2_use_rank_features: bool = True
    winner_v2_use_pairwise_features: bool = True
    winner_v2_use_graph_local_features: bool = True
    winner_v2_use_3d_local_features: bool = True
    winner_v2_train_only_on_hits: bool = True
    winner_v2_loss_weight: float = 1.0
    shortlist_v2_log_every_epoch: bool = True
    enable_two_head_shortlist_winner_v2_1: bool = False
    winner_v2_1_hidden_dim: Optional[int] = None
    winner_v2_1_dropout: float = 0.1
    winner_v2_1_use_existing_candidate_features: bool = True
    winner_v2_1_use_score_gap_features: bool = True
    winner_v2_1_use_rank_features: bool = True
    winner_v2_1_use_pairwise_features: bool = True
    winner_v2_1_use_graph_local_features: bool = True
    winner_v2_1_use_3d_local_features: bool = True
    winner_v2_1_use_top2_gap_features: bool = True
    winner_v2_1_use_normalized_score_features: bool = True
    winner_v2_1_use_shortlist_context_features: bool = True
    winner_v2_1_use_soft_multi_positive_targets: bool = True
    winner_v2_1_train_only_on_hits: bool = True
    winner_v2_1_loss_weight: float = 1.0
    shortlist_v2_1_log_every_epoch: bool = True
    enable_two_head_shortlist_winner_v2_2: bool = False
    winner_v2_2_hidden_dim: Optional[int] = None
    winner_v2_2_dropout: float = 0.1
    winner_v2_2_use_existing_candidate_features: bool = True
    winner_v2_2_use_score_gap_features: bool = True
    winner_v2_2_use_rank_features: bool = True
    winner_v2_2_use_normalized_score_features: bool = True
    winner_v2_2_use_pairwise_features: bool = False
    winner_v2_2_use_graph_local_features: bool = False
    winner_v2_2_use_3d_local_features: bool = False
    winner_v2_2_use_extra_candidate_features: bool = False
    winner_v2_2_use_soft_multi_positive_targets: bool = False
    winner_v2_2_train_only_on_hits: bool = True
    winner_v2_2_loss_weight: float = 1.0
    winner_v2_2_use_source_weighting: bool = True
    winner_v2_2_hard_source_weight: float = 2.0
    winner_v2_2_normal_source_weight: float = 1.0
    winner_v2_2_hard_sources: str = "attnsom,cyp_dbs_external"
    winner_v2_2_log_source_weight_stats: bool = True
    shortlist_v2_2_log_every_epoch: bool = True
    enable_two_head_shortlist_winner_v2_3: bool = False
    winner_v2_3_hidden_dim: Optional[int] = None
    winner_v2_3_dropout: float = 0.1
    winner_v2_3_use_existing_candidate_features: bool = True
    winner_v2_3_use_score_gap_features: bool = True
    winner_v2_3_use_rank_features: bool = True
    winner_v2_3_use_normalized_score_features: bool = True
    winner_v2_3_use_pairwise_features: bool = False
    winner_v2_3_use_graph_local_features: bool = False
    winner_v2_3_use_3d_local_features: bool = False
    winner_v2_3_use_extra_candidate_features: bool = False
    winner_v2_3_use_soft_multi_positive_targets: bool = False
    winner_v2_3_use_source_weighting: bool = False
    winner_v2_3_use_source_oversampling: bool = False
    winner_v2_3_train_only_on_hits: bool = True
    winner_v2_3_loss_weight: float = 1.0
    winner_v2_3_hard_source_weight: float = 2.0
    winner_v2_3_normal_source_weight: float = 1.0
    winner_v2_3_hard_sources: str = "attnsom,cyp_dbs_external"
    winner_v2_3_log_feature_summary: bool = True
    enable_two_head_shortlist_winner_v2_rebuild: bool = False
    enable_two_head_shortlist_winner_v2_rebuild_top12: bool = False
    winner_v2_rebuild_hidden_dim: Optional[int] = None
    winner_v2_rebuild_dropout: float = 0.1
    winner_v2_rebuild_loss_weight: float = 1.0
    winner_v2_rebuild_log_restore_summary: bool = True
    two_head_shortlist_eval_topk: int = 6
    two_head_shortlist_winner_topk: int = 6
    two_head_keep_aux_metrics_at_6: bool = True
    two_head_log_dual_k_metrics: bool = True
    enable_two_head_shortlist_winner_v2_rebuild_hard_source_finetune: bool = False
    hard_source_names: str = "attnsom,cyp_dbs_external"
    hard_source_finetune_require_hit: bool = True
    hard_source_finetune_skip_non_hard_sources: bool = True
    winner_finetune_init_checkpoint_path: str = ""
    hard_source_finetune_lr_scale: float = 0.5
    enable_two_head_shortlist_winner_v2_rebuild_boundary_reranker: bool = False
    boundary_reranker_shortlist_k: int = 12
    boundary_reranker_output_k: int = 6
    boundary_reranker_train_on_rescued_only: bool = True
    boundary_reranker_train_on_hits_only: bool = True
    boundary_reranker_use_pairwise_mode: bool = False
    boundary_reranker_use_listwise_mode: bool = True
    boundary_reranker_hidden_dim: Optional[int] = None
    boundary_reranker_dropout: float = 0.1
    boundary_reranker_loss_weight: float = 1.0
    boundary_reranker_focus_true_rank_min: int = 7
    boundary_reranker_focus_true_rank_max: int = 12
    boundary_reranker_winner_init_checkpoint_path: str = ""
    enable_two_head_shortlist_winner_v2_rebuild_dual_winner_routing: bool = False
    global_winner_checkpoint_path: str = ""
    hard_source_winner_checkpoint_path: str = ""
    dual_winner_route_by_source: bool = True
    dual_winner_use_global_for_non_hard: bool = True
    dual_winner_use_specialist_for_hard: bool = True
    enable_two_head_shortlist_winner_v2_rebuild_context_features: bool = False
    winner_context_use_source_features: bool = True
    winner_context_source_embedding_dim: int = 8
    winner_context_use_hard_source_indicator: bool = True
    winner_context_use_local_competition_features: bool = True
    winner_context_use_relative_top_candidate_features: bool = True
    winner_context_use_geometry_proxy_features: bool = True
    winner_context_use_only_existing_repo_features: bool = True
    winner_context_init_checkpoint_path: str = ""
    enable_two_head_shortlist_winner_v2_rebuild_multisite_pairwise: bool = False
    winner_use_multi_positive_targets: bool = True
    winner_multi_positive_mode: str = "softmax_uniform"
    winner_multi_positive_only_for_multisite: bool = True
    winner_multisite_loss_weight: float = 1.0
    winner_enable_pairwise_ranking: bool = True
    winner_pairwise_margin: float = 0.2
    winner_pairwise_loss_weight: float = 0.5
    winner_pairwise_sample_mode: str = "hard_false_only"
    winner_use_source_embedding: bool = True
    winner_source_embedding_dim: int = 8
    winner_use_source_bias: bool = True
    shortlist_enable_hard_negative_emphasis: bool = False
    shortlist_hard_negative_rank_min: int = 2
    shortlist_hard_negative_rank_max: int = 12
    shortlist_hard_negative_loss_weight: float = 0.0
    shortlist_hard_negative_mode: str = "top_false"
    shortlist_pairwise_margin: float = 0.20
    shortlist_pairwise_loss_weight: float = 0.0
    shortlist_hard_negative_max_per_true: int = 3
    shortlist_hard_negative_sample_mode: str = "top_false_only"
    candidate_mask_mode: str = "hard"
    candidate_mask_logit_bias: float = 2.0
    disable_cyp_task: bool = False
    fixed_cyp_index: int = -1
    fixed_cyp_logit: float = 8.0
    use_local_chemistry_path: bool = False
    local_chem_hidden_dim: int = 64
    local_chem_dropout: float = 0.05
    local_chem_init_scale: float = 0.08
    local_chem_logit_scale: float = 0.05
    use_event_context: bool = False
    use_accessibility_head: bool = False
    use_barrier_head: bool = False
    event_context_hidden_dim: int = 24
    event_context_rounds: int = 3
    accessibility_hidden_dim: int = 16
    barrier_hidden_dim: int = 32
    phase2_context_hidden_dim: int = 96
    phase2_context_dropout: float = 0.05
    phase2_context_init_scale: float = 0.10
    phase2_context_logit_scale: float = 0.05
    use_phase5_boundary_field: bool = False
    use_phase5_accessibility: bool = False
    use_phase5_cyp_profile: bool = False
    phase5_boundary_lmax: int = 2
    phase5_boundary_radius: float = 6.5
    phase5_access_lambda: float = 0.40
    phase5_proposer_hidden_dim: int = 96
    phase5_proposer_dropout: float = 0.05
    phase5_proposer_init_scale: float = 0.10
    phase5_proposer_logit_scale: float = 0.06
    use_phase5_sparse_relay: bool = False
    phase5_sparse_relay_hidden_dim: int = 96
    phase5_sparse_relay_rounds: int = 2
    phase5_sparse_relay_radius: float = 4.5
    phase5_sparse_relay_init_scale: float = 0.08
    use_cyp3a4_state_rescorer: bool = False
    cyp3a4_state_proximity_weight: float = 1.20
    cyp3a4_state_orientation_weight: float = 0.80
    cyp3a4_state_access_weight: float = 1.00
    cyp3a4_state_electronic_weight: float = 0.90
    cyp3a4_state_learned_weight: float = 1.00
    cyp3a4_state_distance_center: float = 4.5
    cyp3a4_state_distance_sigma: float = 1.2
    cyp3a4_state_orientation_alpha: float = 2.0
    cyp3a4_state_access_path_lambda: float = 1.10
    cyp3a4_state_access_crowding_lambda: float = 0.90
    cyp3a4_state_access_radial_lambda: float = 1.15
    cyp3a4_state_access_filter_lambda: float = 0.95
    cyp3a4_state_weight_temperature: float = 0.85
    cyp3a4_state_min_state_weight: float = 0.10
    cyp3a4_state_aggregation_temperature: float = 0.75
    cyp3a4_state_use_mechanistic_gate: bool = True
    use_topk_reranker: bool = False
    topk_reranker_k: int = 8
    topk_reranker_hidden_dim: int = 128
    topk_reranker_heads: int = 4
    topk_reranker_layers: int = 2
    topk_reranker_dropout: float = 0.10
    topk_reranker_residual_scale: float = 0.75
    topk_reranker_use_gate: bool = True
    topk_reranker_gate_bias: float = -2.0
    topk_reranker_ce_weight: float = 0.25
    topk_reranker_margin_weight: float = 0.25
    topk_reranker_margin_value: float = 0.30
    domain_adv_weight: float = 0.0
    domain_adv_grad_scale: float = 0.1
    domain_adv_hidden_dim: int = 64
    source_align_weight: float = 0.0
    source_align_cov_weight: float = 0.5
    use_source_site_heads: bool = False
    source_site_aux_weight: float = 0.0
    source_site_blend_weight: float = 0.0
    source_site_head_names: Tuple[str, ...] = (
        "drugbank",
        "az120",
        "metxbiodb",
        "attnsom",
        "cyp_dbs_external",
        "literature",
    )
    use_nexus_bridge: bool = True
    nexus_wave_hidden_dim: int = 64
    nexus_graph_dim: int = 48
    nexus_memory_capacity: int = 24576
    nexus_memory_topk: int = 16
    nexus_memory_frozen: bool = False
    nexus_rebuild_memory_before_train: bool = True
    nexus_wave_aux_weight: float = 0.10
    nexus_analogical_aux_weight: float = 0.08
    nexus_wave_site_init: float = 0.18
    nexus_analogical_site_init: float = 0.20
    nexus_analogical_cyp_init: float = 0.12
    use_nexus_site_arbiter: bool = True
    use_nexus_sideinfo_features: bool = False
    nexus_sideinfo_hidden_dim: int = 128
    nexus_sideinfo_dropout: float = 0.10
    nexus_sideinfo_init_scale: float = 0.20
    nexus_site_arbiter_hidden_dim: int = 128
    nexus_site_arbiter_dropout: float = 0.20
    nexus_site_label_smoothing: float = 0.05
    nexus_top1_margin_weight: float = 0.25
    nexus_top1_margin_value: float = 0.5
    nexus_lnn_vote_aux_weight: float = 0.01
    nexus_wave_vote_aux_weight: float = 0.01
    nexus_analogical_vote_aux_weight: float = 0.02
    nexus_wave_vote_consistency_weight: float = 0.00
    nexus_analogical_vote_consistency_weight: float = 0.00
    nexus_board_entropy_weight: float = 0.01
    nexus_vote_logit_scale: float = 2.0
    nexus_live_wave_vote_inputs: bool = True
    nexus_live_analogical_vote_inputs: bool = True
    nexus_live_wave_vote_grad_scale: float = 0.02
    nexus_live_analogical_vote_grad_scale: float = 0.02
    nexus_wave_sideinfo_aux_weight: float = 0.0
    nexus_analogical_sideinfo_aux_weight: float = 0.0
    nexus_analogical_cyp_aux_scale: float = 0.10
    nexus_topology_feature_dim: int = 5   # per-atom global topology features (scaffold/centrality/carbonyl/size)
    # Phase 1: Relational Proposer (replaces scalar site head with cross-atom attention)
    use_relational_proposer: bool = False
    relational_proposer_num_heads: int = 4
    relational_proposer_num_layers: int = 2
    relational_proposer_hidden_dim: Optional[int] = None  # defaults to som_branch_dim
    relational_proposer_use_pairwise: bool = True  # include pairwise aggregator (proven 77% acc)
    relational_proposer_dropout: float = 0.1
    relational_proposer_residual_scale: float = 0.1
    relational_proposer_prior_scale_init: float = 0.65
    # Phase 2: Pairwise Reranker (inference-time reranking using pairwise head)
    use_pairwise_reranker: bool = False
    pairwise_reranker_top_k: int = 6
    pairwise_reranker_aggregation: str = "copeland"  # "copeland", "bradley_terry", "sum"
    pairwise_reranker_mode: str = "top1_vs_others"  # "top1_vs_others" (recommended) or "all_pairs"
    pairwise_reranker_swap_threshold: float = 0.6  # For top1_vs_others: min P(challenger > top1) to swap
    pairwise_reranker_temperature: float = 1.0
    pairwise_reranker_checkpoint: Optional[str] = None  # path to pairwise head checkpoint
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cyp_names: Tuple[str, ...] = tuple(MAJOR_CYP_CLASSES)

    def __post_init__(self) -> None:
        if self.shared_hidden_dim is None:
            self.shared_hidden_dim = int(self.hidden_dim)
        if self.som_branch_dim is None:
            self.som_branch_dim = int(self.shared_hidden_dim)
        if self.cyp_branch_dim is None:
            self.cyp_branch_dim = int(self.shared_hidden_dim)
        if self.group_pooling_hidden_dim is None:
            self.group_pooling_hidden_dim = max(32, int(self.shared_hidden_dim) // 2)
        if self.tau_context_dim is None:
            self.tau_context_dim = int(self.shared_hidden_dim)
        if self.shared_encoder_layers is None:
            self.shared_encoder_layers = max(
                1,
                min(int(self.num_liquid_layers), int(self.split_after_layer)),
            )
        self.shared_encoder_layers = max(1, int(self.shared_encoder_layers))
        self.split_after_layer = int(self.shared_encoder_layers)
        self.num_liquid_layers = max(
            int(self.shared_encoder_layers),
            int(self.som_branch_layers),
            int(self.cyp_branch_layers),
        )
        self.manual_prior_fusion_mode = str(self.manual_prior_fusion_mode)
        self.manual_prior_init_scale = min(max(float(self.manual_prior_init_scale), 0.0), 1.5)
        self.model_variant = str(self.model_variant).strip().lower() or "baseline"
        if self.model_variant not in {"baseline", "advanced", "hybrid_selective"}:
            self.model_variant = "baseline"
        self.tau_blend_mode = str(self.tau_blend_mode).strip().lower() or "learnable"
        if self.tau_blend_mode not in {"learnable", "fixed"}:
            self.tau_blend_mode = "learnable"
        if self.tau_min <= 0.0:
            self.tau_min = 1.0e-3
        if self.tau_max <= self.tau_min:
            self.tau_max = self.tau_min + 1.0
        self.max_tunneling_edges_per_node = max(1, int(self.max_tunneling_edges_per_node))
        self.max_tunneling_path_length = max(1, int(self.max_tunneling_path_length))
        self.higher_order_topk = max(2, int(self.higher_order_topk))
        self.higher_order_heads = max(1, int(self.higher_order_heads))
        self.num_deliberation_steps = max(0, int(self.num_deliberation_steps))
        self.tunneling_barrier_min = max(0.0, float(self.tunneling_barrier_min))
        self.tunneling_barrier_max = max(self.tunneling_barrier_min + 1.0e-3, float(self.tunneling_barrier_max))
        self.tunneling_probability_floor = min(max(float(self.tunneling_probability_floor), 1.0e-8), 0.1)
        self.energy_loss_weight = max(0.0, float(self.energy_loss_weight))
        self.energy_loss_clip = max(0.1, float(self.energy_loss_clip))
        self.deliberation_loss_weight = max(0.0, float(self.deliberation_loss_weight))
        self.tau_prior_weight = max(0.0, float(self.tau_prior_weight))
        self.tau_prior_blend = min(max(float(self.tau_prior_blend), 0.0), 1.0)
        self.phase_scale = max(0.0, float(self.phase_scale))
        self.local_tunneling_scale = max(0.0, float(self.local_tunneling_scale))
        self.local_tunneling_clamp = max(0.5, float(self.local_tunneling_clamp))
        self.output_refinement_scale = max(0.0, float(self.output_refinement_scale))
        self.output_refinement_hidden_dim = max(8, int(self.output_refinement_hidden_dim))
        self.tunnel_residual_scale = max(0.0, float(self.tunnel_residual_scale))
        self.tunnel_residual_scale_max = max(self.tunnel_residual_scale + 1.0e-3, float(self.tunnel_residual_scale_max))
        self.energy_value_clip = max(0.5, float(self.energy_value_clip))
        self.deliberation_step_scale = max(0.0, float(self.deliberation_step_scale))
        self.deliberation_max_state_norm = max(1.0, float(self.deliberation_max_state_norm))
        self.instability_hidden_norm_warn = max(1.0, float(self.instability_hidden_norm_warn))
        self.instability_energy_warn = max(0.5, float(self.instability_energy_warn))
        self.nexus_wave_hidden_dim = max(16, int(self.nexus_wave_hidden_dim))
        self.nexus_graph_dim = max(16, int(self.nexus_graph_dim))
        self.nexus_memory_capacity = max(128, int(self.nexus_memory_capacity))
        self.nexus_memory_topk = max(1, int(self.nexus_memory_topk))
        self.nexus_wave_aux_weight = max(0.0, float(self.nexus_wave_aux_weight))
        self.nexus_analogical_aux_weight = max(0.0, float(self.nexus_analogical_aux_weight))
        self.nexus_wave_site_init = min(max(float(self.nexus_wave_site_init), 1.0e-3), 1.0 - 1.0e-3)
        self.nexus_analogical_site_init = min(max(float(self.nexus_analogical_site_init), 1.0e-3), 1.0 - 1.0e-3)
        self.nexus_analogical_cyp_init = min(max(float(self.nexus_analogical_cyp_init), 1.0e-3), 1.0 - 1.0e-3)
        self.use_nexus_sideinfo_features = bool(self.use_nexus_sideinfo_features)
        self.nexus_sideinfo_hidden_dim = max(32, int(self.nexus_sideinfo_hidden_dim))
        self.nexus_sideinfo_dropout = min(max(float(self.nexus_sideinfo_dropout), 0.0), 0.5)
        self.nexus_sideinfo_init_scale = min(max(float(self.nexus_sideinfo_init_scale), 1.0e-3), 1.0)
        self.phase5_boundary_lmax = max(0, min(4, int(self.phase5_boundary_lmax)))
        self.phase5_boundary_radius = max(2.0, float(self.phase5_boundary_radius))
        self.phase5_access_lambda = max(1.0e-3, float(self.phase5_access_lambda))
        self.phase5_proposer_hidden_dim = max(32, int(self.phase5_proposer_hidden_dim))
        self.phase5_proposer_dropout = min(max(float(self.phase5_proposer_dropout), 0.0), 0.5)
        self.phase5_proposer_init_scale = min(max(float(self.phase5_proposer_init_scale), 1.0e-3), 1.0)
        self.phase5_proposer_logit_scale = min(max(float(self.phase5_proposer_logit_scale), 1.0e-3), 1.0)
        self.use_phase5_sparse_relay = bool(self.use_phase5_sparse_relay)
        self.phase5_sparse_relay_hidden_dim = max(32, int(self.phase5_sparse_relay_hidden_dim))
        self.phase5_sparse_relay_rounds = max(1, min(4, int(self.phase5_sparse_relay_rounds)))
        self.phase5_sparse_relay_radius = max(1.0, float(self.phase5_sparse_relay_radius))
        self.phase5_sparse_relay_init_scale = min(max(float(self.phase5_sparse_relay_init_scale), 1.0e-3), 1.0)
        self.use_cyp3a4_state_rescorer = bool(self.use_cyp3a4_state_rescorer)
        self.cyp3a4_state_proximity_weight = max(0.0, float(self.cyp3a4_state_proximity_weight))
        self.cyp3a4_state_orientation_weight = max(0.0, float(self.cyp3a4_state_orientation_weight))
        self.cyp3a4_state_access_weight = max(0.0, float(self.cyp3a4_state_access_weight))
        self.cyp3a4_state_electronic_weight = max(0.0, float(self.cyp3a4_state_electronic_weight))
        self.cyp3a4_state_learned_weight = max(0.0, float(self.cyp3a4_state_learned_weight))
        self.cyp3a4_state_distance_center = max(0.1, float(self.cyp3a4_state_distance_center))
        self.cyp3a4_state_distance_sigma = max(1.0e-3, float(self.cyp3a4_state_distance_sigma))
        self.cyp3a4_state_orientation_alpha = max(0.5, float(self.cyp3a4_state_orientation_alpha))
        self.cyp3a4_state_access_path_lambda = max(0.0, float(self.cyp3a4_state_access_path_lambda))
        self.cyp3a4_state_access_crowding_lambda = max(0.0, float(self.cyp3a4_state_access_crowding_lambda))
        self.cyp3a4_state_access_radial_lambda = max(0.0, float(self.cyp3a4_state_access_radial_lambda))
        self.cyp3a4_state_access_filter_lambda = max(0.0, float(self.cyp3a4_state_access_filter_lambda))
        self.cyp3a4_state_weight_temperature = max(1.0e-3, float(self.cyp3a4_state_weight_temperature))
        self.cyp3a4_state_min_state_weight = min(max(float(self.cyp3a4_state_min_state_weight), 0.0), 0.30)
        self.cyp3a4_state_aggregation_temperature = max(1.0e-3, float(self.cyp3a4_state_aggregation_temperature))
        self.cyp3a4_state_use_mechanistic_gate = bool(self.cyp3a4_state_use_mechanistic_gate)
        self.nexus_site_arbiter_hidden_dim = max(32, int(self.nexus_site_arbiter_hidden_dim))
        self.nexus_site_arbiter_dropout = min(max(float(self.nexus_site_arbiter_dropout), 0.0), 0.5)
        self.nexus_lnn_vote_aux_weight = max(0.0, float(self.nexus_lnn_vote_aux_weight))
        self.nexus_wave_vote_aux_weight = max(0.0, float(self.nexus_wave_vote_aux_weight))
        self.nexus_analogical_vote_aux_weight = max(0.0, float(self.nexus_analogical_vote_aux_weight))
        self.nexus_wave_vote_consistency_weight = max(0.0, float(self.nexus_wave_vote_consistency_weight))
        self.nexus_analogical_vote_consistency_weight = max(0.0, float(self.nexus_analogical_vote_consistency_weight))
        self.nexus_board_entropy_weight = max(0.0, float(self.nexus_board_entropy_weight))
        self.nexus_vote_logit_scale = max(0.1, float(self.nexus_vote_logit_scale))
        self.nexus_live_wave_vote_inputs = bool(self.nexus_live_wave_vote_inputs)
        self.nexus_live_analogical_vote_inputs = bool(self.nexus_live_analogical_vote_inputs)
        self.nexus_live_wave_vote_grad_scale = min(max(float(self.nexus_live_wave_vote_grad_scale), 0.0), 1.0)
        self.nexus_live_analogical_vote_grad_scale = min(max(float(self.nexus_live_analogical_vote_grad_scale), 0.0), 1.0)
        self.nexus_wave_sideinfo_aux_weight = max(0.0, float(self.nexus_wave_sideinfo_aux_weight))
        self.nexus_analogical_sideinfo_aux_weight = max(0.0, float(self.nexus_analogical_sideinfo_aux_weight))
        self.nexus_analogical_cyp_aux_scale = max(0.0, float(self.nexus_analogical_cyp_aux_scale))
        self.use_source_site_heads = bool(self.use_source_site_heads)
        self.source_site_aux_weight = max(0.0, float(self.source_site_aux_weight))
        self.source_site_blend_weight = min(max(float(self.source_site_blend_weight), 0.0), 1.0)
        self.source_site_head_names = tuple(
            str(name).strip().lower().replace("-", "_").replace(" ", "_")
            for name in tuple(self.source_site_head_names or ())
            if str(name).strip()
        )
        self.cyp_site_condition_scale = max(0.0, float(self.cyp_site_condition_scale))
        self.site_logit_bias_warmup_epochs = max(0, int(self.site_logit_bias_warmup_epochs))
        self.site_logit_bias_weight = max(0.0, float(self.site_logit_bias_weight))
        self.site_ranking_weight = max(0.0, float(self.site_ranking_weight))
        self.site_hard_negative_fraction = min(max(float(self.site_hard_negative_fraction), 0.0), 1.0)
        self.site_top1_margin_topk = max(1, int(self.site_top1_margin_topk))
        self.site_top1_margin_decay = min(max(float(self.site_top1_margin_decay), 0.1), 1.0)
        self.site_cover_weight = max(0.0, float(self.site_cover_weight))
        self.site_cover_margin = max(0.0, float(self.site_cover_margin))
        self.site_cover_topk = max(1, int(self.site_cover_topk))
        self.site_shortlist_weight = max(0.0, float(self.site_shortlist_weight))
        self.site_shortlist_temperature = max(1.0e-3, float(self.site_shortlist_temperature))
        self.site_shortlist_topk = max(1, int(self.site_shortlist_topk))
        self.site_use_rank_weighted_shortlist = bool(self.site_use_rank_weighted_shortlist)
        self.site_hard_negative_weight = max(0.0, float(self.site_hard_negative_weight))
        self.site_hard_negative_margin = max(0.0, float(self.site_hard_negative_margin))
        self.site_hard_negative_max_per_true = max(1, int(self.site_hard_negative_max_per_true))
        self.site_use_top_score_hard_neg = bool(self.site_use_top_score_hard_neg)
        self.site_use_graph_local_hard_neg = bool(self.site_use_graph_local_hard_neg)
        self.site_use_3d_local_hard_neg = bool(self.site_use_3d_local_hard_neg)
        self.site_use_rank_weighted_hard_neg = bool(self.site_use_rank_weighted_hard_neg)
        self.site_source_weight_default = max(0.1, float(self.site_source_weight_default))
        self.site_source_weight_drugbank = max(0.1, float(self.site_source_weight_drugbank))
        self.site_source_weight_az120 = max(0.1, float(self.site_source_weight_az120))
        self.site_source_weight_metxbiodb = max(0.1, float(self.site_source_weight_metxbiodb))
        self.site_source_weight_attnsom = max(0.1, float(self.site_source_weight_attnsom))
        self.site_source_weight_cyp_dbs_external = max(0.1, float(self.site_source_weight_cyp_dbs_external))
        self.enable_two_head_shortlist_winner = bool(self.enable_two_head_shortlist_winner)
        self.shortlist_topk = max(1, int(self.shortlist_topk))
        self.shortlist_head_hidden_dim = (
            None if self.shortlist_head_hidden_dim is None else max(1, int(self.shortlist_head_hidden_dim))
        )
        self.shortlist_head_dropout = min(max(float(self.shortlist_head_dropout), 0.0), 0.5)
        self.winner_head_hidden_dim = (
            None if self.winner_head_hidden_dim is None else max(1, int(self.winner_head_hidden_dim))
        )
        self.winner_head_dropout = min(max(float(self.winner_head_dropout), 0.0), 0.5)
        self.shortlist_loss_weight = max(0.0, float(self.shortlist_loss_weight))
        self.winner_loss_weight = max(0.0, float(self.winner_loss_weight))
        self.train_winner_only_on_hits = bool(self.train_winner_only_on_hits)
        self.shortlist_use_existing_site_loss = bool(self.shortlist_use_existing_site_loss)
        self.shortlist_selection_metric = str(self.shortlist_selection_metric or "recall_at_6").strip().lower() or "recall_at_6"
        self.two_head_log_every_epoch = bool(self.two_head_log_every_epoch)
        self.enable_two_head_shortlist_winner_v2 = bool(self.enable_two_head_shortlist_winner_v2)
        self.frozen_shortlist_checkpoint_path = str(self.frozen_shortlist_checkpoint_path or "").strip()
        self.frozen_shortlist_topk = max(1, int(self.frozen_shortlist_topk))
        self.winner_v2_hidden_dim = None if self.winner_v2_hidden_dim is None else max(1, int(self.winner_v2_hidden_dim))
        self.winner_v2_dropout = min(max(float(self.winner_v2_dropout), 0.0), 0.5)
        self.winner_v2_use_existing_candidate_features = bool(self.winner_v2_use_existing_candidate_features)
        self.winner_v2_use_score_gap_features = bool(self.winner_v2_use_score_gap_features)
        self.winner_v2_use_rank_features = bool(self.winner_v2_use_rank_features)
        self.winner_v2_use_pairwise_features = bool(self.winner_v2_use_pairwise_features)
        self.winner_v2_use_graph_local_features = bool(self.winner_v2_use_graph_local_features)
        self.winner_v2_use_3d_local_features = bool(self.winner_v2_use_3d_local_features)
        self.winner_v2_train_only_on_hits = bool(self.winner_v2_train_only_on_hits)
        self.winner_v2_loss_weight = max(0.0, float(self.winner_v2_loss_weight))
        self.shortlist_v2_log_every_epoch = bool(self.shortlist_v2_log_every_epoch)
        self.enable_two_head_shortlist_winner_v2_1 = bool(self.enable_two_head_shortlist_winner_v2_1)
        self.winner_v2_1_hidden_dim = None if self.winner_v2_1_hidden_dim is None else max(1, int(self.winner_v2_1_hidden_dim))
        self.winner_v2_1_dropout = min(max(float(self.winner_v2_1_dropout), 0.0), 0.5)
        self.winner_v2_1_use_existing_candidate_features = bool(self.winner_v2_1_use_existing_candidate_features)
        self.winner_v2_1_use_score_gap_features = bool(self.winner_v2_1_use_score_gap_features)
        self.winner_v2_1_use_rank_features = bool(self.winner_v2_1_use_rank_features)
        self.winner_v2_1_use_pairwise_features = bool(self.winner_v2_1_use_pairwise_features)
        self.winner_v2_1_use_graph_local_features = bool(self.winner_v2_1_use_graph_local_features)
        self.winner_v2_1_use_3d_local_features = bool(self.winner_v2_1_use_3d_local_features)
        self.winner_v2_1_use_top2_gap_features = bool(self.winner_v2_1_use_top2_gap_features)
        self.winner_v2_1_use_normalized_score_features = bool(self.winner_v2_1_use_normalized_score_features)
        self.winner_v2_1_use_shortlist_context_features = bool(self.winner_v2_1_use_shortlist_context_features)
        self.winner_v2_1_use_soft_multi_positive_targets = bool(self.winner_v2_1_use_soft_multi_positive_targets)
        self.winner_v2_1_train_only_on_hits = bool(self.winner_v2_1_train_only_on_hits)
        self.winner_v2_1_loss_weight = max(0.0, float(self.winner_v2_1_loss_weight))
        self.shortlist_v2_1_log_every_epoch = bool(self.shortlist_v2_1_log_every_epoch)
        self.enable_two_head_shortlist_winner_v2_2 = bool(self.enable_two_head_shortlist_winner_v2_2)
        self.winner_v2_2_hidden_dim = None if self.winner_v2_2_hidden_dim is None else max(1, int(self.winner_v2_2_hidden_dim))
        self.winner_v2_2_dropout = min(max(float(self.winner_v2_2_dropout), 0.0), 0.5)
        self.winner_v2_2_use_existing_candidate_features = bool(self.winner_v2_2_use_existing_candidate_features)
        self.winner_v2_2_use_score_gap_features = bool(self.winner_v2_2_use_score_gap_features)
        self.winner_v2_2_use_rank_features = bool(self.winner_v2_2_use_rank_features)
        self.winner_v2_2_use_normalized_score_features = bool(self.winner_v2_2_use_normalized_score_features)
        self.winner_v2_2_use_pairwise_features = bool(self.winner_v2_2_use_pairwise_features)
        self.winner_v2_2_use_graph_local_features = bool(self.winner_v2_2_use_graph_local_features)
        self.winner_v2_2_use_3d_local_features = bool(self.winner_v2_2_use_3d_local_features)
        self.winner_v2_2_use_extra_candidate_features = bool(self.winner_v2_2_use_extra_candidate_features)
        self.winner_v2_2_use_soft_multi_positive_targets = bool(self.winner_v2_2_use_soft_multi_positive_targets)
        self.winner_v2_2_train_only_on_hits = bool(self.winner_v2_2_train_only_on_hits)
        self.winner_v2_2_loss_weight = max(0.0, float(self.winner_v2_2_loss_weight))
        self.winner_v2_2_use_source_weighting = bool(self.winner_v2_2_use_source_weighting)
        self.winner_v2_2_hard_source_weight = max(0.0, float(self.winner_v2_2_hard_source_weight))
        self.winner_v2_2_normal_source_weight = max(0.0, float(self.winner_v2_2_normal_source_weight))
        self.winner_v2_2_hard_sources = str(self.winner_v2_2_hard_sources or "").strip().lower()
        self.winner_v2_2_log_source_weight_stats = bool(self.winner_v2_2_log_source_weight_stats)
        self.shortlist_v2_2_log_every_epoch = bool(self.shortlist_v2_2_log_every_epoch)
        self.enable_two_head_shortlist_winner_v2_3 = bool(self.enable_two_head_shortlist_winner_v2_3)
        self.winner_v2_3_hidden_dim = None if self.winner_v2_3_hidden_dim is None else max(1, int(self.winner_v2_3_hidden_dim))
        self.winner_v2_3_dropout = min(max(float(self.winner_v2_3_dropout), 0.0), 0.5)
        self.winner_v2_3_use_existing_candidate_features = bool(self.winner_v2_3_use_existing_candidate_features)
        self.winner_v2_3_use_score_gap_features = bool(self.winner_v2_3_use_score_gap_features)
        self.winner_v2_3_use_rank_features = bool(self.winner_v2_3_use_rank_features)
        self.winner_v2_3_use_normalized_score_features = bool(self.winner_v2_3_use_normalized_score_features)
        self.winner_v2_3_use_pairwise_features = bool(self.winner_v2_3_use_pairwise_features)
        self.winner_v2_3_use_graph_local_features = bool(self.winner_v2_3_use_graph_local_features)
        self.winner_v2_3_use_3d_local_features = bool(self.winner_v2_3_use_3d_local_features)
        self.winner_v2_3_use_extra_candidate_features = bool(self.winner_v2_3_use_extra_candidate_features)
        self.winner_v2_3_use_soft_multi_positive_targets = bool(self.winner_v2_3_use_soft_multi_positive_targets)
        self.winner_v2_3_use_source_weighting = bool(self.winner_v2_3_use_source_weighting)
        self.winner_v2_3_use_source_oversampling = bool(self.winner_v2_3_use_source_oversampling)
        self.winner_v2_3_train_only_on_hits = bool(self.winner_v2_3_train_only_on_hits)
        self.winner_v2_3_loss_weight = max(0.0, float(self.winner_v2_3_loss_weight))
        self.winner_v2_3_hard_source_weight = max(0.0, float(self.winner_v2_3_hard_source_weight))
        self.winner_v2_3_normal_source_weight = max(0.0, float(self.winner_v2_3_normal_source_weight))
        self.winner_v2_3_hard_sources = str(self.winner_v2_3_hard_sources or "").strip().lower()
        self.winner_v2_3_log_feature_summary = bool(self.winner_v2_3_log_feature_summary)
        self.enable_two_head_shortlist_winner_v2_rebuild = bool(self.enable_two_head_shortlist_winner_v2_rebuild)
        self.enable_two_head_shortlist_winner_v2_rebuild_top12 = bool(
            self.enable_two_head_shortlist_winner_v2_rebuild_top12
        )
        self.winner_v2_rebuild_hidden_dim = (
            None if self.winner_v2_rebuild_hidden_dim is None else max(1, int(self.winner_v2_rebuild_hidden_dim))
        )
        self.winner_v2_rebuild_dropout = min(max(float(self.winner_v2_rebuild_dropout), 0.0), 0.5)
        self.winner_v2_rebuild_loss_weight = max(0.0, float(self.winner_v2_rebuild_loss_weight))
        self.winner_v2_rebuild_log_restore_summary = bool(self.winner_v2_rebuild_log_restore_summary)
        self.two_head_shortlist_eval_topk = max(1, int(self.two_head_shortlist_eval_topk))
        self.two_head_shortlist_winner_topk = max(1, int(self.two_head_shortlist_winner_topk))
        self.two_head_keep_aux_metrics_at_6 = bool(self.two_head_keep_aux_metrics_at_6)
        self.two_head_log_dual_k_metrics = bool(self.two_head_log_dual_k_metrics)
        self.enable_two_head_shortlist_winner_v2_rebuild_hard_source_finetune = bool(
            self.enable_two_head_shortlist_winner_v2_rebuild_hard_source_finetune
        )
        self.hard_source_names = str(self.hard_source_names or "").strip().lower()
        self.hard_source_finetune_require_hit = bool(self.hard_source_finetune_require_hit)
        self.hard_source_finetune_skip_non_hard_sources = bool(self.hard_source_finetune_skip_non_hard_sources)
        self.winner_finetune_init_checkpoint_path = str(self.winner_finetune_init_checkpoint_path or "").strip()
        self.hard_source_finetune_lr_scale = max(0.0, float(self.hard_source_finetune_lr_scale))
        self.enable_two_head_shortlist_winner_v2_rebuild_boundary_reranker = bool(
            self.enable_two_head_shortlist_winner_v2_rebuild_boundary_reranker
        )
        self.boundary_reranker_shortlist_k = max(2, int(self.boundary_reranker_shortlist_k))
        self.boundary_reranker_output_k = max(1, int(self.boundary_reranker_output_k))
        if self.boundary_reranker_output_k > self.boundary_reranker_shortlist_k:
            self.boundary_reranker_output_k = int(self.boundary_reranker_shortlist_k)
        self.boundary_reranker_train_on_rescued_only = bool(self.boundary_reranker_train_on_rescued_only)
        self.boundary_reranker_train_on_hits_only = bool(self.boundary_reranker_train_on_hits_only)
        self.boundary_reranker_use_pairwise_mode = bool(self.boundary_reranker_use_pairwise_mode)
        self.boundary_reranker_use_listwise_mode = bool(self.boundary_reranker_use_listwise_mode)
        self.boundary_reranker_hidden_dim = (
            None if self.boundary_reranker_hidden_dim is None else max(1, int(self.boundary_reranker_hidden_dim))
        )
        self.boundary_reranker_dropout = min(max(float(self.boundary_reranker_dropout), 0.0), 0.5)
        self.boundary_reranker_loss_weight = max(0.0, float(self.boundary_reranker_loss_weight))
        self.boundary_reranker_focus_true_rank_min = max(1, int(self.boundary_reranker_focus_true_rank_min))
        self.boundary_reranker_focus_true_rank_max = max(
            self.boundary_reranker_focus_true_rank_min,
            int(self.boundary_reranker_focus_true_rank_max),
        )
        self.boundary_reranker_winner_init_checkpoint_path = str(
            self.boundary_reranker_winner_init_checkpoint_path or ""
        ).strip()
        self.enable_two_head_shortlist_winner_v2_rebuild_dual_winner_routing = bool(
            self.enable_two_head_shortlist_winner_v2_rebuild_dual_winner_routing
        )
        self.global_winner_checkpoint_path = str(self.global_winner_checkpoint_path or "").strip()
        self.hard_source_winner_checkpoint_path = str(self.hard_source_winner_checkpoint_path or "").strip()
        self.dual_winner_route_by_source = bool(self.dual_winner_route_by_source)
        self.dual_winner_use_global_for_non_hard = bool(self.dual_winner_use_global_for_non_hard)
        self.dual_winner_use_specialist_for_hard = bool(self.dual_winner_use_specialist_for_hard)
        self.enable_two_head_shortlist_winner_v2_rebuild_context_features = bool(
            self.enable_two_head_shortlist_winner_v2_rebuild_context_features
        )
        self.winner_context_use_source_features = bool(self.winner_context_use_source_features)
        self.winner_context_source_embedding_dim = max(1, int(self.winner_context_source_embedding_dim))
        self.winner_context_use_hard_source_indicator = bool(self.winner_context_use_hard_source_indicator)
        self.winner_context_use_local_competition_features = bool(self.winner_context_use_local_competition_features)
        self.winner_context_use_relative_top_candidate_features = bool(
            self.winner_context_use_relative_top_candidate_features
        )
        self.winner_context_use_geometry_proxy_features = bool(self.winner_context_use_geometry_proxy_features)
        self.winner_context_use_only_existing_repo_features = bool(self.winner_context_use_only_existing_repo_features)
        self.winner_context_init_checkpoint_path = str(self.winner_context_init_checkpoint_path or "").strip()
        self.enable_two_head_shortlist_winner_v2_rebuild_multisite_pairwise = bool(
            self.enable_two_head_shortlist_winner_v2_rebuild_multisite_pairwise
        )
        self.winner_use_multi_positive_targets = bool(self.winner_use_multi_positive_targets)
        self.winner_multi_positive_mode = str(self.winner_multi_positive_mode or "softmax_uniform").strip().lower()
        if self.winner_multi_positive_mode not in {"softmax_uniform"}:
            self.winner_multi_positive_mode = "softmax_uniform"
        self.winner_multi_positive_only_for_multisite = bool(self.winner_multi_positive_only_for_multisite)
        self.winner_multisite_loss_weight = max(0.0, float(self.winner_multisite_loss_weight))
        self.winner_enable_pairwise_ranking = bool(self.winner_enable_pairwise_ranking)
        self.winner_pairwise_margin = max(0.0, float(self.winner_pairwise_margin))
        self.winner_pairwise_loss_weight = max(0.0, float(self.winner_pairwise_loss_weight))
        self.winner_pairwise_sample_mode = str(self.winner_pairwise_sample_mode or "hard_false_only").strip().lower()
        if self.winner_pairwise_sample_mode not in {"all_false", "hard_false_only", "top_false_only"}:
            self.winner_pairwise_sample_mode = "hard_false_only"
        self.winner_use_source_embedding = bool(self.winner_use_source_embedding)
        self.winner_source_embedding_dim = max(1, int(self.winner_source_embedding_dim))
        self.winner_use_source_bias = bool(self.winner_use_source_bias)
        self.shortlist_enable_hard_negative_emphasis = bool(self.shortlist_enable_hard_negative_emphasis)
        self.shortlist_hard_negative_rank_min = max(1, int(self.shortlist_hard_negative_rank_min))
        self.shortlist_hard_negative_rank_max = max(
            self.shortlist_hard_negative_rank_min,
            int(self.shortlist_hard_negative_rank_max),
        )
        self.shortlist_hard_negative_loss_weight = max(0.0, float(self.shortlist_hard_negative_loss_weight))
        self.shortlist_hard_negative_mode = str(self.shortlist_hard_negative_mode or "top_false").strip().lower()
        if self.shortlist_hard_negative_mode not in {
            "top_false",
            "rank_window",
            "top_false_plus_rank_window",
            "near_true_neighbors",
        }:
            self.shortlist_hard_negative_mode = "top_false"
        self.shortlist_pairwise_margin = max(0.0, float(self.shortlist_pairwise_margin))
        self.shortlist_pairwise_loss_weight = max(0.0, float(self.shortlist_pairwise_loss_weight))
        self.shortlist_hard_negative_max_per_true = max(1, int(self.shortlist_hard_negative_max_per_true))
        self.shortlist_hard_negative_sample_mode = str(
            self.shortlist_hard_negative_sample_mode or "top_false_only"
        ).strip().lower()
        if self.shortlist_hard_negative_sample_mode not in {"top_false_only", "rank_window", "all_hard_false"}:
            self.shortlist_hard_negative_sample_mode = "top_false_only"
        self.candidate_mask_mode = str(self.candidate_mask_mode).strip().lower() or "hard"
        if self.candidate_mask_mode not in {"hard", "soft", "off"}:
            self.candidate_mask_mode = "hard"
        self.candidate_mask_logit_bias = max(0.0, float(self.candidate_mask_logit_bias))
        self.disable_cyp_task = bool(self.disable_cyp_task)
        self.fixed_cyp_index = int(self.fixed_cyp_index)
        self.fixed_cyp_logit = float(self.fixed_cyp_logit)
        self.use_local_chemistry_path = bool(self.use_local_chemistry_path)
        self.local_chem_hidden_dim = max(16, int(self.local_chem_hidden_dim))
        self.local_chem_dropout = min(max(float(self.local_chem_dropout), 0.0), 0.5)
        self.local_chem_init_scale = min(max(float(self.local_chem_init_scale), 1.0e-3), 1.0)
        self.local_chem_logit_scale = min(max(float(self.local_chem_logit_scale), 1.0e-3), 1.0)
        self.use_event_context = bool(self.use_event_context)
        self.use_accessibility_head = bool(self.use_accessibility_head)
        self.use_barrier_head = bool(self.use_barrier_head)
        self.event_context_hidden_dim = max(8, int(self.event_context_hidden_dim))
        self.event_context_rounds = max(1, int(self.event_context_rounds))
        self.accessibility_hidden_dim = max(8, int(self.accessibility_hidden_dim))
        self.barrier_hidden_dim = max(8, int(self.barrier_hidden_dim))
        self.phase2_context_hidden_dim = max(16, int(self.phase2_context_hidden_dim))
        self.phase2_context_dropout = min(max(float(self.phase2_context_dropout), 0.0), 0.5)
        self.phase2_context_init_scale = min(max(float(self.phase2_context_init_scale), 1.0e-3), 1.0)
        self.phase2_context_logit_scale = min(max(float(self.phase2_context_logit_scale), 1.0e-3), 1.0)
        self.use_topk_reranker = bool(self.use_topk_reranker)
        self.topk_reranker_k = max(2, int(self.topk_reranker_k))
        self.topk_reranker_hidden_dim = max(32, int(self.topk_reranker_hidden_dim))
        self.topk_reranker_heads = max(1, int(self.topk_reranker_heads))
        self.topk_reranker_layers = max(1, int(self.topk_reranker_layers))
        self.topk_reranker_dropout = min(max(float(self.topk_reranker_dropout), 0.0), 0.5)
        self.topk_reranker_residual_scale = max(0.0, float(self.topk_reranker_residual_scale))
        self.topk_reranker_use_gate = bool(self.topk_reranker_use_gate)
        self.topk_reranker_gate_bias = float(self.topk_reranker_gate_bias)
        self.topk_reranker_ce_weight = max(0.0, float(self.topk_reranker_ce_weight))
        self.topk_reranker_margin_weight = max(0.0, float(self.topk_reranker_margin_weight))
        self.topk_reranker_margin_value = max(0.0, float(self.topk_reranker_margin_value))
        self.domain_adv_weight = max(0.0, float(self.domain_adv_weight))
        self.domain_adv_grad_scale = max(0.0, float(self.domain_adv_grad_scale))
        self.domain_adv_hidden_dim = max(16, int(self.domain_adv_hidden_dim))
        self.source_align_weight = max(0.0, float(self.source_align_weight))
        self.source_align_cov_weight = max(0.0, float(self.source_align_cov_weight))

    @property
    def num_cyp_classes(self) -> int:
        return len(self.cyp_names)

    @classmethod
    def baseline(cls, **overrides) -> "ModelConfig":
        return cls(model_variant="baseline", **overrides)

    @classmethod
    def light_advanced(cls, **overrides) -> "ModelConfig":
        params = {
            "model_variant": "advanced",
            "use_physics_residual": True,
            "use_energy_module": False,
            "use_tunneling_module": False,
            "use_graph_tunneling": False,
            "use_phase_augmented_state": False,
            "use_higher_order_coupling": False,
            "use_deliberation_loop": False,
            "num_deliberation_steps": 0,
            "energy_loss_weight": 0.0,
            "deliberation_loss_weight": 0.0,
            "learning_rate": 2e-4,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def full_advanced(cls, **overrides) -> "ModelConfig":
        params = {
            "model_variant": "advanced",
            "use_energy_module": True,
            "use_tunneling_module": True,
            "use_graph_tunneling": True,
            "use_phase_augmented_state": True,
            "use_higher_order_coupling": True,
            "use_physics_residual": True,
            "use_deliberation_loop": True,
            "use_energy_dynamics": True,
            "num_deliberation_steps": 3,
            "energy_loss_weight": 0.02,
            "deliberation_loss_weight": 0.01,
            "tau_prior_weight": 0.02,
            "deliberation_step_scale": 0.05,
            "deliberation_max_state_norm": 8.0,
            "tunnel_residual_scale": 0.025,
            "tunnel_residual_scale_max": 0.15,
            "energy_value_clip": 4.0,
            "energy_loss_clip": 1.5,
            "learning_rate": 2e-4,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def hybrid_selective(cls, **overrides) -> "ModelConfig":
        params = {
            "model_variant": "hybrid_selective",
            "use_physics_residual": True,
            "use_energy_module": False,
            "use_tunneling_module": False,
            "use_graph_tunneling": False,
            "use_phase_augmented_state": False,
            "use_higher_order_coupling": False,
            "use_deliberation_loop": False,
            "num_deliberation_steps": 0,
            "use_local_tunneling_bias": True,
            "local_tunneling_scale": 0.1,
            "local_tunneling_clamp": 5.0,
            "use_output_refinement": True,
            "output_refinement_scale": 0.1,
            "output_refinement_hidden_dim": 64,
            "learning_rate": 2e-4,
        }
        params.update(overrides)
        return cls(**params)


@dataclass
class TrainingConfig:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    precision: str = "float32"
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    epochs: int = 100
    batch_size: int = 8
    gradient_clip: float = 1.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 20
    train_split: float = 0.8
    augment: bool = True
    augment_factor: int = 3
    extras: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.learning_rate = max(1.0e-6, float(self.learning_rate))
        self.weight_decay = max(0.0, float(self.weight_decay))
        self.gradient_clip = max(0.0, float(self.gradient_clip))
        self.max_grad_norm = max(float(self.gradient_clip), float(self.max_grad_norm), 0.0)
