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
    use_cyp_site_conditioning: bool = True   # broadcast CYP logits as per-atom bias before site head
    use_cross_atom_attention: bool = True    # 2-layer self-attention on SoM atom features before site head
    use_bde_prior: bool = True               # learnable BDE→logit residual on top of site head
    bde_feature_index: int = 44             # index of normalized BDE in atom feature vector (stable; XTB appended at end)
    use_nexus_bridge: bool = True
    nexus_wave_hidden_dim: int = 64
    nexus_graph_dim: int = 48
    nexus_memory_capacity: int = 4096
    nexus_memory_topk: int = 32
    nexus_memory_frozen: bool = False
    nexus_rebuild_memory_before_train: bool = True
    nexus_wave_aux_weight: float = 0.10
    nexus_analogical_aux_weight: float = 0.08
    nexus_wave_site_init: float = 0.18
    nexus_analogical_site_init: float = 0.20
    nexus_analogical_cyp_init: float = 0.12
    use_nexus_site_arbiter: bool = True
    nexus_site_arbiter_hidden_dim: int = 128
    nexus_site_arbiter_dropout: float = 0.20
    nexus_site_label_smoothing: float = 0.05
    nexus_top1_margin_weight: float = 0.25
    nexus_top1_margin_value: float = 0.5
    nexus_lnn_vote_aux_weight: float = 0.01
    nexus_wave_vote_aux_weight: float = 0.04
    nexus_analogical_vote_aux_weight: float = 0.04
    nexus_wave_vote_consistency_weight: float = 0.00
    nexus_analogical_vote_consistency_weight: float = 0.00
    nexus_board_entropy_weight: float = 0.01
    nexus_vote_logit_scale: float = 2.0
    nexus_live_wave_vote_inputs: bool = True
    nexus_live_analogical_vote_inputs: bool = True
    nexus_live_wave_vote_grad_scale: float = 0.05
    nexus_live_analogical_vote_grad_scale: float = 0.05
    nexus_analogical_cyp_aux_scale: float = 0.10
    nexus_topology_feature_dim: int = 5   # per-atom global topology features (scaffold/centrality/carbonyl/size)
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
        self.nexus_analogical_cyp_aux_scale = max(0.0, float(self.nexus_analogical_cyp_aux_scale))

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
