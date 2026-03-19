from __future__ import annotations

from pathlib import Path
from typing import Dict

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.features.route_prior import combine_lnn_with_prior
from .config import RecursiveMetabolismConfig


if TORCH_AVAILABLE:
    def load_base_hybrid_checkpoint(checkpoint_path: str | Path, device) -> HybridLNNModel:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_cfg = payload.get("config", {}).get("base_model") or ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            use_3d_branch=True,
            return_intermediate_stats=True,
        ).__dict__
        base_model = LiquidMetabolismNetV2(ModelConfig(**base_cfg))
        model = HybridLNNModel(base_model)
        state_dict = payload.get("model_state_dict") or payload
        model.load_state_dict(state_dict, strict=False)
        return model


    class RecursiveMetabolismModel(nn.Module):
        def __init__(self, base_model: HybridLNNModel, config: RecursiveMetabolismConfig):
            super().__init__()
            self.base_model = base_model
            self.config = config
            step_vocab = max(8, int(config.max_steps) + 2)
            self.step_embedding = nn.Embedding(step_vocab, config.step_embedding_dim)
            self.mol_proj = nn.LazyLinear(config.recursive_hidden_dim // 2)
            self.adjust_head = nn.Sequential(
                nn.LazyLinear(config.recursive_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.recursive_dropout),
                nn.Linear(config.recursive_hidden_dim, config.recursive_hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(config.recursive_hidden_dim // 2, 1),
            )
            self.gate_head = nn.Sequential(
                nn.LazyLinear(config.recursive_hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(config.recursive_hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            self.scale = nn.Parameter(torch.tensor(float(config.recursive_scale_init)))
            if config.freeze_base_model:
                self.set_base_trainable(False)

        def set_base_trainable(self, trainable: bool) -> None:
            for param in self.base_model.parameters():
                param.requires_grad = bool(trainable)

        def _run_base(self, batch: Dict[str, object]) -> Dict[str, object]:
            context = torch.enable_grad() if any(param.requires_grad for param in self.base_model.parameters()) else torch.no_grad()
            with context:
                return self.base_model(batch)

        def forward(self, batch: Dict[str, object]) -> Dict[str, object]:
            base_outputs = self._run_base(batch)
            base_site_logits = base_outputs["site_logits"]
            batch_index = batch["batch"]
            step_numbers = batch["graph_step_numbers"].clamp(min=0, max=self.step_embedding.num_embeddings - 1)
            step_atom = self.step_embedding(step_numbers)[batch_index]

            cyp_probs = torch.softmax(base_outputs["cyp_logits"], dim=-1)[batch_index]
            mol_features = base_outputs.get("mol_features")
            if mol_features is None:
                mol_features = torch.zeros((int(step_numbers.shape[0]), 1), device=base_site_logits.device, dtype=base_site_logits.dtype)
            mol_atom = self.mol_proj(mol_features)[batch_index]

            manual_atom = batch.get("manual_engine_atom_features")
            if manual_atom is None:
                manual_atom = torch.zeros((base_site_logits.size(0), 0), device=base_site_logits.device, dtype=base_site_logits.dtype)
            xtb_atom = batch.get("xtb_atom_features")
            if xtb_atom is None or not self.config.include_xtb_features:
                xtb_atom = torch.zeros((base_site_logits.size(0), 0), device=base_site_logits.device, dtype=base_site_logits.dtype)

            recursive_features = torch.cat(
                [
                    base_site_logits,
                    torch.sigmoid(base_site_logits),
                    step_atom,
                    mol_atom,
                    cyp_probs,
                    manual_atom,
                    xtb_atom,
                ],
                dim=-1,
            )
            delta = self.adjust_head(recursive_features)
            gate = self.gate_head(recursive_features)
            recursive_site_logits = base_site_logits + gate * self.scale * delta
            return {
                "base_outputs": base_outputs,
                "base_site_logits": base_site_logits,
                "recursive_site_logits": recursive_site_logits,
                "cyp_logits": base_outputs["cyp_logits"],
                "diagnostics": {
                    "recursive_gate_mean": float(gate.detach().mean().item()),
                    "recursive_delta_mean": float(delta.detach().mean().item()),
                    "recursive_delta_max": float(delta.detach().abs().max().item()),
                    "xtb_valid_atoms": float(batch.get("xtb_atom_valid_mask", torch.zeros_like(base_site_logits)).float().mean().item())
                    if batch.get("xtb_atom_valid_mask") is not None
                    else 0.0,
                },
            }
else:  # pragma: no cover
    def load_base_hybrid_checkpoint(*args, **kwargs):
        require_torch()

    class RecursiveMetabolismModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
