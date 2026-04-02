from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class ManualEnginePriorEncoder(nn.Module):
        """Project optional manual-engine features and prior logits into model space."""

        def __init__(self, atom_dim: int, mol_dim: int, num_cyp_classes: int):
            super().__init__()
            self.atom_dim = int(atom_dim)
            self.mol_dim = int(mol_dim)
            self.num_cyp_classes = int(num_cyp_classes)
            self.atom_feature_proj = nn.LazyLinear(self.atom_dim)
            self.mol_feature_proj = nn.LazyLinear(self.mol_dim)
            self.atom_logit_proj = nn.LazyLinear(1)
            self.cyp_logit_proj = nn.LazyLinear(self.num_cyp_classes)

        def _to_float_tensor(self, value, device, rows: int):
            if value is None:
                return None
            if not hasattr(value, "to"):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
            else:
                value = value.to(device=device, dtype=torch.float32)
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            if rows and value.size(0) != rows:
                raise ValueError(f"Manual feature rows {value.size(0)} != expected {rows}")
            return value

        def forward(self, batch: Dict[str, object], num_atoms: int, num_molecules: int, device) -> Dict[str, Optional[torch.Tensor]]:
            atom_features = self._to_float_tensor(batch.get("manual_engine_atom_features"), device, num_atoms)
            mol_features = self._to_float_tensor(batch.get("manual_engine_mol_features"), device, num_molecules)
            atom_prior_logits = self._to_float_tensor(batch.get("manual_engine_atom_prior_logits"), device, num_atoms)
            cyp_prior_logits = self._to_float_tensor(batch.get("manual_engine_cyp_prior_logits"), device, num_molecules)

            atom_prior_embedding = self.atom_feature_proj(atom_features) if atom_features is not None else None
            mol_prior_embedding = self.mol_feature_proj(mol_features) if mol_features is not None else None

            if atom_prior_logits is None and atom_features is not None:
                atom_prior_logits = self.atom_logit_proj(atom_features)
            if cyp_prior_logits is None and mol_features is not None:
                cyp_prior_logits = self.cyp_logit_proj(mol_features)

            diagnostics = {
                "atom_prior_present": float(atom_features is not None or atom_prior_logits is not None),
                "mol_prior_present": float(mol_features is not None or cyp_prior_logits is not None),
            }
            return {
                "atom_prior_embedding": atom_prior_embedding,
                "mol_prior_embedding": mol_prior_embedding,
                "atom_prior_logits": atom_prior_logits,
                "cyp_prior_logits": cyp_prior_logits,
                "diagnostics": diagnostics,
            }


    class ResidualFusionHead(nn.Module):
        """Fuse learned residual logits with optional prior logits."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            *,
            prior_feature_dim: int = 0,
            hidden_dim: Optional[int] = None,
            fusion_mode: str = "gated_add",
            dropout: float = 0.1,
            prior_scale_init: float = 0.65,
        ):
            super().__init__()
            self.output_dim = int(output_dim)
            self.prior_feature_dim = int(prior_feature_dim)
            self.fusion_mode = str(fusion_mode)
            hidden_dim = hidden_dim or max(32, input_dim)
            init = min(max(float(prior_scale_init), 1.0e-3), 1.5)
            init_sigmoid = min(max(init / 1.5, 1.0e-3), 1.0 - 1.0e-3)
            self.prior_scale_logit = nn.Parameter(torch.logit(torch.tensor(init_sigmoid, dtype=torch.float32)))
            self.residual_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )
            gate_in = input_dim + self.output_dim + self.prior_feature_dim
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.output_dim),
                nn.Sigmoid(),
            )
            self.concat_proj = nn.Linear(self.output_dim * 2, self.output_dim)

        def forward(self, features, prior_logits=None, prior_features=None):
            residual_logits = self.residual_net(features)
            prior_scale = 1.5 * torch.sigmoid(self.prior_scale_logit).to(device=features.device, dtype=features.dtype)
            if prior_logits is None:
                prior_logits = torch.zeros(
                    features.size(0),
                    self.output_dim,
                    device=features.device,
                    dtype=features.dtype,
                )
            if prior_logits.ndim == 1:
                prior_logits = prior_logits.unsqueeze(-1)
            if self.prior_feature_dim > 0:
                if prior_features is None:
                    prior_features = torch.zeros(
                        features.size(0),
                        self.prior_feature_dim,
                        device=features.device,
                        dtype=features.dtype,
                    )
                elif prior_features.ndim == 1:
                    prior_features = prior_features.unsqueeze(-1)
            else:
                prior_features = torch.zeros(features.size(0), 0, device=features.device, dtype=features.dtype)
            scaled_prior_logits = prior_scale * prior_logits

            if self.fusion_mode == "additive":
                gate = torch.ones_like(residual_logits)
                logits = scaled_prior_logits + residual_logits
            elif self.fusion_mode == "concat_proj":
                gate = torch.ones_like(residual_logits)
                logits = self.concat_proj(torch.cat([scaled_prior_logits, residual_logits], dim=-1))
            else:
                gate = self.gate_net(torch.cat([features, scaled_prior_logits, prior_features], dim=-1))
                logits = scaled_prior_logits + gate * residual_logits

            diagnostics = {
                "residual_abs_mean": float(residual_logits.detach().abs().mean().item()),
                "prior_abs_mean": float(prior_logits.detach().abs().mean().item()),
                "gate_mean": float(gate.detach().mean().item()),
                "prior_scale": float(prior_scale.detach().item()),
            }
            return logits, {
                "residual_logits": residual_logits,
                "prior_logits": scaled_prior_logits,
                "gate": gate,
                "prior_scale": prior_scale,
                "diagnostics": diagnostics,
            }
else:  # pragma: no cover
    class ManualEnginePriorEncoder:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class ResidualFusionHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
