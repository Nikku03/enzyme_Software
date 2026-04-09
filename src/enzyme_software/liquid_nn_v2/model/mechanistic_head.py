"""Mechanistic SoM Head - Encodes CYP450 reaction mechanism as inductive bias.

This module implements a physics-informed approach to Site-of-Metabolism prediction
by explicitly encoding the CYP450 oxidation mechanism:

1. Substrate enters binding pocket and orients based on hydrophobic interactions
2. Fe=O (Compound I) forms at the heme center
3. Hydrogen atom abstraction occurs from nearest accessible C-H bond
4. The reaction rate depends on:
   - Distance to heme iron (exponential falloff)
   - C-H bond strength (BDE - lower is more reactive)
   - Steric accessibility (can Fe=O reach this atom?)
   - Orbital alignment (Fe=O approaches along specific trajectory)
   - Radical stability after H-abstraction (benzylic, allylic stabilization)

The key insight: P(SoM) is NOT just pattern matching on atom features.
It's a competition where atoms with lower activation barriers win.
"""
from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch


# Physical constants and reference values
BDE_REFERENCE_KCAL = 100.0  # Reference C-H BDE in kcal/mol (methane-like)
BDE_MIN_KCAL = 75.0  # Weakest typical C-H (benzylic, allylic)
BDE_MAX_KCAL = 110.0  # Strongest typical C-H
OPTIMAL_HEME_DISTANCE_A = 4.5  # Optimal Fe...H distance in Angstroms
HEME_DISTANCE_DECAY_A = 2.0  # Exponential decay length scale


if TORCH_AVAILABLE:
    class MechanisticSoMHead(nn.Module):
        """
        Physics-informed SoM prediction based on CYP450 mechanism.
        
        Computes P(SoM) based on:
        - Distance to heme (from boundary_field features)
        - Bond dissociation energy (from xTB or estimated)
        - Accessibility (steric + electronic)
        - Orbital alignment (axis cosine from CYP profile)
        - Chemistry prior (SMARTS-based reactivity boost/penalty)
        
        The output is a logit that can be added to the base model's site_logits
        as a mechanistic correction term.
        """
        
        def __init__(
            self,
            *,
            hidden_dim: int = 64,
            dropout: float = 0.1,
            use_learned_weights: bool = True,
            init_scale: float = 0.1,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.use_learned_weights = use_learned_weights
            
            # Input dimensions from various feature sources
            # phase5_atom_features: 18 dims (boundary field + accessibility)
            # local_chem_features: 11 dims  
            # physics_features: variable (BDE, etc.)
            # We'll concatenate what's available
            
            # Learned temperature parameters for each physics term
            # Initialize to encode the known mechanism
            if use_learned_weights:
                # Distance term: exp(-d / lambda)
                # lambda ~ 2-3 Angstroms for optimal reaction
                self.log_distance_lambda = nn.Parameter(torch.tensor(0.7))  # exp(0.7) ~ 2.0
                
                # BDE term: lower BDE = more reactive
                # Scale so that 20 kcal/mol difference gives ~2x rate difference
                self.log_bde_scale = nn.Parameter(torch.tensor(-2.0))  # ~0.14
                
                # Accessibility weight
                self.log_access_weight = nn.Parameter(torch.tensor(0.5))
                
                # Orbital alignment weight (axis cosine term)
                self.log_orbital_weight = nn.Parameter(torch.tensor(0.0))
                
                # Radial gate weight (exponential falloff from optimal distance)
                self.log_radial_weight = nn.Parameter(torch.tensor(0.5))
                
                # Overall scale for mechanistic correction
                self.output_scale = nn.Parameter(torch.tensor(init_scale))
            
            # Optional: small MLP to learn residual corrections
            # This captures effects not in our explicit mechanism
            input_dim = 8  # heme_dist, bde, access, orbital, radial, steric, electro, field
            self.residual_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Initialize residual to near-zero so mechanism dominates initially
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)
            
            # Gate to blend mechanistic vs learned
            self.blend_gate = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            # Initialize gate to favor mechanistic term
            nn.init.constant_(self.blend_gate[-2].bias, -1.0)
        
        def _extract_mechanistic_features(
            self,
            batch: Dict[str, torch.Tensor],
            num_atoms: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> Dict[str, torch.Tensor]:
            """Extract and normalize features relevant to the mechanism."""
            
            def _get(key: str, expected_dim: int = 1) -> torch.Tensor:
                val = batch.get(key)
                if val is None:
                    return torch.zeros(num_atoms, expected_dim, device=device, dtype=dtype)
                t = val.to(device=device, dtype=dtype)
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                return t
            
            # Phase 5 features (boundary field)
            phase5 = _get("phase5_atom_features", 18)
            
            # Extract specific components from phase5
            # Layout: [scalar(1), vector(3), heme_distance(1), axis_cosine(1), 
            #          radial_gate(1), cost(1), score(1), blockage(1), profile(8)]
            if phase5.size(-1) >= 6:
                heme_distance = phase5[:, 4:5]  # heme_distance
                axis_cosine = phase5[:, 5:6]    # axis_cosine  
                radial_gate = phase5[:, 6:7] if phase5.size(-1) > 6 else torch.ones_like(heme_distance)
            else:
                heme_distance = torch.ones(num_atoms, 1, device=device, dtype=dtype) * 5.0
                axis_cosine = torch.zeros(num_atoms, 1, device=device, dtype=dtype)
                radial_gate = torch.ones(num_atoms, 1, device=device, dtype=dtype)
            
            # Local chemistry features
            local_chem = _get("local_chem_features", 11)
            if local_chem.size(-1) >= 5:
                steric_score = local_chem[:, 0:1]
                electro_score = local_chem[:, 1:2]
                field_score = local_chem[:, 2:3]
                access_proxy = local_chem[:, 3:4]
                crowding = local_chem[:, 4:5]
            else:
                steric_score = torch.zeros(num_atoms, 1, device=device, dtype=dtype)
                electro_score = torch.zeros(num_atoms, 1, device=device, dtype=dtype)
                field_score = torch.zeros(num_atoms, 1, device=device, dtype=dtype)
                access_proxy = torch.ones(num_atoms, 1, device=device, dtype=dtype)
                crowding = torch.zeros(num_atoms, 1, device=device, dtype=dtype)
            
            # BDE from physics features
            physics = batch.get("physics_features") or {}
            bde_raw = physics.get("bde_values")
            if bde_raw is not None:
                bde = torch.as_tensor(bde_raw, device=device, dtype=dtype).view(num_atoms, 1)
            else:
                # Default to reference BDE (neutral)
                bde = torch.full((num_atoms, 1), BDE_REFERENCE_KCAL, device=device, dtype=dtype)
            
            # Normalize BDE to [0, 1] where 0 = weak (reactive), 1 = strong (unreactive)
            bde_normalized = (bde - BDE_MIN_KCAL) / (BDE_MAX_KCAL - BDE_MIN_KCAL)
            bde_normalized = bde_normalized.clamp(0.0, 1.0)
            
            # Accessibility = access_proxy - penalty*crowding
            accessibility = (access_proxy - 0.3 * crowding).clamp(0.0, 1.0)
            
            return {
                "heme_distance": heme_distance,
                "axis_cosine": axis_cosine,
                "radial_gate": radial_gate,
                "bde": bde,
                "bde_normalized": bde_normalized,
                "accessibility": accessibility,
                "steric_score": steric_score,
                "electro_score": electro_score,
                "field_score": field_score,
            }
        
        def forward(
            self,
            batch: Dict[str, torch.Tensor],
            base_site_logits: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Compute mechanistic SoM scores.
            
            Returns:
                Dict with:
                - mechanistic_logits: The physics-based SoM prediction
                - combined_logits: base_site_logits + mechanistic correction (if base provided)
                - diagnostics: Dict of intermediate values for debugging
            """
            # Determine dimensions
            if base_site_logits is not None:
                num_atoms = base_site_logits.size(0)
                device = base_site_logits.device
                dtype = base_site_logits.dtype
            else:
                # Infer from batch
                for key in ["phase5_atom_features", "local_chem_features", "x"]:
                    if key in batch and batch[key] is not None:
                        t = batch[key]
                        num_atoms = t.size(0)
                        device = t.device
                        dtype = t.dtype
                        break
                else:
                    raise ValueError("Cannot determine num_atoms from batch")
            
            # Extract mechanistic features
            feats = self._extract_mechanistic_features(batch, num_atoms, device, dtype)
            
            # === PHYSICS-BASED SCORING ===
            
            # 1. Distance term: exponential falloff from heme
            # P ~ exp(-(d - d_opt)^2 / (2 * lambda^2))
            distance_lambda = torch.exp(self.log_distance_lambda).clamp(0.5, 5.0)
            d = feats["heme_distance"]
            d_opt = OPTIMAL_HEME_DISTANCE_A
            distance_score = torch.exp(-((d - d_opt) ** 2) / (2 * distance_lambda ** 2))
            
            # 2. BDE term: lower BDE = more reactive
            # P ~ exp(-BDE / (RT * scale))
            bde_scale = torch.exp(self.log_bde_scale).clamp(0.01, 1.0)
            bde_score = 1.0 - feats["bde_normalized"]  # Invert: low BDE = high score
            
            # 3. Accessibility term
            access_weight = torch.exp(self.log_access_weight).clamp(0.1, 5.0)
            access_score = feats["accessibility"]
            
            # 4. Orbital alignment: |cos(theta)| where theta is angle to Fe=O axis
            orbital_weight = torch.exp(self.log_orbital_weight).clamp(0.1, 5.0)
            orbital_score = torch.abs(feats["axis_cosine"]).clamp(0.0, 1.0)
            
            # 5. Radial gate (from boundary field)
            radial_weight = torch.exp(self.log_radial_weight).clamp(0.1, 5.0)
            radial_score = feats["radial_gate"]
            
            # Combined mechanistic score (log-space for numerical stability)
            # log P(SoM) ~ w1*log(dist) + w2*log(bde) + w3*log(access) + ...
            mechanistic_logit = (
                torch.log(distance_score.clamp(min=1e-6))
                + bde_scale * torch.log(bde_score.clamp(min=1e-6))
                + access_weight * torch.log(access_score.clamp(min=1e-6))
                + orbital_weight * torch.log(orbital_score.clamp(min=0.1))
                + radial_weight * torch.log(radial_score.clamp(min=1e-6))
            )
            
            # === LEARNED RESIDUAL ===
            # Small MLP to capture effects not in explicit mechanism
            residual_input = torch.cat([
                feats["heme_distance"] / 10.0,  # Normalize to ~[0, 1]
                feats["bde_normalized"],
                feats["accessibility"],
                torch.abs(feats["axis_cosine"]),
                feats["radial_gate"],
                feats["steric_score"],
                feats["electro_score"],
                feats["field_score"],
            ], dim=-1)
            
            learned_residual = self.residual_mlp(residual_input)
            blend_gate = self.blend_gate(residual_input)
            
            # Blend mechanistic + learned
            final_logit = (
                self.output_scale * mechanistic_logit
                + blend_gate * learned_residual
            )
            
            # Combine with base logits if provided
            if base_site_logits is not None:
                combined_logits = base_site_logits + final_logit.view_as(base_site_logits)
            else:
                combined_logits = final_logit
            
            return {
                "mechanistic_logits": final_logit,
                "combined_logits": combined_logits,
                "diagnostics": {
                    "distance_score_mean": float(distance_score.mean().item()),
                    "bde_score_mean": float(bde_score.mean().item()),
                    "access_score_mean": float(access_score.mean().item()),
                    "orbital_score_mean": float(orbital_score.mean().item()),
                    "radial_score_mean": float(radial_score.mean().item()),
                    "mechanistic_logit_mean": float(mechanistic_logit.mean().item()),
                    "learned_residual_mean": float(learned_residual.mean().item()),
                    "blend_gate_mean": float(blend_gate.mean().item()),
                    "output_scale": float(self.output_scale.item()),
                    "distance_lambda": float(torch.exp(self.log_distance_lambda).item()),
                    "bde_scale": float(torch.exp(self.log_bde_scale).item()),
                },
            }


else:  # pragma: no cover
    class MechanisticSoMHead:
        def __init__(self, *args, **kwargs):
            require_torch()
