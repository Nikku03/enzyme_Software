"""
Advanced Physics-ML Ensemble for Site-of-Metabolism Prediction.

This module implements a sophisticated ensemble that combines:
1. ML model predictions (learned patterns from data)
2. Physics-based reactivity scores (BDE, accessibility, electronic effects)
3. CYP isoform-specific adjustments
4. Learnable gating mechanism to optimally combine methods

Key insight: Different CYP isoforms have different binding pockets and
preferences. CYP3A4 has a large, flexible pocket preferring large lipophilic
substrates, while CYP2D6 prefers basic nitrogen-containing substrates.

The ensemble learns WHEN to trust physics vs ML for each atom type and
molecular context.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdFreeSASA
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    Chem = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ============================================================================
# CYP ISOFORM-SPECIFIC CHEMISTRY KNOWLEDGE
# ============================================================================

# CYP isoform substrate preferences and binding characteristics
CYP_ISOFORM_PROFILES = {
    "CYP1A2": {
        "description": "Planar aromatic substrates, caffeine, theophylline",
        "preferred_patterns": ["polycyclic_aromatic", "planar_heterocycle"],
        "pocket_size": "small_planar",
        "electronic_preference": "pi_rich",
        "substrate_features": {
            "planarity_weight": 1.5,
            "aromatic_weight": 1.3,
            "size_penalty_threshold": 35,  # atoms
        },
    },
    "CYP2C9": {
        "description": "Acidic drugs, warfarin, NSAIDs",
        "preferred_patterns": ["carboxylic_acid", "acidic_heterocycle"],
        "pocket_size": "medium",
        "electronic_preference": "acidic",
        "substrate_features": {
            "acidic_group_weight": 1.4,
            "lipophilic_weight": 1.2,
        },
    },
    "CYP2C19": {
        "description": "PPIs, clopidogrel, some antidepressants",
        "preferred_patterns": ["imidazole", "pyridine"],
        "pocket_size": "medium",
        "electronic_preference": "basic_weak",
        "substrate_features": {
            "heterocycle_weight": 1.3,
        },
    },
    "CYP2D6": {
        "description": "Basic amines, antipsychotics, beta-blockers",
        "preferred_patterns": ["basic_nitrogen", "aromatic_amine"],
        "pocket_size": "small_deep",
        "electronic_preference": "basic",
        "substrate_features": {
            "basic_nitrogen_weight": 1.6,
            "aromatic_ring_distance": 5.0,  # optimal distance from N to aromatic
        },
    },
    "CYP3A4": {
        "description": "Large lipophilic substrates, macrolides, statins",
        "preferred_patterns": ["large_lipophilic", "polyether"],
        "pocket_size": "large_flexible",
        "electronic_preference": "neutral_lipophilic",
        "substrate_features": {
            "size_preference": "large",
            "flexibility_tolerance": 1.5,
            "lipophilicity_weight": 1.4,
        },
    },
}

# Extended SMARTS patterns with CYP-specific modifiers
EXTENDED_REACTIVITY_RULES = [
    # ========== High Reactivity (0.85-1.0) ==========
    # O-demethylation
    ("o_demethyl_aromatic", "[CH3;X4][OX2][c]", 0.95, "Aromatic O-demethylation"),
    ("o_demethyl_aliphatic", "[CH3;X4][OX2][CX4]", 0.88, "Aliphatic O-demethylation"),
    
    # Benzylic oxidation (stabilized radical)
    ("benzylic_ch2", "[CH2;X4;!R][c]", 0.93, "Benzylic methylene"),
    ("benzylic_ch", "[CH;X4;!R]([c])[!H]", 0.91, "Benzylic methine"),
    ("benzylic_ch3", "[CH3;X4][c]", 0.90, "Benzylic methyl"),
    
    # N-dealkylation
    ("n_methyl_tert", "[CH3;X4][NX3;H0]", 0.92, "Tertiary N-demethylation"),
    ("n_methyl_sec", "[CH3;X4][NX3;H1]", 0.88, "Secondary N-demethylation"),
    ("n_ethyl_alpha", "[CH2;X4][NX3]", 0.85, "N-deethylation (alpha-C)"),
    
    # Allylic oxidation
    ("allylic_ch2", "[CH2;X4;!R][CX3]=[CX3]", 0.88, "Allylic methylene"),
    ("allylic_ch3", "[CH3;X4][CX3]=[CX3]", 0.85, "Allylic methyl"),
    
    # ========== Medium Reactivity (0.60-0.85) ==========
    # Heteroatom oxidation
    ("sulfoxide", "[SX2;!$([S]=*);!$([S-])]", 0.82, "Thioether S-oxidation"),
    ("n_oxide_tert", "[NX3;H0;!$([N+]);!$(N=*);!$(N#*)]([C])([C])[C]", 0.78, "Tertiary amine N-oxidation"),
    ("thiophene_s", "[sX2;r5]", 0.75, "Thiophene S-oxidation"),
    
    # Ring oxidation
    ("piperidine_alpha", "[CH2;R1;r6]@[NX3;R1;r6]", 0.80, "Piperidine alpha-C"),
    ("piperazine_alpha", "[CH2;R1;r6]@[NX3;R1;r6;H0]@[CH2;R1;r6]", 0.78, "Piperazine alpha-C"),
    ("morpholine_alpha", "[CH2;R1;r6]@[OX2;R1;r6]", 0.72, "Morpholine alpha-C"),
    
    # Epoxidation
    ("alkene_terminal", "[CH2;X3]=[CH;X3]", 0.70, "Terminal alkene"),
    ("alkene_internal", "[CH;X3]=[CH;X3]", 0.68, "Internal alkene"),
    ("styrene", "[CH2;X3]=[CH;X3][c]", 0.75, "Styrene epoxidation"),
    
    # Alpha to carbonyl
    ("alpha_ketone", "[CH2;X4][CX3](=O)[#6]", 0.65, "Alpha to ketone"),
    ("alpha_ester", "[CH2;X4][CX3](=O)[OX2]", 0.60, "Alpha to ester"),
    
    # ========== Low Reactivity (0.20-0.60) ==========
    # Aromatic hydroxylation (generally low for CYPs)
    ("aromatic_ch_activated", "[cH;$(c(c)(c)c)]", 0.45, "Para aromatic C-H"),
    ("aromatic_ch_ortho_edg", "[cH;$(c(c[O,N])c)]", 0.50, "Ortho to EDG"),
    ("aromatic_ch", "[cH]", 0.25, "Generic aromatic C-H"),
    
    # Deactivated positions
    ("halogen_adjacent", "[CH2,CH3;X4][F,Cl,Br,I]", 0.15, "Halogen-adjacent (deactivated)"),
    ("nitro_adjacent", "[CH2,CH3;X4][$(c[N+](=O)[O-])]", 0.10, "Nitro-adjacent (deactivated)"),
    ("carbonyl_beta", "[CH2;X4][CX4][CX3]=O", 0.35, "Beta to carbonyl"),
    
    # Non-reactive
    ("quaternary_c", "[CX4;H0]", 0.02, "Quaternary carbon (no H)"),
    ("bridgehead", "[CX4;H1;R2]", 0.05, "Bridgehead carbon"),
]


# Bond Dissociation Energy reference values (kJ/mol)
# Lower BDE = weaker bond = more reactive
BDE_REFERENCE = {
    # C-H bonds
    "methyl_primary": 423,
    "methylene_secondary": 410,
    "methine_tertiary": 400,
    "benzylic": 375,
    "allylic": 370,
    "alpha_heteroatom": 385,
    "aromatic": 473,
    
    # Heteroatom bonds
    "n_methyl": 356,
    "o_methyl": 385,
    "s_oxidation": 310,  # S-lone pair
}


def _compile_extended_patterns():
    """Compile extended SMARTS patterns."""
    if not RDKIT_AVAILABLE:
        return []
    compiled = []
    for name, smarts, score, description in EXTENDED_REACTIVITY_RULES:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                compiled.append((name, pattern, score, description))
        except Exception:
            continue
    return compiled


_EXTENDED_COMPILED = None


def get_extended_patterns():
    global _EXTENDED_COMPILED
    if _EXTENDED_COMPILED is None:
        _EXTENDED_COMPILED = _compile_extended_patterns()
    return _EXTENDED_COMPILED


class AdvancedPhysicsScorer:
    """
    Advanced physics-based SoM scorer with:
    - Extended SMARTS pattern matching
    - BDE-based scoring when xTB data available
    - Solvent-accessible surface area (SASA) for accessibility
    - CYP isoform-specific adjustments
    - Electronic effects (Fukui indices, partial charges)
    """
    
    def __init__(
        self,
        cyp_isoform: str = "CYP3A4",
        use_3d_accessibility: bool = True,
        bde_weight: float = 0.30,
        sasa_weight: float = 0.20,
        electronic_weight: float = 0.15,
        pattern_weight: float = 0.35,
    ):
        self.cyp_isoform = cyp_isoform
        self.cyp_profile = CYP_ISOFORM_PROFILES.get(cyp_isoform, CYP_ISOFORM_PROFILES["CYP3A4"])
        self.use_3d_accessibility = use_3d_accessibility
        
        # Scoring weights
        self.bde_weight = bde_weight
        self.sasa_weight = sasa_weight
        self.electronic_weight = electronic_weight
        self.pattern_weight = pattern_weight
        
        self.patterns = get_extended_patterns()
    
    def _compute_sasa(self, mol, conf_id: int = -1) -> np.ndarray:
        """Compute per-atom solvent accessible surface area."""
        if not RDKIT_AVAILABLE:
            return np.ones(mol.GetNumAtoms())
        
        try:
            # Get radii for SASA calculation
            radii = rdFreeSASA.classifyAtoms(mol)
            # Compute SASA
            sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id)
            
            # Per-atom contribution
            atom_sasa = np.zeros(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                atom_sasa[i] = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id, atomIdx=i)
            
            # Normalize
            max_sasa = atom_sasa.max()
            if max_sasa > 0:
                atom_sasa = atom_sasa / max_sasa
            
            return atom_sasa
        except Exception:
            return np.ones(mol.GetNumAtoms())
    
    def _compute_electronic_effects(
        self,
        mol,
        xtb_charges: Optional[np.ndarray] = None,
        xtb_fukui: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute electronic reactivity from partial charges and Fukui indices.
        
        Fukui f+ (electrophilic attack susceptibility) is key for CYP oxidation:
        - High f+ = electron-rich = more susceptible to oxidation
        """
        num_atoms = mol.GetNumAtoms()
        electronic_scores = np.zeros(num_atoms, dtype=np.float32)
        
        # Use xTB data if available
        if xtb_fukui is not None and len(xtb_fukui) == num_atoms:
            # Fukui f+ directly indicates oxidation susceptibility
            electronic_scores = np.clip(xtb_fukui, 0, 1)
        elif xtb_charges is not None and len(xtb_charges) == num_atoms:
            # Approximate: more negative charge = more electron-rich = more reactive
            # (This is a simplification; Fukui is better)
            electronic_scores = np.clip(-xtb_charges / 0.5 + 0.5, 0, 1)
        else:
            # Fallback: use Gasteiger charges
            try:
                AllChem.ComputeGasteigerCharges(mol)
                for i, atom in enumerate(mol.GetAtoms()):
                    charge = float(atom.GetDoubleProp("_GasteigerCharge"))
                    if not np.isfinite(charge):
                        charge = 0.0
                    # More negative = more electron-rich
                    electronic_scores[i] = np.clip(-charge / 0.3 + 0.5, 0, 1)
            except Exception:
                electronic_scores = np.full(num_atoms, 0.5)
        
        return electronic_scores
    
    def _compute_bde_scores(
        self,
        mol,
        xtb_bde: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score atoms based on bond dissociation energy.
        Lower BDE = weaker bond = more reactive.
        """
        num_atoms = mol.GetNumAtoms()
        bde_scores = np.zeros(num_atoms, dtype=np.float32)
        
        if xtb_bde is not None and len(xtb_bde) == num_atoms:
            # xTB BDE available - use directly
            # Typical range: 300-500 kJ/mol
            # Normalize: lower BDE = higher score
            bde_norm = (500 - np.clip(xtb_bde, 250, 500)) / 250.0
            bde_scores = np.clip(bde_norm, 0, 1)
        else:
            # Estimate BDE from atom environment
            for i, atom in enumerate(mol.GetAtoms()):
                atomic_num = atom.GetAtomicNum()
                if atomic_num != 6:  # Only score carbon C-H bonds
                    continue
                
                num_h = atom.GetTotalNumHs()
                if num_h == 0:
                    continue
                
                # Base BDE from hybridization
                hybridization = atom.GetHybridization()
                if hybridization == Chem.HybridizationType.SP3:
                    # Check degree (primary/secondary/tertiary)
                    degree = atom.GetDegree()
                    if degree == 1:
                        base_bde = BDE_REFERENCE["methyl_primary"]
                    elif degree == 2:
                        base_bde = BDE_REFERENCE["methylene_secondary"]
                    else:
                        base_bde = BDE_REFERENCE["methine_tertiary"]
                elif hybridization == Chem.HybridizationType.SP2:
                    base_bde = BDE_REFERENCE["aromatic"]
                else:
                    base_bde = 420  # default
                
                # Adjust for neighboring groups
                for neighbor in atom.GetNeighbors():
                    n_atomic = neighbor.GetAtomicNum()
                    if neighbor.GetIsAromatic():
                        # Benzylic stabilization
                        base_bde = min(base_bde, BDE_REFERENCE["benzylic"])
                    elif n_atomic == 7:  # N
                        base_bde = min(base_bde, BDE_REFERENCE["alpha_heteroatom"])
                    elif n_atomic == 8:  # O
                        base_bde = min(base_bde, BDE_REFERENCE["alpha_heteroatom"])
                
                # Normalize to score (lower BDE = higher score)
                bde_scores[i] = (500 - np.clip(base_bde, 250, 500)) / 250.0
        
        return bde_scores
    
    def _apply_cyp_adjustments(
        self,
        mol,
        base_scores: np.ndarray,
        pattern_names: np.ndarray,
    ) -> np.ndarray:
        """Apply CYP isoform-specific adjustments to scores."""
        adjusted = base_scores.copy()
        profile = self.cyp_profile
        substrate_features = profile.get("substrate_features", {})
        
        if self.cyp_isoform == "CYP3A4":
            # CYP3A4: large flexible pocket, prefers large lipophilic sites
            # Boost scores for positions that are:
            # - Accessible (not buried)
            # - On flexible parts of molecule
            # - Away from polar groups
            
            lipophilic_weight = substrate_features.get("lipophilicity_weight", 1.4)
            
            for i, atom in enumerate(mol.GetAtoms()):
                # Boost lipophilic carbons
                if atom.GetAtomicNum() == 6:
                    # Count polar neighbors
                    polar_neighbors = sum(
                        1 for n in atom.GetNeighbors() 
                        if n.GetAtomicNum() in [7, 8, 16]
                    )
                    if polar_neighbors == 0:
                        adjusted[i] *= lipophilic_weight
        
        elif self.cyp_isoform == "CYP2D6":
            # CYP2D6: prefers basic nitrogen, looks for aromatic ring at ~5-7 Å
            basic_n_weight = substrate_features.get("basic_nitrogen_weight", 1.6)
            
            # Find basic nitrogens
            basic_n_indices = []
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetAtomicNum() == 7:
                    # Check if basic (not amide, not nitro, etc.)
                    is_basic = True
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 8:  # C=O nearby
                            bond = mol.GetBondBetweenAtoms(i, neighbor.GetIdx())
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                is_basic = False
                    if is_basic:
                        basic_n_indices.append(i)
            
            # Boost sites near basic nitrogen
            for n_idx in basic_n_indices:
                for i in range(mol.GetNumAtoms()):
                    # Topological distance
                    path = Chem.GetShortestPath(mol, n_idx, i)
                    if path and 2 <= len(path) <= 5:
                        adjusted[i] *= 1.2
        
        elif self.cyp_isoform == "CYP1A2":
            # CYP1A2: prefers planar aromatics
            planarity_weight = substrate_features.get("planarity_weight", 1.5)
            aromatic_weight = substrate_features.get("aromatic_weight", 1.3)
            
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetIsAromatic():
                    adjusted[i] *= aromatic_weight
        
        return adjusted
    
    def score_molecule(
        self,
        smiles: str,
        *,
        xtb_bde: Optional[np.ndarray] = None,
        xtb_charges: Optional[np.ndarray] = None,
        xtb_fukui: Optional[np.ndarray] = None,
        atom_coordinates: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Score all atoms for SoM likelihood using physics-based methods.
        
        Args:
            smiles: Molecule SMILES
            xtb_bde: Per-atom BDE values from xTB
            xtb_charges: Per-atom partial charges from xTB
            xtb_fukui: Per-atom Fukui f+ indices from xTB
            atom_coordinates: 3D coordinates for accessibility
            
        Returns:
            Dict with detailed scoring breakdown
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for physics scoring")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add Hs and generate 3D if needed
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        
        # Generate 3D coordinates if not provided
        if atom_coordinates is None and self.use_3d_accessibility:
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
        
        # Initialize score arrays
        pattern_scores = np.zeros(num_atoms, dtype=np.float32)
        pattern_names = np.full(num_atoms, "", dtype=object)
        is_heavy = np.zeros(num_atoms, dtype=np.float32)
        
        # Mark heavy atoms
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                is_heavy[atom.GetIdx()] = 1.0
        
        # 1. Pattern matching
        for name, pattern, score, desc in self.patterns:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                atom_idx = match[0]
                if score > pattern_scores[atom_idx]:
                    pattern_scores[atom_idx] = score
                    pattern_names[atom_idx] = name
        
        # 2. BDE-based scoring
        bde_scores = self._compute_bde_scores(mol, xtb_bde)
        
        # 3. Electronic effects
        electronic_scores = self._compute_electronic_effects(mol, xtb_charges, xtb_fukui)
        
        # 4. SASA-based accessibility
        if self.use_3d_accessibility and mol.GetNumConformers() > 0:
            sasa_scores = self._compute_sasa(mol)
        else:
            sasa_scores = np.ones(num_atoms, dtype=np.float32)
        
        # 5. Combine scores
        combined = (
            self.pattern_weight * pattern_scores +
            self.bde_weight * bde_scores +
            self.electronic_weight * electronic_scores +
            self.sasa_weight * sasa_scores
        )
        
        # 6. Apply CYP-specific adjustments
        adjusted = self._apply_cyp_adjustments(mol, combined, pattern_names)
        
        # 7. Mask non-heavy atoms and normalize
        final_scores = adjusted * is_heavy
        max_score = final_scores.max()
        if max_score > 0:
            final_scores = final_scores / max_score
        
        return {
            "final_scores": final_scores,
            "pattern_scores": pattern_scores,
            "pattern_names": pattern_names,
            "bde_scores": bde_scores,
            "electronic_scores": electronic_scores,
            "sasa_scores": sasa_scores,
            "is_heavy": is_heavy,
            "cyp_isoform": self.cyp_isoform,
        }


if TORCH_AVAILABLE:
    class LearnableEnsembleHead(nn.Module):
        """
        Learnable ensemble head that combines ML predictions with physics features.
        
        Instead of fixed weights, this learns optimal combination based on:
        - Atom features (what type of site is this?)
        - Physics confidence (how confident is the physics prediction?)
        - Molecular context (what does the surrounding environment look like?)
        
        Architecture:
        - Input: [ML_logits, physics_features, atom_features]
        - Gating network decides ML vs physics weight per atom
        - Output: refined logits
        """
        
        def __init__(
            self,
            ml_dim: int = 1,
            physics_dim: int = 6,  # pattern, bde, electronic, sasa, confidence, cyp_match
            atom_feature_dim: int = 128,
            hidden_dim: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            # Combine all inputs
            total_input = ml_dim + physics_dim + atom_feature_dim
            
            # Gating network: decides how much to trust ML vs physics
            self.gate_net = nn.Sequential(
                nn.Linear(total_input, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 2),  # [ml_weight, physics_weight]
            )
            
            # Refinement network: additional learned adjustment
            self.refine_net = nn.Sequential(
                nn.Linear(total_input, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            
            # Physics feature encoder
            self.physics_encoder = nn.Sequential(
                nn.Linear(physics_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, ml_dim),
            )
            
            # Initialize with slight ML preference
            with torch.no_grad():
                self.gate_net[-1].bias.data = torch.tensor([0.5, -0.5])
        
        def forward(
            self,
            ml_logits: torch.Tensor,           # (N,) or (N, 1)
            physics_features: torch.Tensor,     # (N, physics_dim)
            atom_features: torch.Tensor,        # (N, atom_feature_dim)
            candidate_mask: Optional[torch.Tensor] = None,  # (N,)
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Returns:
                Dict with 'ensemble_logits' and diagnostics
            """
            # Ensure correct shapes
            if ml_logits.dim() == 1:
                ml_logits = ml_logits.unsqueeze(-1)
            
            # Concatenate inputs
            combined = torch.cat([ml_logits, physics_features, atom_features], dim=-1)
            
            # Compute gating weights
            gate_logits = self.gate_net(combined)
            gate_weights = F.softmax(gate_logits, dim=-1)
            ml_weight = gate_weights[:, 0:1]
            physics_weight = gate_weights[:, 1:2]
            
            # Encode physics to same space as ML
            physics_encoded = self.physics_encoder(physics_features)
            
            # Weighted combination
            base_ensemble = ml_weight * ml_logits + physics_weight * physics_encoded
            
            # Learned refinement
            refinement = self.refine_net(combined)
            
            # Final output
            ensemble_logits = base_ensemble + 0.1 * refinement
            
            # Apply mask if provided
            if candidate_mask is not None:
                mask = candidate_mask.float().unsqueeze(-1)
                ensemble_logits = ensemble_logits * mask + (-100.0) * (1 - mask)
            
            return {
                "ensemble_logits": ensemble_logits.squeeze(-1),
                "ml_weight": ml_weight.squeeze(-1),
                "physics_weight": physics_weight.squeeze(-1),
                "refinement": refinement.squeeze(-1),
            }


    class PhysicsMLEnsembleModel(nn.Module):
        """
        Complete Physics-ML Ensemble model.
        
        This wraps an existing ML model and adds physics-based scoring
        with a learnable combination head.
        """
        
        def __init__(
            self,
            ml_model: nn.Module,
            physics_scorer: Optional[AdvancedPhysicsScorer] = None,
            ensemble_hidden_dim: int = 64,
            freeze_ml_backbone: bool = True,
        ):
            super().__init__()
            
            self.ml_model = ml_model
            self.physics_scorer = physics_scorer or AdvancedPhysicsScorer()
            self.freeze_ml_backbone = freeze_ml_backbone
            
            # Get atom feature dim from ML model
            config = getattr(ml_model, "config", None)
            atom_dim = 128
            if config is not None:
                atom_dim = int(getattr(config, "som_branch_dim", 
                               getattr(config, "hidden_dim", 128)))
            
            # Learnable ensemble head
            self.ensemble_head = LearnableEnsembleHead(
                ml_dim=1,
                physics_dim=6,
                atom_feature_dim=atom_dim,
                hidden_dim=ensemble_hidden_dim,
            )
            
            if freeze_ml_backbone:
                for param in self.ml_model.parameters():
                    param.requires_grad = False
        
        def _extract_physics_features(
            self,
            smiles_list: List[str],
            batch_index: torch.Tensor,
            xtb_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Extract physics features for all atoms in batch."""
            device = batch_index.device
            num_atoms = batch_index.size(0)
            
            # 6 physics features per atom
            physics_features = torch.zeros(num_atoms, 6, device=device)
            
            atom_offset = 0
            for mol_idx, smiles in enumerate(smiles_list):
                mol_mask = (batch_index == mol_idx)
                mol_num_atoms = mol_mask.sum().item()
                
                try:
                    # Get xTB data if available
                    xtb_bde = None
                    xtb_charges = None
                    if xtb_features is not None:
                        mol_xtb = xtb_features[mol_mask].cpu().numpy()
                        if mol_xtb.shape[1] >= 1:
                            xtb_bde = mol_xtb[:, 0]  # First feature is BDE
                        if mol_xtb.shape[1] >= 2:
                            xtb_charges = mol_xtb[:, 1]
                    
                    result = self.physics_scorer.score_molecule(
                        smiles,
                        xtb_bde=xtb_bde,
                        xtb_charges=xtb_charges,
                    )
                    
                    # Pack features
                    n = min(mol_num_atoms, len(result["final_scores"]))
                    physics_features[atom_offset:atom_offset+n, 0] = torch.tensor(
                        result["pattern_scores"][:n], device=device
                    )
                    physics_features[atom_offset:atom_offset+n, 1] = torch.tensor(
                        result["bde_scores"][:n], device=device
                    )
                    physics_features[atom_offset:atom_offset+n, 2] = torch.tensor(
                        result["electronic_scores"][:n], device=device
                    )
                    physics_features[atom_offset:atom_offset+n, 3] = torch.tensor(
                        result["sasa_scores"][:n], device=device
                    )
                    # Confidence: high pattern score = high confidence
                    physics_features[atom_offset:atom_offset+n, 4] = torch.tensor(
                        (result["pattern_scores"][:n] > 0.5).astype(np.float32), device=device
                    )
                    # Final physics score
                    physics_features[atom_offset:atom_offset+n, 5] = torch.tensor(
                        result["final_scores"][:n], device=device
                    )
                    
                except Exception:
                    pass
                
                atom_offset += mol_num_atoms
            
            return physics_features
        
        def forward(
            self,
            batch: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """Forward pass combining ML and physics."""
            
            # Get ML predictions
            with torch.set_grad_enabled(not self.freeze_ml_backbone):
                ml_outputs = self.ml_model(batch)
            
            ml_logits = ml_outputs.get("site_logits")
            atom_features = ml_outputs.get("atom_features")
            
            if ml_logits is None or atom_features is None:
                return ml_outputs
            
            # Get physics features
            smiles_list = batch.get("smiles", [])
            if isinstance(smiles_list, torch.Tensor):
                smiles_list = []  # Can't use if tensor
            
            batch_index = batch.get("batch", torch.zeros(ml_logits.size(0), dtype=torch.long))
            xtb_features = batch.get("xtb_atom_features")
            
            if smiles_list:
                physics_features = self._extract_physics_features(
                    smiles_list, batch_index, xtb_features
                )
            else:
                # Fallback: use zeros
                physics_features = torch.zeros(
                    ml_logits.size(0), 6, 
                    device=ml_logits.device, dtype=ml_logits.dtype
                )
            
            # Run ensemble head
            candidate_mask = batch.get("candidate_mask")
            ensemble_result = self.ensemble_head(
                ml_logits=ml_logits,
                physics_features=physics_features,
                atom_features=atom_features,
                candidate_mask=candidate_mask,
            )
            
            # Update outputs
            outputs = dict(ml_outputs)
            outputs["site_logits"] = ensemble_result["ensemble_logits"]
            outputs["site_logits_ml"] = ml_logits
            outputs["ml_weight"] = ensemble_result["ml_weight"]
            outputs["physics_weight"] = ensemble_result["physics_weight"]
            outputs["physics_features"] = physics_features
            
            return outputs


def evaluate_ensemble_on_dataset(
    dataset: List[Dict],
    ml_scores_key: str = "ml_scores",
    cyp_isoform: str = "CYP3A4",
    ml_weight: float = 0.55,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate physics-ML ensemble on a dataset.
    
    Args:
        dataset: List of dicts with 'smiles', 'site_labels', and ml_scores_key
        ml_scores_key: Key for ML predictions in dataset
        cyp_isoform: CYP isoform for physics adjustments
        ml_weight: Weight for ML predictions (0-1)
        
    Returns:
        Evaluation metrics
    """
    physics_scorer = AdvancedPhysicsScorer(cyp_isoform=cyp_isoform)
    
    top1_correct = 0
    top3_correct = 0
    top1_physics_only = 0
    top1_ml_only = 0
    total = 0
    
    disagreements_physics_won = 0
    disagreements_ml_won = 0
    agreements = 0
    
    for item in dataset:
        smiles = item.get("smiles", "")
        true_sites = item.get("site_labels", [])
        ml_scores = item.get(ml_scores_key)
        
        if not smiles or not true_sites:
            continue
        
        true_site_set = set(int(s) for s in true_sites if isinstance(s, (int, float)))
        
        try:
            # Get physics scores
            physics_result = physics_scorer.score_molecule(smiles)
            physics_scores = physics_result["final_scores"]
            
            # Prepare ML scores
            if ml_scores is not None:
                ml_array = np.array(ml_scores, dtype=np.float32)
            else:
                ml_array = np.zeros_like(physics_scores)
            
            # Align lengths
            min_len = min(len(physics_scores), len(ml_array))
            physics_scores = physics_scores[:min_len]
            ml_array = ml_array[:min_len]
            is_heavy = physics_result["is_heavy"][:min_len]
            
            # Normalize
            if ml_array.max() > ml_array.min():
                ml_norm = (ml_array - ml_array.min()) / (ml_array.max() - ml_array.min())
            else:
                ml_norm = ml_array
            
            # Ensemble
            ensemble = ml_weight * ml_norm + (1 - ml_weight) * physics_scores
            ensemble = ensemble * is_heavy
            
            # Get predictions
            heavy_mask = is_heavy > 0.5
            if not heavy_mask.any():
                continue
            
            ml_top1 = np.argmax(ml_norm * is_heavy)
            physics_top1 = np.argmax(physics_scores)
            ensemble_top1 = np.argmax(ensemble)
            ensemble_top3 = set(np.argsort(-ensemble)[:3])
            
            # Check accuracy
            if ensemble_top1 in true_site_set:
                top1_correct += 1
            if true_site_set & ensemble_top3:
                top3_correct += 1
            if ml_top1 in true_site_set:
                top1_ml_only += 1
            if physics_top1 in true_site_set:
                top1_physics_only += 1
            
            # Track disagreements
            if ml_top1 != physics_top1:
                if ensemble_top1 in true_site_set:
                    if physics_top1 in true_site_set:
                        disagreements_physics_won += 1
                    else:
                        disagreements_ml_won += 1
            else:
                agreements += 1
            
            total += 1
            
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue
    
    results = {
        "ensemble_top1": top1_correct / max(total, 1),
        "ensemble_top3": top3_correct / max(total, 1),
        "ml_only_top1": top1_ml_only / max(total, 1),
        "physics_only_top1": top1_physics_only / max(total, 1),
        "total": total,
        "agreements": agreements,
        "disagreements_physics_won": disagreements_physics_won,
        "disagreements_ml_won": disagreements_ml_won,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("ENSEMBLE EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total molecules: {total}")
        print(f"ML weight: {ml_weight:.2f}, Physics weight: {1-ml_weight:.2f}")
        print(f"\nAccuracy:")
        print(f"  Ensemble Top-1: {results['ensemble_top1']*100:.1f}%")
        print(f"  Ensemble Top-3: {results['ensemble_top3']*100:.1f}%")
        print(f"  ML-only Top-1:  {results['ml_only_top1']*100:.1f}%")
        print(f"  Physics Top-1:  {results['physics_only_top1']*100:.1f}%")
        print(f"\nML vs Physics disagreements:")
        print(f"  Agreements: {agreements}")
        print(f"  Physics won: {disagreements_physics_won}")
        print(f"  ML won: {disagreements_ml_won}")
    
    return results


if __name__ == "__main__":
    # Quick test
    scorer = AdvancedPhysicsScorer(cyp_isoform="CYP3A4")
    
    # Test on midazolam
    test_smiles = "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
    
    result = scorer.score_molecule(test_smiles)
    print(f"Midazolam physics scores (top 5):")
    top_indices = np.argsort(-result["final_scores"])[:5]
    for idx in top_indices:
        print(f"  Atom {idx}: {result['final_scores'][idx]:.3f} ({result['pattern_names'][idx]})")
