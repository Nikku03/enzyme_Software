"""
Quantum Chemistry-Informed Physics Scoring Module for CYP Site-of-Metabolism Prediction.

This module implements a deep physics-based scoring system that goes beyond simple
SMARTS pattern matching to incorporate:

1. **Density Functional Theory (DFT) Approximations**: Using xTB/GFN2 computed features
   to estimate electronic properties relevant to CYP oxidation:
   - Fukui f+ indices (electrophilic susceptibility)
   - Condensed Fukui functions
   - Partial atomic charges (for radical stabilization)
   
2. **Bond Dissociation Energy (BDE) Calculations**: The rate-limiting step in CYP
   oxidation is often C-H abstraction. Lower BDE = more reactive.
   - Uses xTB vertical/adiabatic BDE when available
   - Falls back to group contribution methods
   
3. **Transition State Theory (TST)**: Estimates reaction barriers using the
   Bell-Evans-Polanyi principle:
   ΔH‡ = α * ΔH_rxn + β
   
4. **Enzyme-Substrate Binding**: CYP-specific binding pocket constraints:
   - Distance to heme iron (optimal ~4-6 Å)
   - Pocket shape complementarity
   - Access channel accessibility
   
5. **Reactive Intermediate Stabilization**: Factors that stabilize the radical
   intermediate formed after H-abstraction:
   - Resonance (benzylic, allylic)
   - Hyperconjugation
   - Spin delocalization

Mathematical Background:
-----------------------
The overall reaction rate for CYP oxidation at site i is:

k_i ∝ exp(-ΔG‡_i / RT) × P_binding × P_accessibility

where:
- ΔG‡_i is the activation free energy (from BDE via BEP relation)
- P_binding is the probability of productive binding pose
- P_accessibility is the geometric accessibility factor

We approximate this as a learned combination of physics features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFreeSASA
    from rdkit.Chem.rdchem import HybridizationType
    RDKIT_AVAILABLE = True
except ImportError:
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


# ============================================================================
# PHYSICAL CONSTANTS AND REFERENCE DATA
# ============================================================================

# Gas constant (kJ/mol/K)
R_GAS = 8.314e-3

# Reference temperature (K)
T_REF = 298.15

# CYP heme-iron parameters
HEME_OPTIMAL_DISTANCE = 5.0  # Angstroms
HEME_DISTANCE_SIGMA = 1.5   # Gaussian width

# Bell-Evans-Polanyi parameters for CYP oxidation
# ΔH‡ = α × ΔH_rxn + β
BEP_ALPHA = 0.7  # Evans-Polanyi slope
BEP_BETA = -250.0  # kJ/mol (intercept)


@dataclass
class AtomReactivityProfile:
    """Comprehensive reactivity profile for a single atom."""
    atom_index: int
    atom_symbol: str
    
    # Electronic properties
    partial_charge: float = 0.0
    fukui_electrophilic: float = 0.0  # f+: susceptibility to nucleophilic attack (by Fe=O)
    fukui_nucleophilic: float = 0.0   # f-: susceptibility to electrophilic attack
    fukui_radical: float = 0.0        # f0: susceptibility to radical attack
    
    # Bond properties
    bde_estimate: float = 400.0  # kJ/mol
    bde_source: str = "group_contribution"  # or "xtb_vertical", "xtb_adiabatic"
    num_hydrogens: int = 0
    
    # Geometric properties
    sasa: float = 1.0  # Solvent accessible surface area fraction
    buried_fraction: float = 0.0
    neighbor_count: int = 0
    
    # CYP-specific
    heme_distance: float = 10.0  # Angstroms (if 3D available)
    channel_accessibility: float = 1.0
    
    # Pattern matching
    reactivity_pattern: str = ""
    pattern_score: float = 0.0
    
    # Final scores
    electronic_score: float = 0.0
    steric_score: float = 0.0
    kinetic_score: float = 0.0  # TST-based
    binding_score: float = 0.0
    final_score: float = 0.0


# ============================================================================
# EXTENDED SMARTS REACTION PATTERNS WITH PHYSICAL JUSTIFICATION
# ============================================================================

# Format: (name, SMARTS, base_score, BDE_estimate_kJ/mol, description)
REACTION_PATTERNS = [
    # === O-Dealkylation (Oxygen α-carbon) ===
    # The C-H adjacent to oxygen is weakened by lone pair donation
    ("O-demethyl_ArOMe", "[CH3;X4][OX2]c", 0.95, 380, "Aromatic methoxy - highly favored"),
    ("O-demethyl_AlkOMe", "[CH3;X4][OX2][CX4]", 0.88, 390, "Aliphatic methoxy"),
    ("O-deethyl_alpha", "[CH2;X4]([OX2])[CH3]", 0.85, 385, "Ethoxy α-carbon"),
    
    # === N-Dealkylation (Nitrogen α-carbon) ===
    # The C-H adjacent to nitrogen is activated by nitrogen lone pair
    ("N-demethyl_tert", "[CH3;X4][NX3;H0]([#6])[#6]", 0.93, 370, "Tertiary amine N-methyl"),
    ("N-demethyl_sec", "[CH3;X4][NX3;H1][#6]", 0.88, 375, "Secondary amine N-methyl"),
    ("N-deethyl_alpha", "[CH2;X4]([NX3])[CH3]", 0.86, 378, "N-ethyl α-carbon"),
    ("N-ring_alpha_6", "[CH2;R1;r6]@[NX3;R1;r6]", 0.84, 382, "Piperidine/piperazine α-carbon"),
    ("N-ring_alpha_5", "[CH2;R1;r5]@[NX3;R1;r5]", 0.82, 384, "Pyrrolidine α-carbon"),
    
    # === Benzylic Oxidation ===
    # Benzylic C-H is weak due to resonance stabilization of radical
    ("benzylic_CH2", "[CH2;X4;!R]c1ccccc1", 0.92, 365, "Benzylic methylene"),
    ("benzylic_CH", "[CH;X4;!R](c1ccccc1)[!H]", 0.90, 360, "Benzylic methine"),
    ("benzylic_CH3", "[CH3;X4]c1ccccc1", 0.89, 368, "Benzylic methyl"),
    ("benzylic_heteroarom", "[CH2,CH3;X4][c;r5,r6]", 0.85, 372, "Heteroaromatic benzylic"),
    
    # === Allylic Oxidation ===
    # Allylic C-H stabilized by π-resonance
    ("allylic_CH2", "[CH2;X4;!R][CX3]=[CX3]", 0.87, 368, "Allylic methylene"),
    ("allylic_CH3", "[CH3;X4][CX3]=[CX3]", 0.84, 372, "Allylic methyl"),
    ("allylic_CH", "[CH;X4;!R]([CX3]=[CX3])[!H]", 0.85, 363, "Allylic methine"),
    
    # === Heteroatom Oxidation ===
    # Direct oxidation of heteroatoms
    ("S_oxidation_thioether", "[SX2;!$(S=*);!$(S-[!#6])]", 0.85, 310, "Thioether S-oxidation"),
    ("S_oxidation_thiophene", "[sX2;r5]", 0.78, 320, "Thiophene S-oxidation"),
    ("N_oxidation_tert", "[NX3;H0;!$(N=*);!$(N#*);!$(N-[O,S,P])]([#6])([#6])[#6]", 0.80, None, "Tertiary amine N-oxide"),
    ("P_oxidation", "[PX3;!$(P=*)]", 0.75, 300, "Phosphine oxidation"),
    
    # === Epoxidation ===
    ("epoxidation_terminal", "[CH2;X3]=[CH;X3]", 0.72, None, "Terminal alkene epoxidation"),
    ("epoxidation_internal", "[CH;X3]=[CH;X3]", 0.68, None, "Internal alkene"),
    ("epoxidation_styrene", "[CH2;X3]=[CH;X3]c", 0.75, None, "Styrene epoxidation"),
    
    # === α-to-Carbonyl ===
    # Weakened by keto-enol tautomerism
    ("alpha_ketone", "[CH2;X4][CX3](=[OX1])[#6]", 0.68, 395, "α to ketone"),
    ("alpha_aldehyde", "[CH2;X4][CX3](=[OX1])[H]", 0.70, 392, "α to aldehyde"),
    ("alpha_ester", "[CH2;X4][CX3](=[OX1])[OX2]", 0.62, 398, "α to ester"),
    
    # === Aromatic Hydroxylation (generally less favored for CYPs) ===
    ("aromatic_ortho_EDG", "[cH;$(c(c[O,N,S])c)]", 0.55, 472, "Aromatic ortho to EDG"),
    ("aromatic_para_EDG", "[cH;$(cc(c[O,N,S])c)]", 0.50, 475, "Aromatic para to EDG"),
    ("aromatic_general", "[cH]", 0.25, 480, "Unactivated aromatic"),
    
    # === Deactivated Positions ===
    ("halogen_adjacent", "[CH2,CH3;X4][F,Cl,Br,I]", 0.12, 420, "Halogen-adjacent (deactivated)"),
    ("CF3_adjacent", "[CH2,CH3;X4][CX4](F)(F)F", 0.08, 430, "CF3-adjacent (strongly deactivated)"),
    ("nitro_adjacent", "[CH2,CH3;X4]c[N+](=O)[O-]", 0.10, 425, "Nitro-adjacent"),
    ("cyano_adjacent", "[CH2,CH3;X4][CX2]#[NX1]", 0.10, 425, "Cyano-adjacent"),
]


def _compile_reaction_patterns():
    """Compile SMARTS patterns with error handling."""
    if not RDKIT_AVAILABLE:
        return []
    
    compiled = []
    for name, smarts, score, bde, desc in REACTION_PATTERNS:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                compiled.append((name, pattern, score, bde, desc))
        except Exception:
            warnings.warn(f"Failed to compile pattern: {name}")
    return compiled


_COMPILED_REACTION_PATTERNS = None


def get_reaction_patterns():
    global _COMPILED_REACTION_PATTERNS
    if _COMPILED_REACTION_PATTERNS is None:
        _COMPILED_REACTION_PATTERNS = _compile_reaction_patterns()
    return _COMPILED_REACTION_PATTERNS


# ============================================================================
# CYP ISOFORM BINDING PROFILES
# ============================================================================

@dataclass
class CYPBindingProfile:
    """Binding characteristics for a specific CYP isoform."""
    name: str
    pocket_volume: float  # Å³
    pocket_shape: str  # "spherical", "elongated", "bilobed"
    preferred_substrate_mw: Tuple[float, float]  # (min, max) daltons
    preferred_logP: Tuple[float, float]  # (min, max)
    key_residues: List[str] = field(default_factory=list)
    heme_access_angle: float = 90.0  # degrees from normal
    
    # Substrate preference weights
    aromatic_preference: float = 1.0
    basic_nitrogen_preference: float = 1.0
    acidic_preference: float = 1.0
    lipophilicity_preference: float = 1.0
    size_penalty_per_100mw: float = 0.0  # Penalty for MW > optimal


CYP_PROFILES = {
    "CYP1A2": CYPBindingProfile(
        name="CYP1A2",
        pocket_volume=375,
        pocket_shape="planar",
        preferred_substrate_mw=(150, 350),
        preferred_logP=(0.5, 3.5),
        key_residues=["F226", "F260", "I386"],
        heme_access_angle=70,
        aromatic_preference=1.5,  # Strong preference for planar aromatics
        basic_nitrogen_preference=0.8,
        lipophilicity_preference=1.0,
        size_penalty_per_100mw=0.15,
    ),
    "CYP2C9": CYPBindingProfile(
        name="CYP2C9",
        pocket_volume=470,
        pocket_shape="L-shaped",
        preferred_substrate_mw=(250, 500),
        preferred_logP=(1.0, 4.5),
        key_residues=["R108", "N204", "F476"],
        heme_access_angle=85,
        acidic_preference=1.4,  # Prefers acidic substrates
        lipophilicity_preference=1.2,
        size_penalty_per_100mw=0.08,
    ),
    "CYP2C19": CYPBindingProfile(
        name="CYP2C19",
        pocket_volume=420,
        pocket_shape="elongated",
        preferred_substrate_mw=(200, 450),
        preferred_logP=(0.5, 4.0),
        key_residues=["F114", "I205", "V292"],
        heme_access_angle=80,
        aromatic_preference=1.2,
        basic_nitrogen_preference=1.1,
        size_penalty_per_100mw=0.10,
    ),
    "CYP2D6": CYPBindingProfile(
        name="CYP2D6",
        pocket_volume=390,
        pocket_shape="narrow_deep",
        preferred_substrate_mw=(200, 400),
        preferred_logP=(0.5, 3.5),
        key_residues=["D301", "E216", "F483"],  # Asp301 is key for basic N binding
        heme_access_angle=75,
        basic_nitrogen_preference=1.6,  # Strong preference for basic amines
        aromatic_preference=1.2,
        size_penalty_per_100mw=0.12,
    ),
    "CYP3A4": CYPBindingProfile(
        name="CYP3A4",
        pocket_volume=1385,  # Large, flexible pocket
        pocket_shape="bilobed",
        preferred_substrate_mw=(300, 800),
        preferred_logP=(1.5, 5.5),
        key_residues=["R105", "R212", "F304", "I369"],
        heme_access_angle=95,
        lipophilicity_preference=1.4,  # Strong preference for lipophilic substrates
        aromatic_preference=1.1,
        size_penalty_per_100mw=0.03,  # Tolerates large substrates
    ),
}


# ============================================================================
# QUANTUM CHEMISTRY-INFORMED SCORING
# ============================================================================

class QuantumPhysicsScorer:
    """
    Advanced physics-based SoM scorer using quantum chemistry approximations.
    
    This scorer combines:
    1. Pattern-based reactivity (SMARTS matching with BDE estimates)
    2. Electronic structure (Fukui indices, partial charges)
    3. Transition state theory (BEP relation)
    4. CYP-specific binding constraints
    """
    
    def __init__(
        self,
        cyp_isoform: str = "CYP3A4",
        temperature: float = T_REF,
        use_3d: bool = True,
        bep_alpha: float = BEP_ALPHA,
        bep_beta: float = BEP_BETA,
    ):
        self.cyp_isoform = cyp_isoform
        self.cyp_profile = CYP_PROFILES.get(cyp_isoform, CYP_PROFILES["CYP3A4"])
        self.temperature = temperature
        self.use_3d = use_3d
        self.bep_alpha = bep_alpha
        self.bep_beta = bep_beta
        self.patterns = get_reaction_patterns()
        
        # Scoring weights (can be learned)
        self.weight_pattern = 0.25
        self.weight_electronic = 0.20
        self.weight_kinetic = 0.25  # TST-based
        self.weight_steric = 0.15
        self.weight_binding = 0.15
    
    def _compute_pattern_scores(self, mol) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Match reaction patterns and extract scores + BDE estimates."""
        num_atoms = mol.GetNumAtoms()
        pattern_scores = np.zeros(num_atoms, dtype=np.float32)
        bde_estimates = np.full(num_atoms, 420.0, dtype=np.float32)  # Default BDE
        pattern_names = [""] * num_atoms
        
        for name, pattern, score, bde, desc in self.patterns:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                atom_idx = match[0]  # First atom is reactive center
                if score > pattern_scores[atom_idx]:
                    pattern_scores[atom_idx] = score
                    if bde is not None:
                        bde_estimates[atom_idx] = bde
                    pattern_names[atom_idx] = name
        
        return pattern_scores, bde_estimates, pattern_names
    
    def _compute_electronic_scores(
        self,
        mol,
        xtb_charges: Optional[np.ndarray] = None,
        xtb_fukui: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute electronic reactivity scores.
        
        The key insight: CYP oxidation involves electrophilic attack by
        the Fe(IV)=O species. Sites with high electron density (negative
        charge, high Fukui f+) are more reactive.
        """
        num_atoms = mol.GetNumAtoms()
        scores = np.zeros(num_atoms, dtype=np.float32)
        
        if xtb_fukui is not None and len(xtb_fukui) == num_atoms:
            # Fukui f+ directly indicates susceptibility to electrophilic attack
            # Normalize to [0, 1]
            f_max = xtb_fukui.max()
            if f_max > 0:
                scores = np.clip(xtb_fukui / f_max, 0, 1)
        elif xtb_charges is not None and len(xtb_charges) == num_atoms:
            # Approximate: more negative charge = more electron-rich
            # Map [-0.5, 0.5] to [1, 0] (more negative = higher score)
            scores = np.clip(-xtb_charges + 0.5, 0, 1)
        else:
            # Fallback: Gasteiger charges
            try:
                AllChem.ComputeGasteigerCharges(mol)
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    charge = float(atom.GetDoubleProp("_GasteigerCharge"))
                    if not np.isfinite(charge):
                        charge = 0.0
                    # More negative = higher score
                    scores[idx] = np.clip(-charge * 2.0 + 0.5, 0, 1)
            except Exception:
                scores = np.full(num_atoms, 0.5, dtype=np.float32)
        
        return scores
    
    def _compute_kinetic_scores(
        self,
        bde_estimates: np.ndarray,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """
        Estimate relative reaction rates using transition state theory.
        
        Bell-Evans-Polanyi principle:
        ΔH‡ = α × ΔH_rxn + β
        
        For H-abstraction by Fe(IV)=O:
        ΔH_rxn ≈ BDE(C-H) - BDE(Fe-H) ≈ BDE(C-H) - 280 kJ/mol
        
        Rate ∝ exp(-ΔG‡/RT) ≈ exp(-ΔH‡/RT) for similar ΔS‡
        """
        if temperature is None:
            temperature = self.temperature
        
        # Estimate activation enthalpy
        # Reference: Fe-H bond strength ~280 kJ/mol
        delta_h_rxn = bde_estimates - 280.0
        delta_h_act = self.bep_alpha * delta_h_rxn + self.bep_beta
        
        # Clamp to reasonable range [10, 120] kJ/mol
        delta_h_act = np.clip(delta_h_act, 10.0, 120.0)
        
        # Relative rate (log scale)
        # k ∝ exp(-ΔH‡/RT)
        log_rate = -delta_h_act / (R_GAS * temperature)
        
        # Normalize to [0, 1]
        log_rate_norm = (log_rate - log_rate.min()) / (log_rate.max() - log_rate.min() + 1e-8)
        
        return log_rate_norm.astype(np.float32)
    
    def _compute_steric_scores(
        self,
        mol,
        conf_id: int = -1,
    ) -> np.ndarray:
        """
        Compute steric accessibility scores.
        
        Uses solvent accessible surface area (SASA) as a proxy for
        accessibility to the CYP active site.
        """
        num_atoms = mol.GetNumAtoms()
        scores = np.ones(num_atoms, dtype=np.float32)
        
        if mol.GetNumConformers() == 0:
            # No 3D - use topological approximation
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                # Fewer heavy neighbors = more accessible
                heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
                scores[idx] = np.exp(-0.2 * heavy_neighbors)
            return scores
        
        try:
            # SASA calculation
            radii = rdFreeSASA.classifyAtoms(mol)
            
            # Per-atom SASA
            atom_sasa = np.zeros(num_atoms)
            for i in range(num_atoms):
                if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:  # Heavy atoms only
                    atom_sasa[i] = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id, atomIdx=i)
            
            # Normalize
            max_sasa = atom_sasa.max()
            if max_sasa > 0:
                scores = (atom_sasa / max_sasa).astype(np.float32)
            
        except Exception:
            pass
        
        return scores
    
    def _compute_binding_scores(
        self,
        mol,
        conf_id: int = -1,
        heme_center: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute CYP binding favorability scores.
        
        Considers:
        1. Distance to heme (optimal ~5 Å)
        2. CYP isoform preferences
        """
        num_atoms = mol.GetNumAtoms()
        scores = np.ones(num_atoms, dtype=np.float32)
        
        # Apply CYP-specific preferences
        profile = self.cyp_profile
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            
            # Basic nitrogen preference (CYP2D6)
            if atom.GetAtomicNum() == 7:
                # Check if basic
                is_basic = atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() > 0
                if not any(n.GetAtomicNum() == 8 and n.GetIsAromatic() == False 
                          for n in atom.GetNeighbors()):
                    if is_basic:
                        scores[idx] *= profile.basic_nitrogen_preference
            
            # Aromatic preference (CYP1A2, CYP2C19)
            if atom.GetIsAromatic():
                scores[idx] *= profile.aromatic_preference
            
            # Lipophilicity (CYP3A4) - carbons far from heteroatoms
            if atom.GetAtomicNum() == 6:
                heteroatom_neighbors = sum(
                    1 for n in atom.GetNeighbors()
                    if n.GetAtomicNum() in [7, 8, 16]
                )
                if heteroatom_neighbors == 0:
                    scores[idx] *= profile.lipophilicity_preference
        
        # Distance to heme if 3D available
        if mol.GetNumConformers() > 0 and heme_center is not None:
            conf = mol.GetConformer(conf_id)
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                pos = np.array(conf.GetAtomPosition(idx))
                dist = np.linalg.norm(pos - heme_center)
                
                # Gaussian penalty for distance from optimal
                dist_score = np.exp(-0.5 * ((dist - HEME_OPTIMAL_DISTANCE) / HEME_DISTANCE_SIGMA) ** 2)
                scores[idx] *= dist_score
        
        return scores
    
    def score_molecule(
        self,
        smiles: str,
        *,
        xtb_bde: Optional[np.ndarray] = None,
        xtb_charges: Optional[np.ndarray] = None,
        xtb_fukui: Optional[np.ndarray] = None,
        heme_center: Optional[np.ndarray] = None,
        return_profiles: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[AtomReactivityProfile]]]:
        """
        Score all atoms in a molecule for SoM likelihood.
        
        Args:
            smiles: SMILES string
            xtb_bde: Per-atom BDE from xTB calculations
            xtb_charges: Per-atom partial charges from xTB
            xtb_fukui: Per-atom Fukui f+ indices from xTB
            heme_center: 3D coordinates of heme center (optional)
            return_profiles: If True, return detailed AtomReactivityProfile objects
            
        Returns:
            Dict containing scores and optional profiles
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        
        # Generate 3D if needed
        if self.use_3d and mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        
        # Heavy atom mask
        is_heavy = np.array([
            1.0 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1 else 0.0
            for i in range(num_atoms)
        ], dtype=np.float32)
        
        # Compute component scores
        pattern_scores, bde_estimates, pattern_names = self._compute_pattern_scores(mol)
        
        # Use xTB BDE if available
        if xtb_bde is not None and len(xtb_bde) == num_atoms:
            # Only override where xTB succeeded (reasonable values)
            valid_xtb = (xtb_bde > 200) & (xtb_bde < 550)
            bde_estimates = np.where(valid_xtb, xtb_bde, bde_estimates)
        
        electronic_scores = self._compute_electronic_scores(mol, xtb_charges, xtb_fukui)
        kinetic_scores = self._compute_kinetic_scores(bde_estimates)
        steric_scores = self._compute_steric_scores(mol)
        binding_scores = self._compute_binding_scores(mol, heme_center=heme_center)
        
        # Combine scores
        combined = (
            self.weight_pattern * pattern_scores +
            self.weight_electronic * electronic_scores +
            self.weight_kinetic * kinetic_scores +
            self.weight_steric * steric_scores +
            self.weight_binding * binding_scores
        )
        
        # Mask non-heavy atoms
        final_scores = combined * is_heavy
        
        # Normalize
        max_score = final_scores.max()
        if max_score > 0:
            final_scores = final_scores / max_score
        
        result = {
            "final_scores": final_scores,
            "pattern_scores": pattern_scores,
            "pattern_names": pattern_names,
            "bde_estimates": bde_estimates,
            "electronic_scores": electronic_scores,
            "kinetic_scores": kinetic_scores,
            "steric_scores": steric_scores,
            "binding_scores": binding_scores,
            "is_heavy": is_heavy,
            "cyp_isoform": self.cyp_isoform,
        }
        
        if return_profiles:
            profiles = []
            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                if atom.GetAtomicNum() <= 1:
                    continue
                
                profile = AtomReactivityProfile(
                    atom_index=i,
                    atom_symbol=atom.GetSymbol(),
                    bde_estimate=float(bde_estimates[i]),
                    bde_source="xtb" if xtb_bde is not None else "pattern",
                    num_hydrogens=atom.GetTotalNumHs(),
                    reactivity_pattern=pattern_names[i],
                    pattern_score=float(pattern_scores[i]),
                    electronic_score=float(electronic_scores[i]),
                    steric_score=float(steric_scores[i]),
                    kinetic_score=float(kinetic_scores[i]),
                    binding_score=float(binding_scores[i]),
                    final_score=float(final_scores[i]),
                )
                profiles.append(profile)
            
            result["profiles"] = profiles
        
        return result


# ============================================================================
# TORCH MODULE FOR LEARNABLE PHYSICS INTEGRATION
# ============================================================================

if TORCH_AVAILABLE:
    
    class LearnablePhysicsIntegration(nn.Module):
        """
        Neural network that learns to optimally combine physics features
        with ML predictions.
        
        Key innovation: Instead of fixed weights, the network learns
        context-dependent weighting based on:
        - Atom type and environment
        - Confidence of physics predictions
        - Molecular context
        """
        
        def __init__(
            self,
            ml_dim: int = 1,
            physics_dim: int = 8,  # pattern, electronic, kinetic, steric, binding, bde, confidence, cyp_match
            atom_feature_dim: int = 128,
            hidden_dim: int = 64,
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.physics_dim = physics_dim
            
            # Physics feature encoder
            self.physics_encoder = nn.Sequential(
                nn.Linear(physics_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Cross-attention: ML attends to physics
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            
            # Atom feature projection
            self.atom_proj = nn.Linear(atom_feature_dim, hidden_dim)
            
            # Gating network
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2 + ml_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),  # [ml_weight, physics_weight, residual_weight]
            )
            
            # Residual prediction head
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            
            # Physics-to-logit projection
            self.physics_to_logit = nn.Linear(hidden_dim, 1)
        
        def forward(
            self,
            ml_logits: torch.Tensor,           # (N,) or (N, 1)
            physics_features: torch.Tensor,     # (N, physics_dim)
            atom_features: torch.Tensor,        # (N, hidden_dim) or (N, atom_feature_dim)
            batch_index: Optional[torch.Tensor] = None,  # (N,)
            candidate_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            if ml_logits.dim() == 1:
                ml_logits = ml_logits.unsqueeze(-1)
            
            N = ml_logits.size(0)
            device = ml_logits.device
            
            # Encode physics features
            physics_encoded = self.physics_encoder(physics_features)  # (N, hidden_dim)
            
            # Project atom features
            atom_encoded = self.atom_proj(atom_features)  # (N, hidden_dim)
            
            # Self-attention over atoms (within each molecule)
            # For simplicity, treat all atoms as one batch
            # In full implementation, would use batch_index for proper batching
            
            # Cross-attention: atom features query physics features
            attn_out, _ = self.cross_attention(
                query=atom_encoded.unsqueeze(0),
                key=physics_encoded.unsqueeze(0),
                value=physics_encoded.unsqueeze(0),
            )
            attn_out = attn_out.squeeze(0)  # (N, hidden_dim)
            
            # Compute gating weights
            gate_input = torch.cat([atom_encoded, attn_out, ml_logits], dim=-1)
            gate_logits = self.gate_net(gate_input)
            gate_weights = F.softmax(gate_logits, dim=-1)
            
            ml_weight = gate_weights[:, 0:1]
            physics_weight = gate_weights[:, 1:2]
            residual_weight = gate_weights[:, 2:3]
            
            # Physics contribution
            physics_logit = self.physics_to_logit(physics_encoded)
            
            # Residual (learned correction)
            residual_input = torch.cat([atom_encoded, attn_out], dim=-1)
            residual = self.residual_head(residual_input)
            
            # Combine
            ensemble_logits = (
                ml_weight * ml_logits +
                physics_weight * physics_logit +
                residual_weight * residual
            )
            
            # Apply candidate mask
            if candidate_mask is not None:
                mask = candidate_mask.float().unsqueeze(-1)
                ensemble_logits = ensemble_logits * mask + (-100.0) * (1 - mask)
            
            return {
                "ensemble_logits": ensemble_logits.squeeze(-1),
                "ml_weight": ml_weight.squeeze(-1),
                "physics_weight": physics_weight.squeeze(-1),
                "residual_weight": residual_weight.squeeze(-1),
                "physics_logit": physics_logit.squeeze(-1),
                "residual": residual.squeeze(-1),
            }


def test_quantum_scorer():
    """Quick test of the quantum physics scorer."""
    scorer = QuantumPhysicsScorer(cyp_isoform="CYP3A4")
    
    # Test molecules
    test_cases = [
        ("Midazolam", "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Omeprazole", "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ]
    
    print("=" * 70)
    print("QUANTUM PHYSICS SCORER TEST")
    print("=" * 70)
    
    for name, smiles in test_cases:
        print(f"\n{name}:")
        try:
            result = scorer.score_molecule(smiles, return_profiles=True)
            
            # Top 3 predictions
            scores = result["final_scores"]
            top_indices = np.argsort(-scores)[:5]
            
            print(f"  Top-5 predictions:")
            for rank, idx in enumerate(top_indices):
                pattern = result["pattern_names"][idx]
                bde = result["bde_estimates"][idx]
                print(f"    {rank+1}. Atom {idx}: score={scores[idx]:.3f}, "
                      f"pattern={pattern or 'none'}, BDE={bde:.0f} kJ/mol")
                
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_quantum_scorer()
