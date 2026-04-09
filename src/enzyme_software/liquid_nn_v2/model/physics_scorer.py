"""Pure Physics/Chemistry Rule-Based SoM Scorer.

This module implements a TRAINING-FREE approach to SoM prediction based on
first-principles understanding of CYP450 metabolism:

1. Reaction Site Rules (from medicinal chemistry knowledge):
   - Benzylic carbons: Highly reactive (C-H BDE ~85 kcal/mol, stabilized radical)
   - Allylic carbons: Highly reactive (similar to benzylic)
   - N-methyl groups: Common N-dealkylation site
   - O-methyl groups: Common O-demethylation site  
   - Alpha to heteroatom: Activated by adjacent N or O
   - Aromatic C-H: Low reactivity (strong C-H bond)
   - Halogen-adjacent: Deactivated

2. CYP3A4-Specific Factors:
   - Large, flexible binding pocket (handles big substrates)
   - Heme distance constraint (~4-6 Å optimal)
   - Preference for lipophilic substrates

3. Steric/Accessibility Factors:
   - Exposed atoms more accessible
   - Crowded positions blocked

The key insight: We don't need to LEARN these rules - they're KNOWN.
We just need to apply them correctly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except Exception:
    Chem = None
    AllChem = None
    Descriptors = None


# ============================================================================
# CHEMISTRY RULES (from literature and medicinal chemistry knowledge)
# ============================================================================

# SMARTS patterns with reactivity scores based on experimental data
# Score represents relative rate of oxidation (higher = more reactive)
REACTIVITY_RULES: List[Tuple[str, str, float, str]] = [
    # High reactivity sites (>0.8)
    ("o_demethyl_aromatic", "[CH3]O[c]",           0.95, "O-demethylation of aromatic methoxy"),
    ("benzylic_ch2",        "[CH2;!R][c]",         0.92, "Benzylic methylene"),  
    ("benzylic_ch3",        "[CH3][c]",            0.90, "Benzylic methyl"),
    ("n_methyl",            "[CH3][NX3]",          0.88, "N-demethylation"),
    ("allylic",             "[CH2,CH3][C]=[C]",    0.85, "Allylic oxidation"),
    ("alpha_n_ch2",         "[CH2][NX3]",          0.82, "Alpha to nitrogen (CH2)"),
    ("alpha_o_ch2",         "[CH2][OX2]",          0.80, "Alpha to oxygen (CH2)"),
    
    # Medium reactivity sites (0.5-0.8)
    ("s_oxidation",         "[SX2;!$([S]=*)]",     0.78, "Thioether S-oxidation"),
    ("n_oxidation",         "[NX3;H0;!$([N+])]",   0.75, "Tertiary amine N-oxidation"),
    ("ring_n_6",            "[NX3;r6;H0]",         0.72, "Piperidine/piperazine N"),
    ("thiophene_s",         "[sX2;r5]",            0.70, "Thiophene S-oxidation"),
    ("epoxidation",         "[CX3]=[CX3]",         0.68, "Alkene epoxidation"),
    ("carbonyl_alpha",      "[CH2,CH3][CX3]=O",    0.65, "Alpha to carbonyl"),
    
    # Low reactivity sites (<0.5)
    ("aromatic_ch",         "[cH]",                0.20, "Aromatic C-H (deactivated)"),
    ("halogen_adjacent",    "[CH2,CH3][F,Cl,Br,I]",0.15, "Halogen-adjacent (deactivated)"),
    ("quaternary",          "[CX4;H0]",            0.05, "Quaternary carbon (no H)"),
]

# Atom type base reactivity (when no SMARTS matches)
ATOM_BASE_REACTIVITY: Dict[int, float] = {
    6: 0.30,   # Carbon (aliphatic)
    7: 0.50,   # Nitrogen
    8: 0.25,   # Oxygen (low, not typically a SoM)
    16: 0.60,  # Sulfur
}


def _compile_patterns():
    """Compile SMARTS patterns once."""
    if Chem is None:
        return []
    compiled = []
    for name, smarts, score, description in REACTIVITY_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            compiled.append((name, pattern, score, description))
    return compiled


_COMPILED_PATTERNS = None


def get_compiled_patterns():
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = _compile_patterns()
    return _COMPILED_PATTERNS


class PhysicsSoMScorer:
    """
    Pure chemistry/physics-based SoM scorer.
    
    NO TRAINING REQUIRED - uses known chemistry rules.
    """
    
    def __init__(
        self,
        accessibility_weight: float = 0.25,
        hydrogen_count_weight: float = 0.15,
        steric_penalty_weight: float = 0.20,
    ):
        self.accessibility_weight = accessibility_weight
        self.hydrogen_count_weight = hydrogen_count_weight
        self.steric_penalty_weight = steric_penalty_weight
        self.patterns = get_compiled_patterns()
    
    def score_molecule(
        self,
        smiles: str,
        *,
        atom_coordinates: Optional[np.ndarray] = None,
        heme_center: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Score all atoms in a molecule for SoM likelihood.
        
        Returns:
            Dict with:
            - scores: (N,) array of SoM scores [0, 1]
            - pattern_matches: (N,) array of which pattern matched
            - reactivity_base: (N,) base reactivity from SMARTS
            - accessibility: (N,) accessibility factor
            - final_scores: (N,) combined scores
        """
        if Chem is None:
            raise RuntimeError("RDKit not available")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        num_heavy = mol.GetNumHeavyAtoms()
        
        # Initialize arrays
        reactivity_base = np.zeros(num_atoms, dtype=np.float32)
        pattern_matches = np.full(num_atoms, "", dtype=object)
        hydrogen_count = np.zeros(num_atoms, dtype=np.float32)
        is_heavy = np.zeros(num_atoms, dtype=np.float32)
        
        # Get atom properties
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            
            # Mark heavy atoms
            if atomic_num > 1:
                is_heavy[idx] = 1.0
            
            # Base reactivity from atom type
            reactivity_base[idx] = ATOM_BASE_REACTIVITY.get(atomic_num, 0.1)
            
            # Count attached hydrogens (more H = more reactive for oxidation)
            hydrogen_count[idx] = atom.GetTotalNumHs()
        
        # Apply SMARTS patterns (highest matching score wins)
        for name, pattern, score, description in self.patterns:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # First atom in match is typically the reactive center
                atom_idx = match[0]
                if score > reactivity_base[atom_idx]:
                    reactivity_base[atom_idx] = score
                    pattern_matches[atom_idx] = name
        
        # Compute accessibility based on 3D coordinates
        accessibility = np.ones(num_atoms, dtype=np.float32)
        if atom_coordinates is not None and atom_coordinates.shape[0] == num_atoms:
            # Simple accessibility: exposed atoms have fewer close neighbors
            coords = np.asarray(atom_coordinates, dtype=np.float32)
            for i in range(num_atoms):
                if is_heavy[i] < 0.5:
                    continue
                # Count atoms within 4 Angstroms
                distances = np.linalg.norm(coords - coords[i], axis=1)
                nearby = np.sum((distances > 0.1) & (distances < 4.0))
                # Fewer nearby atoms = more accessible
                accessibility[i] = np.exp(-0.15 * nearby)
            
            # Distance to heme center penalty
            if heme_center is not None:
                heme_center = np.asarray(heme_center, dtype=np.float32).reshape(1, 3)
                heme_distances = np.linalg.norm(coords - heme_center, axis=1)
                # Optimal distance is 4-6 Angstroms
                distance_factor = np.exp(-0.5 * ((heme_distances - 5.0) / 2.0) ** 2)
                accessibility = accessibility * distance_factor
        
        # Hydrogen count factor (more H = more reactive)
        h_factor = np.clip(hydrogen_count / 3.0, 0.0, 1.0)
        
        # Combine factors
        final_scores = (
            reactivity_base 
            + self.accessibility_weight * accessibility
            + self.hydrogen_count_weight * h_factor
        )
        
        # Mask non-heavy atoms
        final_scores = final_scores * is_heavy
        
        # Normalize to [0, 1]
        max_score = final_scores.max()
        if max_score > 0:
            final_scores = final_scores / max_score
        
        return {
            "scores": final_scores,
            "pattern_matches": pattern_matches,
            "reactivity_base": reactivity_base,
            "accessibility": accessibility,
            "hydrogen_count": hydrogen_count,
            "is_heavy": is_heavy,
            "final_scores": final_scores,
        }
    
    def predict_top_k(
        self,
        smiles: str,
        k: int = 3,
        **kwargs,
    ) -> List[Tuple[int, float, str]]:
        """
        Predict top-k SoM sites.
        
        Returns:
            List of (atom_idx, score, pattern_name) tuples
        """
        result = self.score_molecule(smiles, **kwargs)
        scores = result["final_scores"]
        patterns = result["pattern_matches"]
        is_heavy = result["is_heavy"]
        
        # Get indices of heavy atoms sorted by score
        heavy_indices = np.where(is_heavy > 0.5)[0]
        sorted_indices = heavy_indices[np.argsort(-scores[heavy_indices])]
        
        top_k = []
        for idx in sorted_indices[:k]:
            top_k.append((int(idx), float(scores[idx]), str(patterns[idx])))
        
        return top_k


def evaluate_physics_scorer_on_dataset(
    dataset: List[Dict],
    *,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the physics scorer on a dataset.
    
    Dataset should be list of dicts with 'smiles' and 'site_labels' keys.
    """
    scorer = PhysicsSoMScorer()
    
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    for item in dataset:
        smiles = item.get("smiles", "")
        true_sites = item.get("site_labels", [])
        
        if not smiles or not true_sites:
            continue
        
        try:
            result = scorer.score_molecule(smiles)
            scores = result["final_scores"]
            is_heavy = result["is_heavy"]
            
            # Get top predictions
            heavy_indices = np.where(is_heavy > 0.5)[0]
            if len(heavy_indices) == 0:
                continue
                
            sorted_indices = heavy_indices[np.argsort(-scores[heavy_indices])]
            top1_pred = int(sorted_indices[0]) if len(sorted_indices) > 0 else -1
            top3_pred = set(int(idx) for idx in sorted_indices[:3])
            
            # Convert true sites to set
            true_site_set = set(int(s) for s in true_sites if isinstance(s, (int, float)))
            
            if top1_pred in true_site_set:
                top1_correct += 1
            if true_site_set & top3_pred:
                top3_correct += 1
            
            total += 1
            
        except Exception as e:
            if verbose:
                print(f"Error processing {smiles[:50]}: {e}")
            continue
    
    return {
        "top1_accuracy": top1_correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total_evaluated": total,
    }


if __name__ == "__main__":
    # Quick test
    scorer = PhysicsSoMScorer()
    
    # Test on midazolam (known CYP3A4 substrate)
    # Main sites: 1'-hydroxylation (benzylic) and 4-hydroxylation
    test_smiles = "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"  # Midazolam
    
    result = scorer.score_molecule(test_smiles)
    print(f"Scores: {result['final_scores']}")
    print(f"Top predictions: {scorer.predict_top_k(test_smiles, k=5)}")
