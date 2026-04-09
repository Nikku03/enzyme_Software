#!/usr/bin/env python3
"""
CYP3A4 Dataset Curation Pipeline

This script performs rigorous quality control on SoM labels:
1. Chemical validity checks (can this atom actually be oxidized?)
2. Physics plausibility scoring (does chemistry agree?)
3. Source reliability weighting
4. Multi-site disambiguation
5. Negative sampling for better discrimination

Output:
- curated_cyp3a4_gold.json: High-confidence training set
- curated_cyp3a4_silver.json: Medium-confidence, usable with caution
- manual_review_queue.json: Needs human expert review
- rejected_labels.json: Invalid labels with reasons
- curation_report.md: Detailed analysis

Author: Claude (Anthropic)
Date: 2026-04-09
"""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import sys

import numpy as np

# Add project to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Chemical checks will be limited.")


# ============================================================================
# CHEMISTRY KNOWLEDGE BASE
# ============================================================================

# Common CYP3A4 reaction types and their valid atom targets
CYP3A4_REACTION_TARGETS = {
    "hydroxylation": {
        "valid_atoms": {6},  # Carbon only
        "requires_hydrogen": True,
        "description": "C-H hydroxylation requires carbon with attached hydrogen"
    },
    "n_oxidation": {
        "valid_atoms": {7},  # Nitrogen
        "requires_hydrogen": False,
        "description": "N-oxidation of tertiary amines"
    },
    "s_oxidation": {
        "valid_atoms": {16},  # Sulfur
        "requires_hydrogen": False,
        "description": "S-oxidation of thioethers"
    },
    "o_dealkylation": {
        "valid_atoms": {6},  # The carbon being removed
        "requires_hydrogen": True,
        "description": "O-dealkylation targets the alpha carbon"
    },
    "n_dealkylation": {
        "valid_atoms": {6},  # The carbon being removed
        "requires_hydrogen": True,
        "description": "N-dealkylation targets the alpha carbon"
    },
    "epoxidation": {
        "valid_atoms": {6},  # Alkene carbons
        "requires_hydrogen": False,
        "description": "Epoxidation of C=C double bonds"
    },
}

# Atoms that are NEVER valid SoM targets
IMPOSSIBLE_SOM_ATOMS = {
    9: "Fluorine - not metabolized by CYP",
    17: "Chlorine - not a typical CYP SoM (dehalogenation rare)",
    35: "Bromine - not a typical CYP SoM",
    53: "Iodine - not a typical CYP SoM",
}

# Suspicious but not impossible
SUSPICIOUS_SOM_ATOMS = {
    8: "Oxygen - rarely the actual SoM (usually adjacent carbon)",
    15: "Phosphorus - very rare CYP substrate",
}

# Source reliability tiers
SOURCE_RELIABILITY = {
    # Tier 1: High confidence
    "validated": 1.0,
    "literature": 0.95,
    "AZ120": 0.90,  # AstraZeneca curated dataset
    
    # Tier 2: Medium confidence
    "DrugBank": 0.75,
    "SuperCYP": 0.70,
    
    # Tier 3: Lower confidence (database inference)
    "MetXBioDB": 0.60,
    "metxbiodb": 0.60,
    "CYP_DBs_external": 0.55,
    "MetaPred": 0.50,
    
    # Unknown
    "unknown": 0.40,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AtomAssessment:
    """Assessment of a single atom as potential SoM."""
    atom_idx: int
    atomic_num: int
    symbol: str
    num_hydrogens: int
    is_aromatic: bool
    hybridization: str
    degree: int
    in_ring: bool
    ring_size: int
    
    # Validity flags
    is_chemically_valid: bool = True
    validity_reason: str = ""
    
    # Physics scoring
    physics_score: float = 0.0
    physics_pattern: str = ""
    physics_rank: int = 0
    
    # Final assessment
    confidence: float = 0.0
    flags: List[str] = field(default_factory=list)


@dataclass  
class MoleculeAssessment:
    """Full assessment of a molecule's SoM labels."""
    drug_id: str
    name: str
    smiles: str
    primary_cyp: str
    source: str
    source_reliability: float
    
    # Original labels
    original_sites: List[int] = field(default_factory=list)
    
    # Atom assessments
    atom_assessments: Dict[int, AtomAssessment] = field(default_factory=dict)
    
    # Curated results
    valid_sites: List[int] = field(default_factory=list)
    suspicious_sites: List[int] = field(default_factory=list)
    invalid_sites: List[int] = field(default_factory=list)
    
    # Physics predictions for comparison
    physics_top1: int = -1
    physics_top3: List[int] = field(default_factory=list)
    physics_agrees: bool = False
    
    # Overall assessment
    overall_quality: str = "unknown"  # gold, silver, bronze, reject, review
    quality_score: float = 0.0
    flags: List[str] = field(default_factory=list)
    recommendation: str = ""


# ============================================================================
# CHEMICAL VALIDITY CHECKER
# ============================================================================

class ChemicalValidityChecker:
    """Check if labeled SoM atoms are chemically plausible."""
    
    def __init__(self):
        self.stats = defaultdict(int)
    
    def assess_atom(self, mol, atom_idx: int, reaction_types: List[str] = None) -> AtomAssessment:
        """Assess a single atom for SoM validity."""
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Count hydrogens correctly (works with both implicit and explicit H)
        num_h = atom.GetTotalNumHs()
        # If explicit H's are present, count H neighbors
        if num_h == 0:
            num_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        
        # Get heavy atom degree (excluding H neighbors)
        heavy_degree = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
        
        assessment = AtomAssessment(
            atom_idx=atom_idx,
            atomic_num=atom.GetAtomicNum(),
            symbol=atom.GetSymbol(),
            num_hydrogens=num_h,
            is_aromatic=atom.GetIsAromatic(),
            hybridization=str(atom.GetHybridization()),
            degree=heavy_degree,  # Use heavy atom degree
            in_ring=atom.IsInRing(),
            ring_size=self._get_smallest_ring_size(mol, atom_idx),
        )
        
        # Check 1: Impossible atoms
        if assessment.atomic_num in IMPOSSIBLE_SOM_ATOMS:
            assessment.is_chemically_valid = False
            assessment.validity_reason = IMPOSSIBLE_SOM_ATOMS[assessment.atomic_num]
            assessment.flags.append("IMPOSSIBLE_ATOM")
            self.stats["impossible_atom"] += 1
            return assessment
        
        # Check 2: Suspicious atoms
        if assessment.atomic_num in SUSPICIOUS_SOM_ATOMS:
            assessment.flags.append("SUSPICIOUS_ATOM")
            assessment.validity_reason = SUSPICIOUS_SOM_ATOMS[assessment.atomic_num]
            self.stats["suspicious_atom"] += 1
        
        # Check 3: Carbon without hydrogen (for hydroxylation)
        if assessment.atomic_num == 6 and assessment.num_hydrogens == 0:
            # Quaternary carbon - cannot be hydroxylated
            assessment.is_chemically_valid = False
            assessment.validity_reason = "Quaternary carbon (no H) cannot be hydroxylated"
            assessment.flags.append("NO_HYDROGEN")
            self.stats["quaternary_carbon"] += 1
            return assessment
        
        # Check 4: Terminal atoms with degree 1 (often labeling errors)
        if assessment.degree == 1 and assessment.atomic_num == 8:
            # Terminal oxygen - this is often the metabolite oxygen, not SoM
            assessment.is_chemically_valid = False
            assessment.validity_reason = "Terminal oxygen is metabolite product, not SoM"
            assessment.flags.append("TERMINAL_OXYGEN")
            self.stats["terminal_oxygen"] += 1
            return assessment
        
        # Check 5: Ether oxygens
        if assessment.atomic_num == 8 and assessment.degree == 2:
            # Check if it's an ether (C-O-C)
            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            if neighbors.count(6) == 2:
                # Ether oxygen - SoM should be the adjacent carbon
                assessment.flags.append("ETHER_OXYGEN")
                assessment.validity_reason = "Ether O - SoM is likely adjacent carbon"
                self.stats["ether_oxygen"] += 1
        
        # Check 6: Aromatic carbon reactivity
        if assessment.atomic_num == 6 and assessment.is_aromatic:
            # Aromatic C-H bonds are quite stable
            if assessment.num_hydrogens > 0:
                assessment.flags.append("AROMATIC_CH")
                # Not invalid, just lower reactivity
                self.stats["aromatic_ch"] += 1
        
        # Check 7: Nitrogen checks
        if assessment.atomic_num == 7:
            # N-oxidation requires lone pair (tertiary or secondary amine)
            if assessment.degree == 1:
                assessment.flags.append("PRIMARY_AMINE")
                # Primary amines can still be N-oxidized
            self.stats["nitrogen_som"] += 1
        
        # Check 8: Sulfur checks  
        if assessment.atomic_num == 16:
            # S-oxidation is common for thioethers
            if assessment.degree >= 2:
                assessment.flags.append("THIOETHER")
                self.stats["sulfur_som"] += 1
        
        return assessment
    
    def _get_smallest_ring_size(self, mol, atom_idx: int) -> int:
        """Get smallest ring containing this atom."""
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        min_size = 0
        for ring in atom_rings:
            if atom_idx in ring:
                if min_size == 0 or len(ring) < min_size:
                    min_size = len(ring)
        
        return min_size
    
    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)


# ============================================================================
# PHYSICS PLAUSIBILITY SCORER
# ============================================================================

class PhysicsPlausibilityScorer:
    """Score atoms using chemistry rules and compare with labels."""
    
    # SMARTS patterns with reactivity scores
    REACTIVITY_PATTERNS = [
        # High reactivity (>0.8)
        ("o_demethyl", "[CH3][OX2][c,C]", 0.95),
        ("benzylic_ch2", "[CH2;!R][c]", 0.93),
        ("benzylic_ch3", "[CH3][c]", 0.90),
        ("n_methyl", "[CH3][NX3]", 0.90),
        ("allylic", "[CH2,CH3][C]=[C]", 0.85),
        ("alpha_n", "[CH2,CH][NX3]", 0.82),
        ("alpha_o", "[CH2,CH][OX2]", 0.78),
        
        # Medium reactivity (0.5-0.8)
        ("thioether", "[SX2]([C])([C])", 0.80),
        ("tert_amine_n", "[NX3;H0]([C])([C])[C]", 0.75),
        ("sec_amine_n", "[NX3;H1]([C])[C]", 0.70),
        ("piperidine_alpha", "[CH2;r6]@[NX3;r6]", 0.78),
        ("alkene", "[CX3]=[CX3]", 0.65),
        
        # Lower reactivity (0.2-0.5)
        ("aromatic_ch", "[cH]", 0.25),
        ("aliphatic_ch2", "[CH2]", 0.40),
        ("aliphatic_ch3", "[CH3]", 0.35),
        ("aliphatic_ch", "[CH]", 0.45),
    ]
    
    def __init__(self):
        self.compiled_patterns = []
        if RDKIT_AVAILABLE:
            for name, smarts, score in self.REACTIVITY_PATTERNS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self.compiled_patterns.append((name, pattern, score))
    
    def score_molecule(self, smiles: str) -> Dict[str, Any]:
        """Score all atoms in a molecule."""
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Could not parse SMILES: {smiles}"}
        
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        
        # Initialize scores
        scores = np.zeros(num_atoms, dtype=np.float32)
        patterns = [""] * num_atoms
        is_heavy = np.zeros(num_atoms, dtype=bool)
        
        # Mark heavy atoms and assign base scores
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            
            if atomic_num > 1:
                is_heavy[idx] = True
                
                # Base score by atom type
                if atomic_num == 6:
                    num_h = atom.GetTotalNumHs()
                    if num_h > 0:
                        scores[idx] = 0.3 + 0.1 * min(num_h, 3)
                elif atomic_num == 7:
                    scores[idx] = 0.5
                elif atomic_num == 16:
                    scores[idx] = 0.6
                elif atomic_num == 8:
                    scores[idx] = 0.1  # Oxygen rarely SoM
        
        # Apply SMARTS patterns
        for name, pattern, score in self.compiled_patterns:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                atom_idx = match[0]
                if score > scores[atom_idx]:
                    scores[atom_idx] = score
                    patterns[atom_idx] = name
        
        # Get rankings
        heavy_indices = np.where(is_heavy)[0]
        if len(heavy_indices) == 0:
            return {"error": "No heavy atoms found"}
        
        # Sort by score descending
        sorted_indices = heavy_indices[np.argsort(-scores[heavy_indices])]
        
        # Assign ranks
        ranks = np.zeros(num_atoms, dtype=np.int32)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        return {
            "scores": scores,
            "patterns": patterns,
            "ranks": ranks,
            "is_heavy": is_heavy,
            "top1": int(sorted_indices[0]) if len(sorted_indices) > 0 else -1,
            "top3": [int(idx) for idx in sorted_indices[:3]],
            "top5": [int(idx) for idx in sorted_indices[:5]],
        }


# ============================================================================
# MAIN CURATION PIPELINE
# ============================================================================

class DatasetCurator:
    """Main curation pipeline."""
    
    def __init__(self, cyp_filter: str = "CYP3A4"):
        self.cyp_filter = cyp_filter
        self.validity_checker = ChemicalValidityChecker()
        self.physics_scorer = PhysicsPlausibilityScorer()
        
        # Results
        self.assessments: List[MoleculeAssessment] = []
        self.gold: List[Dict] = []
        self.silver: List[Dict] = []
        self.review_queue: List[Dict] = []
        self.rejected: List[Dict] = []
        
        # Statistics
        self.stats = defaultdict(int)
    
    def load_dataset(self, path: str) -> List[Dict]:
        """Load and filter dataset."""
        with open(path) as f:
            data = json.load(f)
        
        drugs = data.get("drugs", [])
        
        if self.cyp_filter:
            drugs = [d for d in drugs if d.get("primary_cyp") == self.cyp_filter]
        
        print(f"Loaded {len(drugs)} {self.cyp_filter} drugs")
        return drugs
    
    def assess_molecule(self, drug: Dict) -> MoleculeAssessment:
        """Perform full assessment of a molecule."""
        assessment = MoleculeAssessment(
            drug_id=drug.get("id", "unknown"),
            name=drug.get("name", "unknown"),
            smiles=drug.get("smiles", ""),
            primary_cyp=drug.get("primary_cyp", ""),
            source=drug.get("source", "unknown"),
            source_reliability=SOURCE_RELIABILITY.get(drug.get("source", ""), 0.5),
            original_sites=list(drug.get("site_atoms", [])),
        )
        
        if not assessment.smiles:
            assessment.overall_quality = "reject"
            assessment.flags.append("NO_SMILES")
            return assessment
        
        # Parse molecule
        if not RDKIT_AVAILABLE:
            assessment.flags.append("RDKIT_UNAVAILABLE")
            assessment.overall_quality = "review"
            return assessment
        
        mol = Chem.MolFromSmiles(assessment.smiles)
        if mol is None:
            assessment.overall_quality = "reject"
            assessment.flags.append("INVALID_SMILES")
            return assessment
        
        mol_with_h = Chem.AddHs(mol)
        num_heavy = mol.GetNumHeavyAtoms()
        
        # Check 1: Too many labeled sites is suspicious
        num_sites = len(assessment.original_sites)
        if num_sites > num_heavy * 0.3:
            assessment.flags.append("TOO_MANY_SITES")
            self.stats["too_many_sites"] += 1
        
        if num_sites > 10:
            assessment.flags.append("EXCESSIVE_SITES")
            self.stats["excessive_sites"] += 1
        
        # Check 2: Assess each labeled atom
        for site_idx in assessment.original_sites:
            if site_idx >= mol_with_h.GetNumAtoms():
                assessment.invalid_sites.append(site_idx)
                assessment.flags.append(f"INVALID_ATOM_IDX_{site_idx}")
                continue
            
            atom_assessment = self.validity_checker.assess_atom(
                mol_with_h, site_idx, drug.get("reactions", [])
            )
            assessment.atom_assessments[site_idx] = atom_assessment
            
            if atom_assessment.is_chemically_valid:
                if atom_assessment.flags:
                    assessment.suspicious_sites.append(site_idx)
                else:
                    assessment.valid_sites.append(site_idx)
            else:
                assessment.invalid_sites.append(site_idx)
        
        # Check 3: Physics scoring
        physics_result = self.physics_scorer.score_molecule(assessment.smiles)
        
        if "error" not in physics_result:
            assessment.physics_top1 = physics_result["top1"]
            assessment.physics_top3 = physics_result["top3"]
            
            # Add physics info to atom assessments
            for site_idx in assessment.original_sites:
                if site_idx in assessment.atom_assessments:
                    aa = assessment.atom_assessments[site_idx]
                    if site_idx < len(physics_result["scores"]):
                        aa.physics_score = float(physics_result["scores"][site_idx])
                        aa.physics_pattern = physics_result["patterns"][site_idx]
                        aa.physics_rank = int(physics_result["ranks"][site_idx])
            
            # Check if physics agrees with any valid site
            valid_set = set(assessment.valid_sites + assessment.suspicious_sites)
            physics_top3_set = set(assessment.physics_top3)
            
            if valid_set & physics_top3_set:
                assessment.physics_agrees = True
            else:
                assessment.flags.append("PHYSICS_DISAGREES")
                self.stats["physics_disagrees"] += 1
        
        # Calculate quality score
        assessment.quality_score = self._calculate_quality_score(assessment)
        
        # Assign quality tier
        assessment.overall_quality = self._assign_quality_tier(assessment)
        
        # Generate recommendation
        assessment.recommendation = self._generate_recommendation(assessment)
        
        return assessment
    
    def _calculate_quality_score(self, assessment: MoleculeAssessment) -> float:
        """Calculate overall quality score 0-1."""
        score = assessment.source_reliability
        
        # Penalize for issues (but not too harshly)
        if "TOO_MANY_SITES" in assessment.flags:
            score *= 0.85
        if "EXCESSIVE_SITES" in assessment.flags:
            score *= 0.70
        if "PHYSICS_DISAGREES" in assessment.flags:
            score *= 0.90  # Reduced penalty - physics isn't always right
        
        # Factor in validity ratio
        total_sites = len(assessment.original_sites)
        if total_sites > 0:
            valid_count = len(assessment.valid_sites)
            suspicious_count = len(assessment.suspicious_sites)
            # Suspicious sites count as 0.5
            effective_valid = valid_count + 0.5 * suspicious_count
            valid_ratio = effective_valid / total_sites
            score *= (0.3 + 0.7 * valid_ratio)  # Even 0% valid gives 0.3
        
        # Small bonus for physics agreement
        if assessment.physics_agrees:
            score = min(1.0, score * 1.05)
        
        return score
    
    def _assign_quality_tier(self, assessment: MoleculeAssessment) -> str:
        """Assign quality tier based on assessment."""
        # Automatic rejection only for truly invalid cases
        if "INVALID_SMILES" in assessment.flags or "NO_SMILES" in assessment.flags:
            return "reject"
        
        # If ALL sites are chemically invalid, reject
        if not assessment.valid_sites and not assessment.suspicious_sites:
            return "reject"
        
        # Count valid + suspicious as usable
        usable_sites = len(assessment.valid_sites) + len(assessment.suspicious_sites)
        total_sites = len(assessment.original_sites)
        usable_ratio = usable_sites / max(total_sites, 1)
        
        # Gold: High quality - most sites valid, physics agrees, good source
        if (assessment.quality_score >= 0.60 and
            len(assessment.valid_sites) >= 1 and
            usable_ratio >= 0.5 and
            assessment.physics_agrees and
            "EXCESSIVE_SITES" not in assessment.flags):
            return "gold"
        
        # Silver: Medium quality - has valid sites, reasonable quality
        if (assessment.quality_score >= 0.40 and
            len(assessment.valid_sites) >= 1 and
            usable_ratio >= 0.3):
            return "silver"
        
        # Review: Has some usable sites but issues exist
        if usable_sites >= 1:
            return "review"
        
        return "reject"
    
    def _generate_recommendation(self, assessment: MoleculeAssessment) -> str:
        """Generate human-readable recommendation."""
        if assessment.overall_quality == "gold":
            return f"High-confidence. Use valid sites: {assessment.valid_sites}"
        
        if assessment.overall_quality == "silver":
            msg = f"Medium-confidence. Valid sites: {assessment.valid_sites}"
            if assessment.suspicious_sites:
                msg += f". Suspicious: {assessment.suspicious_sites}"
            return msg
        
        if assessment.overall_quality == "review":
            issues = []
            if assessment.invalid_sites:
                issues.append(f"Invalid sites {assessment.invalid_sites}")
            if "PHYSICS_DISAGREES" in assessment.flags:
                issues.append(f"Physics predicts {assessment.physics_top3}")
            if "TOO_MANY_SITES" in assessment.flags:
                issues.append(f"Too many sites ({len(assessment.original_sites)})")
            return "REVIEW NEEDED: " + "; ".join(issues)
        
        return f"Rejected: {', '.join(assessment.flags)}"
    
    def curate_dataset(self, drugs: List[Dict]) -> None:
        """Run full curation pipeline."""
        print(f"\nCurating {len(drugs)} molecules...")
        
        for i, drug in enumerate(drugs):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(drugs)}...")
            
            assessment = self.assess_molecule(drug)
            self.assessments.append(assessment)
            
            # Create curated drug entry
            curated = {
                "id": assessment.drug_id,
                "name": assessment.name,
                "smiles": assessment.smiles,
                "primary_cyp": assessment.primary_cyp,
                "source": assessment.source,
                "original_sites": assessment.original_sites,
                "curated_sites": assessment.valid_sites,
                "suspicious_sites": assessment.suspicious_sites,
                "invalid_sites": assessment.invalid_sites,
                "quality_score": assessment.quality_score,
                "quality_tier": assessment.overall_quality,
                "physics_top3": assessment.physics_top3,
                "physics_agrees": assessment.physics_agrees,
                "flags": assessment.flags,
                "recommendation": assessment.recommendation,
            }
            
            # Add atom-level details for review cases
            if assessment.overall_quality in ["review", "reject"]:
                curated["atom_details"] = {
                    idx: {
                        "symbol": aa.symbol,
                        "valid": aa.is_chemically_valid,
                        "reason": aa.validity_reason,
                        "physics_score": aa.physics_score,
                        "physics_pattern": aa.physics_pattern,
                        "flags": aa.flags,
                    }
                    for idx, aa in assessment.atom_assessments.items()
                }
            
            # Sort into buckets
            if assessment.overall_quality == "gold":
                self.gold.append(curated)
                self.stats["gold"] += 1
            elif assessment.overall_quality == "silver":
                self.silver.append(curated)
                self.stats["silver"] += 1
            elif assessment.overall_quality == "review":
                self.review_queue.append(curated)
                self.stats["review"] += 1
            else:
                self.rejected.append(curated)
                self.stats["rejected"] += 1
        
        print(f"\nCuration complete!")
        print(f"  Gold:     {self.stats['gold']}")
        print(f"  Silver:   {self.stats['silver']}")
        print(f"  Review:   {self.stats['review']}")
        print(f"  Rejected: {self.stats['rejected']}")
    
    def create_training_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """Create final training datasets with negative sampling."""
        
        # Gold dataset: only valid sites
        gold_train = []
        for drug in self.gold:
            entry = {
                "id": drug["id"],
                "name": drug["name"],
                "smiles": drug["smiles"],
                "primary_cyp": drug["primary_cyp"],
                "site_atoms": drug["curated_sites"],
                "source": drug["source"],
                "quality": "gold",
            }
            gold_train.append(entry)
        
        # Silver dataset: valid + suspicious (with caution)
        silver_train = []
        for drug in self.silver:
            entry = {
                "id": drug["id"],
                "name": drug["name"],
                "smiles": drug["smiles"],
                "primary_cyp": drug["primary_cyp"],
                "site_atoms": drug["curated_sites"],  # Only use curated, not suspicious
                "source": drug["source"],
                "quality": "silver",
            }
            silver_train.append(entry)
        
        return gold_train, silver_train
    
    def add_negative_sampling(self, dataset: List[Dict], num_negatives: int = 3) -> List[Dict]:
        """Add hard negative samples for each molecule."""
        if not RDKIT_AVAILABLE:
            return dataset
        
        augmented = []
        
        for drug in dataset:
            smiles = drug.get("smiles", "")
            positive_sites = set(drug.get("site_atoms", []))
            
            if not smiles or not positive_sites:
                augmented.append(drug)
                continue
            
            # Get physics scores
            physics = self.physics_scorer.score_molecule(smiles)
            if "error" in physics:
                augmented.append(drug)
                continue
            
            # Find hard negatives: high physics score but not labeled
            scores = physics["scores"]
            is_heavy = physics["is_heavy"]
            
            candidates = []
            for idx in range(len(scores)):
                if is_heavy[idx] and idx not in positive_sites:
                    candidates.append((idx, scores[idx]))
            
            # Sort by score descending (hardest negatives first)
            candidates.sort(key=lambda x: -x[1])
            
            # Take top N
            hard_negatives = [idx for idx, _ in candidates[:num_negatives]]
            
            # Create augmented entry
            entry = dict(drug)
            entry["hard_negatives"] = hard_negatives
            entry["all_candidates"] = list(positive_sites) + hard_negatives
            entry["labels"] = [1] * len(positive_sites) + [0] * len(hard_negatives)
            
            augmented.append(entry)
        
        return augmented
    
    def generate_report(self) -> str:
        """Generate detailed curation report."""
        lines = [
            "# CYP3A4 Dataset Curation Report",
            "",
            f"**Generated:** 2026-04-09",
            f"**CYP Filter:** {self.cyp_filter}",
            "",
            "## Summary",
            "",
            f"| Category | Count | Percentage |",
            f"|----------|-------|------------|",
        ]
        
        total = sum([self.stats["gold"], self.stats["silver"], 
                     self.stats["review"], self.stats["rejected"]])
        
        for cat in ["gold", "silver", "review", "rejected"]:
            count = self.stats[cat]
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"| {cat.capitalize()} | {count} | {pct:.1f}% |")
        
        lines.append(f"| **Total** | {total} | 100% |")
        
        # Chemical validity stats
        lines.extend([
            "",
            "## Chemical Validity Issues",
            "",
            "| Issue | Count |",
            "|-------|-------|",
        ])
        
        validity_stats = self.validity_checker.get_stats()
        for issue, count in sorted(validity_stats.items(), key=lambda x: -x[1]):
            lines.append(f"| {issue} | {count} |")
        
        # Source breakdown
        lines.extend([
            "",
            "## Quality by Source",
            "",
        ])
        
        source_quality = defaultdict(lambda: {"gold": 0, "silver": 0, "review": 0, "rejected": 0, "reject": 0})
        for a in self.assessments:
            source_quality[a.source][a.overall_quality] += 1
        
        lines.append("| Source | Gold | Silver | Review | Rejected | Gold Rate |")
        lines.append("|--------|------|--------|--------|----------|-----------|")
        
        for source in sorted(source_quality.keys()):
            sq = source_quality[source]
            total_src = sum(sq.values())
            gold_rate = sq["gold"] / total_src * 100 if total_src > 0 else 0
            lines.append(f"| {source} | {sq['gold']} | {sq['silver']} | {sq['review']} | {sq['rejected']} | {gold_rate:.1f}% |")
        
        # Common rejection reasons
        lines.extend([
            "",
            "## Common Issues in Rejected/Review",
            "",
        ])
        
        flag_counts = defaultdict(int)
        for a in self.assessments:
            if a.overall_quality in ["review", "rejected"]:
                for flag in a.flags:
                    flag_counts[flag] += 1
        
        lines.append("| Flag | Count |")
        lines.append("|------|-------|")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"| {flag} | {count} |")
        
        # Sample rejected cases
        lines.extend([
            "",
            "## Sample Rejected Cases (first 10)",
            "",
        ])
        
        for drug in self.rejected[:10]:
            lines.append(f"- **{drug['name']}** ({drug['id']})")
            lines.append(f"  - SMILES: `{drug['smiles'][:60]}...`" if len(drug['smiles']) > 60 else f"  - SMILES: `{drug['smiles']}`")
            lines.append(f"  - Original sites: {drug['original_sites']}")
            lines.append(f"  - Flags: {', '.join(drug['flags'])}")
            lines.append(f"  - Recommendation: {drug['recommendation']}")
            lines.append("")
        
        # Physics disagreement analysis
        lines.extend([
            "",
            "## Physics Disagreement Analysis",
            "",
            "Cases where labeled SoM differs from physics top-3:",
            "",
        ])
        
        disagree_cases = [a for a in self.assessments if "PHYSICS_DISAGREES" in a.flags][:10]
        for a in disagree_cases:
            lines.append(f"- **{a.name}**")
            lines.append(f"  - Labeled: {a.original_sites}")
            lines.append(f"  - Physics top-3: {a.physics_top3}")
            if a.atom_assessments:
                for idx in a.original_sites[:3]:
                    if idx in a.atom_assessments:
                        aa = a.atom_assessments[idx]
                        lines.append(f"  - Site {idx}: {aa.symbol}, physics_score={aa.physics_score:.2f}, pattern={aa.physics_pattern}")
            lines.append("")
        
        # Recommendations
        lines.extend([
            "",
            "## Recommendations",
            "",
            "### For Training:",
            f"1. Use **Gold** dataset ({self.stats['gold']} molecules) for primary training",
            f"2. Optionally add **Silver** dataset ({self.stats['silver']} molecules) with sample weighting",
            f"3. Review queue ({self.stats['review']} molecules) requires manual expert review",
            "",
            "### For Manual Review Priority:",
            "1. Focus on 'PHYSICS_DISAGREES' cases - may indicate novel chemistry or labeling error",
            "2. Check 'TOO_MANY_SITES' cases - may need to identify primary SoM",
            "3. Verify 'SUSPICIOUS_ATOM' cases - especially oxygen-labeled sites",
            "",
            "### Data Quality Improvements:",
            "1. Cross-reference with metabolite structures from DrugBank/HMDB",
            "2. Add external validated datasets (Zaretzki, XenoSite)",
            "3. Consider docking to add 3D binding site information",
        ])
        
        return "\n".join(lines)
    
    def save_results(self, output_dir: str) -> None:
        """Save all curation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create training datasets
        gold_train, silver_train = self.create_training_datasets()
        
        # Add negative sampling
        gold_train_neg = self.add_negative_sampling(gold_train)
        silver_train_neg = self.add_negative_sampling(silver_train)
        
        # Combined training set
        combined_train = gold_train_neg + silver_train_neg
        
        # Save datasets
        datasets = {
            "curated_cyp3a4_gold.json": {
                "description": "High-confidence CYP3A4 SoM dataset",
                "n_drugs": len(gold_train_neg),
                "quality": "gold",
                "drugs": gold_train_neg,
            },
            "curated_cyp3a4_silver.json": {
                "description": "Medium-confidence CYP3A4 SoM dataset",
                "n_drugs": len(silver_train_neg),
                "quality": "silver",
                "drugs": silver_train_neg,
            },
            "curated_cyp3a4_combined.json": {
                "description": "Combined gold+silver CYP3A4 SoM dataset",
                "n_drugs": len(combined_train),
                "quality": "combined",
                "drugs": combined_train,
            },
            "manual_review_queue.json": {
                "description": "Cases requiring manual expert review",
                "n_drugs": len(self.review_queue),
                "drugs": self.review_queue,
            },
            "rejected_labels.json": {
                "description": "Rejected cases with reasons",
                "n_drugs": len(self.rejected),
                "drugs": self.rejected,
            },
        }
        
        for filename, data in datasets.items():
            filepath = output_path / filename
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {filepath} ({data['n_drugs']} drugs)")
        
        # Save report
        report = self.generate_report()
        report_path = output_path / "curation_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved {report_path}")
        
        # Save detailed assessments for debugging
        assessments_data = []
        for a in self.assessments:
            entry = {
                "drug_id": a.drug_id,
                "name": a.name,
                "smiles": a.smiles,
                "source": a.source,
                "source_reliability": a.source_reliability,
                "original_sites": a.original_sites,
                "valid_sites": a.valid_sites,
                "suspicious_sites": a.suspicious_sites,
                "invalid_sites": a.invalid_sites,
                "physics_top3": a.physics_top3,
                "physics_agrees": a.physics_agrees,
                "quality_score": a.quality_score,
                "overall_quality": a.overall_quality,
                "flags": a.flags,
                "recommendation": a.recommendation,
            }
            assessments_data.append(entry)
        
        assessments_path = output_path / "full_assessments.json"
        with open(assessments_path, "w") as f:
            json.dump(assessments_data, f, indent=2)
        print(f"Saved {assessments_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CYP3A4 Dataset Curation Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/curated",
        help="Output directory"
    )
    parser.add_argument(
        "--cyp",
        type=str,
        default="CYP3A4",
        help="CYP isoform to filter"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = args.input
    if not Path(input_path).is_absolute():
        input_path = str(PROJECT_ROOT / input_path)
    
    output_path = args.output
    if not Path(output_path).is_absolute():
        output_path = str(PROJECT_ROOT / output_path)
    
    print("=" * 70)
    print("CYP3A4 DATASET CURATION PIPELINE")
    print("=" * 70)
    
    # Initialize curator
    curator = DatasetCurator(cyp_filter=args.cyp)
    
    # Load data
    drugs = curator.load_dataset(input_path)
    
    # Run curation
    curator.curate_dataset(drugs)
    
    # Save results
    curator.save_results(output_path)
    
    print("\n" + "=" * 70)
    print("CURATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to: {output_path}")
    print(f"\nRecommended next steps:")
    print(f"  1. Review manual_review_queue.json ({curator.stats['review']} cases)")
    print(f"  2. Train on curated_cyp3a4_gold.json ({curator.stats['gold']} high-quality cases)")
    print(f"  3. Add curated_cyp3a4_silver.json ({curator.stats['silver']} cases) if more data needed")


if __name__ == "__main__":
    main()
