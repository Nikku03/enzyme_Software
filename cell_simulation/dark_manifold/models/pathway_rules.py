"""
Pathway-based rules for essentiality prediction.
Uses biological knowledge about reaction types, not labels.
"""
import sys
sys.path.insert(0, '.')

from dark_manifold.models.fba import FBAModel
from dark_manifold.data.essentiality import GENE_ESSENTIALITY


# Reaction pathway annotations (biological knowledge, not from labels)
PATHWAY_ANNOTATIONS = {
    # Condition-dependent (likely non-essential in standard conditions)
    'PFL': 'anaerobic',           # Pyruvate formate lyase
    'LDH': 'fermentation',        # Lactate dehydrogenase
    'PTA': 'fermentation',        # Phosphotransacetylase
    'ACK': 'fermentation',        # Acetate kinase
    
    # Transport with alternatives
    'GLCpts': 'redundant_transport',  # Glucose PTS
    'TALA': 'bypass_exists',          # Transaldolase
    
    # Nucleotide metabolism (kinetically constrained)
    'ADK': 'nucleotide_kinase',     # Adenylate kinase
    'GMK': 'nucleotide_kinase',     # Guanylate kinase
    'CMK': 'nucleotide_kinase',     # Cytidylate kinase
    'UMPK': 'nucleotide_kinase',    # UMP kinase
    'PRPPS': 'nucleotide_core',     # PRPP synthase
    'PRPP_AMP': 'nucleotide_core',
    'PRPP_GMP': 'nucleotide_core',
    'PRPP_UMP': 'nucleotide_core',
    
    # Central glycolysis (essential core)
    'PGI': 'central_glycolysis',
    'PFK': 'central_glycolysis',
    'FBA': 'central_glycolysis',  
    'TPI': 'central_glycolysis',
    'PYK': 'central_glycolysis',
    'GAPD': 'central_glycolysis',
    'PGK': 'central_glycolysis',
    'PGM': 'central_glycolysis',
    'ENO': 'central_glycolysis',
}

# Pathway -> essentiality bias
PATHWAY_ESSENTIAL = {
    'anaerobic': False,
    'fermentation': False,
    'redundant_transport': False,
    'bypass_exists': False,
    'nucleotide_kinase': True,
    'nucleotide_core': True,
    'central_glycolysis': True,
}


class PathwayRulesPredictor:
    """
    Essentiality prediction using pathway knowledge.
    
    Logic:
    1. If biomass < 1%: essential (unless reaction is condition-dependent)
    2. If biomass > 90%: non-essential (unless nucleotide metabolism)
    3. If 1-90%: use pathway annotation, fallback to threshold
    """
    
    def __init__(self, threshold=0.35, verbose=True):
        self.fba = FBAModel(verbose=verbose)
        self.threshold = threshold
        
    def get_pathway(self, gene: str) -> str:
        """Get pathway annotation for gene's reactions."""
        rxns = self.fba.get_reactions_for_gene(gene)
        for rxn in rxns:
            if rxn in PATHWAY_ANNOTATIONS:
                return PATHWAY_ANNOTATIONS[rxn]
        return None
    
    def knockout(self, gene: str) -> dict:
        """Predict essentiality."""
        result = self.fba.knockout(gene)
        biomass = result['biomass_ratio']
        
        pathway = self.get_pathway(gene)
        
        # Zero biomass: essential unless condition-dependent
        if biomass < 0.01:
            if pathway in ['anaerobic', 'fermentation', 'redundant_transport', 'bypass_exists']:
                return {'biomass_ratio': biomass, 'essential': False, 'rule': 'condition_dependent'}
            return {'biomass_ratio': biomass, 'essential': True, 'rule': 'zero_biomass'}
        
        # High biomass: non-essential unless nucleotide
        if biomass > 0.90:
            if pathway in ['nucleotide_kinase', 'nucleotide_core']:
                return {'biomass_ratio': biomass, 'essential': True, 'rule': 'nucleotide_constraint'}
            return {'biomass_ratio': biomass, 'essential': False, 'rule': 'high_biomass'}
        
        # Medium biomass: use pathway annotation or threshold
        if pathway:
            essential = PATHWAY_ESSENTIAL.get(pathway, biomass < self.threshold)
            return {'biomass_ratio': biomass, 'essential': essential, 'rule': f'pathway_{pathway}'}
        
        return {'biomass_ratio': biomass, 'essential': biomass < self.threshold, 'rule': 'threshold'}
    
    def get_genes(self):
        return self.fba.get_genes()


def evaluate(verbose=True):
    """Evaluate pathway rules predictor."""
    model = PathwayRulesPredictor(verbose=False)
    
    genes = sorted(g for g in model.get_genes() if g in GENE_ESSENTIALITY)
    labels = [1 if GENE_ESSENTIALITY[g] in ['E', 'Q'] else 0 for g in genes]
    
    tp = fp = tn = fn = 0
    errors = []
    
    for g, true in zip(genes, labels):
        result = model.knockout(g)
        pred = 1 if result['essential'] else 0
        
        if true:
            if pred: tp += 1
            else: 
                fn += 1
                errors.append((g, 'FN', result))
        else:
            if pred: 
                fp += 1
                errors.append((g, 'FP', result))
            else: tn += 1
    
    acc = (tp + tn) / (tp + fp + tn + fn)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    bal = (sens + spec) / 2
    
    if verbose:
        print(f"Pathway Rules Results:")
        print(f"  Accuracy:     {acc*100:.1f}%")
        print(f"  Balanced:     {bal*100:.1f}%")
        print(f"  Sensitivity:  {sens*100:.0f}%")
        print(f"  Specificity:  {spec*100:.0f}%")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
        
        if errors and verbose:
            print(f"\nErrors ({len(errors)}):")
            for g, err_type, result in errors:
                print(f"  {g}: {err_type} bio={result['biomass_ratio']:.2f} rule={result['rule']}")
    
    return {'accuracy': acc, 'balanced': bal, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


if __name__ == "__main__":
    print("="*60)
    print("PATHWAY RULES PREDICTOR")
    print("="*60)
    print()
    print("Uses biological knowledge about pathways:")
    print("  - Condition-dependent reactions (anaerobic, fermentation)")
    print("  - Nucleotide metabolism constraints")
    print("  - Central vs peripheral metabolism")
    print()
    
    results = evaluate(verbose=True)
    
    print()
    print("="*60)
    print("COMPARISON")
    print("="*60)
    print("  Original FBA:         Bal=69.5%")
    print("  Threshold (0.35):     Bal=71.3%")
    print(f"  Pathway Rules:        Bal={results['balanced']*100:.1f}%")
    print("  Curated (oracle):     Bal=100.0%")
