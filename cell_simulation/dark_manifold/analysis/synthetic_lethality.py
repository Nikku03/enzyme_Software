"""
Synthetic Lethality Analysis

Gene pairs viable alone, lethal together.
Valuable for:
- Drug target discovery (target redundant pathways)
- Understanding genetic interactions
- Predicting combination therapies
"""

from __future__ import annotations
import time
from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import numpy as np


@dataclass
class SLPair:
    """Synthetic lethal gene pair."""
    gene_a: str
    gene_b: str
    biomass_ratio: float
    pathway_a: str = ""
    pathway_b: str = ""


@dataclass 
class SLScreen:
    """Results of synthetic lethality screen."""
    viable_genes: List[str]
    essential_genes: List[str]
    sl_pairs: List[SLPair]
    total_pairs_tested: int
    runtime_seconds: float


def run_sl_screen(fba_model, verbose: bool = True) -> SLScreen:
    """
    Run synthetic lethality screen.
    
    Tests all pairs of non-essential genes.
    SL pair = both viable alone, lethal together.
    """
    genes = sorted(fba_model.get_genes())
    
    # Single knockouts
    if verbose:
        print("Single knockouts...")
    
    viable = []
    essential = []
    
    for g in genes:
        result = fba_model.knockout(g)
        if result['essential']:
            essential.append(g)
        else:
            viable.append(g)
    
    if verbose:
        print(f"  Viable: {len(viable)}, Essential: {len(essential)}")
    
    # Double knockouts (only viable pairs)
    pairs = list(combinations(viable, 2))
    
    if verbose:
        print(f"Testing {len(pairs)} pairs...")
    
    start = time.time()
    sl_pairs = []
    
    for ga, gb in pairs:
        result = fba_model.double_knockout(ga, gb)
        if result['essential']:
            sl_pairs.append(SLPair(
                gene_a=ga,
                gene_b=gb,
                biomass_ratio=float(result['biomass_ratio'])
            ))
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"Found {len(sl_pairs)} SL pairs in {elapsed:.1f}s")
    
    return SLScreen(
        viable_genes=viable,
        essential_genes=essential,
        sl_pairs=sl_pairs,
        total_pairs_tested=len(pairs),
        runtime_seconds=elapsed
    )


def analyze_sl_network(screen: SLScreen) -> Dict:
    """
    Analyze SL network structure.
    
    Returns hub genes, clusters, pathway enrichment.
    """
    # Count interactions per gene
    gene_counts = {}
    for pair in screen.sl_pairs:
        gene_counts[pair.gene_a] = gene_counts.get(pair.gene_a, 0) + 1
        gene_counts[pair.gene_b] = gene_counts.get(pair.gene_b, 0) + 1
    
    # Hub genes (many SL partners)
    sorted_genes = sorted(gene_counts.items(), key=lambda x: -x[1])
    hubs = [(g, c) for g, c in sorted_genes if c >= 2]
    
    # Build adjacency
    adj = {g: set() for g in gene_counts}
    for pair in screen.sl_pairs:
        adj[pair.gene_a].add(pair.gene_b)
        adj[pair.gene_b].add(pair.gene_a)
    
    # Find cliques (fully connected subsets)
    cliques = []
    genes_in_sl = list(gene_counts.keys())
    
    for i, g1 in enumerate(genes_in_sl):
        for g2 in genes_in_sl[i+1:]:
            if g2 in adj[g1]:
                # Check for triangles
                common = adj[g1] & adj[g2]
                for g3 in common:
                    clique = tuple(sorted([g1, g2, g3]))
                    if clique not in cliques:
                        cliques.append(clique)
    
    return {
        'hub_genes': hubs,
        'gene_sl_counts': gene_counts,
        'cliques': cliques,
        'n_genes_in_sl': len(gene_counts),
        'n_pairs': len(screen.sl_pairs),
        'density': len(screen.sl_pairs) / (len(gene_counts) * (len(gene_counts)-1) / 2) if len(gene_counts) > 1 else 0
    }


def predict_sl_from_features(
    fba_model,
    feature_extractor,
    gene_a: str,
    gene_b: str,
) -> Dict:
    """
    Predict SL using feature similarity.
    
    Hypothesis: genes in same pathway more likely SL
    (redundant function).
    """
    feat_a = feature_extractor.extract(gene_a).features
    feat_b = feature_extractor.extract(gene_b).features
    
    # Feature similarity
    dist = np.linalg.norm(feat_a - feat_b)
    similarity = 1 / (1 + dist)
    
    # Pathway overlap (shared metabolites via reactions)
    # Genes sharing metabolites = potential redundancy
    
    # Simple heuristic: similar features + both non-essential = SL candidate
    single_a = fba_model.knockout(gene_a)
    single_b = fba_model.knockout(gene_b)
    
    both_viable = not single_a['essential'] and not single_b['essential']
    
    sl_score = similarity if both_viable else 0.0
    
    return {
        'gene_a': gene_a,
        'gene_b': gene_b,
        'feature_similarity': similarity,
        'both_viable': both_viable,
        'sl_score': sl_score,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')
    from dark_manifold.models.fba import get_fba_model
    
    fba = get_fba_model(verbose=False)
    screen = run_sl_screen(fba, verbose=True)
    
    print("\n=== SL NETWORK ANALYSIS ===")
    analysis = analyze_sl_network(screen)
    
    print(f"Genes in SL network: {analysis['n_genes_in_sl']}")
    print(f"SL pairs: {analysis['n_pairs']}")
    print(f"Network density: {analysis['density']:.2%}")
    
    print("\nHub genes:")
    for gene, count in analysis['hub_genes'][:10]:
        print(f"  {gene}: {count} partners")
    
    print(f"\nCliques (triangles): {len(analysis['cliques'])}")
    for clique in analysis['cliques'][:5]:
        print(f"  {clique}")
