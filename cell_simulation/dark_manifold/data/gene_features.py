"""
dark_manifold/data/gene_features.py

Gene feature extraction for neural refinement.

Extracts features for each gene:
1. FBA-derived features (biomass impact, blocked reactions)
2. Network topology (centrality, hub status)
3. Redundancy (isozymes, alternative pathways)
4. Expression proxy (based on reaction count - real data would be better)
5. Thermodynamic features (reversibility)

These features feed into the GeneEncoder for similarity-based retrieval.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


@dataclass
class GeneFeatures:
    """Feature vector for a single gene."""
    gene_id: str
    features: np.ndarray  # Shape: (15,)
    
    # Individual feature values for interpretability
    biomass_ratio: float = 0.0
    num_blocked_reactions: int = 0
    flux_variability: float = 0.0
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    is_hub: bool = False
    clustering_coefficient: float = 0.0
    isozyme_count: int = 0
    has_alternative_pathway: bool = False
    pathway_redundancy: float = 0.0
    expression_level: float = 0.0
    protein_halflife: float = 0.0
    reaction_delta_g: float = 0.0
    is_irreversible: bool = False


class GeneFeatureExtractor:
    """
    Extract features for genes in the metabolic model.
    
    Features combine FBA results with network topology and redundancy metrics.
    """
    
    def __init__(self, fba_model=None, verbose: bool = False):
        """
        Initialize feature extractor.
        
        Args:
            fba_model: FBAModel instance. If None, will create one.
            verbose: Print progress messages.
        """
        self.verbose = verbose
        
        if fba_model is None:
            from ..models.fba import get_fba_model
            self.fba = get_fba_model(verbose=verbose)
        else:
            self.fba = fba_model
        
        # Build gene-reaction network
        self._build_network()
        
        # Cache knockout results
        self._knockout_cache: Dict[str, Dict] = {}
    
    def _build_network(self):
        """Build gene network from stoichiometry matrix."""
        try:
            import networkx as nx
            self._nx_available = True
        except ImportError:
            self._nx_available = False
            if self.verbose:
                print("Warning: networkx not available, using simplified topology")
            return
        
        self.G = nx.DiGraph()
        
        # Add genes as nodes
        for gene in self.fba.get_genes():
            self.G.add_node(gene)
        
        # Connect genes that share metabolites
        # Two genes are connected if the product of one reaction
        # is the substrate of another
        stoich = self.fba.stoich
        gene_to_rxns = self.fba.gene_to_rxns
        
        for gene_a in gene_to_rxns:
            rxns_a = gene_to_rxns[gene_a]
            products_a = set()
            
            # Get products of gene_a's reactions
            for rxn_idx in rxns_a:
                for met_idx in range(stoich.n_mets):
                    if stoich.S[met_idx, rxn_idx] > 0:  # Product
                        products_a.add(met_idx)
            
            # Check if any other gene uses these as substrates
            for gene_b in gene_to_rxns:
                if gene_a == gene_b:
                    continue
                rxns_b = gene_to_rxns[gene_b]
                
                for rxn_idx in rxns_b:
                    for met_idx in range(stoich.n_mets):
                        if stoich.S[met_idx, rxn_idx] < 0:  # Substrate
                            if met_idx in products_a:
                                self.G.add_edge(gene_a, gene_b)
                                break
        
        # Precompute centrality metrics
        if self.verbose:
            print(f"Built gene network: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        self._degree_centrality = nx.degree_centrality(self.G)
        
        # Betweenness can be slow, use approximation for large graphs
        if self.G.number_of_nodes() > 100:
            self._betweenness = nx.betweenness_centrality(self.G, k=min(50, self.G.number_of_nodes()))
        else:
            self._betweenness = nx.betweenness_centrality(self.G)
        
        self._closeness = nx.closeness_centrality(self.G)
        self._clustering = nx.clustering(self.G.to_undirected())
        
        # Compute hub threshold (mean + 2*std of degree)
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        self._hub_threshold = np.mean(degrees) + 2 * np.std(degrees) if degrees else 0
    
    def _get_knockout_result(self, gene: str) -> Dict:
        """Get cached knockout result or compute it."""
        if gene not in self._knockout_cache:
            self._knockout_cache[gene] = self.fba.knockout(gene)
        return self._knockout_cache[gene]
    
    def _count_isozymes(self, gene: str) -> int:
        """
        Count isozymes for a gene.
        
        Isozymes are other genes that catalyze the same reaction(s).
        """
        gene_rxns = set(self.fba.gene_to_rxns.get(gene, []))
        if not gene_rxns:
            return 0
        
        isozyme_count = 0
        for other_gene, other_rxns in self.fba.gene_to_rxns.items():
            if other_gene != gene:
                if gene_rxns & set(other_rxns):  # Shared reactions
                    isozyme_count += 1
        
        return isozyme_count
    
    def _has_alternative_pathway(self, gene: str) -> bool:
        """
        Check if there's an alternative pathway bypassing this gene.
        
        Simple heuristic: If knockout still allows some biomass production,
        there must be an alternative route.
        """
        result = self._get_knockout_result(gene)
        return result['biomass_ratio'] > 0.1  # >10% growth = alternative exists
    
    def _get_pathway_redundancy(self, gene: str) -> float:
        """
        Compute pathway redundancy score.
        
        Based on:
        - Number of isozymes
        - Whether alternative pathways exist
        - Network connectivity
        """
        isozymes = self._count_isozymes(gene)
        alt_path = self._has_alternative_pathway(gene)
        
        # Normalize isozyme count (most genes have 0-3 isozymes)
        isozyme_score = min(isozymes / 3.0, 1.0)
        
        # Alternative pathway score
        alt_score = 1.0 if alt_path else 0.0
        
        # Combine
        return 0.5 * isozyme_score + 0.5 * alt_score
    
    def _estimate_expression(self, gene: str) -> Tuple[float, float]:
        """
        Estimate expression level and protein half-life.
        
        Without real transcriptomics data, we use proxy features:
        - Expression: Number of reactions (more reactions = likely higher expression)
        - Half-life: Based on gene function (essential genes often more stable)
        
        Returns:
            (expression_level, protein_halflife) - both normalized 0-1
        """
        n_reactions = len(self.fba.gene_to_rxns.get(gene, []))
        
        # Normalize (most genes have 1-5 reactions)
        expression = min(n_reactions / 5.0, 1.0)
        
        # Essential genes tend to have stable proteins
        result = self._get_knockout_result(gene)
        stability = 1.0 - result['biomass_ratio']  # More essential = more stable
        
        return expression, stability
    
    def _get_thermodynamic_features(self, gene: str) -> Tuple[float, bool]:
        """
        Get thermodynamic features for gene's reactions.
        
        Returns:
            (delta_g_proxy, is_irreversible)
        """
        rxn_indices = self.fba.gene_to_rxns.get(gene, [])
        if not rxn_indices:
            return 0.0, False
        
        stoich = self.fba.stoich
        
        # Check reversibility
        n_irreversible = 0
        for rxn_idx in rxn_indices:
            if stoich.lb[rxn_idx] >= 0:  # lb >= 0 means irreversible
                n_irreversible += 1
        
        is_irreversible = n_irreversible > len(rxn_indices) / 2
        
        # Without actual thermodynamic data, use irreversibility as proxy
        # Irreversible reactions tend to be more thermodynamically favorable
        delta_g_proxy = 1.0 if is_irreversible else 0.5
        
        return delta_g_proxy, is_irreversible
    
    def extract(self, gene: str) -> GeneFeatures:
        """
        Extract all features for a gene.
        
        Returns:
            GeneFeatures with 15-dimensional feature vector
        """
        # FBA features
        ko_result = self._get_knockout_result(gene)
        biomass_ratio = ko_result['biomass_ratio']
        
        # Count blocked reactions
        rxn_indices = self.fba.gene_to_rxns.get(gene, [])
        num_blocked = len(rxn_indices)
        
        # Flux variability proxy (based on reversibility)
        stoich = self.fba.stoich
        reversible_count = sum(1 for i in rxn_indices if stoich.lb[i] < 0)
        flux_var = reversible_count / max(len(rxn_indices), 1)
        
        # Network topology
        if self._nx_available:
            degree = self._degree_centrality.get(gene, 0.0)
            between = self._betweenness.get(gene, 0.0)
            close = self._closeness.get(gene, 0.0)
            cluster = self._clustering.get(gene, 0.0)
            is_hub = self.G.degree(gene) > self._hub_threshold if gene in self.G else False
        else:
            # Simplified: use reaction count as proxy
            degree = min(num_blocked / 5.0, 1.0)
            between = degree * 0.5
            close = degree * 0.5
            cluster = 0.5
            is_hub = num_blocked > 3
        
        # Redundancy
        isozyme_count = self._count_isozymes(gene)
        has_alt = self._has_alternative_pathway(gene)
        redundancy = self._get_pathway_redundancy(gene)
        
        # Expression (proxy)
        expression, halflife = self._estimate_expression(gene)
        
        # Thermodynamics
        delta_g, irreversible = self._get_thermodynamic_features(gene)
        
        # Build feature vector
        features = np.array([
            biomass_ratio,           # 0
            num_blocked / 10.0,      # 1: normalize
            flux_var,                # 2
            degree,                  # 3
            between,                 # 4
            close,                   # 5
            float(is_hub),           # 6
            cluster,                 # 7
            isozyme_count / 3.0,     # 8: normalize
            float(has_alt),          # 9
            redundancy,              # 10
            expression,              # 11
            halflife,                # 12
            delta_g,                 # 13
            float(irreversible),     # 14
        ], dtype=np.float32)
        
        return GeneFeatures(
            gene_id=gene,
            features=features,
            biomass_ratio=biomass_ratio,
            num_blocked_reactions=num_blocked,
            flux_variability=flux_var,
            degree_centrality=degree,
            betweenness_centrality=between,
            closeness_centrality=close,
            is_hub=is_hub,
            clustering_coefficient=cluster,
            isozyme_count=isozyme_count,
            has_alternative_pathway=has_alt,
            pathway_redundancy=redundancy,
            expression_level=expression,
            protein_halflife=halflife,
            reaction_delta_g=delta_g,
            is_irreversible=irreversible,
        )
    
    def extract_all(self) -> Dict[str, GeneFeatures]:
        """Extract features for all genes in model."""
        features = {}
        genes = self.fba.get_genes()
        
        for i, gene in enumerate(genes):
            if self.verbose and (i + 1) % 20 == 0:
                print(f"Extracting features: {i+1}/{len(genes)}")
            features[gene] = self.extract(gene)
        
        return features
    
    def to_tensor(self, genes: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features as a numpy array for batch processing.
        
        Args:
            genes: List of gene IDs. If None, use all genes.
        
        Returns:
            (features_array, gene_list) where features_array is (N, 15)
        """
        if genes is None:
            genes = self.fba.get_genes()
        
        features = []
        for gene in genes:
            gf = self.extract(gene)
            features.append(gf.features)
        
        return np.stack(features), genes


# Convenience function
def extract_gene_features(gene: str, fba_model=None) -> np.ndarray:
    """Quick feature extraction for a single gene."""
    extractor = GeneFeatureExtractor(fba_model, verbose=False)
    return extractor.extract(gene).features


if __name__ == "__main__":
    print("Testing GeneFeatureExtractor...")
    
    extractor = GeneFeatureExtractor(verbose=True)
    
    # Test single gene
    gene = 'JCVISYN3A_0207'  # pfkA
    features = extractor.extract(gene)
    
    print(f"\nFeatures for {gene} (pfkA):")
    print(f"  biomass_ratio: {features.biomass_ratio:.3f}")
    print(f"  blocked_reactions: {features.num_blocked_reactions}")
    print(f"  degree_centrality: {features.degree_centrality:.3f}")
    print(f"  is_hub: {features.is_hub}")
    print(f"  isozyme_count: {features.isozyme_count}")
    print(f"  has_alternative: {features.has_alternative_pathway}")
    print(f"  Feature vector: {features.features}")
    
    # Test batch extraction
    print("\nExtracting all features...")
    X, genes = extractor.to_tensor()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of genes: {len(genes)}")
