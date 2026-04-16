"""
Tier 1: FBA-based endpoint predictor
=====================================

Wraps V37 FBA for fast essentiality prediction + feature extraction.
This is the fast, baseline tier — microseconds per query, 85.6% accuracy.

Its second job: emit a feature vector per gene that Tier 2 can use to
correct FBA's systematic errors.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, '/home/claude/enzyme_repo/cell_simulation/v37_full_imb155')
from core_cell_simulator import (
    CoreCellSimulator,
    GENE_ESSENTIALITY,
    GENE_NAMES,
)


class Tier1FBA:
    """
    Endpoint predictor: (gene_id) -> (essential_bool, confidence, features)
    
    Uses V37 FBA as the workhorse. Emits a 10-feature vector per gene
    that captures information FBA's binary decision loses.
    """
    
    def __init__(self, verbose=True):
        if not verbose:
            # Silence V37's print statements
            import builtins
            orig_print = builtins.print
            builtins.print = lambda *a, **k: None
            try:
                self.sim = CoreCellSimulator()
            finally:
                builtins.print = orig_print
        else:
            self.sim = CoreCellSimulator()
        
        # Precompute reaction-level features
        self._build_reaction_features()
    
    def _build_reaction_features(self):
        """
        Precompute features that are fixed across all knockouts:
        - Gene -> number of reactions catalyzed
        - Gene -> reaction degree centrality in metabolite graph
        - Gene -> whether any reaction is a dead-end producer/consumer
        """
        self.gene_n_rxns = {}
        self.gene_centrality = {}
        self.gene_is_singleton = {}  # Gene is the only catalyst for its reaction
        
        S = self.sim.S.S  # stoichiometry matrix (n_mets x n_rxns)
        n_mets, n_rxns = S.shape
        
        # Reaction degree = how many metabolites it touches
        rxn_degree = np.abs(S).sum(axis=0).flatten()
        rxn_degree = np.asarray(rxn_degree).flatten()
        
        # Metabolite degree = how many reactions touch it
        met_degree = (np.abs(S) > 0).sum(axis=1).flatten()
        met_degree = np.asarray(met_degree).flatten()
        
        # Reaction -> genes catalyzing it (inverse of gene_rxns)
        rxn_to_genes = {}
        for gene, rxn_list in self.sim.gene_rxns.items():
            for r in rxn_list:
                rxn_to_genes.setdefault(r, []).append(gene)
        
        for gene, rxn_list in self.sim.gene_rxns.items():
            n = len(rxn_list)
            self.gene_n_rxns[gene] = n
            
            # Centrality: mean metabolite degree of metabolites in gene's reactions
            met_degs = []
            is_only = 0
            for r in rxn_list:
                col = S[:, r]
                col_arr = np.asarray(col.todense()).flatten() if hasattr(col, 'todense') else np.asarray(col).flatten()
                touched = np.where(np.abs(col_arr) > 0)[0]
                for m in touched:
                    met_degs.append(met_degree[m])
                # Singleton: only this gene catalyzes this reaction
                if len(rxn_to_genes.get(r, [])) == 1:
                    is_only += 1
            
            self.gene_centrality[gene] = np.mean(met_degs) if met_degs else 0.0
            self.gene_is_singleton[gene] = is_only / max(n, 1)
    
    def predict(self, gene_id: str) -> Dict:
        """Predict essentiality and extract features for a gene."""
        start = time.perf_counter()
        
        if gene_id not in self.sim.gene_rxns:
            return {
                'gene': gene_id,
                'fba_essential': None,
                'features': None,
                'time_us': 0,
            }
        
        ko = self.sim.knockout(gene_id)
        elapsed_us = (time.perf_counter() - start) * 1e6
        
        # Feature vector (10 dims) — Tier 2 will learn corrections using these
        features = np.array([
            ko['biomass_ratio'],                    # 0: FBA biomass ratio
            float(ko['essential']),                 # 1: FBA binary call
            self.gene_n_rxns[gene_id],              # 2: # reactions gene catalyzes
            self.gene_centrality[gene_id],          # 3: mean metabolite degree
            self.gene_is_singleton[gene_id],        # 4: fraction of reactions where gene is sole catalyst
            float(ko['biomass_ratio'] > 0.99),      # 5: essentially no effect
            float(0.3 < ko['biomass_ratio'] < 0.8), # 6: moderate effect (hard zone)
            float(ko['biomass_ratio'] < 0.01),      # 7: lethal zone
            np.log1p(self.gene_n_rxns[gene_id]),    # 8: log n_rxns
            np.log1p(self.gene_centrality[gene_id]),# 9: log centrality
        ], dtype=np.float32)
        
        return {
            'gene': gene_id,
            'fba_essential': ko['essential'],
            'fba_biomass_ratio': ko['biomass_ratio'],
            'features': features,
            'time_us': elapsed_us,
        }
    
    def predict_all(self, gene_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Batch prediction. Returns (features [N,10], fba_calls [N], timing dict)."""
        feats = []
        fba_calls = []
        total_time = 0
        n_ok = 0
        
        for g in gene_ids:
            r = self.predict(g)
            if r['features'] is None:
                continue
            feats.append(r['features'])
            fba_calls.append(r['fba_essential'])
            total_time += r['time_us']
            n_ok += 1
        
        timing = {
            'total_us': total_time,
            'per_gene_us': total_time / max(n_ok, 1),
            'n_genes': n_ok,
        }
        return np.array(feats), np.array(fba_calls, dtype=bool), timing


def benchmark():
    """Validate V37 accuracy and measure throughput."""
    print("=" * 60)
    print("TIER 1: FBA Endpoint Predictor")
    print("=" * 60)
    
    t1 = Tier1FBA(verbose=True)
    
    # Get labels for Hutchison 2016 genes
    labeled_genes = [g for g in GENE_ESSENTIALITY if g in t1.sim.gene_rxns]
    print(f"\nGenes with FBA coverage AND experimental labels: {len(labeled_genes)}")
    
    y_true = np.array([GENE_ESSENTIALITY[g] in ('E', 'Q') for g in labeled_genes])
    
    feats, fba_calls, timing = t1.predict_all(labeled_genes)
    
    tp = int(((fba_calls == True) & (y_true == True)).sum())
    fp = int(((fba_calls == True) & (y_true == False)).sum())
    tn = int(((fba_calls == False) & (y_true == False)).sum())
    fn = int(((fba_calls == False) & (y_true == True)).sum())
    
    acc = (tp + tn) / len(labeled_genes)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (sens + spec)
    
    print(f"\nTier 1 (FBA alone):")
    print(f"  Accuracy:          {acc*100:.1f}%")
    print(f"  Balanced accuracy: {bal_acc*100:.1f}%")
    print(f"  Sensitivity:       {sens*100:.1f}%")
    print(f"  Specificity:       {spec*100:.1f}%")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Speed: {timing['per_gene_us']:.1f} µs/gene")
    
    return t1, labeled_genes, feats, y_true


if __name__ == '__main__':
    benchmark()
