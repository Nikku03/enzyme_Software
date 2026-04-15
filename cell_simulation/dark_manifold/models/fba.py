"""
dark_manifold/models/fba.py

Flux Balance Analysis (FBA) model for JCVI-syn3A.

This is the SACRED V37 core - DO NOT MODIFY without validation.
Achieves 85.6% accuracy on Hutchison 2016 essentiality data.

Based on iMB155 reconstruction (Breuer et al. 2019 eLife):
- 304 metabolites
- 338 reactions  
- 155 genes

IMPORTANT: This module wraps the original V37 implementation.
Do not reimplement - the exact parameters matter for accuracy.

Author: Naresh Chhillar, 2026
"""

import sys
import os
from typing import Dict, List, Optional

# Add V37 to path for import
_v37_path = os.path.join(os.path.dirname(__file__), '..', '..', 'v37_full_imb155')
_v37_path = os.path.abspath(_v37_path)
if _v37_path not in sys.path:
    sys.path.insert(0, _v37_path)

# Import the original V37 implementation
from core_cell_simulator import CoreCellSimulator


class FBAModel:
    """
    Flux Balance Analysis model for gene essentiality prediction.
    
    SACRED V37 CORE - 85.6% accuracy on Hutchison 2016 data.
    
    This is a thin wrapper around the original V37 CoreCellSimulator.
    DO NOT MODIFY the underlying implementation.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize FBA model."""
        # Suppress verbose output if needed
        if not verbose:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self._core = CoreCellSimulator()
        else:
            self._core = CoreCellSimulator()
        
        # Expose key attributes
        self.gene_to_rxns = self._core.gene_rxns
        self.wt_biomass = self._core.wt_biomass
        self.stoich = self._core.S
    
    def knockout(self, gene: str) -> Dict:
        """
        Simulate single gene knockout.
        
        Returns dict with:
            - gene: gene ID
            - essential: True if knockout is lethal
            - viable: True if knockout survives
            - biomass: absolute biomass flux
            - biomass_ratio: relative to wild-type
            - time_ms: computation time
        """
        return self._core.knockout(gene)
    
    def double_knockout(self, gene_a: str, gene_b: str) -> Dict:
        """
        Simulate double gene knockout for synthetic lethality.
        
        Note: This uses sequential knockouts. For true epistasis,
        would need proper GPR logic, but this is sufficient for
        FBA-based synthetic lethality screening.
        """
        import time
        start = time.time()
        
        # Get flux with both genes knocked out
        flux = self._core._solve_fba([gene_a, gene_b])
        elapsed_ms = (time.time() - start) * 1000
        
        if flux is None:
            biomass = 0
        else:
            biomass = flux[self._core.obj_idx] if self._core.obj_idx >= 0 else 0
        
        ratio = biomass / self.wt_biomass if self.wt_biomass > 0 else 0
        essential = biomass < 0.01 * self.wt_biomass
        
        return {
            'gene_a': gene_a,
            'gene_b': gene_b,
            'viable': not essential,
            'essential': essential,
            'biomass': biomass,
            'biomass_ratio': ratio,
            'time_ms': elapsed_ms,
        }
    
    def get_genes(self) -> List[str]:
        """Get list of all genes in the model."""
        return list(self.gene_to_rxns.keys())
    
    def get_reactions_for_gene(self, gene: str) -> List[str]:
        """Get reaction IDs catalyzed by a gene."""
        if gene not in self.gene_to_rxns:
            return []
        return [self.stoich.rxn_ids[i] for i in self.gene_to_rxns[gene]]
    
    def run_all_knockouts(self) -> Dict:
        """Run knockouts and compute accuracy vs experimental data."""
        return self._core.run_all_knockouts()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_cached_model: Optional[FBAModel] = None

def get_fba_model(verbose: bool = False) -> FBAModel:
    """Get or create cached FBA model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = FBAModel(verbose=verbose)
    return _cached_model


def predict_essentiality(gene: str) -> bool:
    """Quick essentiality prediction for a single gene."""
    model = get_fba_model()
    result = model.knockout(gene)
    return result['essential']


if __name__ == "__main__":
    # Quick test
    print("Testing FBA Model wrapper...")
    model = FBAModel()
    
    print("\nTesting knockouts:")
    result = model.knockout('JCVISYN3A_0207')  # pfkA
    print(f"  pfkA: {'ESSENTIAL' if result['essential'] else 'viable'} (ratio: {result['biomass_ratio']:.2%})")
    
    result = model.knockout('JCVISYN3A_0449')  # ldh
    print(f"  ldh: {'ESSENTIAL' if result['essential'] else 'viable'} (ratio: {result['biomass_ratio']:.2%})")
    
    print("\nTesting double knockout:")
    result = model.double_knockout('JCVISYN3A_0449', 'JCVISYN3A_0485')  # ldh + ackA
    print(f"  ldh + ackA: {'ESSENTIAL' if result['essential'] else 'viable'}")
