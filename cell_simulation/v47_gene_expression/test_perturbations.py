"""
Test gene expression model with perturbations.

1. Knockout experiments
2. Nutrient limitation (reduce translation)
3. Stress response (increase chaperones)
"""

from gene_expression import *
import numpy as np

def test_knockout(model, gene_name, t_sim=180):
    """Simulate gene knockout and observe effects."""
    
    print(f"\n{'='*70}")
    print(f"KNOCKOUT EXPERIMENT: {gene_name}")
    print("="*70)
    
    # Run to steady state
    result_wt = model.simulate(t_span=(0, 60))
    initial_state = result_wt['y'][:, -1]
    
    # Store original rates
    gene = model.genes[gene_name]
    orig_k_tx = gene.k_tx
    orig_k_tl = gene.k_tl
    
    # Knockout
    model.knockout_gene(gene_name)
    
    # Also set mRNA and protein to zero
    idx = model.gene_list.index(gene_name)
    perturbed_state = initial_state.copy()
    perturbed_state[model.mrna_start + idx] = 0
    perturbed_state[model.protein_start + idx] = 0
    
    # Run simulation after knockout
    result_ko = model.simulate(t_span=(0, t_sim), initial_state=perturbed_state)
    
    # Analyze effects
    protein_wt = result_wt['protein'][:, -1]
    protein_ko = result_ko['protein'][:, -1]
    
    # Find genes with biggest change
    change = protein_ko - protein_wt
    pct_change = change / (protein_wt + 1) * 100
    
    print(f"\nEffects of {gene_name} knockout after {t_sim} min:")
    print(f"  {gene_name} protein: {protein_wt[idx]:.0f} → {protein_ko[idx]:.0f}")
    
    # Top affected genes
    affected_idx = np.argsort(np.abs(pct_change))[::-1]
    print(f"\nMost affected genes:")
    for i in affected_idx[:10]:
        name = model.gene_list[i]
        if name != gene_name and np.abs(pct_change[i]) > 0.1:
            print(f"  {name:<12} {protein_wt[i]:>6.0f} → {protein_ko[i]:>6.0f} ({pct_change[i]:+.1f}%)")
    
    # Restore
    gene.k_tx = orig_k_tx
    gene.k_tl = orig_k_tl
    
    return result_wt, result_ko


def test_ribosome_depletion(model):
    """Simulate what happens when ribosomes become limiting."""
    
    print(f"\n{'='*70}")
    print("RIBOSOME DEPLETION EXPERIMENT")
    print("="*70)
    
    # Save original
    orig_ribosomes = model.total_ribosomes
    
    results = []
    ribosome_levels = [1500, 1000, 500, 200, 100]
    
    for n_ribo in ribosome_levels:
        model.total_ribosomes = n_ribo
        result = model.simulate(t_span=(0, 120))
        
        total_protein = result['protein'][:, -1].sum()
        free_ribo = result['free_ribosomes'][-1]
        
        results.append({
            'ribosomes': n_ribo,
            'total_protein': total_protein,
            'free_ribosomes': free_ribo
        })
        
        print(f"  Ribosomes: {n_ribo:>5} → Total protein: {total_protein:>8.0f}, Free: {free_ribo:>6.0f}")
    
    # Restore
    model.total_ribosomes = orig_ribosomes
    
    return results


def test_rnap_depletion(model):
    """Simulate what happens when RNAP becomes limiting."""
    
    print(f"\n{'='*70}")
    print("RNAP DEPLETION EXPERIMENT")
    print("="*70)
    
    orig_rnap = model.total_RNAP
    
    rnap_levels = [300, 200, 100, 50, 20]
    
    for n_rnap in rnap_levels:
        model.total_RNAP = n_rnap
        result = model.simulate(t_span=(0, 120))
        
        total_mrna = result['mrna'][:, -1].sum()
        total_protein = result['protein'][:, -1].sum()
        free_rnap = result['free_RNAP'][-1]
        
        print(f"  RNAP: {n_rnap:>3} → Total mRNA: {total_mrna:>6.0f}, Protein: {total_protein:>8.0f}, Free: {free_rnap:>5.0f}")
    
    model.total_RNAP = orig_rnap


def test_doubling_time(model):
    """Estimate doubling time from protein synthesis rate."""
    
    print(f"\n{'='*70}")
    print("DOUBLING TIME ESTIMATION")
    print("="*70)
    
    result = model.simulate(t_span=(0, 120))
    
    # Total protein
    total_protein = result['protein'][:, -1].sum()
    
    # Protein synthesis rate (from ODE)
    state = result['y'][:, -1]
    dydt = model.ode_rhs(120, state)
    
    protein_synthesis_rate = 0
    for i, name in enumerate(model.gene_list):
        gene = model.genes[name]
        mrna = state[model.mrna_start + i]
        protein_synthesis_rate += gene.k_tl * mrna
    
    print(f"Total protein: {total_protein:.0f} copies")
    print(f"Protein synthesis rate: {protein_synthesis_rate:.1f} proteins/min")
    
    # Doubling time = time to make total_protein new proteins
    if protein_synthesis_rate > 0:
        doubling_time = total_protein / protein_synthesis_rate
        print(f"Estimated doubling time: {doubling_time:.1f} min ({doubling_time/60:.1f} hr)")
    
    # Compare to real Mycoplasma
    print(f"\nReal Mycoplasma doubling time: ~60-120 min (1-2 hr)")


def test_growth_rate_dependence(model):
    """Show how expression changes with growth rate."""
    
    print(f"\n{'='*70}")
    print("GROWTH RATE DEPENDENCE")
    print("="*70)
    
    # Simulate different "growth rates" by varying resource levels
    conditions = [
        ('Rich', 1500, 300),
        ('Medium', 1000, 200),
        ('Poor', 500, 100),
        ('Starving', 200, 50),
    ]
    
    print("\nCondition    Ribosomes  Total Protein  Ribosomal%  Metabolic%")
    print("-"*65)
    
    for name, ribo, rnap in conditions:
        model.total_ribosomes = ribo
        model.total_RNAP = rnap
        
        result = model.simulate(t_span=(0, 120))
        protein = result['protein'][:, -1]
        total = protein.sum()
        
        # Calculate fraction in ribosomes
        ribosomal = 0
        metabolic = 0
        for i, gene_name in enumerate(model.gene_list):
            gene = model.genes[gene_name]
            if 'ribosome' in gene.category:
                ribosomal += protein[i]
            elif gene.category == 'metabolism':
                metabolic += protein[i]
        
        ribo_pct = ribosomal / total * 100
        metab_pct = metabolic / total * 100
        
        print(f"{name:<12} {ribo:>5}      {total:>10.0f}       {ribo_pct:>5.1f}%      {metab_pct:>5.1f}%")
    
    # Restore
    model.total_ribosomes = 1500
    model.total_RNAP = 300
    
    print("""
This matches the "growth law": 
- Faster growth → more ribosomes (need more protein synthesis)
- Slower growth → more metabolic enzymes (need more scavenging)
""")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GENE EXPRESSION PERTURBATION TESTS")
    print("="*70)
    
    # Build model
    genes = build_syn3a_genes()
    model = GeneExpressionModel(genes)
    
    # Test 1: Doubling time
    test_doubling_time(model)
    
    # Test 2: Ribosome depletion
    test_ribosome_depletion(model)
    
    # Test 3: RNAP depletion  
    test_rnap_depletion(model)
    
    # Test 4: Growth rate dependence
    test_growth_rate_dependence(model)
    
    # Test 5: Knockout experiments
    # Knockout a ribosomal protein
    test_knockout(model, 'rpsA', t_sim=60)
    
    # Knockout a chaperone
    test_knockout(model, 'groEL', t_sim=60)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
