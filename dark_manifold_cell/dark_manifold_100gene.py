"""
DARK MANIFOLD VIRTUAL CELL - 100 GENE JCVI-syn3A
=================================================

Comprehensive model covering:
- Glycolysis (10 genes)
- Pentose Phosphate Pathway (7 genes)
- Nucleotide Synthesis (15 genes)
- Amino Acid Metabolism (20 genes)
- Lipid Synthesis (10 genes)
- Energy Metabolism (10 genes)
- Transcription/Translation Core (15 genes)
- Transporters (13 genes)

Based on real JCVI-syn3A gene annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# ============================================================================
# JCVI-syn3A GENE DEFINITIONS (100 genes)
# ============================================================================

PATHWAYS = {
    'glycolysis': {
        'genes': ['ptsI', 'ptsH', 'pgi', 'pfkA', 'fba', 'tpiA', 'gapA', 'pgk', 'gpmI', 'eno', 'pyk', 'ldh'],
        'metabolites': ['Glc_ext', 'Glc', 'G6P', 'F6P', 'FBP', 'DHAP', 'GAP', 'BPG', 'PG3', 'PG2', 'PEP', 'Pyr', 'Lac'],
    },
    'pentose_phosphate': {
        'genes': ['zwf', 'pgl', 'gnd', 'rpe', 'rpiA', 'tktA', 'talA'],
        'metabolites': ['G6P', 'PGL', 'PG6', 'Ru5P', 'X5P', 'R5P', 'S7P', 'E4P'],
    },
    'nucleotide_synthesis': {
        'genes': ['purA', 'purB', 'purC', 'purD', 'purE', 'purF', 'purH', 'purK', 'purL', 'purM', 'purN', 'pyrB', 'pyrC', 'pyrD', 'pyrE'],
        'metabolites': ['PRPP', 'IMP', 'AMP', 'GMP', 'UMP', 'CMP', 'dATP', 'dGTP', 'dCTP', 'dTTP'],
    },
    'amino_acid': {
        'genes': ['alaS', 'argS', 'asnS', 'aspS', 'cysS', 'glnS', 'gluS', 'glyS', 'hisS', 'ileS', 
                  'leuS', 'lysS', 'metS', 'pheS', 'proS', 'serS', 'thrS', 'trpS', 'tyrS', 'valS'],
        'metabolites': ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                       'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val'],
    },
    'lipid_synthesis': {
        'genes': ['accA', 'accB', 'accC', 'accD', 'fabD', 'fabF', 'fabG', 'fabH', 'fabI', 'fabZ'],
        'metabolites': ['AcCoA', 'MalCoA', 'ACP', 'FA_C16', 'FA_C18', 'PG', 'CL'],
    },
    'energy': {
        'genes': ['atpA', 'atpB', 'atpC', 'atpD', 'atpE', 'atpF', 'atpG', 'atpH', 'ndk', 'adk'],
        'metabolites': ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NADH', 'NADP', 'NADPH'],
    },
    'transcription_translation': {
        'genes': ['rpoA', 'rpoB', 'rpoC', 'rpoD', 'rpsA', 'rpsB', 'rplA', 'rplB', 'fusA', 'tufA',
                  'tsf', 'infA', 'infB', 'infC', 'prfA'],
        'metabolites': ['NTP_pool', 'AA_pool', 'tRNA_charged', 'mRNA', 'Protein'],
    },
    'transport': {
        'genes': ['glcU', 'nupC', 'potA', 'potB', 'potC', 'potD', 'oppA', 'oppB', 'oppC', 'oppD',
                  'oppF', 'secA', 'secY'],
        'metabolites': ['Glc_ext', 'Nuc_ext', 'Spd_ext', 'AA_ext', 'Peptide_ext'],
    },
}

def build_gene_list():
    """Build complete gene list with indices."""
    genes = []
    gene_to_pathway = {}
    
    for pathway, data in PATHWAYS.items():
        for gene in data['genes']:
            if gene not in genes:
                genes.append(gene)
                gene_to_pathway[gene] = pathway
    
    return genes, gene_to_pathway

def build_metabolite_list():
    """Build complete metabolite list."""
    metabolites = []
    
    for pathway, data in PATHWAYS.items():
        for met in data['metabolites']:
            if met not in metabolites:
                metabolites.append(met)
    
    return metabolites

GENES, GENE_TO_PATHWAY = build_gene_list()
METABOLITES = build_metabolite_list()

print(f"Total genes: {len(GENES)}")
print(f"Total metabolites: {len(METABOLITES)}")


# ============================================================================
# GROUND TRUTH: 100-Gene Metabolic Network
# ============================================================================

class Syn3A100GeneModel:
    """
    Realistic 100-gene metabolic model based on JCVI-syn3A.
    Uses simplified but biologically accurate kinetics.
    """
    
    def __init__(self):
        self.n_genes = len(GENES)
        self.n_mets = len(METABOLITES)
        self.genes = GENES
        self.metabolites = METABOLITES
        
        np.random.seed(42)
        
        # Build stoichiometry matrix
        self.S = self._build_stoichiometry()
        
        # Kinetic parameters
        self.Vmax = np.random.uniform(0.5, 2.0, self.n_genes)
        self.Km = np.random.uniform(0.1, 1.0, self.n_genes)
        
        # Regulatory network (sparse)
        self.W_reg = self._build_regulation()
        
        # Enzyme expression levels
        self.expression = np.random.uniform(0.5, 2.0, self.n_genes)
        
    def _build_stoichiometry(self) -> np.ndarray:
        """Build stoichiometry matrix: S[met, gene] = effect of gene on metabolite."""
        S = np.zeros((self.n_mets, self.n_genes))
        
        # Glycolysis: linear pathway
        glyc_genes = PATHWAYS['glycolysis']['genes']
        glyc_mets = PATHWAYS['glycolysis']['metabolites']
        
        for i, gene in enumerate(glyc_genes):
            if gene in GENES:
                g_idx = GENES.index(gene)
                if i < len(glyc_mets) - 1:
                    sub_idx = METABOLITES.index(glyc_mets[i]) if glyc_mets[i] in METABOLITES else -1
                    prod_idx = METABOLITES.index(glyc_mets[i+1]) if glyc_mets[i+1] in METABOLITES else -1
                    if sub_idx >= 0:
                        S[sub_idx, g_idx] = -1
                    if prod_idx >= 0:
                        S[prod_idx, g_idx] = 1
        
        # ATP production in glycolysis
        atp_idx = METABOLITES.index('ATP') if 'ATP' in METABOLITES else -1
        adp_idx = METABOLITES.index('ADP') if 'ADP' in METABOLITES else -1
        
        if atp_idx >= 0 and adp_idx >= 0:
            for gene in ['pgk', 'pyk']:
                if gene in GENES:
                    g_idx = GENES.index(gene)
                    S[atp_idx, g_idx] = 1
                    S[adp_idx, g_idx] = -1
            
            # pfkA consumes ATP
            if 'pfkA' in GENES:
                g_idx = GENES.index('pfkA')
                S[atp_idx, g_idx] = -1
                S[adp_idx, g_idx] = 1
        
        # Energy genes produce ATP
        for gene in PATHWAYS['energy']['genes']:
            if gene in GENES and 'atp' in gene.lower():
                g_idx = GENES.index(gene)
                if atp_idx >= 0:
                    S[atp_idx, g_idx] = 0.5
                if adp_idx >= 0:
                    S[adp_idx, g_idx] = -0.5
        
        # Add some random connections for other pathways
        for pathway, data in PATHWAYS.items():
            if pathway in ['glycolysis', 'energy']:
                continue
            
            for gene in data['genes']:
                if gene in GENES:
                    g_idx = GENES.index(gene)
                    # Random substrate and product
                    for met in data['metabolites'][:3]:
                        if met in METABOLITES:
                            m_idx = METABOLITES.index(met)
                            S[m_idx, g_idx] = np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.8)
        
        return S
    
    def _build_regulation(self) -> np.ndarray:
        """Build gene regulatory network."""
        W = np.zeros((self.n_genes, self.n_genes))
        
        # Each gene regulated by 2-5 others
        for i in range(self.n_genes):
            n_reg = np.random.randint(2, 6)
            regulators = np.random.choice(self.n_genes, n_reg, replace=False)
            for r in regulators:
                if r != i:
                    W[i, r] = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.5)
        
        # Key regulatory interactions
        # pfkA inhibited by ATP (via atpA expression)
        if 'pfkA' in GENES and 'atpA' in GENES:
            W[GENES.index('pfkA'), GENES.index('atpA')] = -0.5
        
        return W
    
    def dynamics(self, state: np.ndarray, enzyme_mask: np.ndarray = None) -> np.ndarray:
        """Compute system dynamics."""
        if enzyme_mask is None:
            enzyme_mask = np.ones(self.n_genes)
        
        enzymes = state[:self.n_genes] * enzyme_mask
        mets = state[self.n_genes:]
        
        dstate = np.zeros_like(state)
        
        # Enzyme dynamics (slow, based on regulation)
        reg_effect = np.tanh(self.W_reg @ enzymes)
        dstate[:self.n_genes] = 0.01 * (self.expression * (1 + 0.5 * reg_effect) - enzymes)
        
        # Metabolite dynamics
        for g in range(self.n_genes):
            # Find substrate for this enzyme
            substrates = np.where(self.S[:, g] < 0)[0]
            products = np.where(self.S[:, g] > 0)[0]
            
            if len(substrates) > 0:
                # Michaelis-Menten rate
                S_conc = np.mean([mets[s] for s in substrates])
                rate = self.Vmax[g] * enzymes[g] * S_conc / (self.Km[g] + S_conc + 1e-6)
                
                for s in substrates:
                    dstate[self.n_genes + s] -= rate * abs(self.S[s, g])
                for p in products:
                    dstate[self.n_genes + p] += rate * abs(self.S[p, g])
        
        # ATP consumption (background cellular processes)
        atp_idx = METABOLITES.index('ATP') if 'ATP' in METABOLITES else -1
        if atp_idx >= 0:
            dstate[self.n_genes + atp_idx] -= 0.1 * mets[atp_idx]
        
        return dstate
    
    def simulate(self, initial: np.ndarray, n_steps: int = 100, 
                 enzyme_mask: np.ndarray = None, dt: float = 0.05) -> np.ndarray:
        """Simulate the system."""
        traj = [initial.copy()]
        state = initial.copy()
        
        for _ in range(n_steps):
            dstate = self.dynamics(state, enzyme_mask)
            state = state + dt * dstate
            state = np.clip(state, 0.01, 50)
            traj.append(state.copy())
        
        return np.array(traj)
    
    def get_initial_state(self) -> np.ndarray:
        """Get initial state."""
        state = np.zeros(self.n_genes + self.n_mets)
        
        # Enzyme levels
        state[:self.n_genes] = self.expression
        
        # Metabolite levels
        state[self.n_genes:] = np.random.uniform(0.5, 2.0, self.n_mets)
        
        # Set ATP/ADP ratio
        atp_idx = METABOLITES.index('ATP') if 'ATP' in METABOLITES else -1
        adp_idx = METABOLITES.index('ADP') if 'ADP' in METABOLITES else -1
        if atp_idx >= 0:
            state[self.n_genes + atp_idx] = 3.0
        if adp_idx >= 0:
            state[self.n_genes + adp_idx] = 1.0
        
        return state


# ============================================================================
# DARK MANIFOLD MODEL
# ============================================================================

class DarkManifold100Gene(nn.Module):
    """
    Dark Manifold architecture for 100-gene syn3A model.
    """
    
    def __init__(self, n_genes: int, n_mets: int, hidden_dim: int = 256):
        super().__init__()
        self.n_genes = n_genes
        self.n_mets = n_mets
        self.total_dim = n_genes + n_mets
        
        # Dark field encoder
        self.dark_encoder = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Enzyme-metabolite interaction matrix (learnable stoichiometry)
        self.W_stoich = nn.Parameter(torch.zeros(n_mets, n_genes))
        
        # Gene regulatory network
        self.W_reg = nn.Parameter(torch.zeros(n_genes, n_genes))
        
        # Metabolite dynamics network
        self.met_dynamics = nn.Sequential(
            nn.Linear(hidden_dim + n_mets, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_mets),
        )
        
        # Gene expression dynamics
        self.gene_dynamics = nn.Sequential(
            nn.Linear(hidden_dim + n_genes, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_genes),
        )
        
    def forward(self, state: torch.Tensor, enzyme_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        batch = state.shape[0]
        
        if enzyme_mask is None:
            enzyme_mask = torch.ones(batch, self.n_genes, device=state.device)
        
        enzymes = state[:, :self.n_genes] * enzyme_mask
        mets = state[:, self.n_genes:]
        
        # Dark field encoding
        masked_state = torch.cat([enzymes, mets], dim=-1)
        dark = self.dark_encoder(masked_state)
        
        # Enzyme-mediated metabolite changes
        W_s = torch.tanh(self.W_stoich) * 0.5
        stoich_effect = (W_s @ enzymes.T).T  # [batch, n_mets]
        
        # Metabolite dynamics
        met_input = torch.cat([dark, mets], dim=-1)
        dmets = self.met_dynamics(met_input)
        dmets = dmets + stoich_effect * mets  # Substrate-dependent
        
        # Gene expression dynamics (slow)
        W_r = torch.tanh(self.W_reg) * 0.3
        reg_effect = (W_r @ enzymes.T).T
        gene_input = torch.cat([dark, enzymes], dim=-1)
        dgenes = self.gene_dynamics(gene_input)
        dgenes = 0.1 * (dgenes + reg_effect)  # Slow dynamics
        
        # Update
        new_enzymes = enzymes + 0.01 * dgenes
        new_enzymes = F.relu(new_enzymes) + 0.01
        new_enzymes = new_enzymes * enzyme_mask  # Maintain knockout
        
        new_mets = mets + 0.02 * dmets
        new_mets = F.relu(new_mets) + 0.01
        
        return torch.cat([new_enzymes, new_mets], dim=-1)
    
    def simulate(self, initial: torch.Tensor, n_steps: int, 
                 enzyme_mask: torch.Tensor = None) -> torch.Tensor:
        """Simulate trajectory."""
        trajectory = [initial]
        state = initial
        
        for _ in range(n_steps):
            state = self.forward(state, enzyme_mask)
            trajectory.append(state)
        
        return torch.stack(trajectory)
    
    def get_learned_stoichiometry(self) -> np.ndarray:
        """Get learned stoichiometry matrix."""
        return torch.tanh(self.W_stoich).detach().cpu().numpy()
    
    def get_learned_regulation(self) -> np.ndarray:
        """Get learned regulatory network."""
        return torch.tanh(self.W_reg).detach().cpu().numpy()


# ============================================================================
# TRAINING
# ============================================================================

def train_100_gene_model(
    n_epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
):
    """Train the 100-gene Dark Manifold model."""
    
    if verbose:
        print("=" * 70)
        print("DARK MANIFOLD - 100 GENE JCVI-syn3A MODEL")
        print("=" * 70)
        print(f"\nDevice: {device}")
        print(f"Genes: {len(GENES)}")
        print(f"Metabolites: {len(METABOLITES)}")
    
    # Ground truth
    gt = Syn3A100GeneModel()
    
    # Generate training data
    if verbose:
        print("\n[1] GENERATING TRAINING DATA...")
    
    data = []
    initial = gt.get_initial_state()
    
    # Wild-type trajectories
    for _ in range(100):
        init = initial * (1 + 0.2 * np.random.randn(len(initial)))
        init = np.clip(init, 0.1, 20)
        traj = gt.simulate(init, n_steps=50)
        mask = np.ones(gt.n_genes)
        
        for i in range(len(traj) - 1):
            data.append((traj[i], traj[i+1], mask))
    
    # Knockout trajectories (sample of genes)
    ko_genes = list(range(0, gt.n_genes, 5))  # Every 5th gene
    if verbose:
        print(f"   Training knockouts for {len(ko_genes)} genes...")
    
    for ko in ko_genes:
        mask = np.ones(gt.n_genes)
        mask[ko] = 0
        
        for _ in range(10):
            init = initial.copy()
            init[:gt.n_genes] *= mask
            init = init * (1 + 0.1 * np.random.randn(len(init)))
            init = np.clip(init, 0.1, 20)
            
            traj = gt.simulate(init, n_steps=50, enzyme_mask=mask)
            for i in range(len(traj) - 1):
                data.append((traj[i], traj[i+1], mask))
    
    if verbose:
        print(f"   Total samples: {len(data)}")
    
    # Convert to tensors
    inputs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(device)
    targets = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(device)
    masks = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32).to(device)
    
    # Model
    model = DarkManifold100Gene(gt.n_genes, gt.n_mets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    # Training
    if verbose:
        print("\n[2] TRAINING...")
    
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(n_epochs):
        model.train()
        
        # Random batch
        idx = torch.randint(0, len(inputs), (batch_size,))
        batch_in = inputs[idx]
        batch_target = targets[idx]
        batch_mask = masks[idx]
        
        # Forward
        pred = model(batch_in, batch_mask)
        loss = F.mse_loss(pred, batch_target)
        
        # Sparsity regularization
        loss = loss + 0.0001 * (torch.abs(model.W_stoich).mean() + 
                                torch.abs(model.W_reg).mean())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        history['loss'].append(loss.item())
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    return model, gt, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, gt, device='cpu', verbose=True):
    """Evaluate the trained model."""
    
    model.eval()
    model = model.to(device)
    
    results = {
        'trajectory_corr': [],
        'knockout_results': [],
        'atp_predictions': [],
    }
    
    if verbose:
        print("\n[3] EVALUATION")
        print("-" * 50)
    
    # Trajectory prediction
    if verbose:
        print("\n   A. TRAJECTORY PREDICTION")
    
    initial = gt.get_initial_state()
    gt_traj = gt.simulate(initial, n_steps=100)
    
    with torch.no_grad():
        model_traj = model.simulate(
            torch.tensor(initial, dtype=torch.float32).unsqueeze(0).to(device),
            n_steps=100
        ).cpu().squeeze(1).numpy()
    
    for t in [25, 50, 75, 100]:
        gt_mets = gt_traj[t, gt.n_genes:]
        model_mets = model_traj[t, gt.n_genes:]
        corr = np.corrcoef(gt_mets, model_mets)[0, 1]
        results['trajectory_corr'].append(corr)
        if verbose:
            print(f"      t={t}: {corr:.4f}")
    
    avg_traj = np.mean(results['trajectory_corr'])
    if verbose:
        print(f"      Average: {avg_traj:.4f}")
    
    # Knockout predictions
    if verbose:
        print("\n   B. KNOCKOUT PREDICTIONS")
    
    # Test important genes
    test_genes = ['ptsI', 'pfkA', 'pyk', 'atpA', 'rpoB', 'fusA', 'accA', 'purF']
    test_indices = [GENES.index(g) for g in test_genes if g in GENES]
    
    atp_idx = METABOLITES.index('ATP') if 'ATP' in METABOLITES else 0
    
    for ko_idx in test_indices:
        gene_name = GENES[ko_idx]
        
        with torch.no_grad():
            # Wild-type
            wt_traj = gt.simulate(gt.get_initial_state(), n_steps=100)
            wt_atp = wt_traj[-1, gt.n_genes + atp_idx]
            
            # Knockout
            mask = np.ones(gt.n_genes)
            mask[ko_idx] = 0
            ko_init = gt.get_initial_state()
            ko_init[:gt.n_genes] *= mask
            ko_traj = gt.simulate(ko_init, n_steps=100, enzyme_mask=mask)
            ko_atp = ko_traj[-1, gt.n_genes + atp_idx]
            
            gt_delta = ko_atp - wt_atp
            
            # Model
            model_wt = model.simulate(
                torch.tensor(gt.get_initial_state(), dtype=torch.float32).unsqueeze(0).to(device),
                n_steps=100
            )[-1, 0, gt.n_genes + atp_idx].cpu().item()
            
            model_ko = model.simulate(
                torch.tensor(ko_init, dtype=torch.float32).unsqueeze(0).to(device),
                100,
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            )[-1, 0, gt.n_genes + atp_idx].cpu().item()
            
            model_delta = model_ko - model_wt
            
            results['knockout_results'].append({
                'gene': gene_name,
                'gt_atp': gt_delta,
                'model_atp': model_delta,
            })
            
            sign_match = '✓' if np.sign(gt_delta) == np.sign(model_delta) else '✗'
            if verbose:
                print(f"      {gene_name:6s}: GT={gt_delta:+.2f}, Model={model_delta:+.2f} {sign_match}")
    
    # Summary metrics
    gt_atps = np.array([r['gt_atp'] for r in results['knockout_results']])
    model_atps = np.array([r['model_atp'] for r in results['knockout_results']])
    
    if np.std(gt_atps) > 0.01:
        atp_corr = np.corrcoef(gt_atps, model_atps)[0, 1]
    else:
        atp_corr = 0
    
    sign_correct = sum(1 for r in results['knockout_results'] 
                       if np.sign(r['gt_atp']) == np.sign(r['model_atp']))
    
    results['summary'] = {
        'trajectory_corr': avg_traj,
        'atp_corr': atp_corr,
        'sign_accuracy': sign_correct / len(results['knockout_results']),
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  JCVI-syn3A 100-GENE MODEL                                         ║
╠════════════════════════════════════════════════════════════════════╣
║  Trajectory Correlation:     {avg_traj:.3f}                                ║
║  ATP Prediction Correlation: {atp_corr:.3f}                                ║
║  Sign Accuracy:              {sign_correct}/{len(results['knockout_results'])} ({100*results['summary']['sign_accuracy']:.0f}%)                              ║
╚════════════════════════════════════════════════════════════════════╝
""")
        
        if avg_traj > 0.9 and results['summary']['sign_accuracy'] > 0.6:
            print("🟢 SUCCESS: Dark Manifold learns 100-gene syn3A metabolism!")
        elif avg_traj > 0.7:
            print("🟡 PROMISING: Good trajectory prediction, knockout prediction improving")
        else:
            print("🔴 NEEDS MORE TRAINING")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Train
    model, gt, history = train_100_gene_model(
        n_epochs=500,
        batch_size=128,
        verbose=True,
    )
    
    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = evaluate_model(model, gt, device=device, verbose=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'genes': GENES,
        'metabolites': METABOLITES,
        'results': results,
    }, 'dark_manifold_100gene.pt')
    
    print("\n✓ Model saved to dark_manifold_100gene.pt")
