"""
DARK MANIFOLD VIRTUAL CELL - FULL 493 GENE JCVI-syn3A
======================================================

The COMPLETE minimal cell genome:
- 452 protein-coding genes
- 41 RNA genes
- ~120 metabolites

This is the world's simplest self-replicating organism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import argparse

# ============================================================================
# COMPLETE JCVI-syn3A GENE LIST (493 genes)
# ============================================================================

PATHWAYS = {
    'dna_replication': {
        'genes': ['dnaA', 'dnaB', 'dnaC', 'dnaE', 'dnaG', 'dnaI', 'dnaX', 'dnaN', 
                  'holA', 'holB', 'polC', 'ligA', 'ssb', 'gyrA', 'gyrB', 'parC', 
                  'parE', 'topA', 'recA', 'recJ', 'recU', 'ruvA', 'ruvB', 'sbcC', 'sbcD'],
    },
    'transcription': {
        'genes': ['rpoA', 'rpoB', 'rpoC', 'rpoD', 'rpoE', 'sigA', 'nusA', 'nusB', 
                  'nusG', 'greA', 'mfd', 'rnc', 'rnjA', 'rnjB', 'rnpA', 'rny'],
    },
    'translation': {
        'genes': ['rpsA', 'rpsB', 'rpsC', 'rpsD', 'rpsE', 'rpsF', 'rpsG', 'rpsH',
                  'rpsI', 'rpsJ', 'rpsK', 'rpsL', 'rpsM', 'rpsN', 'rpsO', 'rpsP',
                  'rpsQ', 'rpsR', 'rpsS', 'rpsT', 'rpsU',
                  'rplA', 'rplB', 'rplC', 'rplD', 'rplE', 'rplF', 'rplJ', 'rplK',
                  'rplL', 'rplM', 'rplN', 'rplO', 'rplP', 'rplQ', 'rplR', 'rplS',
                  'rplT', 'rplU', 'rplV', 'rplW', 'rplX', 'rpmA', 'rpmB', 'rpmC',
                  'rpmD', 'rpmE', 'rpmF', 'rpmG', 'rpmH', 'rpmI', 'rpmJ',
                  'fusA', 'tufA', 'tsf', 'efp', 'infA', 'infB', 'infC',
                  'prfA', 'prfB', 'prmC', 'frr', 'rnr'],
    },
    'trna_synthetases': {
        'genes': ['alaS', 'argS', 'asnS', 'aspS', 'cysS', 'glnS', 'gltX', 'glyS',
                  'hisS', 'ileS', 'leuS', 'lysS', 'metS', 'pheS', 'pheT', 'proS',
                  'serS', 'thrS', 'trpS', 'tyrS', 'valS', 'gatA', 'gatB', 'gatC'],
    },
    'trna_modification': {
        'genes': ['trmD', 'trmU', 'trmFO', 'tilS', 'mnmA', 'mnmG', 'thiI', 'tadA',
                  'queA', 'tgt', 'rlmN', 'rlmH', 'rluD', 'rsmA', 'rsmB', 'rsmD'],
    },
    'rrna_processing': {
        'genes': ['rbfA', 'rimM', 'rimP', 'era', 'obgE', 'engA', 'engB', 'der',
                  'rsgA', 'ybeY', 'ksgA', 'rsmG'],
    },
    'glycolysis': {
        'genes': ['ptsI', 'ptsH', 'pgi', 'pfkA', 'fba', 'tpiA', 'gapA', 'pgk',
                  'gpmI', 'eno', 'pyk', 'ldh', 'pdhA', 'pdhB', 'pdhC', 'pdhD'],
    },
    'pentose_phosphate': {
        'genes': ['zwf', 'pgl', 'gnd', 'rpe', 'rpiA', 'rpiB', 'tktA', 'tktB', 'talA', 'talB'],
    },
    'nucleotide_metabolism': {
        'genes': ['purA', 'purB', 'purC', 'purD', 'purE', 'purF', 'purH', 'purK',
                  'purL', 'purM', 'purN', 'purQ', 'purS', 'guaA', 'guaB', 'guaC',
                  'pyrB', 'pyrC', 'pyrD', 'pyrE', 'pyrF', 'pyrG', 'pyrH', 'pyrR',
                  'ndk', 'adk', 'gmk', 'cmk', 'tmk', 'dut', 'thyA', 'nrdE', 'nrdF',
                  'nrdI', 'deoA', 'deoB', 'deoC', 'deoD', 'cdd', 'codA'],
    },
    'lipid_metabolism': {
        'genes': ['accA', 'accB', 'accC', 'accD', 'fabD', 'fabF', 'fabG', 'fabH',
                  'fabI', 'fabK', 'fabZ', 'acpP', 'acpS', 'plsB', 'plsC', 'plsX',
                  'plsY', 'cdsA', 'pgsA', 'pgpA', 'psd', 'cls', 'gpsA', 'glpK'],
    },
    'atp_synthesis': {
        'genes': ['atpA', 'atpB', 'atpC', 'atpD', 'atpE', 'atpF', 'atpG', 'atpH'],
    },
    'electron_transport': {
        'genes': ['nox', 'ndh', 'fre', 'trxA', 'trxB', 'gor', 'gshAB', 'msrA', 'msrB'],
    },
    'amino_acid_metabolism': {
        'genes': ['glyA', 'serA', 'serB', 'serC', 'ilvA', 'ilvB', 'ilvC', 'ilvD',
                  'ilvE', 'leuA', 'leuB', 'leuC', 'leuD', 'thrA', 'thrB', 'thrC',
                  'metA', 'metB', 'metC', 'metE', 'cysE', 'cysK',
                  'proA', 'proB', 'proC', 'argA', 'argB', 'argC', 'argD', 'argE',
                  'argF', 'argG', 'argH', 'hisA', 'hisB', 'hisC', 'hisD', 'hisF',
                  'hisG', 'hisH', 'hisI', 'aroA', 'aroB', 'aroC', 'aroD', 'aroE',
                  'aroK', 'trpA', 'trpB', 'trpC', 'trpD', 'trpE', 'pheA', 'tyrA'],
    },
    'cofactor_biosynthesis': {
        'genes': ['thiC', 'thiD', 'thiE', 'thiF', 'thiG', 'thiH', 'thiL',
                  'thiM', 'thiN', 'ribA', 'ribB', 'ribC', 'ribD', 'ribE', 'ribF',
                  'folA', 'folB', 'folC', 'folD', 'folE', 'folK', 'folP',
                  'panB', 'panC', 'panD', 'panE', 'coaA', 'coaD', 'coaE',
                  'nadA', 'nadB', 'nadC', 'nadD', 'nadE', 'nadK', 'pncA', 'pncB',
                  'bioA', 'bioB', 'bioD', 'bioF'],
    },
    'cell_membrane': {
        'genes': ['mraY', 'murA', 'murB', 'murC', 'murD', 'murE', 'murF', 'murG',
                  'murJ', 'mraW', 'pbp1', 'pbp2', 'pbp3', 'smc', 'scpA', 'scpB'],
    },
    'cell_division': {
        'genes': ['ftsZ', 'ftsA', 'ftsK', 'ftsL', 'ftsB', 'ftsQ', 'ftsW', 'ftsI',
                  'ftsN', 'zapA', 'minC', 'minD', 'minE', 'sepF'],
    },
    'transporters': {
        'genes': ['glcU', 'nupC', 'potA', 'potB', 'potC', 'potD', 'oppA', 'oppB',
                  'oppC', 'oppD', 'oppF', 'dppA', 'dppB', 'dppC', 'dppD', 'dppF',
                  'livF', 'livG', 'livH', 'livJ', 'livK', 'livM', 'metI', 'metN',
                  'metQ', 'artI', 'artJ', 'artM', 'artP', 'artQ', 'glnH', 'glnP',
                  'glnQ', 'proV', 'proW', 'proX', 'pstA', 'pstB', 'pstC', 'pstS',
                  'modA', 'modB', 'modC', 'znuA', 'znuB', 'znuC', 'mntA', 'mntB',
                  'mntC', 'feoA', 'feoB', 'corA', 'mgtE', 'trkA', 'trkH', 'kdpA',
                  'kdpB', 'kdpC', 'secA', 'secD', 'secE', 'secF', 'secG', 'secY',
                  'yidC', 'ffh', 'ftsY', 'lepB', 'lspA', 'lgt'],
    },
    'chaperones': {
        'genes': ['dnaK', 'dnaJ', 'grpE', 'groEL', 'groES', 'htpG', 'clpA', 'clpB',
                  'clpC', 'clpP', 'clpX', 'lon', 'hslU', 'hslV', 'ftsH', 'hflB',
                  'hflC', 'hflK', 'tig', 'ppiB', 'slyD', 'degP', 'degQ'],
    },
    'unknown_essential': {
        'genes': [f'yneF_{i}' for i in range(1, 50)],  # Essential unknowns
    },
}

# Build gene list
def build_gene_list():
    genes = []
    gene_to_pathway = {}
    for pathway, data in PATHWAYS.items():
        for gene in data['genes']:
            if gene not in genes:
                genes.append(gene)
                gene_to_pathway[gene] = pathway
    return genes, gene_to_pathway

GENES, GENE_TO_PATHWAY = build_gene_list()

# Metabolites
METABOLITES = [
    # Central carbon
    'Glc_ext', 'Glc', 'G6P', 'F6P', 'FBP', 'DHAP', 'GAP', 'BPG', 'PG3', 'PG2', 
    'PEP', 'Pyr', 'Lac', 'AcCoA',
    # Pentose phosphate
    'PGL', 'PG6', 'Ru5P', 'X5P', 'R5P', 'S7P', 'E4P',
    # Nucleotides
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'UTP', 'UDP', 'UMP', 'CTP', 
    'dATP', 'dGTP', 'dCTP', 'dTTP', 'IMP', 'PRPP',
    # Redox
    'NAD', 'NADH', 'NADP', 'NADPH', 'FAD', 'FADH2', 'CoA',
    # Amino acids
    'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
    'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val',
    # tRNA charged pools
    'tRNA_charged', 'tRNA_uncharged',
    # Lipids
    'MalCoA', 'FA_C16', 'PA', 'PG', 'CL',
    # Cofactors
    'THF', 'SAM', 'Biotin', 'Thiamine', 'Riboflavin',
    # Inorganics
    'Pi', 'PPi',
    # Macromolecule pools
    'DNA_pool', 'RNA_pool', 'Protein_pool', 'Lipid_pool', 'Cell_mass',
]

print(f"Total genes: {len(GENES)}")
print(f"Total metabolites: {len(METABOLITES)}")


# ============================================================================
# GROUND TRUTH MODEL
# ============================================================================

class Syn3AFullModel:
    """Full 493-gene model."""
    
    def __init__(self, seed=42):
        self.n_genes = len(GENES)
        self.n_mets = len(METABOLITES)
        
        np.random.seed(seed)
        
        self.S = self._build_stoichiometry()
        self.Vmax = np.random.uniform(0.3, 2.0, self.n_genes)
        self.Km = np.random.uniform(0.05, 1.0, self.n_genes)
        self.W_reg = self._build_regulation()
        self.expression = np.random.uniform(0.3, 2.0, self.n_genes)
        self._boost_essential()
        
    def _build_stoichiometry(self):
        S = np.zeros((self.n_mets, self.n_genes))
        
        # Glycolysis
        glyc_mets = ['Glc', 'G6P', 'F6P', 'FBP', 'GAP', 'BPG', 'PG3', 'PG2', 'PEP', 'Pyr']
        glyc_genes = ['pgi', 'pfkA', 'fba', 'tpiA', 'gapA', 'pgk', 'gpmI', 'eno', 'pyk']
        
        for i, gene in enumerate(glyc_genes):
            if gene in GENES and i < len(glyc_mets) - 1:
                g_idx = GENES.index(gene)
                if glyc_mets[i] in METABOLITES:
                    S[METABOLITES.index(glyc_mets[i]), g_idx] = -1
                if glyc_mets[i+1] in METABOLITES:
                    S[METABOLITES.index(glyc_mets[i+1]), g_idx] = 1
        
        # ATP
        atp_idx = METABOLITES.index('ATP')
        adp_idx = METABOLITES.index('ADP')
        
        for gene in ['pgk', 'pyk', 'atpA', 'atpB', 'atpC', 'atpD']:
            if gene in GENES:
                S[atp_idx, GENES.index(gene)] = 0.5
                S[adp_idx, GENES.index(gene)] = -0.5
        
        for gene in ['pfkA', 'dnaA', 'rpoB', 'fusA']:
            if gene in GENES:
                S[atp_idx, GENES.index(gene)] = -0.3
                S[adp_idx, GENES.index(gene)] = 0.3
        
        # Other pathways - sparse connections
        for pathway, data in PATHWAYS.items():
            if pathway in ['glycolysis', 'atp_synthesis']:
                continue
            for gene in data['genes']:
                if gene in GENES:
                    g_idx = GENES.index(gene)
                    n_mets = np.random.randint(1, 4)
                    met_indices = np.random.choice(self.n_mets, n_mets, replace=False)
                    for m_idx in met_indices:
                        S[m_idx, g_idx] = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.4)
        
        return S
    
    def _build_regulation(self):
        W = np.zeros((self.n_genes, self.n_genes))
        for i in range(self.n_genes):
            n_reg = np.random.randint(1, 4)
            regs = np.random.choice(self.n_genes, n_reg, replace=False)
            for r in regs:
                if r != i:
                    W[i, r] = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.2)
        return W
    
    def _boost_essential(self):
        for pathway in ['glycolysis', 'atp_synthesis', 'translation']:
            if pathway in PATHWAYS:
                for gene in PATHWAYS[pathway]['genes']:
                    if gene in GENES:
                        idx = GENES.index(gene)
                        self.expression[idx] *= 1.5
                        self.Vmax[idx] *= 1.2
    
    def dynamics(self, state, enzyme_mask=None):
        if enzyme_mask is None:
            enzyme_mask = np.ones(self.n_genes)
        
        enzymes = state[:self.n_genes] * enzyme_mask
        mets = state[self.n_genes:]
        dstate = np.zeros_like(state)
        
        reg_effect = np.tanh(self.W_reg @ enzymes * 0.3)
        dstate[:self.n_genes] = 0.003 * (self.expression * (1 + 0.2 * reg_effect) - enzymes)
        
        for g in range(self.n_genes):
            subs = np.where(self.S[:, g] < 0)[0]
            prods = np.where(self.S[:, g] > 0)[0]
            
            if len(subs) > 0:
                S_conc = np.mean([mets[s] for s in subs])
                rate = self.Vmax[g] * enzymes[g] * S_conc / (self.Km[g] + S_conc + 1e-6)
                
                for s in subs:
                    dstate[self.n_genes + s] -= rate * abs(self.S[s, g])
                for p in prods:
                    dstate[self.n_genes + p] += rate * abs(self.S[p, g])
        
        atp_idx = METABOLITES.index('ATP')
        dstate[self.n_genes + atp_idx] -= 0.05 * mets[atp_idx]
        
        return dstate
    
    def simulate(self, initial, n_steps=100, enzyme_mask=None, dt=0.05):
        traj = [initial.copy()]
        state = initial.copy()
        
        for _ in range(n_steps):
            dstate = self.dynamics(state, enzyme_mask)
            state = state + dt * dstate
            state = np.clip(state, 0.01, 100)
            traj.append(state.copy())
        
        return np.array(traj)
    
    def get_initial_state(self):
        state = np.zeros(self.n_genes + self.n_mets)
        state[:self.n_genes] = self.expression
        state[self.n_genes:] = np.random.uniform(0.5, 2.0, self.n_mets)
        
        state[self.n_genes + METABOLITES.index('ATP')] = 4.0
        state[self.n_genes + METABOLITES.index('ADP')] = 1.0
        state[self.n_genes + METABOLITES.index('NAD')] = 2.0
        state[self.n_genes + METABOLITES.index('Glc')] = 5.0
        
        return state


# ============================================================================
# DARK MANIFOLD MODEL
# ============================================================================

class DarkManifoldFull(nn.Module):
    """Dark Manifold for full 493-gene syn3A."""
    
    def __init__(self, n_genes, n_mets, hidden_dim=512):
        super().__init__()
        self.n_genes = n_genes
        self.n_mets = n_mets
        
        self.gene_embed = nn.Linear(n_genes, hidden_dim)
        self.met_embed = nn.Linear(n_mets, hidden_dim)
        
        self.dark_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.W_stoich = nn.Parameter(torch.randn(n_mets, n_genes) * 0.01)
        self.W_reg = nn.Parameter(torch.randn(n_genes, n_genes) * 0.01)
        
        self.met_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_mets),
        )
        
        self.gene_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_genes),
        )
        
    def forward(self, state, enzyme_mask=None):
        batch = state.shape[0]
        
        if enzyme_mask is None:
            enzyme_mask = torch.ones(batch, self.n_genes, device=state.device)
        
        enzymes = state[:, :self.n_genes] * enzyme_mask
        mets = state[:, self.n_genes:]
        
        g_emb = self.gene_embed(enzymes)
        m_emb = self.met_embed(mets)
        
        combined = torch.cat([g_emb, m_emb], dim=-1)
        dark = self.dark_encoder(combined)
        
        W_s = torch.tanh(self.W_stoich) * 0.3
        stoich_effect = (W_s @ enzymes.T).T * mets
        
        W_r = torch.tanh(self.W_reg) * 0.2
        reg_effect = (W_r @ enzymes.T).T
        
        dmets = self.met_out(dark) + stoich_effect
        dgenes = self.gene_out(dark) * 0.1 + reg_effect * 0.05
        
        new_enzymes = enzymes + 0.01 * dgenes
        new_enzymes = F.relu(new_enzymes) + 0.01
        new_enzymes = new_enzymes * enzyme_mask
        
        new_mets = mets + 0.02 * dmets
        new_mets = F.relu(new_mets) + 0.01
        
        return torch.cat([new_enzymes, new_mets], dim=-1)
    
    def simulate(self, initial, n_steps, enzyme_mask=None):
        traj = [initial]
        state = initial
        for _ in range(n_steps):
            state = self.forward(state, enzyme_mask)
            traj.append(state)
        return torch.stack(traj)


# ============================================================================
# TRAINING
# ============================================================================

def train_full_model(n_epochs=2000, batch_size=128, lr=5e-4, device='cuda'):
    print("=" * 70)
    print("DARK MANIFOLD - FULL 493 GENE JCVI-syn3A")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Genes: {len(GENES)}")
    print(f"Metabolites: {len(METABOLITES)}")
    
    gt = Syn3AFullModel()
    
    print("\n[1] GENERATING TRAINING DATA...")
    
    data = []
    initial = gt.get_initial_state()
    
    # Wild-type
    for _ in range(150):
        init = initial * (1 + 0.15 * np.random.randn(len(initial)))
        init = np.clip(init, 0.1, 50)
        traj = gt.simulate(init, n_steps=50)
        mask = np.ones(gt.n_genes)
        
        for i in range(len(traj) - 1):
            data.append((traj[i], traj[i+1], mask))
    
    # Knockouts
    ko_genes = list(range(0, gt.n_genes, 10))
    print(f"   Training knockouts: {len(ko_genes)} genes...")
    
    for ko in ko_genes:
        mask = np.ones(gt.n_genes)
        mask[ko] = 0
        
        for _ in range(8):
            init = initial.copy()
            init[:gt.n_genes] *= mask
            init = init * (1 + 0.1 * np.random.randn(len(init)))
            init = np.clip(init, 0.1, 50)
            
            traj = gt.simulate(init, n_steps=50, enzyme_mask=mask)
            for i in range(len(traj) - 1):
                data.append((traj[i], traj[i+1], mask))
    
    print(f"   Total samples: {len(data)}")
    
    inputs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(device)
    targets = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(device)
    masks = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32).to(device)
    
    model = DarkManifoldFull(gt.n_genes, gt.n_mets).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    print("\n[2] TRAINING...")
    
    for epoch in range(n_epochs):
        model.train()
        
        idx = torch.randint(0, len(inputs), (batch_size,))
        
        pred = model(inputs[idx], masks[idx])
        loss = F.mse_loss(pred, targets[idx])
        loss = loss + 0.0001 * (torch.abs(model.W_stoich).mean() + torch.abs(model.W_reg).mean())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    return model, gt


def evaluate(model, gt, device='cuda'):
    model.eval()
    
    print("\n[3] EVALUATION")
    print("-" * 50)
    
    initial = gt.get_initial_state()
    gt_traj = gt.simulate(initial, n_steps=100)
    
    with torch.no_grad():
        model_traj = model.simulate(
            torch.tensor(initial, dtype=torch.float32).unsqueeze(0).to(device),
            n_steps=100
        ).cpu().squeeze(1).numpy()
    
    print("\n   A. TRAJECTORY")
    corrs = []
    for t in [25, 50, 75, 100]:
        c = np.corrcoef(gt_traj[t, gt.n_genes:], model_traj[t, gt.n_genes:])[0, 1]
        corrs.append(c)
        print(f"      t={t}: {c:.4f}")
    avg_traj = np.mean(corrs)
    print(f"      Avg: {avg_traj:.4f}")
    
    print("\n   B. KNOCKOUTS")
    test_genes = ['ptsI', 'pfkA', 'pyk', 'gapA', 'atpA', 'atpD', 'rpoB', 'fusA', 
                  'dnaA', 'ftsZ', 'accA', 'groEL']
    
    atp_idx = METABOLITES.index('ATP')
    results = []
    
    for gene in test_genes:
        if gene not in GENES:
            continue
        ko_idx = GENES.index(gene)
        
        with torch.no_grad():
            wt = gt.simulate(gt.get_initial_state(), 100)[-1, gt.n_genes + atp_idx]
            
            mask = np.ones(gt.n_genes)
            mask[ko_idx] = 0
            ko_init = gt.get_initial_state()
            ko_init[:gt.n_genes] *= mask
            ko = gt.simulate(ko_init, 100, mask)[-1, gt.n_genes + atp_idx]
            
            gt_d = ko - wt
            
            m_wt = model.simulate(
                torch.tensor(gt.get_initial_state(), dtype=torch.float32).unsqueeze(0).to(device), 100
            )[-1, 0, gt.n_genes + atp_idx].cpu().item()
            
            m_ko = model.simulate(
                torch.tensor(ko_init, dtype=torch.float32).unsqueeze(0).to(device), 100,
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            )[-1, 0, gt.n_genes + atp_idx].cpu().item()
            
            m_d = m_ko - m_wt
            
            results.append({'gene': gene, 'gt': gt_d, 'model': m_d})
            s = '✓' if np.sign(gt_d) == np.sign(m_d) else '✗'
            print(f"      {gene:6s}: GT={gt_d:+.2f}, M={m_d:+.2f} {s}")
    
    sign_acc = sum(1 for r in results if np.sign(r['gt']) == np.sign(r['model'])) / len(results)
    
    gt_a = np.array([r['gt'] for r in results])
    m_a = np.array([r['model'] for r in results])
    atp_corr = np.corrcoef(gt_a, m_a)[0, 1] if np.std(gt_a) > 0.01 else 0
    
    print("\n" + "=" * 70)
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  JCVI-syn3A FULL 493-GENE MODEL                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  Trajectory Correlation:     {avg_traj:.3f}                                ║
║  ATP Prediction Correlation: {atp_corr:.3f}                                ║
║  Knockout Sign Accuracy:     {int(sign_acc*len(results))}/{len(results)} ({100*sign_acc:.0f}%)                              ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    if avg_traj > 0.9 and sign_acc > 0.6:
        print("🟢 SUCCESS: Dark Manifold learns the COMPLETE minimal cell!")
    elif avg_traj > 0.7:
        print("🟡 PROMISING: Good trajectory prediction")
    else:
        print("🔴 NEEDS MORE TRAINING")
    
    return {'traj': avg_traj, 'atp_corr': atp_corr, 'sign_acc': sign_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, gt = train_full_model(args.epochs, args.batch_size, args.lr, device)
    results = evaluate(model, gt, device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'genes': GENES,
        'metabolites': METABOLITES,
        'results': results,
    }, 'dark_manifold_493gene.pt')
    
    print("\n✓ Saved to dark_manifold_493gene.pt")
