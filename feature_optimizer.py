"""
COMPREHENSIVE FEATURE OPTIMIZATION FOR SOM PREDICTION

This extracts ALL features we've developed and uses educated hit-and-try
to find the optimal combination for Zaretzki dataset.

Strategy:
1. Extract all 27 features for each atom
2. Start with known good combinations
3. Systematically test feature importance
4. Optimize weights using hill climbing
5. Track best model per category
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract all 27 features for each atom in a molecule."""
    
    def __init__(self, mol):
        self.mol = mol
        self.n = mol.GetNumAtoms()
        self._compute_laplacian()
        self._compute_all_features()
    
    def _compute_laplacian(self):
        """Compute graph Laplacian and its eigendecomposition."""
        self.A = np.zeros((self.n, self.n))
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            w = 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
            self.A[i, j] = self.A[j, i] = w
        
        self.L = np.diag(self.A.sum(axis=1)) - self.A
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)
    
    def _compute_all_features(self):
        """Compute all features for all atoms."""
        self.features = {i: {} for i in range(self.n)}
        
        for i in range(self.n):
            atom = self.mol.GetAtomWithIdx(i)
            f = self.features[i]
            
            # Basic info
            f['is_carbon'] = atom.GetAtomicNum() == 6
            f['n_H'] = atom.GetTotalNumHs()
            f['is_aromatic'] = atom.GetIsAromatic()
            f['n_C_neighbors'] = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
            
            # Skip non-carbons for detailed features
            if not f['is_carbon']:
                for key in ['flex', 'amp', 'tunnel', 'topo', 'zpe', 'exchange', 
                           'pi', 'edge', 'vdw', 'pauli', 'jt', 'pjt', 'vib_asym',
                           'homo', 'lumo', 'fukui', 'dual', 'delta_G', 'rate',
                           'steric', 'exposure']:
                    f[key] = 0
                continue
            
            # === GRAPH LAPLACIAN FEATURES ===
            
            # 1. Flexibility
            f['flex'] = 1.0 / (sum(self.eigenvectors[i, k]**2 
                              for k in range(max(1, self.n-3), self.n)) + 0.1)
            
            # 2. Amplitude
            f['amp'] = sum(self.eigenvectors[i, k]**2 / (self.eigenvalues[k] + 0.1) 
                          for k in range(1, min(5, self.n)))
            
            # 3. Tunneling (raw, without alpha boost)
            f['tunnel_raw'] = f['flex'] * (1 + 0.3 * f['n_H'])
            
            # 3b. Tunneling with alpha boost
            tunnel = f['tunnel_raw']
            for nbr in atom.GetNeighbors():
                z = nbr.GetAtomicNum()
                if z == 7: tunnel *= 1.5
                elif z == 8: tunnel *= 1.4
                elif z == 16: tunnel *= 1.3
            f['tunnel'] = tunnel
            
            # 4. Topological charge
            phase = sum(abs(self.eigenvectors[i, k] - self.eigenvectors[j, k]) 
                       for nbr in atom.GetNeighbors() 
                       for j in [nbr.GetIdx()] for k in range(1, min(5, self.n)))
            f['topo'] = 1.0 / (phase + 0.1)
            
            # 5. Zero-point energy
            f['zpe'] = sum(self.eigenvalues[k] * self.eigenvectors[i, k]**2 
                         for k in range(1, self.n))
            
            # 6. Exchange coupling
            exchange = 0
            for bond in atom.GetBonds():
                j = bond.GetOtherAtomIdx(i)
                bo = bond.GetBondTypeAsDouble()
                overlap = sum(self.eigenvectors[i, k] * self.eigenvectors[j, k] 
                             for k in range(1, min(5, self.n)))
                exchange += bo * abs(overlap)
            f['exchange'] = exchange
            
            # 7. π-density
            if f['is_aromatic']:
                f['pi'] = sum(self.eigenvectors[i, k]**2 
                             for k in range(1, self.n) 
                             if 0.1 < self.eigenvalues[k] < 3.0)
            else:
                f['pi'] = 0
            
            # 8. Edge accessibility
            if f['is_aromatic']:
                f['edge'] = sum(1 for nbr in atom.GetNeighbors() 
                               if not nbr.GetIsAromatic())
            else:
                f['edge'] = 0
            
            # 9. van der Waals proxy
            f['vdw'] = sum(1.0 / (abs(i - j) + 1) 
                         for j in range(self.n) 
                         if self.A[i, j] == 0 and i != j)
            
            # 10. Pauli repulsion
            n_heavy = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1)
            f['pauli'] = 1.0 / (1.0 + 0.3 * n_heavy)
            
            # === SYMMETRY BREAKING FEATURES ===
            
            # 11. Jahn-Teller susceptibility
            jt = 0
            for k in range(1, self.n-1):
                for l in range(k+1, self.n):
                    gap = abs(self.eigenvalues[k] - self.eigenvalues[l])
                    if gap < 0.3:
                        jt += (self.eigenvectors[i, k]**2 * 
                               self.eigenvectors[i, l]**2 / (gap + 0.01))
            f['jt'] = jt
            
            # 12. Pseudo-Jahn-Teller
            pjt = 0
            for k in range(1, self.n-1):
                for l in range(k+1, self.n):
                    gap = abs(self.eigenvalues[k] - self.eigenvalues[l])
                    if 0.3 <= gap < 1.0:
                        pjt += (self.eigenvectors[i, k]**2 * 
                                self.eigenvectors[i, l]**2 / (gap + 0.1))
            f['pjt'] = pjt
            
            # 13. Vibrational asymmetry
            f['vib_asym'] = np.std([self.eigenvectors[i, k]**2 
                                   for k in range(1, min(10, self.n))])
            
            # === FMO FEATURES ===
            # Build Hückel Hamiltonian
            H = np.zeros((self.n, self.n))
            for bond in self.mol.GetBonds():
                bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                beta = 1.0 if bond.GetIsAromatic() else 0.7 * bond.GetBondTypeAsDouble()
                H[bi, bj] = H[bj, bi] = -beta
            
            ALPHA = {6: 0.0, 7: -0.5, 8: -1.0, 9: -1.5, 16: -0.3}
            for j in range(self.n):
                z = self.mol.GetAtomWithIdx(j).GetAtomicNum()
                H[j, j] = ALPHA.get(z, 0.0)
            
            h_eig, h_vec = np.linalg.eigh(H)
            n_pi = sum(1 for j in range(self.n) 
                      if self.mol.GetAtomWithIdx(j).GetIsAromatic() or
                      any(b.GetBondTypeAsDouble() > 1 
                          for b in self.mol.GetAtomWithIdx(j).GetBonds()))
            n_occ = max(1, min(n_pi // 2, self.n - 1))
            
            # 14. HOMO density
            f['homo'] = h_vec[i, n_occ - 1]**2
            
            # 15. LUMO density
            f['lumo'] = h_vec[i, n_occ]**2 if n_occ < self.n else 0
            
            # 16. Fukui f⁰
            f['fukui'] = 0.5 * (f['homo'] + f['lumo'])
            
            # 17. Dual descriptor
            f['dual'] = f['lumo'] - f['homo']
            
            # === TST FEATURES ===
            
            # 18. Activation energy proxy
            if f['is_aromatic']:
                base_G = 80
            else:
                if f['n_C_neighbors'] >= 3:
                    base_G = 50  # Tertiary
                elif f['n_C_neighbors'] == 2:
                    base_G = 58  # Secondary
                elif f['n_C_neighbors'] == 1:
                    base_G = 65  # Primary
                else:
                    base_G = 70  # Methyl
            
            for nbr in atom.GetNeighbors():
                z = nbr.GetAtomicNum()
                if z == 7: base_G -= 15
                elif z == 8: base_G -= 12
                elif z == 16: base_G -= 10
                elif nbr.GetIsAromatic() and not f['is_aromatic']: 
                    base_G -= 18
            
            f['delta_G'] = base_G
            
            # 19. Rate
            f['rate'] = np.exp(-base_G / 2.48)
            
            # === ACCESSIBILITY FEATURES ===
            
            # 20. Steric accessibility
            f['steric'] = 1.0 / (1.0 + 0.2 * n_heavy)
            
            # 21. Surface exposure proxy
            f['exposure'] = f['pauli'] * (1 + 0.1 * f['edge'])
    
    def get_feature_vector(self, atom_idx, feature_names):
        """Get normalized feature vector for an atom."""
        return [self.features[atom_idx].get(name, 0) for name in feature_names]
    
    def get_all_normalized(self, feature_names):
        """Get normalized features for all atoms."""
        raw = np.array([[self.features[i].get(name, 0) for name in feature_names] 
                       for i in range(self.n)])
        
        # Normalize each feature
        normalized = np.zeros_like(raw)
        for j in range(len(feature_names)):
            col = raw[:, j]
            r = col.max() - col.min()
            if r > 1e-10:
                normalized[:, j] = (col - col.min()) / r
            else:
                normalized[:, j] = 0.5
        
        return normalized


def score_molecule(mol, feature_names, weights, chem_params):
    """Score all atoms in a molecule using given features and weights."""
    extractor = FeatureExtractor(mol)
    n = mol.GetNumAtoms()
    
    # Get normalized features
    features = extractor.get_all_normalized(feature_names)
    
    scores = np.full(n, -np.inf)
    
    for i in range(n):
        f = extractor.features[i]
        
        if not f['is_carbon']:
            continue
        
        if f['n_H'] == 0 and not f['is_aromatic']:
            continue
        
        # Base score from features
        base = sum(features[i, j] * weights[j] for j in range(len(weights)))
        
        # Chemical multipliers
        atom = mol.GetAtomWithIdx(i)
        
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: alpha_mult = max(alpha_mult, chem_params['aN'])
            elif z == 8: alpha_mult = max(alpha_mult, chem_params['aO'])
            elif z == 16: alpha_mult = max(alpha_mult, chem_params['aS'])
        
        benz_mult = 1.0
        if not f['is_aromatic'] and f['n_H'] > 0:
            if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
                benz_mult = chem_params['benz']
        
        tert_mult = 1.0 + chem_params['tert'] * (f['n_C_neighbors'] - 1) if f['n_C_neighbors'] > 1 else 1.0
        
        h_factor = (1 + chem_params['hf'] * f['n_H']) if f['n_H'] > 0 else 0.5
        
        scores[i] = base * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def evaluate_model(data, feature_names, weights, chem_params):
    """Evaluate model on dataset."""
    t1 = t3 = n = 0
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        
        if not smiles or not sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        try:
            scores = score_molecule(mol, feature_names, weights, chem_params)
        except:
            continue
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])
        
        n += 1
        if ranked[0] in sites:
            t1 += 1
        if any(r in sites for r in ranked[:3]):
            t3 += 1
    
    return t1/n if n > 0 else 0, t3/n if n > 0 else 0, n


def optimize_model(data, feature_names, n_trials=500, verbose=True):
    """Optimize weights and chemical parameters."""
    n_features = len(feature_names)
    
    # Initialize with uniform weights
    best_weights = np.ones(n_features) / n_features
    best_chem = {'aN': 1.7, 'aO': 1.8, 'aS': 1.65, 'benz': 1.5, 'tert': 0.1, 'hf': 0.1}
    best_t1 = 0
    
    for trial in range(n_trials):
        # Perturb weights
        weights = best_weights + np.random.uniform(-0.1, 0.1, n_features)
        weights = np.maximum(weights, 0.01)
        weights /= weights.sum()  # Normalize
        
        # Perturb chemical params
        chem = {k: max(0.01, v + np.random.uniform(-0.15, 0.15)) 
               for k, v in best_chem.items()}
        for k in ['aN', 'aO', 'aS', 'benz']:
            chem[k] = max(1.0, min(3.0, chem[k]))
        
        t1, t3, n = evaluate_model(data, feature_names, weights, chem)
        
        if t1 > best_t1:
            best_t1 = t1
            best_weights = weights.copy()
            best_chem = chem.copy()
            
            if verbose:
                print(f"  Trial {trial}: Top-1={t1*100:.1f}%, Top-3={t3*100:.1f}%")
    
    return best_weights, best_chem, best_t1


if __name__ == '__main__':
    # Load Zaretzki data
    with open('data/curated/merged_cyp3a4_extended.json') as f:
        all_data = json.load(f)['drugs']
    
    zaretzki = [d for d in all_data if d.get('source') == 'Zaretzki']
    print(f"Zaretzki dataset: {len(zaretzki)} molecules")
    
    # Test basic feature extraction
    test_smiles = zaretzki[0]['smiles']
    test_mol = Chem.MolFromSmiles(test_smiles)
    extractor = FeatureExtractor(test_mol)
    
    print(f"\nTest molecule: {test_smiles}")
    print(f"Atoms: {test_mol.GetNumAtoms()}")
    print(f"Features per atom: {len([k for k in extractor.features[0].keys() if k not in ['is_carbon', 'n_H', 'is_aromatic', 'n_C_neighbors']])}")
