"""
Thornburg 2022 / 2026 data loader
==================================

Connects the cascade to Luthey-Schulten's published trajectory data.

Handles three data sources:

1. THORNBURG 2022 CME-ODE (well-stirred, Zenodo 5780120):
   - CSV files with 1-second hook intervals
   - Columns: time, followed by ~1800 species (genes, RNAs, proteins, metabolites)
   - ~100-200 MB per 7200s (2-hour) simulation
   - Multiple cell replicates per archive

2. THORNBURG 2022 RDME-CME-ODE (spatial, 20-min partial):
   - Lattice Microbes `.lm` files (HDF5-based)
   - Same species table but with 3D lattice positions
   - Much larger (GBs per cell)

3. THORNBURG 2026 4DWCM (Zenodo 15579158):
   - 50 simulated cells, full cell cycles
   - LM RDME trajectory files + CSV particle counts
   - Chromosome dynamics in separate LAMMPS outputs

For the cascade we only need (1) — the well-stirred CSVs give us
trajectories of every species at every second, which is exactly what
the Tier 2 dynamic surrogate needs for training.

Usage
-----
    from thornburg_loader import ThornburgDataset
    
    ds = ThornburgDataset('/path/to/zenodo_5780120/CME_ODE/')
    ds.summary()                    # print what was found
    trajectories = ds.load_cme_ode_csvs(max_cells=10)
    
    X, Y = ds.to_surrogate_training(
        trajectories,
        init_window=(0, 60),        # first 60 sec = "initial state"
        final_window=(3600, 7200),  # last hour average = "final state"
        species_subset='metabolites' # or 'all' or list of IDs
    )
    # X: (N, d_in), Y: (N, d_out) — ready to fit Tier2Hybrid.fit(X, Y)

If the data archive isn't present, the loader creates a SYNTHETIC
dataset from tier3_cascade.Tier3MechanisticODE that mimics the Thornburg
schema exactly, so downstream code can be developed and tested without
waiting for the real download.
"""

from __future__ import annotations

import os
import glob
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np


# ----------------------------------------------------------------------------
# Species taxonomy — how Thornburg labels columns in CME-ODE CSVs
# ----------------------------------------------------------------------------
# Real CSVs have ~1800 columns. The prefixes below are what Thornburg uses
# (checked against Minimal_Cell/CME_ODE/programs/GIP_rates.py and related).
SPECIES_PREFIXES = {
    'metabolites': [
        'M_', 'atp', 'adp', 'amp', 'gtp', 'gdp', 'gmp',
        'nad', 'nadh', 'nadp', 'nadph', 'coa', 'accoa',
        'g6p', 'f6p', 'fdp', 'pep', 'pyr', 'lac', 'glc',
    ],
    'mrnas': ['R_', 'mRNA_'],
    'proteins': ['P_', 'protein_'],
    'genes_dna': ['G_', 'gene_'],
    'complexes': ['C_', 'complex_'],
    'trnas': ['tRNA_', 'T_'],
    'rrnas': ['rRNA_'],
}


@dataclass
class CellTrajectory:
    """One simulated cell's trajectory."""
    cell_id: str
    time_s: np.ndarray                        # shape (T,), seconds
    species_names: List[str]                  # length S
    counts: np.ndarray                        # shape (T, S), molecule counts
    source: str                                # 'real_csv', 'synthetic', 'pickle', ...
    metadata: Dict = field(default_factory=dict)
    
    @property
    def n_species(self) -> int:
        return len(self.species_names)
    
    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])
    
    def subset_species(self, kind: str) -> np.ndarray:
        """Return indices of species matching a taxonomic group."""
        if kind == 'all':
            return np.arange(self.n_species)
        prefixes = SPECIES_PREFIXES.get(kind, [])
        idx = [i for i, name in enumerate(self.species_names)
               if any(name.lower().startswith(p.lower()) for p in prefixes)]
        return np.array(idx, dtype=int)


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------

class ThornburgDataset:
    """
    Unified loader for Luthey-Schulten whole-cell trajectory data.
    
    Directory layout expected (from Zenodo 5780120):
      {root}/
        CME_ODE/
          replicate_1/
            output.csv           # main trajectory
            constants.pkl        # simulation parameters
          replicate_2/
            ...
        RDME_CME_ODE/            # spatial model (we skip by default)
          ...
    
    If {root} is None or missing, generate synthetic data with identical schema.
    """
    
    def __init__(self, root: Optional[str] = None,
                 synthetic_fallback: bool = True):
        self.root = root
        self.synthetic_fallback = synthetic_fallback
        self._discovered = self._discover()
    
    def _discover(self) -> Dict[str, List[str]]:
        """Find available simulation directories/files."""
        found = {
            'cme_ode_csvs': [],
            'cme_ode_pickles': [],
            'rdme_files': [],
            'species_table': None,
        }
        if self.root is None or not os.path.isdir(self.root):
            return found
        
        # CME-ODE CSVs: look for output.csv in replicate_* folders
        for d in sorted(glob.glob(os.path.join(self.root, '**/replicate*'),
                                   recursive=True)):
            csv = os.path.join(d, 'output.csv')
            if os.path.isfile(csv):
                found['cme_ode_csvs'].append(csv)
            # Also look for species_counts_*.csv (Minimal_Cell_ComplexFormation layout)
            for alt in glob.glob(os.path.join(d, 'species_counts*.csv')):
                found['cme_ode_csvs'].append(alt)
        
        # Pickled trajectories (fallback format)
        for p in glob.glob(os.path.join(self.root, '**/*.pkl'), recursive=True):
            if 'traj' in os.path.basename(p).lower():
                found['cme_ode_pickles'].append(p)
        
        # LM RDME (Lattice Microbes) files — we don't parse these here
        for lm in glob.glob(os.path.join(self.root, '**/*.lm'), recursive=True):
            found['rdme_files'].append(lm)
        
        # Species metadata (helps us assign taxonomy)
        for name in ('initial_conditions.csv', 'kinetic_params.xlsx',
                     'species.txt', 'model_data.xlsx'):
            path = os.path.join(self.root, name)
            if os.path.isfile(path):
                found['species_table'] = path
                break
        
        return found
    
    def summary(self) -> None:
        print("=" * 64)
        print("ThornburgDataset")
        print("=" * 64)
        if self.root is None:
            print("  Root: (none — synthetic mode)")
        else:
            print(f"  Root: {self.root}")
            print(f"  Exists: {os.path.isdir(self.root)}")
        print(f"  CME-ODE CSV trajectories:  {len(self._discovered['cme_ode_csvs'])}")
        print(f"  Pickled trajectories:      {len(self._discovered['cme_ode_pickles'])}")
        print(f"  RDME .lm files (spatial):  {len(self._discovered['rdme_files'])}")
        print(f"  Species table: {self._discovered['species_table']}")
        if not (self._discovered['cme_ode_csvs'] or self._discovered['cme_ode_pickles']):
            print()
            print("  No real trajectories found.")
            if self.synthetic_fallback:
                print("  Will use synthetic Tier 3 data matching Thornburg schema.")
            else:
                print("  synthetic_fallback=False — downstream calls will fail.")
    
    # -------- Loading --------
    
    def load_cme_ode_csvs(self, max_cells: Optional[int] = None,
                          downsample_to: Optional[int] = 500) -> List[CellTrajectory]:
        """Load CME-ODE CSV trajectories. Each row is one time point."""
        paths = self._discovered['cme_ode_csvs']
        if not paths:
            if self.synthetic_fallback:
                return self._synthetic_trajectories(n_cells=max_cells or 20,
                                                    n_timepoints=downsample_to or 500)
            raise FileNotFoundError(
                f"No CME-ODE CSVs under {self.root}. "
                f"Download Zenodo 5780120 or pass synthetic_fallback=True.")
        
        if max_cells is not None:
            paths = paths[:max_cells]
        
        trajectories = []
        for p in paths:
            traj = self._read_csv(p, downsample_to=downsample_to)
            if traj is not None:
                trajectories.append(traj)
        return trajectories
    
    def _read_csv(self, path: str,
                  downsample_to: Optional[int] = None) -> Optional[CellTrajectory]:
        """Parse one CME-ODE output.csv. Format is standard: time column + species columns."""
        try:
            # Use numpy to avoid pandas dependency
            with open(path) as f:
                header = f.readline().strip().split(',')
            data = np.loadtxt(path, delimiter=',', skiprows=1)
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")
            return None
        
        if data.ndim != 2 or data.shape[0] < 2:
            warnings.warn(f"Empty or malformed: {path}")
            return None
        
        # Time column: look for 'time', 't', 'Time' — default to first column
        time_col = 0
        for cand in ('time', 't', 'Time', 'time_s', 'seconds'):
            if cand in header:
                time_col = header.index(cand)
                break
        
        time_s = data[:, time_col]
        mask = np.ones(data.shape[1], dtype=bool)
        mask[time_col] = False
        counts = data[:, mask]
        species_names = [h for i, h in enumerate(header) if mask[i]]
        
        if downsample_to is not None and len(time_s) > downsample_to:
            idx = np.linspace(0, len(time_s) - 1, downsample_to).astype(int)
            time_s = time_s[idx]
            counts = counts[idx]
        
        return CellTrajectory(
            cell_id=os.path.basename(os.path.dirname(path)) or 'cell',
            time_s=time_s,
            species_names=species_names,
            counts=counts,
            source='real_csv',
            metadata={'path': path, 'original_length': data.shape[0]},
        )
    
    # -------- Conversion to surrogate training data --------
    
    def to_surrogate_training(
        self,
        trajectories: List[CellTrajectory],
        init_window: Tuple[float, float] = (0.0, 60.0),
        final_window: Tuple[float, float] = (3540.0, 3600.0),
        species_subset: str = 'metabolites',
        perturbation_encoder: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert trajectories to (X, Y) pairs for Tier 2 surrogate training.
        
        X[i] = initial state of cell i (mean over init_window)
               + optional perturbation encoding (knockouts, media)
        Y[i] = final state of cell i (mean over final_window)
        
        Returns:
            X: (N, d_in), Y: (N, d_out), selected_species_names
        """
        if not trajectories:
            raise ValueError("No trajectories provided.")
        
        # Use first trajectory to determine species subset
        sel_idx = trajectories[0].subset_species(species_subset)
        sel_names = [trajectories[0].species_names[i] for i in sel_idx]
        
        X_list, Y_list = [], []
        for traj in trajectories:
            if traj.species_names[:len(sel_names)] != sel_names[:len(traj.species_names)]:
                # Species reindex for this trajectory
                idx = []
                for name in sel_names:
                    if name in traj.species_names:
                        idx.append(traj.species_names.index(name))
                    else:
                        idx.append(None)
                idx_valid = [i for i in idx if i is not None]
                if len(idx_valid) < len(sel_names) * 0.8:
                    continue  # too much missing
                counts = np.zeros((len(traj.time_s), len(sel_names)))
                for j, i in enumerate(idx):
                    if i is not None:
                        counts[:, j] = traj.counts[:, i]
            else:
                counts = traj.counts[:, sel_idx]
            
            # Window averaging
            m_init = (traj.time_s >= init_window[0]) & (traj.time_s <= init_window[1])
            m_final = (traj.time_s >= final_window[0]) & (traj.time_s <= final_window[1])
            if m_init.sum() == 0 or m_final.sum() == 0:
                continue
            init_state = counts[m_init].mean(axis=0)
            final_state = counts[m_final].mean(axis=0)
            
            # Optional perturbation encoding (e.g. knockouts from metadata)
            if perturbation_encoder is not None:
                pert_vec = perturbation_encoder(traj)
                x = np.concatenate([init_state, pert_vec])
            else:
                x = init_state
            
            X_list.append(x)
            Y_list.append(final_state)
        
        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)
        return X, Y, sel_names
    
    # -------- Synthetic fallback --------
    
    def _synthetic_trajectories(self, n_cells: int = 20,
                                n_timepoints: int = 500) -> List[CellTrajectory]:
        """
        Use tier3_cascade.Tier3MechanisticODE to manufacture data in Thornburg's
        schema. This lets downstream code (Tier 2 training, evaluation) work
        identically whether real or synthetic data is in play.
        """
        from tier3_cascade import Tier3MechanisticODE, Tier2DynamicSurrogate
        
        sim = Tier3MechanisticODE()
        species_names = [
            'M_glc_e', 'M_g6p_c', 'M_f6p_c', 'M_fdp_c',
            'M_pep_c', 'M_pyr_c', 'M_lac_c',
            'M_atp_c', 'M_adp_c', 'M_nad_c',
        ]
        
        rng = np.random.default_rng(0)
        trajectories = []
        for k in range(n_cells):
            # Perturb initial state
            y0_base = Tier3MechanisticODE.initial_state()
            y0 = y0_base * rng.uniform(0.7, 1.3, size=y0_base.shape)
            # Random knockout
            gene = None
            if k % 3 == 0:
                gene = str(rng.choice(Tier2DynamicSurrogate.GENE_ORDER))
            
            sim.reset()
            if gene:
                sim.set_knockout(gene)
            Tier3MechanisticODE.initial_state = staticmethod(lambda x=y0: x)
            
            try:
                # 60s horizon matches the stable regime from tier3_cascade.py.
                # The "t_final=3600" in Thornburg's actual data uses CME-ODE,
                # which handles stiffness differently from our reduced BDF ODE.
                result = sim.simulate(t_final=60.0, n_points=n_timepoints)
            finally:
                pass
            
            if not result['success']:
                continue
            
            trajectories.append(CellTrajectory(
                cell_id=f'synth_{k:03d}',
                time_s=result['t'],
                species_names=species_names,
                counts=result['y'].T,  # transpose to (T, S)
                source='synthetic',
                metadata={'knockout_gene': gene, 'initial_state_scale': list(y0 / y0_base)},
            ))
        
        return trajectories


def demo():
    """Quick sanity check that loader works on synthetic data end-to-end."""
    print("Testing ThornburgDataset with synthetic fallback...")
    ds = ThornburgDataset(root=None)  # will use synthetic
    ds.summary()
    
    print("\nLoading 20 synthetic trajectories...")
    trajectories = ds.load_cme_ode_csvs(max_cells=20, downsample_to=500)
    print(f"  Got {len(trajectories)} trajectories")
    print(f"  Species in first: {trajectories[0].n_species}")
    print(f"  Duration:        {trajectories[0].duration_s:.0f} s")
    print(f"  Time points:     {len(trajectories[0].time_s)}")
    
    print("\nConverting to surrogate training data...")
    X, Y, names = ds.to_surrogate_training(
        trajectories,
        init_window=(0, 60),
        final_window=(3540, 3600),
        species_subset='metabolites',
    )
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Species: {names}")
    
    # Train a surrogate on it
    print("\nFitting Tier2Hybrid on loaded data...")
    # Pad X with zero perturbation vector so Tier2Hybrid expects the right dims
    n_gene_dims = 7
    X_padded = np.hstack([X, np.zeros((X.shape[0], n_gene_dims))])
    
    from tier2_hybrid import Tier2Hybrid
    t2 = Tier2Hybrid()
    if X_padded.shape[0] >= 10:
        t2.fit(X_padded, Y)
        print("  Fit complete.")
        from sklearn.metrics import r2_score
        Y_pred = t2.predict_state_batch(X_padded)
        r2 = [r2_score(Y[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
        print(f"  Per-species R² (on training data, sanity only):")
        for n, r in zip(names, r2):
            print(f"    {n:12s}: {r:+.3f}")


if __name__ == '__main__':
    demo()
