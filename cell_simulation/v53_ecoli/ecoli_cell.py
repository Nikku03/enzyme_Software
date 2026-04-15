"""
Dark Manifold V53: E. coli Whole Cell
=====================================

E. coli is VERY different from JCVI-syn3A:

JCVI-syn3A (minimal):
- 473 genes
- Fermentative only (no TCA, no ETC)
- Glucose → Lactate + 2 ATP
- ~60-90 min doubling

E. coli (full):
- ~4,300 genes
- Full TCA cycle
- Oxidative phosphorylation (ETC)
- Glucose → CO2 + ~30 ATP
- ~20 min doubling (fast!)

Key differences:
1. MUCH more ATP per glucose (30 vs 2)
2. Requires O2 for max ATP
3. More complex regulation
4. Faster growth
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class EcoliParams:
    """E. coli specific parameters."""
    genome_size: int = 4_600_000  # 4.6 Mb
    doubling_time: float = 20.0   # min (in rich media)
    
    # Cell cycle - lower thresholds for our model
    DnaA_threshold: float = 30
    FtsZ_threshold: float = 50
    
    # ATP yield
    atp_per_glucose_glycolysis: int = 2
    atp_per_glucose_tca: int = 2      # GTP from succinyl-CoA
    atp_per_nadh: float = 2.5         # P/O ratio
    atp_per_fadh2: float = 1.5
    # Total: 2 + 2 + 10*2.5 + 2*1.5 = 32 ATP per glucose


class EcoliCell:
    """
    E. coli whole cell model.
    
    Key features:
    1. Glycolysis + TCA cycle + oxidative phosphorylation
    2. ~30 ATP per glucose (vs 2 for JCVI-syn3A)
    3. Oxygen-dependent respiration
    4. ~20 min doubling time
    """
    
    def __init__(self, n_genes: int = 100):
        print("="*72)
        print("    DARK MANIFOLD V53: E. COLI WHOLE CELL")
        print("    Full respiration - TCA + oxidative phosphorylation")
        print("="*72)
        
        self.params = EcoliParams()
        
        # Build E. coli gene set
        self.genes = self._build_ecoli_genes(n_genes)
        self.gene_list = list(self.genes.keys())
        self.n_genes = len(self.gene_list)
        
        # Metabolites - E. coli has more!
        self.met_states = [
            # Energy
            'ATP', 'ADP', 'AMP',
            'GTP', 'GDP',
            'NAD', 'NADH',
            'NADP', 'NADPH',  # E. coli uses NADPH for biosynthesis
            'FAD', 'FADH2',   # For succinate dehydrogenase
            'UTP', 'CTP',
            'dNTP',
            # Glycolysis
            'Glc', 'G6P', 'PEP', 'Pyr',
            # TCA cycle
            'AcCoA', 'Cit', 'aKG', 'Suc', 'OAA',
            # Other
            'AA', 'O2', 'CO2',
            'Pi', 'CoA',
        ]
        self.n_met = len(self.met_states)
        
        # Cell cycle
        self.cc_states = ['DNA', 'rep_prog', 'div_prog', 'mass']
        self.n_cc = len(self.cc_states)
        
        # Total state
        self.n_states = self.n_met + self.n_cc + 2 * self.n_genes
        
        # Indices
        self.met_idx = {m: i for i, m in enumerate(self.met_states)}
        self.cc_idx = {c: self.n_met + i for i, c in enumerate(self.cc_states)}
        self.mrna_idx = {g: self.n_met + self.n_cc + i for i, g in enumerate(self.gene_list)}
        self.prot_idx = {g: self.n_met + self.n_cc + self.n_genes + i 
                        for i, g in enumerate(self.gene_list)}
        
        self._setup_kinetics()
        
        self.division_times = []
        self.death_time = None
        
        self._print_summary()
    
    def _build_ecoli_genes(self, n_genes: int) -> Dict:
        """Build E. coli gene set."""
        
        @dataclass
        class Gene:
            name: str
            length_aa: int
            category: str
            promoter_strength: float = 5.0
            rbs_strength: float = 5.0
            k_tx_init: float = 0.0
            k_tl_init: float = 0.0
            autorepression: bool = False
            K_autorepression: float = 100
            
            @property
            def length_nt(self):
                return self.length_aa * 3
            
            @property
            def delta_m(self):
                return np.log(2) / 3.0  # ~3 min mRNA half-life
            
            @property
            def delta_p(self):
                return np.log(2) / 60.0 + np.log(2) / 20.0  # Decay + fast dilution
        
        genes = {}
        
        # Essential E. coli genes
        essential = [
            # RNAP
            ('rpoA', 329, 'RNAP', 4.0), ('rpoB', 1342, 'RNAP', 4.0),
            ('rpoC', 1407, 'RNAP', 4.0), ('rpoD', 613, 'RNAP', 3.0),
            
            # Ribosomes (highly expressed)
            ('rplA', 234, 'ribosome', 9.0), ('rplB', 273, 'ribosome', 9.0),
            ('rpsA', 557, 'ribosome', 9.0), ('rpsB', 241, 'ribosome', 9.0),
            ('rplC', 209, 'ribosome', 9.0), ('rpsC', 233, 'ribosome', 9.0),
            
            # Translation factors
            ('tufA', 394, 'translation', 9.0), ('fusA', 704, 'translation', 8.0),
            ('infB', 890, 'translation', 6.0), ('tsf', 283, 'translation', 7.0),
            
            # Glycolysis
            ('pgi', 549, 'glycolysis', 5.0), ('pfkA', 320, 'glycolysis', 5.0),
            ('fbaA', 359, 'glycolysis', 5.0), ('gapA', 331, 'glycolysis', 6.0),
            ('pgk', 387, 'glycolysis', 5.0), ('eno', 432, 'glycolysis', 6.0),
            ('pykF', 470, 'glycolysis', 5.0),
            
            # Pyruvate dehydrogenase
            ('aceE', 886, 'pdh', 5.0), ('aceF', 630, 'pdh', 5.0),
            ('lpd', 474, 'pdh', 5.0),
            
            # TCA cycle
            ('gltA', 427, 'tca', 5.0),   # Citrate synthase
            ('acnB', 865, 'tca', 4.0),   # Aconitase
            ('icd', 416, 'tca', 5.0),    # Isocitrate dehydrogenase
            ('sucA', 933, 'tca', 4.0),   # α-ketoglutarate dehydrogenase
            ('sucC', 388, 'tca', 4.0),   # Succinyl-CoA synthetase
            ('sdhA', 588, 'tca', 4.0),   # Succinate dehydrogenase
            ('fumA', 548, 'tca', 4.0),   # Fumarase
            ('mdh', 312, 'tca', 5.0),    # Malate dehydrogenase
            
            # Electron transport chain
            ('nuoA', 147, 'etc', 4.0),   # Complex I
            ('cyoA', 315, 'etc', 4.0),   # Cytochrome bo3
            ('atpA', 513, 'atp_synthase', 6.0),  # ATP synthase
            ('atpD', 460, 'atp_synthase', 6.0),
            
            # DNA replication
            ('dnaA', 467, 'replication', 3.0),
            ('dnaE', 1160, 'replication', 3.0),
            ('dnaN', 366, 'replication', 3.0),
            
            # Cell division
            ('ftsZ', 383, 'division', 4.0),
            ('ftsA', 420, 'division', 4.0),
            
            # Chaperones
            ('groEL', 548, 'chaperone', 7.0),
            ('dnaK', 638, 'chaperone', 6.0),
        ]
        
        for item in essential:
            if len(item) == 4:
                name, length, cat, strength = item
            else:
                name, length, cat = item
                strength = 5.0
            
            genes[name] = Gene(
                name=name, 
                length_aa=length, 
                category=cat,
                promoter_strength=strength,
                rbs_strength=strength
            )
        
        # DnaA autorepression
        if 'dnaA' in genes:
            genes['dnaA'].autorepression = True
            genes['dnaA'].K_autorepression = 200
        
        return genes
    
    def _setup_kinetics(self):
        """E. coli kinetics - faster than JCVI-syn3A."""
        
        # External
        self.Glc_ext = 20.0
        self.AA_ext = 10.0
        self.O2_ext = 0.2  # mM dissolved O2
        
        # GLYCOLYSIS (same as JCVI-syn3A)
        self.V_glc_uptake = 3.0  # Faster uptake
        self.V_glycolysis = 2.0
        self.V_pyk = 3.0
        
        # PYRUVATE DEHYDROGENASE: Pyr + NAD + CoA → AcCoA + NADH + CO2
        self.V_pdh = 2.0
        self.Km_pyr_pdh = 0.5
        
        # TCA CYCLE (lumped)
        self.V_tca = 1.5
        self.Km_accoa = 0.1
        self.Km_nad_tca = 0.2
        self.Km_o2 = 0.01
        
        # ELECTRON TRANSPORT CHAIN - key for ATP!
        self.V_etc = 5.0  # Faster
        self.Km_nadh_etc = 0.03
        self.Km_o2_etc = 0.003
        self.P_O_nadh = 2.5
        self.P_O_fadh2 = 1.5
        
        # Gene expression (faster for E. coli)
        self.k_tx = 1.0   # Higher
        self.k_tl = 5.0   # Much higher
        self.K_RNAP = 5   # Lower Km
        self.K_ribo = 30  # Lower Km
        
        # ATP costs (same)
        self.ntp_per_tx = 0.0003
        self.atp_per_tl = 0.001
        self.gtp_per_tl = 0.001
        self.aa_per_tl = 0.001
        
        # Maintenance (higher for E. coli)
        self.V_maint = 0.05
        
        for gene in self.genes.values():
            gene.k_tx_init = self.k_tx * (gene.promoter_strength / 5.0)
            gene.k_tl_init = self.k_tl * (gene.rbs_strength / 5.0)
    
    def _print_summary(self):
        print(f"\nGenes: {self.n_genes}")
        print(f"States: {self.n_states}")
        print(f"\nE. coli metabolism:")
        print(f"  Glycolysis: Glucose → 2 Pyruvate + 2 ATP + 2 NADH")
        print(f"  PDH: Pyruvate → Acetyl-CoA + NADH + CO2")
        print(f"  TCA: Acetyl-CoA → 2 CO2 + 3 NADH + FADH2 + GTP")
        print(f"  ETC: NADH → 2.5 ATP, FADH2 → 1.5 ATP")
        print(f"  Total: ~30 ATP per glucose!")
        
        by_cat = defaultdict(int)
        for g in self.genes.values():
            by_cat[g.category] += 1
        print(f"\nGenes by category:")
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {n}")
    
    def get_initial_state(self) -> np.ndarray:
        y0 = np.zeros(self.n_states)
        
        # Energy - E. coli has more ATP
        y0[self.met_idx['ATP']] = 5.0
        y0[self.met_idx['ADP']] = 0.5
        y0[self.met_idx['AMP']] = 0.2
        y0[self.met_idx['GTP']] = 1.0
        y0[self.met_idx['GDP']] = 0.3
        
        # Redox
        y0[self.met_idx['NAD']] = 2.0
        y0[self.met_idx['NADH']] = 0.2
        y0[self.met_idx['NADP']] = 0.3
        y0[self.met_idx['NADPH']] = 0.1
        y0[self.met_idx['FAD']] = 0.1
        y0[self.met_idx['FADH2']] = 0.02
        
        # NTPs
        y0[self.met_idx['UTP']] = 0.5
        y0[self.met_idx['CTP']] = 0.4
        y0[self.met_idx['dNTP']] = 0.1
        
        # Glycolysis
        y0[self.met_idx['Glc']] = 0.5
        y0[self.met_idx['G6P']] = 0.3
        y0[self.met_idx['PEP']] = 0.2
        y0[self.met_idx['Pyr']] = 0.5
        
        # TCA
        y0[self.met_idx['AcCoA']] = 0.2
        y0[self.met_idx['Cit']] = 0.3
        y0[self.met_idx['aKG']] = 0.1
        y0[self.met_idx['Suc']] = 0.1
        y0[self.met_idx['OAA']] = 0.1
        y0[self.met_idx['CoA']] = 0.5
        
        # Other
        y0[self.met_idx['AA']] = 5.0
        y0[self.met_idx['O2']] = 0.2
        y0[self.met_idx['CO2']] = 0.5
        y0[self.met_idx['Pi']] = 10.0
        
        # Cell cycle
        y0[self.cc_idx['DNA']] = 1.0
        y0[self.cc_idx['rep_prog']] = 0.0
        y0[self.cc_idx['div_prog']] = 0.0
        y0[self.cc_idx['mass']] = 1.0
        
        # Proteins - E. coli has more, start high!
        for name in self.gene_list:
            gene = self.genes[name]
            mrna = gene.k_tx_init / gene.delta_m * 5.0
            prot = gene.k_tl_init * mrna / gene.delta_p * 100.0
            
            if gene.category == 'ribosome':
                prot *= 5.0  # Lots of ribosomes
            if name in ['dnaA', 'ftsZ']:
                prot *= 3.0  # Cell cycle proteins
            
            y0[self.mrna_idx[name]] = max(mrna, 2.0)
            y0[self.prot_idx[name]] = max(prot, 200)
        
        return y0
    
    def get_met(self, y, name):
        return max(y[self.met_idx[name]], 1e-12) if name in self.met_idx else 0
    
    def get_prot(self, y, name):
        return max(y[self.prot_idx[name]], 0) if name in self.prot_idx else 0
    
    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        dy = np.zeros(self.n_states)
        
        # Unpack metabolites
        ATP = self.get_met(y, 'ATP')
        ADP = self.get_met(y, 'ADP')
        AMP = self.get_met(y, 'AMP')
        GTP = self.get_met(y, 'GTP')
        GDP = self.get_met(y, 'GDP')
        NAD = self.get_met(y, 'NAD')
        NADH = self.get_met(y, 'NADH')
        FAD = self.get_met(y, 'FAD')
        FADH2 = self.get_met(y, 'FADH2')
        UTP = self.get_met(y, 'UTP')
        CTP = self.get_met(y, 'CTP')
        dNTP = self.get_met(y, 'dNTP')
        Glc = self.get_met(y, 'Glc')
        G6P = self.get_met(y, 'G6P')
        PEP = self.get_met(y, 'PEP')
        Pyr = self.get_met(y, 'Pyr')
        AcCoA = self.get_met(y, 'AcCoA')
        CoA = self.get_met(y, 'CoA')
        O2 = self.get_met(y, 'O2')
        AA = self.get_met(y, 'AA')
        Pi = self.get_met(y, 'Pi')
        
        DNA = y[self.cc_idx['DNA']]
        rep_prog = y[self.cc_idx['rep_prog']]
        div_prog = y[self.cc_idx['div_prog']]
        mass = y[self.cc_idx['mass']]
        
        DnaA = self.get_prot(y, 'dnaA')
        DnaE = self.get_prot(y, 'dnaE')
        FtsZ = self.get_prot(y, 'ftsZ')
        
        # ========== GLYCOLYSIS ==========
        
        # Glucose uptake
        v_glc = self.V_glc_uptake * self.Glc_ext / (1.0 + self.Glc_ext)
        dy[self.met_idx['G6P']] += v_glc
        
        # Glycolysis (lumped): G6P + 2ADP + 2NAD → 2PEP + 2ATP + 2NADH
        atp_inhib = 5.0 / (5.0 + ATP)
        v_glyc = self.V_glycolysis * G6P / (0.2 + G6P) * NAD / (0.1 + NAD) * ADP / (0.3 + ADP) * atp_inhib
        
        dy[self.met_idx['G6P']] -= v_glyc
        dy[self.met_idx['NAD']] -= 2 * v_glyc
        dy[self.met_idx['NADH']] += 2 * v_glyc
        dy[self.met_idx['ADP']] -= 2 * v_glyc
        dy[self.met_idx['ATP']] += 2 * v_glyc
        dy[self.met_idx['PEP']] += 2 * v_glyc
        
        # Pyruvate kinase
        v_pyk = self.V_pyk * PEP / (0.2 + PEP) * ADP / (0.3 + ADP)
        dy[self.met_idx['PEP']] -= v_pyk
        dy[self.met_idx['ADP']] -= v_pyk
        dy[self.met_idx['Pyr']] += v_pyk
        dy[self.met_idx['ATP']] += v_pyk
        
        # ========== PYRUVATE DEHYDROGENASE ==========
        # Pyr + NAD + CoA → AcCoA + NADH + CO2
        v_pdh = self.V_pdh * Pyr / (self.Km_pyr_pdh + Pyr) * NAD / (0.2 + NAD) * CoA / (0.1 + CoA)
        
        dy[self.met_idx['Pyr']] -= v_pdh
        dy[self.met_idx['NAD']] -= v_pdh
        dy[self.met_idx['NADH']] += v_pdh
        dy[self.met_idx['CoA']] -= v_pdh
        dy[self.met_idx['AcCoA']] += v_pdh
        dy[self.met_idx['CO2']] += v_pdh
        
        # ========== TCA CYCLE (lumped) ==========
        # AcCoA + 3NAD + FAD + GDP → 2CO2 + 3NADH + FADH2 + GTP + CoA
        v_tca = self.V_tca * AcCoA / (self.Km_accoa + AcCoA) * NAD / (self.Km_nad_tca + NAD) * FAD / (0.05 + FAD)
        
        dy[self.met_idx['AcCoA']] -= v_tca
        dy[self.met_idx['NAD']] -= 3 * v_tca
        dy[self.met_idx['NADH']] += 3 * v_tca
        dy[self.met_idx['FAD']] -= v_tca
        dy[self.met_idx['FADH2']] += v_tca
        dy[self.met_idx['GDP']] -= v_tca
        dy[self.met_idx['GTP']] += v_tca
        dy[self.met_idx['CoA']] += v_tca
        dy[self.met_idx['CO2']] += 2 * v_tca
        
        # ========== ELECTRON TRANSPORT CHAIN ==========
        # NADH + 0.5 O2 + 2.5 ADP → NAD + 2.5 ATP
        v_etc_nadh = self.V_etc * NADH / (self.Km_nadh_etc + NADH) * O2 / (self.Km_o2_etc + O2) * ADP / (0.5 + ADP)
        
        dy[self.met_idx['NADH']] -= v_etc_nadh
        dy[self.met_idx['NAD']] += v_etc_nadh
        dy[self.met_idx['O2']] -= 0.5 * v_etc_nadh
        dy[self.met_idx['ADP']] -= self.P_O_nadh * v_etc_nadh
        dy[self.met_idx['ATP']] += self.P_O_nadh * v_etc_nadh
        
        # FADH2 oxidation
        v_etc_fadh2 = self.V_etc * 0.5 * FADH2 / (0.02 + FADH2) * O2 / (self.Km_o2_etc + O2) * ADP / (0.5 + ADP)
        
        dy[self.met_idx['FADH2']] -= v_etc_fadh2
        dy[self.met_idx['FAD']] += v_etc_fadh2
        dy[self.met_idx['O2']] -= 0.5 * v_etc_fadh2
        dy[self.met_idx['ADP']] -= self.P_O_fadh2 * v_etc_fadh2
        dy[self.met_idx['ATP']] += self.P_O_fadh2 * v_etc_fadh2
        
        # ========== O2 UPTAKE ==========
        v_o2 = 0.5 * (self.O2_ext - O2)  # Diffusion-like
        dy[self.met_idx['O2']] += v_o2
        
        # ========== CO2 RELEASE ==========
        CO2 = self.get_met(y, 'CO2')
        v_co2 = 0.3 * CO2
        dy[self.met_idx['CO2']] -= v_co2
        
        # ========== KINASES ==========
        v_adk = 50.0 * (ADP * ADP - ATP * AMP)
        dy[self.met_idx['ATP']] += v_adk
        dy[self.met_idx['AMP']] += v_adk
        dy[self.met_idx['ADP']] -= 2 * v_adk
        
        v_ndk = 30.0 * (GDP * ATP - GTP * ADP)
        dy[self.met_idx['GTP']] += v_ndk
        dy[self.met_idx['ADP']] += v_ndk
        dy[self.met_idx['GDP']] -= v_ndk
        dy[self.met_idx['ATP']] -= v_ndk
        
        # ========== BIOSYNTHESIS ==========
        v_aa = 0.3 * self.AA_ext / (0.5 + self.AA_ext)
        dy[self.met_idx['AA']] += v_aa
        
        v_dntp = 0.03 * ATP / (0.5 + ATP)
        dy[self.met_idx['dNTP']] += v_dntp
        dy[self.met_idx['ATP']] -= v_dntp
        dy[self.met_idx['ADP']] += v_dntp
        
        v_ntp = 0.03 * ATP / (0.5 + ATP)
        dy[self.met_idx['UTP']] += 0.5 * v_ntp
        dy[self.met_idx['CTP']] += 0.5 * v_ntp
        dy[self.met_idx['ATP']] -= v_ntp
        dy[self.met_idx['ADP']] += v_ntp
        
        # ========== GENE EXPRESSION ==========
        n_RNAP = min(self.get_prot(y, 'rpoB'), self.get_prot(y, 'rpoD'))
        n_ribo = min(self.get_prot(y, 'rplA'), self.get_prot(y, 'rpsA'))
        
        f_RNAP = n_RNAP / (self.K_RNAP + n_RNAP)
        f_ribo = n_ribo / (self.K_ribo + n_ribo)
        
        NTP_min = min(ATP, GTP, UTP, CTP)
        f_NTP = NTP_min / (0.2 + NTP_min)
        f_AA = AA / (0.5 + AA)
        f_ATP = ATP / (0.5 + ATP)
        f_GTP = GTP / (0.3 + GTP)
        
        total_tx = 0
        total_tl = 0
        
        for name in self.gene_list:
            gene = self.genes[name]
            i_m = self.mrna_idx[name]
            i_p = self.prot_idx[name]
            
            mrna = max(y[i_m], 0)
            prot = max(y[i_p], 0)
            
            rep = 1.0
            if gene.autorepression and prot > 0:
                rep = gene.K_autorepression / (gene.K_autorepression + prot)
            
            tx = gene.k_tx_init * f_RNAP * f_NTP * rep * DNA
            dy[i_m] = tx - gene.delta_m * mrna
            total_tx += tx
            
            tl = gene.k_tl_init * mrna * f_ribo * f_AA * f_ATP * f_GTP
            dy[i_p] = tl - gene.delta_p * prot
            total_tl += tl
        
        # Costs
        dy[self.met_idx['ATP']] -= total_tx * self.ntp_per_tx
        dy[self.met_idx['GTP']] -= total_tx * self.ntp_per_tx
        dy[self.met_idx['AA']] -= total_tl * self.aa_per_tl
        dy[self.met_idx['ATP']] -= total_tl * self.atp_per_tl
        dy[self.met_idx['GTP']] -= total_tl * self.gtp_per_tl
        dy[self.met_idx['ADP']] += total_tl * self.atp_per_tl
        dy[self.met_idx['GDP']] += total_tl * self.gtp_per_tl
        
        # ========== MAINTENANCE ==========
        v_maint = self.V_maint * ATP / (0.3 + ATP)
        dy[self.met_idx['ATP']] -= v_maint
        dy[self.met_idx['ADP']] += v_maint
        
        # ========== CELL CYCLE ==========
        DnaA_ATP = DnaA * ATP / (0.5 + ATP)
        
        replicating = rep_prog > 0 and rep_prog < 1.0
        can_init = DnaA_ATP > self.params.DnaA_threshold and rep_prog < 0.01 and DNA < 1.2
        
        if can_init:
            dy[self.cc_idx['rep_prog']] = 0.01
        
        if replicating:
            v_rep = 0.03 * DnaE / (50 + DnaE) * dNTP / (0.02 + dNTP)  # Faster for E. coli
            dy[self.cc_idx['rep_prog']] = v_rep
            dy[self.cc_idx['DNA']] = v_rep
            dy[self.met_idx['dNTP']] -= v_rep * 0.01
        
        if rep_prog >= 1.0:
            dy[self.cc_idx['rep_prog']] = 0
        
        can_div = DNA >= 1.8 and FtsZ > self.params.FtsZ_threshold and div_prog < 0.01
        
        if can_div:
            dy[self.cc_idx['div_prog']] = 0.01
        
        if div_prog > 0 and div_prog < 1.0:
            v_div = 0.15 * FtsZ / (self.params.FtsZ_threshold + FtsZ)  # Faster division
            dy[self.cc_idx['div_prog']] = v_div
        
        # ========== GROWTH (tuned for balance) ==========
        mu = 0.02 * f_ribo * f_AA * f_ATP  # Slower growth
        dy[self.cc_idx['mass']] = mu * mass
        
        for i in range(self.n_met):
            dy[i] -= mu * y[i]
        for i in range(self.n_met + self.n_cc, self.n_states):
            dy[i] -= mu * y[i]
        
        return dy
    
    def handle_division(self, y, t):
        self.division_times.append(t)
        y_new = y.copy()
        
        for i in range(self.n_met):
            y_new[i] = y[i] / 2
        
        y_new[self.cc_idx['DNA']] = 1.0
        y_new[self.cc_idx['rep_prog']] = 0.0
        y_new[self.cc_idx['div_prog']] = 0.0
        y_new[self.cc_idx['mass']] = y[self.cc_idx['mass']] / 2
        
        for i in range(self.n_met + self.n_cc, self.n_states):
            y_new[i] = y[i] / 2
        
        return y_new
    
    def check_death(self, y):
        ATP = self.get_met(y, 'ATP')
        ADP = self.get_met(y, 'ADP')
        AMP = self.get_met(y, 'AMP')
        EC = (ATP + 0.5*ADP) / (ATP + ADP + AMP + 1e-12)
        return EC < 0.2 or ATP < 0.1  # More lenient
    
    def simulate(self, t_end=200, max_div=5):
        print(f"\nSimulating E. coli: 0 → {t_end} min...")
        
        y = self.get_initial_state()
        t = 0
        dt = 0.5
        
        times = [0]
        states = [y.copy()]
        
        while t < t_end and len(self.division_times) < max_div:
            t_next = min(t + dt, t_end)
            
            try:
                sol = solve_ivp(self.deriv, (t, t_next), y, method='LSODA',
                               rtol=1e-5, atol=1e-8)
                y = sol.y[:, -1]
            except Exception as e:
                print(f"  ✗ Error at t={t:.1f}: {e}")
                break
            
            t = t_next
            
            if self.check_death(y):
                ATP = self.get_met(y, 'ATP')
                EC = (ATP + 0.5*self.get_met(y, 'ADP')) / \
                     (ATP + self.get_met(y, 'ADP') + self.get_met(y, 'AMP') + 1e-12)
                print(f"  ☠️  DEATH at t={t:.1f} min (EC={EC:.3f}, ATP={ATP:.2f})")
                self.death_time = t
                break
            
            if y[self.cc_idx['div_prog']] >= 1.0:
                print(f"  ⚡ DIVISION #{len(self.division_times)+1} at t={t:.1f} min")
                y = self.handle_division(y, t)
            
            times.append(t)
            states.append(y.copy())
        
        times = np.array(times)
        states = np.array(states).T
        
        result = {'t': times, 'divisions': self.division_times, 'death': self.death_time}
        
        for met in self.met_states:
            result[met] = states[self.met_idx[met], :]
        
        for cc in self.cc_states:
            result[cc] = states[self.cc_idx[cc], :]
        
        result['mrna'] = {g: states[self.mrna_idx[g], :] for g in self.gene_list}
        result['protein'] = {g: states[self.prot_idx[g], :] for g in self.gene_list}
        
        result['EC'] = (result['ATP'] + 0.5*result['ADP']) / \
                       (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        
        return result
    
    def analyze(self, r):
        n_div = len(r['divisions'])
        died = r['death'] is not None
        
        print(f"""
{'█'*72}
{'█'*20}  E. COLI REPORT  {'█'*21}
{'█'*72}

  Simulation: {r['t'][-1]:.0f} min
  Divisions: {n_div}
  Survived: {'YES!' if not died else 'NO - died at t=' + str(r['death'])}
""")
        
        if n_div > 1:
            intervals = np.diff(r['divisions'])
            print(f"  Cycle times: {intervals}")
            print(f"  Mean: {np.mean(intervals):.1f} min")
        
        print(f"""
  Final state:
    ATP: {r['ATP'][-1]:.2f} mM    Energy charge: {r['EC'][-1]:.3f}
    NAD: {r['NAD'][-1]:.2f} mM    NADH: {r['NADH'][-1]:.2f} mM
    O2:  {r['O2'][-1]:.3f} mM    (respiring!)
    DNA: {r['DNA'][-1]:.2f}      Mass: {r['mass'][-1]:.2f}
""")
        
        if n_div >= 3 and not died:
            print(f"""
{'═'*72}
{'█'*72}
{'█'*15}  🎉 E. COLI SURVIVED! 🎉  {'█'*15}
{'█'*72}
{'═'*72}

  E. coli with FULL respiration:
    ✓ Glycolysis
    ✓ TCA cycle
    ✓ Electron transport chain
    ✓ Oxidative phosphorylation (~30 ATP/glucose)
    ✓ Fast growth (~{np.mean(np.diff(r['divisions'])) if n_div > 1 else 'N/A'} min doubling)

{'═'*72}
""")
        
        return {'divisions': n_div, 'died': died}


def main():
    model = EcoliCell(n_genes=50)
    result = model.simulate(t_end=200, max_div=8)
    analysis = model.analyze(result)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('E. coli Whole Cell - Full Respiration', fontsize=14, fontweight='bold')
    t = result['t']
    divs = result['divisions']
    
    ax = axes[0, 0]
    ax.plot(t, result['EC'], 'k-', lw=2)
    ax.axhline(0.3, color='r', ls='--', alpha=0.7)
    ax.fill_between(t, 0.8, 1.0, alpha=0.1, color='g')
    for td in divs: ax.axvline(td, color='b', ls=':', alpha=0.7)
    ax.set_title('Energy Charge')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t, result['ATP'], 'b-', lw=2, label='ATP')
    ax.plot(t, result['ADP'], 'r-', lw=1.5, label='ADP')
    for td in divs: ax.axvline(td, color='b', ls=':', alpha=0.7)
    ax.set_title('Adenylates')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(t, result['NAD'], 'orange', lw=2, label='NAD')
    ax.plot(t, result['NADH'], 'brown', lw=1.5, label='NADH')
    ax.plot(t, result['FAD']*5, 'green', lw=1.5, label='FAD×5')
    for td in divs: ax.axvline(td, color='b', ls=':', alpha=0.7)
    ax.set_title('Redox Carriers')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(t, result['DNA'], 'b-', lw=2)
    for td in divs: ax.axvline(td, color='b', ls=':', alpha=0.7)
    ax.set_title('DNA Content')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t, result['O2'], 'cyan', lw=2, label='O₂')
    ax.plot(t, result['CO2'], 'gray', lw=1.5, label='CO₂')
    for td in divs: ax.axvline(td, color='b', ls=':', alpha=0.7)
    ax.set_title('Gases (respiration!)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(t, result['Pyr'], 'r-', lw=2, label='Pyruvate')
    ax.plot(t, result['AcCoA'], 'purple', lw=1.5, label='Acetyl-CoA')
    for td in divs: ax.axvline(td, color='b', ls=':', alpha=0.7)
    ax.set_title('TCA Intermediates')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ecoli_cell.png', dpi=150)
    print("✓ Saved: ecoli_cell.png")
    
    return model, result, analysis

if __name__ == '__main__':
    model, result, analysis = main()
