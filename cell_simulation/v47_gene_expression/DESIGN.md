# V47: Gene Expression Dynamics
## Transcription + Translation for JCVI-syn3A

### Goal
Simulate the DYNAMICS of gene expression:
- mRNA levels over time
- Protein levels over time
- Response to perturbations

### The Biology

```
        TRANSCRIPTION              TRANSLATION
DNA ──────────────────> mRNA ──────────────────> Protein
        (RNAP)                    (Ribosome)
        
        kTX                        kTL
Gene ────────> mRNA ────────> Protein
               │                  │
               ▼ δm               ▼ δp
             (decay)           (decay/dilution)
```

### Core Equations (per gene i)

```
d[mRNA_i]/dt = kTX_i - δm_i * [mRNA_i]

d[Protein_i]/dt = kTL_i * [mRNA_i] - δp_i * [Protein_i]
```

Where:
- kTX_i = transcription rate (mRNA/min)
- kTL_i = translation rate (protein/mRNA/min)
- δm_i = mRNA decay rate (1/min)
- δp_i = protein decay rate (1/min)

### Typical E. coli / Mycoplasma Parameters

| Parameter | E. coli | Mycoplasma | Units |
|-----------|---------|------------|-------|
| mRNA half-life | 3-8 min | 2-5 min | min |
| Protein half-life | 20 min - hours | similar | min |
| Transcription rate | 10-100 | 5-50 | mRNA/gene/hr |
| Translation rate | 10-1000 | 10-500 | protein/mRNA/hr |
| Ribosomes/cell | 20,000-70,000 | 500-2,000 | count |
| RNAP/cell | 1,500-5,000 | 100-500 | count |

### Resources to Model

1. **RNAP (RNA Polymerase)**
   - Limited pool
   - Competes for genes
   - Elongation speed: ~40-80 nt/s

2. **Ribosomes**
   - Limited pool (major resource!)
   - Competes for mRNAs
   - Elongation speed: ~15-20 aa/s

3. **tRNAs**
   - Charged vs uncharged
   - Codon-specific availability

4. **NTPs (for transcription)**
   - ATP, GTP, CTP, UTP

5. **Amino acids (for translation)**
   - 20 types, different pools

### Simplification Levels

**Level 1: Simple ODEs (start here)**
- Fixed rates per gene
- No resource competition
- ~500 genes × 2 variables = 1000 ODEs

**Level 2: Resource-limited**
- RNAP and ribosome pools
- Competition for resources
- Michaelis-Menten kinetics

**Level 3: Full mechanistic**
- Elongation steps
- Codon-specific translation
- Initiation/elongation/termination

### Implementation Plan

1. Build gene database with expression parameters
2. Simple ODE model (Level 1)
3. Add resource competition (Level 2)
4. Validate against known data
5. Perturbation experiments
