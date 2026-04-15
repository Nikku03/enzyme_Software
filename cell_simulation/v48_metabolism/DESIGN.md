# V48: Metabolism Module
## ATP, NTPs, Amino Acids - The Fuel of Life

### The Big Picture

```
                    GLUCOSE
                       │
                       ▼
              ┌────────────────┐
              │   GLYCOLYSIS   │
              │                │
              │  Glucose ───►  │
              │      ATP       │
              │    Pyruvate    │
              └───────┬────────┘
                      │
          ┌──────────┴──────────┐
          ▼                     ▼
    ┌──────────┐          ┌──────────┐
    │   TCA    │          │ FERMENT  │
    │  CYCLE   │          │ (Lactate)│
    └────┬─────┘          └──────────┘
         │
         ▼
    ┌──────────┐
    │ OXIDATIVE│
    │  PHOS    │
    │  (ATP)   │
    └──────────┘
```

### What JCVI-syn3A Actually Has

JCVI-syn3A is a **minimal cell** - it has:
- ✅ Glycolysis (makes ATP)
- ✅ Pentose phosphate pathway (makes NADPH, ribose-5-P)
- ❌ NO TCA cycle
- ❌ NO oxidative phosphorylation
- ✅ F1Fo ATP synthase (but runs on membrane potential from?)

It's basically a **fermentative** organism that:
1. Takes up glucose
2. Runs glycolysis → ATP + pyruvate
3. Converts pyruvate → lactate (regenerates NAD+)
4. Uses ATP for everything else

### Key Metabolite Pools to Model

```
ENERGY:
  ATP ⟷ ADP + Pi        (main energy currency)
  GTP ⟷ GDP + Pi        (translation, signaling)
  
NUCLEOTIDES (for RNA synthesis):
  ATP, GTP, CTP, UTP    (NTPs)
  
DEOXYNUCLEOTIDES (for DNA synthesis):
  dATP, dGTP, dCTP, dTTP (dNTPs)

AMINO ACIDS (20 types for protein synthesis):
  Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile,
  Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val

COFACTORS:
  NAD+/NADH             (redox)
  NADP+/NADPH           (biosynthesis)
  CoA                   (acyl transfer)
  FAD/FADH2             (redox)
```

### Coupling to Gene Expression

```
TRANSCRIPTION consumes:
  - ATP, GTP, CTP, UTP (1 per nucleotide)
  - Rate limited by NTP availability

TRANSLATION consumes:
  - 2 GTP per amino acid (EF-Tu, EF-G)
  - 2 ATP per amino acid (tRNA charging)
  - 1 of each amino acid
  - Rate limited by AA availability

DNA REPLICATION consumes:
  - dATP, dGTP, dCTP, dTTP
```

### Implementation Strategy

**Level 1: Simplified pools (start here)**
- ATP/ADP pool with regeneration
- Single "NTP" pool
- Single "amino acid" pool
- Michaelis-Menten kinetics

**Level 2: Full pools**
- All 4 NTPs separately
- All 20 amino acids separately
- Nucleotide interconversion

**Level 3: Full FBA**
- Stoichiometric matrix
- Flux balance analysis
- Optimal flux distribution

### Key Parameters (Mycoplasma/E. coli)

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| ATP/cell | 1-10 mM | ~3×10⁶ molecules | BioNumbers |
| ATP turnover | ~1 mM/s | ~50% per second | Literature |
| Glycolysis flux | 0.5-2 mmol/gDW/hr | | iMB155 |
| AA concentration | 0.1-1 mM each | | BioNumbers |
| NTP concentration | 1-5 mM total | | BioNumbers |
