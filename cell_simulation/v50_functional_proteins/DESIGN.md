# V50: Functional Gene Products

## The Problem

Right now our proteins are just counts. But proteins DO things:
- **Enzymes** catalyze metabolic reactions (glycolysis, etc.)
- **Ribosomes** make more proteins
- **RNAP** makes mRNA
- **Chaperones** fold other proteins
- **DNA polymerase** replicates DNA

## The Solution: Close the Loop

```
Gene Expression → Protein Levels → Enzyme Activity → Metabolic Flux
      ↑                                                    │
      └────────────────────────────────────────────────────┘
                    (ATP, NTPs, AAs from metabolism)
```

## Key Functional Categories

### 1. METABOLIC ENZYMES
Protein levels determine Vmax of reactions:
- pgi, pfkA, fbaA, tpiA, gapA, pgk, eno, pykF → Glycolysis
- ldh → Lactate production
- ndk, adk, gmk, cmk → Nucleotide kinases

**Implementation:**
```python
# Glycolysis flux depends on enzyme levels
v_glycolysis = k_cat * [enzyme] * [substrate] / (Km + [substrate])
```

### 2. GENE EXPRESSION MACHINERY
- **RNAP (rpoA, rpoB, rpoC, rpoD)** → Transcription rate
- **Ribosomes (30S + 50S proteins)** → Translation rate
- **tuf, fusA, etc.** → Elongation factors

**Implementation:**
```python
# Transcription rate depends on RNAP levels
total_RNAP = f([rpoA], [rpoB], [rpoC], [rpoD])
tx_rate = k_tx * total_RNAP * promoter_strength
```

### 3. CHAPERONES
- **GroEL/GroES** → Protein folding
- **DnaK/DnaJ/GrpE** → Protein folding

**Implementation:**
```python
# Fraction of properly folded proteins
f_folded = [GroEL] / (K_fold + [GroEL])
active_protein = total_protein * f_folded
```

### 4. DNA REPLICATION (for cell division)
- **DnaA** → Initiation
- **DnaB, DnaC** → Helicase loading
- **DnaE, DnaN** → DNA polymerase III
- **GyrA, GyrB** → Topoisomerase

## Emergent Behaviors

With functional proteins, we get EMERGENT PROPERTIES:

1. **Growth rate emerges** from ribosome levels
2. **Metabolic flux emerges** from enzyme levels
3. **Gene expression capacity emerges** from RNAP levels
4. **Stress response** when chaperones are limiting

## Implementation Strategy

1. Define functional relationships (enzyme → flux)
2. Add to integrated model
3. Watch emergent behaviors arise
4. Validate against known biology
