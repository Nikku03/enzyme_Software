# CYP3A4 Dataset Curation Report

**Generated:** 2026-04-09
**CYP Filter:** CYP3A4

## Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Gold | 128 | 33.1% |
| Silver | 137 | 35.4% |
| Review | 73 | 18.9% |
| Rejected | 49 | 12.7% |
| **Total** | 387 | 100% |

## Chemical Validity Issues

| Issue | Count |
|-------|-------|
| quaternary_carbon | 235 |
| aromatic_ch | 154 |
| nitrogen_som | 88 |
| suspicious_atom | 83 |
| terminal_oxygen | 68 |
| ether_oxygen | 15 |
| impossible_atom | 9 |
| sulfur_som | 4 |

## Quality by Source

| Source | Gold | Silver | Review | Rejected | Gold Rate |
|--------|------|--------|--------|----------|-----------|
| AZ120 | 40 | 9 | 2 | 0 | 57.1% |
| CYP_DBs_external | 0 | 10 | 12 | 0 | 0.0% |
| DrugBank | 48 | 67 | 16 | 0 | 35.6% |
| MetXBioDB | 35 | 42 | 25 | 0 | 29.4% |
| MetaPred | 0 | 1 | 0 | 0 | 0.0% |
| literature | 2 | 0 | 0 | 0 | 100.0% |
| metxbiodb | 2 | 8 | 18 | 0 | 6.7% |
| validated | 1 | 0 | 0 | 0 | 100.0% |

## Common Issues in Rejected/Review

| Flag | Count |
|------|-------|
| PHYSICS_DISAGREES | 54 |
| TOO_MANY_SITES | 18 |
| EXCESSIVE_SITES | 13 |

## Sample Rejected Cases (first 10)

- **Cmpd 115** (az120:92f1ecb14c5e:CYP3A4)
  - SMILES: `COc1ccc(CCO[C@@H]2CCCC[C@H]2N2CC[C@@H](O)C2)cc1OC`
  - Original sites: [19]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **Abiraterone** (DB05812)
  - SMILES: `C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(c3cccnc3)=C...`
  - Original sites: [17]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **Delamanid** (DB11637)
  - SMILES: `C[C@]1(COc2ccc(N3CCC(Oc4ccc(OC(F)(F)F)cc4)CC3)cc2)Cn2cc([N+]...`
  - Original sites: [7, 13, 16, 18]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **L_775606** (metxbiodb:HYDKUVPEVOHALC-UHFFFAOYSA-N:CYP3A4)
  - SMILES: `Fc1cccc(CCN2CCN(CCCc3cc4cc(-n5cnnc5)ccc4[nH]3)CC2)c1`
  - Original sites: [1]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **N_ac_PCBC** (metxbiodb:GEAIQETWXSVTMO-CBABADBVSA-N:CYP3A4)
  - SMILES: `N[C@@H](CS/C(Cl)=C(/Cl)C(Cl)=C(Cl)Cl)C(=O)O`
  - Original sites: [8]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **TAXOL** (metxbiodb:RCINICONZNJXQF-YUKVPPSFSA-N:CYP3A4)
  - SMILES: `CC(=O)O[C@H]1C(=O)[C@]2(C)[C@@H](O)C[C@H]3OC[C@@]3(OC(C)=O)[...`
  - Original sites: [31]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **cp_195543** (metxbiodb:NZQDWKCNBOELAI-KSFYIVLOSA-N:CYP3A4)
  - SMILES: `O=C(O)c1ccc(C(F)(F)F)cc1-c1ccc2c(c1)OC[C@H](Cc1ccccc1)[C@H]2...`
  - Original sites: [1]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **dhea** (metxbiodb:FMGSKLZLMKYGDP-USOAJAOKSA-N:CYP3A4)
  - SMILES: `C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]...`
  - Original sites: [14]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **flucloxacillin** (metxbiodb:UIOFUWFRIANQPC-JKIFEVAISA-N:CYP3A4)
  - SMILES: `Cc1onc(-c2c(F)cccc2Cl)c1C(=O)N[C@@H]1C(=O)N2[C@@H]1SC(C)(C)[...`
  - Original sites: [5]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES

- **indiplon** (metxbiodb:CBIAWPMZSFFRGN-UHFFFAOYSA-N:CYP3A4)
  - SMILES: `CC(=O)N(C)c1cccc(-c2ccnc3c(C(=O)c4cccs4)cnn23)c1`
  - Original sites: [1]
  - Flags: PHYSICS_DISAGREES
  - Recommendation: Rejected: PHYSICS_DISAGREES


## Physics Disagreement Analysis

Cases where labeled SoM differs from physics top-3:

- **Cmpd 115**
  - Labeled: [19]
  - Physics top-3: [0, 24, 6]
  - Site 19: O, physics_score=0.10, pattern=

- **Abiraterone**
  - Labeled: [17]
  - Physics top-3: [6, 9, 24]
  - Site 17: C, physics_score=0.00, pattern=

- **Anastrozole**
  - Labeled: [0, 2, 3, 6]
  - Physics top-3: [8, 4, 9]
  - Site 0: C, physics_score=0.35, pattern=aliphatic_ch3
  - Site 2: C, physics_score=0.35, pattern=aliphatic_ch3
  - Site 3: C, physics_score=0.00, pattern=

- **Budesonide**
  - Labeled: [12, 22, 24]
  - Physics top-3: [10, 3, 5]
  - Site 12: C, physics_score=0.45, pattern=aliphatic_ch
  - Site 22: C, physics_score=0.40, pattern=aliphatic_ch2
  - Site 24: C, physics_score=0.35, pattern=aliphatic_ch3

- **CLARITHROMYCIN**
  - Labeled: [11, 34]
  - Physics top-3: [15, 39, 32]
  - Site 11: C, physics_score=0.40, pattern=aliphatic_ch2
  - Site 34: C, physics_score=0.78, pattern=alpha_o

- **CORTISOL**
  - Labeled: [11]
  - Physics top-3: [8, 12, 24]
  - Site 11: C, physics_score=0.45, pattern=aliphatic_ch

- **Clindamycin**
  - Labeled: [0, 2, 4, 19]
  - Physics top-3: [25, 5, 26]
  - Site 0: C, physics_score=0.35, pattern=aliphatic_ch3
  - Site 2: C, physics_score=0.40, pattern=aliphatic_ch2
  - Site 4: C, physics_score=0.40, pattern=aliphatic_ch2

- **Delamanid**
  - Labeled: [7, 13, 16, 18]
  - Physics top-3: [9, 25, 11]
  - Site 7: C, physics_score=0.00, pattern=
  - Site 13: C, physics_score=0.00, pattern=
  - Site 16: C, physics_score=0.00, pattern=

- **Dexamethasone**
  - Labeled: [8, 19, 21]
  - Physics top-3: [6, 17, 26]
  - Site 8: C, physics_score=0.45, pattern=aliphatic_ch
  - Site 19: C, physics_score=0.40, pattern=aliphatic_ch2
  - Site 21: C, physics_score=0.35, pattern=aliphatic_ch3

- **Eplerenone**
  - Labeled: [10, 29]
  - Physics top-3: [0, 5, 16]
  - Site 10: C, physics_score=0.40, pattern=aliphatic_ch2
  - Site 29: C, physics_score=0.45, pattern=aliphatic_ch


## Recommendations

### For Training:
1. Use **Gold** dataset (128 molecules) for primary training
2. Optionally add **Silver** dataset (137 molecules) with sample weighting
3. Review queue (73 molecules) requires manual expert review

### For Manual Review Priority:
1. Focus on 'PHYSICS_DISAGREES' cases - may indicate novel chemistry or labeling error
2. Check 'TOO_MANY_SITES' cases - may need to identify primary SoM
3. Verify 'SUSPICIOUS_ATOM' cases - especially oxygen-labeled sites

### Data Quality Improvements:
1. Cross-reference with metabolite structures from DrugBank/HMDB
2. Add external validated datasets (Zaretzki, XenoSite)
3. Consider docking to add 3D binding site information