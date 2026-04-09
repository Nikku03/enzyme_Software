# CYP3A4 Dataset Cleanup Changelog

**Date:** 2026-04-09
**Version:** Phase 0 v1

## Summary

- Original molecules: 699
- Final molecules: 696
- Removed: 3
- Site corrections: 2

## Confidence Distribution

- High confidence: 195
- Medium confidence: 455
- Low confidence: 46

---

## Removed Molecules (Wrong Enzyme)

### NNK
- **Reason:** NNK α-hydroxylation primarily by CYP2A6/2A13, not CYP3A4
- **Primary enzyme:** CYP2A6/CYP2A13

### nicotine
- **Reason:** CYP2A6 handles 70-80% via 5'-hydroxylation → cotinine. CYP3A4 not significant.
- **Primary enzyme:** CYP2A6

### N_nitrosonornicotine
- **Reason:** CYP2A6 primary for 5'-hydroxylation; CYP3A4 does 2'-OH at different site
- **Primary enzyme:** CYP2A6

---

## Site Corrections

### zileuton
- **Old site:** atom 2
- **New site:** atom 10
- **Reason:** CYP3A4 exclusively does sulfoxidation; ring hydroxylation is CYP1A2/CYP2C9

### diclofenac
- **Old site:** atom 1
- **New site:** atom 16
- **Reason:** CYP2C9 does 4'-hydroxylation (>99%); CYP3A4 does 5-hydroxylation (minor)

---

## Flagged: Minor CYP3A4 Pathway

- **phenprocoumon:** CYP2C9 dominant (S-phenprocoumon); CYP3A4 minor
- **mianserin:** CYP2D6/CYP1A2 are dominant; CYP3A4 minor contributor
- **hydromorphone:** UGT-mediated glucuronidation dominant; CYP role minimal

---

## Flagged: Insufficient Evidence

- **mesoridazine:** Limited specific CYP3A4 data
- **rifalazil:** Limited literature on CYP3A4 specificity
- **tamarixetin:** Flavonoid, complex metabolism
- **ezlopitant:** NK1 antagonist, limited metabolism data
- **voriconazole:** CYP2C19/CYP3A4 both contribute
- **reduced_diclofenac:** Derivative, limited data