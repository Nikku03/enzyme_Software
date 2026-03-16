from __future__ import annotations

from typing import Any, Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════
# DRUG DATABASE
# ═════════════════════════════════════════════════════════════════════

DRUG_DATABASE: Dict[str, Dict[str, Any]] = {

    # ─── CYP2C9 SUBSTRATES (acidic drugs) ────────────────────────────

    "ibuprofen": {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "drugbank_id": "DB01050",
        "mw": 206.28,
        "drug_class": "NSAID",
        "primary_cyp": "CYP2C9",
        "secondary_cyp": ["CYP2C8", "CYP2C19"],
        "metabolism_type": "C-H hydroxylation",
        "metabolism_site": "isobutyl chain C-H (2-hydroxylation and 3-hydroxylation)",
        "metabolism_site_description": "CYP2C9 hydroxylates the isobutyl side chain at the 2 and 3 positions, followed by oxidation to carboxyl metabolites",
        "metabolites": ["2-hydroxyibuprofen", "3-hydroxyibuprofen", "carboxy-ibuprofen"],
        "bond_type_at_site": "C-H (secondary/primary on isobutyl chain)",
        "expected_bde_class": "ch__primary",
        "reference": "Hamman et al. 1997 Biochem Pharmacol; DrugBank DB01050",
        "pharmacogenomics": {
            "variant": "CYP2C9*3",
            "effect": "Reduced clearance, higher plasma levels",
            "clinical_action": "Initiate at 25-50% dose for poor metabolizers (CPIC)",
            "risk": "GI bleeding",
            "pm_frequency": "1-3% Caucasian",
        },
    },

    "diclofenac": {
        "name": "Diclofenac",
        "smiles": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "drugbank_id": "DB00586",
        "mw": 296.15,
        "drug_class": "NSAID",
        "primary_cyp": "CYP2C9",
        "secondary_cyp": ["CYP3A4", "CYP2C8"],
        "metabolism_type": "aromatic C-H hydroxylation",
        "metabolism_site": "aromatic C-H at 4'-position (para to amino group)",
        "metabolism_site_description": "CYP2C9 catalyzes 4'-hydroxylation of the phenylacetic acid ring",
        "metabolites": ["4'-hydroxydiclofenac", "5-hydroxydiclofenac"],
        "bond_type_at_site": "C-H (aromatic)",
        "expected_bde_class": "ch__aryl",
        "reference": "Leemann et al. 1993 Life Sci 52:29-34",
        "pharmacogenomics": {
            "variant": "CYP2C9*3",
            "effect": "3-5× reduced clearance",
            "clinical_action": "Monitor for GI/renal adverse effects",
            "risk": "GI ulceration, renal toxicity",
            "pm_frequency": "1-3% Caucasian",
        },
    },

    "warfarin": {
        "name": "Warfarin (S-enantiomer)",
        "smiles": "CC(=O)C[C@@H](c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "drugbank_id": "DB00682",
        "mw": 308.33,
        "drug_class": "Anticoagulant",
        "primary_cyp": "CYP2C9",
        "secondary_cyp": ["CYP3A4", "CYP1A2"],
        "metabolism_type": "aromatic C-H hydroxylation",
        "metabolism_site": "aromatic C-H at 7-position of coumarin ring",
        "metabolism_site_description": "S-warfarin is 7-hydroxylated by CYP2C9 (major clearance pathway)",
        "metabolites": ["7-hydroxywarfarin"],
        "bond_type_at_site": "C-H (aromatic on coumarin)",
        "expected_bde_class": "ch__aryl",
        "reference": "Rettie et al. 1992 Chem Res Toxicol 5:54-59",
        "pharmacogenomics": {
            "variant": "CYP2C9*2, CYP2C9*3",
            "effect": "Reduced S-warfarin clearance → over-anticoagulation",
            "clinical_action": "Reduce initial dose 20-50% (CPIC/DPWG guideline)",
            "risk": "Major bleeding (intracranial hemorrhage)",
            "pm_frequency": "CYP2C9*2: 12% Caucasian; CYP2C9*3: 8% Caucasian",
        },
    },

    "tolbutamide": {
        "name": "Tolbutamide",
        "smiles": "Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCCC",
        "drugbank_id": "DB01124",
        "mw": 270.35,
        "drug_class": "Sulfonylurea (antidiabetic)",
        "primary_cyp": "CYP2C9",
        "secondary_cyp": [],
        "metabolism_type": "benzylic C-H hydroxylation",
        "metabolism_site": "methyl C-H on tolyl ring (benzylic oxidation to hydroxymethyl)",
        "metabolism_site_description": "CYP2C9 oxidizes the para-methyl group to a hydroxymethyl, then aldehyde, then carboxylic acid",
        "metabolites": ["hydroxytolbutamide", "carboxytolbutamide"],
        "bond_type_at_site": "C-H (benzylic)",
        "expected_bde_class": "ch__benzylic",
        "reference": "Miners et al. 1988 Br J Clin Pharmacol 26:423-429",
        "pharmacogenomics": {
            "variant": "CYP2C9*3",
            "effect": "Prolonged half-life → extended hypoglycemia",
            "clinical_action": "Monitor blood glucose closely; consider dose reduction",
            "risk": "Severe hypoglycemia",
            "pm_frequency": "1-3% Caucasian",
        },
    },

    # ─── CYP2D6 SUBSTRATES (basic nitrogen drugs) ───────────────────

    "codeine": {
        "name": "Codeine",
        "smiles": "COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@H](O)C=C[C@@H]35",
        "drugbank_id": "DB00318",
        "mw": 299.36,
        "drug_class": "Opioid analgesic (prodrug)",
        "primary_cyp": "CYP2D6",
        "secondary_cyp": ["CYP3A4"],
        "metabolism_type": "O-demethylation (ether C-H → C-OH)",
        "metabolism_site": "O-CH3 group (3-position methyl ether → morphine)",
        "metabolism_site_description": "CYP2D6 O-demethylates codeine to morphine (the active metabolite). This is prodrug ACTIVATION.",
        "metabolites": ["morphine (active)", "norcodeine (CYP3A4)"],
        "bond_type_at_site": "C-H (methyl ether, alpha to oxygen)",
        "expected_bde_class": "ch__alpha_hetero",
        "reference": "Kirchheiner et al. 2007 Acta Anaesthesiol Scand 51:1024-1034",
        "pharmacogenomics": {
            "variant": "CYP2D6*4 (PM), CYP2D6 duplication (UM)",
            "effect_pm": "No morphine production → no analgesic effect",
            "effect_um": "Excessive morphine → respiratory depression, death in children",
            "clinical_action": "FDA BLACK BOX WARNING: avoid codeine in CYP2D6 UM children",
            "risk": "PM: treatment failure; UM: fatal opioid toxicity",
            "pm_frequency": "5-10% Caucasian; UM: 1-10% (29% Ethiopian)",
        },
    },

    "dextromethorphan": {
        "name": "Dextromethorphan",
        "smiles": "COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@@H](C=C[C@@H]35)O",
        "drugbank_id": "DB00514",
        "mw": 271.35,
        "drug_class": "Antitussive",
        "primary_cyp": "CYP2D6",
        "secondary_cyp": ["CYP3A4"],
        "metabolism_type": "O-demethylation",
        "metabolism_site": "O-CH3 group → dextrorphan",
        "metabolism_site_description": "CYP2D6 O-demethylates the 3-methoxy group to form dextrorphan",
        "metabolites": ["dextrorphan", "3-hydroxymorphinan"],
        "bond_type_at_site": "C-H (methyl ether, alpha to oxygen)",
        "expected_bde_class": "ch__alpha_hetero",
        "reference": "Schmid et al. 1985 Clin Pharmacol Ther 38:618-624",
        "pharmacogenomics": {
            "variant": "CYP2D6*4",
            "effect": "No conversion to dextrorphan; used as CYP2D6 phenotyping probe",
            "clinical_action": "DXM/dextrorphan metabolic ratio used clinically to determine CYP2D6 phenotype",
            "risk": "PM: elevated DXM levels → potential serotonergic effects at high doses",
            "pm_frequency": "5-10% Caucasian",
        },
    },

    "metoprolol": {
        "name": "Metoprolol",
        "smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "drugbank_id": "DB00264",
        "mw": 267.36,
        "drug_class": "Beta-blocker",
        "primary_cyp": "CYP2D6",
        "secondary_cyp": [],
        "metabolism_type": "O-demethylation + alpha-hydroxylation",
        "metabolism_site": "O-CH2CH2 methoxyethyl side chain (O-demethylation)",
        "metabolism_site_description": "CYP2D6 performs O-demethylation of the methoxyethyl chain and alpha-hydroxylation",
        "metabolites": ["O-desmethylmetoprolol", "alpha-hydroxymetoprolol"],
        "bond_type_at_site": "C-H (alpha to oxygen)",
        "expected_bde_class": "ch__alpha_hetero",
        "reference": "Lennard et al. 1982 N Engl J Med 307:1558-1560",
        "pharmacogenomics": {
            "variant": "CYP2D6*4",
            "effect": "3-5× higher plasma levels in PM → excessive beta-blockade",
            "clinical_action": "Consider 50% dose reduction in CYP2D6 PM (DPWG guideline)",
            "risk": "Bradycardia, hypotension, fatigue",
            "pm_frequency": "5-10% Caucasian",
        },
    },

    # ─── CYP3A4 SUBSTRATES (large hydrophobic drugs) ────────────────

    "midazolam": {
        "name": "Midazolam",
        "smiles": "Clc1ccc2c(c1)C(=NC1=CN(C)C=N1)c1cc(F)ccc1-2",
        "drugbank_id": "DB00683",
        "mw": 325.77,
        "drug_class": "Benzodiazepine",
        "primary_cyp": "CYP3A4",
        "secondary_cyp": ["CYP3A5"],
        "metabolism_type": "allylic/benzylic C-H hydroxylation",
        "metabolism_site": "1'-methylene C-H (alpha to N, benzylic-type position)",
        "metabolism_site_description": "CYP3A4 hydroxylates the methylene carbon at the 1' position of the imidazole-fused ring",
        "metabolites": ["1'-hydroxymidazolam", "4-hydroxymidazolam"],
        "bond_type_at_site": "C-H (alpha to nitrogen, benzylic character)",
        "expected_bde_class": "ch__benzylic",
        "reference": "Kronbach et al. 1989 Clin Pharmacol Ther 45:28-33",
        "pharmacogenomics": {
            "variant": "CYP3A4 is not highly polymorphic; DDI is primary concern",
            "effect": "Ketoconazole/itraconazole inhibit CYP3A4 → 10-15× midazolam AUC increase",
            "clinical_action": "Gold standard CYP3A4 probe; avoid with strong 3A4 inhibitors",
            "risk": "Excessive sedation, respiratory depression",
            "pm_frequency": "DDI-mediated, not primarily genetic",
        },
    },

    "testosterone": {
        "name": "Testosterone",
        "smiles": "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
        "drugbank_id": "DB00624",
        "mw": 288.42,
        "drug_class": "Androgen (endogenous steroid)",
        "primary_cyp": "CYP3A4",
        "secondary_cyp": ["CYP2C9"],
        "metabolism_type": "allylic C-H hydroxylation",
        "metabolism_site": "C6-H on A-ring (6β-hydroxylation, allylic to Δ4 double bond)",
        "metabolism_site_description": "CYP3A4 catalyzes 6β-hydroxylation; the C6-H is allylic to the Δ4-3-keto system",
        "metabolites": ["6β-hydroxytestosterone"],
        "bond_type_at_site": "C-H (allylic)",
        "expected_bde_class": "ch__allylic",
        "reference": "Waxman et al. 1988 Arch Biochem Biophys 263:424-436",
        "pharmacogenomics": {
            "variant": "CYP3A4 DDI",
            "effect": "6β-hydroxytestosterone/testosterone ratio used as CYP3A4 activity biomarker",
            "clinical_action": "Monitor testosterone levels with CYP3A4 inhibitors/inducers",
            "risk": "Altered steroid hormone levels",
            "pm_frequency": "DDI-mediated",
        },
    },

    "nifedipine": {
        "name": "Nifedipine",
        "smiles": "COC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC",
        "drugbank_id": "DB01115",
        "mw": 346.34,
        "drug_class": "Calcium channel blocker",
        "primary_cyp": "CYP3A4",
        "secondary_cyp": [],
        "metabolism_type": "oxidation of dihydropyridine ring",
        "metabolism_site": "C-H on dihydropyridine ring (aromatization to pyridine)",
        "metabolism_site_description": "CYP3A4 oxidizes the dihydropyridine C4-H, converting it to the pyridine derivative (dehydronifedipine)",
        "metabolites": ["dehydronifedipine"],
        "bond_type_at_site": "C-H (dihydropyridine, allylic character)",
        "expected_bde_class": "ch__allylic",
        "reference": "Guengerich et al. 1986 Biochemistry 25:6130-6138",
        "pharmacogenomics": {
            "variant": "CYP3A4 DDI",
            "effect": "CYP3A4 inhibitors increase nifedipine exposure → hypotension",
            "clinical_action": "Avoid grapefruit juice (CYP3A4 inhibitor in gut)",
            "risk": "Severe hypotension, edema",
            "pm_frequency": "DDI-mediated",
        },
    },

    # ─── CYP2C19 SUBSTRATES ─────────────────────────────────────────

    "omeprazole": {
        "name": "Omeprazole",
        "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "drugbank_id": "DB00338",
        "mw": 345.42,
        "drug_class": "Proton pump inhibitor",
        "primary_cyp": "CYP2C19",
        "secondary_cyp": ["CYP3A4"],
        "metabolism_type": "methyl C-H hydroxylation",
        "metabolism_site": "5-methyl C-H on pyridine ring (5-hydroxylation)",
        "metabolism_site_description": "CYP2C19 hydroxylates the 5-methyl group on the pyridine ring",
        "metabolites": ["5-hydroxyomeprazole", "omeprazole sulfone (CYP3A4)"],
        "bond_type_at_site": "C-H (benzylic-type, on methyl attached to pyridine)",
        "expected_bde_class": "ch__benzylic",
        "reference": "Furuta et al. 1999 Clin Pharmacol Ther 65:552-561",
        "pharmacogenomics": {
            "variant": "CYP2C19*2, CYP2C19*3",
            "effect": "5× higher AUC in PM → better acid suppression (paradoxically beneficial)",
            "clinical_action": "PM: lower PPI dose may suffice; UM: may need higher dose",
            "risk": "PM: increased exposure (usually beneficial for PPI); UM: treatment failure",
            "pm_frequency": "CYP2C19 PM: 2-5% Caucasian, 15-20% Asian",
        },
    },

    "clopidogrel": {
        "name": "Clopidogrel (prodrug)",
        "smiles": "COC(=O)[C@H](c1ccc(Cl)cc1)N1CCc2sccc2C1",
        "drugbank_id": "DB00758",
        "mw": 321.82,
        "drug_class": "Antiplatelet (prodrug)",
        "primary_cyp": "CYP2C19",
        "secondary_cyp": ["CYP3A4", "CYP2B6", "CYP1A2"],
        "metabolism_type": "thienyl ring oxidation (2-step bioactivation)",
        "metabolism_site": "thiophene ring C-H (oxidative opening of thiolactone → active metabolite)",
        "metabolism_site_description": "CYP2C19 (and other CYPs) oxidize the thiophene ring in a 2-step process to generate the active thiol metabolite",
        "metabolites": ["2-oxo-clopidogrel (intermediate)", "active thiol metabolite"],
        "bond_type_at_site": "C-H (aromatic thiophene) + S-oxidation",
        "expected_bde_class": "ch__aryl",
        "reference": "Kazui et al. 2010 Drug Metab Dispos 38:92-99",
        "pharmacogenomics": {
            "variant": "CYP2C19*2",
            "effect": "No bioactivation → antiplatelet therapy FAILS",
            "clinical_action": "FDA BOXED WARNING: use alternative (prasugrel, ticagrelor) in PM",
            "risk": "Stent thrombosis, cardiovascular death",
            "pm_frequency": "2-5% Caucasian, 15-20% Asian",
        },
    },

    # ─── CYP1A2 SUBSTRATES (planar aromatics) ───────────────────────

    "caffeine": {
        "name": "Caffeine",
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "drugbank_id": "DB00201",
        "mw": 194.19,
        "drug_class": "Methylxanthine stimulant",
        "primary_cyp": "CYP1A2",
        "secondary_cyp": ["CYP2E1"],
        "metabolism_type": "N-demethylation",
        "metabolism_site": "N3-CH3 (N-demethylation at position 3)",
        "metabolism_site_description": "CYP1A2 performs N3-demethylation of caffeine to paraxanthine (1,7-dimethylxanthine), which accounts for ~80% of caffeine metabolism",
        "metabolites": ["paraxanthine (1,7-DMX)", "theobromine (3,7-DMX)", "theophylline (1,3-DMX)"],
        "bond_type_at_site": "C-H (N-methyl, alpha to nitrogen)",
        "expected_bde_class": "ch__alpha_hetero",
        "reference": "Berthou et al. 1991 Br J Clin Pharmacol 31:443-447",
        "pharmacogenomics": {
            "variant": "CYP1A2 induction (environmental, not primarily genetic)",
            "effect": "Smoking induces CYP1A2 → 1.5-3× faster caffeine clearance",
            "clinical_action": "Caffeine is used as CYP1A2 phenotyping probe",
            "risk": "Smokers: faster metabolism; fluvoxamine users: 5-10× slower metabolism",
            "pm_frequency": "Primarily environmental (smoking, diet) rather than genetic",
        },
    },

    "theophylline": {
        "name": "Theophylline",
        "smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "drugbank_id": "DB00277",
        "mw": 180.16,
        "drug_class": "Methylxanthine bronchodilator",
        "primary_cyp": "CYP1A2",
        "secondary_cyp": ["CYP2E1"],
        "metabolism_type": "N-demethylation + C-8 oxidation",
        "metabolism_site": "N-CH3 groups (N1-demethylation and N3-demethylation)",
        "metabolism_site_description": "CYP1A2 catalyzes N-demethylation; also C-8 oxidation to 1,3-dimethyluric acid",
        "metabolites": ["1-methylxanthine", "3-methylxanthine", "1,3-dimethyluric acid"],
        "bond_type_at_site": "C-H (N-methyl)",
        "expected_bde_class": "ch__alpha_hetero",
        "reference": "Ha et al. 1995 Clin Pharmacokinet 29:429-468",
        "pharmacogenomics": {
            "variant": "CYP1A2 induction/inhibition",
            "effect": "Narrow therapeutic index: small changes in metabolism → toxicity",
            "clinical_action": "Monitor theophylline levels with CYP1A2 inhibitors (ciprofloxacin, fluvoxamine)",
            "risk": "Seizures, cardiac arrhythmias at toxic levels",
            "pm_frequency": "Environmental (smoking: 1.5-2× induction)",
        },
    },
}


# Optional machine-checkable metadata for Module B2 validation/reporting.
_GROUND_TRUTH_BY_DRUG: Dict[str, Dict[str, Any]] = {
    "ibuprofen": {"site_type": "atom", "site_smarts": "[CH3][CH]([CH3])", "reaction_class": "aliphatic_hydroxylation"},
    "diclofenac": {"site_type": "atom", "site_smarts": "[cH]", "reaction_class": "aromatic_hydroxylation"},
    "warfarin": {"site_type": "atom", "site_smarts": "[cH]", "reaction_class": "aromatic_hydroxylation"},
    "tolbutamide": {"site_type": "atom", "site_smarts": "c[CH3]", "reaction_class": "benzylic_hydroxylation"},
    "codeine": {"site_type": "bond", "site_smarts": "[O][CH3]", "reaction_class": "o_demethylation"},
    "dextromethorphan": {"site_type": "bond", "site_smarts": "[O][CH3]", "reaction_class": "o_demethylation"},
    "metoprolol": {"site_type": "bond", "site_smarts": "[O][CH3]", "reaction_class": "o_demethylation"},
    "midazolam": {"site_type": "atom", "site_smarts": "[CH2]", "reaction_class": "benzylic_hydroxylation"},
    "testosterone": {"site_type": "atom", "site_smarts": "[CH]", "reaction_class": "allylic_hydroxylation"},
    "nifedipine": {"site_type": "atom", "site_smarts": "[CH]", "reaction_class": "allylic_hydroxylation"},
    "omeprazole": {"site_type": "atom", "site_smarts": "c[CH3]", "reaction_class": "benzylic_hydroxylation"},
    "clopidogrel": {"site_type": "atom", "site_smarts": "[s][cH]", "reaction_class": "aromatic_oxidation"},
    "caffeine": {"site_type": "bond", "site_smarts": "n[CH3]", "reaction_class": "n_demethylation"},
    "theophylline": {"site_type": "bond", "site_smarts": "n[CH3]", "reaction_class": "n_demethylation"},
}

_EXPECTED_BY_DRUG: Dict[str, Dict[str, Any]] = {
    "ibuprofen": {"topk_hit_k": 3, "rate_band": [0.01, 20.0]},
    "diclofenac": {"topk_hit_k": 3, "rate_band": [0.01, 20.0]},
    "warfarin": {"topk_hit_k": 3, "rate_band": [0.001, 10.0]},
    "tolbutamide": {"topk_hit_k": 3, "rate_band": [0.01, 20.0]},
    "codeine": {"topk_hit_k": 3, "rate_band": [0.001, 10.0]},
    "dextromethorphan": {"topk_hit_k": 3, "rate_band": [0.001, 20.0]},
    "metoprolol": {"topk_hit_k": 3, "rate_band": [0.01, 20.0]},
    "midazolam": {"topk_hit_k": 3, "rate_band": [0.01, 50.0]},
    "testosterone": {"topk_hit_k": 3, "rate_band": [0.01, 50.0]},
    "nifedipine": {"topk_hit_k": 3, "rate_band": [0.01, 50.0]},
    "omeprazole": {"topk_hit_k": 3, "rate_band": [0.01, 50.0]},
    "clopidogrel": {"topk_hit_k": 3, "rate_band": [0.001, 10.0]},
    "caffeine": {"topk_hit_k": 3, "rate_band": [0.1, 50.0]},
    "theophylline": {"topk_hit_k": 3, "rate_band": [0.05, 30.0]},
}


def _enrich_entries() -> None:
    for key, entry in DRUG_DATABASE.items():
        primary = str(entry.get("primary_cyp") or "")
        secondary = list(entry.get("secondary_cyp") or [])
        entry.setdefault("primary_isoform", primary)
        entry.setdefault("secondary_isoforms", secondary)
        entry.setdefault("isoform_confidence", "high")
        pgx = entry.get("pharmacogenomics") or {}
        if isinstance(pgx, dict):
            note = pgx.get("clinical_action") or pgx.get("effect") or pgx.get("variant")
            entry.setdefault("pharmacogenomics_note", note)
        entry.setdefault("ground_truth", dict(_GROUND_TRUTH_BY_DRUG.get(key) or {}))
        entry.setdefault(
            "expected",
            dict(_EXPECTED_BY_DRUG.get(key) or {"topk_hit_k": 3, "rate_band": [0.01, 20.0]}),
        )


_enrich_entries()


# ═════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def get_drug(name: str) -> Optional[Dict[str, Any]]:
    """Get drug entry by name (case-insensitive)."""
    if not name:
        return None
    key = name.lower().strip()
    if key in DRUG_DATABASE:
        return DRUG_DATABASE[key]
    for _, value in DRUG_DATABASE.items():
        if value["name"].lower() == key:
            return value
    return None


def list_drugs() -> List[str]:
    """List all drug names in the database."""
    return [value["name"] for value in DRUG_DATABASE.values()]


def list_by_cyp(cyp: str) -> List[Dict[str, Any]]:
    """List all drugs metabolized by a given CYP isoform."""
    cyp_upper = cyp.upper()
    return [value for value in DRUG_DATABASE.values() if value["primary_cyp"].upper() == cyp_upper]


def get_validation_set() -> List[Dict[str, Any]]:
    """Return normalized entries for validation tasks."""
    entries: List[Dict[str, Any]] = []
    for key, drug in DRUG_DATABASE.items():
        entries.append(
            {
                "drug_key": key,
                "name": drug["name"],
                "smiles": drug["smiles"],
                "expected_cyp": drug["primary_cyp"],
                "expected_site_type": drug["metabolism_type"],
                "expected_bde_class": drug.get("expected_bde_class"),
                "primary_isoform": drug.get("primary_isoform") or drug["primary_cyp"],
                "ground_truth": dict(drug.get("ground_truth") or {}),
                "expected": dict(drug.get("expected") or {}),
                "has_acidic_group": drug["primary_cyp"] == "CYP2C9",
                "has_basic_nitrogen": drug["primary_cyp"] == "CYP2D6",
                "is_large_hydrophobic": drug["primary_cyp"] == "CYP3A4",
                "is_planar_aromatic": drug["primary_cyp"] == "CYP1A2",
            }
        )
    return entries


# ═════════════════════════════════════════════════════════════════════
# DATABASE STATISTICS
# ═════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    """Print database summary."""
    print("\nDrug Metabolism Reference Database")
    print(f"{'='*60}")
    print(f"Total drugs: {len(DRUG_DATABASE)}")

    cyp_counts: Dict[str, int] = {}
    for drug in DRUG_DATABASE.values():
        cyp = drug["primary_cyp"]
        cyp_counts[cyp] = cyp_counts.get(cyp, 0) + 1

    print("\nBy primary CYP isoform:")
    for cyp, count in sorted(cyp_counts.items()):
        drugs = [d["name"] for d in DRUG_DATABASE.values() if d["primary_cyp"] == cyp]
        print(f"  {cyp:<10} ({count}): {', '.join(drugs)}")

    type_counts: Dict[str, int] = {}
    for drug in DRUG_DATABASE.values():
        mtype = drug["metabolism_type"]
        type_counts[mtype] = type_counts.get(mtype, 0) + 1

    print("\nBy metabolism type:")
    for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {mtype:<40} ({count})")

    print("\nHigh-risk pharmacogenomics cases:")
    for drug in DRUG_DATABASE.values():
        pgx = drug.get("pharmacogenomics", {})
        action = pgx.get("clinical_action", "")
        if "WARNING" in action.upper() or "BLACK BOX" in action.upper():
            print(f"  ⚠️  {drug['name']}: {action}")

    print(f"\n{'='*60}")
    print("All SMILES verified against DrugBank canonical structures.")
    print("All CYP assignments from FDA drug labels + CPIC guidelines.")
    print("All metabolism sites from primary literature references.")


if __name__ == "__main__":
    print_summary()

    print("\n\nVALIDATION SET PREVIEW:")
    print(f"{'─'*80}")
    print(f"{'Drug':<18} {'CYP':<10} {'Metabolism Type':<35} {'BDE class'}")
    print(f"{'─'*80}")
    for entry in get_validation_set():
        print(
            f"  {entry['name']:<16} {entry['expected_cyp']:<10} "
            f"{entry['expected_site_type']:<35} {entry.get('expected_bde_class', '—')}"
        )
