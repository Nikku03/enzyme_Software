from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from enzyme_software.calibration.drug_metabolism_db import DRUG_DATABASE, get_drug


# Activity score convention: 0=no function, 0.5=decreased, 1=normal, >1=increased.
ALLELE_ACTIVITY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "CYP2D6": {
        "*1": {"activity": 1.0, "function": "normal"},
        "*2": {"activity": 1.0, "function": "normal"},
        "*3": {"activity": 0.0, "function": "no_function"},
        "*4": {"activity": 0.0, "function": "no_function"},
        "*5": {"activity": 0.0, "function": "no_function", "note": "gene_deletion"},
        "*6": {"activity": 0.0, "function": "no_function"},
        "*9": {"activity": 0.5, "function": "decreased"},
        "*10": {"activity": 0.5, "function": "decreased"},
        "*17": {"activity": 0.5, "function": "decreased"},
        "*41": {"activity": 0.5, "function": "decreased"},
        "*1xN": {"activity": 2.0, "function": "increased", "note": "gene_duplication"},
        "*2xN": {"activity": 2.0, "function": "increased", "note": "gene_duplication"},
    },
    "CYP2C9": {
        "*1": {"activity": 1.0, "function": "normal"},
        "*2": {"activity": 0.5, "function": "decreased"},
        "*3": {"activity": 0.0, "function": "no_function"},
        "*5": {"activity": 0.0, "function": "no_function"},
        "*6": {"activity": 0.0, "function": "no_function"},
        "*8": {"activity": 0.5, "function": "decreased"},
        "*11": {"activity": 0.5, "function": "decreased"},
    },
    "CYP2C19": {
        "*1": {"activity": 1.0, "function": "normal"},
        "*2": {"activity": 0.0, "function": "no_function"},
        "*3": {"activity": 0.0, "function": "no_function"},
        "*17": {"activity": 1.5, "function": "increased", "note": "gain_of_function"},
    },
    "CYP1A2": {
        "*1": {"activity": 1.0, "function": "normal"},
        "*1F": {"activity": 1.5, "function": "increased", "note": "inducible_allele"},
    },
    "VKORC1": {
        "GG": {
            "activity": 1.0,
            "function": "normal",
            "note": "wild_type_normal_vitamin_k_recycling",
        },
        "GA": {
            "activity": 0.75,
            "function": "decreased",
            "note": "heterozygous_approximately_25pct_less_vkorc1",
        },
        "AA": {
            "activity": 0.5,
            "function": "low",
            "note": "homozygous_variant_high_warfarin_sensitivity",
        },
    },
}


PHENOTYPE_THRESHOLDS: Dict[str, List[tuple[float, float, str, str]]] = {
    "CYP2D6": [
        (0.0, 0.0, "poor_metabolizer", "PM"),
        (0.25, 1.25, "intermediate_metabolizer", "IM"),
        (1.5, 2.25, "normal_metabolizer", "NM"),
        (2.5, 99.0, "ultrarapid_metabolizer", "UM"),
    ],
    "CYP2C9": [
        (0.0, 0.0, "poor_metabolizer", "PM"),
        (0.5, 1.0, "intermediate_metabolizer", "IM"),
        (1.5, 2.0, "normal_metabolizer", "NM"),
    ],
    "CYP2C19": [
        (0.0, 0.0, "poor_metabolizer", "PM"),
        (0.5, 1.0, "intermediate_metabolizer", "IM"),
        (2.0, 2.0, "normal_metabolizer", "NM"),
        (2.5, 99.0, "ultrarapid_metabolizer", "UM"),
    ],
    "VKORC1": [
        (0.5, 0.5, "high_sensitivity", "HS"),
        (0.75, 0.75, "intermediate_sensitivity", "IS"),
        (1.0, 1.0, "normal_sensitivity", "NS"),
    ],
}


ALLELE_FREQUENCIES: Dict[str, Dict[str, Dict[str, float]]] = {
    "CYP2D6": {
        "caucasian": {
            "*1": 0.35,
            "*2": 0.25,
            "*3": 0.02,
            "*4": 0.20,
            "*5": 0.03,
            "*6": 0.01,
            "*9": 0.02,
            "*10": 0.02,
            "*41": 0.08,
            "*1xN": 0.02,
        },
        "african": {
            "*1": 0.30,
            "*2": 0.15,
            "*4": 0.07,
            "*5": 0.06,
            "*10": 0.05,
            "*17": 0.20,
            "*41": 0.10,
            "*1xN": 0.05,
        },
        "east_asian": {
            "*1": 0.30,
            "*2": 0.15,
            "*4": 0.01,
            "*5": 0.06,
            "*10": 0.40,
            "*41": 0.03,
            "*1xN": 0.01,
        },
    },
    "CYP2C9": {
        "caucasian": {"*1": 0.80, "*2": 0.12, "*3": 0.08},
        "african": {
            "*1": 0.88,
            "*2": 0.03,
            "*3": 0.02,
            "*5": 0.02,
            "*8": 0.03,
            "*11": 0.02,
        },
        "east_asian": {"*1": 0.96, "*2": 0.00, "*3": 0.04},
    },
    "CYP2C19": {
        "caucasian": {"*1": 0.63, "*2": 0.15, "*3": 0.005, "*17": 0.21},
        "african": {"*1": 0.55, "*2": 0.17, "*3": 0.01, "*17": 0.26},
        "east_asian": {"*1": 0.60, "*2": 0.30, "*3": 0.08, "*17": 0.02},
    },
    "VKORC1": {
        "caucasian": {"GG": 0.37, "GA": 0.47, "AA": 0.16},
        "african": {"GG": 0.70, "GA": 0.27, "AA": 0.03},
        "east_asian": {"GG": 0.10, "GA": 0.40, "AA": 0.50},
    },
}


QUANTITATIVE_PK: Dict[str, Dict[str, Any]] = {
    "warfarin": {
        "standard_dose_mg": 5.0,
        "therapeutic_range": "INR_2.0_to_3.0",
        "pk_by_phenotype": {
            "CYP2C9": {
                "NM": {
                    "auc_fold": 1.0,
                    "half_life_h": 36,
                    "clearance_change": "normal",
                    "dose_mg": 5.0,
                },
                "IM": {
                    "auc_fold": 1.5,
                    "half_life_h": 50,
                    "clearance_change": "approximately_30pct_reduced",
                    "dose_mg": 3.75,
                },
                "PM": {
                    "auc_fold": 2.5,
                    "half_life_h": 70,
                    "clearance_change": "approximately_60pct_reduced",
                    "dose_mg": 2.5,
                },
            },
            "VKORC1": {
                "NS": {
                    "sensitivity": "normal",
                    "dose_adjustment": 0.0,
                },
                "IS": {
                    "sensitivity": "increased",
                    "dose_adjustment": -1.5,
                },
                "HS": {
                    "sensitivity": "high",
                    "dose_adjustment": -2.5,
                },
            },
        },
        "reference": "Gage_et_al_2008_Clin_Pharmacol_Ther",
    },
    "codeine": {
        "standard_dose_mg": 30.0,
        "pk_by_phenotype": {
            "CYP2D6": {
                "NM": {
                    "morphine_conversion_pct": 10,
                    "auc_fold": 1.0,
                    "half_life_h": 3.0,
                    "dose_mg": 30.0,
                },
                "IM": {
                    "morphine_conversion_pct": 5,
                    "auc_fold": 0.7,
                    "half_life_h": 3.5,
                    "dose_mg": 30.0,
                },
                "PM": {
                    "morphine_conversion_pct": 0,
                    "auc_fold": 1.3,
                    "half_life_h": 4.0,
                    "dose_mg": None,
                    "note": "AVOID",
                },
                "UM": {
                    "morphine_conversion_pct": 30,
                    "auc_fold": 0.5,
                    "half_life_h": 2.0,
                    "dose_mg": None,
                    "note": "CONTRAINDICATED",
                },
            },
        },
        "reference": "Kirchheiner_et_al_2007_Acta_Anaesthesiol_Scand",
    },
    "clopidogrel": {
        "standard_dose_mg": 75.0,
        "pk_by_phenotype": {
            "CYP2C19": {
                "NM": {
                    "active_metabolite_auc_fold": 1.0,
                    "platelet_inhibition_pct": 50,
                    "dose_mg": 75.0,
                },
                "IM": {
                    "active_metabolite_auc_fold": 0.6,
                    "platelet_inhibition_pct": 30,
                    "dose_mg": 75.0,
                    "note": "consider_alternative",
                },
                "PM": {
                    "active_metabolite_auc_fold": 0.25,
                    "platelet_inhibition_pct": 10,
                    "dose_mg": None,
                    "note": "AVOID_use_prasugrel_or_ticagrelor",
                },
                "UM": {
                    "active_metabolite_auc_fold": 1.5,
                    "platelet_inhibition_pct": 65,
                    "dose_mg": 75.0,
                },
            },
        },
        "reference": "Mega_et_al_2009_NEJM",
    },
    "ibuprofen": {
        "standard_dose_mg": 400.0,
        "pk_by_phenotype": {
            "CYP2C9": {
                "NM": {
                    "auc_fold": 1.0,
                    "half_life_h": 2.5,
                    "clearance_change": "normal",
                    "dose_mg": 400.0,
                },
                "IM": {
                    "auc_fold": 1.5,
                    "half_life_h": 3.5,
                    "clearance_change": "approximately_30pct_reduced",
                    "dose_mg": 400.0,
                    "note": "lowest_effective_dose",
                },
                "PM": {
                    "auc_fold": 2.0,
                    "half_life_h": 5.0,
                    "clearance_change": "approximately_50pct_reduced",
                    "dose_mg": 200.0,
                },
            },
        },
        "reference": "Theken_et_al_2020_Clin_Pharmacol_Ther",
    },
    "omeprazole": {
        "standard_dose_mg": 20.0,
        "pk_by_phenotype": {
            "CYP2C19": {
                "NM": {"auc_fold": 1.0, "half_life_h": 1.0, "dose_mg": 20.0},
                "IM": {"auc_fold": 2.0, "half_life_h": 2.0, "dose_mg": 20.0},
                "PM": {
                    "auc_fold": 5.0,
                    "half_life_h": 4.0,
                    "dose_mg": 10.0,
                    "note": "lower_dose_sufficient",
                },
                "UM": {
                    "auc_fold": 0.5,
                    "half_life_h": 0.5,
                    "dose_mg": 40.0,
                    "note": "may_need_double_dose",
                },
            },
        },
        "reference": "Lima_et_al_2021_Clin_Pharmacol_Ther",
    },
    "metoprolol": {
        "standard_dose_mg": 100.0,
        "pk_by_phenotype": {
            "CYP2D6": {
                "NM": {"auc_fold": 1.0, "half_life_h": 3.5, "dose_mg": 100.0},
                "IM": {"auc_fold": 1.5, "half_life_h": 5.0, "dose_mg": 100.0},
                "PM": {"auc_fold": 5.0, "half_life_h": 9.0, "dose_mg": 50.0},
                "UM": {
                    "auc_fold": 0.4,
                    "half_life_h": 2.0,
                    "dose_mg": 150.0,
                    "note": "or_switch_to_bisoprolol",
                },
            },
        },
        "reference": "DPWG_and_Swen_et_al_2011_Clin_Pharmacol_Ther",
    },
}


CPIC_GUIDELINES: Dict[tuple[str, str], Dict[str, Any]] = {
    ("warfarin", "CYP2C9"): {
        "cpic_level": "A",
        "recommendations": {
            "NM": {"dose_adjustment": "standard", "action": "Use standard dosing algorithm."},
            "IM": {
                "dose_adjustment": "reduce_25pct",
                "action": "Reduce initial dose by about 25%. Consider lower maintenance dose.",
            },
            "PM": {
                "dose_adjustment": "reduce_50pct",
                "action": "Reduce initial dose by about 50%. Frequent INR monitoring required.",
            },
        },
        "additional": "Consider VKORC1 genotype for comprehensive dosing.",
        "reference": "Johnson_et_al_2017_Clin_Pharmacol_Ther",
    },
    ("codeine", "CYP2D6"): {
        "cpic_level": "A",
        "recommendations": {
            "NM": {"dose_adjustment": "standard", "action": "Use standard codeine dose."},
            "IM": {
                "dose_adjustment": "standard_or_reduce",
                "action": "Use standard dose; monitor for reduced efficacy.",
            },
            "PM": {
                "dose_adjustment": "avoid",
                "action": "Avoid codeine. Use non-CYP2D6-dependent analgesic.",
            },
            "UM": {
                "dose_adjustment": "contraindicated",
                "action": "Avoid codeine. Contraindicated due to morphine toxicity risk.",
            },
        },
        "fda_warning": "BLACK_BOX: deaths reported in CYP2D6 ultrarapid-metabolizer children post-tonsillectomy",
        "reference": "Crews_et_al_2014_Clin_Pharmacol_Ther",
    },
    ("clopidogrel", "CYP2C19"): {
        "cpic_level": "A",
        "recommendations": {
            "NM": {"dose_adjustment": "standard", "action": "Standard 75 mg daily dose."},
            "IM": {
                "dose_adjustment": "alternative_preferred",
                "action": "Consider prasugrel or ticagrelor if no contraindication.",
            },
            "PM": {
                "dose_adjustment": "avoid",
                "action": "Avoid clopidogrel. Use prasugrel or ticagrelor.",
            },
            "UM": {
                "dose_adjustment": "standard",
                "action": "Standard dose; may have enhanced antiplatelet response.",
            },
        },
        "fda_warning": "BOXED_WARNING: diminished antiplatelet effect in CYP2C19 poor metabolizers",
        "reference": "Scott_et_al_2013_Clin_Pharmacol_Ther",
    },
    ("ibuprofen", "CYP2C9"): {
        "cpic_level": "A",
        "recommendations": {
            "NM": {"dose_adjustment": "standard", "action": "Standard dosing."},
            "IM": {
                "dose_adjustment": "start_low",
                "action": "Initiate at lowest recommended dose for shortest duration.",
            },
            "PM": {
                "dose_adjustment": "reduce_50pct",
                "action": "Initiate at 25-50% of starting dose or use alternate NSAID.",
            },
        },
        "reference": "Theken_et_al_2020_Clin_Pharmacol_Ther",
    },
    ("omeprazole", "CYP2C19"): {
        "cpic_level": "B",
        "recommendations": {
            "NM": {"dose_adjustment": "standard", "action": "Standard 20 mg daily."},
            "IM": {"dose_adjustment": "standard", "action": "Standard dose."},
            "PM": {"dose_adjustment": "reduce", "action": "Consider lower dose."},
            "UM": {
                "dose_adjustment": "increase",
                "action": "Increase dose by 50-100% for eradication regimens if needed.",
            },
        },
        "reference": "Lima_et_al_2021_Clin_Pharmacol_Ther",
    },
    ("metoprolol", "CYP2D6"): {
        "cpic_level": "B",
        "recommendations": {
            "NM": {"dose_adjustment": "standard", "action": "Standard dosing per indication."},
            "IM": {
                "dose_adjustment": "standard",
                "action": "Standard dosing; monitor heart rate and blood pressure.",
            },
            "PM": {
                "dose_adjustment": "reduce_50pct",
                "action": "Consider about 50% dose reduction or switch to bisoprolol.",
            },
            "UM": {
                "dose_adjustment": "increase_or_switch",
                "action": "May require higher dose or switch for reliable beta-blockade.",
            },
        },
        "reference": "DPWG_guidance_Swen_et_al_2011_Clin_Pharmacol_Ther",
    },
}


CYP_INHIBITORS: Dict[str, Dict[str, List[str]]] = {
    "CYP3A4": {
        "strong": ["ketoconazole", "itraconazole", "clarithromycin", "ritonavir", "grapefruit_juice"],
        "moderate": ["erythromycin", "fluconazole", "diltiazem", "verapamil"],
        "weak": ["cimetidine"],
    },
    "CYP2D6": {
        "strong": ["fluoxetine", "paroxetine", "bupropion", "quinidine"],
        "moderate": ["duloxetine", "sertraline"],
    },
    "CYP2C9": {"strong": ["fluconazole"], "moderate": ["amiodarone", "fluvoxamine"]},
    "CYP2C19": {"strong": ["fluvoxamine", "fluoxetine"], "moderate": ["omeprazole"]},
    "CYP1A2": {
        "strong": ["fluvoxamine", "ciprofloxacin"],
        "environmental": ["smoking_cessation"],
    },
}


CYP_INDUCERS: Dict[str, List[str]] = {
    "CYP3A4": ["rifampin", "carbamazepine", "phenytoin", "st_johns_wort"],
    "CYP1A2": ["smoking", "charbroiled_meat", "cruciferous_vegetables"],
    "CYP2C9": ["rifampin"],
    "CYP2C19": ["rifampin"],
}


def determine_phenotype(gene: str, diplotype: str) -> Dict[str, Any]:
    gene_norm = str(gene or "").upper()
    raw = str(diplotype or "").strip()

    if gene_norm == "VKORC1":
        allele = ALLELE_ACTIVITY.get("VKORC1", {}).get(raw)
        if allele is None:
            return {"error": f"Unknown VKORC1 genotype {raw}"}
        activity_score = float(allele["activity"])
        phenotype = "unknown"
        abbr = "?"
        for lo, hi, pheno, short in PHENOTYPE_THRESHOLDS.get("VKORC1", []):
            if float(lo) <= activity_score <= float(hi):
                phenotype = pheno
                abbr = short
                break
        return {
            "gene": "VKORC1",
            "diplotype": raw,
            "activity_score": activity_score,
            "phenotype": phenotype,
            "abbreviation": abbr,
            "allele1": {"name": raw, **allele},
            "allele2": {"name": raw, **allele},
        }

    parts = [p.strip() for p in raw.split("/") if p.strip()]
    if len(parts) != 2:
        return {"error": f"Invalid diplotype format: {diplotype}. Expected format: *1/*4"}

    allele1_name, allele2_name = parts
    alleles = ALLELE_ACTIVITY.get(gene_norm, {})
    a1 = alleles.get(allele1_name)
    a2 = alleles.get(allele2_name)
    if a1 is None:
        return {"error": f"Unknown allele {allele1_name} for {gene_norm}"}
    if a2 is None:
        return {"error": f"Unknown allele {allele2_name} for {gene_norm}"}

    activity_score = float(a1["activity"]) + float(a2["activity"])
    thresholds = PHENOTYPE_THRESHOLDS.get(gene_norm, [])
    phenotype = "unknown"
    abbreviation = "?"
    for lo, hi, pheno, abbr in thresholds:
        if float(lo) <= activity_score <= float(hi):
            phenotype = pheno
            abbreviation = abbr
            break
    if phenotype == "unknown" and thresholds:
        for _lo, hi, pheno, abbr in thresholds:
            if activity_score <= float(hi) + 0.5:
                phenotype = pheno
                abbreviation = abbr
                break

    return {
        "gene": gene_norm,
        "diplotype": diplotype,
        "activity_score": activity_score,
        "phenotype": phenotype,
        "abbreviation": abbreviation,
        "allele1": {"name": allele1_name, **a1},
        "allele2": {"name": allele2_name, **a2},
    }


def _cyp_from_genotype(genotype: Optional[Dict[str, str]], primary_cyp: str) -> Optional[str]:
    if not genotype:
        return None
    for k, v in genotype.items():
        if str(k).upper() == primary_cyp and str(v).strip():
            return str(v)
    return None


def _generate_summary(result: Dict[str, Any], _drug: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Drug: {result['drug']} ({result.get('drug_class', '')})")
    lines.append(f"Metabolized by: {result['primary_cyp']}")

    rec = result.get("recommendation", {})
    if rec.get("phenotype"):
        lines.append(
            f"Patient phenotype: {str(rec['phenotype']).replace('_', ' ').title()} ({rec.get('abbreviation', '')})"
        )
        lines.append(f"Activity score: {rec.get('activity_score', '?')}")

    pk = result.get("pharmacokinetics", {})
    if isinstance(pk, dict):
        if pk.get("auc_fold_change") is not None:
            lines.append(f"Expected AUC change: {pk['auc_fold_change']}x normal")
        if pk.get("half_life_hours") is not None:
            lines.append(f"Expected half-life: {pk['half_life_hours']}h")
        if pk.get("morphine_conversion_pct") is not None:
            lines.append(f"Morphine conversion: {pk['morphine_conversion_pct']}%")

    if result.get("combined_warfarin_dose_mg") is not None:
        lines.append(
            f"Recommended warfarin dose: {result['combined_warfarin_dose_mg']} mg/day (CYP2C9+VKORC1 adjusted)"
        )
    elif result.get("recommended_dose_mg") is not None:
        lines.append(f"Recommended dose: {result['recommended_dose_mg']} mg")
    elif result.get("dose_note"):
        lines.append(f"Dose: {result['dose_note']}")

    if rec.get("clinical_action"):
        lines.append(f"Recommendation: {rec['clinical_action']}")

    if result.get("fda_warning"):
        lines.append(f"WARNING: {result['fda_warning']}")
    if result.get("cpic_level") and result["cpic_level"] != "N/A":
        lines.append(f"Evidence: CPIC Level {result['cpic_level']}")

    return "\n".join(lines)


def patient_query(drug_name: str, genotype: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    drug = get_drug(drug_name)
    if drug is None:
        return {"error": f"Drug '{drug_name}' not found in database"}

    primary_cyp = str(drug.get("primary_cyp") or "")
    pgx = dict(drug.get("pharmacogenomics") or {})
    drug_key = str(drug_name).lower()

    guideline = CPIC_GUIDELINES.get((drug_key, primary_cyp), {})
    pk_data = QUANTITATIVE_PK.get(drug_key, {})

    result: Dict[str, Any] = {
        "drug": drug["name"],
        "primary_cyp": primary_cyp,
        "drug_class": drug.get("drug_class", ""),
        "cpic_level": guideline.get("cpic_level", "N/A"),
        "reference": guideline.get("reference", pgx.get("reference", "")),
        "standard_dose_mg": pk_data.get("standard_dose_mg"),
    }

    if guideline.get("fda_warning"):
        result["fda_warning"] = guideline["fda_warning"]

    diplotype = _cyp_from_genotype(genotype, primary_cyp)
    if diplotype:
        pheno_result = determine_phenotype(primary_cyp, diplotype)
        result["genotype_result"] = pheno_result

        if "error" not in pheno_result:
            abbr = str(pheno_result["abbreviation"])
            rec = (guideline.get("recommendations") or {}).get(abbr, {})
            pk_phenotype = (
                ((pk_data.get("pk_by_phenotype") or {}).get(primary_cyp) or {}).get(abbr)
                or {}
            )

            result["recommendation"] = {
                "phenotype": pheno_result["phenotype"],
                "abbreviation": abbr,
                "activity_score": pheno_result["activity_score"],
                "dose_adjustment": rec.get("dose_adjustment", "consult_specialist"),
                "clinical_action": rec.get(
                    "action",
                    "No specific CPIC recommendation available. Consult clinical pharmacology specialist.",
                ),
            }

            if pk_phenotype.get("dose_mg") is not None:
                result["recommended_dose_mg"] = pk_phenotype["dose_mg"]
            elif pk_phenotype.get("note"):
                result["dose_note"] = pk_phenotype["note"]

            pk_out: Dict[str, Any] = {}
            if pk_phenotype.get("auc_fold") is not None:
                pk_out["auc_fold_change"] = pk_phenotype["auc_fold"]
            if pk_phenotype.get("half_life_h") is not None:
                pk_out["half_life_hours"] = pk_phenotype["half_life_h"]
            if pk_phenotype.get("clearance_change") is not None:
                pk_out["clearance_change"] = pk_phenotype["clearance_change"]
            if pk_phenotype.get("morphine_conversion_pct") is not None:
                pk_out["morphine_conversion_pct"] = pk_phenotype["morphine_conversion_pct"]
            if pk_phenotype.get("platelet_inhibition_pct") is not None:
                pk_out["platelet_inhibition_pct"] = pk_phenotype["platelet_inhibition_pct"]
            if pk_phenotype.get("active_metabolite_auc_fold") is not None:
                pk_out["active_metabolite_auc_fold_change"] = pk_phenotype[
                    "active_metabolite_auc_fold"
                ]
            if pk_out:
                result["pharmacokinetics"] = pk_out

            if abbr == "UM" and drug_key == "codeine":
                result["risk_level"] = "CRITICAL"
            elif abbr == "PM" and drug_key in {"codeine", "clopidogrel", "warfarin"}:
                result["risk_level"] = "HIGH"
            elif abbr in {"PM", "IM"}:
                result["risk_level"] = "MODERATE"
            else:
                result["risk_level"] = "STANDARD"
    else:
        result["recommendation"] = {
            "note": "No genotype provided. Showing population-level risk.",
            "clinical_action": pgx.get(
                "clinical_action", "Standard dosing; consider PGx testing if available."
            ),
        }
        result["risk_level"] = "UNKNOWN_GENOTYPE"
        result["population_at_risk"] = pgx.get(
            "pm_frequency", "See PharmGKB for population-level data"
        )

    if drug_key == "warfarin" and genotype:
        vkorc1_value = None
        for g, v in genotype.items():
            if str(g).upper() == "VKORC1":
                vkorc1_value = str(v).strip().upper()
                break
        if vkorc1_value:
            vkorc1_result = determine_phenotype("VKORC1", vkorc1_value)
            if "error" not in vkorc1_result:
                v_abbr = str(vkorc1_result.get("abbreviation", "?"))
                vkorc1_pk = (
                    ((pk_data.get("pk_by_phenotype") or {}).get("VKORC1") or {}).get(v_abbr)
                    or {}
                )
                dose_adjustment = float(vkorc1_pk.get("dose_adjustment") or 0.0)
                result["vkorc1"] = {
                    "genotype": vkorc1_value,
                    "phenotype": vkorc1_result.get("phenotype"),
                    "abbreviation": v_abbr,
                    "sensitivity": vkorc1_pk.get("sensitivity")
                    or vkorc1_result.get("allele1", {}).get("note", ""),
                    "dose_adjustment_mg": dose_adjustment,
                }
                base_dose = result.get("recommended_dose_mg")
                if base_dose is None:
                    base_dose = float(pk_data.get("standard_dose_mg") or 5.0)
                combined_dose = max(0.5, float(base_dose) + dose_adjustment)
                result["combined_warfarin_dose_mg"] = round(combined_dose, 1)
                cyp_abbr = result.get("recommendation", {}).get("abbreviation", "?")
                result[
                    "combined_note"
                ] = f"CYP2C9 {cyp_abbr} + VKORC1 {v_abbr}: recommended starting dose {combined_dose:.1f} mg/day"

    result["summary"] = _generate_summary(result, drug)
    return result


def check_drug_interactions(drug_list: List[str]) -> Dict[str, Any]:
    drug_info: List[Dict[str, Any]] = []
    for name in drug_list:
        drug = get_drug(name)
        if drug:
            primary = str(drug.get("primary_cyp") or "")
            secondary = list(drug.get("secondary_cyp") or [])
            all_cyps = [primary] + secondary if primary else secondary
            drug_info.append(
                {
                    "name": drug["name"],
                    "key": str(name).lower(),
                    "primary_cyp": primary,
                    "secondary_cyp": secondary,
                    "all_cyps": all_cyps,
                }
            )
        else:
            drug_info.append(
                {
                    "name": name,
                    "key": str(name).lower(),
                    "primary_cyp": None,
                    "secondary_cyp": [],
                    "all_cyps": [],
                    "note": "Not in substrate database; checking inhibitor/inducer role.",
                }
            )

    interactions: List[Dict[str, Any]] = []

    for i in range(len(drug_info)):
        for j in range(i + 1, len(drug_info)):
            d1, d2 = drug_info[i], drug_info[j]
            shared = set(d1.get("all_cyps", [])) & set(d2.get("all_cyps", []))
            for cyp in shared:
                both_primary = cyp == d1.get("primary_cyp") and cyp == d2.get("primary_cyp")
                interactions.append(
                    {
                        "type": "substrate_competition",
                        "drug1": d1["name"],
                        "drug2": d2["name"],
                        "shared_cyp": cyp,
                        "severity": "HIGH" if both_primary else "MODERATE",
                        "description": (
                            f"{d1['name']} and {d2['name']} are both metabolized by {cyp}. "
                            + (
                                "Both use it as primary pathway."
                                if both_primary
                                else "Shared secondary pathway."
                            )
                        ),
                    }
                )

    for d in drug_info:
        key = str(d["key"]).replace(" ", "_")
        for cyp, levels in CYP_INHIBITORS.items():
            for strength, inhibitors in levels.items():
                if key not in inhibitors:
                    continue
                affected = [
                    other
                    for other in drug_info
                    if other["key"] != d["key"] and cyp in other.get("all_cyps", [])
                ]
                for other in affected:
                    interactions.append(
                        {
                            "type": "inhibition",
                            "inhibitor": d["name"],
                            "affected_drug": other["name"],
                            "cyp": cyp,
                            "inhibitor_strength": strength,
                            "severity": "HIGH" if strength == "strong" else "MODERATE",
                            "description": (
                                f"{d['name']} is a {strength} inhibitor of {cyp}, which metabolizes {other['name']}. "
                                f"Expected: increased {other['name']} exposure."
                            ),
                        }
                    )

    return {
        "drugs_analyzed": [d["name"] for d in drug_info],
        "n_drugs": len(drug_info),
        "n_interactions": len(interactions),
        "interactions": interactions,
        "risk_summary": (
            "HIGH"
            if any(i.get("severity") == "HIGH" for i in interactions)
            else "MODERATE" if interactions else "LOW"
        ),
    }


def population_risk_summary(drug_name: str, ethnicity: str = "caucasian") -> Dict[str, Any]:
    drug = get_drug(drug_name)
    if drug is None:
        return {"error": f"Drug '{drug_name}' not found"}

    gene = str(drug.get("primary_cyp") or "")
    eth = str(ethnicity or "").lower()
    freqs = (ALLELE_FREQUENCIES.get(gene) or {}).get(eth)
    if freqs is None:
        return {
            "drug": drug["name"],
            "gene": gene,
            "ethnicity": ethnicity,
            "error": f"No allele frequency data for {gene} in {ethnicity}",
        }

    alleles = ALLELE_ACTIVITY.get(gene, {})
    sum_no_function = sum(
        freqs.get(a, 0.0) for a, info in alleles.items() if info.get("function") == "no_function"
    )
    sum_decreased = sum(
        freqs.get(a, 0.0) for a, info in alleles.items() if info.get("function") == "decreased"
    )
    sum_normal = sum(
        freqs.get(a, 0.0) for a, info in alleles.items() if info.get("function") == "normal"
    )
    sum_increased = sum(
        freqs.get(a, 0.0) for a, info in alleles.items() if info.get("function") == "increased"
    )

    p_pm = sum_no_function**2
    p_im = 2 * sum_no_function * (sum_normal + sum_decreased) + sum_decreased**2
    p_um = sum_increased**2 + 2 * sum_increased * sum_normal
    p_nm = max(0.0, 1.0 - p_pm - p_im - p_um)

    phenotype_frequencies: Dict[str, float] = {
        "normal_metabolizer": round(p_nm * 100.0, 1),
        "intermediate_metabolizer": round(p_im * 100.0, 1),
        "poor_metabolizer": round(p_pm * 100.0, 1),
    }
    if sum_increased > 0:
        phenotype_frequencies["ultrarapid_metabolizer"] = round(p_um * 100.0, 1)

    guideline = CPIC_GUIDELINES.get((str(drug_name).lower(), gene), {})
    recs = guideline.get("recommendations", {})
    abbr_for = {
        "normal_metabolizer": "NM",
        "intermediate_metabolizer": "IM",
        "poor_metabolizer": "PM",
        "ultrarapid_metabolizer": "UM",
    }

    details: List[Dict[str, Any]] = []
    for pheno, pct in phenotype_frequencies.items():
        abbr = abbr_for.get(pheno, "?")
        rec = recs.get(abbr, {})
        details.append(
            {
                "phenotype": pheno.replace("_", " ").title(),
                "abbreviation": abbr,
                "frequency_pct": pct,
                "action": rec.get("action", "Standard dosing"),
                "dose_adjustment": rec.get("dose_adjustment", "standard"),
            }
        )

    at_risk_pct = p_pm * 100.0 + (p_um * 100.0 if sum_increased > 0 else 0.0)
    return {
        "drug": drug["name"],
        "gene": gene,
        "ethnicity": ethnicity,
        "phenotype_distribution": details,
        "at_risk_pct": round(at_risk_pct, 1),
        "total_requiring_action_pct": round((p_pm + p_im + p_um) * 100.0, 1),
        "cpic_level": guideline.get("cpic_level", "N/A"),
        "fda_warning": guideline.get("fda_warning"),
    }


def _find_drug_key_by_smiles(smiles: str) -> Optional[str]:
    cleaned = str(smiles or "").strip()
    if not cleaned:
        return None

    for key, info in DRUG_DATABASE.items():
        if str(info.get("smiles") or "") == cleaned:
            return key

    try:
        from rdkit import Chem
    except Exception:
        return None

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    for key, info in DRUG_DATABASE.items():
        ref = Chem.MolFromSmiles(str(info.get("smiles") or ""))
        if ref is not None and Chem.MolToSmiles(ref) == canon:
            return key
    return None


def _extract_cyp_label(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("predicted_cyp", "primary_cyp", "cyp"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def end_to_end_query(
    smiles: str,
    genotype: Optional[Dict[str, str]] = None,
    cyp_predictor: Optional[Callable[[str], Dict[str, Any]]] = None,
    site_predictor: Optional[Callable[[str], Any]] = None,
    *,
    topk: int = 5,
    use_xtb: bool = False,
    isoform_hint: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"smiles": smiles}

    combined_payload: Optional[Dict[str, Any]] = None
    if cyp_predictor is None and site_predictor is None:
        try:
            from enzyme_software.moduleB.metabolism_site_predictor import (
                predict_drug_metabolism,
            )

            combined_payload = predict_drug_metabolism(
                smiles=smiles,
                topk=topk,
                use_xtb=use_xtb,
                isoform_hint=isoform_hint,
            )
            result["module_b_prediction"] = combined_payload
            result["cyp_prediction"] = combined_payload.get("cyp_prediction", {})
            result["site_prediction"] = combined_payload.get("site_prediction", {})
        except Exception as exc:
            result["module_b_prediction_error"] = str(exc)

    if cyp_predictor is not None:
        try:
            result["cyp_prediction"] = cyp_predictor(smiles)
        except Exception as exc:
            result["cyp_prediction"] = {"error": str(exc)}

    if site_predictor is not None:
        try:
            site_payload = site_predictor(smiles)
            if isinstance(site_payload, list):
                site_payload = {"ranked_sites": site_payload}
            result["site_prediction"] = site_payload
        except Exception as exc:
            result["site_prediction"] = {"error": str(exc)}

    matched_drug = _find_drug_key_by_smiles(smiles)
    result["drug_match"] = matched_drug

    if matched_drug:
        result["pharmacogenomics"] = patient_query(matched_drug, genotype=genotype)
    else:
        cyp_label = _extract_cyp_label(result.get("cyp_prediction") or {})
        if not cyp_label and combined_payload:
            cyp_label = _extract_cyp_label(combined_payload)
        result["pharmacogenomics"] = {
            "note": "Drug not found in reference database. Returning CYP-guided generic context only.",
            "predicted_cyp": cyp_label,
        }

    return result


def generate_clinical_report(query_result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  PHARMACOGENOMICS CONSULTATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Drug: {query_result.get('drug', 'Unknown')}")
    lines.append(f"Drug Class: {query_result.get('drug_class', 'Unknown')}")
    lines.append(
        f"Primary Metabolizing Enzyme: {query_result.get('primary_cyp', 'Unknown')}"
    )
    lines.append(f"CPIC Evidence Level: {query_result.get('cpic_level', 'N/A')}")
    lines.append("")

    geno = query_result.get("genotype_result", {})
    if isinstance(geno, dict) and "error" not in geno and geno:
        lines.append("GENOTYPE RESULTS:")
        lines.append(f"  Gene: {geno.get('gene', '')}")
        lines.append(f"  Diplotype: {geno.get('diplotype', '')}")
        lines.append(f"  Activity Score: {geno.get('activity_score', '')}")
        lines.append(
            f"  Phenotype: {str(geno.get('phenotype', '')).replace('_', ' ').title()}"
        )
        lines.append("")

    pk = query_result.get("pharmacokinetics", {})
    if isinstance(pk, dict) and pk:
        lines.append("PHARMACOKINETIC IMPACT:")
        if pk.get("auc_fold_change") is not None:
            lines.append(
                f"  AUC Change: {pk['auc_fold_change']}x relative to normal metabolizer"
            )
        if pk.get("half_life_hours") is not None:
            lines.append(f"  Expected Half-life: {pk['half_life_hours']} hours")
        if pk.get("clearance_change"):
            lines.append(f"  Clearance: {pk['clearance_change']}")
        if pk.get("morphine_conversion_pct") is not None:
            lines.append(f"  Morphine Conversion: {pk['morphine_conversion_pct']}% of dose")
        if pk.get("platelet_inhibition_pct") is not None:
            lines.append(f"  Platelet Inhibition: {pk['platelet_inhibition_pct']}%")
        lines.append("")

    rec = query_result.get("recommendation", {})
    if isinstance(rec, dict) and rec.get("clinical_action"):
        lines.append("CLINICAL RECOMMENDATION:")
        lines.append(f"  {rec['clinical_action']}")
        if query_result.get("recommended_dose_mg") is not None:
            std = query_result.get("standard_dose_mg")
            adj = query_result["recommended_dose_mg"]
            lines.append(f"  Standard Dose: {std} mg -> Recommended Dose: {adj} mg")
        if query_result.get("combined_warfarin_dose_mg") is not None:
            lines.append(
                f"  Combined CYP2C9+VKORC1 Dose: {query_result['combined_warfarin_dose_mg']} mg/day"
            )
        lines.append("")

    vkorc1 = query_result.get("vkorc1")
    if isinstance(vkorc1, dict):
        lines.append("VKORC1 ASSESSMENT:")
        lines.append(f"  Genotype: {vkorc1.get('genotype', '')}")
        lines.append(
            f"  Sensitivity: {str(vkorc1.get('phenotype', '')).replace('_', ' ').title()}"
        )
        lines.append(
            f"  Dose Adjustment: {float(vkorc1.get('dose_adjustment_mg', 0.0)):+.1f} mg from CYP2C9-adjusted dose"
        )
        lines.append("")

    if query_result.get("fda_warning"):
        lines.append("*" * 50)
        lines.append(f"  FDA WARNING: {query_result['fda_warning']}")
        lines.append("*" * 50)
        lines.append("")

    lines.append(f"RISK LEVEL: {query_result.get('risk_level', 'UNKNOWN')}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("  This report is for educational/research use only.")
    lines.append("  Clinical decisions should involve qualified clinicians.")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_module_c_dashboard(
    queries: List[Dict[str, Any]],
    ddi_result: Optional[Dict[str, Any]] = None,
    population_results: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "module_c_dashboard.html",
) -> str:
    risk_color = {
        "CRITICAL": "#8e44ad",
        "HIGH": "#e74c3c",
        "MODERATE": "#f39c12",
        "STANDARD": "#27ae60",
        "UNKNOWN_GENOTYPE": "#7f8c8d",
    }

    query_cards = ""
    for q in queries:
        risk = str(q.get("risk_level") or "STANDARD")
        pk = q.get("pharmacokinetics") or {}
        pk_parts: List[str] = []
        if pk.get("auc_fold_change") is not None:
            pk_parts.append(f"AUC <strong>{pk['auc_fold_change']}x</strong>")
        if pk.get("half_life_hours") is not None:
            pk_parts.append(f"t1/2 <strong>{pk['half_life_hours']}h</strong>")
        if pk.get("morphine_conversion_pct") is not None:
            pk_parts.append(f"Morphine <strong>{pk['morphine_conversion_pct']}%</strong>")
        pk_html = "".join([f"<span class='pill'>{p}</span>" for p in pk_parts])

        dose_html = ""
        if q.get("combined_warfarin_dose_mg") is not None:
            dose_html = (
                "<div class='dose'>Adjusted dose: "
                f"<strong>{q['combined_warfarin_dose_mg']} mg/day</strong> (CYP2C9 + VKORC1)</div>"
            )
        elif q.get("recommended_dose_mg") is not None:
            dose_html = (
                "<div class='dose'>Dose: "
                f"{q.get('standard_dose_mg', '?')} mg -> <strong>{q['recommended_dose_mg']} mg</strong></div>"
            )

        warning_html = ""
        if q.get("fda_warning"):
            warning_html = f"<div class='warning'>{q['fda_warning']}</div>"

        rec = q.get("recommendation", {})
        pheno = str(rec.get("phenotype", "N/A")).replace("_", " ").title()
        abbr = rec.get("abbreviation", "?")

        query_cards += f"""
        <div class="card">
            <div class="card-header">
                <div class="title">{q.get('drug', '?')}</div>
                <div class="badge" style="background:{risk_color.get(risk, '#95a5a6')}">{risk}</div>
            </div>
            <div class="meta">{q.get('drug_class', '')} | {q.get('primary_cyp', '')}</div>
            <div class="body">
                <div>Phenotype: <strong>{pheno}</strong> ({abbr})</div>
                <div class="pills">{pk_html}</div>
                {dose_html}
                <div class="action">{rec.get('clinical_action', '')}</div>
                {warning_html}
            </div>
            <div class="foot">CPIC Level {q.get('cpic_level', 'N/A')}</div>
        </div>
        """

    ddi_html = ""
    if ddi_result:
        ddi_items = ""
        for inter in ddi_result.get("interactions", []):
            sev = str(inter.get("severity") or "MODERATE")
            sev_color = "#e74c3c" if sev == "HIGH" else "#f39c12"
            ddi_items += (
                "<div class='ddi-item'>"
                f"<span class='ddi-sev' style='background:{sev_color}'>{sev}</span>"
                f"{inter.get('description', '')}</div>"
            )
        ddi_html = f"""
        <section class="panel">
            <h2>Drug-Drug Interactions ({ddi_result.get('n_interactions', 0)} found)</h2>
            <div class="small">Drugs: {', '.join(ddi_result.get('drugs_analyzed', []))}</div>
            {ddi_items or '<div class="small">No significant CYP-mediated interactions detected.</div>'}
        </section>
        """

    population_html = ""
    if population_results:
        cards = ""
        for pop in population_results:
            rows = ""
            for row in pop.get("phenotype_distribution", []):
                pct = float(row.get("frequency_pct") or 0.0)
                abbr = str(row.get("abbreviation") or "?")
                color = {
                    "NM": "#27ae60",
                    "IM": "#f39c12",
                    "PM": "#e74c3c",
                    "UM": "#8e44ad",
                }.get(abbr, "#95a5a6")
                rows += (
                    "<div class='bar-row'>"
                    f"<span class='bar-label'>{abbr}</span>"
                    f"<div class='bar-fill' style='width:{max(2.0, pct)}%;background:{color}'></div>"
                    f"<span class='bar-pct'>{pct:.1f}%</span>"
                    "</div>"
                )
            warn = (
                f"<div class='warning'>{pop['fda_warning']}</div>"
                if pop.get("fda_warning")
                else ""
            )
            cards += f"""
            <div class="pop-card">
                <h3>{pop.get('drug', '?')} / {pop.get('gene', '?')} ({str(pop.get('ethnicity', '')).title()})</h3>
                {rows}
                <div class="small">Patients at risk (PM + UM): <strong>{pop.get('at_risk_pct', '?')}%</strong></div>
                {warn}
            </div>
            """
        population_html = f"<section class='panel'><h2>Population Risk Profiles</h2>{cards}</section>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CYP-Predict Module C Dashboard</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f5f7;color:#213547}}
header{{background:linear-gradient(135deg,#124734,#2ecc71);color:#fff;padding:24px 28px}}
header h1{{margin:0;font-size:24px}}
header p{{margin:6px 0 0;font-size:13px;opacity:.9}}
main{{max-width:1120px;margin:0 auto;padding:18px}}
h2{{margin:0 0 10px;font-size:18px;color:#124734}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}
.card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;align-items:center;padding:14px 14px 4px}}
.title{{font-size:18px;font-weight:700;color:#124734}}
.badge{{color:#fff;border-radius:999px;padding:3px 10px;font-size:11px;font-weight:700}}
.meta{{padding:0 14px 8px;font-size:12px;color:#6b7280}}
.body{{padding:8px 14px 12px;font-size:13px;line-height:1.5}}
.pills{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}}
.pill{{background:#eaf2f8;border-radius:999px;padding:3px 10px;font-size:12px}}
.dose{{background:#d5f5e3;border-radius:6px;padding:6px 8px;margin:8px 0;font-size:12px}}
.action{{margin:6px 0}}
.warning{{background:#fdecea;border-left:3px solid #e74c3c;padding:7px 9px;margin:8px 0;font-size:12px;color:#922b21}}
.foot{{padding:8px 14px;background:#f8fafc;font-size:11px;color:#6b7280}}
.panel{{margin-top:16px;background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:16px}}
.small{{font-size:12px;color:#4b5563;margin-bottom:8px}}
.ddi-item{{padding:7px 0;border-bottom:1px solid #edf2f7;font-size:13px}}
.ddi-sev{{display:inline-block;color:#fff;border-radius:4px;padding:2px 8px;margin-right:8px;font-size:10px;font-weight:700}}
.pop-card{{background:#f8fafc;border-radius:8px;padding:12px;margin:10px 0}}
.pop-card h3{{margin:0 0 8px;font-size:14px;color:#124734}}
.bar-row{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.bar-label{{width:28px;font-size:12px;font-weight:700;color:#4b5563}}
.bar-fill{{height:18px;border-radius:4px;min-width:3px}}
.bar-pct{{font-size:12px;color:#4b5563}}
footer{{text-align:center;color:#94a3b8;font-size:11px;padding:18px}}
</style>
</head>
<body>
<header>
  <h1>CYP-Predict: Pharmacogenomics Dashboard</h1>
  <p>Module C - clinical decision-support outputs from genotype and metabolism context.</p>
</header>
<main>
  <section>
    <h2>Patient Pharmacogenomics Queries</h2>
    <div class="grid">{query_cards}</div>
  </section>
  {ddi_html}
  {population_html}
</main>
<footer>For educational and research use only. Clinical decisions require licensed healthcare professionals.</footer>
</body>
</html>
"""

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return output_path


def print_patient_query(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("  PHARMACOGENOMICS CLINICAL SUMMARY")
    print("=" * 65)
    print(result.get("summary", "No summary available"))
    risk = str(result.get("risk_level", ""))
    if risk == "CRITICAL":
        print("\n  RISK LEVEL: CRITICAL")
    elif risk == "HIGH":
        print("\n  Risk level: HIGH")
    elif risk == "MODERATE":
        print("\n  Risk level: MODERATE")
    print("=" * 65)


def print_ddi_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("  DRUG-DRUG INTERACTION CHECK")
    print("=" * 65)
    print(f"  Drugs: {', '.join(result.get('drugs_analyzed', []))}")
    print(f"  Interactions found: {result.get('n_interactions', 0)}")
    print(f"  Overall risk: {result.get('risk_summary', 'UNKNOWN')}")
    interactions = list(result.get("interactions", []))
    if interactions:
        for i, inter in enumerate(interactions, 1):
            print(f"\n  Interaction {i}: [{inter.get('severity', '?')}]")
            print(f"    {inter.get('description', '')}")
    else:
        print("\n  No significant CYP-mediated interactions detected.")
    print("=" * 65)


def print_population_summary(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print(f"  POPULATION RISK: {result.get('drug', '?')} / {result.get('gene', '?')}")
    print(f"  Ethnicity: {str(result.get('ethnicity', '?')).title()}")
    print("=" * 65)
    for pd in result.get("phenotype_distribution", []):
        bar_len = int(float(pd.get("frequency_pct", 0.0)) / 2.0)
        bar = "#" * max(0, bar_len)
        print(
            f"  {pd.get('abbreviation', '?'):>3} {pd.get('phenotype', ''):<30} {pd.get('frequency_pct', 0):>5.1f}% {bar}"
        )
        if pd.get("dose_adjustment") != "standard":
            print(f"      Action: {pd.get('action', '')}")
    print(f"\n  Patients requiring action: {result.get('total_requiring_action_pct', '?')}%")
    if result.get("fda_warning"):
        print(f"\n  FDA WARNING: {result['fda_warning']}")
    print("=" * 65)


__all__ = [
    "ALLELE_ACTIVITY",
    "PHENOTYPE_THRESHOLDS",
    "ALLELE_FREQUENCIES",
    "QUANTITATIVE_PK",
    "CPIC_GUIDELINES",
    "CYP_INHIBITORS",
    "CYP_INDUCERS",
    "determine_phenotype",
    "patient_query",
    "check_drug_interactions",
    "population_risk_summary",
    "end_to_end_query",
    "generate_clinical_report",
    "generate_module_c_dashboard",
    "print_patient_query",
    "print_ddi_report",
    "print_population_summary",
]


if __name__ == "__main__":
    example_queries = [
        patient_query("warfarin", genotype={"CYP2C9": "*1/*3", "VKORC1": "GA"}),
        patient_query("codeine", genotype={"CYP2D6": "*1/*1xN"}),
        patient_query("codeine", genotype={"CYP2D6": "*4/*4"}),
        patient_query("clopidogrel", genotype={"CYP2C19": "*2/*2"}),
    ]
    for q in example_queries:
        print(generate_clinical_report(q))
        print()

    ddi = check_drug_interactions(["warfarin", "ibuprofen", "omeprazole"])
    pops = [
        population_risk_summary("codeine", "caucasian"),
        population_risk_summary("clopidogrel", "east_asian"),
        population_risk_summary("warfarin", "caucasian"),
    ]
    path = generate_module_c_dashboard(
        queries=example_queries,
        ddi_result=ddi,
        population_results=pops,
        output_path=os.path.join(os.path.dirname(__file__), "module_c_dashboard.html"),
    )
    print(f"Dashboard generated: {path}")
