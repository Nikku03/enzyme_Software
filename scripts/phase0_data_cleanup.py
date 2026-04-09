#!/usr/bin/env python3
"""
Phase 0: CYP3A4 Dataset Cleanup

This script addresses label quality issues identified through literature verification:
1. Removes molecules where CYP3A4 is not the primary enzyme
2. Corrects wrong site annotations
3. Adds label confidence scores
4. Flags molecules with minor CYP3A4 pathways

Usage:
    python scripts/phase0_data_cleanup.py
"""

import json
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


# ============================================================================
# CORRECTION RULES (Based on Literature Verification)
# ============================================================================

# Molecules to REMOVE entirely (CYP3A4 is NOT the primary/significant enzyme)
REMOVE_WRONG_ENZYME = {
    "nicotine": {
        "reason": "CYP2A6 handles 70-80% via 5'-hydroxylation → cotinine. CYP3A4 not significant.",
        "primary_enzyme": "CYP2A6",
        "reference": "Messina et al. 1997; Nakajima et al. 1996"
    },
    "n_nitrosonornicotine": {
        "reason": "CYP2A6 primary for 5'-hydroxylation; CYP3A4 does 2'-OH at different site",
        "primary_enzyme": "CYP2A6",
        "reference": "Jalas et al. 2005"
    },
    "nnk": {
        "reason": "NNK α-hydroxylation primarily by CYP2A6/2A13, not CYP3A4",
        "primary_enzyme": "CYP2A6/CYP2A13",
        "reference": "Smith et al. 2003"
    },
}

# Molecules to CORRECT (wrong site annotated for CYP3A4)
# NOTE: These indices are for the JSON dataset (not CSV gold_hard_source which uses different indices)
CORRECT_SITE_ANNOTATION = {
    "zileuton": {
        "old_atom_idx": 2,  # Currently annotated in JSON: aromatic C (atom 2)
        "new_atom_idx": 10,  # Correct: Sulfur atom (sulfoxidation by CYP3A4)
        "reason": "CYP3A4 exclusively does sulfoxidation; ring hydroxylation is CYP1A2/CYP2C9",
        "reaction_type": "sulfoxidation",
        "reference": "Machinist et al. 1995"
    },
    "diclofenac": {
        "old_atom_idx": 1,  # Currently in JSON: carboxylic C (atom 1) - clearly wrong
        "new_atom_idx": 16,  # Correct: meta-C on dichlorophenyl (position 5) for CYP3A4
        "reason": "CYP2C9 does 4'-hydroxylation (>99%); CYP3A4 does 5-hydroxylation (minor)",
        "reaction_type": "5-hydroxylation",
        "reference": "Leemann et al. 1993"
    },
}

# Molecules to FLAG as low confidence (CYP3A4 is minor pathway)
FLAG_MINOR_PATHWAY = {
    "mianserin": {
        "confidence": "low",
        "reason": "CYP2D6/CYP1A2 are dominant; CYP3A4 minor contributor",
        "primary_enzymes": ["CYP2D6", "CYP1A2"],
        "reference": "Dahl et al. 1994"
    },
    "phenprocoumon": {
        "confidence": "low",
        "reason": "CYP2C9 dominant (S-phenprocoumon); CYP3A4 minor",
        "primary_enzymes": ["CYP2C9"],
        "reference": "Ufer et al. 2004"
    },
    "hydromorphone": {
        "confidence": "low",
        "reason": "UGT-mediated glucuronidation dominant; CYP role minimal",
        "primary_enzymes": ["UGT2B7"],
        "reference": "Coffman et al. 1998"
    },
}

# Molecules with VERIFIED correct CYP3A4 labels
VERIFIED_CORRECT = {
    "quinine": {
        "confidence": "high",
        "reason": "3-hydroxylation is definitive CYP3A4 probe reaction",
        "reference": "Mirghani et al. 1999"
    },
    "hydrocodone": {
        "confidence": "high",
        "reason": "N-demethylation to norhydrocodone confirmed CYP3A4",
        "reference": "Hutchinson et al. 2004"
    },
}

# Molecules with INSUFFICIENT evidence (keep but flag)
INSUFFICIENT_EVIDENCE = {
    "rifalazil": {"confidence": "medium", "reason": "Limited literature on CYP3A4 specificity"},
    "ezlopitant": {"confidence": "medium", "reason": "NK1 antagonist, limited metabolism data"},
    "tamarixetin": {"confidence": "medium", "reason": "Flavonoid, complex metabolism"},
    "5betacholestane-3_7_12_trihydroxy": {"confidence": "medium", "reason": "Bile acid precursor, complex"},
    "voriconazole": {"confidence": "medium", "reason": "CYP2C19/CYP3A4 both contribute"},
    "mesoridazine": {"confidence": "medium", "reason": "Limited specific CYP3A4 data"},
    "reduced_diclofenac": {"confidence": "medium", "reason": "Derivative, limited data"},
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize drug name for matching."""
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def find_drug_by_name(drugs: List[Dict], name: str) -> Tuple[Optional[int], Optional[Dict]]:
    """Find drug in dataset by normalized name."""
    norm_name = normalize_name(name)
    for i, drug in enumerate(drugs):
        drug_name = normalize_name(drug.get("molecule_name", drug.get("name", drug.get("id", ""))))
        if drug_name == norm_name or normalize_name(drug.get("id", "")) == norm_name:
            return i, drug
    return None, None


def add_confidence_field(drug: Dict, confidence: str, reason: str, 
                         action: str = "none", reference: str = "") -> Dict:
    """Add label confidence metadata to drug entry."""
    drug = copy.deepcopy(drug)
    drug["label_confidence"] = confidence
    drug["confidence_reason"] = reason
    drug["cleanup_action"] = action
    if reference:
        drug["confidence_reference"] = reference
    drug["cleanup_timestamp"] = datetime.now().isoformat()
    return drug


# ============================================================================
# MAIN CLEANUP FUNCTION
# ============================================================================

def cleanup_cyp3a4_dataset(input_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Clean the CYP3A4 dataset:
    1. Remove wrong-enzyme molecules
    2. Correct wrong site annotations
    3. Add confidence scores to all molecules
    
    Returns cleanup report.
    """
    # Load dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    drugs = data.get("drugs", [])
    original_count = len(drugs)
    
    # Initialize report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_path,
        "original_drug_count": original_count,
        "actions": {
            "removed": [],
            "site_corrected": [],
            "flagged_minor_pathway": [],
            "flagged_insufficient_evidence": [],
            "verified_correct": [],
            "default_confidence": [],
        },
        "summary": {},
    }
    
    cleaned_drugs = []
    
    for drug in drugs:
        drug_name = drug.get("molecule_name", drug.get("name", drug.get("id", "")))
        norm_name = normalize_name(drug_name)
        
        # Check if should be REMOVED
        if norm_name in REMOVE_WRONG_ENZYME:
            info = REMOVE_WRONG_ENZYME[norm_name]
            report["actions"]["removed"].append({
                "name": drug_name,
                "smiles": drug.get("canonical_smiles", drug.get("smiles", "")),
                "reason": info["reason"],
                "primary_enzyme": info["primary_enzyme"],
            })
            continue  # Skip this drug
        
        # Check if site needs CORRECTION
        if norm_name in CORRECT_SITE_ANNOTATION:
            info = CORRECT_SITE_ANNOTATION[norm_name]
            old_sites = drug.get("site_atoms", drug.get("metabolism_sites", []))
            
            # Replace old index with new
            new_sites = []
            for site in old_sites:
                if site == info["old_atom_idx"]:
                    new_sites.append(info["new_atom_idx"])
                else:
                    new_sites.append(site)
            
            drug = copy.deepcopy(drug)
            drug["site_atoms"] = new_sites
            drug["metabolism_sites"] = new_sites
            
            # Update som field if present
            if "som" in drug:
                new_som = []
                for s in drug["som"]:
                    if s.get("atom_idx") == info["old_atom_idx"]:
                        new_som.append({"atom_idx": info["new_atom_idx"]})
                    else:
                        new_som.append(s)
                drug["som"] = new_som
            
            drug = add_confidence_field(
                drug, 
                confidence="high",
                reason=f"Site corrected: {info['reason']}",
                action="site_corrected",
                reference=info.get("reference", "")
            )
            drug["original_site_atoms"] = old_sites
            drug["correction_details"] = info
            
            report["actions"]["site_corrected"].append({
                "name": drug_name,
                "old_site": info["old_atom_idx"],
                "new_site": info["new_atom_idx"],
                "reason": info["reason"],
            })
            
            cleaned_drugs.append(drug)
            continue
        
        # Check if should be FLAGGED as minor pathway
        if norm_name in FLAG_MINOR_PATHWAY:
            info = FLAG_MINOR_PATHWAY[norm_name]
            drug = add_confidence_field(
                drug,
                confidence=info["confidence"],
                reason=f"Minor pathway: {info['reason']}",
                action="flagged_minor_pathway",
                reference=info.get("reference", "")
            )
            drug["primary_enzymes_literature"] = info.get("primary_enzymes", [])
            
            report["actions"]["flagged_minor_pathway"].append({
                "name": drug_name,
                "confidence": info["confidence"],
                "reason": info["reason"],
            })
            
            cleaned_drugs.append(drug)
            continue
        
        # Check if VERIFIED correct
        if norm_name in VERIFIED_CORRECT:
            info = VERIFIED_CORRECT[norm_name]
            drug = add_confidence_field(
                drug,
                confidence=info["confidence"],
                reason=f"Verified: {info['reason']}",
                action="verified_correct",
                reference=info.get("reference", "")
            )
            
            report["actions"]["verified_correct"].append({
                "name": drug_name,
                "confidence": info["confidence"],
            })
            
            cleaned_drugs.append(drug)
            continue
        
        # Check if INSUFFICIENT evidence
        if norm_name in INSUFFICIENT_EVIDENCE:
            info = INSUFFICIENT_EVIDENCE[norm_name]
            drug = add_confidence_field(
                drug,
                confidence=info["confidence"],
                reason=info["reason"],
                action="flagged_insufficient_evidence",
            )
            
            report["actions"]["flagged_insufficient_evidence"].append({
                "name": drug_name,
                "confidence": info["confidence"],
                "reason": info["reason"],
            })
            
            cleaned_drugs.append(drug)
            continue
        
        # DEFAULT: Assign confidence based on source
        source = drug.get("source", drug.get("molecule_source", "unknown"))
        
        source_confidence_map = {
            "drugbank": "high",
            "metxbiodb": "high",
            "MetXBioDB": "high",
            "peng_external": "high",
            "Peng_external": "high",
            "rudik_external": "high",
            "Rudik_external": "high",
            "attnsom": "medium",
            "ATTNSOM": "medium",
            "cyp_dbs_external": "low",
            "CYP_DBs_external": "low",
        }
        
        confidence = source_confidence_map.get(source, "medium")
        drug = add_confidence_field(
            drug,
            confidence=confidence,
            reason=f"Default confidence from source: {source}",
            action="default_confidence",
        )
        
        report["actions"]["default_confidence"].append({
            "name": drug_name,
            "source": source,
            "confidence": confidence,
        })
        
        cleaned_drugs.append(drug)
    
    # Build summary
    report["summary"] = {
        "original_count": original_count,
        "final_count": len(cleaned_drugs),
        "removed_count": len(report["actions"]["removed"]),
        "site_corrected_count": len(report["actions"]["site_corrected"]),
        "flagged_minor_count": len(report["actions"]["flagged_minor_pathway"]),
        "flagged_insufficient_count": len(report["actions"]["flagged_insufficient_evidence"]),
        "verified_correct_count": len(report["actions"]["verified_correct"]),
        "default_confidence_count": len(report["actions"]["default_confidence"]),
        "confidence_breakdown": {
            "high": sum(1 for d in cleaned_drugs if d.get("label_confidence") == "high"),
            "medium": sum(1 for d in cleaned_drugs if d.get("label_confidence") == "medium"),
            "low": sum(1 for d in cleaned_drugs if d.get("label_confidence") == "low"),
        }
    }
    
    # Build output dataset
    output_data = copy.deepcopy(data)
    output_data["drugs"] = cleaned_drugs
    output_data["n_drugs"] = len(cleaned_drugs)
    output_data["cleanup_metadata"] = {
        "cleanup_version": "phase0_v1",
        "cleanup_timestamp": report["timestamp"],
        "removed_molecules": [r["name"] for r in report["actions"]["removed"]],
        "site_corrections": {r["name"]: {"old": r["old_site"], "new": r["new_site"]} 
                           for r in report["actions"]["site_corrected"]},
    }
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned dataset
    output_path = output_dir / "cyp3a4_cleaned_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save report
    report_path = output_dir / "cleanup_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save changelog
    changelog_path = output_dir / "CHANGELOG.md"
    with open(changelog_path, 'w') as f:
        f.write(generate_changelog(report))
    
    return report


def generate_changelog(report: Dict) -> str:
    """Generate human-readable changelog."""
    lines = [
        "# CYP3A4 Dataset Cleanup Changelog",
        "",
        f"**Date:** {report['timestamp'][:10]}",
        f"**Version:** Phase 0 v1",
        "",
        "## Summary",
        "",
        f"- Original molecules: {report['summary']['original_count']}",
        f"- Final molecules: {report['summary']['final_count']}",
        f"- Removed: {report['summary']['removed_count']}",
        f"- Site corrections: {report['summary']['site_corrected_count']}",
        "",
        "## Confidence Distribution",
        "",
        f"- High confidence: {report['summary']['confidence_breakdown']['high']}",
        f"- Medium confidence: {report['summary']['confidence_breakdown']['medium']}",
        f"- Low confidence: {report['summary']['confidence_breakdown']['low']}",
        "",
        "---",
        "",
        "## Removed Molecules (Wrong Enzyme)",
        "",
    ]
    
    for r in report["actions"]["removed"]:
        lines.append(f"### {r['name']}")
        lines.append(f"- **Reason:** {r['reason']}")
        lines.append(f"- **Primary enzyme:** {r['primary_enzyme']}")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Site Corrections",
        "",
    ])
    
    for r in report["actions"]["site_corrected"]:
        lines.append(f"### {r['name']}")
        lines.append(f"- **Old site:** atom {r['old_site']}")
        lines.append(f"- **New site:** atom {r['new_site']}")
        lines.append(f"- **Reason:** {r['reason']}")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Flagged: Minor CYP3A4 Pathway",
        "",
    ])
    
    for r in report["actions"]["flagged_minor_pathway"]:
        lines.append(f"- **{r['name']}:** {r['reason']}")
    
    lines.extend([
        "",
        "---",
        "",
        "## Flagged: Insufficient Evidence",
        "",
    ])
    
    for r in report["actions"]["flagged_insufficient_evidence"]:
        lines.append(f"- **{r['name']}:** {r['reason']}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Paths
    repo_root = Path(__file__).parent.parent
    input_path = repo_root / "data" / "prepared_training" / "cyp3a4_merged_dataset_local" / "cyp3a4_merged_dataset.json"
    output_dir = repo_root / "data" / "cleaned"
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print()
    
    report = cleanup_cyp3a4_dataset(str(input_path), str(output_dir))
    
    print("=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print()
    print(f"Original molecules: {report['summary']['original_count']}")
    print(f"Final molecules:    {report['summary']['final_count']}")
    print()
    print(f"Removed (wrong enzyme):     {report['summary']['removed_count']}")
    print(f"Site corrections:           {report['summary']['site_corrected_count']}")
    print(f"Flagged (minor pathway):    {report['summary']['flagged_minor_count']}")
    print(f"Flagged (insufficient):     {report['summary']['flagged_insufficient_count']}")
    print(f"Verified correct:           {report['summary']['verified_correct_count']}")
    print()
    print("Confidence breakdown:")
    print(f"  High:   {report['summary']['confidence_breakdown']['high']}")
    print(f"  Medium: {report['summary']['confidence_breakdown']['medium']}")
    print(f"  Low:    {report['summary']['confidence_breakdown']['low']}")
    print()
    print(f"Outputs saved to: {output_dir}")
