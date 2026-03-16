from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


SUPPORTED_CYP_ISOFORMS = {
    "CYP1A2",
    "CYP2A6",
    "CYP2B6",
    "CYP2C8",
    "CYP2C9",
    "CYP2C19",
    "CYP2D6",
    "CYP2E1",
    "CYP3A4",
}

REACTION_KEYWORDS: Dict[str, List[str]] = {
    "hydroxylation": ["hydroxylat", "hydroxy"],
    "n_dealkylation": ["n-dealkylat", "n-demethylat", "deamination", "n-deethylat"],
    "o_dealkylation": ["o-dealkylat", "o-demethylat", "o-deethylat"],
    "s_oxidation": ["s-oxidat", "sulfoxid"],
    "epoxidation": ["epoxid"],
    "n_oxidation": ["n-oxid"],
    "glucuronidation": ["glucuronid"],
    "reduction": ["reduc"],
}


def extract_cyp_enzymes(text: str) -> List[str]:
    """Extract normalized CYP isoforms from free-text metabolism annotations."""
    if not text:
        return []
    normalized = re.sub(r"[\-/]", "", text.upper())
    matches = re.findall(r"CYP\s*(\d[A-Z]\d+)", normalized, re.IGNORECASE)
    enzymes: List[str] = []
    for match in matches:
        cyp = f"CYP{match.upper()}"
        if cyp in SUPPORTED_CYP_ISOFORMS and cyp not in enzymes:
            enzymes.append(cyp)
    return enzymes


def extract_reaction_types(text: str) -> List[str]:
    """Extract coarse reaction-type labels from DrugBank metabolism text."""
    if not text:
        return []
    text_lower = text.lower()
    reactions: List[str] = []
    for reaction_type, keywords in REACTION_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            reactions.append(reaction_type)
    return reactions


def parse_drugbank_xml(xml_path: str | Path) -> List[Dict[str, object]]:
    """
    Extract drugs with CYP metabolism annotations from DrugBank XML using streaming iterparse.

    This avoids building the entire ~GB-scale XML tree in memory.
    """
    xml_path = Path(xml_path)
    ns_uri = "http://www.drugbank.ca"
    ns_prefix = f"{{{ns_uri}}}"
    drugs: List[Dict[str, object]] = []

    context = ET.iterparse(str(xml_path), events=("end",))
    for _event, elem in context:
        if elem.tag != f"{ns_prefix}drug":
            continue

        drugbank_id = None
        name = None
        smiles = None
        metabolism_text = ""

        for child in elem:
            if child.tag == f"{ns_prefix}drugbank-id" and child.attrib.get("primary") == "true" and child.text:
                drugbank_id = child.text
            elif child.tag == f"{ns_prefix}name" and child.text:
                name = child.text
            elif child.tag == f"{ns_prefix}metabolism" and child.text:
                metabolism_text = child.text.strip()
            elif child.tag == f"{ns_prefix}calculated-properties" and smiles is None:
                for prop in child.findall(f"{ns_prefix}property"):
                    kind = prop.find(f"{ns_prefix}kind")
                    value = prop.find(f"{ns_prefix}value")
                    if kind is not None and value is not None and kind.text == "SMILES" and value.text:
                        smiles = value.text
                        break

        enzymes = extract_cyp_enzymes(metabolism_text)
        reactions = extract_reaction_types(metabolism_text)

        if smiles and enzymes:
            drugs.append(
                {
                    "drugbank_id": drugbank_id,
                    "name": name,
                    "smiles": smiles,
                    "metabolism_text": metabolism_text,
                    "enzymes": enzymes,
                    "reactions": reactions,
                }
            )

        elem.clear()

    return drugs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DrugBank XML into CYP metabolism records.")
    parser.add_argument("xml_path", help="Path to DrugBank XML")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    drugs = parse_drugbank_xml(args.xml_path)
    print(f"parsed={len(drugs)}")
    if args.output:
        Path(args.output).write_text(json.dumps(drugs, indent=2))
        print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
