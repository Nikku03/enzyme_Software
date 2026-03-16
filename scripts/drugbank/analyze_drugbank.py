from __future__ import annotations

import json
from collections import defaultdict


def analyze_drugbank(json_path: str = "data/cyp_metabolism_dataset.json"):
    with open(json_path) as f:
        data = json.load(f)
    drugs = data if isinstance(data, list) else data.get("drugs", data.get("compounds", []))
    print(f"Total drugs: {len(drugs)}")
    if not drugs:
        print("No drugs found!")
        return []

    print("\n" + "=" * 60)
    print("SAMPLE DRUG STRUCTURE")
    print("=" * 60)
    print(json.dumps(drugs[0], indent=2)[:2000])

    print("\n" + "=" * 60)
    print("FIELD ANALYSIS")
    print("=" * 60)
    field_counts = defaultdict(int)
    field_examples = {}
    for drug in drugs:
        for key, value in drug.items():
            field_counts[key] += 1
            if key not in field_examples and value:
                field_examples[key] = str(value)[:100]
    for field, count in sorted(field_counts.items()):
        pct = count / len(drugs) * 100
        print(f"{field}: {count} ({pct:.1f}%) - Example: {field_examples.get(field, 'N/A')}")

    print("\n" + "=" * 60)
    print("CYP DISTRIBUTION")
    print("=" * 60)
    cyp_counts = defaultdict(int)
    for drug in drugs:
        cyp = drug.get("primary_cyp") or drug.get("cyp") or drug.get("enzyme")
        if cyp:
            cyp_counts[cyp] += 1
    for cyp, count in sorted(cyp_counts.items(), key=lambda x: -x[1]):
        print(f"  {cyp}: {count}")

    print("\n" + "=" * 60)
    print("SoM FIELD CHECK")
    print("=" * 60)
    for field in ["site_atoms", "metabolism_sites", "sites", "som", "metabolism_description", "metabolism", "pathway"]:
        count = sum(1 for d in drugs if d.get(field))
        if count > 0:
            example = next((d[field] for d in drugs if d.get(field)), None)
            print(f"{field}: {count} drugs have this field")
            print(f"  Example: {str(example)[:200]}")
    return drugs


if __name__ == "__main__":
    analyze_drugbank()
