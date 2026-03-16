from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES


CYP_ORDER = list(ALL_CYP_CLASSES)


def count_by_cyp(drugs: List[Dict]) -> Dict[str, int]:
    counts = defaultdict(int)
    for drug in drugs:
        cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "")
        if cyp in CYP_ORDER:
            counts[cyp] += 1
    return dict(counts)


def oversample_rare_classes(
    drugs: List[Dict],
    target_per_class: int = 100,
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    by_cyp = defaultdict(list)
    for drug in drugs:
        cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "")
        if cyp in CYP_ORDER:
            by_cyp[cyp].append(drug)

    oversampled: List[Dict] = []
    for cyp in CYP_ORDER:
        cyp_drugs = list(by_cyp.get(cyp, []))
        if not cyp_drugs:
            continue
        oversampled.extend(cyp_drugs)
        if len(cyp_drugs) < target_per_class:
            n_needed = target_per_class - len(cyp_drugs)
            oversampled.extend(rng.choice(cyp_drugs) for _ in range(n_needed))

    rng.shuffle(oversampled)
    return oversampled
