from __future__ import annotations

from collections import OrderedDict

FUNCTIONAL_GROUP_SMARTS = OrderedDict(
    [
        ("amide", "[NX3][CX3](=[OX1])"),
        ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
        ("ester", "[CX3](=O)[OX2][#6]"),
        ("sulfonamide", "[SX4](=[OX1])(=[OX1])([NX3])"),
        ("phenol", "[OX2H1]c"),
        ("alcohol", "[OX2H1][CX4]"),
        ("ketone", "[CX3](=O)([#6])[#6]"),
        ("aldehyde", "[CX3H1](=O)"),
        ("ether", "[OX2]([#6])[#6]"),
        ("primary_amine", "[NX3H2;!$(N=*);!$(N#*)]"),
        ("secondary_amine", "[NX3H1;!$(N=*);!$(N#*);!$(NC=O)]"),
        ("tertiary_amine", "[NX3H0;!$(N=*);!$(N#*);!$(NC=O)]"),
        ("sulfide", "[SX2]([#6])[#6]"),
        ("heteroaromatic", "[nR,oR,sR]"),
        ("aromatic_ring", "[a]"),
        ("aliphatic_chain", "[CX4H2,CX4H3;!R]"),
    ]
)
