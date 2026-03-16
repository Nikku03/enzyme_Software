"""Layer 2 protein structure database and scaffold helpers.

This module centralizes real PDB scaffold metadata per enzyme family so
Module 1/2 can consume non-synthetic topology priors and residue targets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


SCAFFOLD_DATABASE: Dict[str, Dict[str, Any]] = {
    "cytochrome_P450": {
        "structures": [
            {
                "pdb_id": "1FAG",
                "name": "P450-BM3 (CYP102A1)",
                "chain": "A",
                "priority": 1,
                "pocket_metrics": {
                    "active_site_volume_A3": 530,
                    "channel_1_diameter_A": 3.8,
                    "channel_2_diameter_A": 4.2,
                    "fe_to_channel_mouth_A": 12.5,
                },
                "active_site": {
                    "first_shell_residues": [
                        {"resid": 87, "resname": "PHE"},
                        {"resid": 82, "resname": "ALA"},
                        {"resid": 75, "resname": "LEU"},
                        {"resid": 78, "resname": "VAL"},
                    ],
                    "access_channel_residues": [
                        {"resid": 42, "resname": "PHE"},
                        {"resid": 47, "resname": "ARG"},
                        {"resid": 51, "resname": "TYR"},
                    ],
                },
            },
            {
                "pdb_id": "2CPP",
                "name": "P450cam (CYP101A1)",
                "chain": "A",
                "priority": 2,
                "pocket_metrics": {
                    "active_site_volume_A3": 390,
                    "channel_1_diameter_A": 3.2,
                    "fe_to_channel_mouth_A": 10.0,
                },
                "active_site": {
                    "first_shell_residues": [
                        {"resid": 295, "resname": "VAL"},
                        {"resid": 297, "resname": "ASP"},
                        {"resid": 96, "resname": "TYR"},
                    ]
                },
            },
            {
                "pdb_id": "1TQN",
                "name": "CYP3A4",
                "chain": "A",
                "priority": 3,
                "pocket_metrics": {
                    "active_site_volume_A3": 1385,
                    "channel_1_diameter_A": 6.0,
                },
            },
        ]
    },
    "non_heme_iron_oxygenase": {
        "structures": [
            {
                "pdb_id": "1OS7",
                "name": "TauD",
                "chain": "A",
                "priority": 1,
                "pocket_metrics": {"active_site_volume_A3": 280, "fe_to_surface_A": 8.0},
                "active_site": {
                    "first_shell_residues": [
                        {"resid": 99, "resname": "HIS"},
                        {"resid": 101, "resname": "ASP"},
                        {"resid": 255, "resname": "HIS"},
                        {"resid": 270, "resname": "ARG"},
                    ]
                },
            },
            {
                "pdb_id": "1NDO",
                "name": "Naphthalene dioxygenase",
                "chain": "A",
                "priority": 2,
                "pocket_metrics": {"active_site_volume_A3": 420},
            },
        ]
    },
    "serine_hydrolase": {
        "structures": [
            {
                "pdb_id": "1CEX",
                "name": "Cutinase",
                "chain": "A",
                "priority": 1,
                "pocket_metrics": {
                    "active_site_volume_A3": 180,
                    "oxyanion_hole_nh_distance_A": 2.8,
                    "channel_1_diameter_A": 3.1,
                },
                "active_site": {
                    "catalytic_residues": {
                        "Ser120": {"resid": 120, "resname": "SER"},
                        "His188": {"resid": 188, "resname": "HIS"},
                        "Asp175": {"resid": 175, "resname": "ASP"},
                    },
                    "oxyanion_hole": [
                        {"resid": 121, "resname": "GLN"},
                        {"resid": 42, "resname": "SER"},
                    ],
                    "first_shell_residues": [
                        {"resid": 81, "resname": "LEU"},
                        {"resid": 184, "resname": "VAL"},
                    ],
                },
            },
            {
                "pdb_id": "1SBN",
                "name": "Subtilisin BPN'",
                "chain": "A",
                "priority": 2,
                "pocket_metrics": {"active_site_volume_A3": 350, "channel_1_diameter_A": 3.4},
            },
            {
                "pdb_id": "1LPB",
                "name": "CalB",
                "chain": "A",
                "priority": 3,
                "pocket_metrics": {"active_site_volume_A3": 220, "channel_1_diameter_A": 3.2},
            },
        ]
    },
    "haloalkane_dehalogenase": {
        "structures": [
            {
                "pdb_id": "1CQW",
                "name": "DhaA",
                "chain": "A",
                "priority": 1,
                "pocket_metrics": {
                    "active_site_volume_A3": 150,
                    "tunnel_length_A": 15.0,
                    "tunnel_bottleneck_radius_A": 1.4,
                    "asp_to_tunnel_mouth_A": 12.0,
                    "channel_1_diameter_A": 2.8,
                },
                "active_site": {
                    "halide_stabilizing": [
                        {"resid": 109, "resname": "TRP"},
                        {"resid": 41, "resname": "ASN"},
                    ],
                    "tunnel_residues": [
                        {"resid": 177, "resname": "LEU"},
                        {"resid": 176, "resname": "CYS"},
                        {"resid": 245, "resname": "VAL"},
                    ],
                },
            },
            {
                "pdb_id": "1MJ5",
                "name": "LinB",
                "chain": "A",
                "priority": 2,
                "pocket_metrics": {
                    "active_site_volume_A3": 200,
                    "tunnel_length_A": 12.0,
                    "tunnel_bottleneck_radius_A": 1.8,
                    "channel_1_diameter_A": 3.2,
                },
            },
        ]
    },
}


VARIANT_RESIDUE_MAP: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "cytochrome_P450": {
        "1FAG": {
            "Substrate positioning tunnel": {
                "target_residues": [
                    {"resid": 87, "current": "PHE", "mutations": ["ALA", "GLY", "VAL", "LEU"]},
                    {"resid": 82, "current": "ALA", "mutations": ["GLY", "VAL"]},
                    {"resid": 42, "current": "PHE", "mutations": ["ALA", "LEU"]},
                ]
            },
            "Radical rebound cage": {
                "target_residues": [
                    {"resid": 75, "current": "LEU", "mutations": ["PHE", "ILE", "VAL"]},
                    {"resid": 78, "current": "VAL", "mutations": ["ALA", "LEU"]},
                ]
            },
            "Metal center optimization": {
                "target_residues": [{"resid": 400, "current": "CYS", "note": "Do not mutate axial ligand"}],
            },
        }
    },
    "serine_hydrolase": {
        "1CEX": {
            "Oxyanion hole strengthening": {
                "target_residues": [{"resid": 121, "current": "GLN"}],
                "nearby_tuning": [{"resid": 42, "current": "SER", "mutations": ["THR", "ALA"]}],
            },
            "Substrate positioning tunnel": {
                "target_residues": [
                    {"resid": 81, "current": "LEU", "mutations": ["ALA", "VAL", "PHE"]},
                    {"resid": 184, "current": "VAL", "mutations": ["ALA", "LEU"]},
                ]
            },
        }
    },
    "haloalkane_dehalogenase": {
        "1CQW": {
            "Substrate positioning tunnel": {
                "target_residues": [
                    {"resid": 177, "current": "LEU", "mutations": ["TRP", "GLY", "ALA"]},
                    {"resid": 176, "current": "CYS", "mutations": ["TYR", "PHE"]},
                    {"resid": 245, "current": "VAL", "mutations": ["ALA", "LEU"]},
                ]
            },
            "Halide stabilization pocket": {
                "target_residues": [
                    {"resid": 109, "current": "TRP", "note": "Essential halide stabilizer"},
                    {"resid": 41, "current": "ASN", "mutations": ["GLN", "HIS"]},
                ]
            },
        }
    },
}


def get_family_structures(enzyme_family: Optional[str]) -> List[Dict[str, Any]]:
    data = SCAFFOLD_DATABASE.get(str(enzyme_family or ""))
    if not isinstance(data, dict):
        return []
    structures = data.get("structures") or []
    if not isinstance(structures, list):
        return []
    return sorted(
        [entry for entry in structures if isinstance(entry, dict)],
        key=lambda entry: int(entry.get("priority") or 999),
    )


def select_scaffold(
    enzyme_family: Optional[str],
    substrate_smiles: Optional[str] = None,
    substrate_properties: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return highest-priority structure for family (v1 policy)."""
    structures = get_family_structures(enzyme_family)
    if not structures:
        return None
    return dict(structures[0])


def resolve_variant_targets(
    enzyme_family: Optional[str],
    pdb_id: Optional[str],
    variant_label: Optional[str],
) -> Dict[str, Any]:
    fam = str(enzyme_family or "")
    pid = str(pdb_id or "")
    label = str(variant_label or "")
    return (
        (VARIANT_RESIDUE_MAP.get(fam) or {}).get(pid, {}).get(label, {})
        if fam and pid and label
        else {}
    ) or {}

