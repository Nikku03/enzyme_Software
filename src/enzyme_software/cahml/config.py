from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES


REACTION_TYPES = [
    "aliphatic_hydroxylation",
    "aromatic_hydroxylation",
    "benzylic_hydroxylation",
    "n_demethylation",
    "o_demethylation",
    "oxidation",
    "n_oxidation",    # tertiary amine → N-oxide
    "s_oxidation",    # thioether/thiophene → sulfoxide
    "epoxidation",    # alkene/arene → epoxide
]

REACTION_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(REACTION_TYPES)}

SITE_SMARTS_PATTERNS: Dict[str, str] = {
    # ── Heteroatom oxidation sites (highest priority in classify_site_types) ────
    "s_oxidation":         "[S;X2;!$([S]=*)]",               # thioether S, S-oxidation by CYP3A4/2C19
    "thiophene_s":         "[s;r5]",                          # aromatic thiophene S, reactive S-oxide
    "n_oxidation":         "[N;X3;H0;!$([N+]);!$([N]=*)]",   # tertiary amine N-oxide by CYP3A4
    "primary_aro_amine":   "[NH2;!R][c]",                     # aniline N-hydroxylation
    "ring_nitrogen_6":     "[N;r6;X3]",                       # piperidine/piperazine/morpholine N
    # ── Classic high-reactivity C-H sites ───────────────────────────────────────
    "benzylic":            "[CH2,CH3][c]",
    "allylic":             "[CH2,CH3][C]=[C]",
    "o_methyl_aromatic":   "[CH3]O[c]",
    "n_methyl":            "[CH3]N",
    "alpha_to_nitrogen":   "[CH2,CH3][N;!$([N]=*)]",
    "alpha_to_oxygen":     "[CH2,CH3]O",
    "tertiary_ch":         "[CH]([C])([C])[C]",
    "alkene_epoxidation":  "[C]=[C]",                         # alkene epoxidation
    "carbonyl_alpha":      "[CH2,CH3][C]=O",                  # alpha to carbonyl
    # ── Deactivated / blocked sites ─────────────────────────────────────────────
    "aromatic_unactivated": "[cH]",
    "methyl_no_activation": "[CH3][C;!$([C]=*);!$([C]#[C])]",
    "halogen_adjacent":    "[C][F,Cl,Br,I]",                  # deactivated by electron-withdrawing halogen
    "quaternary":          "[C;X4;H0]",
    "carbonyl":            "[C]=O",
    "carboxyl":            "[C](=O)[O]",
}

PHYSICS_RULES: Dict[str, Tuple[str, float, str]] = {
    # ── Heteroatom oxidation boosts ──────────────────────────────────────────────
    "s_oxidation":         ("boost", 1.8,  "Thioether S-oxidation handle — CYP3A4/2C19"),
    "thiophene_s":         ("boost", 1.6,  "Thiophene S-oxidation → reactive sulfoxide intermediate"),
    "n_oxidation":         ("boost", 1.6,  "Tertiary amine N-oxidation handle — CYP3A4"),
    "primary_aro_amine":   ("boost", 1.3,  "Primary aromatic amine N-hydroxylation"),
    "ring_nitrogen_6":     ("boost", 1.5,  "Piperidine/piperazine N — common CYP substrate handle"),
    # ── C-H oxidation boosts ─────────────────────────────────────────────────────
    "benzylic":            ("boost", 2.0,  "Stabilized benzylic radical"),
    "allylic":             ("boost", 1.7,  "Stabilized allylic radical"),
    "o_methyl_aromatic":   ("boost", 2.3,  "Classic O-demethylation handle"),
    "n_methyl":            ("boost", 2.0,  "Classic N-demethylation handle"),
    "alpha_to_nitrogen":   ("boost", 1.4,  "Activated alpha-to-nitrogen site"),
    "alpha_to_oxygen":     ("boost", 1.25, "Activated alpha-to-oxygen site"),
    "tertiary_ch":         ("boost", 1.3,  "Tertiary radical stabilization"),
    "alkene_epoxidation":  ("boost", 1.4,  "Alkene epoxidation target"),
    "carbonyl_alpha":      ("boost", 1.3,  "Alpha to carbonyl — activated C-H position"),
    # ── Deactivated / penalized sites ────────────────────────────────────────────
    "aromatic_unactivated": ("penalize", 0.5, "Unactivated aromatic C-H"),
    "methyl_no_activation": ("penalize", 0.7, "Unactivated methyl group"),
    "halogen_adjacent":    ("penalize", 0.6,  "Halogen-adjacent C — deactivated by electron withdrawal"),
    # ── Hard blocks (no chemistry possible) ─────────────────────────────────────
    "quaternary":          ("block", 0.0, "Quaternary carbon — no hydrogen available"),
    "carbonyl":            ("block", 0.0, "Carbonyl carbon — not a C-H metabolic site"),
    "carboxyl":            ("block", 0.0, "Carboxyl carbon — not a C-H metabolic site"),
}

CYP_SUBSTRATE_PATTERNS = {
    "CYP1A2": {
        "preferred_substrates": ["planar_aromatic", "polycyclic"],
        "typical_reactions": ["aromatic_hydroxylation", "n_demethylation", "n_oxidation"],
    },
    "CYP2C9": {
        "preferred_substrates": ["acidic", "lipophilic", "aromatic"],
        "typical_reactions": ["aromatic_hydroxylation", "benzylic_hydroxylation", "s_oxidation"],
    },
    "CYP2C19": {
        "preferred_substrates": ["basic_nitrogen", "two_aromatic_rings"],
        "typical_reactions": ["n_demethylation", "aromatic_hydroxylation", "s_oxidation", "oxidation"],
    },
    "CYP2D6": {
        "preferred_substrates": ["basic_nitrogen_5_7A_from_site", "lipophilic"],
        "typical_reactions": ["aliphatic_hydroxylation", "o_demethylation"],
    },
    "CYP3A4": {
        "preferred_substrates": ["large_lipophilic", "mw_over_300"],
        "typical_reactions": ["aliphatic_hydroxylation", "n_demethylation", "n_oxidation", "s_oxidation", "epoxidation"],
    },
}


@dataclass
class CAHMLConfig:
    checkpoint_dir: str = "checkpoints/cahml"
    artifact_dir: str = "artifacts/cahml"
    cache_dir: str = "cache/cahml"
    n_models: int = 3
    model_names: List[str] = field(default_factory=lambda: ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"])
    mol_fingerprint_dim: int = 2048
    mol_descriptor_dim: int = 10
    atom_raw_feature_dim: int = 15
    smarts_pattern_dim: int = len(SITE_SMARTS_PATTERNS)
    hidden_dim: int = 64
    n_cyp_classes: int = len(MAJOR_CYP_CLASSES)
    n_reaction_types: int = len(REACTION_TYPES)
    use_physics_constraints: bool = True
    constraint_boost_factor: float = 1.5
    constraint_penalty_factor: float = 0.3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    dropout: float = 0.1
    epochs: int = 100
    patience: int = 15
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    use_uncertainty: bool = True
    uncertainty_threshold: float = 0.3
    site_loss_weight: float = 1.0
    cyp_loss_weight: float = 0.5
    reaction_loss_weight: float = 0.25
    use_base_cyp_prior: bool = True
    mirank_weight: float = 1.0
    bce_weight: float = 0.3
    listmle_weight: float = 0.5
    focal_weight: float = 0.2
    ranking_margin: float = 1.0
    hard_negative_fraction: float | None = 0.5

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
