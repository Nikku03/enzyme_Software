from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.cahml.config import SITE_SMARTS_PATTERNS


RAW_ATOM_FEATURE_NAMES = [
    "atomic_num_norm",
    "is_aromatic",
    "total_h_norm",
    "formal_charge_norm",
    "in_ring",
    "in_ring5",
    "in_ring6",
    "is_sp",
    "is_sp2",
    "is_sp3",
    "nbr_c_norm",
    "nbr_n_norm",
    "nbr_o_norm",
    "degree_norm",
    "valence_norm",
]


@dataclass
class ChemistryFeatures:
    smiles: str
    mol_features_raw: "torch.Tensor"
    atom_features_raw: "torch.Tensor"
    smarts_matches: "torch.Tensor"
    site_types: List[str]
    num_atoms: int


if TORCH_AVAILABLE:
    class MoleculeEncoder(nn.Module):
        def __init__(self, *, fingerprint_dim: int = 2048, descriptor_dim: int = 10, hidden_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            raw_dim = fingerprint_dim + descriptor_dim
            self.output_dim = int(hidden_dim)
            self.net = nn.Sequential(
                nn.Linear(raw_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
            return self.net(raw_features.float())


    class AtomChemistryEncoder(nn.Module):
        def __init__(self, *, raw_feature_dim: int = 15, smarts_dim: int = len(SITE_SMARTS_PATTERNS), hidden_dim: int = 32, dropout: float = 0.1):
            super().__init__()
            self.output_dim = int(hidden_dim)
            self.net = nn.Sequential(
                nn.Linear(raw_feature_dim + smarts_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, atom_features_raw: torch.Tensor, smarts_matches: torch.Tensor) -> torch.Tensor:
            combined = torch.cat([atom_features_raw.float(), smarts_matches.float()], dim=-1)
            return self.net(combined)
else:  # pragma: no cover
    class MoleculeEncoder:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class AtomChemistryEncoder:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()


class ChemistryFeatureExtractor:
    def __init__(self, fingerprint_dim: int = 2048):
        require_torch()
        self.fingerprint_dim = int(fingerprint_dim)
        self.pattern_names = list(SITE_SMARTS_PATTERNS.keys())
        self.patterns = {
            name: Chem.MolFromSmarts(smarts)
            for name, smarts in SITE_SMARTS_PATTERNS.items()
        }

    def _safe_mol(self, smiles: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(smiles)
        return mol

    def compute_fingerprint(self, mol: Chem.Mol) -> torch.Tensor:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fingerprint_dim)
        return torch.tensor(list(fp), dtype=torch.float32)

    def compute_descriptors(self, mol: Chem.Mol) -> torch.Tensor:
        values = [
            Descriptors.MolWt(mol) / 700.0,
            Descriptors.MolLogP(mol) / 8.0,
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.NumHDonors(mol) / 8.0,
            Descriptors.NumHAcceptors(mol) / 12.0,
            Descriptors.NumRotatableBonds(mol) / 20.0,
            rdMolDescriptors.CalcNumRings(mol) / 8.0,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 6.0,
            rdMolDescriptors.CalcNumHeterocycles(mol) / 6.0,
            rdMolDescriptors.CalcFractionCSP3(mol),
        ]
        return torch.tensor(values, dtype=torch.float32)

    def compute_atom_features(self, mol: Chem.Mol, atom_idx: int) -> torch.Tensor:
        atom = mol.GetAtomWithIdx(atom_idx)
        features = [
            atom.GetAtomicNum() / 20.0,
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs() / 4.0,
            atom.GetFormalCharge() / 2.0,
            float(atom.IsInRing()),
            float(atom.IsInRingSize(5)),
            float(atom.IsInRingSize(6)),
            float(atom.GetHybridization() == Chem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            len([n for n in atom.GetNeighbors() if n.GetSymbol() == "C"]) / 4.0,
            len([n for n in atom.GetNeighbors() if n.GetSymbol() == "N"]) / 3.0,
            len([n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]) / 3.0,
            atom.GetDegree() / 4.0,
            atom.GetTotalValence() / 6.0,
        ]
        return torch.tensor(features, dtype=torch.float32)

    def compute_smarts_matches(self, mol: Chem.Mol) -> torch.Tensor:
        n_atoms = mol.GetNumAtoms()
        out = torch.zeros((n_atoms, len(self.pattern_names)), dtype=torch.float32)
        for j, name in enumerate(self.pattern_names):
            pattern = self.patterns.get(name)
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                for atom_idx in match:
                    if 0 <= atom_idx < n_atoms:
                        out[atom_idx, j] = 1.0
        return out

    def classify_site_types(self, smarts_matches: torch.Tensor, atom_features_raw: torch.Tensor) -> List[str]:
        site_types: List[str] = []
        for atom_idx in range(smarts_matches.shape[0]):
            matches = smarts_matches[atom_idx]
            if atom_features_raw[atom_idx, 0].item() <= 0.0:
                site_types.append("invalid")
                continue
            assigned = "generic_site"
            for j, name in enumerate(self.pattern_names):
                if matches[j].item() > 0.5:
                    assigned = name
                    break
            site_types.append(assigned)
        return site_types

    def extract(self, smiles: str) -> Optional[ChemistryFeatures]:
        mol = self._safe_mol(smiles)
        if mol is None:
            return None
        n_atoms = mol.GetNumAtoms()
        mol_features_raw = torch.cat([self.compute_fingerprint(mol), self.compute_descriptors(mol)], dim=0)
        atom_features_raw = torch.stack([self.compute_atom_features(mol, idx) for idx in range(n_atoms)], dim=0)
        smarts_matches = self.compute_smarts_matches(mol)
        site_types = self.classify_site_types(smarts_matches, atom_features_raw)
        return ChemistryFeatures(
            smiles=smiles,
            mol_features_raw=mol_features_raw,
            atom_features_raw=atom_features_raw,
            smarts_matches=smarts_matches,
            site_types=site_types,
            num_atoms=n_atoms,
        )
