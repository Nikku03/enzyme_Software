from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

try:
    from rdkit import Chem

    _RDKIT_OK = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    _RDKIT_OK = False

from nexus.models.mol_llama_wrapper import LatentBlueprint, MolLlamaWrapper
from nexus.models.structural_diffusion import Jacobian_Hook, StructuralDiffusion3D


SUPPORTED_ELEMENTS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


@dataclass
class NEXUS_Seed:
    pos: torch.Tensor
    z: torch.Tensor
    latent_blueprint: LatentBlueprint
    smiles: str
    atom_symbols: List[str]
    chirality_codes: torch.Tensor
    jacobian_hook: Jacobian_Hook | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "NEXUS_Seed":
        return NEXUS_Seed(
            pos=self.pos.to(device),
            z=self.z.to(device),
            latent_blueprint=self.latent_blueprint.to(device),
            smiles=self.smiles,
            atom_symbols=list(self.atom_symbols),
            chirality_codes=self.chirality_codes.to(device),
            jacobian_hook=self.jacobian_hook,
            metadata=dict(self.metadata),
        )


class NEXT_Mol_Generative_Agency(nn.Module):
    def __init__(
        self,
        model_path_llama: str | Path | None,
        model_path_diffusion: str | Path | None = None,
        *,
        latent_dim: int = 256,
        hidden_dim: int = 192,
        diffusion_steps: int = 6,
        freeze_llm: bool = True,
        allow_remote_weights: bool = False,
    ) -> None:
        super().__init__()
        self.model_path_llama = str(model_path_llama or "")
        self.model_path_diffusion = str(model_path_diffusion or "")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.fast_path_scheduler = "2026-fast-path"

        self.mol_llama = MolLlamaWrapper(
            model_path=self.model_path_llama,
            latent_dim=self.latent_dim,
            freeze_backbone=freeze_llm,
            allow_remote_weights=allow_remote_weights,
        )
        self.structural_diffusion = StructuralDiffusion3D(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            diffusion_steps=self.diffusion_steps,
            supported_atomic_numbers=list(SUPPORTED_ELEMENTS.values()),
        )

        if self.model_path_diffusion:
            path = Path(self.model_path_diffusion)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
                self.structural_diffusion.load_state_dict(state_dict or payload, strict=False)

    def _require_rdkit(self) -> None:
        if not _RDKIT_OK:
            raise ImportError("RDKit is required for NEXT_Mol_Generative_Agency SMILES validation")

    def _validate_and_parse(self, smiles: str):
        self._require_rdkit()
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol

    def _extract_atomic_numbers(self, mol) -> tuple[torch.Tensor, List[str]]:
        atomic_numbers: List[int] = []
        atom_symbols: List[str] = []
        for atom in mol.GetAtoms():
            atomic_num = int(atom.GetAtomicNum())
            symbol = str(atom.GetSymbol())
            if atomic_num not in SUPPORTED_ELEMENTS.values():
                raise ValueError(f"Unsupported element {symbol} (Z={atomic_num}) in NEXT_Mol_Generative_Agency")
            atomic_numbers.append(atomic_num)
            atom_symbols.append(symbol)
        return torch.tensor(atomic_numbers, dtype=torch.long), atom_symbols

    def _extract_chirality_codes(self, mol) -> torch.Tensor:
        codes = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
        for atom in mol.GetAtoms():
            tag = str(atom.GetChiralTag())
            code = 0
            if tag == "CHI_TETRAHEDRAL_CCW":
                code = 1
            elif tag == "CHI_TETRAHEDRAL_CW":
                code = -1
            bond_bias = 0
            for bond in atom.GetBonds():
                stereo = str(bond.GetStereo())
                if stereo == "STEREOE":
                    bond_bias = max(bond_bias, 2)
                elif stereo == "STEREOZ":
                    bond_bias = min(bond_bias, -2)
            if code == 0 and bond_bias != 0:
                code = bond_bias
            codes[atom.GetIdx()] = int(max(-2, min(2, code)))
        return codes

    def _chirality_signature(self, smiles: str, chirality_codes: torch.Tensor) -> torch.Tensor:
        signature = torch.zeros(8, dtype=torch.float32)
        signature[0] = float(smiles.count("@@"))
        signature[1] = float(max(0, smiles.count("@") - 2 * smiles.count("@@")))
        signature[2] = float(smiles.count("/"))
        signature[3] = float(smiles.count("\\"))
        signature[4] = float((chirality_codes == 1).sum().item())
        signature[5] = float((chirality_codes == -1).sum().item())
        signature[6] = float((chirality_codes == 2).sum().item())
        signature[7] = float((chirality_codes == -2).sum().item())
        return signature

    def smiles_to_latent(self, smiles: str) -> LatentBlueprint:
        mol = self._validate_and_parse(smiles)
        chirality_codes = self._extract_chirality_codes(mol)
        signature = self._chirality_signature(smiles, chirality_codes)
        return self.mol_llama.encode_one(smiles, chirality_signature=signature)

    def latent_to_3d(
        self,
        latent: LatentBlueprint,
        atomic_numbers: torch.Tensor,
        chirality_codes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Jacobian_Hook]:
        device = next(self.parameters()).device
        z = atomic_numbers.to(device)
        chirality = chirality_codes.to(device) if chirality_codes is not None else None
        coords, jacobian_hook = self.structural_diffusion(latent.to(device), z, chirality)
        coords = coords.requires_grad_(True)
        return coords, jacobian_hook

    def generate_seed(self, smiles: str) -> NEXUS_Seed:
        mol = self._validate_and_parse(smiles)
        atomic_numbers, atom_symbols = self._extract_atomic_numbers(mol)
        chirality_codes = self._extract_chirality_codes(mol)
        latent = self.mol_llama.encode_one(smiles, chirality_signature=self._chirality_signature(smiles, chirality_codes))
        pos, jacobian_hook = self.latent_to_3d(latent, atomic_numbers, chirality_codes)
        return NEXUS_Seed(
            pos=pos,
            z=atomic_numbers.to(pos.device),
            latent_blueprint=latent.to(pos.device),
            smiles=str(smiles),
            atom_symbols=atom_symbols,
            chirality_codes=chirality_codes.to(pos.device),
            jacobian_hook=jacobian_hook,
            metadata={
                "scheduler": self.fast_path_scheduler,
                "llm_source": latent.source,
                "num_atoms": int(atomic_numbers.numel()),
                "chiral_centers": int((chirality_codes != 0).sum().item()),
            },
        )

    def jacobian_link(self, objective: torch.Tensor, seed: NEXUS_Seed) -> torch.Tensor:
        scalar_objective = objective if objective.ndim == 0 else objective.sum()
        grad = torch.autograd.grad(
            outputs=scalar_objective,
            inputs=seed.pos,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        return grad

    def forward(self, smiles_list: str | Sequence[str]) -> NEXUS_Seed | List[NEXUS_Seed]:
        if isinstance(smiles_list, str):
            return self.generate_seed(smiles_list)
        return [self.generate_seed(smiles) for smiles in smiles_list]
