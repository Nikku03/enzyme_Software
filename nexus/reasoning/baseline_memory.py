"""
nexus/reasoning/baseline_memory.py

Phase 1 of the Analogical Reasoning Engine: ECFP4-based memory bank with
MCS-guided transport.

Retrieval uses cosine similarity over L2-normalised Morgan fingerprints.
The leave-one-out anti-cheat discards any hit with similarity > 0.999 so
that training queries never trivially retrieve themselves.

Transport uses RDKit's rdFMCS to find the Maximum Common Substructure, then
walks the atom-index mapping to transfer the historical SoM label onto the
query molecule's atom space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from nexus.data.metabolic_dataset import NUM_MORPHISM_CLASSES, REACTION_TAXONOMY

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFMCS
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False


# Same key priority order as ZaretzkiMetabolicDataset so both systems read
# the same label from whatever SDF property is actually present.
_SOM_KEYS = (
    "PRIMARY_SOM",
    "SOM_IDX",
    "SoM_IDX",
    "SOM",
    "SOM_INDEX",
    "SECONDARY_SOM",
    "SITE_OF_METABOLISM",
)

_REACTION_PROP_KEYS = (
    "REACTIONS",
    "REACTION",
    "REACTION_TYPE",
    "REACTION_TYPES",
    "METABOLISM_TYPE",
    "METABOLISM_TYPES",
)

_LABEL_SOURCE_KEYS = (
    "LABEL_SOURCE",
    "SITE_SOURCE",
    "ANNOTATION_SOURCE",
    "SOURCE",
)


def _extract_som_idx(mol) -> Optional[int]:
    """Return the 0-based primary SoM atom index, or None if unavailable."""
    for key in _SOM_KEYS:
        if not mol.HasProp(key):
            continue
        raw = str(mol.GetProp(key)).strip()
        if not raw:
            continue
        token = raw.split()[0].split(",")[0].split(";")[0].strip()
        try:
            idx = int(float(token)) - 1   # SDF is 1-based
        except Exception:
            continue
        if 0 <= idx < mol.GetNumAtoms():
            return idx
    return None


def _morgan_fp_tensor(mol, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    except ImportError:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float32)


def scaffold_key(mol) -> str:
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return str(scaffold or "")
    except Exception:
        return ""


def _tokenize_prop(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    normalized = text.replace(";", ",").replace("|", ",").replace("/", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _normalize_reaction_name(name: str) -> str:
    value = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "hydroxylation": "hydroxylation",
        "aliphatic_hydroxylation": "aliphatic_hydroxylation",
        "aromatic_hydroxylation": "aromatic_hydroxylation",
        "n_dealkylation": "dealkylation",
        "o_dealkylation": "dealkylation",
        "dealkylation": "dealkylation",
        "n_oxidation": "oxidation_n_s",
        "s_oxidation": "oxidation_n_s",
        "oxidation_n_s": "oxidation_n_s",
        "epoxidation": "epoxidation",
    }
    return aliases.get(value, value)


def _extract_reaction_names(mol) -> List[str]:
    reactions: List[str] = []
    for key in _REACTION_PROP_KEYS:
        if not mol.HasProp(key):
            continue
        reactions.extend(_tokenize_prop(mol.GetProp(key)))
    deduped: List[str] = []
    seen = set()
    for reaction in reactions:
        normalized = _normalize_reaction_name(reaction)
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _extract_label_source(mol) -> str:
    for key in _LABEL_SOURCE_KEYS:
        if mol.HasProp(key):
            value = str(mol.GetProp(key)).strip()
            if value:
                return value
    return ""


def _label_confidence_from_source(source: str) -> float:
    value = str(source or "").strip().lower()
    if not value:
        return 0.0
    if any(token in value for token in ("assay", "clinical", "validated", "manual", "curated")):
        return 1.0
    if any(token in value for token in ("drugbank", "metxbio", "literature", "reported")):
        return 0.75
    if any(token in value for token in ("heuristic", "smarts", "inferred", "predicted", "rule")):
        return 0.5
    return 0.5


def build_morphism_supervision(mol, som_idx: int, *, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor, bool, float]:
    num_atoms = int(mol.GetNumAtoms())
    morphism_target = torch.zeros((num_atoms, NUM_MORPHISM_CLASSES), dtype=torch.float32, device=device)
    morphism_loss_mask = torch.zeros((num_atoms, NUM_MORPHISM_CLASSES), dtype=torch.float32, device=device)
    if not (0 <= int(som_idx) < num_atoms):
        return morphism_target, morphism_loss_mask, False, 0.0

    reactions = _extract_reaction_names(mol)
    atom = mol.GetAtomWithIdx(int(som_idx))
    classes: List[int] = []
    seen = set()
    for reaction in reactions:
        canonical = _normalize_reaction_name(reaction)
        if canonical == "hydroxylation":
            canonical = "aromatic_hydroxylation" if atom.GetIsAromatic() else "aliphatic_hydroxylation"
        elif canonical == "oxidation_n_s":
            if atom.GetSymbol() not in {"N", "S"}:
                continue
        elif canonical == "epoxidation":
            if atom.GetSymbol() != "C":
                continue
        elif canonical == "dealkylation":
            if atom.GetSymbol() not in {"C", "N", "O"}:
                continue
        class_idx = REACTION_TAXONOMY.get(canonical)
        if class_idx is None or class_idx in seen:
            continue
        classes.append(class_idx)
        seen.add(class_idx)

    has_label = len(classes) > 0
    confidence = _label_confidence_from_source(_extract_label_source(mol))
    if has_label:
        if confidence <= 0.0:
            confidence = 0.5
        morphism_target[int(som_idx), classes] = 1.0
        morphism_loss_mask[int(som_idx), :] = 1.0
    return morphism_target, morphism_loss_mask, has_label, confidence


@dataclass
class MemoryRetrievalResult:
    """All outputs produced by a single retrieve-and-transport call."""
    analogical_pred: torch.Tensor               # [N_query] one-hot float, 0 if transport failed
    confidence: float                           # cosine similarity to retrieved neighbour
    retrieved_mol: object                       # RDKit Mol
    retrieved_som_idx: int                      # 0-based SoM on the retrieved molecule
    transport_succeeded: bool                   # False when MCS mapping missed the SoM atom
    mcs_size: int                               # number of atoms in the MCS (0 if MCS failed)
    query_embed: Optional[torch.Tensor] = None          # [embed_dim] MechanismEncoder output for query
    retrieved_embed_detached: Optional[torch.Tensor] = None  # [embed_dim] detached encoder output for retrieved mol
    embedding_space: str = "euclidean"
    transport_backend: str = "mcs"
    transport_support_size: int = 0
    transported_mass: float = 0.0
    retrieved_same_query: bool = False
    transport_distill_loss: Optional[torch.Tensor] = None
    neuralgw_used_exact: bool = False
    neuralgw_confidence: float = 0.0
    neuralgw_distill_loss: float = 0.0
    transport_plan: Optional[torch.Tensor] = None
    retrieved_node_multivectors: Optional[torch.Tensor] = None
    transport_error_message: Optional[str] = None
    retrieval_mix_count: int = 1
    retrieval_mix_entropy: float = 0.0
    retrieved_morphism_target: Optional[torch.Tensor] = None
    retrieved_morphism_loss_mask: Optional[torch.Tensor] = None
    retrieved_has_morphism_label: bool = False
    retrieved_label_confidence: float = 0.0
    retrieved_scaffold: str = ""
    retrieval_candidate_count: int = 1
    retrieval_mechanism_overlap: float = 0.0
    retrieval_diversity_score: float = 0.0
    neuralgw_route_reason: str = "unavailable"


class BaselineMemoryBank:
    """
    ECFP4 Euclidean memory bank with MCS-guided SoM transport.

    Usage
    -----
    bank = BaselineMemoryBank(device='cpu')
    bank.populate_from_mols(list_of_rdkit_mols)
    result = bank.retrieve_and_transport(query_mol)
    """

    def __init__(
        self,
        device: str = "cpu",
        fp_radius: int = 2,
        fp_bits: int = 2048,
        identity_threshold: float = 0.999,
        mcs_timeout: int = 2,
    ) -> None:
        if not _RDKIT_OK:
            raise ImportError("RDKit is required for BaselineMemoryBank")
        self.device = device
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.identity_threshold = identity_threshold
        self.mcs_timeout = mcs_timeout

        self.historical_mols: List = []
        self.historical_soms: List[int] = []
        self.memory_embeddings: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate_from_mols(self, mols: List) -> None:
        """Build the L2-normalised ECFP4 memory matrix from a list of RDKit Mols.

        Molecules whose SoM cannot be determined are silently skipped so
        the memory bank only contains labelled examples.
        """
        print(f"Populating Analogical Memory Bank from {len(mols)} input molecules...")
        fps: List[torch.Tensor] = []
        skipped = 0

        for mol in mols:
            som_idx = _extract_som_idx(mol)
            if som_idx is None:
                skipped += 1
                continue
            fps.append(_morgan_fp_tensor(mol, self.fp_radius, self.fp_bits))
            self.historical_mols.append(mol)
            self.historical_soms.append(som_idx)

        if not fps:
            raise RuntimeError("No labelled molecules found — memory bank is empty.")

        self.memory_embeddings = F.normalize(
            torch.stack(fps).to(self.device), p=2, dim=1
        )
        print(
            f"Memory Bank Active: {len(self.historical_mols)} molecules "
            f"({skipped} skipped — no SoM label)."
        )

    # ------------------------------------------------------------------
    # Retrieval + Transport
    # ------------------------------------------------------------------

    def retrieve_and_transport(self, query_mol) -> MemoryRetrievalResult:
        """
        Retrieve the nearest labelled analogue (excluding self-matches) and
        transport its SoM label onto the query via Maximum Common Substructure.

        Returns a MemoryRetrievalResult; `analogical_pred` is all-zero when
        the MCS either timed out or the SoM atom fell outside the shared
        scaffold (the 'alien appendage' case).
        """
        if self.memory_embeddings is None:
            raise RuntimeError("Call populate_from_mols() before retrieve_and_transport().")

        N_query = query_mol.GetNumAtoms()
        analogical_pred = torch.zeros(N_query, dtype=torch.float32, device=self.device)

        # ── 1. Fingerprint similarity ──────────────────────────────────
        q_fp = _morgan_fp_tensor(query_mol, self.fp_radius, self.fp_bits)
        q_fp = F.normalize(q_fp.unsqueeze(0).to(self.device), p=2, dim=1)
        sim = torch.matmul(q_fp, self.memory_embeddings.T).squeeze(0)  # [M]

        k = min(2, len(self.historical_mols))
        top_scores, top_indices = torch.topk(sim, k)

        # Anti-cheat: if the top hit is the query itself, use the runner-up
        best_idx = int(top_indices[0].item())
        confidence = float(top_scores[0].item())
        if confidence > self.identity_threshold and k > 1:
            best_idx = int(top_indices[1].item())
            confidence = float(top_scores[1].item())

        retrieved_mol = self.historical_mols[best_idx]
        retrieved_som = self.historical_soms[best_idx]
        morph_target, morph_mask, has_morph, morph_conf = build_morphism_supervision(
            retrieved_mol,
            retrieved_som,
            device=self.device,
        )

        # ── 2. MCS-guided transport ────────────────────────────────────
        mcs_size = 0
        transport_ok = False

        res = rdFMCS.FindMCS(
            [query_mol, retrieved_mol],
            timeout=self.mcs_timeout,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )

        if not res.canceled and res.numAtoms > 0:
            mcs_size = res.numAtoms
            mcs_mol = Chem.MolFromSmarts(res.smartsString)
            match_query = query_mol.GetSubstructMatch(mcs_mol)
            match_retrieved = retrieved_mol.GetSubstructMatch(mcs_mol)

            if match_query and match_retrieved:
                try:
                    # Walk the MCS mapping: retrieved_som -> MCS position -> query atom
                    mcs_pos = match_retrieved.index(retrieved_som)
                    mapped_query_idx = match_query[mcs_pos]
                    analogical_pred[mapped_query_idx] = 1.0
                    transport_ok = True
                except ValueError:
                    # retrieved_som is outside the MCS (alien appendage) — leave zeros
                    pass

        return MemoryRetrievalResult(
            analogical_pred=analogical_pred,
            confidence=confidence,
            retrieved_mol=retrieved_mol,
            retrieved_som_idx=retrieved_som,
            transport_succeeded=transport_ok,
            mcs_size=mcs_size,
            retrieved_morphism_target=morph_target,
            retrieved_morphism_loss_mask=morph_mask,
            retrieved_has_morphism_label=has_morph,
            retrieved_label_confidence=morph_conf,
        )

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def batch_stats(self, mols: List, true_soms: List[int]) -> dict:
        """
        Run retrieve-and-transport on every molecule and compute:
          - transport_success_rate : fraction where MCS mapped the SoM
          - top1_accuracy          : fraction where predicted == true SoM
          - mean_confidence        : average retrieval cosine similarity
          - mean_mcs_size          : average MCS atom count
        """
        n = len(mols)
        successes = top1_hits = 0
        total_conf = total_mcs = 0.0

        for mol, true_som in zip(mols, true_soms):
            r = self.retrieve_and_transport(mol)
            total_conf += r.confidence
            total_mcs += r.mcs_size
            if r.transport_succeeded:
                successes += 1
                pred_atom = int(r.analogical_pred.argmax().item())
                if pred_atom == true_som:
                    top1_hits += 1

        return {
            "n": n,
            "transport_success_rate": successes / n,
            "top1_accuracy": top1_hits / n,
            "mean_confidence": total_conf / n,
            "mean_mcs_size": total_mcs / n,
        }
