"""Substrate Reality Engine (SRE) - Atomic Truth Registry (ATR)

This is the *foundation* that prevents index chaos.

Principles:
- One atom = one UUID (canonical identity).
- All mappings are explicit, bidirectional, and validated.
- Roles are validated (lightweight chemistry sanity rules).

This module is intentionally minimal: it does not decide mechanisms or do QM.
It only guarantees that *atom identity + mappings + roles* are correct.

Requires: RDKit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Iterable, Union
import uuid

from rdkit import Chem

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


class AtomRole(str, Enum):
    ELECTROPHILE = "electrophile"
    NUCLEOPHILE = "nucleophile"
    LEAVING_GROUP = "leaving_group"
    STABILIZER = "stabilizer"          # e.g., oxyanion hole acceptor oxygen
    GENERAL_ACID = "general_acid"
    GENERAL_BASE = "general_base"
    METAL_BINDER = "metal_binder"
    RADICAL_SITE = "radical_site"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AtomRef:
    """ATR-backed atom reference (no raw indices outside original_index)."""

    atr: "AtomicTruthRegistry"
    atom_id: str
    source: Optional[str] = None
    note: Optional[str] = None

    def validate_exists(self) -> None:
        if self.atom_id not in self.atr._atoms:
            raise ValueError(f"AtomRef atom_id not in ATR: {self.atom_id}")

    @property
    def truth(self) -> "AtomTruth":
        self.validate_exists()
        return self.atr.get_by_uuid(self.atom_id)

    @property
    def element(self) -> str:
        return self.truth.element

    @property
    def formal_charge(self) -> int:
        return self.truth.formal_charge

    @property
    def is_aromatic(self) -> bool:
        return self.truth.is_aromatic

    @property
    def hybridization(self) -> str:
        return self.truth.hybridization

    @property
    def original_index(self) -> int:
        return int(self.truth.parent_index)


@dataclass
class AtomTruth:
    """Single source of truth for one atom."""

    atom_id: str
    element: str
    atomic_number: int
    formal_charge: int
    is_aromatic: bool
    hybridization: str
    canonical_rank: int

    # Parent/fragment indices (optional, explicit)
    parent_index: Optional[int] = None
    fragment_index: Optional[int] = None

    # Computed properties (optional)
    partial_charge: Optional[float] = None

    # role -> confidence + provenance
    roles: Dict[AtomRole, float] = field(default_factory=dict)
    role_sources: Dict[AtomRole, str] = field(default_factory=dict)

    def can_play_role(self, role: AtomRole) -> bool:
        """Lightweight chemistry sanity checks.

        These are *guards*, not full chemistry. They prevent obviously wrong assignments.
        """

        el = self.element
        q = self.formal_charge

        if role == AtomRole.ELECTROPHILE:
            # Common electrophiles: carbonyl carbon (C), phosphoryl (P), sulfonyl (S)
            return el in {"C", "P", "S"} and q <= 1

        if role == AtomRole.NUCLEOPHILE:
            # Common nucleophiles: O, N, S (and sometimes halides, but keep strict)
            return el in {"O", "N", "S"} and q >= -2

        if role == AtomRole.LEAVING_GROUP:
            # Typical leaving groups: halogens, O/N in esters/amides, sulfonates, etc.
            return el in {"F", "Cl", "Br", "I", "O", "N", "S"}

        if role == AtomRole.STABILIZER:
            # Stabilizers are often hetero atoms near developing charge (O/N)
            return el in {"O", "N", "S"}

        if role in {AtomRole.GENERAL_ACID, AtomRole.GENERAL_BASE}:
            return el in {"O", "N", "S"}

        if role == AtomRole.METAL_BINDER:
            return el in {"O", "N", "S", "P"}

        # UNKNOWN / RADICAL_SITE: allow
        return True


class ATRConsistencyError(ValueError):
    pass


class AtomicTruthRegistry:
    """Manages AtomTruth objects and all mappings.

    Parent molecule is the single canonical source. Fragments must be registered
    with explicit atom mapping back to the parent.
    """

    def __init__(self, mol: Chem.Mol, *, parent_smiles: str, keep_hs: bool = False):
        self.parent_smiles = parent_smiles
        self.keep_hs = keep_hs

        # Store a sanitized copy
        self.parent_mol = Chem.Mol(mol)

        # Atom UUIDs are stored as RDKit properties to keep them tied to the mol.
        self._atoms: Dict[str, AtomTruth] = {}
        self._parent_index_to_id: Dict[int, str] = {}
        self._warnings: List[str] = []

        # fragment_id -> (fragment_mol, fragment_index_to_id, id_to_fragment_index)
        self._fragments: Dict[str, Dict[str, Any]] = {}

        self._init_parent_atoms()

    @classmethod
    def from_smiles(cls, smiles: str, *, keep_hs: bool = False) -> "AtomicTruthRegistry":
        prep = prepare_mol(smiles)
        if prep.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = prep.mol
        mol = Chem.AddHs(mol) if keep_hs else Chem.Mol(mol)
        parent_smiles = prep.canonical_smiles or Chem.MolToSmiles(mol, canonical=True)
        return cls(mol, parent_smiles=parent_smiles, keep_hs=keep_hs)

    @classmethod
    def from_molblock(
        cls, molblock: str, *, keep_hs: bool = False
    ) -> "AtomicTruthRegistry":
        mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=not keep_hs)
        if mol is None:
            raise ValueError("Invalid molblock")
        mol = Chem.AddHs(mol) if keep_hs else Chem.Mol(mol)
        parent_smiles = Chem.MolToSmiles(mol, canonical=True)
        return cls(mol, parent_smiles=parent_smiles, keep_hs=keep_hs)

    # ---------- parent atom identity ----------

    def _init_parent_atoms(self) -> None:
        ranks = list(Chem.CanonicalRankAtoms(self.parent_mol))
        for atom in self.parent_mol.GetAtoms():
            idx = atom.GetIdx()
            aid = str(uuid.uuid4())
            atom.SetProp("_ATR_ID", aid)

            truth = AtomTruth(
                atom_id=aid,
                element=atom.GetSymbol(),
                atomic_number=atom.GetAtomicNum(),
                formal_charge=int(atom.GetFormalCharge()),
                is_aromatic=bool(atom.GetIsAromatic()),
                hybridization=str(atom.GetHybridization()),
                canonical_rank=int(ranks[idx]) if idx < len(ranks) else -1,
                parent_index=idx,
            )

            self._atoms[aid] = truth
            self._parent_index_to_id[idx] = aid

    def atom_id_from_parent_index(self, parent_index: int) -> str:
        if parent_index not in self._parent_index_to_id:
            raise KeyError(f"Parent atom index not in registry: {parent_index}")
        return self._parent_index_to_id[parent_index]

    def parent_index_from_atom_id(self, atom_id: str) -> int:
        return int(self.atom(atom_id).parent_index)

    def atom(self, atom_ref_or_id: Union[AtomRef, str]) -> AtomTruth:
        atom_id = atom_ref_or_id.atom_id if isinstance(atom_ref_or_id, AtomRef) else atom_ref_or_id
        try:
            return self._atoms[atom_id]
        except KeyError as e:
            raise KeyError(f"Unknown atom_id: {atom_id}") from e

    def ref_from_parent_index(self, parent_index: int) -> AtomRef:
        return AtomRef(self, self.atom_id_from_parent_index(parent_index))

    def get_by_uuid(self, atom_id: str) -> AtomTruth:
        return self.atom(atom_id)

    def get_by_parent_index(self, parent_index: int) -> AtomTruth:
        return self.atom(self.atom_id_from_parent_index(parent_index))

    # ---------- roles ----------

    def assign_role(
        self, atom_ref_or_id: Union[AtomRef, str], role: AtomRole, confidence: float, source: str
    ) -> List[str]:
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        at = self.atom(atom_ref_or_id)
        warnings: List[str] = []
        if role == AtomRole.ELECTROPHILE and at.element == "H":
            raise ATRConsistencyError(
                f"Impossible role assignment: {role} cannot be assigned to H (atom_id={at.atom_id})"
            )
        if not at.can_play_role(role):
            warnings.append(
                f"Unlikely role assignment: {role} on {at.element} (q={at.formal_charge})"
            )
        prev = at.roles.get(role, 0.0)
        if confidence >= prev:
            at.roles[role] = confidence
            at.role_sources[role] = source
        if warnings:
            self._warnings.extend(warnings)
        return warnings

    # ---------- fragments and mappings ----------

    def register_fragment(
        self,
        fragment_mol: Chem.Mol,
        atom_map_parent_to_fragment: Dict[int, int],
        *,
        fragment_id: Optional[str] = None,
    ) -> str:
        """Register a fragment with an explicit mapping.

        atom_map_parent_to_fragment: {parent_atom_index -> fragment_atom_index}

        Validations:
        - indices exist in both mols
        - element match
        - mapping is injective (no two parents map to same fragment atom)
        """

        if fragment_id is None:
            fragment_id = str(uuid.uuid4())

        frag = Chem.Mol(fragment_mol)

        # Validate injective mapping
        inv_seen: Dict[int, int] = {}
        for p_idx, f_idx in atom_map_parent_to_fragment.items():
            if f_idx in inv_seen:
                raise ATRConsistencyError(
                    f"Non-injective mapping: fragment atom {f_idx} mapped from parents {inv_seen[f_idx]} and {p_idx}"
                )
            inv_seen[f_idx] = p_idx

        frag_index_to_id: Dict[int, str] = {}
        id_to_frag_index: Dict[str, int] = {}

        for p_idx, f_idx in atom_map_parent_to_fragment.items():
            if p_idx < 0 or p_idx >= self.parent_mol.GetNumAtoms():
                raise ATRConsistencyError(f"Parent index out of range: {p_idx}")
            if f_idx < 0 or f_idx >= frag.GetNumAtoms():
                raise ATRConsistencyError(f"Fragment index out of range: {f_idx}")

            p_atom = self.parent_mol.GetAtomWithIdx(p_idx)
            f_atom = frag.GetAtomWithIdx(f_idx)
            if p_atom.GetSymbol() != f_atom.GetSymbol():
                raise ATRConsistencyError(
                    f"Element mismatch at mapping parent {p_idx}({p_atom.GetSymbol()}) -> fragment {f_idx}({f_atom.GetSymbol()})"
                )

            aid = self.atom_id_from_parent_index(p_idx)
            frag_index_to_id[f_idx] = aid
            id_to_frag_index[aid] = f_idx

            # back-fill fragment_index on AtomTruth
            self._atoms[aid].fragment_index = f_idx

        self._fragments[fragment_id] = {
            "mol": frag,
            "parent_to_fragment": dict(atom_map_parent_to_fragment),
            "fragment_to_parent": {f: p for p, f in atom_map_parent_to_fragment.items()},
            "fragment_index_to_id": frag_index_to_id,
            "id_to_fragment_index": id_to_frag_index,
        }

        return fragment_id

    def fragment_atom_ref(self, fragment_id: str, fragment_index: int) -> AtomRef:
        frag_info = self._fragments.get(fragment_id)
        if frag_info is None:
            raise KeyError(f"Unknown fragment_id: {fragment_id}")
        if fragment_index not in frag_info["fragment_index_to_id"]:
            raise KeyError(f"Fragment index not mapped: {fragment_index}")
        return AtomRef(self, frag_info["fragment_index_to_id"][fragment_index])

    def parent_index_from_fragment_index(self, fragment_id: str, fragment_index: int) -> int:
        frag_info = self._fragments.get(fragment_id)
        if frag_info is None:
            raise KeyError(f"Unknown fragment_id: {fragment_id}")
        if fragment_index not in frag_info["fragment_to_parent"]:
            raise KeyError(f"Fragment index not mapped: {fragment_index}")
        return int(frag_info["fragment_to_parent"][fragment_index])

    # ---------- validation / debug ----------

    def validate(self) -> List[str]:
        """Return a list of issues. Empty list means OK."""
        issues: List[str] = []

        # Parent mapping completeness
        if len(self._parent_index_to_id) != self.parent_mol.GetNumAtoms():
            issues.append("Parent atom registry size mismatch")

        # Ensure every parent index maps to AtomTruth with correct parent_index
        for idx, aid in self._parent_index_to_id.items():
            at = self._atoms.get(aid)
            if at is None:
                issues.append(f"Missing AtomTruth for id {aid}")
                continue
            if at.parent_index != idx:
                issues.append(f"AtomTruth parent_index mismatch for atom_id {aid}: {at.parent_index} != {idx}")

        # Fragment validations
        for fid, finfo in self._fragments.items():
            frag = finfo["mol"]
            p2f = finfo["parent_to_fragment"]
            f2p = finfo["fragment_to_parent"]

            if len(p2f) != len(f2p):
                issues.append(f"Fragment {fid}: p2f/f2p size mismatch")

            for p_idx, f_idx in p2f.items():
                if f2p.get(f_idx) != p_idx:
                    issues.append(f"Fragment {fid}: mapping inconsistency at p{p_idx}<->f{f_idx}")
                # element match already checked on register, but re-check
                if self.parent_mol.GetAtomWithIdx(p_idx).GetSymbol() != frag.GetAtomWithIdx(f_idx).GetSymbol():
                    issues.append(f"Fragment {fid}: element mismatch at p{p_idx}<->f{f_idx}")

        return issues

    def concordance_table(self) -> List[Dict[str, Any]]:
        """Convenient debug view: one row per parent atom."""
        rows: List[Dict[str, Any]] = []
        for idx in range(self.parent_mol.GetNumAtoms()):
            aid = self._parent_index_to_id[idx]
            at = self._atoms[aid]
            rows.append(
                {
                    "parent_index": idx,
                    "atom_id": aid,
                    "element": at.element,
                    "charge": at.formal_charge,
                    "aromatic": at.is_aromatic,
                    "hybridization": at.hybridization,
                    "fragment_index": at.fragment_index,
                    "roles": {r.value: c for r, c in at.roles.items()},
                }
            )
        return rows

    def warnings(self) -> List[str]:
        return list(self._warnings)


# -------- chemical semantics layer (Step 2) --------


class GroupRole(str, Enum):
    CARBONYL_C = "carbonyl_c"
    CARBONYL_O = "carbonyl_o"
    HETERO_ATTACH = "hetero_attach"
    LEAVING_GROUP = "leaving_group"
    HALOGEN = "halogen"
    ALLYLIC_C = "allylic_c"
    ARYL_C = "aryl_c"
    EPOXIDE_O = "epoxide_o"
    EPOXIDE_C1 = "epoxide_c1"
    EPOXIDE_C2 = "epoxide_c2"
    ALPHA_C = "alpha_c"
    BETA_C = "beta_c"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BondRef:
    a: AtomRef
    b: AtomRef
    bond_order: float
    is_aromatic: bool
    rdkit_bond_type: Optional[str] = None

    def pair_ids(self) -> Tuple[str, str]:
        return (self.a.atom_id, self.b.atom_id)

    def pair_indices(self) -> Tuple[int, int]:
        return (self.a.original_index, self.b.original_index)

    def validate_connected(self, mol: Chem.Mol) -> None:
        if mol is None:
            raise ValueError("validate_connected requires an RDKit mol")
        bond = mol.GetBondBetweenAtoms(self.a.original_index, self.b.original_index)
        if bond is None:
            raise ValueError(
                f"No bond between atoms {self.a.original_index} and {self.b.original_index}"
            )


@dataclass
class ChemicalGroup:
    group_type: str
    atoms: List[AtomRef]
    roles: Dict[GroupRole, AtomRef]
    bonds: List[BondRef] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def validate(self, mol: Optional[Chem.Mol] = None, spec: Optional["FunctionalGroupSpec"] = None) -> List[str]:
        return validate_group(mol, self, spec)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_type": self.group_type,
            "confidence": self.confidence,
            "roles": {role.value: ref.atom_id for role, ref in self.roles.items()},
            "atoms": [ref.atom_id for ref in self.atoms],
            "bonds": [bond.pair_ids() for bond in self.bonds],
            "properties": dict(self.properties),
            "evidence": dict(self.evidence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class FunctionalGroupSpec:
    name: str
    group_type: str
    smarts: str
    role_map: Dict[int, GroupRole]
    required_roles: Set[GroupRole]
    expected_bonds: List[Tuple[GroupRole, GroupRole, str]]
    min_confidence: float = 0.5


@dataclass
class GroupDetectionResult:
    groups: List[ChemicalGroup] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# -------- Step 3: Bond Resolution Engine --------


@dataclass(frozen=True)
class BondCandidate:
    atom_ids: Tuple[str, str]
    bond_order: float
    is_aromatic: bool
    roles: Optional[Tuple[GroupRole, GroupRole]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BondResolutionResult:
    selected: Optional[BondCandidate] = None
    candidates: List[BondCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    method: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.selected is not None and not self.errors


def _candidate_sort_key(atr: AtomicTruthRegistry, cand: BondCandidate) -> Tuple[int, int, int, int]:
    a_id, b_id = cand.atom_ids
    a = atr.get_by_uuid(a_id)
    b = atr.get_by_uuid(b_id)

    def _rank(atom: AtomTruth) -> int:
        return atom.canonical_rank if atom.canonical_rank is not None else 10**6

    pair = sorted(
        [(int(a.parent_index), _rank(a)), (int(b.parent_index), _rank(b))],
        key=lambda t: (t[1], t[0]),
    )
    return (pair[0][1], pair[0][0], pair[1][1], pair[1][0])


def resolve_bond_by_indices(
    mol: Chem.Mol, atr: AtomicTruthRegistry, indices: Tuple[int, int]
) -> BondResolutionResult:
    result = BondResolutionResult(method="indices")
    try:
        a_idx, b_idx = int(indices[0]), int(indices[1])
    except Exception as exc:
        result.errors.append(f"indices invalid: {exc}")
        return result

    bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
    if bond is None:
        result.errors.append(f"no bond between indices {a_idx}-{b_idx}")
        return result

    a_id = atr.atom_id_from_parent_index(a_idx)
    b_id = atr.atom_id_from_parent_index(b_idx)
    cand = BondCandidate(
        atom_ids=(a_id, b_id),
        bond_order=float(bond.GetBondTypeAsDouble()),
        is_aromatic=bool(_bond_is_aromatic_context(bond)),
        evidence={"method": "indices", "indices": (a_idx, b_idx)},
    )
    result.candidates.append(cand)
    result.selected = cand
    return result


def resolve_bond_by_smarts(
    mol: Chem.Mol,
    atr: AtomicTruthRegistry,
    smarts: str,
    bond_atom_indices: Tuple[int, int],
) -> BondResolutionResult:
    result = BondResolutionResult(method="smarts")
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        result.errors.append(f"invalid SMARTS: {smarts}")
        return result

    for match in mol.GetSubstructMatches(patt, uniquify=True):
        try:
            a_idx = int(match[bond_atom_indices[0]])
            b_idx = int(match[bond_atom_indices[1]])
        except Exception as exc:
            result.errors.append(f"bond_atom_indices invalid for match: {exc}")
            continue
        bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
        if bond is None:
            continue
        cand = BondCandidate(
            atom_ids=(
                atr.atom_id_from_parent_index(a_idx),
                atr.atom_id_from_parent_index(b_idx),
            ),
            bond_order=float(bond.GetBondTypeAsDouble()),
            is_aromatic=bool(_bond_is_aromatic_context(bond)),
            evidence={
                "method": "smarts",
                "smarts": smarts,
                "match_indices": match,
                "bond_atom_indices": bond_atom_indices,
            },
        )
        result.candidates.append(cand)

    if not result.candidates:
        result.errors.append("no SMARTS matches produced a valid bond")
        return result

    result.candidates.sort(key=lambda c: _candidate_sort_key(atr, c))
    result.selected = result.candidates[0]
    return result


def resolve_bond_by_roles(
    mol: Chem.Mol,
    atr: AtomicTruthRegistry,
    groups: Iterable[ChemicalGroup],
    role_a: GroupRole,
    role_b: GroupRole,
    *,
    group_type: Optional[str] = None,
) -> BondResolutionResult:
    result = BondResolutionResult(method="roles")
    for group in groups:
        if group_type and group.group_type != group_type:
            continue
        if role_a not in group.roles or role_b not in group.roles:
            continue
        a = group.roles[role_a]
        b = group.roles[role_b]
        bond = mol.GetBondBetweenAtoms(a.original_index, b.original_index)
        if bond is None:
            continue
        cand = BondCandidate(
            atom_ids=(a.atom_id, b.atom_id),
            bond_order=float(bond.GetBondTypeAsDouble()),
            is_aromatic=bool(_bond_is_aromatic_context(bond)),
            roles=(role_a, role_b),
            evidence={"method": "roles", "group_type": group.group_type},
        )
        result.candidates.append(cand)

    if not result.candidates:
        result.errors.append("no role-based candidates found")
        return result

    result.candidates.sort(key=lambda c: _candidate_sort_key(atr, c))
    result.selected = result.candidates[0]
    return result


def resolve_bond_by_element_pair(
    mol: Chem.Mol,
    atr: AtomicTruthRegistry,
    element_a: str,
    element_b: str,
) -> BondResolutionResult:
    """Resolve bond by element pair (e.g., C-F, C-Cl, C-H).

    For C-H bonds, explicit hydrogens are added since RDKit doesn't
    include them by default.
    """
    result = BondResolutionResult(method="element_pair")
    candidates = []

    # Check if we need explicit hydrogens (for C-H, N-H, O-H bonds)
    needs_explicit_h = "H" in (element_a, element_b)
    search_mol = mol
    h_atom_offset = 0

    if needs_explicit_h:
        # Add explicit hydrogens to find X-H bonds
        search_mol = Chem.AddHs(mol)
        h_atom_offset = mol.GetNumAtoms()  # Original atoms come first

    for bond in search_mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        a_sym = a.GetSymbol()
        b_sym = b.GetSymbol()
        if (a_sym == element_a and b_sym == element_b) or (a_sym == element_b and b_sym == element_a):
            a_idx = a.GetIdx()
            b_idx = b.GetIdx()

            # For H atoms added by AddHs, we need to handle ATR lookup differently
            if needs_explicit_h:
                # Get the non-H atom's ATR id
                if a_sym == "H":
                    heavy_idx = b_idx
                    h_idx = a_idx
                else:
                    heavy_idx = a_idx
                    h_idx = b_idx

                # Only the heavy atom exists in the original ATR
                if heavy_idx < mol.GetNumAtoms():
                    heavy_id = atr.atom_id_from_parent_index(heavy_idx)
                    if heavy_id:
                        # Create a synthetic H atom id based on heavy atom
                        h_id = f"{heavy_id}_H{h_idx}"
                        candidates.append(
                            BondCandidate(
                                atom_ids=(heavy_id, h_id) if a_sym != "H" else (h_id, heavy_id),
                                bond_order=bond.GetBondTypeAsDouble(),
                                is_aromatic=False,
                                roles=None,
                            )
                        )
            else:
                a_id = atr.atom_id_from_parent_index(a_idx)
                b_id = atr.atom_id_from_parent_index(b_idx)
                if a_id and b_id:
                    candidates.append(
                        BondCandidate(
                            atom_ids=(a_id, b_id),
                            bond_order=bond.GetBondTypeAsDouble(),
                            is_aromatic=bond.GetIsAromatic(),
                            roles=None,
                        )
                    )

    if not candidates:
        result.errors.append(f"No {element_a}-{element_b} bonds found.")
        return result
    result.candidates = candidates
    result.selected = candidates[0]
    return result


def resolve_bond(
    smiles_or_mol: Union[str, Chem.Mol],
    *,
    atr: Optional[AtomicTruthRegistry] = None,
    target_indices: Optional[Tuple[int, int]] = None,
    target_smarts: Optional[str] = None,
    target_smarts_bond: Optional[Tuple[int, int]] = None,
    target_group_type: Optional[str] = None,
    target_role_pair: Optional[Tuple[GroupRole, GroupRole]] = None,
    target_element_pair: Optional[Tuple[str, str]] = None,
    specs: Optional[List[FunctionalGroupSpec]] = None,
) -> BondResolutionResult:
    if isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
        if atr is None:
            atr = AtomicTruthRegistry(mol, parent_smiles=Chem.MolToSmiles(mol, canonical=True))
    else:
        prep = prepare_mol(smiles_or_mol)
        mol = prep.mol
        if mol is None:
            return BondResolutionResult(errors=[f"Invalid SMILES: {smiles_or_mol}"])
        if atr is None:
            parent_smiles = prep.canonical_smiles or Chem.MolToSmiles(mol, canonical=True)
            atr = AtomicTruthRegistry(mol, parent_smiles=parent_smiles, keep_hs=False)

    if target_indices is not None:
        return resolve_bond_by_indices(mol, atr, target_indices)

    if target_smarts and target_smarts_bond is not None:
        return resolve_bond_by_smarts(mol, atr, target_smarts, target_smarts_bond)

    if target_element_pair is not None:
        return resolve_bond_by_element_pair(mol, atr, target_element_pair[0], target_element_pair[1])

    if target_role_pair is not None:
        if specs is None:
            detection = detect_groups(mol, atr=atr)
            groups = detection.groups
        else:
            detection = detect_groups(mol, atr=atr, specs=specs)
            groups = detection.groups
        return resolve_bond_by_roles(
            mol,
            atr,
            groups,
            target_role_pair[0],
            target_role_pair[1],
            group_type=target_group_type,
        )

    return BondResolutionResult(errors=["no target specified for bond resolution"])


def _allowed_elements_for_role(role: GroupRole) -> Optional[Set[str]]:
    mapping: Dict[GroupRole, Set[str]] = {
        GroupRole.CARBONYL_C: {"C"},
        GroupRole.CARBONYL_O: {"O"},
        GroupRole.HETERO_ATTACH: {"O", "N", "S", "P"},
        GroupRole.LEAVING_GROUP: {"F", "Cl", "Br", "I", "O", "N", "S", "P"},
        GroupRole.HALOGEN: {"F", "Cl", "Br", "I"},
        GroupRole.EPOXIDE_O: {"O"},
        GroupRole.EPOXIDE_C1: {"C"},
        GroupRole.EPOXIDE_C2: {"C"},
        GroupRole.ALPHA_C: {"C"},
        GroupRole.BETA_C: {"C"},
        GroupRole.ALLYLIC_C: {"C"},
        GroupRole.ARYL_C: {"C"},
    }
    return mapping.get(role)


def _bond_is_aromatic_context(bond: Chem.Bond) -> bool:
    if bond.GetIsAromatic():
        return True
    return bond.GetBeginAtom().GetIsAromatic() or bond.GetEndAtom().GetIsAromatic()


def validate_required_roles(group: ChemicalGroup, required: Optional[Set[GroupRole]] = None) -> List[str]:
    req = required or set()
    missing = [role for role in req if role not in group.roles]
    if missing:
        return [f"{group.group_type}: missing required roles: {', '.join(r.value for r in missing)}"]
    return []


def validate_roles_elements(group: ChemicalGroup) -> List[str]:
    issues: List[str] = []
    for role, atom in group.roles.items():
        allowed = _allowed_elements_for_role(role)
        if allowed is None:
            continue
        if atom.element not in allowed:
            issues.append(
                f"{group.group_type}: role {role.value} has {atom.element}, expected {sorted(allowed)}"
            )
    return issues


def validate_roles_bond_orders(
    mol: Optional[Chem.Mol], group: ChemicalGroup, spec: Optional[FunctionalGroupSpec]
) -> List[str]:
    if mol is None or spec is None:
        return []
    issues: List[str] = []
    for role_a, role_b, expectation in spec.expected_bonds:
        if role_a not in group.roles or role_b not in group.roles:
            issues.append(
                f"{group.group_type}: expected bond {role_a.value}-{role_b.value} missing role"
            )
            continue
        a = group.roles[role_a]
        b = group.roles[role_b]
        bond = mol.GetBondBetweenAtoms(a.original_index, b.original_index)
        if bond is None:
            issues.append(
                f"{group.group_type}: expected bond {role_a.value}-{role_b.value} not found"
            )
            continue
        order = float(bond.GetBondTypeAsDouble())
        if expectation == "any":
            continue
        if expectation == "aromatic":
            if _bond_is_aromatic_context(bond):
                continue
            issues.append(
                f"{group.group_type}: expected aromatic bond between {role_a.value}-{role_b.value}"
            )
        elif expectation == "single":
            if bond.GetIsAromatic() or order >= 1.5:
                issues.append(
                    f"{group.group_type}: expected single bond between {role_a.value}-{role_b.value}"
                )
        elif expectation == "double":
            if bond.GetIsAromatic() or order < 1.5:
                issues.append(
                    f"{group.group_type}: expected double bond between {role_a.value}-{role_b.value}"
                )
    return issues


def validate_connected_subgraph(mol: Optional[Chem.Mol], group: ChemicalGroup) -> List[str]:
    if mol is None:
        return []
    if not group.atoms:
        return [f"{group.group_type}: group has no atoms"]
    idxs = {a.original_index for a in group.atoms}
    adj: Dict[int, Set[int]] = {idx: set() for idx in idxs}
    for bond in group.bonds:
        a_idx, b_idx = bond.pair_indices()
        if a_idx in idxs and b_idx in idxs:
            adj[a_idx].add(b_idx)
            adj[b_idx].add(a_idx)
    visited: Set[int] = set()
    stack = [next(iter(idxs))]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for nb in adj.get(cur, set()):
            if nb not in visited:
                stack.append(nb)
    if visited != idxs:
        return [f"{group.group_type}: group atoms not connected as a subgraph"]
    return []


def validate_group(
    mol: Optional[Chem.Mol],
    group: ChemicalGroup,
    spec: Optional[FunctionalGroupSpec] = None,
) -> List[str]:
    issues: List[str] = []
    issues.extend(validate_required_roles(group, spec.required_roles if spec else None))
    issues.extend(validate_roles_elements(group))
    issues.extend(validate_roles_bond_orders(mol, group, spec))
    issues.extend(validate_connected_subgraph(mol, group))
    return issues


def _detect_groups_for_spec_internal(
    mol: Chem.Mol, atr: AtomicTruthRegistry, spec: FunctionalGroupSpec
) -> Tuple[List[ChemicalGroup], List[str], List[str]]:
    groups: List[ChemicalGroup] = []
    warnings: List[str] = []
    errors: List[str] = []

    patt = Chem.MolFromSmarts(spec.smarts)
    if patt is None:
        errors.append(f"{spec.name}: invalid SMARTS '{spec.smarts}'")
        return groups, warnings, errors

    for match in mol.GetSubstructMatches(patt, uniquify=True):
        roles: Dict[GroupRole, AtomRef] = {}
        try:
            for smarts_idx, role in spec.role_map.items():
                if smarts_idx >= len(match):
                    raise ValueError(
                        f"{spec.name}: role_map index {smarts_idx} out of range for match"
                    )
                atom_idx = int(match[smarts_idx])
                atom_id = atr.atom_id_from_parent_index(atom_idx)
                roles[role] = AtomRef(atr, atom_id, source="smarts")
        except Exception as exc:
            errors.append(f"{spec.name}: role assignment failed ({exc})")
            continue

        atoms = [AtomRef(atr, atr.atom_id_from_parent_index(int(idx)), source="smarts") for idx in match]
        bonds: List[BondRef] = []
        for i in range(len(match)):
            for j in range(i + 1, len(match)):
                bond = mol.GetBondBetweenAtoms(int(match[i]), int(match[j]))
                if bond is None:
                    continue
                bonds.append(
                    BondRef(
                        a=atoms[i],
                        b=atoms[j],
                        bond_order=float(bond.GetBondTypeAsDouble()),
                        is_aromatic=bool(_bond_is_aromatic_context(bond)),
                        rdkit_bond_type=str(bond.GetBondType()),
                    )
                )

        group = ChemicalGroup(
            group_type=spec.group_type,
            atoms=atoms,
            roles=roles,
            bonds=bonds,
            confidence=0.8,
            evidence={
                "spec": spec.name,
                "smarts": spec.smarts,
                "match_atom_ids": tuple(a.atom_id for a in atoms),
                "method": "smarts",
            },
        )

        issues = validate_group(mol, group, spec)
        if issues:
            group.warnings.extend(issues)
            fatal = any("missing required roles" in msg or "expected bond" in msg for msg in issues)
            group.confidence = 0.0 if fatal else max(0.0, 0.8 - 0.1 * len(issues))

        if group.confidence >= spec.min_confidence:
            groups.append(group)
        else:
            warnings.append(
                f"{spec.name}: group discarded (confidence={group.confidence:.2f})"
            )
    return groups, warnings, errors


def detect_groups_for_spec(
    mol: Chem.Mol, atr: AtomicTruthRegistry, spec: FunctionalGroupSpec
) -> List[ChemicalGroup]:
    groups, _, _ = _detect_groups_for_spec_internal(mol, atr, spec)
    return groups


def detect_groups(
    smiles_or_mol: Union[str, Chem.Mol],
    atr: Optional[AtomicTruthRegistry] = None,
    specs: Optional[List[FunctionalGroupSpec]] = None,
) -> GroupDetectionResult:
    warnings: List[str] = []
    errors: List[str] = []

    if isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
        if atr is None:
            parent_smiles = Chem.MolToSmiles(mol, canonical=True)
            atr = AtomicTruthRegistry(mol, parent_smiles=parent_smiles, keep_hs=False)
    else:
        prep = prepare_mol(smiles_or_mol)
        mol = prep.mol
        if mol is None:
            return GroupDetectionResult(groups=[], warnings=[], errors=[f"Invalid SMILES: {smiles_or_mol}"])
        if atr is None:
            parent_smiles = prep.canonical_smiles or Chem.MolToSmiles(mol, canonical=True)
            atr = AtomicTruthRegistry(mol, parent_smiles=parent_smiles, keep_hs=False)

    if specs is None:
        specs = [
            FunctionalGroupSpec(
                name="ester",
                group_type="ester",
                smarts="[CX3](=O)[OX2][#6]",
                role_map={0: GroupRole.CARBONYL_C, 1: GroupRole.CARBONYL_O, 2: GroupRole.HETERO_ATTACH},
                required_roles={GroupRole.CARBONYL_C, GroupRole.CARBONYL_O, GroupRole.HETERO_ATTACH},
                expected_bonds=[
                    (GroupRole.CARBONYL_C, GroupRole.CARBONYL_O, "double"),
                    (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH, "single"),
                ],
                min_confidence=0.5,
            ),
            FunctionalGroupSpec(
                name="amide",
                group_type="amide",
                smarts="[CX3](=O)[NX3]",
                role_map={0: GroupRole.CARBONYL_C, 1: GroupRole.CARBONYL_O, 2: GroupRole.HETERO_ATTACH},
                required_roles={GroupRole.CARBONYL_C, GroupRole.CARBONYL_O, GroupRole.HETERO_ATTACH},
                expected_bonds=[
                    (GroupRole.CARBONYL_C, GroupRole.CARBONYL_O, "double"),
                    (GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH, "single"),
                ],
                min_confidence=0.5,
            ),
            FunctionalGroupSpec(
                name="aryl_halide",
                group_type="aryl_halide",
                smarts="[c][F,Cl,Br,I]",
                role_map={0: GroupRole.ARYL_C, 1: GroupRole.HALOGEN},
                required_roles={GroupRole.HALOGEN},
                expected_bonds=[(GroupRole.ARYL_C, GroupRole.HALOGEN, "aromatic")],
                min_confidence=0.5,
            ),
            FunctionalGroupSpec(
                name="epoxide",
                group_type="epoxide",
                smarts="[OX2]1[CX4][CX4]1",
                role_map={0: GroupRole.EPOXIDE_O, 1: GroupRole.EPOXIDE_C1, 2: GroupRole.EPOXIDE_C2},
                required_roles={GroupRole.EPOXIDE_O, GroupRole.EPOXIDE_C1, GroupRole.EPOXIDE_C2},
                expected_bonds=[
                    (GroupRole.EPOXIDE_O, GroupRole.EPOXIDE_C1, "single"),
                    (GroupRole.EPOXIDE_O, GroupRole.EPOXIDE_C2, "single"),
                    (GroupRole.EPOXIDE_C1, GroupRole.EPOXIDE_C2, "single"),
                ],
                min_confidence=0.5,
            ),
        ]

    all_groups: List[ChemicalGroup] = []
    for spec in specs:
        spec_groups, spec_warnings, spec_errors = _detect_groups_for_spec_internal(
            mol, atr, spec
        )
        all_groups.extend(spec_groups)
        warnings.extend(spec_warnings)
        errors.extend(spec_errors)

    return GroupDetectionResult(groups=all_groups, warnings=warnings, errors=errors)


# ---- small helper for tests / demos ----

def make_fragment_via_atom_indices(parent_mol: Chem.Mol, atom_indices: List[int]) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Create a fragment submol containing a set of parent atom indices.

    Returns (fragment_mol, parent_to_fragment_map).

    This is not the final SmartFragmentBuilder; it's just to test ATR.
    """

    atom_indices_sorted = sorted(set(atom_indices))
    em = Chem.EditableMol(Chem.Mol())

    p_to_f: Dict[int, int] = {}
    for p_idx in atom_indices_sorted:
        a = parent_mol.GetAtomWithIdx(p_idx)
        f_idx = em.AddAtom(Chem.Atom(a.GetAtomicNum()))
        p_to_f[p_idx] = f_idx

    # add bonds within the selected set
    selected = set(atom_indices_sorted)
    for b in parent_mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in selected and a2 in selected:
            em.AddBond(p_to_f[a1], p_to_f[a2], b.GetBondType())

    frag = em.GetMol()
    Chem.SanitizeMol(frag)
    return frag, p_to_f
