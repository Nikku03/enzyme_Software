from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from contextlib import nullcontext

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg
import math

from enzyme_software.modules.sre_atr import (
    AtomicTruthRegistry,
    ChemicalGroup,
    GroupRole,
)


class FragmentBuildError(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class CapStrategy(str, Enum):
    HYDROGEN = "hydrogen"
    METHYL = "methyl"
    GENERIC_R_GROUP = "generic_r_group"


@dataclass(frozen=True)
class CutBond:
    parent_bond_idx: int
    parent_atom_indices: Tuple[int, int]
    parent_atom_uuids: Tuple[str, str]
    bond_order: float
    is_aromatic: bool


@dataclass(frozen=True)
class CapRecord:
    cut_bond: CutBond
    kept_atom_uuid: str
    removed_atom_uuid: str
    strategy: CapStrategy
    cap_atom_symbol: str
    notes: str


@dataclass
class Fragment:
    frag_mol: Chem.Mol
    frag_smiles: str
    parent_smiles: str
    mapped_smiles: Optional[str]
    parent_uuid_to_frag_idx: Dict[str, int]
    frag_idx_to_parent_uuid: Dict[int, str]
    kept_parent_uuids: Set[str]
    cut_bonds: List[CutBond]
    cap_records: List[CapRecord]
    warnings: List[str] = field(default_factory=list)
    build_notes: List[str] = field(default_factory=list)
    context_metrics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    fragment_3d: Optional["Fragment3D"] = None


@dataclass
class GeometryMetrics:
    bond_lengths: Dict[str, float] = field(default_factory=dict)
    bond_angles_deg: Dict[str, float] = field(default_factory=dict)
    clash_count: int = 0


@dataclass
class Fragment3D:
    mol_2d: Chem.Mol
    role_to_frag_idx: Dict[str, int]
    warnings: List[str] = field(default_factory=list)

    mol_3d: Optional[Chem.Mol] = None
    conformer_id: Optional[int] = None
    validation: Dict[str, bool] = field(default_factory=dict)
    metrics: GeometryMetrics = field(default_factory=GeometryMetrics)

    def build(
        self,
        n_conformers: int = 10,
        seed: int = 0xC0FFEE,
        max_iters: int = 500,
    ) -> "Fragment3D":
        self.mol_3d, self.conformer_id = self._generate_3d_conformer(
            self.mol_2d, n_conformers=n_conformers, seed=seed, max_iters=max_iters
        )
        if self.mol_3d is None:
            self.validation["has_3d"] = False
            return self

        self.validation["has_3d"] = True
        self.validation.update(self._validate_3d_structure(self.mol_3d, self.conformer_id))
        self.metrics = self._compute_geometric_metrics(self.mol_3d, self.conformer_id)
        return self

    def _generate_3d_conformer(
        self,
        mol_2d: Chem.Mol,
        n_conformers: int,
        seed: int,
        max_iters: int,
    ) -> Tuple[Optional[Chem.Mol], Optional[int]]:
        try:
            mol = Chem.AddHs(mol_2d)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed)
            params.pruneRmsThresh = 0.25
            params.useSmallRingTorsions = True
            params.useBasicKnowledge = True

            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
            if not conf_ids:
                self.warnings.append("3d_embed_failed: no conformers generated")
                return None, None

            mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if mmff_props is not None:
                # RDKit expects MMFFOptimizeMoleculeConfs(mol, ...) without props.
                results = AllChem.MMFFOptimizeMoleculeConfs(
                    mol, maxIters=max_iters, mmffVariant="MMFF94s"
                )
                best = min(
                    [(i, e) for i, (st, e) in enumerate(results) if st == 0],
                    key=lambda x: x[1],
                    default=None,
                )
                if best is None:
                    self.warnings.append("mmff_opt_failed: using first conformer")
                    return mol, int(conf_ids[0])
                return mol, int(conf_ids[best[0]])
            else:
                results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
                best = min([(i, e) for i, (st, e) in enumerate(results) if st == 0], key=lambda x: x[1], default=None)
                if best is None:
                    self.warnings.append("uff_opt_failed: using first conformer")
                    return mol, int(conf_ids[0])
                return mol, int(conf_ids[best[0]])
        except Exception as e:
            self.warnings.append(f"3d_exception: {type(e).__name__}: {e}")
            return None, None

    def _validate_3d_structure(self, mol_3d: Chem.Mol, conf_id: int) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        checks["roles_present"] = all(r in self.role_to_frag_idx for r in self.role_to_frag_idx.keys())

        if "carbonyl_c" in self.role_to_frag_idx and "carbonyl_o" in self.role_to_frag_idx:
            c = self.role_to_frag_idx["carbonyl_c"]
            o = self.role_to_frag_idx["carbonyl_o"]
            bond = mol_3d.GetBondBetweenAtoms(int(c), int(o))
            checks["carbonyl_bond_exists"] = bond is not None
        else:
            checks["carbonyl_bond_exists"] = True

        checks["no_hard_clashes"] = (self._count_clashes(mol_3d, conf_id) < 3)
        return checks

    def _compute_geometric_metrics(self, mol_3d: Chem.Mol, conf_id: int) -> GeometryMetrics:
        m = GeometryMetrics()
        conf = mol_3d.GetConformer(int(conf_id))

        def bond_len(name: str, a_role: str, b_role: str):
            if a_role in self.role_to_frag_idx and b_role in self.role_to_frag_idx:
                a = int(self.role_to_frag_idx[a_role])
                b = int(self.role_to_frag_idx[b_role])
                m.bond_lengths[name] = float(GetBondLength(conf, a, b))

        bond_len("carbonyl_C_O", "carbonyl_c", "carbonyl_o")
        bond_len("acyl_C_hetero", "carbonyl_c", "hetero_attach")

        if all(k in self.role_to_frag_idx for k in ("carbonyl_o", "carbonyl_c", "hetero_attach")):
            o = int(self.role_to_frag_idx["carbonyl_o"])
            c = int(self.role_to_frag_idx["carbonyl_c"])
            x = int(self.role_to_frag_idx["hetero_attach"])
            m.bond_angles_deg["O_C_X"] = float(GetAngleDeg(conf, o, c, x))

        m.clash_count = self._count_clashes(mol_3d, conf_id)
        return m

    def _count_clashes(self, mol_3d: Chem.Mol, conf_id: int, thresh: float = 0.75) -> int:
        conf = mol_3d.GetConformer(int(conf_id))
        heavy = [a.GetIdx() for a in mol_3d.GetAtoms() if a.GetAtomicNum() > 1]
        clash = 0
        for i in range(len(heavy)):
            pi = conf.GetAtomPosition(heavy[i])
            for j in range(i + 1, len(heavy)):
                pj = conf.GetAtomPosition(heavy[j])
                d = (pi - pj).Length()
                if d < thresh:
                    clash += 1
        return clash


class ChemicallyAwareFragmentBuilder:
    def __init__(self, *, max_heavy_atoms: int = 40, default_radius: int = 2):
        self.max_heavy_atoms = max_heavy_atoms
        self.default_radius = default_radius
        self.small_ring_max = 4

    def build_from_group(
        self,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        group: ChemicalGroup,
        *,
        pH: Optional[float] = None,
        radius: Optional[int] = None,
        cap_strategy: CapStrategy = CapStrategy.HYDROGEN,
        keep_conjugation: bool = True,
        keep_rings: bool = True,
        keep_hbond_partners: bool = True,
        keep_charged_neighbors: bool = True,
    ) -> Fragment:
        core = {a.atom_id for a in group.atoms}
        for role_atom in group.roles.values():
            core.add(role_atom.atom_id)
        role_uuid_map = {role.value: atom.atom_id for role, atom in group.roles.items()}
        return self._build_fragment(
            atr=atr,
            parent_mol=parent_mol,
            core_atom_uuids=core,
            group_type=group.group_type,
            pH=pH,
            radius=radius,
            cap_strategy=cap_strategy,
            keep_conjugation=keep_conjugation,
            keep_rings=keep_rings,
            keep_hbond_partners=keep_hbond_partners,
            keep_charged_neighbors=keep_charged_neighbors,
            role_uuid_map=role_uuid_map,
        )

    def build_from_bond(
        self,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        bond_atom_uuids: Tuple[str, str],
        *,
        radius: Optional[int] = None,
        cap_strategy: CapStrategy = CapStrategy.HYDROGEN,
        keep_conjugation: bool = True,
        keep_rings: bool = True,
        keep_hbond_partners: bool = True,
        keep_charged_neighbors: bool = True,
    ) -> Fragment:
        core = {bond_atom_uuids[0], bond_atom_uuids[1]}
        return self._build_fragment(
            atr=atr,
            parent_mol=parent_mol,
            core_atom_uuids=core,
            group_type="bond",
            radius=radius,
            cap_strategy=cap_strategy,
            keep_conjugation=keep_conjugation,
            keep_rings=keep_rings,
            keep_hbond_partners=keep_hbond_partners,
            keep_charged_neighbors=keep_charged_neighbors,
            role_uuid_map={},
        )

    def _build_fragment(
        self,
        *,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        core_atom_uuids: Set[str],
        group_type: str,
        pH: Optional[float] = None,
        radius: Optional[int] = None,
        cap_strategy: CapStrategy = CapStrategy.HYDROGEN,
        keep_conjugation: bool = True,
        keep_rings: bool = True,
        keep_hbond_partners: bool = True,
        keep_charged_neighbors: bool = True,
        role_uuid_map: Optional[Dict[str, str]] = None,
    ) -> Fragment:
        if parent_mol is None:
            raise FragmentBuildError("parent_mol is None")
        if not core_atom_uuids:
            raise FragmentBuildError("core_atom_uuids is empty")

        base_radius = self.default_radius if radius is None else int(radius)
        attempt_radius = [base_radius, base_radius + 1, base_radius + 2]
        last_error: Optional[Dict[str, Any]] = None

        ring_atom_sets, ring_bond_sets = self._compute_ring_sets(parent_mol)

        for radius_try in attempt_radius:
            try:
                kept_indices, protected_bonds, provenance, warnings = self._select_atoms(
                    atr=atr,
                    parent_mol=parent_mol,
                    core_atom_uuids=core_atom_uuids,
                    group_type=group_type,
                    radius=radius_try,
                    keep_conjugation=keep_conjugation,
                    keep_rings=keep_rings,
                    keep_hbond_partners=keep_hbond_partners,
                    keep_charged_neighbors=keep_charged_neighbors,
                    ring_atom_sets=ring_atom_sets,
                    ring_bond_sets=ring_bond_sets,
                )
                cut_bonds = self._identify_cut_bonds(
                    atr,
                    parent_mol,
                    kept_indices,
                    protected_bonds,
                )
                frag_mol, mapping = self._build_submol(
                    atr, parent_mol, kept_indices
                )
                cap_records = self._apply_caps(
                    atr, parent_mol, frag_mol, mapping, cut_bonds, cap_strategy
                )
                warnings.extend(
                    self._validate_fragment(
                        frag_mol,
                        mapping,
                        atr,
                        parent_mol,
                        kept_indices,
                        cap_strategy,
                        expects_ring=bool(protected_bonds),
                    )
                )
                frag_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
                parent_smiles = atr.parent_smiles or Chem.MolToSmiles(parent_mol, canonical=True)
                mapped_smiles = None
                try:
                    mapped_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
                except Exception:
                    mapped_smiles = None

                build_notes: List[str] = []
                if not cut_bonds:
                    build_notes.append(
                        "no_capping_needed: selected subgraph had no external bonds"
                    )

                context_metrics = self._compute_context_metrics(
                    atr,
                    parent_mol,
                    frag_mol,
                    kept_indices,
                    cut_bonds,
                    cap_records,
                )
                role_to_frag_idx: Dict[str, int] = {}
                if role_uuid_map:
                    for role_name, uuid in role_uuid_map.items():
                        if uuid in mapping["parent_uuid_to_frag_idx"]:
                            role_to_frag_idx[role_name] = mapping["parent_uuid_to_frag_idx"][uuid]
                fragment_3d = Fragment3D(
                    mol_2d=frag_mol,
                    role_to_frag_idx=role_to_frag_idx,
                ).build()
                frag = Fragment(
                    frag_mol=frag_mol,
                    frag_smiles=frag_smiles,
                    parent_smiles=parent_smiles,
                    mapped_smiles=mapped_smiles,
                    parent_uuid_to_frag_idx=mapping["parent_uuid_to_frag_idx"],
                    frag_idx_to_parent_uuid=mapping["frag_idx_to_parent_uuid"],
                    kept_parent_uuids=set(mapping["parent_uuid_to_frag_idx"].keys()),
                    cut_bonds=cut_bonds,
                    cap_records=cap_records,
                    warnings=warnings,
                    build_notes=build_notes,
                    context_metrics=context_metrics,
                    provenance=provenance,
                    fragment_3d=fragment_3d,
                )
                return frag
            except FragmentBuildError as exc:
                last_error = {"message": str(exc), "details": exc.details}
                continue

        raise FragmentBuildError(
            "fragment build failed after retries",
            details=last_error or {},
        )

    def _select_atoms(
        self,
        *,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        core_atom_uuids: Set[str],
        group_type: str,
        radius: int,
        keep_conjugation: bool,
        keep_rings: bool,
        keep_hbond_partners: bool,
        keep_charged_neighbors: bool,
        ring_atom_sets: List[Set[int]],
        ring_bond_sets: List[Set[int]],
    ) -> Tuple[Set[int], Set[int], Dict[str, Any], List[str]]:
        warnings: List[str] = []
        provenance: Dict[str, Any] = {
            "group_type": group_type,
            "radius": radius,
            "rules": [],
        }

        core_indices: Set[int] = set()
        for uid in core_atom_uuids:
            idx = self._resolve_parent_index(atr, parent_mol, uid)
            if idx is None:
                warnings.append(f"core_uuid_unresolved: {uid}")
                continue
            core_indices.add(idx)
        kept = set(core_indices)
        protected_bonds: Set[int] = set()

        if keep_rings:
            ring_added = self._expand_rings(parent_mol, kept, ring_atom_sets)
            if ring_added:
                provenance["rules"].append("ring_preservation")
            small_ring_added, protected_bonds = self._protect_small_rings(
                parent_mol, kept, ring_atom_sets, ring_bond_sets
            )
            if small_ring_added:
                provenance["rules"].append("small_ring_preservation")

        if keep_conjugation:
            conj_added = self._expand_conjugation(parent_mol, kept)
            if conj_added:
                provenance["rules"].append("conjugation")

        if keep_charged_neighbors:
            charged_added = self._expand_charged_neighbors(parent_mol, kept, core_indices)
            if charged_added:
                provenance["rules"].append("charged_neighbors")
            acid_added = self._expand_nearby_carboxyl(parent_mol, kept, core_indices)
            if acid_added:
                provenance["rules"].append("carboxyl_neighbors")

        if keep_hbond_partners:
            hbond_added = self._expand_hbond_partners(parent_mol, kept, core_indices)
            if hbond_added:
                provenance["rules"].append("hbond_partners")

        radius_added = self._expand_by_radius(parent_mol, kept, kept, radius)
        if radius_added:
            provenance["rules"].append("radius")

        ideal_kept = set(kept)
        protected_ring_atoms = self._protected_ring_atoms(
            ring_atom_sets, ring_bond_sets, protected_bonds
        )
        kept, truncation_warning = self._truncate_by_max(
            parent_mol,
            ideal_kept,
            core_indices,
            protected_ring_atoms=protected_ring_atoms,
        )
        if truncation_warning:
            warnings.append(truncation_warning)
            provenance["rules"].append("truncation")

        return kept, protected_bonds, provenance, warnings

    def _compute_ring_sets(
        self, mol: Chem.Mol
    ) -> Tuple[List[Set[int]], List[Set[int]]]:
        ring_atom_sets: List[Set[int]] = []
        ring_bond_sets: List[Set[int]] = []
        for ring in mol.GetRingInfo().AtomRings():
            ring_set = set(ring)
            ring_atom_sets.append(ring_set)
            bond_set: Set[int] = set()
            for i in range(len(ring)):
                a = ring[i]
                b = ring[(i + 1) % len(ring)]
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is not None:
                    bond_set.add(int(bond.GetIdx()))
            ring_bond_sets.append(bond_set)
        return ring_atom_sets, ring_bond_sets

    def _resolve_parent_index(
        self, atr: AtomicTruthRegistry, mol: Chem.Mol, atom_id: str
    ) -> Optional[int]:
        """Resolve ATR UUID to parent index, with implicit-H tolerance."""
        try:
            return atr.parent_index_from_atom_id(atom_id)
        except Exception:
            pass
        if "_H" in atom_id:
            base_uuid, suffix = atom_id.rsplit("_H", 1)
            try:
                parent_idx = atr.parent_index_from_atom_id(base_uuid)
            except Exception:
                return None
            try:
                h_index = int(suffix)
            except ValueError:
                return None
            parent_atom = mol.GetAtomWithIdx(int(parent_idx))
            count = 0
            for neighbor in parent_atom.GetNeighbors():
                if neighbor.GetSymbol() == "H":
                    if count == h_index:
                        return neighbor.GetIdx()
                    count += 1
            return None
        return None

    def _expand_rings(
        self, mol: Chem.Mol, kept: Set[int], ring_atom_sets: List[Set[int]]
    ) -> bool:
        added = False
        changed = True
        while changed:
            changed = False
            for ring_set in ring_atom_sets:
                if not ring_set & kept:
                    continue
                new_atoms = ring_set - kept
                if new_atoms:
                    kept.update(new_atoms)
                    added = True
                    changed = True
        return added

    def _protect_small_rings(
        self,
        mol: Chem.Mol,
        kept: Set[int],
        ring_atom_sets: List[Set[int]],
        ring_bond_sets: List[Set[int]],
    ) -> Tuple[bool, Set[int]]:
        added = False
        protected_bonds: Set[int] = set()
        changed = True
        while changed:
            changed = False
            for ring_set, bond_set in zip(ring_atom_sets, ring_bond_sets):
                if len(ring_set) > self.small_ring_max:
                    continue
                if ring_set & kept:
                    new_atoms = ring_set - kept
                    if new_atoms:
                        kept.update(new_atoms)
                        added = True
                        changed = True
                    protected_bonds.update(bond_set)
        return added, protected_bonds

    def _protected_ring_atoms(
        self,
        ring_atom_sets: List[Set[int]],
        ring_bond_sets: List[Set[int]],
        protected_bonds: Set[int],
    ) -> Set[int]:
        if not protected_bonds:
            return set()
        protected_atoms: Set[int] = set()
        for ring_set, bond_set in zip(ring_atom_sets, ring_bond_sets):
            if bond_set & protected_bonds:
                protected_atoms.update(ring_set)
        return protected_atoms

    def _expand_conjugation(self, mol: Chem.Mol, kept: Set[int]) -> bool:
        added = False
        for idx in list(kept):
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() >= 1.5:
                    nb = bond.GetOtherAtomIdx(idx)
                    if nb not in kept:
                        kept.add(nb)
                        added = True
            if self._is_carbonyl_carbon(atom):
                for bond in atom.GetBonds():
                    nb = bond.GetOtherAtomIdx(idx)
                    if nb not in kept:
                        kept.add(nb)
                        added = True
        return added

    def _expand_charged_neighbors(
        self, mol: Chem.Mol, kept: Set[int], core: Set[int]
    ) -> bool:
        added = False
        for idx in list(kept):
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetFormalCharge() != 0:
                for nb in atom.GetNeighbors():
                    n_idx = nb.GetIdx()
                    if n_idx not in kept:
                        kept.add(n_idx)
                        added = True
        return added

    def _expand_nearby_carboxyl(
        self, mol: Chem.Mol, kept: Set[int], core: Set[int]
    ) -> bool:
        added = False
        patt = Chem.MolFromSmarts("[CX3](=O)[OX1,OX2H,OX2-]")
        if patt is None:
            return False
        dmat = Chem.GetDistanceMatrix(mol)
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            if any(min(dmat[i][c] for c in core) <= 2 for i in match):
                for idx in match:
                    if idx not in kept:
                        kept.add(idx)
                        added = True
        return added

    def _expand_hbond_partners(
        self, mol: Chem.Mol, kept: Set[int], core: Set[int]
    ) -> bool:
        added = False
        dmat = Chem.GetDistanceMatrix(mol)
        for idx in range(mol.GetNumAtoms()):
            if idx in kept:
                continue
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() not in {"O", "N", "S"}:
                continue
            if any(dmat[idx][c] <= 2 for c in core):
                kept.add(idx)
                added = True
        return added

    def _expand_by_radius(
        self, mol: Chem.Mol, kept: Set[int], seed: Set[int], radius: int
    ) -> bool:
        if radius <= 0:
            return False
        added = False
        dmat = Chem.GetDistanceMatrix(mol)
        for idx in range(mol.GetNumAtoms()):
            if idx in kept:
                continue
            if any(dmat[idx][c] <= radius for c in seed):
                kept.add(idx)
                added = True
        return added

    def _truncate_by_max(
        self,
        mol: Chem.Mol,
        ideal_kept: Set[int],
        core: Set[int],
        *,
        protected_ring_atoms: Set[int],
    ) -> Tuple[Set[int], Optional[str]]:
        def _heavy_count(indices: Set[int]) -> int:
            return sum(1 for i in indices if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

        ideal_heavy = _heavy_count(ideal_kept)
        if ideal_heavy <= self.max_heavy_atoms:
            return ideal_kept, None

        must_keep = set(core) | set(protected_ring_atoms)
        core_heavy = {
            i for i in must_keep if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        }
        core_count = len(core_heavy)
        if core_count > self.max_heavy_atoms:
            warning = (
                f"truncation: core heavy atoms ({core_count}) exceed max_heavy_atoms={self.max_heavy_atoms}"
            )
            return must_keep, warning

        dmat = Chem.GetDistanceMatrix(mol)
        distances = []
        for idx in ideal_kept:
            if idx in must_keep:
                continue
            min_dist = min(dmat[idx][c] for c in core)
            distances.append((min_dist, -self._importance_score(mol, idx), idx))
        distances.sort()

        new_kept = set(must_keep)
        for _, _, idx in distances:
            if mol.GetAtomWithIdx(idx).GetAtomicNum() <= 1:
                new_kept.add(idx)
                continue
            if _heavy_count(new_kept) >= self.max_heavy_atoms:
                break
            new_kept.add(idx)

        warning = (
            f"truncation: ideal selection {ideal_heavy} heavy atoms > "
            f"max_heavy_atoms={self.max_heavy_atoms}; kept {_heavy_count(new_kept)}"
        )
        return new_kept, warning

    def _importance_score(self, mol: Chem.Mol, idx: int) -> int:
        atom = mol.GetAtomWithIdx(idx)
        score = 0
        if atom.GetFormalCharge() != 0:
            score += 3
        if atom.GetSymbol() in {"O", "N", "S", "P"}:
            score += 2
        if atom.GetIsAromatic():
            score += 1
        return score

    def _identify_cut_bonds(
        self,
        atr: AtomicTruthRegistry,
        mol: Chem.Mol,
        kept_indices: Set[int],
        protected_bonds: Set[int],
    ) -> List[CutBond]:
        max_iters = 8
        iter_count = 0
        while iter_count < max_iters:
            iter_count += 1
            changed = False
            cut_bonds: List[CutBond] = []

            for bond in mol.GetBonds():
                a = bond.GetBeginAtomIdx()
                b = bond.GetEndAtomIdx()
                in_kept = a in kept_indices
                out_kept = b in kept_indices
                if in_kept == out_kept:
                    continue
                bond_idx = int(bond.GetIdx())
                if bond_idx in protected_bonds:
                    kept_indices.add(a)
                    kept_indices.add(b)
                    changed = True
                    continue
                cut_bonds.append(
                    CutBond(
                        parent_bond_idx=bond_idx,
                        parent_atom_indices=(a, b),
                        parent_atom_uuids=(
                            atr.atom_id_from_parent_index(a),
                            atr.atom_id_from_parent_index(b),
                        ),
                        bond_order=float(bond.GetBondTypeAsDouble()),
                        is_aromatic=bool(bond.GetIsAromatic()),
                    )
                )

            if not changed:
                return cut_bonds

        raise FragmentBuildError(
            "protected ring expansion did not converge",
            details={"kept_count": len(kept_indices)},
        )

    def _build_submol(
        self,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        kept_indices: Set[int],
    ) -> Tuple[Chem.Mol, Dict[str, Dict[Any, Any]]]:
        if not kept_indices:
            raise FragmentBuildError("kept_indices is empty")

        rw = Chem.RWMol()
        parent_to_frag: Dict[int, int] = {}
        parent_uuid_to_frag_idx: Dict[str, int] = {}
        frag_idx_to_parent_uuid: Dict[int, str] = {}

        for p_idx in sorted(kept_indices):
            atom = parent_mol.GetAtomWithIdx(p_idx)
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            new_atom.SetHybridization(atom.GetHybridization())
            f_idx = rw.AddAtom(new_atom)
            parent_to_frag[p_idx] = f_idx
            atom_uuid = atr.atom_id_from_parent_index(p_idx)
            rw.GetAtomWithIdx(f_idx).SetProp("parent_uuid", atom_uuid)
            parent_uuid_to_frag_idx[atom_uuid] = f_idx
            frag_idx_to_parent_uuid[f_idx] = atom_uuid

        for bond in parent_mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a not in kept_indices or b not in kept_indices:
                continue
            bond_type = bond.GetBondType()
            rw.AddBond(parent_to_frag[a], parent_to_frag[b], bond_type)
            if bond.GetIsAromatic():
                rbond = rw.GetBondBetweenAtoms(parent_to_frag[a], parent_to_frag[b])
                if rbond is not None:
                    rbond.SetIsAromatic(True)

        frag = rw.GetMol()
        return frag, {
            "parent_uuid_to_frag_idx": parent_uuid_to_frag_idx,
            "frag_idx_to_parent_uuid": frag_idx_to_parent_uuid,
        }

    def _apply_caps(
        self,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        frag_mol: Chem.Mol,
        mapping: Dict[str, Dict[Any, Any]],
        cut_bonds: List[CutBond],
        cap_strategy: CapStrategy,
    ) -> List[CapRecord]:
        if cap_strategy == CapStrategy.GENERIC_R_GROUP:
            cap_symbol = "*"
        cap_records: List[CapRecord] = []
        rw = Chem.RWMol(frag_mol)
        uuid_to_frag = mapping["parent_uuid_to_frag_idx"]

        for cut in cut_bonds:
            a_uuid, b_uuid = cut.parent_atom_uuids
            a_idx, b_idx = cut.parent_atom_indices
            if a_uuid in uuid_to_frag and b_uuid in uuid_to_frag:
                continue
            if a_uuid in uuid_to_frag:
                kept_uuid = a_uuid
                removed_uuid = b_uuid
            else:
                kept_uuid = b_uuid
                removed_uuid = a_uuid

            kept_parent_idx = atr.parent_index_from_atom_id(kept_uuid)
            kept_atom = parent_mol.GetAtomWithIdx(kept_parent_idx)

            strategy = cap_strategy
            cap_atom_symbol = "H"
            if cap_strategy == CapStrategy.METHYL:
                if kept_atom.GetIsAromatic() or kept_atom.GetHybridization() in (
                    Chem.HybridizationType.SP2,
                    Chem.HybridizationType.SP,
                ):
                    cap_atom_symbol = "H"
                    strategy = CapStrategy.HYDROGEN
                else:
                    cap_atom_symbol = "C"

            if cap_strategy == CapStrategy.GENERIC_R_GROUP:
                cap_atom_symbol = "*"

            kept_frag_idx = uuid_to_frag[kept_uuid]
            cap_atom = Chem.Atom(cap_atom_symbol)
            cap_idx = rw.AddAtom(cap_atom)
            rw.AddBond(kept_frag_idx, cap_idx, Chem.BondType.SINGLE)
            cap_records.append(
                CapRecord(
                    cut_bond=cut,
                    kept_atom_uuid=kept_uuid,
                    removed_atom_uuid=removed_uuid,
                    strategy=strategy,
                    cap_atom_symbol=cap_atom_symbol,
                    notes="cap added",
                )
            )

        frag = rw.GetMol()
        block_logs = getattr(rdBase, "BlockLogs", None)
        log_context = block_logs() if callable(block_logs) else nullcontext()
        with log_context:
            try:
                Chem.SanitizeMol(frag)
            except Exception as exc:
                raise FragmentBuildError("fragment sanitization failed after capping", details={"error": str(exc)})

        return cap_records

    def _validate_fragment(
        self,
        frag_mol: Chem.Mol,
        mapping: Dict[str, Dict[Any, Any]],
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        kept_indices: Set[int],
        cap_strategy: CapStrategy,
        expects_ring: bool,
    ) -> List[str]:
        warnings: List[str] = []
        uuid_to_frag = mapping["parent_uuid_to_frag_idx"]
        frag_to_uuid = mapping["frag_idx_to_parent_uuid"]
        block_logs = getattr(rdBase, "BlockLogs", None)
        log_context = block_logs() if callable(block_logs) else nullcontext()

        if len(uuid_to_frag) != len(frag_to_uuid):
            warnings.append("mapping is not bijective")

        for idx in range(frag_mol.GetNumAtoms()):
            atom = frag_mol.GetAtomWithIdx(idx)
            if atom.HasProp("parent_uuid") and atom.GetProp("parent_uuid") not in uuid_to_frag:
                warnings.append(f"fragment atom {idx} has unknown parent_uuid")

        if cap_strategy != CapStrategy.GENERIC_R_GROUP:
            if any(atom.GetAtomicNum() == 0 for atom in frag_mol.GetAtoms()):
                warnings.append("fragment contains dummy atoms")

        with log_context:
            try:
                Chem.SanitizeMol(frag_mol)
            except Exception as exc:
                warnings.append(f"sanitize_failed: {exc}")

        if expects_ring and frag_mol.GetRingInfo().NumRings() == 0:
            warnings.append("ring perception failed: fragment lost ring context")

        # Charge conservation on retained atoms (ignore caps)
        parent_charge = sum(parent_mol.GetAtomWithIdx(i).GetFormalCharge() for i in kept_indices)
        frag_charge = 0
        for idx in range(frag_mol.GetNumAtoms()):
            atom = frag_mol.GetAtomWithIdx(idx)
            if atom.HasProp("parent_uuid"):
                frag_charge += atom.GetFormalCharge()
        if parent_charge != frag_charge:
            warnings.append(
                f"charge_mismatch_on_retained_atoms: parent={parent_charge} frag={frag_charge}"
            )

        # Ring preservation warning if any retained atom was in a ring but fragment has none
        if any(parent_mol.GetAtomWithIdx(i).IsInRing() for i in kept_indices):
            if frag_mol.GetRingInfo().NumRings() == 0:
                warnings.append("ring_lost: retained atoms came from ring")

        for parent_uuid, frag_idx in uuid_to_frag.items():
            parent_atom = atr.get_by_uuid(parent_uuid)
            frag_atom = frag_mol.GetAtomWithIdx(frag_idx)
            if parent_atom.is_aromatic and not frag_atom.GetIsAromatic():
                warnings.append(
                    f"aromaticity lost for atom {parent_uuid}"
                )

        return warnings

    def _compute_context_metrics(
        self,
        atr: AtomicTruthRegistry,
        parent_mol: Chem.Mol,
        frag_mol: Chem.Mol,
        kept_indices: Set[int],
        cut_bonds: List[CutBond],
        cap_records: List[CapRecord],
    ) -> Dict[str, Any]:
        def _heavy_count_mol(mol: Chem.Mol) -> int:
            return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

        def _heavy_count_indices(indices: Set[int]) -> int:
            return sum(1 for i in indices if parent_mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

        parent_heavy = _heavy_count_mol(parent_mol)
        kept_heavy = _heavy_count_indices(kept_indices)
        ring_atoms_kept = sum(
            1 for i in kept_indices if parent_mol.GetAtomWithIdx(i).IsInRing()
        )
        aromatic_atoms_kept = sum(
            1 for i in kept_indices if parent_mol.GetAtomWithIdx(i).GetIsAromatic()
        )
        aromaticity_preserved = True
        for idx in range(frag_mol.GetNumAtoms()):
            atom = frag_mol.GetAtomWithIdx(idx)
            if atom.HasProp("parent_uuid"):
                parent_uuid = atom.GetProp("parent_uuid")
                parent_atom = atr.get_by_uuid(parent_uuid)
                if parent_atom.is_aromatic and not atom.GetIsAromatic():
                    aromaticity_preserved = False
                    break

        conjugated_bonds_kept = sum(
            1 for b in frag_mol.GetBonds() if b.GetIsConjugated()
        )

        formal_charge_parent = sum(
            parent_mol.GetAtomWithIdx(i).GetFormalCharge() for i in kept_indices
        )
        formal_charge_frag = sum(
            frag_mol.GetAtomWithIdx(i).GetFormalCharge()
            for i in range(frag_mol.GetNumAtoms())
            if frag_mol.GetAtomWithIdx(i).HasProp("parent_uuid")
        )

        hbond_donors = sum(
            1
            for atom in frag_mol.GetAtoms()
            if atom.GetSymbol() in {"O", "N", "S"} and atom.GetTotalNumHs() > 0
        )
        hbond_acceptors = sum(
            1
            for atom in frag_mol.GetAtoms()
            if atom.GetSymbol() in {"O", "N", "S"} and atom.GetFormalCharge() <= 0
        )

        return {
            "kept_heavy_atoms": kept_heavy,
            "parent_heavy_atoms": parent_heavy,
            "kept_ratio": round(float(kept_heavy) / float(parent_heavy), 3) if parent_heavy else 0.0,
            "cuts": len(cut_bonds),
            "caps": len(cap_records),
            "ring_atoms_kept": ring_atoms_kept,
            "rings_preserved": frag_mol.GetRingInfo().NumRings(),
            "aromatic_atoms_kept": aromatic_atoms_kept,
            "aromaticity_preserved": aromaticity_preserved,
            "hbond_donors_kept": hbond_donors,
            "hbond_acceptors_kept": hbond_acceptors,
            "formal_charge_kept_parent": formal_charge_parent,
            "formal_charge_kept_frag": formal_charge_frag,
            "conjugated_bonds_kept": conjugated_bonds_kept,
        }

    @staticmethod
    def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
        if atom.GetSymbol() != "C":
            return False
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() >= 1.5 and bond.GetOtherAtom(atom).GetSymbol() == "O":
                return True
        return False
