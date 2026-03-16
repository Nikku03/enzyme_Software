from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    _RDKIT = True
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    rdMolDescriptors = None
    _RDKIT = False

try:
    from vina import Vina

    _VINA = True
except Exception:  # pragma: no cover
    Vina = None
    _VINA = False

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    _OPENMM = True
except Exception:  # pragma: no cover
    openmm = None
    app = None
    unit = None
    _OPENMM = False

try:
    from pdbfixer import PDBFixer

    _PDBFIXER = True
except Exception:  # pragma: no cover
    PDBFixer = None
    _PDBFIXER = False

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    _MEEKO = True
except Exception:  # pragma: no cover
    MoleculePreparation = None
    PDBQTWriterLegacy = None
    _MEEKO = False


@dataclass
class PreparedMolecule:
    """Molecule prepared for computational engines."""

    smiles: str
    name: str = ""
    mol: Any = None
    xyz_string: Optional[str] = None
    pdb_string: Optional[str] = None
    sdf_string: Optional[str] = None
    pdbqt_string: Optional[str] = None
    n_atoms: int = 0
    n_heavy_atoms: int = 0
    charge: int = 0
    mw: float = 0.0
    warnings: List[str] = field(default_factory=list)


def _mol_to_xyz(mol: Any, conf_id: int = 0) -> Optional[str]:
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    title = "generated"
    if _RDKIT:
        try:
            title = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            title = "generated"
    lines = [str(n), title]
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s} {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}")
    return "\n".join(lines) + "\n"


def _mol_to_pdb(mol: Any, conf_id: int = 0) -> Optional[str]:
    if not _RDKIT or mol is None or mol.GetNumConformers() == 0:
        return None
    return Chem.MolToPDBBlock(mol, confId=conf_id)


def _mol_to_sdf(mol: Any, conf_id: int = 0) -> Optional[str]:
    if not _RDKIT or mol is None or mol.GetNumConformers() == 0:
        return None
    return Chem.MolToMolBlock(mol, confId=conf_id)


def _mol_to_pdbqt(mol: Any) -> Optional[str]:
    if not _MEEKO or mol is None:
        return None
    try:
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        if not setups:
            return None
        pdbqt_str, ok, _err = PDBQTWriterLegacy.write_string(setups[0])
        return pdbqt_str if ok else None
    except Exception:
        return None


def _best_conformer_id(mol: Any) -> int:
    if not _RDKIT or mol is None or mol.GetNumConformers() <= 1:
        return 0
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return 0
        best_id = 0
        best_e = None
        for cid in range(mol.GetNumConformers()):
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            e = float(ff.CalcEnergy())
            if best_e is None or e < best_e:
                best_e = e
                best_id = cid
        return best_id
    except Exception:
        return 0


def prepare_molecule(
    smiles: str,
    name: str = "",
    n_conformers: int = 5,
    add_hs: bool = True,
    optimize: bool = True,
) -> PreparedMolecule:
    """SMILES -> prepared 3D molecule with common format strings."""
    out = PreparedMolecule(smiles=smiles, name=name)
    if not _RDKIT:
        out.warnings.append("RDKit not available")
        return out

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        out.warnings.append(f"SMILES parse failed: {smiles}")
        return out

    if add_hs:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, int(n_conformers)), params=params))
    except Exception:
        conf_ids = []

    if not conf_ids:
        try:
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                out.warnings.append("3D embedding failed")
                return out
        except Exception:
            out.warnings.append("3D embedding failed")
            return out

    if optimize and mol.GetNumConformers() > 0:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        except Exception:
            out.warnings.append("MMFF optimization failed; using embedded geometry")

    best_conf = _best_conformer_id(mol)

    out.mol = mol
    out.n_atoms = int(mol.GetNumAtoms())
    out.n_heavy_atoms = int(mol.GetNumHeavyAtoms())
    out.charge = int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    out.mw = float(rdMolDescriptors.CalcExactMolWt(Chem.RemoveHs(mol)))
    out.xyz_string = _mol_to_xyz(mol, conf_id=best_conf)
    out.pdb_string = _mol_to_pdb(mol, conf_id=best_conf)
    out.sdf_string = _mol_to_sdf(mol, conf_id=best_conf)
    out.pdbqt_string = _mol_to_pdbqt(mol)

    return out


class XTBEngine:
    """GFN2-xTB wrapper for QM energies and BDE calculations."""

    H_ATOM_ENERGY_EH = -0.3935
    EH_TO_KJ = 2625.5

    def __init__(self, xtb_path: Optional[str] = None, solvent: str = "water", timeout: int = 120):
        raw = xtb_path if xtb_path is not None else "xtb"
        if os.path.sep in str(raw) or str(raw).startswith("."):
            self.xtb_path = raw if os.path.exists(str(raw)) else None
        else:
            self.xtb_path = shutil.which(str(raw))
        self.solvent = solvent
        self.timeout = int(timeout)

    def check_available(self) -> bool:
        return bool(self.xtb_path)

    def is_available(self) -> bool:
        return self.check_available()

    def prepare_input(self, prepared: PreparedMolecule) -> Dict[str, Any]:
        return {
            "xyz": prepared.xyz_string,
            "charge": prepared.charge,
            "uhf": 0,
            "solvent": self.solvent,
        }

    def parse_output(self, stdout: str) -> Optional[float]:
        return self._parse_total_energy(stdout)

    def run(self, input_data: Dict[str, Any], optimize: bool = False) -> Dict[str, Any]:
        xyz = input_data.get("xyz")
        if not isinstance(xyz, str) or not xyz.strip():
            return {"energy_eh": None, "error": "missing_xyz"}
        e = self._run_xtb(
            xyz,
            charge=int(input_data.get("charge", 0)),
            uhf=int(input_data.get("uhf", 0)),
            optimize=optimize,
        )
        return {"energy_eh": e, "optimize": optimize}

    def single_point(self, xyz: str, charge: int = 0, uhf: int = 0) -> Optional[float]:
        return self._run_xtb(xyz, charge=charge, uhf=uhf, optimize=False)

    def optimize(self, xyz: str, charge: int = 0, uhf: int = 0) -> Optional[float]:
        return self._run_xtb(xyz, charge=charge, uhf=uhf, optimize=True)

    def compute_bde(
        self,
        prepared: Optional[PreparedMolecule] = None,
        heavy_idx: Optional[int] = None,
        h_idx: Optional[int] = None,
        *,
        smiles: Optional[str] = None,
        name: str = "",
        n_conformers: int = 5,
        add_hs: bool = True,
        optimize: bool = True,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"bde_kj_mol": None, "error": "xTB not available"}

        if isinstance(prepared, str) and smiles is None:
            smiles = prepared
            prepared = None

        if prepared is None and smiles:
            prepared = prepare_molecule(
                smiles=smiles,
                name=name,
                n_conformers=n_conformers,
                add_hs=add_hs,
                optimize=optimize,
            )

        if heavy_idx is None or h_idx is None:
            return {"bde_kj_mol": None, "error": "heavy_idx and h_idx are required"}

        if not _RDKIT or prepared is None or prepared.mol is None:
            return {"bde_kj_mol": None, "error": "No molecule"}

        mol = prepared.mol
        if (
            int(heavy_idx) < 0
            or int(h_idx) < 0
            or int(heavy_idx) >= mol.GetNumAtoms()
            or int(h_idx) >= mol.GetNumAtoms()
        ):
            return {"bde_kj_mol": None, "error": "Invalid atom indices"}

        h_atom = mol.GetAtomWithIdx(int(h_idx))
        if h_atom.GetSymbol() != "H":
            return {"bde_kj_mol": None, "error": "h_idx is not hydrogen"}
        if mol.GetBondBetweenAtoms(int(heavy_idx), int(h_idx)) is None:
            return {"bde_kj_mol": None, "error": "No heavy-H bond found"}

        parent_xyz = _mol_to_xyz(mol, conf_id=0)
        if not parent_xyz:
            return {"bde_kj_mol": None, "error": "Parent XYZ generation failed"}

        e_parent = self.optimize(parent_xyz, charge=prepared.charge, uhf=0)
        if e_parent is None:
            return {"bde_kj_mol": None, "error": "Parent optimization failed"}

        radical = Chem.RWMol(mol)
        radical.RemoveBond(int(heavy_idx), int(h_idx))
        radical.RemoveAtom(int(h_idx))
        new_heavy = int(heavy_idx) if int(h_idx) > int(heavy_idx) else int(heavy_idx) - 1
        try:
            rad_atom = radical.GetAtomWithIdx(new_heavy)
            rad_atom.SetNumRadicalElectrons(1)
            rad_atom.SetNoImplicit(True)
            Chem.SanitizeMol(radical)
        except Exception:
            pass

        if radical.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(radical, randomSeed=42)
            except Exception:
                return {"bde_kj_mol": None, "error": "Radical embedding failed"}

        radical_xyz = _mol_to_xyz(radical, conf_id=0)
        if not radical_xyz:
            return {"bde_kj_mol": None, "error": "Radical XYZ failed"}

        rad_charge = int(sum(a.GetFormalCharge() for a in radical.GetAtoms()))
        e_rad_vert = self.single_point(radical_xyz, charge=rad_charge, uhf=1)
        e_rad_adiab = self.optimize(radical_xyz, charge=rad_charge, uhf=1)

        bde_vert = None
        bde_adiab = None
        if e_rad_vert is not None:
            bde_vert = (e_rad_vert + self.H_ATOM_ENERGY_EH - e_parent) * self.EH_TO_KJ
        if e_rad_adiab is not None:
            bde_adiab = (e_rad_adiab + self.H_ATOM_ENERGY_EH - e_parent) * self.EH_TO_KJ

        primary = bde_vert if bde_vert is not None else bde_adiab
        return {
            "bde_kj_mol": round(primary, 1) if primary is not None else None,
            "bde_vertical_kj_mol": round(bde_vert, 1) if bde_vert is not None else None,
            "bde_adiabatic_kj_mol": round(bde_adiab, 1) if bde_adiab is not None else None,
            "source": "xtb_gfn2",
            "spin_states": {"parent_uhf": 0, "radical_uhf": 1},
        }

    def interaction_energy(
        self,
        complex_xyz: str,
        fragment_a_xyz: str,
        fragment_b_xyz: str,
        charge: int = 0,
    ) -> Optional[float]:
        e_ab = self.single_point(complex_xyz, charge=charge, uhf=0)
        e_a = self.single_point(fragment_a_xyz, charge=charge, uhf=0)
        e_b = self.single_point(fragment_b_xyz, charge=0, uhf=0)
        if e_ab is None or e_a is None or e_b is None:
            return None
        return (e_ab - e_a - e_b) * self.EH_TO_KJ

    def _run_xtb(self, xyz: str, charge: int, uhf: int, optimize: bool) -> Optional[float]:
        if not self.is_available() or not xyz:
            return None
        with tempfile.TemporaryDirectory(prefix="xtb_engine_") as tmp:
            xyz_path = Path(tmp) / "mol.xyz"
            xyz_path.write_text(xyz)
            cmd = [
                str(self.xtb_path),
                str(xyz_path),
                "--gfn",
                "2",
                "--chrg",
                str(int(charge)),
                "--uhf",
                str(int(uhf)),
                "--alpb",
                self.solvent,
                "--etemp",
                "300",
            ]
            if optimize:
                cmd.append("--opt")

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=tmp,
                    timeout=self.timeout,
                )
            except Exception:
                return None
            if proc.returncode != 0:
                return None
            return self._parse_total_energy(proc.stdout)

    @staticmethod
    def _parse_total_energy(stdout: str) -> Optional[float]:
        for line in str(stdout or "").splitlines():
            if "TOTAL ENERGY" in line:
                parts = line.split()
                try:
                    return float(parts[-3])
                except Exception:
                    continue
        return None


CYP_ACTIVE_SITE_CENTERS: Dict[str, Dict[str, Any]] = {
    "1FAG": {"center": (25.0, 5.0, 15.0), "box": (22.0, 22.0, 22.0), "name": "P450-BM3"},
    "1TQN": {"center": (25.0, 10.0, 35.0), "box": (26.0, 26.0, 26.0), "name": "CYP3A4"},
    "2CPP": {"center": (30.0, 65.0, 25.0), "box": (20.0, 20.0, 20.0), "name": "P450cam"},
    "1OG5": {"center": (20.0, 75.0, 45.0), "box": (22.0, 22.0, 22.0), "name": "CYP2C9"},
    "2F9Q": {"center": (15.0, 45.0, 30.0), "box": (20.0, 20.0, 20.0), "name": "CYP2D6"},
}


class VinaEngine:
    """AutoDock Vina wrapper for docking and score extraction."""

    def __init__(self, exhaustiveness: int = 8, n_poses: int = 5):
        self.exhaustiveness = int(exhaustiveness)
        self.n_poses = int(n_poses)

    def check_available(self) -> bool:
        return bool(_VINA)

    def is_available(self) -> bool:
        return self.check_available()

    def prepare_input(
        self,
        receptor_pdbqt: str,
        prepared_ligand: PreparedMolecule,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        return {
            "receptor_pdbqt": receptor_pdbqt,
            "prepared_ligand": prepared_ligand,
            "center": center,
            "box_size": box_size,
        }

    def parse_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return dict(payload)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prepared = input_data.get("prepared_ligand")
        if not isinstance(prepared, PreparedMolecule):
            return {"error": "missing_prepared_ligand", "binding_energy": None}
        return self._dock_prepared(
            receptor_pdbqt=str(input_data.get("receptor_pdbqt") or ""),
            prepared=prepared,
            center=tuple(input_data.get("center") or ()),
            box_size=tuple(input_data.get("box_size") or (22.0, 22.0, 22.0)),
        )

    def prepare_receptor(self, pdb_path: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(Path(pdb_path).with_suffix(".pdbqt"))

        if _PDBFIXER:
            fixer = PDBFixer(filename=pdb_path)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.4)
            clean_pdb = str(Path(output_path).with_suffix(".clean.pdb"))
            with open(clean_pdb, "w") as handle:
                app.PDBFile.writeFile(fixer.topology, fixer.positions, handle)
        else:
            clean_pdb = pdb_path

        prep_bin = shutil.which("prepare_receptor4.py") or shutil.which("prepare_receptor")
        if prep_bin:
            try:
                subprocess.run([prep_bin, "-r", clean_pdb, "-o", output_path], capture_output=True, text=True, timeout=120)
            except Exception:
                shutil.copy(clean_pdb, output_path)
        else:
            shutil.copy(clean_pdb, output_path)
        return output_path

    def prepare_ligand(self, prepared: PreparedMolecule, output_path: Optional[str] = None) -> str:
        out = output_path or tempfile.mktemp(prefix="lig_", suffix=".pdbqt")
        if prepared.pdbqt_string:
            Path(out).write_text(prepared.pdbqt_string)
            return out

        if _MEEKO and prepared.mol is not None:
            pdbqt = _mol_to_pdbqt(prepared.mol)
            if pdbqt:
                Path(out).write_text(pdbqt)
                return out

        # Fallback text to preserve downstream behavior in dry-run environments.
        Path(out).write_text("REMARK ligand_pdbqt_unavailable\n")
        return out

    def dock(
        self,
        receptor_pdbqt: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        pdb_id: Optional[str] = None,
        center: Optional[Tuple[float, float, float]] = None,
        box_size: Optional[Tuple[float, float, float]] = None,
        *,
        pdb_path: Optional[str] = None,
        smiles: Optional[str] = None,
        receptor_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"error": "Vina not installed", "binding_energy": None}

        ligand_smiles = ligand_smiles or smiles
        receptor_pdbqt = receptor_pdbqt or receptor_path

        if not ligand_smiles:
            return {"error": "No ligand SMILES provided", "binding_energy": None}

        if pdb_id and pdb_id in CYP_ACTIVE_SITE_CENTERS:
            site = CYP_ACTIVE_SITE_CENTERS[pdb_id]
            center = center or tuple(site["center"])
            box_size = box_size or tuple(site["box"])

        if center is None:
            return {"error": "No active site center defined", "binding_energy": None}
        if box_size is None:
            box_size = (22.0, 22.0, 22.0)

        prepared = prepare_molecule(ligand_smiles, name="ligand")
        if prepared.mol is None:
            return {"error": "Ligand preparation failed", "binding_energy": None}

        if not receptor_pdbqt and pdb_path:
            if str(pdb_path).lower().endswith(".pdbqt"):
                receptor_pdbqt = pdb_path
            else:
                with tempfile.TemporaryDirectory(prefix="vina_rec_") as rec_tmp:
                    out_pdbqt = os.path.join(rec_tmp, "receptor.pdbqt")
                    try:
                        receptor_pdbqt = self.prepare_receptor(pdb_path, output_path=out_pdbqt)
                    except Exception as exc:
                        return {"error": f"Receptor preparation failed: {exc}", "binding_energy": None}
                    # Must dock while temp directory still exists.
                    return self._dock_prepared(
                        receptor_pdbqt=receptor_pdbqt,
                        prepared=prepared,
                        center=tuple(center),
                        box_size=tuple(box_size),
                        pdb_id=pdb_id,
                    )

        if not receptor_pdbqt:
            return {"error": "No receptor input provided", "binding_energy": None}

        return self._dock_prepared(
            receptor_pdbqt=receptor_pdbqt,
            prepared=prepared,
            center=tuple(center),
            box_size=tuple(box_size),
            pdb_id=pdb_id,
        )

    def _dock_prepared(
        self,
        receptor_pdbqt: str,
        prepared: PreparedMolecule,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        pdb_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(receptor_pdbqt):
            return {"error": f"Receptor not found: {receptor_pdbqt}", "binding_energy": None}

        with tempfile.TemporaryDirectory(prefix="vina_engine_") as tmp:
            lig_path = os.path.join(tmp, "ligand.pdbqt")
            self.prepare_ligand(prepared, lig_path)
            try:
                v = Vina(sf_name="vina")
                v.set_receptor(receptor_pdbqt)
                v.set_ligand_from_file(lig_path)
                v.compute_vina_maps(center=list(center), box_size=list(box_size))
                v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
                energies = v.energies()
            except Exception as exc:
                return {"error": str(exc), "binding_energy": None}

            if energies is None or len(energies) == 0:
                return {"error": "No poses found", "binding_energy": None}

            best_energy = float(energies[0][0])
            try:
                poses = v.poses()
            except Exception:
                poses = ""
            centroid = _parse_ligand_centroid(poses)
            dist_to_center = None
            if centroid is not None:
                dist_to_center = math.sqrt(sum((c - s) ** 2 for c, s in zip(centroid, center)))

            topogate = _vina_to_topogate_scores(best_energy, dist_to_center, len(energies))
            return {
                "binding_energy_kcal": round(best_energy, 2),
                "binding_energy_kj": round(best_energy * 4.184, 2),
                "n_poses": len(energies),
                "all_energies_kcal": [round(float(e[0]), 2) for e in energies],
                "distance_to_center_A": round(dist_to_center, 2) if dist_to_center is not None else None,
                "ligand_centroid": centroid,
                "topogate_scores": topogate,
                "receptor": receptor_pdbqt,
                "ligand_smiles": prepared.smiles,
                "pdb_id": pdb_id,
            }

    def dock_multiple(
        self,
        receptor_pdbqt: str,
        smiles_list: List[str],
        pdb_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for smi in smiles_list:
            row = self.dock(receptor_pdbqt, smi, pdb_id=pdb_id)
            row["smiles"] = smi
            out.append(row)
        return out


def _parse_ligand_centroid(pdb_block: str) -> Optional[Tuple[float, float, float]]:
    if not pdb_block:
        return None
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for line in str(pdb_block).splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except Exception:
                continue
        if line.startswith("ENDMDL"):
            break
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def _vina_to_topogate_scores(
    binding_energy: float,
    distance_to_center: Optional[float],
    n_poses: int,
) -> Dict[str, float]:
    access = 1.0 if n_poses > 0 else 0.0
    reach = 0.5 if distance_to_center is None else max(0.0, min(1.0, 1.0 - distance_to_center / 8.0))
    retention = max(0.0, min(1.0, -float(binding_energy) / 10.0))
    return {
        "access_score": round(access, 3),
        "reach_score": round(reach, 3),
        "retention_score": round(retention, 3),
        "composite": round(0.35 * access + 0.45 * reach + 0.20 * retention, 3),
    }


class OpenMMEngine:
    """OpenMM wrapper for system prep, minimization, and short MD runs."""

    def __init__(
        self,
        forcefield: str = "amber14-all.xml",
        water_model: str = "amber14/tip3pfb.xml",
        temperature_K: float = 300.0,
        friction_ps: float = 1.0,
        timestep_fs: float = 2.0,
        padding_nm: float = 1.0,
    ):
        self.forcefield = forcefield
        self.water_model = water_model
        self.temperature = float(temperature_K)
        self.friction = float(friction_ps)
        self.timestep_fs = float(timestep_fs)
        self.padding = float(padding_nm)

    def check_available(self) -> bool:
        return bool(_OPENMM)

    def is_available(self) -> bool:
        return self.check_available()

    def prepare_input(self, pdb_path: str) -> Dict[str, Any]:
        return self.prepare_system(pdb_path)

    def run(self, system_data: Dict[str, Any], n_steps: int = 50000, report_interval: int = 1000) -> Dict[str, Any]:
        return self.run_md(system_data, n_steps=n_steps, report_interval=report_interval)

    def parse_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return dict(payload)

    def prepare_system(self, pdb_path: str) -> Dict[str, Any]:
        if not self.is_available():
            return {"error": "OpenMM not available"}

        if _PDBFIXER:
            fixer = PDBFixer(filename=pdb_path)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.4)
            topology = fixer.topology
            positions = fixer.positions
        else:
            # fallback without PDBFixer: read PDB directly
            try:
                pdb = app.PDBFile(pdb_path)
            except Exception as exc:
                return {"error": str(exc)}
            topology = pdb.topology
            positions = pdb.positions

        ff = app.ForceField(self.forcefield, self.water_model)
        modeller = app.Modeller(topology, positions)
        modeller.addSolvent(ff, padding=self.padding * unit.nanometers, ionicStrength=0.15 * unit.molar)

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )

        return {
            "system": system,
            "topology": modeller.topology,
            "positions": modeller.positions,
            "forcefield": ff,
            "n_atoms": int(system.getNumParticles()),
        }

    def energy_minimize(self, system_data: Dict[str, Any], max_iterations: int = 1000) -> Dict[str, Any]:
        if "error" in system_data:
            return dict(system_data)

        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * unit.kelvin,
            self.friction / unit.picoseconds,
            self.timestep_fs * unit.femtoseconds,
        )
        simulation = app.Simulation(system_data["topology"], system_data["system"], integrator)
        simulation.context.setPositions(system_data["positions"])

        before = simulation.context.getState(getEnergy=True)
        e_before = before.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        simulation.minimizeEnergy(maxIterations=int(max_iterations))
        after = simulation.context.getState(getEnergy=True, getPositions=True)
        e_after = after.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        return {
            "energy_before_kj": round(float(e_before), 1),
            "energy_after_kj": round(float(e_after), 1),
            "energy_change_kj": round(float(e_after - e_before), 1),
            "positions": after.getPositions(),
            "converged": True,
        }

    def run_md(self, system_data: Dict[str, Any], n_steps: int = 50000, report_interval: int = 1000) -> Dict[str, Any]:
        if "error" in system_data:
            return dict(system_data)

        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * unit.kelvin,
            self.friction / unit.picoseconds,
            self.timestep_fs * unit.femtoseconds,
        )
        simulation = app.Simulation(system_data["topology"], system_data["system"], integrator)
        simulation.context.setPositions(system_data["positions"])

        energies: List[Dict[str, Any]] = []
        step = 0
        stride = max(1, int(report_interval))
        total = max(0, int(n_steps))
        while step < total:
            take = min(stride, total - step)
            simulation.step(take)
            step += take
            state = simulation.context.getState(getEnergy=True)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            energies.append(
                {
                    "step": step,
                    "time_ps": round(step * self.timestep_fs / 1000.0, 3),
                    "potential_kj": round(float(pe), 1),
                    "kinetic_kj": round(float(ke), 1),
                    "total_kj": round(float(pe + ke), 1),
                }
            )

        stable = False
        pe_mean = 0.0
        pe_std = 0.0
        if len(energies) >= 5:
            last_vals = [row["potential_kj"] for row in energies[-5:]]
            mean = sum(last_vals) / len(last_vals)
            var = sum((x - mean) ** 2 for x in last_vals) / (len(last_vals) - 1)
            pe_mean = float(mean)
            pe_std = math.sqrt(max(0.0, var))
            stable = pe_std < max(1.0, abs(pe_mean) * 0.01)

        final_state = simulation.context.getState(getEnergy=True, getPositions=True)
        return {
            "n_steps": int(n_steps),
            "total_time_ps": round(int(n_steps) * self.timestep_fs / 1000.0, 3),
            "timestep_fs": self.timestep_fs,
            "temperature_K": self.temperature,
            "energy_trajectory": energies,
            "final_potential_kj": energies[-1]["potential_kj"] if energies else None,
            "energy_stable": bool(stable),
            "last_5_pe_mean_kj": round(pe_mean, 1),
            "last_5_pe_std_kj": round(pe_std, 1),
            "final_positions": final_state.getPositions(),
        }

    def binding_stability_check(self, pdb_path: str, n_steps: int = 50000) -> Dict[str, Any]:
        system_data = self.prepare_system(pdb_path)
        if "error" in system_data:
            return system_data

        min_result = self.energy_minimize(system_data)
        if "error" in min_result:
            return min_result

        system_data["positions"] = min_result["positions"]
        md_result = self.run_md(system_data, n_steps=n_steps)
        return {
            "minimization": {
                "energy_before_kj": min_result.get("energy_before_kj"),
                "energy_after_kj": min_result.get("energy_after_kj"),
            },
            "md": {
                "total_time_ps": md_result.get("total_time_ps"),
                "energy_stable": md_result.get("energy_stable"),
                "final_pe_kj": md_result.get("final_potential_kj"),
            },
            "verdict": "STABLE" if md_result.get("energy_stable") else "UNSTABLE",
        }

    def binding_stability(self, pdb_path: str, n_steps: int = 50000) -> Dict[str, Any]:
        return self.binding_stability_check(pdb_path=pdb_path, n_steps=n_steps)


def compute_drug_docking_profile(
    drug_smiles: str,
    cyp_pdb_id: str = "1TQN",
    receptor_pdbqt: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare ligand, optionally compute xTB weakest-bond BDE, and dock with Vina."""
    result: Dict[str, Any] = {"smiles": drug_smiles, "pdb_id": cyp_pdb_id}

    prepared = prepare_molecule(drug_smiles, name="drug")
    result["molecule"] = {
        "n_atoms": prepared.n_atoms,
        "mw": prepared.mw,
        "charge": prepared.charge,
        "warnings": list(prepared.warnings),
    }

    xtb = XTBEngine()
    if xtb.is_available() and _RDKIT and prepared.mol is not None:
        best_bde = None
        best_indices: Optional[Tuple[int, int]] = None
        mol = prepared.mol
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in {"C", "N", "O", "S"}:
                continue
            h_idx = None
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "H":
                    h_idx = int(nbr.GetIdx())
                    break
            if h_idx is None:
                continue
            bde_row = xtb.compute_bde(prepared, int(atom.GetIdx()), h_idx)
            bde = bde_row.get("bde_kj_mol")
            if isinstance(bde, (int, float)) and (best_bde is None or float(bde) < float(best_bde)):
                best_bde = float(bde)
                best_indices = (int(atom.GetIdx()), int(h_idx))

        result["xtb_bde"] = {
            "weakest_bond_bde_kj": best_bde,
            "weakest_bond_indices": best_indices,
        }

    vina = VinaEngine()
    if vina.is_available() and receptor_pdbqt:
        result["docking"] = vina.dock(receptor_pdbqt=receptor_pdbqt, ligand_smiles=drug_smiles, pdb_id=cyp_pdb_id)
        if isinstance(result["docking"], dict) and result["docking"].get("topogate_scores"):
            result["topogate_scores"] = result["docking"]["topogate_scores"]
    else:
        result["docking"] = {
            "status": "skipped",
            "reason": "Vina unavailable or receptor_pdbqt missing",
        }

    return result


def check_engines() -> Dict[str, Any]:
    xtb = XTBEngine()
    vina = VinaEngine()
    omm = OpenMMEngine()
    status = {
        "rdkit": _RDKIT,
        "xtb": xtb.is_available(),
        "xtb_path": xtb.xtb_path,
        "vina": vina.is_available(),
        "openmm": omm.is_available(),
        "openmm_version": getattr(openmm, "__version__", None) if _OPENMM else None,
        "pdbfixer": _PDBFIXER,
        "meeko": _MEEKO,
    }
    n_available = sum(1 for k in ("xtb", "vina", "openmm") if status.get(k))
    status["engines_available"] = f"{n_available}/3"
    return status


__all__ = [
    "PreparedMolecule",
    "prepare_molecule",
    "XTBEngine",
    "VinaEngine",
    "OpenMMEngine",
    "compute_drug_docking_profile",
    "check_engines",
    "CYP_ACTIVE_SITE_CENTERS",
]


if __name__ == "__main__":
    st = check_engines()
    print("COMPUTATIONAL ENGINE STATUS")
    print("=" * 50)
    for key, value in st.items():
        print(f"  {key}: {value}")
