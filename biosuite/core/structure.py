"""
Protein structure analysis and visualization.

Parses PDB files, computes structural properties, and provides
3D visualization. All pure Python via Biopython.
"""
import os
import numpy as np
from dataclasses import dataclass, field

try:
    from Bio import PDB
    from Bio.PDB import PDBParser, DSSP, SASA, NeighborSearch, PPBuilder
    from Bio.PDB.Polypeptide import PPBuilder
    HAS_BIO = True
except ImportError:
    HAS_BIO = False


@dataclass
class StructureInfo:
    name: str
    num_atoms: int = 0
    num_residues: int = 0
    num_chains: int = 0
    chains: list = field(default_factory=list)
    resolution: float = 0.0
    method: str = ""
    molecule_type: str = ""
    molecular_weight: float = 0.0
    secondary_structure: dict = field(default_factory=dict)
    ramachandran: dict = field(default_factory=dict)
    surface_area: float = 0.0
    message: str = ""


from .utils import has_tool as _has_tool


def check_structure_tools():
    return {'dssp': _has_tool('mkdssp') or _has_tool('dssp')}


def parse_pdb(filepath):
    if not HAS_BIO:
        return None, "Biopython not installed"
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', filepath)
        return structure, None
    except Exception as e:
        return None, str(e)


def get_structure_info(structure, pdb_id="unknown"):
    if not HAS_BIO or structure is None:
        return StructureInfo(name=pdb_id, message="No structure loaded")

    model = structure[0]
    atoms = list(model.get_atoms())
    residues = list(model.get_residues())
    chains = [c.id for c in model.get_chains()]

    info = StructureInfo(
        name=pdb_id,
        num_atoms=len(atoms),
        num_residues=len(residues),
        num_chains=len(chains),
        chains=chains,
    )

    # Extract metadata from header
    try:
        header = structure.header
        info.resolution = header.get('resolution', 0.0)
        info.method = header.get('structure_method', 'unknown')
        info.molecule_type = header.get('molecule_type', 'unknown')
    except Exception:
        pass

    return info


def compute_secondary_structure(structure):
    if not HAS_BIO:
        return {}
    try:
        model = structure[0]
        tools = check_structure_tools()
        if tools.get('dssp'):
            dssp = DSSP(model, structure.full_id[0] if hasattr(structure, 'full_id') else 'protein')
            ss = {}
            for key, val in dssp:
                res_name = f"{val[1]}{key[1][1]}"
                ss[res_name] = val[2]
            return ss
    except Exception:
        pass

    # Fallback: simple secondary structure from phi/psi angles
    ppb = PPBuilder()
    ss_counts = {'H': 0, 'E': 0, 'C': 0}
    for pp in ppb.build_peptides(structure[0]):
        phi_psi = pp.get_phi_psi_list()
        for phi, psi in phi_psi:
            if phi is None or psi is None:
                ss_counts['C'] += 1
            elif -150 < phi < -30 and -75 < psi < 50:
                ss_counts['H'] += 1
            elif -180 < phi < -40 and 90 < psi < 180:
                ss_counts['E'] += 1
            else:
                ss_counts['C'] += 1
    return ss_counts


def compute_ramachandran(structure):
    if not HAS_BIO:
        return {}
    ppb = PPBuilder()
    angles = {'phi': [], 'psi': []}
    for pp in ppb.build_peptides(structure[0]):
        phi_psi = pp.get_phi_psi_list()
        for phi, psi in phi_psi:
            if phi is not None:
                angles['phi'].append(np.degrees(phi))
            if psi is not None:
                angles['psi'].append(np.degrees(psi))
    return angles


def compute_sasa(structure):
    if not HAS_BIO:
        return 0.0
    try:
        sasa = SASA.ShrakeRupley()
        sasa.compute(structure[0], level='R')
        total = sum(r.sasa for r in structure[0].get_residues())
        return total
    except Exception:
        return 0.0


def find_binding_sites(structure, ligand_residue='HEM', radius=8.0):
    if not HAS_BIO:
        return []
    model = structure[0]
    ligand_atoms = []
    for residue in model.get_residues():
        if residue.get_resname() == ligand_residue:
            ligand_atoms.extend(list(residue.get_atoms()))

    if not ligand_atoms:
        return []

    ns = NeighborSearch(list(model.get_atoms()))
    nearby = ns.search(ligand_atoms[0].vector, radius, 'R')

    binding_residues = []
    for res in nearby:
        if res.get_resname() != ligand_residue:
            binding_residues.append({
                'residue': res.get_resname(),
                'chain': res.get_parent().id,
                'number': res.get_id()[1]
            })
    return binding_residues


def load_pdb(pdb_id=None, filepath=None):
    if not HAS_BIO:
        return None, "Biopython not installed"

    parser = PDBParser(QUIET=True)

    if filepath and os.path.exists(filepath):
        return parser.get_structure('protein', filepath), None

    if pdb_id:
        from Bio.PDB import PDBList
        try:
            pdbl = PDBList()
            local_file = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb')
            return parser.get_structure(pdb_id, local_file), None
        except Exception as e:
            return None, f"Could not download PDB {pdb_id}: {e}"

    return None, "No PDB ID or file path provided"


def full_analysis(pdb_id=None, filepath=None):
    structure, err = load_pdb(pdb_id=pdb_id, filepath=filepath)
    if err:
        return StructureInfo(name=pdb_id or 'unknown', message=err)

    info = get_structure_info(structure, pdb_id or 'loaded')
    info.secondary_structure = compute_secondary_structure(structure)
    info.ramachandran = compute_ramachandran(structure)
    info.surface_area = compute_sasa(structure)

    return info


def format_structure_report(info):
    lines = [
        "=== Protein Structure Report ===",
        f"Name: {info.name}",
        f"Atoms: {info.num_atoms:,}",
        f"Residues: {info.num_residues}",
        f"Chains: {info.num_chains} ({', '.join(info.chains)})",
        f"Resolution: {info.resolution:.2f} Å" if info.resolution else "",
        f"Method: {info.method}" if info.method else "",
        f"Surface area: {info.surface_area:.0f} Å²" if info.surface_area else "",
    ]
    if info.secondary_structure:
        if isinstance(info.secondary_structure, dict) and 'H' in info.secondary_structure:
            ss = info.secondary_structure
            total = sum(ss.values())
            lines.append(f"Secondary structure: {ss.get('H',0)/total*100:.0f}% helix, "
                        f"{ss.get('E',0)/total*100:.0f}% sheet, "
                        f"{ss.get('C',0)/total*100:.0f}% coil" if total > 0 else "")
    if info.ramachandran and 'phi' in info.ramachandran:
        phi = info.ramachandran['phi']
        psi = info.ramachandran['psi']
        if phi:
            lines.append(f"Ramachandran: {len(phi)} angles (phi: {np.mean(phi):.1f}±{np.std(phi):.1f}°)")
    if info.message:
        lines.append(f"Note: {info.message}")
    return '\n'.join(lines)
