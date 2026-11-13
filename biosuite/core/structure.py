"""
Protein structure analysis and visualization.

Parses PDB files, computes structural properties, and provides
3D visualization. All pure Python via Biopython.
"""
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from .utils import has_tool as _has_tool

try:
    from Bio.PDB import DSSP, SASA, NeighborSearch, PDBParser, PPBuilder
    from Bio.PDB.PDBExceptions import PDBException, PDBConstructionException
    HAS_BIO = True
    # A PDB read fails in exactly three ways: the path is unusable (OSError), the
    # bytes are not text or the file is empty (ValueError), or a record is
    # malformed (Biopython's PDBException/PDBConstructionException pair).
    _PDB_READ_FAILURES = (OSError, ValueError, PDBException, PDBConstructionException)
    # What running DSSP can report: the binary is missing (OSError), its version
    # line does not parse (AttributeError from Biopython's re.search(...).group),
    # its output is unreadable (ValueError/KeyError/IndexError), the extension is
    # not one Biopython recognises (AssertionError), a stand-in object is not a
    # map (TypeError), or the rows disagree with the structure (PDBException).
    _DSSP_FAILURES = (OSError, ValueError, KeyError, IndexError, TypeError,
                      AttributeError, AssertionError, PDBException)
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


def check_structure_tools():
    return {'dssp': _has_tool('mkdssp') or _has_tool('dssp')}


def _first_model(structure):
    """Return ``structure[0]``, or None when the parsed file carries no model.

    A file with no ATOM record — prose, a FASTA, a truncated download — parses
    without raising, and Biopython then answers ``structure[0]`` with
    ``KeyError: 0``.  Every entry point below reports a model-less structure as
    an empty measurement instead of propagating that key error.
    """
    try:
        return structure[0]
    except (KeyError, IndexError, TypeError):
        return None


def parse_pdb(filepath, structure_id='protein'):
    if not HAS_BIO:
        return None, "Biopython not installed"
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure_id, filepath)
        return structure, None
    except _PDB_READ_FAILURES as e:
        return None, str(e)


def get_structure_info(structure, pdb_id="unknown"):
    if not HAS_BIO or structure is None:
        return StructureInfo(name=pdb_id, message="No structure loaded")

    model = _first_model(structure)
    if model is None:
        return StructureInfo(name=pdb_id, message="Structure contains no model")

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
        # A deposition without a REMARK 2 resolution line carries None, not a
        # missing key: `header.get('resolution', 0.0)` then stored None in a
        # field declared `float`, so any numeric use downstream was a TypeError.
        info.resolution = header.get('resolution') or 0.0
        info.method = header.get('structure_method', 'unknown')
        info.molecule_type = header.get('molecule_type', 'unknown')
    except AttributeError as exc:
        # Only `.header` / `.get` can fail here (a Model, Chain or a hand-built
        # object has none).  The old `except Exception: pass` hid that — and any
        # real bug in the block above — behind a silently empty metadata set.
        warnings.warn(f"PDB header metadata unavailable: {exc}", RuntimeWarning, stacklevel=2)
        info.message = f"No header metadata ({type(exc).__name__}: {exc})"

    return info


def compute_secondary_structure(structure, filepath=None):
    if not HAS_BIO:
        return {}
    model = _first_model(structure)
    if model is None:
        return {}
    try:
        tools = check_structure_tools()
        # DSSP needs the original PDB filename: the old code passed the
        # structure id (not a path), so it always raised and silently fell
        # back to the phi/psi heuristic even with mkdssp installed.
        if tools.get('dssp') and filepath and os.path.exists(filepath):
            dssp = DSSP(model, filepath)
            ss = {'H': 0, 'E': 0, 'C': 0}
            # Biopython's DSSP map yields one flat row per residue, not a
            # (key, row) pair, so the documented access is by the residue key
            # with the secondary structure code at index 2.
            for key in dssp.keys():
                cls = dssp[key][2]
                if cls in ('H', 'G', 'I'):
                    ss['H'] += 1
                elif cls in ('E', 'B'):
                    ss['E'] += 1
                else:
                    ss['C'] += 1
            return ss
    except Exception as exc:
        # Biopython raises a *bare* Exception ("DSSP failed to produce an
        # output") when the program runs but writes nothing, so the set below
        # cannot be listed on its own.  Anything that is neither one of those
        # failures nor that bare marker is a bug here or in the library, and it
        # propagates instead of being downgraded to the heuristic behind a
        # warning.
        if type(exc) is not Exception and not isinstance(exc, _DSSP_FAILURES):
            raise
        warnings.warn(f"DSSP could not be run on {filepath!r}: {exc}; "
                      f"falling back to the phi/psi heuristic", RuntimeWarning, stacklevel=2)

    # Fallback: simple secondary structure from phi/psi angles.
    # ``get_phi_psi_list()`` returns RADIANS; the windows below are degrees, so
    # the angles are converted first.  Reading the raw radians against a -150..-30
    # window made every residue coil and both classification branches unreachable.
    ppb = PPBuilder()
    ss_counts = {'H': 0, 'E': 0, 'C': 0}
    for pp in ppb.build_peptides(model):
        phi_psi = pp.get_phi_psi_list()
        for phi, psi in phi_psi:
            if phi is None or psi is None:
                ss_counts['C'] += 1
                continue
            phi, psi = float(np.degrees(phi)), float(np.degrees(psi))
            if -150 < phi < -30 and -75 < psi < 50:
                ss_counts['H'] += 1
            elif -180 < phi < -40 and 90 < psi < 180:
                ss_counts['E'] += 1
            else:
                ss_counts['C'] += 1
    return ss_counts


def compute_ramachandran(structure):
    """Return the backbone angles as two lists that stay in step.

    ``phi[i]`` and ``psi[i]`` describe the *same* residue: a residue missing
    either angle — the first or the last of its peptide — is skipped.  Sorting
    the two angles into independent lists paired one residue's phi with its
    neighbour's psi, and at a chain break with another chain's residue, so
    anything zipping them plotted points that exist in no real residue.
    """
    if not HAS_BIO:
        return {}
    model = _first_model(structure)
    if model is None:
        return {}
    ppb = PPBuilder()
    angles = {'phi': [], 'psi': []}
    for pp in ppb.build_peptides(model):
        for phi, psi in pp.get_phi_psi_list():
            if phi is None or psi is None:
                continue
            angles['phi'].append(float(np.degrees(phi)))
            angles['psi'].append(float(np.degrees(psi)))
    return angles


def compute_sasa(structure):
    """Total solvent-accessible surface in Å², or 0.0 when it cannot be computed.

    A failed calculation is warned about: 0.0 is otherwise indistinguishable
    from a completely buried structure, and the report drops the line silently.
    """
    if not HAS_BIO:
        return 0.0
    model = _first_model(structure)
    if model is None:
        return 0.0
    try:
        sasa = SASA.ShrakeRupley()
        sasa.compute(model, level='R')
        total = sum(r.sasa for r in model.get_residues())
        return total
    except (AttributeError, TypeError, ValueError) as exc:
        warnings.warn(f"SASA could not be computed: {type(exc).__name__}: {exc}",
                      RuntimeWarning, stacklevel=2)
        return 0.0


def find_binding_sites(structure, ligand_residue='HEM', radius=8.0):
    """Residues with an atom within ``radius`` of any atom of ``ligand_residue``.

    Crystallographic water is skipped: a buried waters shell is not a binding
    site, and on a real deposition it outnumbers the contacting residues.
    """
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError(f"radius must be a positive number, got {radius!r}.")
    if not HAS_BIO:
        return []
    model = _first_model(structure)
    if model is None:
        return []
    ligand_atoms = []
    for residue in model.get_residues():
        if residue.get_resname() == ligand_residue:
            ligand_atoms.extend(list(residue.get_atoms()))

    if not ligand_atoms:
        return []

    ns = NeighborSearch(list(model.get_atoms()))
    # Every ligand atom seeds the search.  Two defects were here: ``Atom.vector``
    # is not an attribute of Biopython's Atom (AttributeError on any real ligand),
    # and only the first atom was used, so residues contacting the rest of the
    # ligand were dropped and the answer depended on atom ordering.
    binding_residues = []
    seen = set()
    for ligand_atom in ligand_atoms:
        for res in ns.search(ligand_atom.coord, radius, 'R'):
            if res.get_resname() == ligand_residue:
                continue
            if res.id[0] == 'W':
                continue
            key = (res.get_parent().id, res.get_id())
            if key in seen:
                continue
            seen.add(key)
            binding_residues.append({
                'residue': res.get_resname(),
                'chain': res.get_parent().id,
                'number': res.get_id()[1]
            })
    return binding_residues


def load_pdb(pdb_id=None, filepath=None):
    if not HAS_BIO:
        return None, "Biopython not installed"

    if filepath and os.path.exists(filepath):
        return parse_pdb(filepath)

    if pdb_id:
        from Bio.PDB import PDBList
        try:
            pdbl = PDBList()
            local_file = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb')
        # The transfer can fail in as many ways as there are transports
        # (socket, TLS, HTTP status, proxy), none of them enumerable here, and
        # each one is handed back to the caller as text rather than swallowed.
        except Exception as e:
            return None, f"Could not download PDB {pdb_id}: {e}"
        # Parsing gets its own message: a deposition that downloaded but cannot
        # be read is not a download failure.
        return parse_pdb(local_file, pdb_id)

    return None, "No PDB ID or file path provided"


def full_analysis(pdb_id=None, filepath=None):
    structure, err = load_pdb(pdb_id=pdb_id, filepath=filepath)
    if err:
        return StructureInfo(name=pdb_id or 'unknown', message=err)

    info = get_structure_info(structure, pdb_id or 'loaded')
    info.secondary_structure = compute_secondary_structure(structure, filepath=filepath)
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
        if phi:
            lines.append(f"Ramachandran: {len(phi)} angles (phi: {np.mean(phi):.1f}±{np.std(phi):.1f}°)")
    if info.message:
        lines.append(f"Note: {info.message}")
    # The optional sections above contribute '' placeholders when they carry no
    # value; they must not reach the caller as blank report lines.
    return '\n'.join(line for line in lines if line)
