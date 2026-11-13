"""
Molecular dynamics simulation with dual-mode execution.

Pure Python with numpy as default engine, OpenMM as optional.
Implements:
- Lennard-Jones potential with cutoff, tail correction and 1-2/1-3 exclusions
- Harmonic bond and angle terms inferred from the distances in the file
- Velocity Verlet integrator
- Berendsen thermostat for temperature control
- Steepest-descent energy minimization
- Proper PDB parsing with residue/chain metadata
- Correct radius of gyration calculation
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from openmm import LangevinMiddleIntegrator, unit
    from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile, Simulation
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

from .log import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants and default parameters
# ═══════════════════════════════════════════════════════════════════════════════

BOLTZMANN_K = 1.380649e-23      # J/K
AVOGADRO = 6.02214076e23        # mol^-1
KCAL_TO_KJ = 4.184              # 1 kcal/mol = 4.184 kJ/mol
KJ_TO_KCAL = 1.0 / KCAL_TO_KJ
amu_to_kg = 1.66053906660e-27   # 1 amu in kg
angstrom_to_m = 1e-10

# Standard atomic weights (amu), keyed by the PDB element symbol.  A mass table
# that mixes element symbols with PDB atom names cannot work: 'CA' is the alpha
# carbon of every residue *and* the element symbol for calcium, so the old
# single table gave a calcium ion carbon's mass.  Lookup is by element only, and
# the entries that matter for biomolecules include phosphorus and the metals.
ELEMENT_MASSES = {
    'H': 1.008, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'P': 30.974, 'S': 32.06, 'CL': 35.45, 'SE': 78.971,
    'NA': 22.990, 'MG': 24.305, 'K': 39.098, 'CA': 40.078, 'MN': 54.938,
    'FE': 55.845, 'CO': 58.933, 'NI': 58.693, 'CU': 63.546, 'ZN': 65.38,
    'MO': 95.95, 'CD': 112.414, 'I': 126.904, 'BR': 79.904, 'HG': 200.592,
}
UNKNOWN_ELEMENT_MASS = 12.011   # carbon, the most common heavy atom in a protein

# Default LJ parameters (OPLS-AA-like, sigma in Å, epsilon in kcal/mol).  Keyed
# by PDB atom name first and element symbol second.
DEFAULT_LJ = {
    'C':  {'sigma': 3.5, 'epsilon': 0.066},
    'CA': {'sigma': 3.5, 'epsilon': 0.066},
    'CB': {'sigma': 3.5, 'epsilon': 0.066},
    'N':  {'sigma': 3.25, 'epsilon': 0.17},
    'O':  {'sigma': 3.0, 'epsilon': 0.17},
    'S':  {'sigma': 3.6, 'epsilon': 0.25},
    'H':  {'sigma': 2.5, 'epsilon': 0.03},
}
# Fallback
_DEFAULT_LJ_ENTRY = {'sigma': 3.4, 'epsilon': 0.066}

# PDB is a fixed-column format: a value that needs more characters than its
# field reserves shifts every later column, and the record then reads back as
# something else entirely.  write_pdb checks each field and refuses.
ALT_LOC_STRATEGIES = ('first', 'last', 'all')
TOOLS = ('auto', 'openmm', 'builtin')


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Atom:
    """Single PDB atom record."""
    serial: int
    name: str
    alt_loc: str
    res_name: str
    chain_id: str
    res_seq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    temp_factor: float
    element: str
    charge: float = 0.0


@dataclass
class PDBStructure:
    """Parsed PDB file contents."""
    atoms: List[Atom]
    title: str = ""
    remarks: List[str] = field(default_factory=list)
    num_models: int = 1

    @property
    def coordinates(self) -> np.ndarray:
        """(N, 3) float array — an empty structure is still a coordinate set."""
        if not self.atoms:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=np.float64)

    @property
    def atom_names(self) -> List[str]:
        return [a.name.strip() for a in self.atoms]

    @property
    def chain_ids(self) -> List[str]:
        return list(dict.fromkeys(a.chain_id for a in self.atoms))

    @property
    def residue_names(self) -> List[str]:
        return [a.res_name.strip() for a in self.atoms]

    @property
    def elements(self) -> List[str]:
        return [a.element.strip() for a in self.atoms]

    @property
    def masses(self) -> np.ndarray:
        """Atomic masses in amu, from the element symbol on each record."""
        unknown: Dict[str, int] = {}
        values = []
        for a in self.atoms:
            symbol = a.element.strip().upper()
            mass = ELEMENT_MASSES.get(symbol)
            if mass is None:
                unknown[symbol] = unknown.get(symbol, 0) + 1
                mass = UNKNOWN_ELEMENT_MASS
            values.append(mass)
        if unknown:
            logger.warning(
                "PDB: no atomic mass for %s; treated as %.3f amu.",
                ", ".join(f"'{s}' ({c} atom(s))" for s, c in sorted(unknown.items())),
                UNKNOWN_ELEMENT_MASS,
            )
        return np.array(values, dtype=np.float64)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


@dataclass
class LJParameters:
    """Per-atom LJ parameters."""
    sigma: np.ndarray    # Å
    epsilon: np.ndarray  # kcal/mol


@dataclass
class MDSimulationResult:
    """What a run measured.

    ``energy``, ``kinetic_energy`` and ``temperature`` are ``None`` until a run
    actually produced them: a simulation that was refused because the file was
    missing used to report 0.0 kcal/mol and 300.0 K, which is a invented
    measurement, not an empty one.
    """
    engine: str
    steps: int = 0
    energy: Optional[float] = None
    kinetic_energy: Optional[float] = None
    temperature: Optional[float] = None
    rmsd: List[float] = field(default_factory=list)
    radius_gyration: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    temperature_history: List[float] = field(default_factory=list)
    output_pdb: str = ""
    message: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PDB parsing
# ═══════════════════════════════════════════════════════════════════════════════

_ELEMENT_MAP = {
    'H': 'H', 'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S', 'P': 'P',
    'F': 'F', 'B': 'B', 'I': 'I', 'K': 'K',
}

# Two-letter ion symbols that a bare atom name may stand for.  'CD' (cadmium)
# and 'CA' (calcium) are deliberately absent: CD1/CD2 are leucine and
# phenylalanine atom names and CA is the alpha carbon of every residue, so
# guessing them from a name would relabel ordinary protein atoms as metals.
_ION_NAMES = ('FE', 'ZN', 'MG', 'MN', 'CU', 'CO', 'NI', 'SE', 'CL', 'BR',
              'NA', 'MO', 'HG')


def _guess_element(atom_name: str) -> str:
    """Guess an element from an atom name (PDB convention: column-aligned naming).

    Used only when the file has no element column.  A name that is exactly an
    ion symbol, or a symbol followed by a digit ('FE1' in an iron-sulfur
    cluster), resolves to that element; everything else takes its first letter.
    """
    clean = atom_name.strip().lstrip('0123456789').upper()
    if clean in _ION_NAMES:
        return clean
    if len(clean) >= 3 and clean[:2] in _ION_NAMES and clean[2:].isdigit():
        return clean[:2]
    first = clean[0] if clean else 'C'
    if first in _ELEMENT_MAP:
        return first
    return 'C'  # fallback


def parse_pdb(pdb_path: str, alt_loc_strategy: str = 'first') -> PDBStructure:
    """
    Parse a PDB file into a PDBStructure.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    alt_loc_strategy : str
        How to handle alternate locations: 'first' (default), 'last', or 'all'.

    Returns
    -------
    PDBStructure

    Raises
    ------
    ValueError
        If ``alt_loc_strategy`` is not one of the three documented strategies.
    """
    if alt_loc_strategy not in ALT_LOC_STRATEGIES:
        raise ValueError(
            f"alt_loc_strategy must be one of {ALT_LOC_STRATEGIES}, "
            f"got {alt_loc_strategy!r}."
        )

    atoms: List[Atom] = []
    remarks: List[str] = []
    title = ""
    seen_alt: Dict[Tuple[str, int, str, str, str], int] = {}
    models = 0
    skipped_models = 0
    unreadable_coordinates = 0
    first_unreadable_line = None
    coerced_serial = 0
    coerced_res_seq = 0

    with open(pdb_path, 'r', encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            line = raw.rstrip('\n\r')
            record = line[:6].strip()

            if record == 'TITLE':
                title = (title + ' ' + line[10:].strip()).strip()
            elif record == 'REMARK':
                remarks.append(line)
            elif record == 'MODEL':
                # A multi-model file (NMR ensembles, trajectory dumps) repeats
                # the same atoms once per model.  Concatenating them produced a
                # structure with 20x the atoms of the molecule, bonded across
                # model boundaries, so only the first model is read and the
                # total is reported in num_models.
                models += 1
            elif record in ('ATOM', 'HETATM'):
                if models > 1:
                    skipped_models += 1
                    continue
                try:
                    serial = int(line[6:11])
                except ValueError:
                    serial = len(atoms) + 1
                    coerced_serial += 1
                name = line[12:16].strip()
                alt_loc = line[16].strip() or ' '
                res_name = line[17:20].strip()
                chain_id = line[21].strip() or 'A'
                try:
                    res_seq = int(line[22:26])
                except ValueError:
                    res_seq = 0
                    coerced_res_seq += 1
                icode = line[26].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    unreadable_coordinates += 1
                    if first_unreadable_line is None:
                        first_unreadable_line = number
                    continue
                try:
                    occupancy = float(line[54:60])
                except ValueError:
                    occupancy = 1.0
                try:
                    temp_factor = float(line[60:66])
                except ValueError:
                    temp_factor = 0.0

                # Element
                element = ''
                if len(line) >= 78:
                    element = line[76:78].strip()
                if not element:
                    element = _guess_element(name)

                atom = Atom(
                    serial=serial, name=name, alt_loc=alt_loc,
                    res_name=res_name, chain_id=chain_id,
                    res_seq=res_seq, icode=icode,
                    x=x, y=y, z=z,
                    occupancy=occupancy, temp_factor=temp_factor,
                    element=element,
                )

                # Alt location handling.  ``seen_alt`` maps the residue's atom
                # slot to the list index the conformer was stored at, so 'last'
                # can overwrite it.  The key carries chain and residue identity
                # as well as the name: keyed on (name, res_seq) alone, two chains
                # that both hold a 'CA' of residue 1 were treated as conformers
                # of each other and one whole chain was thrown away.
                key = (chain_id, res_seq, icode, res_name, name)
                if alt_loc != ' ':
                    if key in seen_alt:
                        if alt_loc_strategy == 'first':
                            continue  # skip subsequent alt locs
                        if alt_loc_strategy == 'last':
                            atoms[seen_alt[key]] = atom
                            continue
                    seen_alt[key] = len(atoms)

                atoms.append(atom)

    if unreadable_coordinates:
        logger.warning(
            "PDB %s: %d atom record(s) skipped because their coordinate "
            "columns could not be read (first at line %d).",
            pdb_path, unreadable_coordinates, first_unreadable_line,
        )
    if coerced_serial:
        logger.warning(
            "PDB %s: %d atom record(s) had an unreadable serial number and were "
            "numbered from their row instead.",
            pdb_path, coerced_serial,
        )
    if coerced_res_seq:
        logger.warning(
            "PDB %s: %d atom record(s) had an unreadable residue number and were "
            "given residue 0 instead.",
            pdb_path, coerced_res_seq,
        )
    if models > 1:
        logger.warning(
            "PDB %s holds %d models; only the first was read (%d atom(s), "
            "%d record(s) from the later models ignored).",
            pdb_path, models, len(atoms), skipped_models,
        )

    return PDBStructure(atoms=atoms, title=title, remarks=remarks,
                        num_models=max(models, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Lennard-Jones potential
# ═══════════════════════════════════════════════════════════════════════════════

def _as_float_coords(coords, label: str = "coords") -> np.ndarray:
    """Coerce a coordinate array to a float (N, 3) view or reject it.

    Integer coordinates used to reach ``np.zeros_like(coords)`` inside the force
    builders, which made every force an integer and raised a casting error from
    the middle of the minimiser.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{label} must be an (N, 3) array of coordinates, got shape {arr.shape}."
        )
    return arr


def _require_matching_length(values: np.ndarray, n: int, label: str) -> None:
    """Reject a per-atom parameter array that does not describe this system."""
    if len(values) != n:
        raise ValueError(
            f"{label} has {len(values)} entries but the coordinate set has "
            f"{n} atom(s); they must describe the same system."
        )


def assign_lj_parameters(atoms: List[Atom]) -> LJParameters:
    """Assign LJ parameters from atom names / element types.

    An atom whose name and element are both outside the parameter table gets the
    documented generic entry, and that substitution is reported once per type
    rather than silently — a phosphorus or a zinc modelled with carbon's sigma
    and epsilon is not the atom the user put in the file.
    """
    n = len(atoms)
    sigma = np.zeros(n, dtype=np.float64)
    epsilon = np.zeros(n, dtype=np.float64)
    generic: Dict[str, int] = {}
    for i, atom in enumerate(atoms):
        name = atom.name.strip().upper()
        elem = atom.element.strip().upper()
        params = DEFAULT_LJ.get(name) or DEFAULT_LJ.get(elem)
        if params is None:
            params = _DEFAULT_LJ_ENTRY
            key = f"{name or '(blank)'} / {elem or '(blank)'}"
            generic[key] = generic.get(key, 0) + 1
        sigma[i] = params['sigma']
        epsilon[i] = params['epsilon']
    if generic:
        logger.warning(
            "LJ: %s have no tabulated parameters and use the generic "
            "sigma=%.2f Å, epsilon=%.3f kcal/mol.",
            ", ".join(f"'{k}' ({c} atom(s))" for k, c in sorted(generic.items())),
            _DEFAULT_LJ_ENTRY['sigma'], _DEFAULT_LJ_ENTRY['epsilon'],
        )
    return LJParameters(sigma=sigma, epsilon=epsilon)


def lj_cutoff_correction(sigma: np.ndarray, epsilon: np.ndarray,
                         cutoff: float, n: int) -> float:
    """
    Long-range tail correction for the LJ energy of ``n`` atoms.

        U_tail = (8/3) * pi * N * rho * <eps * sigma^3 *
                 [ (sigma/rc)^9 / 3 - (sigma/rc)^3 / 2 ] >

    ``rho`` is the number density.  No box is known here, so the module's
    standing approximation is one (2*cutoff)^3 cell per atom, which makes rho
    fixed and the correction extensive in N.  (It used to multiply by N/V with
    V = N * (2*cutoff)^3, which cancels N exactly: the same value came back for
    one atom and for a thousand, so a bigger system got no more tail than a
    single pair.)  Averaging sigma and epsilon over the atoms is the mean-field
    part of the approximation; a real pair sum would scale as N^2.

    Returns the energy in kcal/mol; 0.0 for an empty system.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be a positive distance in Å, got {cutoff}.")
    if n <= 0 or len(sigma) == 0:
        return 0.0
    avg_sigma = float(np.mean(sigma))
    avg_eps = float(np.mean(epsilon))
    sr = avg_sigma / cutoff
    sr3 = sr ** 3
    sr9 = sr3 ** 3
    vol_per_atom = (2.0 * cutoff) ** 3
    density = 1.0 / vol_per_atom
    correction = (8.0 / 3.0) * math.pi * n * density * avg_eps * avg_sigma**3 * (
        sr9 / 3.0 - sr3 / 2.0
    )
    return correction


def compute_lj_energy(coords: np.ndarray, lj: LJParameters,
                      cutoff: float = 10.0,
                      exclusions: Optional[frozenset] = None) -> Tuple[float, np.ndarray]:
    """
    Compute Lennard-Jones energy and forces with cutoff and tail correction.

    Uses Lorentz-Berthelot mixing rules for unlike pairs:
        sigma_ij = (sigma_i + sigma_j) / 2
        eps_ij = sqrt(eps_i * eps_j)

    Parameters
    ----------
    coords : (N, 3) array in Å
    lj : LJParameters
    cutoff : float, cutoff distance in Å
    exclusions : set of (i, j) index pairs, i < j, whose LJ term is switched off

    Returns
    -------
    energy : float (kcal/mol)
    forces : (N, 3) array (kcal/mol/Å)

    Notes
    -----
    ``exclusions`` exists because bonded neighbours sit 1.2-1.5 Å apart while
    every heavy atom here carries a sigma of 3.0-3.6 Å: applying the full LJ
    curve to those pairs puts each bond ~10,000 kcal/mol and ~85,000 kcal/mol/Å
    inside the repulsive wall, so fifteen integration steps of a six-atom chain
    threw an atom to 17,951 Å.  Standard force fields exclude 1-2 and 1-3 pairs
    from the LJ sum for exactly this reason; ``compute_forces`` builds that set
    from the bond and angle lists it is already given.
    """
    coords = _as_float_coords(coords)
    n = len(coords)
    _require_matching_length(lj.sigma, n, "lj.sigma")
    _require_matching_length(lj.epsilon, n, "lj.epsilon")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be a positive distance in Å, got {cutoff}.")
    forces = np.zeros((n, 3), dtype=np.float64)
    energy = 0.0
    cutoff_sq = cutoff * cutoff
    coincident = 0

    sigma_i = lj.sigma
    epsilon_i = lj.epsilon

    # Offsets of the excluded partners inside each row of the upper triangle.
    offsets: Dict[int, List[int]] = {}
    if exclusions:
        for pair in exclusions:
            i, j = pair
            if not (0 <= i < j < n):
                raise ValueError(
                    f"LJ exclusion {pair} does not name an ordered pair of atoms "
                    f"in a system of {n} atom(s)."
                )
            offsets.setdefault(i, []).append(j - i - 1)

    for i in range(n - 1):
        # Vectorized distances from atom i to atoms i+1..N-1
        diff = coords[i + 1:] - coords[i]          # (M, 3)
        dist_sq = np.sum(diff * diff, axis=1)      # (M,)
        mask = (dist_sq < cutoff_sq) & (dist_sq > 1e-6)
        coincident += int(np.count_nonzero((dist_sq < cutoff_sq) & ~mask))
        if i in offsets:
            mask[offsets[i]] = False
        if not np.any(mask):
            continue

        idx = np.where(mask)[0]
        r2 = dist_sq[idx]
        r = np.sqrt(r2)

        # Lorentz-Berthelot mixing
        sigma_ij = 0.5 * (sigma_i[i] + sigma_i[i + 1 + idx])
        eps_ij = np.sqrt(epsilon_i[i] * epsilon_i[i + 1 + idx])

        sr2 = (sigma_ij / r) ** 2     # (sigma/r)^2
        sr6 = sr2 ** 3                 # (sigma/r)^6
        sr12 = sr6 ** 2                # (sigma/r)^12

        # V = 4 * eps * (sr12 - sr6)
        pair_energy = 4.0 * eps_ij * (sr12 - sr6)
        energy += np.sum(pair_energy)

        # F_i = -dV/dr * (r_ij_vec / r) with r_ij_vec = coords[j] - coords[i]
        # and dV/dr = (24*eps/r) * (sr6 - 2*sr12), so
        # F_vec_i = -24 * eps * (2*sr12 - sr6) / r2 * r_ij_vec.
        # The sign here used to be +, i.e. the code returned +grad(V): inside
        # the repulsive wall it pulled overlapping atoms TOGETHER, so
        # steepest descent climbed the potential and never converged.
        f_scalar = -24.0 * eps_ij * (2.0 * sr12 - sr6) / r2

        f_vec = f_scalar[:, np.newaxis] * diff[idx]   # (M, 3)
        forces[i] += np.sum(f_vec, axis=0)
        for k, j in enumerate(idx):
            forces[i + 1 + j] -= f_vec[k]

    # Tail correction (approximate)
    energy += lj_cutoff_correction(lj.sigma, lj.epsilon, cutoff, n)

    if coincident:
        logger.warning(
            "LJ: %d atom pair(s) are closer than 0.001 Å and were left out of "
            "the energy and the forces; a duplicated atom in the input produces "
            "an interaction that silently does nothing.",
            coincident,
        )

    return energy, forces


# ═══════════════════════════════════════════════════════════════════════════════
# Spring (harmonic bond) potential
# ═══════════════════════════════════════════════════════════════════════════════

def _build_bond_list(pdb: PDBStructure, max_bond_len: float = 2.0) -> List[Tuple[int, int, float]]:
    """
    Infer bonds from inter-atomic distances.
    Returns list of (i, j, equilibrium_length) in Å.

    The equilibrium length is the distance the file happens to hold, so a clash
    in the input becomes a bond that holds the clash in place; that is the price
    of bonding without a topology.
    """
    if max_bond_len <= 0:
        raise ValueError(
            f"max_bond_len must be a positive distance in Å, got {max_bond_len}."
        )
    coords = pdb.coordinates
    n = pdb.n_atoms
    bonds = []
    for i in range(n - 1):
        diff = coords[i + 1:] - coords[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        close = np.where((dist < max_bond_len) & (dist > 0.5))[0]
        for k in close:
            bonds.append((i, i + 1 + k, float(dist[k])))
    if n >= 2 and not bonds:
        logger.warning(
            "Bonds: no pair of the %d atom(s) is within %.2f Å, so the run has "
            "no bond or angle term at all — a CA-only trace at 3.8 Å spacing "
            "behaves like a box of independent atoms.",
            n, max_bond_len,
        )
    return bonds


def _check_topology_indices(pairs: List[tuple], n: int, kind: str) -> None:
    """Fail on a bond/angle that names an atom the coordinate set does not have."""
    arity = 2 if kind == 'bond' else 3
    for entry in pairs:
        for position in entry[:arity]:
            if not 0 <= position < n:
                raise ValueError(
                    f"{kind} {tuple(entry)} refers to atom {position} of a "
                    f"system with {n} atom(s)."
                )


def compute_bond_energy(coords: np.ndarray, bonds: List[Tuple[int, int, float]],
                        k_bond: float = 500.0) -> Tuple[float, np.ndarray]:
    """
    Harmonic bond potential: E = 0.5 * k * (r - r0)^2.
    k_bond in kcal/mol/Å^2.
    """
    coords = _as_float_coords(coords)
    _check_topology_indices(bonds, len(coords), 'bond')
    forces = np.zeros((len(coords), 3), dtype=np.float64)
    energy = 0.0
    for i, j, r0 in bonds:
        diff = coords[j] - coords[i]
        r = np.sqrt(np.sum(diff * diff))
        if r < 1e-8:
            continue
        dr = r - r0
        energy += 0.5 * k_bond * dr * dr
        # F = -k * (r - r0) * (r_vec / r)
        f_scalar = -k_bond * dr / r
        f_vec = f_scalar * diff
        forces[i] -= f_vec
        forces[j] += f_vec
    return energy, forces


# ═══════════════════════════════════════════════════════════════════════════════
# Angle potential
# ═══════════════════════════════════════════════════════════════════════════════

def _build_angle_list(bonds: List[Tuple[int, int, float]]) -> List[Tuple[int, int, int]]:
    """Build angle list from bond connectivity."""
    adj: Dict[int, list] = {}
    for i, j, _ in bonds:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    angles = []
    seen = set()
    for center, neighbors in adj.items():
        for a in neighbors:
            for b in neighbors:
                if a < b:
                    key = (a, center, b)
                    if key not in seen:
                        seen.add(key)
                        angles.append(key)
    return angles


def compute_angle_energy(coords: np.ndarray, angles: List[Tuple[int, int, int]],
                         k_angle: float = 50.0) -> Tuple[float, np.ndarray]:
    """
    Harmonic angle potential: E = 0.5 * k * (theta - theta0)^2.
    k_angle in kcal/mol/rad^2, theta0 = 1.9106 rad (109.5 deg, tetrahedral default).
    """
    coords = _as_float_coords(coords)
    _check_topology_indices(angles, len(coords), 'angle')
    forces = np.zeros((len(coords), 3), dtype=np.float64)
    energy = 0.0
    theta0 = math.radians(109.5)

    for i, j, k in angles:
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        n1 = np.sqrt(np.sum(v1 * v1))
        n2 = np.sqrt(np.sum(v2 * v2))
        if n1 < 1e-8 or n2 < 1e-8:
            continue
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        theta = math.acos(cos_theta)
        dtheta = theta - theta0
        energy += 0.5 * k_angle * dtheta * dtheta

        if abs(math.sin(theta)) < 1e-8:
            continue
        # Gradient of the angle potential w.r.t. coords (simplified).
        # grad_v1 / grad_v2 are dE/dcoords_i and dE/dcoords_k, so the force
        # is their NEGATIVE; adding them instead (as this line used to) made
        # the angle term push atoms further from the equilibrium angle.
        coeff = k_angle * dtheta / math.sin(theta)
        # d(theta)/d(v1) = (cos_theta * v1/n1 - v2/n2) / (n1)
        grad_v1 = coeff * (cos_theta * v1 / (n1 * n1) - v2 / (n1 * n2))
        grad_v2 = coeff * (cos_theta * v2 / (n2 * n2) - v1 / (n1 * n2))
        forces[i] -= grad_v1
        forces[k] -= grad_v2
        forces[j] += (grad_v1 + grad_v2)

    return energy, forces


# ═══════════════════════════════════════════════════════════════════════════════
# Total force computation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_lj_exclusions(bonds: List[Tuple[int, int, float]],
                         angles: List[Tuple[int, int, int]]) -> frozenset:
    """1-2 and 1-3 atom pairs, which must not feel each other through LJ.

    A pair joined by a bond, or separated by one bonded atom, is already held at
    its distance by the bond and angle terms; letting the Lennard-Jones curve
    act on it as well adds a repulsion of thousands of kcal/mol/Å at the 1.5 Å
    of a real bond length.
    """
    excluded = set()
    for i, j, _r0 in bonds:
        excluded.add((min(i, j), max(i, j)))
    for a, _centre, b in angles:
        if a != b:
            excluded.add((min(a, b), max(a, b)))
    return frozenset(excluded)


def compute_forces(coords: np.ndarray, lj: LJParameters,
                   bonds: List, angles: List,
                   cutoff: float = 10.0) -> Tuple[float, np.ndarray]:
    """
    Total potential energy and forces from LJ + bonds + angles.

    Bonded (1-2) and angle-coupled (1-3) pairs are left out of the LJ sum; see
    ``_build_lj_exclusions``.
    """
    exclusions = _build_lj_exclusions(bonds, angles) if bonds else None
    e_lj, f_lj = compute_lj_energy(coords, lj, cutoff, exclusions)
    e_bond, f_bond = compute_bond_energy(coords, bonds)
    e_angle, f_angle = compute_angle_energy(coords, angles)
    total_e = e_lj + e_bond + e_angle
    total_f = f_lj + f_bond + f_angle
    return total_e, total_f


# ═══════════════════════════════════════════════════════════════════════════════
# Steepest-descent energy minimisation
# ═══════════════════════════════════════════════════════════════════════════════

def minimize_energy(coords: np.ndarray, lj: LJParameters,
                    bonds: List, angles: List,
                    max_steps: int = 500,
                    convergence_tol: float = 1e-3,
                    cutoff: float = 10.0,
                    initial_step: float = 0.01) -> Tuple[np.ndarray, List[float]]:
    """
    Steepest-descent energy minimisation.

    Parameters
    ----------
    coords : (N, 3) initial coordinates
    lj : LJParameters
    bonds : list of (i, j, r0)
    angles : list of (i, j, k)
    max_steps : int
    convergence_tol : float – stop when max|F| < tol
    cutoff : float – LJ cutoff in Å
    initial_step : float – initial step size in Å

    Returns
    -------
    coords : minimised (N, 3)
    energy_history : list of energies, one per iteration plus the starting point

    Raises
    ------
    ValueError
        If a step count, tolerance or step size is not a usable positive number.

    Notes
    -----
    Convergence is judged on max|F|, the largest force component on any atom,
    which is what the tolerance means in the docstring.  The Frobenius norm of
    the whole force array used to be compared against it instead, so the same
    tolerance asked for a gradient smaller by sqrt(3N) as the system grew — a
    200-atom sample was held to a criterion 24 times tighter than a pair.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool):
        raise ValueError(f"max_steps must be an integer, got {max_steps!r}.")
    if max_steps < 0:
        raise ValueError(f"max_steps cannot be negative, got {max_steps}.")
    if convergence_tol <= 0:
        raise ValueError(
            f"convergence_tol must be a positive force in kcal/mol/Å, got {convergence_tol}."
        )
    if initial_step <= 0:
        raise ValueError(
            f"initial_step must be a positive displacement in Å, got {initial_step}."
        )

    coords = _as_float_coords(coords).copy()
    energy_history = []
    step_size = initial_step

    energy, forces = compute_forces(coords, lj, bonds, angles, cutoff)
    energy_history.append(energy)
    gradient = float(np.max(np.abs(forces))) if forces.size else 0.0

    converged = gradient < convergence_tol
    for _step in range(max_steps):
        if gradient < convergence_tol:
            break

        # Normalize force direction, scale by step_size
        direction = forces / gradient
        trial_coords = coords + step_size * direction

        trial_energy, trial_forces = compute_forces(trial_coords, lj, bonds, angles, cutoff)

        if trial_energy < energy:
            # Accept
            coords = trial_coords
            energy = trial_energy
            forces = trial_forces
            gradient = float(np.max(np.abs(forces))) if forces.size else 0.0
            step_size *= 1.2  # increase step
        else:
            # Reject, reduce step
            step_size *= 0.5

        energy_history.append(energy)
        converged = gradient < convergence_tol
        if step_size < 1e-12 and not converged:
            break

    if not converged:
        logger.warning(
            "Minimisation ended at max|F| = %.4g kcal/mol/Å, above the "
            "convergence tolerance %.4g, after %d step(s) (step size had "
            "shrunk to %.3g Å); the structure is not at a minimum.",
            gradient, convergence_tol, len(energy_history) - 1, step_size,
        )

    return coords, energy_history


# ═══════════════════════════════════════════════════════════════════════════════
# Velocity Verlet integrator + Berendsen thermostat
# ═══════════════════════════════════════════════════════════════════════════════

def _init_velocities(masses: np.ndarray, target_temp: float,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Initialise velocities from a Maxwell-Boltzmann distribution.

    Parameters
    ----------
    masses : (N,) in amu
    target_temp : K, 0.0 for a still system
    rng : numpy random generator; a fresh one is drawn when omitted

    Returns
    -------
    velocities : (N, 3) in Å/fs  (1 Å/fs = 1e5 m/s)
    """
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError(
            f"masses must be a non-empty (N,) array, got shape {masses.shape}."
        )
    if np.any(masses <= 0):
        bad = int(np.argmax(masses <= 0))
        raise ValueError(
            f"mass of atom {bad} is {masses[bad]} amu; every atom needs a "
            f"positive mass to be given a thermal velocity."
        )
    if target_temp < 0:
        raise ValueError(
            f"target temperature cannot be negative, got {target_temp} K."
        )

    if rng is None:
        rng = np.random.default_rng()

    # sigma_v = sqrt(k_B * T / m) in real units, then convert to Å/fs
    # k_B = 1.380649e-23 J/K, m in amu -> kg, v in m/s -> Å/fs (*1e5)
    kB = BOLTZMANN_K
    amu_kg = amu_to_kg
    angstrom_fs = 1e-5  # 1 Å/fs = 1e5 m/s -> 1 m/s = 1e-5 Å/fs

    sigma_v = np.sqrt(kB * target_temp / (masses * amu_kg))  # m/s
    sigma_v *= angstrom_fs  # Å/fs

    vel = rng.normal(0.0, 1.0, size=(len(masses), 3))
    vel *= sigma_v[:, np.newaxis]

    # Remove centre-of-mass velocity
    total_momentum = np.sum(masses[:, np.newaxis] * vel, axis=0)
    total_mass = np.sum(masses)
    vel -= total_momentum / total_mass

    return vel


def compute_temperature(velocities: np.ndarray, masses: np.ndarray) -> float:
    """
    Instantaneous temperature from kinetic energy.

    T = (2 / (3N)) * KE / k_B
    """
    velocities = _as_float_coords(velocities, "velocities")
    masses = np.asarray(masses, dtype=np.float64)
    _require_matching_length(masses, len(velocities), "masses")
    n = len(masses)
    if n == 0:
        return 0.0
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)

    # Convert KE from amu*(A/fs)^2 to Joules
    # 1 amu = 1.66054e-27 kg, 1 Å/fs = 1e5 m/s
    ke_joules = ke * amu_to_kg * 1e10  # amu * (A/fs)^2 -> kg*(m/s)^2 = J

    T = 2.0 * ke_joules / (3.0 * n * BOLTZMANN_K)
    return float(T)


def berendsen_thermostat(velocities: np.ndarray,
                         current_temp: float, target_temp: float,
                         tau: float = 0.1, dt: float = 1.0) -> np.ndarray:
    """
    Berendsen velocity rescaling thermostat.

    lambda = sqrt(1 + (dt/tau) * (T_target/T_current - 1))

    Parameters
    ----------
    tau : coupling time constant in same units as dt (fs)
    dt : timestep in fs

    Returns
    -------
    rescaled velocities

    Notes
    -----
    The scaling factor is mass-independent by construction — every velocity is
    multiplied by the same lambda — and the masses the caller used to hand in
    were never read: ``current_temp`` already carries the mass weighting from
    ``compute_temperature``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be a positive coupling time, got {tau}.")
    if dt <= 0:
        raise ValueError(f"dt must be a positive timestep, got {dt}.")
    if target_temp < 0:
        raise ValueError(f"target temperature cannot be negative, got {target_temp}.")
    if current_temp < 1e-6:
        return velocities
    lam_sq = 1.0 + (dt / tau) * (target_temp / current_temp - 1.0)
    if lam_sq < 0:
        # dt/tau larger than the relative gap lets lambda go imaginary; the
        # documented behaviour is a full quench, not a crash.
        lam_sq = 0.0
    lam = math.sqrt(lam_sq)
    return velocities * lam


def velocity_verlet_step(coords: np.ndarray, velocities: np.ndarray,
                         forces: np.ndarray, masses: np.ndarray,
                         lj: LJParameters, bonds: List, angles: List,
                         dt: float = 1.0, cutoff: float = 10.0,
                         target_temp: float = 300.0,
                         tau: float = 0.1,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    One Velocity Verlet step with Berendsen thermostat.

    Parameters
    ----------
    dt : timestep in fs
    cutoff : LJ cutoff in Å
    target_temp : thermostat target in K
    tau : Berendsen coupling constant in fs

    Returns
    -------
    new_coords, new_velocities, new_forces, potential_energy
    """
    coords = _as_float_coords(coords)
    velocities = _as_float_coords(velocities, "velocities")
    forces = _as_float_coords(forces, "forces")
    masses = np.asarray(masses, dtype=np.float64)
    n = len(coords)
    _require_matching_length(masses, n, "masses")
    _require_matching_length(velocities, n, "velocities")
    _require_matching_length(forces, n, "forces")
    if np.any(masses <= 0):
        bad = int(np.argmax(masses <= 0))
        raise ValueError(
            f"mass of atom {bad} is {masses[bad]} amu; a zero or negative mass "
            f"turns F = ma into an infinite acceleration."
        )
    if dt <= 0:
        raise ValueError(
            f"dt must be a positive timestep in fs, got {dt}; a zero dt leaves "
            f"the atoms still while the thermostat still rescales their "
            f"velocities."
        )
    if target_temp < 0:
        raise ValueError(f"target temperature cannot be negative, got {target_temp}.")
    # Convert forces: F in kcal/mol/Å, m in amu, need a in Å/fs²
    # F = m * a  -> a = F/m
    # kcal/mol/Å / amu = kcal/(mol*amu*Å)
    # 1 kcal/mol = 4184 J/mol = 4184/6.022e23 J = 6.947e-21 J
    # a [m/s²] = F[N] / m[kg]
    # F[N] = F[kcal/mol/Å] * 4184/(6.022e23) / (1e-10)
    # m[kg] = m[amu] * 1.66054e-27
    # a[m/s²] = F[kcal/mol/Å] * (4184 / 6.022e23 / 1e-10) / (m[amu] * 1.66054e-27)
    # Then convert to Å/fs²: 1 Å/fs² = 1e-10/(1e-15)² = 1e20 m/s²
    # So a[Å/fs²] = a[m/s²] / 1e20

    kcal_to_j = 4184.0 / AVOGADRO   # J per (kcal/mol)
    # F[N] = F[kcal/mol/A] * kcal_to_j / 1e-10  (energy per Å -> energy per m)
    # a[m/s^2] = F[N] / m[kg];  a[A/fs^2] = a[m/s^2] / 1e20
    # -> conv = kcal_to_j * 1e10 / (amu_to_kg * 1e20) = kcal_to_j / (amu_to_kg * 1e10)
    conv = kcal_to_j / (amu_to_kg * 1e10)  # kcal/mol/Å per amu -> Å/fs²

    acc = forces * conv / masses[:, np.newaxis]  # Å/fs²

    # Half-step velocity update
    v_half = velocities + 0.5 * dt * acc

    # Position update
    new_coords = coords + dt * v_half

    # Compute new forces
    new_energy, new_forces = compute_forces(new_coords, lj, bonds, angles, cutoff)
    new_acc = new_forces * conv / masses[:, np.newaxis]

    # Second half-step velocity update
    new_velocities = v_half + 0.5 * dt * new_acc

    # Berendsen thermostat
    cur_temp = compute_temperature(new_velocities, masses)
    new_velocities = berendsen_thermostat(
        new_velocities, cur_temp, target_temp, tau=tau, dt=dt
    )

    return new_coords, new_velocities, new_forces, new_energy


# ═══════════════════════════════════════════════════════════════════════════════
# RMSD and radius of gyration
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rmsd(coords: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-square deviation between two coordinate sets.

    Both sets must describe the same atoms in the same order; a mismatched
    length used to broadcast or raise deep inside numpy, and an empty pair
    produced nan from the mean of nothing.
    """
    coords = _as_float_coords(coords)
    reference = _as_float_coords(reference, "reference")
    if coords.shape != reference.shape:
        raise ValueError(
            f"RMSD needs two coordinate sets of the same shape, got "
            f"{coords.shape} and {reference.shape}."
        )
    if len(coords) == 0:
        raise ValueError("RMSD of an empty coordinate set is undefined.")
    diff = coords - reference
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_radius_of_gyration(coords: np.ndarray, masses: np.ndarray) -> float:
    """
    Mass-weighted radius of gyration.

    Rg = sqrt( sum_i m_i |r_i - COM|^2 / sum_i m_i )
    """
    coords = _as_float_coords(coords)
    masses = np.asarray(masses, dtype=np.float64)
    _require_matching_length(masses, len(coords), "masses")
    total_mass = float(np.sum(masses))
    if len(coords) == 0 or total_mass <= 0:
        raise ValueError(
            f"radius of gyration needs atoms with mass, got {len(coords)} "
            f"atom(s) totalling {total_mass} amu."
        )
    com = np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass
    diff = coords - com
    rg = np.sqrt(np.sum(masses * np.sum(diff * diff, axis=1)) / total_mass)
    return float(rg)


# ═══════════════════════════════════════════════════════════════════════════════
# PDB output
# ═══════════════════════════════════════════════════════════════════════════════

def _fit(text: str, width: int, field: str, label: str) -> str:
    """Keep a PDB column to the number of characters the format reserves."""
    if len(text) > width:
        raise ValueError(
            f"PDB {field} of {label} renders as {text!r}, {len(text)} characters "
            f"in a {width}-column field. Writing it anyway shifts every later "
            f"column, and the record is dropped when the file is read back."
        )
    return text


def _checked_position(position: np.ndarray, label: str) -> None:
    """Reject a position that has no PDB representation."""
    for name, value in zip(("x", "y", "z"), position):
        if not np.isfinite(value):
            raise ValueError(
                f"PDB {name} coordinate of {label} is {float(value)!r}; a nan or "
                f"infinite position cannot be written to a file."
            )


def write_pdb(coords: np.ndarray, output_file: str,
              pdb: Optional[PDBStructure] = None,
              model_num: int = 1):
    """
    Write a PDB file, preserving original metadata when available.

    Raises
    ------
    ValueError
        If a value cannot be represented inside the fixed PDB columns, or if
        ``model_num`` is not a positive model index.
    """
    if model_num < 1:
        raise ValueError(f"model_num must be 1 or greater, got {model_num}.")
    coords = _as_float_coords(coords)
    n_atoms = len(coords)

    if pdb is not None and pdb.n_atoms != n_atoms:
        logger.warning(
            "PDB output: %d coordinate row(s) but %d parsed atom(s); the "
            "residue names, chains and B-factors cannot be matched up, so "
            "every atom is written as the ALA A 1 placeholder.",
            n_atoms, pdb.n_atoms,
        )

    with open(output_file, 'w', encoding="utf-8") as f:
        if model_num == 1:
            f.write("REMARK   Generated by BioSuite MD simulation\n")
        f.write("MODEL     "
                + _fit(f"{model_num:4d}", 4, "model number", f"model {model_num}") + "\n")

        if pdb is not None and pdb.n_atoms == n_atoms:
            for i, atom in enumerate(pdb.atoms):
                label = (f"atom {atom.serial} ({atom.name} of "
                         f"{atom.res_name} {atom.chain_id}{atom.res_seq})")
                x, y, z = coords[i]
                _checked_position((x, y, z), label)
                f.write(
                    "ATOM  "
                    + _fit(f"{atom.serial:5d}", 5, "serial", label)
                    + " "
                    + _fit(f"{atom.name:<4s}", 4, "atom name", label)
                    + _fit(f"{atom.alt_loc:<1s}", 1, "alternate location", label)
                    + _fit(f"{atom.res_name:>3s}", 3, "residue name", label)
                    + " "
                    + _fit(f"{atom.chain_id:<1s}", 1, "chain id", label)
                    + _fit(f"{atom.res_seq:4d}", 4, "residue number", label)
                    + _fit(f"{atom.icode:<1s}", 1, "insertion code", label)
                    + "   "
                    + _fit(f"{x:8.3f}", 8, "x coordinate", label)
                    + _fit(f"{y:8.3f}", 8, "y coordinate", label)
                    + _fit(f"{z:8.3f}", 8, "z coordinate", label)
                    + _fit(f"{atom.occupancy:6.2f}", 6, "occupancy", label)
                    + _fit(f"{atom.temp_factor:6.2f}", 6, "B-factor", label)
                    + "          "
                    + _fit(f"{atom.element:>2s}", 2, "element", label)
                    + "\n"
                )
        else:
            for i, (x, y, z) in enumerate(coords):
                label = f"atom {i + 1}"
                _checked_position((x, y, z), label)
                f.write(
                    "ATOM  "
                    + _fit(f"{i + 1:5d}", 5, "serial", label)
                    + f" {'CA':<4s} {'ALA':>3s} A   1    "
                    + _fit(f"{x:8.3f}", 8, "x coordinate", label)
                    + _fit(f"{y:8.3f}", 8, "y coordinate", label)
                    + _fit(f"{z:8.3f}", 8, "z coordinate", label)
                    + "  1.00  0.00           C  \n"
                )
        f.write("ENDMDL\n")
        f.write("END\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Builtin simulation driver
# ═══════════════════════════════════════════════════════════════════════════════

def _builtin_simulate(pdb_file: str, output_pdb: Optional[str] = None,
                      steps: int = 1000, temperature: float = 300.0,
                      dt: float = 1.0, cutoff: float = 10.0,
                      minimize_steps: int = 200, tau: float = 100.0,
                      seed: Optional[int] = None) -> MDSimulationResult:
    """
    Run energy minimisation followed by MD with Velocity Verlet + Berendsen.

    Parameters
    ----------
    pdb_file : path to PDB
    output_pdb : optional output path
    steps : MD steps after minimisation
    temperature : target temperature in K
    dt : timestep in fs
    cutoff : LJ cutoff in Å
    minimize_steps : number of minimisation steps before MD
    tau : Berendsen coupling time in fs
    seed : seed for the Maxwell-Boltzmann draw; None leaves it unseeded

    Returns
    -------
    MDSimulationResult
    """
    for name, value in (("steps", steps), ("minimize_steps", minimize_steps)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        if value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}.")
    if dt <= 0:
        raise ValueError(f"dt must be a positive timestep in fs, got {dt}.")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be a positive distance in Å, got {cutoff}.")
    if tau <= 0:
        raise ValueError(f"tau must be a positive coupling time in fs, got {tau}.")
    if temperature < 0:
        raise ValueError(
            f"temperature cannot be negative, got {temperature} K; the "
            f"Maxwell-Boltzmann spread sqrt(kB*T/m) would be imaginary."
        )

    pdb = parse_pdb(pdb_file)
    if pdb.n_atoms == 0:
        return MDSimulationResult(
            engine='builtin',
            message=f"No atoms found in {pdb_file}.",
        )

    coords = pdb.coordinates.copy()
    masses = pdb.masses
    lj = assign_lj_parameters(pdb.atoms)
    bonds = _build_bond_list(pdb)
    angles = _build_angle_list(bonds)

    # ── Energy Minimisation ──────────────────────────────────────────────────
    coords, min_energy_history = minimize_energy(
        coords, lj, bonds, angles,
        max_steps=minimize_steps, cutoff=cutoff,
    )

    # ── Initialise velocities ────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    velocities = _init_velocities(masses, temperature, rng=rng)
    energy, forces = compute_forces(coords, lj, bonds, angles, cutoff)

    initial_coords = coords.copy()
    energy_history = list(min_energy_history)
    temp_history: List[float] = []
    rg_history: List[float] = []

    # Record initial observables
    temp_history.append(compute_temperature(velocities, masses))
    rg_history.append(compute_radius_of_gyration(coords, masses))

    # ── Velocity Verlet MD loop ──────────────────────────────────────────────
    for step in range(steps):
        coords, velocities, forces, energy = velocity_verlet_step(
            coords, velocities, forces, masses,
            lj, bonds, angles,
            dt=dt, cutoff=cutoff,
            target_temp=temperature, tau=tau,
        )
        energy_history.append(energy)

        # Sample every ten steps, and always the last one: reading the final
        # temperature off this list used to report step 10 of a 15-step run.
        if (step + 1) % 10 == 0 or step == steps - 1:
            temp_history.append(compute_temperature(velocities, masses))
            rg_history.append(compute_radius_of_gyration(coords, masses))

    # ── Observables ──────────────────────────────────────────────────────────
    rmsd_val = compute_rmsd(coords, initial_coords)
    # amu*(A/fs)^2 -> J (per simulated system) -> J/mol -> kJ/mol -> kcal/mol.
    # The raw joule value used to be reported as-is under a 'kcal/mol' label,
    # which made the report print 0.00 for every run.
    final_ke = (0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
                * amu_to_kg * 1e10 * AVOGADRO * KJ_TO_KCAL / 1000.0)  # kcal/mol

    if output_pdb is None:
        # splitext, not str.replace('.pdb', ...): an input named e.g.
        # '1ubq.ent' or '1UBQ.PDB' matched nothing and the minimised
        # coordinates were written straight back over the source file.
        suffix = '_minimized.pdb' if steps == 0 else '_md_final.pdb'
        output_pdb = os.path.splitext(pdb_file)[0] + suffix
    write_pdb(coords, output_pdb, pdb)

    return MDSimulationResult(
        engine='builtin',
        steps=minimize_steps + steps,
        energy=float(energy_history[-1]),
        kinetic_energy=float(final_ke),
        temperature=float(temp_history[-1]),
        rmsd=[rmsd_val],
        radius_gyration=rg_history,
        energy_history=energy_history,
        temperature_history=temp_history,
        output_pdb=output_pdb,
        message=(
            f"Minimised {minimize_steps} steps, MD {steps} steps "
            f"(dt={dt} fs, cutoff={cutoff} Å, tau={tau} fs), "
            f"T_target={temperature} K"
            + (f", seed={seed}" if seed is not None else "")
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OpenMM wrapper
# ═══════════════════════════════════════════════════════════════════════════════

OPENMM_MINIMISE_ITERATIONS = 500


def _openmm_simulate(pdb_file, output_pdb=None, steps=1000, temperature=300):
    """Minimise and integrate with OpenMM.

    Returns ``(result, None)`` on success and ``(None, reason)`` when the engine
    is unavailable or the run failed, so the caller decides what to do instead of
    recognising a failure by the wording of a message.  The final positions are
    always written, with a name derived from the input when the caller does not
    choose one, exactly as the built-in engine does.
    """
    if not HAS_OPENMM:
        return None, "openmm is not installed"
    try:
        if steps < 0:
            raise ValueError(f"steps cannot be negative, got {steps}.")
        if temperature < 0:
            raise ValueError(f"temperature cannot be negative, got {temperature}.")

        pdb = PDBFile(pdb_file)
        forcefield = ForceField('amber14-all.xml')
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometers)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            constraints=HBonds,
        )
        integrator = LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1 / unit.picosecond,
            2 * unit.femtoseconds,
        )
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        simulation.minimizeEnergy(maxIterations=OPENMM_MINIMISE_ITERATIONS)
        # The MD itself: minimizeEnergy alone never moved a single step, so a
        # call that reported "1000 steps" had integrated nothing.
        simulation.step(steps)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        potential_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic_kj = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        # Keep the nanometre Quantity: PDBFile.writeFile expects positions with
        # length units, and the bare float array it used to receive is either
        # rejected or written 1e7 times too small depending on the OpenMM version.
        positions = state.getPositions(asNumpy=True)

        if output_pdb is None:
            output_pdb = os.path.splitext(pdb_file)[0] + (
                '_minimized.pdb' if steps == 0 else '_md_final.pdb'
            )
        # writeFile takes an open handle; passing bare open() left it to the
        # garbage collector, so the trajectory file was flushed late and the
        # descriptor stayed open for the rest of the run.
        with open(output_pdb, 'w', encoding="utf-8") as handle:
            PDBFile.writeFile(simulation.topology, positions, handle)

        result = MDSimulationResult(
            engine='openmm',
            steps=steps,
            # OpenMM reports kJ/mol; every other number this module publishes is
            # kcal/mol, so the conversion happens here rather than in the label.
            energy=float(potential_kj) * KJ_TO_KCAL,
            kinetic_energy=float(kinetic_kj) * KJ_TO_KCAL,
            temperature=float(temperature),
            output_pdb=output_pdb,
            message=(
                f"OpenMM: minimised {OPENMM_MINIMISE_ITERATIONS} iterations, "
                f"integrated {steps} steps, potential energy "
                f"{potential_kj:.2f} kJ/mol"
            ),
        )
        return result, None
    except Exception as exc:
        logger.warning("OpenMM run failed: %s: %s", type(exc).__name__, exc)
        return None, f"{type(exc).__name__}: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def check_md_tools():
    """Which optional MD engines this install can reach."""
    return {'openmm': HAS_OPENMM}


def run_simulation(pdb_file, output_pdb=None, steps=1000, temperature=300,
                   tool='auto', dt=1.0, cutoff=10.0, minimize_steps=200,
                   tau=100.0, seed=None):
    """
    Run molecular dynamics simulation.

    Parameters
    ----------
    pdb_file : str
    output_pdb : str, optional
    steps : int – MD integration steps
    temperature : float – target temperature in K
    tool : 'auto', 'openmm', or 'builtin'
    dt : float – timestep in fs (builtin only)
    cutoff : float – LJ cutoff in Å (builtin only)
    minimize_steps : int – minimisation steps before MD (builtin only)
    tau : float – Berendsen coupling time in fs (builtin only)
    seed : int, optional – seeds the Maxwell-Boltzmann draw (builtin only) so a
        reported trajectory can be reproduced

    Raises
    ------
    ValueError
        If ``tool`` is not one of the three documented engines.
    """
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {TOOLS}, got {tool!r}.")

    if not os.path.isfile(pdb_file):
        reason = "is a directory" if os.path.isdir(pdb_file) else "not found"
        return MDSimulationResult(
            engine='none',
            message=f"File {reason}: {pdb_file}",
        )

    if tool in ('openmm', 'auto') and HAS_OPENMM:
        result, failure = _openmm_simulate(pdb_file, output_pdb, steps, temperature)
        if result is not None:
            return result
        if tool == 'openmm':
            # The user named this engine; running another one quietly would
            # report a simulation that was not the one asked for.
            return MDSimulationResult(
                engine='openmm',
                message=f"OpenMM could not run {pdb_file}: {failure}",
            )
        logger.warning(
            "OpenMM failed (%s); continuing with the built-in engine, which "
            "uses a coarse-grained force field and no solvent.", failure
        )
    elif tool == 'openmm':
        return MDSimulationResult(
            engine='openmm',
            message=f"OpenMM could not run {pdb_file}: openmm is not installed",
        )

    return _builtin_simulate(
        pdb_file, output_pdb, steps, temperature,
        dt=dt, cutoff=cutoff, minimize_steps=minimize_steps, tau=tau, seed=seed,
    )


def format_md_report(result: MDSimulationResult) -> str:
    """Human-readable report string."""
    def number(value: Optional[float], spec: str, unit: str) -> str:
        if value is None:
            return "(not run)"
        return f"{value:{spec}}{unit}"

    lines = [
        "═══ Molecular Dynamics Report ═══",
        f"  Engine        : {result.engine}",
        f"  Total steps   : {result.steps}",
        f"  Potential E   : {number(result.energy, '.2f', ' kcal/mol')}",
        f"  Kinetic E     : {number(result.kinetic_energy, '.2f', ' kcal/mol')}",
        f"  Temperature   : {number(result.temperature, '.1f', ' K')}",
    ]
    if result.rmsd:
        lines.append(f"  RMSD          : {result.rmsd[-1]:.3f} Å")
    if result.radius_gyration:
        lines.append(f"  Rg (final)    : {result.radius_gyration[-1]:.3f} Å")
        if len(result.radius_gyration) > 1:
            lines.append(f"  Rg (initial)  : {result.radius_gyration[0]:.3f} Å")
    if result.output_pdb:
        lines.append(f"  Output PDB    : {result.output_pdb}")
    if result.message:
        lines.append(f"  Note          : {result.message}")
    lines.append("═" * 40)
    return "\n".join(lines)
