"""
Molecular dynamics simulation with dual-mode execution.

Pure Python with numpy as default engine, OpenMM as optional.
Implements:
- Lennard-Jones potential with cutoff and tail corrections
- Velocity Verlet integrator
- Berendsen thermostat for temperature control
- Steepest-descent energy minimization
- Proper PDB parsing with residue/chain metadata
- Correct radius of gyration calculation
"""
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

try:
    from openmm.app import PDBFile, ForceField, Simulation, Modeller
    from openmm import unit, LangevinMiddleIntegrator
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants and default parameters
# ═══════════════════════════════════════════════════════════════════════════════

BOLTZMANN_K = 1.380649e-23      # J/K
AVOGADRO = 6.02214076e23        # mol^-1
KCAL_TO_KJ = 4.184              # 1 kcal/mol = 4.184 kJ/mol
KJ_TO_KCAL = 1.0 / KCAL_TO_KJ
amu_to_kg = 1.66053906660e-27   # 1 amu in kg
angstrom_to_m = 1e-10
AMBER_ATOMIC_MASS = {           # amu – common amino-acid heavy atoms
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06,
    'H': 1.008,  'CA': 12.011, 'CB': 12.011,
}

# Default LJ parameters (OPLS-AA-like, sigma in Å, epsilon in kcal/mol)
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
        """Atomic masses in amu."""
        return np.array([
            AMBER_ATOMIC_MASS.get(a.element.strip().upper(), 12.011)
            for a in self.atoms
        ])

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
    engine: str
    steps: int = 0
    energy: float = 0.0
    kinetic_energy: float = 0.0
    temperature: float = 300.0
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
    'FE': 'FE', 'ZN': 'ZN', 'MG': 'MG', 'CA': 'CA',
}


def _guess_element(atom_name: str) -> str:
    """Guess element from atom name (PDB convention: first letter is element)."""
    clean = atom_name.strip().lstrip('0123456789')
    first = clean[0].upper() if clean else 'C'
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
    """
    atoms: List[Atom] = []
    remarks: List[str] = []
    title = ""
    seen_alt: Dict[Tuple[str, int], bool] = {}

    with open(pdb_path, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n\r')
            record = line[:6].strip()

            if record == 'TITLE':
                title = (title + ' ' + line[10:].strip()).strip()
            elif record == 'REMARK':
                remarks.append(line)
            elif record in ('ATOM', 'HETATM'):
                try:
                    serial = int(line[6:11])
                except ValueError:
                    serial = len(atoms) + 1
                name = line[12:16].strip()
                alt_loc = line[16].strip() or ' '
                res_name = line[17:20].strip()
                chain_id = line[21].strip() or 'A'
                try:
                    res_seq = int(line[22:26])
                except ValueError:
                    res_seq = 0
                icode = line[26].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
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

                # Alt location handling
                key = (name, res_seq)
                if alt_loc != ' ':
                    if key in seen_alt:
                        if alt_loc_strategy == 'first':
                            continue  # skip subsequent alt locs
                        elif alt_loc_strategy == 'last':
                            # will be replaced below
                            pass
                    seen_alt[key] = True

                atoms.append(Atom(
                    serial=serial, name=name, alt_loc=alt_loc,
                    res_name=res_name, chain_id=chain_id,
                    res_seq=res_seq, icode=icode,
                    x=x, y=y, z=z,
                    occupancy=occupancy, temp_factor=temp_factor,
                    element=element,
                ))

    return PDBStructure(atoms=atoms, title=title, remarks=remarks)


# ═══════════════════════════════════════════════════════════════════════════════
# Lennard-Jones potential
# ═══════════════════════════════════════════════════════════════════════════════

def assign_lj_parameters(atoms: List[Atom]) -> LJParameters:
    """Assign LJ parameters from atom names / element types."""
    n = len(atoms)
    sigma = np.zeros(n)
    epsilon = np.zeros(n)
    for i, atom in enumerate(atoms):
        name = atom.name.strip().upper()
        elem = atom.element.strip().upper()
        params = DEFAULT_LJ.get(name) or DEFAULT_LJ.get(elem) or _DEFAULT_LJ_ENTRY
        sigma[i] = params['sigma']
        epsilon[i] = params['epsilon']
    return LJParameters(sigma=sigma, epsilon=epsilon)


def lj_cutoff_correction(sigma: np.ndarray, epsilon: np.ndarray,
                         cutoff: float, n: int) -> float:
    """
    Long-range tail correction for LJ energy.

    U_tail = (8/3) * pi * N/V * sum_ij eps_ij * sigma_ij^3 *
             [ (sigma_ij/cutoff)^9 / 3 - (sigma_ij/cutoff)^3 / 2 ]

    Returns energy in kcal/mol (assumes box volume = (cutoff*2)^3 per atom for a
    rough approximation).
    """
    # Use a simple pairwise average approximation
    n_pairs = n * (n - 1) / 2
    avg_sigma = np.mean(sigma)
    avg_eps = np.mean(epsilon)
    sr = avg_sigma / cutoff
    sr3 = sr ** 3
    sr9 = sr3 ** 3
    vol_per_atom = (2.0 * cutoff) ** 3
    V = vol_per_atom * max(n, 1)
    correction = (8.0 / 3.0) * math.pi * n / V * avg_eps * avg_sigma**3 * (
        sr9 / 3.0 - sr3 / 2.0
    )
    return correction


def compute_lj_energy(coords: np.ndarray, lj: LJParameters,
                      cutoff: float = 10.0) -> Tuple[float, np.ndarray]:
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

    Returns
    -------
    energy : float (kcal/mol)
    forces : (N, 3) array (kcal/mol/Å)
    """
    n = len(coords)
    forces = np.zeros_like(coords)
    energy = 0.0
    cutoff_sq = cutoff * cutoff

    sigma_i = lj.sigma
    epsilon_i = lj.epsilon

    for i in range(n - 1):
        # Vectorized distances from atom i to atoms i+1..N-1
        diff = coords[i + 1:] - coords[i]          # (M, 3)
        dist_sq = np.sum(diff * diff, axis=1)      # (M,)
        mask = (dist_sq < cutoff_sq) & (dist_sq > 1e-6)
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

        # F = -dV/dr * (r_vec / r) = 24 * eps * (2*sr12 - sr6) / r  * (r_vec/r)
        # So F_vec = 24 * eps * (2*sr12 - sr6) / r2 * diff
        f_scalar = 24.0 * eps_ij * (2.0 * sr12 - sr6) / r2

        f_vec = f_scalar[:, np.newaxis] * diff[idx]   # (M, 3)
        forces[i] += np.sum(f_vec, axis=0)
        for k, j in enumerate(idx):
            forces[i + 1 + j] -= f_vec[k]

    # Tail correction (approximate)
    energy += lj_cutoff_correction(lj.sigma, lj.epsilon, cutoff, n)

    return energy, forces


# ═══════════════════════════════════════════════════════════════════════════════
# Spring (harmonic bond) potential
# ═══════════════════════════════════════════════════════════════════════════════

def _build_bond_list(pdb: PDBStructure, max_bond_len: float = 2.0) -> List[Tuple[int, int, float]]:
    """
    Infer bonds from inter-atomic distances.
    Returns list of (i, j, equilibrium_length) in Å.
    """
    coords = pdb.coordinates
    n = pdb.n_atoms
    bonds = []
    for i in range(n - 1):
        diff = coords[i + 1:] - coords[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        close = np.where((dist < max_bond_len) & (dist > 0.5))[0]
        for k in close:
            bonds.append((i, i + 1 + k, dist[k]))
    return bonds


def compute_bond_energy(coords: np.ndarray, bonds: List[Tuple[int, int, float]],
                        k_bond: float = 500.0) -> Tuple[float, np.ndarray]:
    """
    Harmonic bond potential: E = 0.5 * k * (r - r0)^2.
    k_bond in kcal/mol/Å^2.
    """
    forces = np.zeros_like(coords)
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
    forces = np.zeros_like(coords)
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
        # Gradient of angle w.r.t. coords (simplified)
        coeff = k_angle * dtheta / math.sin(theta)
        # d(theta)/d(v1) = (cos_theta * v1/n1 - v2/n2) / (n1)
        grad_v1 = coeff * (cos_theta * v1 / (n1 * n1) - v2 / (n1 * n2))
        grad_v2 = coeff * (cos_theta * v2 / (n2 * n2) - v1 / (n1 * n2))
        forces[i] += grad_v1
        forces[k] += grad_v2
        forces[j] -= (grad_v1 + grad_v2)

    return energy, forces


# ═══════════════════════════════════════════════════════════════════════════════
# Total force computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_forces(coords: np.ndarray, lj: LJParameters,
                   bonds: List, angles: List,
                   cutoff: float = 10.0) -> Tuple[float, np.ndarray]:
    """
    Total potential energy and forces from LJ + bonds + angles.
    """
    e_lj, f_lj = compute_lj_energy(coords, lj, cutoff)
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
    energy_history : list of energies per step
    """
    coords = coords.copy()
    energy_history = []
    step_size = initial_step

    energy, forces = compute_forces(coords, lj, bonds, angles, cutoff)
    energy_history.append(energy)

    for step in range(max_steps):
        fmag = np.sqrt(np.sum(forces * forces))
        if fmag < convergence_tol:
            break

        # Normalize force direction, scale by step_size
        direction = forces / fmag
        trial_coords = coords + step_size * direction

        trial_energy, trial_forces = compute_forces(trial_coords, lj, bonds, angles, cutoff)

        if trial_energy < energy:
            # Accept
            coords = trial_coords
            energy = trial_energy
            forces = trial_forces
            step_size *= 1.2  # increase step
        else:
            # Reject, reduce step
            step_size *= 0.5

        energy_history.append(energy)

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
    target_temp : K

    Returns
    -------
    velocities : (N, 3) in Å/fs  (1 Å/fs = 1e5 m/s)
    """
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
    n = len(masses)
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)

    # Convert KE from amu*(Å/fs)^2 to Joules
    # 1 amu = 1.66054e-27 kg, 1 Å/fs = 1e5 m/s
    ke_joules = ke * amu_to_kg * 1e10  # amu * (Å/fs)^2 -> kg*(m/s)^2 = J

    if n <= 0:
        return 0.0
    T = 2.0 * ke_joules / (3.0 * n * BOLTZMANN_K)
    return T


def berendsen_thermostat(velocities: np.ndarray, masses: np.ndarray,
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
    """
    if current_temp < 1e-6:
        return velocities
    lam_sq = 1.0 + (dt / tau) * (target_temp / current_temp - 1.0)
    if lam_sq < 0:
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
    conv = kcal_to_j / (amu_to_kg * 1e20)  # kcal/mol/Å per amu -> Å/fs²

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
        new_velocities, masses, cur_temp, target_temp, tau=tau, dt=dt
    )

    return new_coords, new_velocities, new_forces, new_energy


# ═══════════════════════════════════════════════════════════════════════════════
# RMSD and radius of gyration
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rmsd(coords: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-square deviation between two coordinate sets."""
    diff = coords - reference
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_radius_of_gyration(coords: np.ndarray, masses: np.ndarray) -> float:
    """
    Mass-weighted radius of gyration.

    Rg = sqrt( sum_i m_i |r_i - COM|^2 / sum_i m_i )
    """
    com = np.sum(masses[:, np.newaxis] * coords, axis=0) / np.sum(masses)
    diff = coords - com
    rg = np.sqrt(np.sum(masses * np.sum(diff * diff, axis=1)) / np.sum(masses))
    return float(rg)


# ═══════════════════════════════════════════════════════════════════════════════
# PDB output
# ═══════════════════════════════════════════════════════════════════════════════

def write_pdb(coords: np.ndarray, output_file: str,
              pdb: Optional[PDBStructure] = None,
              model_num: int = 1):
    """Write a PDB file, preserving original metadata when available."""
    with open(output_file, 'w') as f:
        if model_num == 1:
            f.write("REMARK   Generated by BioSuite MD simulation\n")
        f.write(f"MODEL     {model_num:4d}\n")

        n_atoms = len(coords)
        if pdb and pdb.n_atoms == n_atoms:
            for i, atom in enumerate(pdb.atoms):
                x, y, z = coords[i]
                f.write(
                    f"ATOM  {atom.serial:5d} {atom.name:4s}{atom.alt_loc}"
                    f"{atom.res_name:>3s} {atom.chain_id}"
                    f"{atom.res_seq:4d}{atom.icode}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{atom.occupancy:6.2f}{atom.temp_factor:6.2f}"
                    f"          {atom.element:>2s}  \n"
                )
        else:
            for i, (x, y, z) in enumerate(coords):
                f.write(
                    f"ATOM  {i+1:5d} {'CA':4s} "
                    f"ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           C  \n"
                )
        f.write(f"ENDMDL\n")
        f.write("END\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Builtin simulation driver
# ═══════════════════════════════════════════════════════════════════════════════

def _builtin_simulate(pdb_file: str, output_pdb: Optional[str] = None,
                      steps: int = 1000, temperature: float = 300.0,
                      dt: float = 1.0, cutoff: float = 10.0,
                      minimize_steps: int = 200) -> MDSimulationResult:
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

    Returns
    -------
    MDSimulationResult
    """
    pdb = parse_pdb(pdb_file)
    if pdb.n_atoms == 0:
        return MDSimulationResult(engine='builtin',
                                  message="No atoms found in PDB file.")

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
    velocities = _init_velocities(masses, temperature)
    energy, forces = compute_forces(coords, lj, bonds, angles, cutoff)

    initial_coords = coords.copy()
    energy_history = list(min_energy_history)
    temp_history: List[float] = []
    rg_history: List[float] = []

    # Record initial observables
    T0 = compute_temperature(velocities, masses)
    temp_history.append(T0)
    rg_history.append(compute_radius_of_gyration(coords, masses))

    # ── Velocity Verlet MD loop ──────────────────────────────────────────────
    for step in range(steps):
        coords, velocities, forces, energy = velocity_verlet_step(
            coords, velocities, forces, masses,
            lj, bonds, angles,
            dt=dt, cutoff=cutoff,
            target_temp=temperature, tau=100.0,  # tau = 100 fs (weak coupling)
        )
        energy_history.append(energy)

        if step % 10 == 0:
            T = compute_temperature(velocities, masses)
            temp_history.append(T)
            rg_history.append(compute_radius_of_gyration(coords, masses))

    # ── Observables ──────────────────────────────────────────────────────────
    rmsd_val = compute_rmsd(coords, initial_coords)
    final_ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2) * amu_to_kg * 1e10

    if output_pdb is None:
        output_pdb = pdb_file.replace('.pdb', '_minimized.pdb')
    write_pdb(coords, output_pdb, pdb)

    return MDSimulationResult(
        engine='builtin',
        steps=minimize_steps + steps,
        energy=energy_history[-1],
        kinetic_energy=float(final_ke),
        temperature=temp_history[-1] if temp_history else temperature,
        rmsd=[rmsd_val],
        radius_gyration=rg_history,
        energy_history=energy_history,
        temperature_history=temp_history,
        output_pdb=output_pdb,
        message=(
            f"Minimised {minimize_steps} steps, MD {steps} steps "
            f"(dt={dt} fs, cutoff={cutoff} Å), "
            f"T_target={temperature} K"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy helper – kept for backward compat with tests
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_coords_from_pdb(pdb_file):
    """Extract coordinates only (legacy API)."""
    pdb = parse_pdb(pdb_file)
    return pdb.coordinates


def _builtin_minimize(pdb_file, output_pdb=None, max_steps=1000):
    """Legacy API: energy minimisation only (no MD), returns MDSimulationResult."""
    return _builtin_simulate(
        pdb_file, output_pdb, steps=0, temperature=300.0,
        minimize_steps=max_steps,
    )


def _write_pdb(coords, output_file, atom_names=None):
    """Write minimal PDB (legacy API)."""
    with open(output_file, 'w') as f:
        for i, (x, y, z) in enumerate(coords):
            name = atom_names[i] if atom_names and i < len(atom_names) else 'CA'
            f.write(f"ATOM  {i+1:5d} {name:4s} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        f.write("END\n")


# ═══════════════════════════════════════════════════════════════════════════════
# OpenMM wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def _openmm_simulate(pdb_file, output_pdb=None, steps=1000, temperature=300):
    if not HAS_OPENMM:
        return None
    try:
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

        simulation.minimizeEnergy(maxIterations=steps)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)

        if output_pdb:
            PDBFile.writeFile(simulation.topology, positions, open(output_pdb, 'w'))

        return MDSimulationResult(
            engine='openmm',
            steps=steps,
            energy=round(energy, 2),
            output_pdb=output_pdb or '',
            message=f"OpenMM: energy={energy:.2f} kJ/mol",
        )
    except Exception as e:
        return MDSimulationResult(engine='openmm', message=f"OpenMM error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def check_md_tools():
    return {'openmm': HAS_OPENMM}


def run_simulation(pdb_file, output_pdb=None, steps=1000, temperature=300,
                   tool='auto', dt=1.0, cutoff=10.0, minimize_steps=200):
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
    """
    if not os.path.exists(pdb_file):
        return MDSimulationResult(engine='none',
                                  message=f"File not found: {pdb_file}")

    tools = check_md_tools()
    if tool in ('openmm', 'auto') and tools['openmm']:
        result = _openmm_simulate(pdb_file, output_pdb, steps, temperature)
        if result and not result.message.startswith("OpenMM error"):
            return result

    return _builtin_simulate(
        pdb_file, output_pdb, steps, temperature,
        dt=dt, cutoff=cutoff, minimize_steps=minimize_steps,
    )


def format_md_report(result: MDSimulationResult) -> str:
    """Human-readable report string."""
    lines = [
        "═══ Molecular Dynamics Report ═══",
        f"  Engine        : {result.engine}",
        f"  Total steps   : {result.steps}",
        f"  Potential E   : {result.energy:.2f} kcal/mol",
        f"  Kinetic E     : {result.kinetic_energy:.2f} kcal/mol",
        f"  Temperature   : {result.temperature:.1f} K",
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
