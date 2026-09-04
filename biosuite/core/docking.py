"""
Molecular docking with dual-mode execution.

Pure Python simple docking scoring as default, AutoDock Vina as optional.
"""
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np

from .log import get_logger
from .utils import PerformanceWarning

logger = get_logger(__name__)


@dataclass
class DockingResult:
    engine: str
    binding_energy: float = 0.0
    poses: list = field(default_factory=list)
    best_pose_file: str = ""
    num_poses: int = 0
    message: str = ""


@dataclass
class Pose:
    rank: int
    energy: float
    x: float
    y: float
    z: float


from .utils import has_tool as _has_tool


def check_docking_tools():
    return {'vina': _has_tool('vina') or _has_tool('autodock-vina')}


# ── Pure Python Docking Score ───────────────────────────────────────────────

def _parse_pdb_atoms(pdb_file, chain=None):
    atoms = []
    malformed = 0
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    ch = line[21]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if chain is None or ch == chain:
                        atoms.append({'name': atom_name, 'res': res_name,
                                     'chain': ch, 'x': x, 'y': y, 'z': z})
                except (ValueError, IndexError):
                    # Malformed coordinate columns: skip the record but say so
                    # rather than dropping atoms silently.
                    malformed += 1
                    continue
    if malformed:
        logger.warning("%s: skipped %d ATOM/HETATM record(s) with unparseable "
                       "coordinates", pdb_file, malformed)
    return atoms


def _score_contacts(rec_coords, lig_coords, cutoff=5.0, clash=3.0):
    """Crude contact potential for a ligand placement.

    Sums an attractive ``-0.5 / (d + 0.1)`` term over receptor/ligand atom
    pairs within *cutoff* angstroms and adds a steep repulsive penalty for
    pairs closer than *clash* angstroms.

    Warning:
        This is a **heuristic geometric score**, not a physics-based free
        energy.  It has no solvation, electrostatics, torsional or entropic
        terms and it is not calibrated against experimental affinities.  The
        value is expressed in arbitrary units and must not be interpreted as
        kcal/mol.  Install AutoDock Vina for a validated scoring function.
    """
    if len(rec_coords) == 0 or len(lig_coords) == 0:
        return 0.0
    # (n_lig, n_rec) pairwise distances, computed in blocks so a large
    # receptor does not allocate an n_lig x n_rec array all at once.
    energy = 0.0
    block = max(1, int(2_000_000 // max(len(rec_coords), 1)))
    for start in range(0, len(lig_coords), block):
        chunk = lig_coords[start:start + block]
        d = np.sqrt(((chunk[:, None, :] - rec_coords[None, :, :]) ** 2).sum(-1))
        near = d < cutoff
        if np.any(near):
            energy -= float(np.sum(1.0 / (d[near] + 0.1)) * 0.5)
        clashing = d < clash
        if np.any(clashing):
            # Soft-core repulsion: burying the ligand inside the protein must
            # not be scored as the best possible pose.
            energy += float(np.sum((clash - d[clashing]) ** 2)) * 2.0
    return float(energy)


def _compute_binding_energy(receptor_atoms, ligand_atoms):
    """Score the ligand *as supplied*, without searching for a pose.

    Warning:
        Arbitrary units, heuristic — see :func:`_score_contacts`.
    """
    if not receptor_atoms or not ligand_atoms:
        return 0.0
    rec_coords = np.array([[a['x'], a['y'], a['z']] for a in receptor_atoms],
                          dtype=float)
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms],
                          dtype=float)
    return round(_score_contacts(rec_coords, lig_coords), 2)


def _random_rotation_matrices(rng, n):
    """*n* uniformly distributed 3D rotation matrices (Shoemake's method)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q = np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=1)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def _builtin_dock(receptor_file, ligand_file, center=None, num_poses=5,
                  box_size=(20.0, 20.0, 20.0), n_trials=400, seed=0):
    """Rigid-body pose search with a heuristic contact score.

    The ligand is rigidly rotated and translated inside a search box around
    *center*; every candidate placement is **actually scored** and the best
    *num_poses* placements are returned, with the coordinates of the ligand
    centroid in that pose.

    Previously this function computed a single score for the input geometry
    and then invented ``num_poses`` results by adding Gaussian noise to the
    receptor centroid and uniform noise to that score, so the reported
    "poses" and their "energies" were random numbers unrelated to any docking
    calculation.

    Warning:
        The scoring function is a crude contact potential in **arbitrary
        units** (see :func:`_score_contacts`); the search is a random rigid
        placement scan with no flexibility, no solvation and no calibration.
        Use it for a quick geometric sanity check only.  Install AutoDock
        Vina for quantitative work.

    Args:
        seed: RNG seed.  The search is stochastic, so results are only
            reproducible for a fixed seed; the default makes repeated runs
            deterministic.
    """
    rec_atoms = _parse_pdb_atoms(receptor_file)
    lig_atoms = _parse_pdb_atoms(ligand_file)

    if not rec_atoms or not lig_atoms:
        return DockingResult(engine='builtin',
                             message="Could not parse atoms from input files")
    if num_poses < 1:
        raise ValueError("num_poses must be >= 1")

    rec_coords = np.array([[a['x'], a['y'], a['z']] for a in rec_atoms], dtype=float)
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in lig_atoms], dtype=float)

    if center is None:
        center = rec_coords.mean(axis=0)
    center = np.asarray(center, dtype=float)
    box = np.asarray(box_size, dtype=float)

    lig_local = lig_coords - lig_coords.mean(axis=0)
    rng = np.random.default_rng(seed)
    n_trials = max(int(n_trials), num_poses)
    rotations = _random_rotation_matrices(rng, n_trials)
    offsets = (rng.random((n_trials, 3)) - 0.5) * box

    scored = []
    for rot, offset in zip(rotations, offsets):
        placed = lig_local @ rot.T + center + offset
        scored.append((_score_contacts(rec_coords, placed), center + offset))
    scored.sort(key=lambda item: item[0])

    poses = [
        Pose(rank=rank, energy=round(float(score), 2),
             x=round(float(pos[0]), 3),
             y=round(float(pos[1]), 3),
             z=round(float(pos[2]), 3))
        for rank, (score, pos) in enumerate(scored[:num_poses], start=1)
    ]

    return DockingResult(
        engine='builtin',
        binding_energy=poses[0].energy,
        poses=poses,
        num_poses=len(poses),
        message=(f"Built-in heuristic rigid-body search: {len(poses)} poses "
                 f"from {n_trials} trials, best contact score="
                 f"{poses[0].energy:.2f} (arbitrary units, not kcal/mol)")
    )


# ── Vina Wrapper ────────────────────────────────────────────────────────────

def _parse_vina_output_pdbqt(path):
    """Parse `MODEL <i>` / `REMARK VINA RESULT: <energy>` records into
    (rank, energy) pairs — the only data written for docking poses."""
    energies = []
    import re
    pat = re.compile(r"REMARK\s+VINA RESULT:\s*(-?\d+\.?\d*)")
    try:
        with open(path) as fh:
            for line in fh:
                m = pat.match(line.strip())
                if m:
                    energies.append(float(m.group(1)))
    except OSError:
        return []
    return [(i + 1, e) for i, e in enumerate(energies)]


def _vina_dock(receptor_pdbqt, ligand_pdbqt, center, box_size, num_poses):
    """Run vina; return (rank, energy) pose list or None on failure."""
    # tempfile.mktemp is a TOCTOU race and the file was never deleted;
    # use mkstemp and always unlink afterwards.
    fd, out_path = tempfile.mkstemp(suffix='.pdbqt')
    os.close(fd)
    cmd = ['vina', '--receptor', receptor_pdbqt, '--ligand', ligand_pdbqt,
           '--center_x', str(center[0]), '--center_y', str(center[1]),
           '--center_z', str(center[2]),
           '--size_x', str(box_size[0]), '--size_y', str(box_size[1]),
           '--size_z', str(box_size[2]),
           '--num_modes', str(num_poses), '--out', out_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode == 0:
            return _parse_vina_output_pdbqt(out_path)
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def dock(receptor_file, ligand_file, center=None, box_size=(20, 20, 20),
         num_poses=5, tool='auto'):
    if not os.path.exists(receptor_file):
        return DockingResult(engine='none', message=f"Receptor not found: {receptor_file}")
    if not os.path.exists(ligand_file):
        return DockingResult(engine='none', message=f"Ligand not found: {ligand_file}")

    tools = check_docking_tools()
    if tool in ('vina', 'auto') and tools['vina']:
        vina_poses = _vina_dock(receptor_file, ligand_file, center or [0, 0, 0],
                                box_size, num_poses)
        if vina_poses:
            # Previously the vina run was accepted but its output was
            # thrown away, leaving energy=0 / poses=[] — a silent lie.
            cx, cy, cz = (center or [0, 0, 0])
            poses = [Pose(rank=rank, energy=energy, x=cx, y=cy, z=cz)
                     for rank, energy in vina_poses]
            return DockingResult(
                engine='vina',
                binding_energy=poses[0].energy,
                poses=poses,
                num_poses=len(poses),
                message=(f"AutoDock Vina (external): {len(poses)} poses, "
                         f"best energy={poses[0].energy:.2f} kcal/mol"))

    warnings.warn(
        "AutoDock Vina not found. Falling back to the built-in heuristic "
        "rigid-body search, whose scores are in arbitrary units and are NOT "
        "binding free energies. For quantitative docking install AutoDock "
        "Vina (http://vina.scripps.edu/).",
        PerformanceWarning, stacklevel=2
    )
    return _builtin_dock(receptor_file, ligand_file, center, num_poses,
                         box_size=box_size)


def format_docking_report(result):
    heuristic = result.engine == 'builtin'
    unit = "(arbitrary units)" if heuristic else "kcal/mol"
    lines = [
        "=== Molecular Docking Report ===",
        f"Engine: {result.engine}",
        f"Poses generated: {result.num_poses}",
        f"Best score: {result.binding_energy:.2f} {unit}",
    ]
    if heuristic:
        lines.append(
            "WARNING: heuristic contact score from the built-in rigid-body "
            "search; not a binding free energy and not comparable to Vina.")
    if result.poses:
        lines.append("\nTop poses:")
        for p in result.poses[:3]:
            lines.append(f"  Pose {p.rank}: {p.energy:.2f} {unit} "
                         f"at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)
