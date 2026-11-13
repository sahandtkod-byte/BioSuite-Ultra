"""
Molecular docking with dual-mode execution.

Pure Python simple docking scoring as default, AutoDock Vina as optional.
"""
import math
import os
import re
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

# The unit each engine's numbers may be quoted in.  An engine that never ran has
# no unit at all, which is why this is a lookup and not a `!= 'builtin'` test.
ENGINE_UNITS = {'vina': 'kcal/mol', 'builtin': '(arbitrary units)'}

# The built-in search samples this many placements per run.
DEFAULT_TRIALS = 400


def check_docking_tools():
    return {'vina': _has_tool('vina') or _has_tool('autodock-vina')}


# ── Argument domains ────────────────────────────────────────────────────────

def _as_point(value, name):
    """Return *value* as three finite numbers, or raise ValueError."""
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be three numbers (x, y, z), got {value!r}.") from None
    if len(numbers) != 3:
        raise ValueError(
            f"{name} must be three numbers (x, y, z), got {len(numbers)}: {value!r}.")
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}.")
    return numbers


def _as_box_size(value):
    box = _as_point(value, "box_size")
    if any(size <= 0 for size in box):
        raise ValueError(f"box_size must be positive in every dimension, got {value!r}.")
    return box


def _as_pose_count(num_poses):
    if isinstance(num_poses, bool) or not isinstance(num_poses, int):
        raise ValueError(f"num_poses must be an integer >= 1, got {num_poses!r}.")
    if num_poses < 1:
        raise ValueError(f"num_poses must be >= 1, got {num_poses}.")
    return num_poses


# ── Pure Python Docking Score ───────────────────────────────────────────────

def _xyz(line):
    """The (x, y, z) of a fixed-width ATOM/HETATM record, or None.

    Columns 31-54 hold the three coordinates; a short line or a non-numeric
    field is not a coordinate record and is reported to the caller, which
    counts the ones it threw away.
    """
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except (ValueError, IndexError):
        return None


def _parse_pdb_atoms(pdb_file, chain=None):
    atoms = []
    malformed = 0
    with open(pdb_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    ch = line[21]
                except IndexError:
                    malformed += 1
                    continue
                if chain is not None and ch != chain:
                    continue
                coordinate = _xyz(line)
                if coordinate is None:
                    # Malformed coordinate columns: skip the record but say so
                    # rather than dropping atoms silently.
                    malformed += 1
                    continue
                x, y, z = coordinate
                atoms.append({'name': atom_name, 'res': res_name,
                              'chain': ch, 'x': x, 'y': y, 'z': z})
    if malformed:
        logger.warning("%s: skipped %d ATOM/HETATM record(s) with unparseable "
                       "coordinates", pdb_file, malformed)
    return atoms


def _centroid(coordinate_list):
    """Mean position of a list of (x, y, z) tuples, NaN when there is none.

    NaN rather than (0, 0, 0): a missing position must not look like the origin
    of the coordinate system, which is where the old code put every Vina pose.
    """
    if not coordinate_list:
        return [float('nan')] * 3
    array = np.asarray(coordinate_list, dtype=float)
    return [float(value) for value in array.mean(axis=0)]


def _receptor_centroid(receptor_file):
    """Centroid of a receptor's atoms, or None when it carries no atoms."""
    atoms = _parse_pdb_atoms(receptor_file)
    if not atoms:
        return None
    return _centroid([(a['x'], a['y'], a['z']) for a in atoms])


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
                  box_size=(20.0, 20.0, 20.0), n_trials=DEFAULT_TRIALS, seed=0):
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
    # Checked before the files are read: a run with a bad pose count should
    # fail at once rather than after parsing two PDB files.
    num_poses = _as_pose_count(num_poses)
    if center is not None:
        center = _as_point(center, "center")
    box = _as_box_size(box_size)

    rec_atoms = _parse_pdb_atoms(receptor_file)
    lig_atoms = _parse_pdb_atoms(ligand_file)

    if not rec_atoms or not lig_atoms:
        return DockingResult(engine='builtin',
                             message="Could not parse atoms from input files")

    rec_coords = np.array([[a['x'], a['y'], a['z']] for a in rec_atoms], dtype=float)
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in lig_atoms], dtype=float)

    if center is None:
        center = [float(value) for value in rec_coords.mean(axis=0)]
    center = np.asarray(center, dtype=float)

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

_VINA_RESULT = re.compile(r"REMARK\s+VINA RESULT:\s*(-?\d+(?:\.\d+)?)")


def _parse_vina_output_pdbqt(path):
    """Read `(rank, energy, centroid)` for every pose in a Vina output file.

    Each ``REMARK VINA RESULT`` line opens one pose and the ATOM/HETATM records
    that follow it belong to that pose, so the position reported for a pose is
    the centroid of **its own** atoms.  Reading only the REMARK lines — which is
    all this used to do — threw the coordinates away and left every pose at one
    place: the report printed the centre of the search box as the position of
    every pose, so three different poses looked like the same placement three
    times.  A pose whose file carries no atoms keeps NaN coordinates rather
    than a position nobody measured.
    """
    poses = []
    malformed = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                match = _VINA_RESULT.match(line.strip())
                if match:
                    poses.append([float(match.group(1)), []])
                elif poses and line.startswith(('ATOM', 'HETATM')):
                    coordinate = _xyz(line)
                    if coordinate is None:
                        malformed += 1
                    else:
                        poses[-1][1].append(coordinate)
    except OSError:
        return []
    if malformed:
        logger.warning("%s: skipped %d pose ATOM/HETATM record(s) with "
                       "unparseable coordinates", path, malformed)
    return [(rank, energy, _centroid(coordinates))
            for rank, (energy, coordinates) in enumerate(poses, start=1)]


def _vina_run(cmd, out_path):
    """Execute the Vina command line; return (poses, failure_reason)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        reason = f"the vina process could not be run: {exc}"
        logger.warning("[docking] %s", reason)
        return [], reason

    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        reason = (f"vina exited with code {r.returncode}"
                  + (f": {detail[0]}" if detail else ""))
        logger.warning("[docking] %s", reason)
        return [], reason

    poses = _parse_vina_output_pdbqt(out_path)
    if not poses:
        reason = "vina wrote no 'REMARK VINA RESULT' pose record"
        logger.warning("[docking] %s", reason)
        return [], reason
    return poses, ""


def _vina_dock(receptor_pdbqt, ligand_pdbqt, center, box_size, num_poses):
    """Run vina; return (poses, failure_reason).

    The reason matters because the caller has to tell "vina is not installed"
    apart from "vina ran and failed": both used to be reported as the first.
    """
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
        return _vina_run(cmd, out_path)
    finally:
        try:
            os.unlink(out_path)
        except OSError as exc:
            logger.debug("[docking] pose file %s could not be removed: %s",
                         out_path, exc)


# ── Public API ──────────────────────────────────────────────────────────────

def dock(receptor_file, ligand_file, center=None, box_size=(20, 20, 20),
         num_poses=5, tool='auto'):
    """Dock *ligand_file* into *receptor_file*.

    Args:
        center: the (x, y, z) centre of the search box, in Å.  Defaults to the
            receptor's own centroid, for both engines: an unspecified centre
            used to mean the origin (0, 0, 0) for Vina while the built-in
            search used the receptor, so a real receptor sitting 300 Å away
            from the origin got a box of empty space and nobody was told.
    """
    for label, path in (("Receptor", receptor_file), ("Ligand", ligand_file)):
        if not os.path.exists(path):
            return DockingResult(engine='none', message=f"{label} not found: {path}")
        if not os.path.isfile(path):
            return DockingResult(engine='none', message=f"{label} is not a file: {path}")

    num_poses = _as_pose_count(num_poses)
    box_size = _as_box_size(box_size)
    if center is not None:
        center = _as_point(center, "center")

    tools = check_docking_tools()
    if tool in ('vina', 'auto') and tools['vina']:
        box_centre = (center if center is not None
                      else _receptor_centroid(receptor_file))
        if box_centre is None:
            vina_poses, failure = [], ("the receptor file carries no ATOM or "
                                       "HETATM records, so the search box has "
                                       "no centre to be built around")
        else:
            vina_poses, failure = _vina_dock(receptor_file, ligand_file,
                                             box_centre, box_size, num_poses)
        if vina_poses:
            # Previously the vina run was accepted but its output was
            # thrown away, leaving energy=0 / poses=[] — a silent lie.
            poses = [Pose(rank=rank, energy=energy,
                          x=coordinate[0], y=coordinate[1], z=coordinate[2])
                     for rank, energy, coordinate in vina_poses]
            return DockingResult(
                engine='vina',
                binding_energy=poses[0].energy,
                poses=poses,
                num_poses=len(poses),
                message=(f"AutoDock Vina (external): {len(poses)} poses, "
                         f"best energy={poses[0].energy:.2f} kcal/mol"))

    if tools['vina']:
        text = ("AutoDock Vina was tried but produced no usable poses"
                + (f" ({failure})" if failure else "")
                + ". Falling back to the built-in heuristic rigid-body search, "
                  "whose scores are in arbitrary units and are NOT binding free "
                  "energies. This is not a Vina result.")
    else:
        text = ("AutoDock Vina not found. Falling back to the built-in heuristic "
                "rigid-body search, whose scores are in arbitrary units and are NOT "
                "binding free energies. For quantitative docking install AutoDock "
                "Vina (http://vina.scripps.edu/).")
    warnings.warn(text, PerformanceWarning, stacklevel=2)
    return _builtin_dock(receptor_file, ligand_file, center, num_poses,
                         box_size=box_size)


def _format_pose_position(pose):
    """Where a pose sits, or an explicit statement that the file never said."""
    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.z)):
        return " at a position the pose file does not record"
    return f" at ({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f})"


def format_docking_report(result):
    unit = ENGINE_UNITS.get(result.engine)
    heuristic = unit == ENGINE_UNITS['builtin']
    lines = [
        "=== Molecular Docking Report ===",
        f"Engine: {result.engine}",
        f"Poses generated: {result.num_poses}",
        # An engine that never ran has no score to quote a unit for: claiming
        # kcal/mol for a docking that did not happen is how a missing file read
        # as a measured result of 0.00 kcal/mol.
        (f"Best score: {result.binding_energy:.2f} {unit}" if unit else
         f"Best score: {result.binding_energy:.2f} (no scoring engine ran)"),
    ]
    if heuristic:
        lines.append(
            "WARNING: heuristic contact score from the built-in rigid-body "
            "search; not a binding free energy and not comparable to Vina.")
    if result.poses:
        lines.append("\nTop poses:")
        for p in result.poses[:3]:
            lines.append(f"  Pose {p.rank}: {p.energy:.2f} {unit or ''}"
                         f"{_format_pose_position(p)}")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)

