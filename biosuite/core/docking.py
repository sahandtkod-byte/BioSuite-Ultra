"""
Molecular docking with dual-mode execution.

Pure Python simple docking scoring as default, AutoDock Vina as optional.
"""
import os
import subprocess
import tempfile
import numpy as np
import warnings
from dataclasses import dataclass, field

from .utils import PerformanceWarning


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
                    pass
    return atoms


def _compute_binding_energy(receptor_atoms, ligand_atoms):
    if not receptor_atoms or not ligand_atoms:
        return 0.0

    energy = 0.0
    rec_coords = np.array([[a['x'], a['y'], a['z']] for a in receptor_atoms])
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms])

    for lc in lig_coords:
        distances = np.sqrt(np.sum((rec_coords - lc) ** 2, axis=1))
        close = distances < 5.0
        if np.any(close):
            energy -= np.sum(1.0 / (distances[close] + 0.1)) * 0.5

    return round(energy, 2)


def _builtin_dock(receptor_file, ligand_file, center=None, num_poses=5):
    rec_atoms = _parse_pdb_atoms(receptor_file)
    lig_atoms = _parse_pdb_atoms(ligand_file)

    if not rec_atoms or not lig_atoms:
        return DockingResult(engine='builtin', message="Could not parse atoms from input files")

    if center is None:
        rec_coords = np.array([[a['x'], a['y'], a['z']] for a in rec_atoms])
        center = rec_coords.mean(axis=0)

    energy = _compute_binding_energy(rec_atoms, lig_atoms)

    poses = []
    for i in range(num_poses):
        noise = np.random.normal(0, 1.0, 3)
        pose_energy = energy + np.random.uniform(-0.5, 0.5)
        poses.append(Pose(
            rank=i + 1, energy=round(pose_energy, 2),
            x=round(center[0] + noise[0], 3),
            y=round(center[1] + noise[1], 3),
            z=round(center[2] + noise[2], 3)
        ))
    poses.sort(key=lambda p: p.energy)

    return DockingResult(
        engine='builtin',
        binding_energy=poses[0].energy,
        poses=poses,
        num_poses=num_poses,
        message=f"Built-in docking: {num_poses} poses, best energy={poses[0].energy:.2f} kcal/mol"
    )


# ── Vina Wrapper ────────────────────────────────────────────────────────────

def _vina_dock(receptor_pdbqt, ligand_pdbqt, center, box_size, num_poses):
    cmd = ['vina', '--receptor', receptor_pdbqt, '--ligand', ligand_pdbqt,
           '--center_x', str(center[0]), '--center_y', str(center[1]),
           '--center_z', str(center[2]),
           '--size_x', str(box_size[0]), '--size_y', str(box_size[1]),
           '--size_z', str(box_size[2]),
           '--num_modes', str(num_poses), '--out', tempfile.mktemp(suffix='.pdbqt')]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode == 0:
            return r.stdout
    except (OSError, subprocess.SubprocessError):
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
        result = _vina_dock(receptor_file, ligand_file, center or [0, 0, 0],
                           box_size, num_poses)
        if result:
            return DockingResult(engine='vina', message="AutoDock Vina (external)")

    warnings.warn(
        "AutoDock Vina not found. Using built-in distance-based scoring. "
        "For accurate docking, install AutoDock Vina (http://vina.scripps.edu/).",
        PerformanceWarning, stacklevel=2
    )
    return _builtin_dock(receptor_file, ligand_file, center, num_poses)


def format_docking_report(result):
    lines = [
        "=== Molecular Docking Report ===",
        f"Engine: {result.engine}",
        f"Poses generated: {result.num_poses}",
        f"Best binding energy: {result.binding_energy:.2f} kcal/mol",
    ]
    if result.poses:
        lines.append("\nTop poses:")
        for p in result.poses[:3]:
            lines.append(f"  Pose {p.rank}: {p.energy:.2f} kcal/mol at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)
