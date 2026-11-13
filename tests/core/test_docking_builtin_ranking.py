"""The built-in rigid-body search must return a real ranking of real placements.

An earlier version of this engine scored the input geometry once and then invented
`num_poses` results by adding noise to that score and to the receptor centroid, so
the "poses" carried no information about the receptor at all.  Everything here is
written so that a fabricated ranking cannot pass it.
"""
import numpy as np
import pytest

from biosuite.core import docking as dk


def shell_receptor(tmp_path, name="shell.pdb", radius=12.0, n=300):
    """A hollow spherical shell of atoms: a receptor with a central cavity."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coords = directions * radius
    path = tmp_path / name
    path.write_text("".join(
        f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        for i, (x, y, z) in enumerate(coords)), encoding="utf-8")
    return str(path)


def ligand(tmp_path, name="lig.pdb", offset=(40.0, 40.0, 40.0)):
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]],
                      dtype=float) + np.asarray(offset, dtype=float)
    path = tmp_path / name
    path.write_text("".join(
        f"HETATM{i + 1:5d}  C   LIG B{i + 1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        for i, (x, y, z) in enumerate(coords)), encoding="utf-8")
    return str(path)


def coords_of(path):
    return np.array([[a['x'], a['y'], a['z']] for a in dk._parse_pdb_atoms(path)])


def test_the_reported_energies_are_a_sorted_ranking_with_consecutive_ranks(tmp_path):
    result = dk._builtin_dock(shell_receptor(tmp_path), ligand(tmp_path), num_poses=8)

    energies = [p.energy for p in result.poses]
    assert energies == sorted(energies), energies
    assert [p.rank for p in result.poses] == list(range(1, 9))
    assert result.num_poses == len(result.poses) == 8
    assert result.binding_energy == energies[0]


def test_a_best_pose_beats_a_placement_far_outside_the_receptor(tmp_path):
    """Impossible if the energies were noise around one constant score."""
    receptor = shell_receptor(tmp_path)
    result = dk._builtin_dock(receptor, ligand(tmp_path), num_poses=6, seed=0)

    assert result.poses[0].energy < dk._score_contacts(
        coords_of(receptor), np.array([[500.0, 500.0, 500.0]]))


def test_the_search_finds_the_cavity_it_was_pointed_at(tmp_path):
    receptor = shell_receptor(tmp_path, radius=12.0)
    lig = ligand(tmp_path, offset=(60.0, 60.0, 60.0))

    result = dk._builtin_dock(receptor, lig, center=(0, 0, 0), box_size=(30, 30, 30),
                              num_poses=3, seed=1)

    best = result.poses[0]
    assert np.linalg.norm([best.x, best.y, best.z]) < 14.0, best
    assert best.energy < 0, best


def test_trials_are_never_fewer_than_the_poses_asked_for(tmp_path):
    receptor, lig = shell_receptor(tmp_path), ligand(tmp_path)

    result = dk._builtin_dock(receptor, lig, center=(0, 0, 0), num_poses=25, n_trials=4)

    assert len(result.poses) == 25, result.num_poses
    assert "25 poses from 25 trials" in result.message, result.message


def test_a_fixed_seed_reproduces_the_run_and_another_seed_does_not(tmp_path):
    receptor, lig = shell_receptor(tmp_path), ligand(tmp_path)
    key = lambda r: [(p.energy, p.x, p.y, p.z) for p in r.poses]

    first = dk._builtin_dock(receptor, lig, num_poses=5, seed=7)
    second = dk._builtin_dock(receptor, lig, num_poses=5, seed=7)
    third = dk._builtin_dock(receptor, lig, num_poses=5, seed=8)

    assert key(first) == key(second)
    assert [(p.x, p.y, p.z) for p in first.poses] != [(p.x, p.y, p.z) for p in third.poses]


def test_every_pose_position_is_inside_the_search_box(tmp_path):
    """A position outside the box would be a number nobody placed there."""
    center = [10.0, -20.0, 5.0]
    halves = (4.0, 5.0, 6.0)

    result = dk._builtin_dock(shell_receptor(tmp_path), ligand(tmp_path),
                              center=center, box_size=(8.0, 10.0, 12.0), num_poses=6)

    for pose in result.poses:
        for coordinate, axis, half in zip((pose.x, pose.y, pose.z), center, halves):
            assert abs(coordinate - axis) <= half + 1e-3, (pose, center)


def test_the_search_is_a_rigid_move_of_the_ligand_not_a_new_shape(tmp_path):
    """Internal ligand distances are preserved by rotation plus translation.

    A pose that rescaled the ligand would score better simply by packing more
    atoms into the receptor, which is why the sampled matrices must be rotations.
    """
    lig_coords = coords_of(ligand(tmp_path))

    rng = np.random.default_rng(3)
    matrices = dk._random_rotation_matrices(rng, 50)
    assert matrices.shape == (50, 3, 3), matrices.shape
    assert np.allclose(np.linalg.det(matrices), 1.0, atol=1e-9), \
        "a reflection is not a rotation: it turns the ligand into its mirror image"
    assert np.allclose(matrices @ np.transpose(matrices, (0, 2, 1)),
                       np.eye(3), atol=1e-9)

    moved = lig_coords @ matrices[0].T + np.array([7.0, -3.0, 2.0])

    def pairwise(coordinates):
        return np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1))

    assert np.allclose(pairwise(moved), pairwise(lig_coords), atol=1e-9)


def test_a_run_that_picked_nothing_leaves_the_message_to_say_why(tmp_path):
    empty = tmp_path / "no_atoms.pdb"
    empty.write_text("HEADER only\n", encoding="utf-8")

    result = dk._builtin_dock(shell_receptor(tmp_path), str(empty))

    assert result.poses == [] and result.num_poses == 0
    assert result.message == "Could not parse atoms from input files", result.message
    assert result.binding_energy == 0.0
