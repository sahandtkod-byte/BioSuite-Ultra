"""The built-in contact potential: what it rewards, and what its blocks must not change.

The score is a heuristic in arbitrary units, so the only claims worth pinning are
its shape (near attracts, overlapping repels, far contributes nothing) and the
fact that it is computed in blocks — a block boundary that changed the total would
make a large receptor score differently from a small one for the same geometry.
"""
import numpy as np
import pytest

from biosuite.core import docking as dk


def grid(*points):
    return np.array(points, dtype=float)


def test_an_empty_side_scores_zero_in_both_directions():
    coords = grid((0.0, 0.0, 0.0))

    assert dk._score_contacts(coords, grid()) == 0.0
    assert dk._score_contacts(grid(), coords) == 0.0


def test_the_binding_energy_of_a_missing_side_is_zero_without_reaching_the_score():
    one = [{'x': 0.0, 'y': 0.0, 'z': 0.0}]

    assert dk._compute_binding_energy([], one) == 0.0
    assert dk._compute_binding_energy(one, []) == 0.0


def test_a_contacting_pair_attracts():
    """4 Å apart: inside the 5 Å cutoff, outside the 3 Å clash radius."""
    score = dk._score_contacts(grid((0.0, 0.0, 0.0)), grid((4.0, 0.0, 0.0)))

    assert score < 0.0, score
    assert score == pytest.approx(-0.5 / 4.1, abs=1e-12)


def test_a_pair_beyond_the_cutoff_contributes_nothing():
    assert dk._score_contacts(grid((0.0, 0.0, 0.0)), grid((9.0, 0.0, 0.0))) == 0.0


def test_overlapping_atoms_are_penalised_and_lose_to_a_contact():
    """Regression guard: the score used to fall monotonically with distance.

    The best "pose" was then the ligand superimposed on the protein, which is
    physically impossible and scored as the strongest binding.
    """
    receptor = [{'x': 0.0, 'y': 0.0, 'z': 0.0}, {'x': 1.0, 'y': 0.0, 'z': 0.0}]
    clashing = dk._compute_binding_energy(
        receptor, [{'x': 0.5, 'y': 0.0, 'z': 0.0}])
    contacting = dk._compute_binding_energy(
        receptor, [{'x': 4.0, 'y': 0.0, 'z': 0.0}])

    assert clashing > 0.0, clashing
    assert clashing > contacting, (clashing, contacting)


def test_the_repulsive_term_grows_faster_than_the_attraction_shrinks():
    """Halving an already-clashing distance more than doubles the penalty."""
    receptor = grid((0.0, 0.0, 0.0))
    near = dk._score_contacts(receptor, grid((2.0, 0.0, 0.0)))
    nearer = dk._score_contacts(receptor, grid((1.0, 0.0, 0.0)))

    assert nearer > near * 4 > near > 0.0, (near, nearer)


def test_the_blocked_matrix_gives_the_unblocked_total():
    """1500 x 1500 atom pairs cross the ~2e6-element block boundary.

    The pairwise distances are computed in blocks so a large receptor cannot
    allocate the whole matrix at once; a block boundary that dropped or doubled a
    pair would make the same geometry score differently by receptor size, and
    that is invisible unless the two are compared directly.
    """
    rng = np.random.default_rng(11)
    receptor = rng.normal(scale=25.0, size=(1500, 3))
    ligand = rng.normal(scale=25.0, size=(1500, 3))
    distances = np.sqrt(((ligand[:, None, :] - receptor[None, :, :]) ** 2).sum(-1))
    expected = (-(1.0 / (distances[distances < 5.0] + 0.1)).sum() * 0.5
                + ((3.0 - distances[distances < 3.0]) ** 2).sum() * 2.0)

    assert dk._score_contacts(receptor, ligand) == pytest.approx(expected, rel=1e-12)


def test_the_binding_energy_is_the_rounded_score_of_the_supplied_geometry():
    receptor = [{'x': 0.0, 'y': 0.0, 'z': 0.0}, {'x': 0.0, 'y': 0.0, 'z': 0.0}]
    ligand = [{'x': 4.0, 'y': 0.0, 'z': 0.0}]
    raw = dk._score_contacts(grid((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), grid((4.0, 0.0, 0.0)))

    assert dk._compute_binding_energy(receptor, ligand) == round(raw, 2)
