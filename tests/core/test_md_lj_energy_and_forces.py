"""biosuite.core.md_simulation: the Lennard-Jones pair energy and its forces.

Two things are pinned here that are easy to get wrong and hard to notice: the
sign of the force (a plus sign makes the integrator climb the potential instead
of descending it) and what the function does with input it was never meant to
receive — integer coordinates, an atom count that disagrees with the parameter
array, a pair sitting on top of itself.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def pair_lj(sigma=3.5, epsilon=0.066):
    return md.LJParameters(sigma=np.array([sigma, sigma]),
                           epsilon=np.array([epsilon, epsilon]))


def analytic(r, sigma=3.5, epsilon=0.066):
    """(energy, |dV/dr|) for one LJ pair, straight from V = 4 eps (s/r)^12 - (s/r)^6."""
    sr = sigma / r
    energy = 4.0 * epsilon * (sr ** 12 - sr ** 6)
    gradient = 24.0 * epsilon * (2.0 * sr ** 12 - sr ** 6) / r
    return energy, gradient


def test_the_minimum_of_the_curve_is_where_the_textbook_puts_it():
    sigma, epsilon = 3.5, 0.066
    r_min = 2 ** (1.0 / 6.0) * sigma
    coords = np.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]])
    lj = pair_lj(sigma, epsilon)
    energy, forces = md.compute_lj_energy(coords, lj, cutoff=10.0)
    tail = md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 2)
    assert energy - tail == pytest.approx(-epsilon, abs=1e-9)
    assert np.max(np.abs(forces)) == pytest.approx(0.0, abs=1e-9)


def test_the_force_on_each_atom_is_the_negative_gradient_and_obeys_newton():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    lj = pair_lj()
    energy, forces = md.compute_lj_energy(coords, lj, cutoff=10.0)
    _pair, gradient = analytic(3.0)
    assert forces[0][0] == pytest.approx(-gradient, rel=1e-12)
    assert forces[1][0] == pytest.approx(gradient, rel=1e-12)
    assert np.sum(forces, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_the_repulsive_wall_pushes_atoms_apart_not_together():
    """The sign bug this pins: a '+' here made minimisation climb the potential."""
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    _e, forces = md.compute_lj_energy(coords, pair_lj(), cutoff=10.0)
    assert forces[0][0] < 0 < forces[1][0]


def test_the_energy_is_the_pair_sum_plus_the_tail_correction():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    lj = pair_lj()
    energy, _f = md.compute_lj_energy(coords, lj, cutoff=10.0)
    pair, _g = analytic(3.0)
    assert energy == pytest.approx(pair + md.lj_cutoff_correction(
        lj.sigma, lj.epsilon, 10.0, 2), rel=1e-12)


def test_pairs_beyond_the_cutoff_contribute_nothing_but_the_tail():
    coords = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    lj = pair_lj()
    energy, forces = md.compute_lj_energy(coords, lj, cutoff=10.0)
    assert energy == pytest.approx(md.lj_cutoff_correction(
        lj.sigma, lj.epsilon, 10.0, 2), rel=1e-12)
    assert np.all(forces == 0.0)


def test_unlike_atoms_use_the_lorentz_berthelot_rules():
    """sigma averages and epsilon takes the geometric mean of the two partners."""
    lj = md.LJParameters(sigma=np.array([3.0, 4.0]), epsilon=np.array([0.1, 0.9]))
    coords = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    energy, _f = md.compute_lj_energy(coords, lj, cutoff=20.0)
    # (3.0 + 4.0) / 2 = 3.5 A and sqrt(0.1 * 0.9) = 0.3 kcal/mol.
    mixed, _g = analytic(3.5, sigma=3.5, epsilon=0.3)
    tail = md.lj_cutoff_correction(lj.sigma, lj.epsilon, 20.0, 2)
    assert energy - tail == pytest.approx(mixed, rel=1e-12)


def test_the_numeric_gradient_matches_the_analytic_forces_on_a_random_cluster():
    """Five atoms, every pair inside the cutoff, finite differences at h = 1e-5."""
    rng = np.random.default_rng(3)
    coords = rng.normal(0, 2.0, (5, 3))
    lj = md.LJParameters(np.full(5, 3.4), np.full(5, 0.066))
    energy, forces = md.compute_lj_energy(coords, lj, cutoff=10.0)
    step = 1e-5
    numeric = np.zeros_like(coords)
    for i in range(5):
        for d in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, d] += step
            minus[i, d] -= step
            numeric[i, d] = -(md.compute_lj_energy(plus, lj, 10.0)[0]
                              - md.compute_lj_energy(minus, lj, 10.0)[0]) / (2 * step)
    scale = np.maximum(np.abs(forces), 1e-9)
    assert np.max(np.abs(forces - numeric) / scale) < 1e-5


def test_integer_coordinates_are_read_as_coordinates_not_rejected():
    """np.zeros_like(coords) used to inherit the integer dtype and crash mid-step."""
    lj = pair_lj()
    energy, forces = md.compute_lj_energy(np.array([[0, 0, 0], [4, 0, 0]]), lj, 10.0)
    assert forces.dtype == np.float64
    assert np.max(np.abs(forces)) > 0
    assert np.isfinite(energy)


def test_two_coincident_atoms_are_reported_instead_of_vanishing(caplog):
    coords = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    lj = pair_lj()
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        energy, forces = md.compute_lj_energy(coords, lj, cutoff=10.0)
    assert energy == pytest.approx(
        md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 2), rel=1e-12)
    assert np.all(forces == 0.0)
    assert "1 atom pair(s)" in caplog.text
    assert "closer than 0.001" in caplog.text


def test_a_duplicated_atom_in_a_real_structure_is_counted_once_per_pair(caplog):
    coords = np.zeros((4, 3))
    coords[1] = coords[0]
    coords[3] = coords[2]
    coords[2] = [5.0, 0.0, 0.0]
    coords[3] = [5.0, 0.0, 0.0]
    lj = md.LJParameters(np.full(4, 3.5), np.full(4, 0.066))
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        md.compute_lj_energy(coords, lj, cutoff=10.0)
    assert "2 atom pair(s)" in caplog.text


def test_a_malformed_coordinate_array_is_rejected_with_its_shape():
    lj = md.LJParameters(np.array([3.5, 3.5]), np.array([0.066, 0.066]))
    with pytest.raises(ValueError, match=r"\(N, 3\) array of coordinates, got shape \(2,\)"):
        md.compute_lj_energy(np.array([0.0, 1.0]), lj, 10.0)
    with pytest.raises(ValueError, match=r"got shape \(2, 2\)"):
        md.compute_lj_energy(np.zeros((2, 2)), lj, 10.0)


def test_a_parameter_array_that_describes_a_different_system_is_refused():
    with pytest.raises(ValueError,
                       match=r"lj\.sigma has 3 entries but the coordinate set has 2"):
        md.compute_lj_energy(np.zeros((2, 3)),
                             md.LJParameters(np.zeros(3), np.zeros(3)), 10.0)
    with pytest.raises(ValueError,
                       match=r"lj\.epsilon has 3 entries but the coordinate set has 2"):
        md.compute_lj_energy(np.zeros((2, 3)),
                             md.LJParameters(np.zeros(2), np.zeros(3)), 10.0)


@pytest.mark.parametrize("cutoff", [0, 0.0, -2.0])
def test_an_unusable_cutoff_is_named_rather_than_divided_by(cutoff):
    with pytest.raises(ValueError, match="cutoff must be a positive distance"):
        md.compute_lj_energy(np.array([[0.0, 0, 0], [3.0, 0, 0]]), pair_lj(), cutoff)


def test_a_single_atom_has_no_pair_term_and_still_earns_its_tail():
    lj = md.LJParameters(np.array([3.5]), np.array([0.066]))
    energy, forces = md.compute_lj_energy(np.array([[0.0, 0.0, 0.0]]), lj, 10.0)
    assert np.all(forces == 0.0)
    assert energy == pytest.approx(md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 1))


def test_three_atoms_sum_their_three_pairs():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    lj = md.LJParameters(np.full(3, 3.5), np.full(3, 0.066))
    energy, _f = md.compute_lj_energy(coords, lj, cutoff=10.0)
    pairs = (analytic(3.0)[0] + analytic(4.0)[0]
             + analytic(5.0)[0])
    assert energy - md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 3) == \
        pytest.approx(pairs, rel=1e-12)
