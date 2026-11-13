"""biosuite.core.md_simulation: steepest descent and what it promises to stop on.

The docstring says the run stops when max|F| falls under the tolerance; the code
compared the Frobenius norm of the whole force array, which grows as sqrt(3N), so
the same tolerance meant something different for every system size.  The other
half of this file is the part a caller cannot see: whether the search actually
converged, which is now said out loud.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def lj_pair(sigma=3.5, epsilon=0.066, n=2):
    return md.LJParameters(np.full(n, sigma), np.full(n, epsilon))


def three_pairs_apart(r):
    return np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0],
                     [60.0, 0.0, 0.0], [60.0 + r, 0.0, 0.0],
                     [120.0, 0.0, 0.0], [120.0 + r, 0.0, 0.0]])


def test_two_carbons_minimise_to_the_lennard_jones_minimum():
    """Positive control for the whole force field: 2^(1/6) sigma = 3.9286 A."""
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    relaxed, history = md.minimize_energy(coords, lj_pair(), [], [], max_steps=400)
    distance = float(np.linalg.norm(relaxed[1] - relaxed[0]))
    assert distance == pytest.approx(2 ** (1.0 / 6.0) * 3.5, abs=0.01)
    assert history[0] > history[-1], "the energy went down, as advertised"
    assert len(history) == 23


def test_the_stopping_rule_is_the_largest_single_component_not_the_array_norm():
    """Measured on this fixture: max|F| = 0.0977 while the norm is 0.2393.

    Under the old comparison this system was nowhere near converged; under the
    documented one it already satisfies tol = 0.1 and the loop never moves.
    """
    coords = three_pairs_apart(3.74)
    lj = lj_pair(n=6)
    _e, forces = md.compute_forces(coords, lj, [], [])
    assert np.max(np.abs(forces)) < 0.1 <= np.sqrt(np.sum(forces * forces))
    _relaxed, history = md.minimize_energy(coords, lj, [], [], max_steps=200,
                                           convergence_tol=0.1)
    assert len(history) == 1, "converged before the first iteration"


def test_a_run_that_never_converges_says_so(caplog):
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        md.minimize_energy(coords, lj_pair(), [], [], max_steps=2)
    assert "Minimisation ended at max|F|" in caplog.text
    assert "above the convergence tolerance" in caplog.text
    assert "the structure is not at a minimum" in caplog.text


def test_a_converged_run_says_nothing(caplog):
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        md.minimize_energy(coords, lj_pair(), [], [], max_steps=400)
    assert caplog.text == ""


def test_zero_minimisation_steps_returns_the_input_energy_only():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    relaxed, history = md.minimize_energy(coords, lj_pair(), [], [], max_steps=0)
    assert history == [md.compute_forces(coords, lj_pair(), [], [])[0]]
    np.testing.assert_array_equal(relaxed, coords)


def test_the_input_array_is_never_written_to():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    before = coords.copy()
    md.minimize_energy(coords, lj_pair(), [], [], max_steps=50)
    np.testing.assert_array_equal(coords, before)


def test_integer_coordinates_are_minimised_rather_than_crashing():
    coords = np.array([[0, 0, 0], [3, 0, 0]])
    relaxed, history = md.minimize_energy(coords, lj_pair(), [], [], max_steps=100)
    assert relaxed.dtype == np.float64
    assert len(history) > 1


def test_a_step_size_that_underflows_ends_the_search_instead_of_spinning():
    """Measured: the search stops after 153 iterations of its 500-step budget.

    With an unreachable tolerance the loop keeps taking and rejecting smaller
    steps until the step size is below 1e-12 A, where a trial move is
    indistinguishable from standing still; the old code would have spent all 500
    force evaluations doing nothing.
    """
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    _relaxed, history = md.minimize_energy(coords, lj_pair(), [], [], max_steps=500,
                                           convergence_tol=1e-30, initial_step=1e-9)
    assert len(history) == 154, "one entry per iteration plus the starting point"
    assert 3.0 < _relaxed[1][0] < 2 ** (1.0 / 6.0) * 3.5, "part way, then out of steps"
    assert history[-1] < history[0]


def test_a_convergence_tolerance_below_any_printable_force_still_warns(caplog):
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        md.minimize_energy(coords, lj_pair(), [], [], max_steps=3,
                           convergence_tol=1e-12)
    assert "above the convergence tolerance 1e-12" in caplog.text
    assert "after 3 step(s)" in caplog.text


@pytest.mark.parametrize("value", [-1, -100])
def test_a_negative_step_budget_is_refused(value):
    with pytest.raises(ValueError, match="max_steps cannot be negative"):
        md.minimize_energy(np.zeros((2, 3)), lj_pair(), [], [], max_steps=value)


@pytest.mark.parametrize("value", [1.5, "20", None, True])
def test_a_step_budget_that_is_not_a_count_is_refused(value):
    with pytest.raises(ValueError, match="max_steps must be an integer"):
        md.minimize_energy(np.zeros((2, 3)), lj_pair(), [], [], max_steps=value)


@pytest.mark.parametrize("tol", [0, -1e-3])
def test_a_tolerance_that_cannot_be_met_is_refused(tol):
    with pytest.raises(ValueError, match="convergence_tol must be a positive force"):
        md.minimize_energy(np.zeros((2, 3)), lj_pair(), [], [], convergence_tol=tol)


@pytest.mark.parametrize("step", [0, 0.0, -0.01])
def test_a_zero_or_negative_first_step_would_only_halve_itself_forever(step):
    with pytest.raises(ValueError, match="initial_step must be a positive displacement"):
        md.minimize_energy(np.zeros((2, 3)), lj_pair(), [], [], initial_step=step)


def test_a_rejected_move_still_records_the_current_energy():
    """The history is one entry per iteration, whether or not that step moved.

    Measured: starting 0.03 A short of the minimum with a 1 A first step, the
    trial overshoots and is refused, so the coordinates never change and the
    recorded energy repeats.
    """
    coords = np.array([[0.0, 0.0, 0.0], [3.9, 0.0, 0.0]])
    relaxed, history = md.minimize_energy(coords, lj_pair(), [], [],
                                          max_steps=1, initial_step=1.0)
    assert history == pytest.approx([-0.065994, -0.065994], abs=5e-7)
    np.testing.assert_array_equal(relaxed, coords)
