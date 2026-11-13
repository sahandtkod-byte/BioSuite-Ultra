"""biosuite.core.md_simulation: bond angles and the angles inferred from bonds.

The angle list is derived from the bond graph, so a bond written twice must not
buy the same angle twice, and an angle term whose gradient points the wrong way
holds a molecule at the wrong shape.  The potential here is harmonic around a
single hard-coded tetrahedral 109.5 degrees, so the numbers in these tests come
from that one constant.
"""
import math

import numpy as np
import pytest

from biosuite.core import md_simulation as md

K = 50.0
THETA0 = math.radians(109.5)


def test_the_angle_list_comes_from_the_bond_graph():
    """Each tuple is (outer, centre, outer) — the middle index is the vertex."""
    bonds = [(0, 1, 1.45), (1, 2, 1.45), (2, 3, 1.45)]
    assert md._build_angle_list(bonds) == [(0, 1, 2), (1, 2, 3)]


def test_a_centre_atom_with_one_neighbour_has_no_angle():
    assert md._build_angle_list([(0, 1, 1.45)]) == []


def test_a_duplicated_bond_does_not_buy_the_angle_twice():
    bonds = [(0, 1, 1.45), (1, 2, 1.45), (0, 1, 1.45)]
    assert md._build_angle_list(bonds) == [(0, 1, 2)]


def test_a_star_of_four_neighbours_gives_every_pair_once():
    bonds = [(1, 0, 1.45), (2, 0, 1.45), (3, 0, 1.45), (4, 0, 1.45)]
    angles = md._build_angle_list(bonds)
    assert len(angles) == 6
    assert sorted(angles) == sorted([(a, 0, b) for a, b in
                                     [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]])


def test_an_exact_tetrahedral_angle_holds_no_energy():
    """Two arms at 109.5 degrees: the reference case for the whole term."""
    half = math.radians(109.5) / 2.0
    coords = np.array([[-math.sin(half), math.cos(half), 0.0],
                       [0.0, 0.0, 0.0],
                       [math.sin(half), math.cos(half), 0.0]])
    energy, forces = md.compute_angle_energy(coords, [(0, 1, 2)], k_angle=K)
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(forces)) == pytest.approx(0.0, abs=1e-12)


def test_a_straight_angle_costs_the_documented_amount_and_exerts_no_force():
    coords = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    energy, forces = md.compute_angle_energy(coords, [(0, 1, 2)], k_angle=K)
    assert energy == pytest.approx(0.5 * K * (math.pi - THETA0) ** 2)
    assert np.all(forces == 0.0), "sin(theta) = 0 has no well-defined gradient"


def test_two_arms_on_the_same_side_are_the_expensive_degenerate_case():
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    energy, forces = md.compute_angle_energy(coords, [(0, 1, 2)], k_angle=K)
    assert energy == pytest.approx(0.5 * K * THETA0 ** 2)
    assert np.all(forces == 0.0), "sin(theta) = 0 again, this time at zero degrees"


def test_a_right_angle_moves_the_atoms_back_toward_the_equilibrium():
    """90 deg is narrower than 109.5, so each arm is pushed away from the other."""
    coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    energy, forces = md.compute_angle_energy(coords, [(0, 1, 2)], k_angle=K)
    assert energy == pytest.approx(0.5 * K * (math.pi / 2 - THETA0) ** 2)
    assert forces[0][0] < 0, "the upper arm slides away from the other one"
    assert forces[2][1] < 0
    assert np.sum(forces, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_the_angle_gradient_is_the_negative_of_its_numeric_derivative():
    rng = np.random.default_rng(5)
    coords = rng.normal(0.0, 1.2, (4, 3))
    angles = [(0, 1, 2), (1, 2, 3)]

    def energy_of(c):
        return md.compute_angle_energy(c, angles, k_angle=K)[0]

    _e, forces = md.compute_angle_energy(coords, angles, k_angle=K)
    step = 1e-6
    numeric = np.zeros_like(coords)
    for i in range(4):
        for d in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, d] += step
            minus[i, d] -= step
            numeric[i, d] = -(energy_of(plus) - energy_of(minus)) / (2 * step)
    assert np.max(np.abs(forces - numeric) / np.maximum(np.abs(numeric), 1e-9)) < 1e-5


def test_an_angle_through_a_coincident_centre_is_skipped():
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    energy, forces = md.compute_angle_energy(coords, [(0, 1, 2)], k_angle=K)
    assert (energy, np.all(forces == 0.0)) == (0.0, True)


def test_no_angles_is_a_zero_energy_not_an_error():
    coords = np.zeros((3, 3))
    energy, forces = md.compute_angle_energy(coords, [])
    assert energy == 0.0 and forces.shape == (3, 3)


def test_an_angle_that_names_an_absent_atom_is_refused_by_number():
    coords = np.zeros((2, 3))
    with pytest.raises(ValueError,
                       match=r"angle \(0, 1, 7\) refers to atom 7 of a system "
                             r"with 2 atom\(s\)"):
        md.compute_angle_energy(coords, [(0, 1, 7)])


def test_integer_coordinates_are_promoted_before_the_forces_are_filled():
    coords = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    _energy, forces = md.compute_angle_energy(coords, [(0, 1, 2)], k_angle=K)
    assert forces.dtype == np.float64
    assert np.any(forces != 0.0)
