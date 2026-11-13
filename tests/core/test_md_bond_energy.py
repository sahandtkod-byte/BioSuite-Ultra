"""biosuite.core.md_simulation: the harmonic bond term E = 1/2 k (r - r0)^2.

Two tests here do work that a "is it finite" check cannot: the force must point
so as to *close* a stretched bond and *open* a compressed one, and the numeric
gradient of the energy has to agree with the analytic force.  A sign error in
either direction turns minimisation into a climb up the potential.
"""
import numpy as np
import pytest

from biosuite.core import md_simulation as md

K = 500.0


def test_a_stretched_bond_holds_its_documented_energy():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    energy, forces = md.compute_bond_energy(coords, [(0, 1, 1.0)], k_bond=K)
    assert energy == pytest.approx(0.5 * K * (2.0 - 1.0) ** 2)
    assert forces[0][0] > 0 > forces[1][0], "a stretched bond pulls its atoms together"


def test_a_compressed_bond_pushes_its_atoms_apart():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    energy, forces = md.compute_bond_energy(coords, [(0, 1, 3.0)], k_bond=K)
    assert energy == pytest.approx(0.5 * K * 1.0)
    assert forces[0][0] < 0 < forces[1][0]


def test_a_bond_at_its_rest_length_carries_neither_energy_nor_force():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.45, 0.0]])
    energy, forces = md.compute_bond_energy(coords, [(0, 1, 1.45)], k_bond=K)
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert np.all(forces == 0.0)


def test_the_forces_are_the_negative_gradient_of_the_energy():
    rng = np.random.default_rng(11)
    coords = rng.normal(0.0, 1.5, (4, 3))
    bonds = [(0, 1, 1.3), (1, 2, 1.5), (2, 3, 1.1), (0, 3, 2.0)]

    def energy_of(c):
        return md.compute_bond_energy(c, bonds, k_bond=K)[0]

    _e, forces = md.compute_bond_energy(coords, bonds, k_bond=K)
    step = 1e-6
    numeric = np.zeros_like(coords)
    for i in range(4):
        for d in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, d] += step
            minus[i, d] -= step
            numeric[i, d] = -(energy_of(plus) - energy_of(minus)) / (2 * step)
    assert np.max(np.abs(forces - numeric)) < 1e-4
    assert np.sum(forces, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_two_coincident_atoms_are_skipped_instead_of_dividing_by_zero():
    """r < 1e-8 has no direction to pull along, so the bond contributes nothing."""
    coords = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    energy, forces = md.compute_bond_energy(coords, [(0, 1, 1.45)], k_bond=K)
    assert energy == 0.0
    assert np.all(forces == 0.0)


def test_no_bonds_is_a_zero_energy_not_an_error():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    energy, forces = md.compute_bond_energy(coords, [])
    assert energy == 0.0
    assert forces.shape == (2, 3) and np.all(forces == 0.0)


def test_integer_coordinates_do_not_truncate_the_forces():
    coords = np.array([[0, 0, 0], [2, 0, 0]])
    energy, forces = md.compute_bond_energy(coords, [(0, 1, 1.0)], k_bond=K)
    assert forces.dtype == np.float64
    assert forces[0][0] == pytest.approx(500.0)


def test_a_bond_that_names_an_absent_atom_is_refused_by_number():
    coords = np.zeros((2, 3))
    with pytest.raises(ValueError,
                       match=r"bond \(0, 5, 1\.0\) refers to atom 5 of a system "
                             r"with 2 atom\(s\)"):
        md.compute_bond_energy(coords, [(0, 5, 1.0)])
    with pytest.raises(ValueError, match="refers to atom -1"):
        md.compute_bond_energy(coords, [(-1, 0, 1.0)])


def test_a_malformed_coordinate_array_is_rejected_with_its_shape():
    with pytest.raises(ValueError, match=r"got shape \(3,\)"):
        md.compute_bond_energy(np.array([0.0, 1.0, 2.0]), [(0, 1, 1.0)])


def test_several_bonds_sum_independently():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    bonds = [(0, 1, 1.0), (1, 2, 1.0)]
    energy, _f = md.compute_bond_energy(coords, bonds, k_bond=K)
    assert energy == pytest.approx(2 * 0.5 * K * 1.0)
