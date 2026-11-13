"""biosuite.core.md_simulation: bonded atoms must not also feel the LJ wall.

A carbon-nitrogen bond is 1.45 A long while these Lennard-Jones sigmas are
3.0-3.6 A, so every bonded pair sits far inside the repulsive wall: measured at
1.45 A, one such pair carries 10,276 kcal/mol of energy and 85,255 kcal/mol/A of
force.  The first fifteen integration steps of a chain of six carbons threw an
atom to 17,951 A, and the report printed the result as if it were physics.
Standard force fields exclude 1-2 and 1-3 pairs, which is what these tests pin.
"""
import numpy as np
import pytest

from biosuite.core import md_simulation as md


def atom(serial, name="CA", element="C", res_name="ALA", chain="A", res_seq=1,
         x=0.0, y=0.0, z=0.0):
    return md.Atom(serial=serial, name=name, alt_loc=" ", res_name=res_name,
                   chain_id=chain, res_seq=res_seq, icode=" ", x=x, y=y, z=z,
                   occupancy=1.0, temp_factor=0.0, element=element)


LJ = md.LJParameters(sigma=np.array([3.5, 3.5]), epsilon=np.array([0.066, 0.066]))


def test_an_excluded_pair_contributes_no_energy_and_no_force():
    coords = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]])
    tail = md.lj_cutoff_correction(LJ.sigma, LJ.epsilon, 10.0, 2)
    plain_e, plain_f = md.compute_lj_energy(coords, LJ, 10.0)
    excl_e, excl_f = md.compute_lj_energy(coords, LJ, 10.0, frozenset({(0, 1)}))
    assert plain_e - tail > 6000, "the wall really is that steep at a bond length"
    assert np.max(np.abs(plain_f)) > 50000
    assert excl_e == pytest.approx(tail, rel=1e-12)
    assert np.all(excl_f == 0.0)


def test_the_total_energy_of_a_relaxed_bond_is_zero_not_a_clash():
    """compute_forces excludes the pairs the bond and angle lists name."""
    coords = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]])
    bonds = [(0, 1, 1.45)]
    tail = md.lj_cutoff_correction(LJ.sigma, LJ.epsilon, 10.0, 2)
    energy, forces = md.compute_forces(coords, LJ, bonds, [])
    assert energy == pytest.approx(tail, rel=1e-12), "bond at r0 and no LJ term"
    assert np.max(np.abs(forces)) == pytest.approx(0.0, abs=1e-12)


def test_a_1_3_pair_is_excluded_through_the_angle_list():
    coords = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [2.5, 1.0, 0.0]])
    bonds = [(0, 1, 1.45), (1, 2, 1.45)]
    angles = [(0, 1, 2)]
    lj = md.LJParameters(np.full(3, 3.5), np.full(3, 0.066))
    tail = md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 3)
    e_angle, _f = md.compute_angle_energy(coords, angles)
    energy, _f2 = md.compute_forces(coords, lj, bonds, angles)
    assert energy == pytest.approx(tail + e_angle, rel=1e-12), \
        "bond and angle terms only; the 2.69 A 1-3 pair costs no LJ energy"
    unexcluded, _f3 = md.compute_lj_energy(coords, lj, 10.0)
    assert unexcluded > 9000, "without the exclusions this cluster is a clash"


def test_only_the_pairs_the_caller_named_are_excluded():
    """A chain at its bond lengths: the 1-3 pair survives until angles name it."""
    coords = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [2.9, 0.0, 0.0]])
    lj = md.LJParameters(np.full(3, 3.5), np.full(3, 0.066))
    bonds = [(0, 1, 1.45), (1, 2, 1.45)]
    assert md._build_lj_exclusions(bonds, []) == frozenset({(0, 1), (1, 2)})
    energy, _f = md.compute_forces(coords, lj, bonds, [])

    def lj_pair(r, sigma=3.5, epsilon=0.066):
        sr = sigma / r
        return 4.0 * epsilon * (sr ** 12 - sr ** 6)

    expected = (lj_pair(2.9) + md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 3))
    assert energy == pytest.approx(expected, rel=1e-9), \
        "the two bonded pairs are silent, the 2.9 A 1-3 pair is not"
    with_angles = md.compute_forces(coords, lj, bonds, [(0, 1, 2)])[0]
    assert with_angles - md.compute_angle_energy(coords, [(0, 1, 2)])[0] == \
        pytest.approx(md.lj_cutoff_correction(lj.sigma, lj.epsilon, 10.0, 3), rel=1e-9)


def test_the_exclusion_set_is_symmetric_and_ordered_by_index():
    bonds = [(5, 2, 1.4), (1, 3, 1.5)]
    excluded = md._build_lj_exclusions(bonds, [(3, 1, 7)])
    assert excluded == frozenset({(2, 5), (1, 3), (3, 7)})


def test_a_degenerate_angle_with_one_arm_excludes_nothing():
    """(0, 1, 0) names the same atom twice: there is no 1-3 pair in it."""
    assert md._build_lj_exclusions([], [(0, 1, 0)]) == frozenset()


def test_no_bonds_means_no_exclusions():
    assert md._build_lj_exclusions([], []) == frozenset()


def test_an_exclusion_that_names_an_atom_the_system_does_not_have_is_refused():
    coords = np.zeros((2, 3))
    with pytest.raises(ValueError, match="does not name an ordered pair of atoms"):
        md.compute_lj_energy(coords, LJ, 10.0, frozenset({(0, 4)}))
    with pytest.raises(ValueError, match="does not name an ordered pair of atoms"):
        md.compute_lj_energy(coords, LJ, 10.0, frozenset({(1, 0)}))


def test_a_real_backbone_stays_inside_the_pdb_coordinate_range():
    """The regression this whole file exists for: five residues, 1.5 A bonds.

    Before the exclusions existed, 15 integration steps at dt = 1 fs reached
    x = 17,951 A, which the PDB writer can represent no more than a nan could.
    """
    coords = np.array([[i * 1.45, 0.0, 0.0] for i in range(6)])
    lj = md.LJParameters(np.full(6, 3.5), np.full(6, 0.066))
    bonds = [(i, i + 1, 1.45) for i in range(5)]
    angles = md._build_angle_list(bonds)
    velocities = md._init_velocities(np.full(6, 12.011), 300.0,
                                     rng=np.random.default_rng(7))
    _e, forces = md.compute_forces(coords, lj, bonds, angles, 10.0)
    for _step in range(15):
        coords, velocities, forces, _e = md.velocity_verlet_step(
            coords, velocities, forces, np.full(6, 12.011), lj, bonds, angles,
            dt=0.1, cutoff=10.0, target_temp=300.0, tau=100.0)
    assert np.max(np.abs(coords)) < 100.0, "the run stays writable"
    assert np.isfinite(coords).all()
