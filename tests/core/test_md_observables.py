"""biosuite.core.md_simulation: RMSD and the radius of gyration.

Both are the numbers a reader takes to the report, and both were undefined for
the inputs that used to reach them: the mean of an empty list is nan, and a
structure whose masses sum to zero divides by zero.  An undefined measurement
printed as a number is worse than no measurement.
"""
import numpy as np
import pytest

from biosuite.core import md_simulation as md


def test_rmsd_of_two_identical_sets_is_zero():
    coords = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    assert md.compute_rmsd(coords, coords.copy()) == 0.0


def test_rmsd_is_the_root_of_the_mean_squared_displacement():
    a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert md.compute_rmsd(a, b) == pytest.approx(np.sqrt((1.0 + 4.0) / 2))


def test_rmsd_does_not_need_the_same_number_of_atoms_to_look_symmetric():
    """The value is a sum of squares, so it is symmetric even though it is signed-free."""
    a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 0.5]])
    assert md.compute_rmsd(a, b) == pytest.approx(md.compute_rmsd(b, a))


def test_rmsd_of_one_atom_moved_one_angstrom():
    assert md.compute_rmsd(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3))) == 1.0


def test_rmsd_of_mismatched_coordinate_sets_is_refused():
    with pytest.raises(ValueError, match=r"same shape, got \(2, 3\) and \(3, 3\)"):
        md.compute_rmsd(np.zeros((2, 3)), np.zeros((3, 3)))


def test_rmsd_of_an_empty_set_is_refused_rather_than_reported_as_nan():
    with pytest.raises(ValueError, match="RMSD of an empty coordinate set"):
        md.compute_rmsd(np.zeros((0, 3)), np.zeros((0, 3)))


def test_the_radius_of_gyration_of_a_symmetric_pair_is_its_half_distance():
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert md.compute_radius_of_gyration(coords, np.array([12.0, 12.0])) == \
        pytest.approx(1.0)


def test_the_radius_of_gyration_is_mass_weighted():
    """COM sits at 2.25 A: Rg = sqrt((1*2.25^2 + 3*0.75^2)/4) = sqrt(1.6875)."""
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rg = md.compute_radius_of_gyration(coords, np.array([1.0, 3.0]))
    assert rg == pytest.approx(1.6875 ** 0.5)
    unweighted = md.compute_radius_of_gyration(coords, np.array([1.0, 1.0]))
    assert unweighted == pytest.approx(1.5)


def test_a_structure_translated_in_space_keeps_its_radius(tmp_path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 3.0]])
    masses = np.array([12.0, 14.0, 16.0])
    assert md.compute_radius_of_gyration(coords, masses) == pytest.approx(
        md.compute_radius_of_gyration(coords + 500.0, masses))


def test_a_single_atom_has_no_extent():
    assert md.compute_radius_of_gyration(np.array([[1.0, 2.0, 3.0]]),
                                        np.array([12.0])) == 0.0


def test_massless_atoms_are_refused_by_their_total(tmp_path):
    with pytest.raises(ValueError, match="needs atoms with mass, got 2 atom\\(s\\) "
                                         "totalling 0.0 amu"):
        md.compute_radius_of_gyration(np.zeros((2, 3)), np.zeros(2))


def test_an_empty_structure_is_refused():
    with pytest.raises(ValueError, match="radius of gyration needs atoms with mass"):
        md.compute_radius_of_gyration(np.zeros((0, 3)), np.zeros(0))


def test_a_mass_array_of_the_wrong_length_is_named():
    with pytest.raises(ValueError, match=r"masses has 2 entries but the coordinate "
                                         r"set has 3"):
        md.compute_radius_of_gyration(np.zeros((3, 3)), np.array([12.0, 12.0]))


def test_integer_coordinates_are_measured_not_rounded():
    coords = np.array([[0, 0, 0], [1, 0, 0]])
    assert md.compute_radius_of_gyration(coords, np.array([1, 1])) == pytest.approx(0.5)
    assert md.compute_rmsd(coords, np.array([[0, 0, 0], [0, 1, 0]])) == pytest.approx(1.0)


def test_equal_masses_make_the_radius_the_spread_about_the_centroid():
    """With one mass everywhere, Rg is by definition the RMSD from the centre."""
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    masses = np.full(3, 12.011)
    centre = np.mean(coords, axis=0)
    assert md.compute_radius_of_gyration(coords, masses) == pytest.approx(
        md.compute_rmsd(coords, np.tile(centre, (3, 1))))
