"""biosuite.core.md_simulation: the Lennard-Jones long-range tail correction.

The correction is a constant added per atom, so its arithmetic has to be
extensive or a big system is charged the same as a small one.  It used to divide
N by a volume that was itself N times the per-atom cell — the N cancelled
exactly, and one atom and a thousand atoms got the same answer.
"""
import math

import numpy as np
import pytest

from biosuite.core import md_simulation as md


SIGMA = np.array([3.5, 3.5])
EPSILON = np.array([0.066, 0.066])


def expected(n, sigma=3.5, epsilon=0.066, cutoff=10.0):
    """The formula as the docstring states it: (8/3) pi N rho eps s^3 [...]."""
    sr = sigma / cutoff
    density = 1.0 / (2.0 * cutoff) ** 3
    return (8.0 / 3.0) * math.pi * n * density * epsilon * sigma ** 3 * (
        sr ** 9 / 3.0 - sr ** 3 / 2.0
    )


def test_the_correction_matches_its_own_documented_formula():
    got = md.lj_cutoff_correction(SIGMA, EPSILON, cutoff=10.0, n=7)
    assert got == pytest.approx(expected(7), rel=1e-12)


def test_the_correction_is_extensive_in_the_number_of_atoms():
    """Measured before the fix: n=1, 100 and 1000 all returned -6.3448e-05."""
    one = md.lj_cutoff_correction(SIGMA, EPSILON, 10.0, 1)
    assert md.lj_cutoff_correction(SIGMA, EPSILON, 10.0, 100) == pytest.approx(one * 100)
    assert md.lj_cutoff_correction(SIGMA, EPSILON, 10.0, 1000) == pytest.approx(one * 1000)
    assert one == pytest.approx(-6.344804796e-05, rel=1e-9)


def test_the_correction_is_negative_at_an_ordinary_cutoff():
    """The attractive tail dominates at rc >> sigma, so the sign is a check."""
    assert md.lj_cutoff_correction(SIGMA, EPSILON, 10.0, 10) < 0


def test_an_empty_system_gets_no_correction_rather_than_nan():
    """np.mean of an empty array is nan, and nan used to poison the whole energy."""
    assert md.lj_cutoff_correction(np.array([]), np.array([]), 10.0, 0) == 0.0
    assert md.lj_cutoff_correction(SIGMA, EPSILON, 10.0, 0) == 0.0


@pytest.mark.parametrize("cutoff", [0, 0.0, -1.0, -0.5])
def test_a_non_positive_cutoff_is_refused(cutoff):
    """The old code divided by it: ZeroDivisionError for 0, nonsense for a negative."""
    with pytest.raises(ValueError, match="cutoff must be a positive distance"):
        md.lj_cutoff_correction(SIGMA, EPSILON, cutoff, 2)


def test_mixed_parameters_are_averaged_as_the_mean_field_assumption_says():
    sigma = np.array([3.0, 4.0])
    epsilon = np.array([0.1, 0.9])
    got = md.lj_cutoff_correction(sigma, epsilon, 10.0, 2)
    assert got == pytest.approx(expected(2, sigma=3.5, epsilon=0.5), rel=1e-12)


def test_a_shorter_cutoff_moves_the_correction_the_way_the_formula_says():
    far = md.lj_cutoff_correction(SIGMA, EPSILON, 12.0, 5)
    near = md.lj_cutoff_correction(SIGMA, EPSILON, 5.0, 5)
    assert abs(near) > abs(far), "cutting the range sooner leaves more uncounted tail"
