"""biosuite.core.md_simulation: the Maxwell-Boltzmann velocity a run starts from.

Every observable after this line depends on the draw: the reported temperature,
the kinetic energy, the radius of gyration's drift.  A negative target used to
produce nan velocities that travelled all the way into the report, and there was
no way to ask for the same starting conditions twice.
"""
import numpy as np
import pytest

from biosuite.core import md_simulation as md

MASS = 12.011


def sigma_v_am_per_fs(temperature, mass_amu=MASS):
    """The most probable speed scale sqrt(kB T / m), in the module's own units."""
    return (md.BOLTZMANN_K * temperature / (mass_amu * md.amu_to_kg)) ** 0.5 * 1e-5


def test_the_drawn_speeds_have_the_documented_width():
    masses = np.full(4000, MASS)
    vel = md._init_velocities(masses, 300.0, rng=np.random.default_rng(1))
    assert np.std(vel) == pytest.approx(sigma_v_am_per_fs(300.0), rel=0.03)


def test_the_draw_reaches_about_the_temperature_it_was_asked_for():
    masses = np.full(600, MASS)
    vel = md._init_velocities(masses, 300.0, rng=np.random.default_rng(4))
    assert md.compute_temperature(vel, masses) == pytest.approx(300.0, rel=0.05)


def test_the_centre_of_mass_does_not_drift():
    masses = np.array([1.008, 12.011, 15.999, 32.06])
    vel = md._init_velocities(masses, 310.0, rng=np.random.default_rng(9))
    assert np.sum(masses[:, np.newaxis] * vel, axis=0) == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12)


def test_a_colder_target_moves_everything_less():
    masses = np.full(200, MASS)
    hot = md._init_velocities(masses, 400.0, rng=np.random.default_rng(2))
    cold = md._init_velocities(masses, 100.0, rng=np.random.default_rng(2))
    assert np.std(hot) / np.std(cold) == pytest.approx(2.0, rel=0.02)


def test_zero_kelvin_is_still_not_an_error():
    vel = md._init_velocities(np.array([MASS, 15.999]), 0.0,
                              rng=np.random.default_rng(0))
    assert np.all(vel == 0.0)


def test_the_same_seed_gives_the_same_velocities():
    masses = np.array([MASS, 1.008, 14.007])
    a = md._init_velocities(masses, 300.0, rng=np.random.default_rng(12345))
    b = md._init_velocities(masses, 300.0, rng=np.random.default_rng(12345))
    c = md._init_velocities(masses, 300.0, rng=np.random.default_rng(12346))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_generator_is_created_when_the_caller_brings_none():
    vel = md._init_velocities(np.array([MASS]), 300.0)
    assert vel.shape == (1, 3)
    assert np.isfinite(vel).all()


def test_integer_masses_do_not_integer_the_velocities():
    vel = md._init_velocities(np.array([12, 14]), 300.0, rng=np.random.default_rng(3))
    assert vel.dtype == np.float64
    assert not np.array_equal(vel, np.round(vel)), "the draw is not a whole number"


@pytest.mark.parametrize("temperature", [-1.0, -300.0, -0.5])
def test_a_negative_temperature_is_refused_not_solved_as_a_square_root(temperature):
    """sqrt(kB * -T / m) used to be nan, and every later number inherited it."""
    with pytest.raises(ValueError, match="target temperature cannot be negative"):
        md._init_velocities(np.array([MASS]), temperature)


@pytest.mark.parametrize("masses", [[], [[]], [0.0], [-1.0], [12.0, 0.0]])
def test_a_mass_array_that_cannot_be_used_is_named(masses):
    arr = np.array(masses, dtype=float)
    with pytest.raises(ValueError):
        md._init_velocities(arr, 300.0)


def test_a_zero_mass_is_reported_by_atom_number():
    with pytest.raises(ValueError, match="mass of atom 1 is 0.0 amu"):
        md._init_velocities(np.array([12.011, 0.0]), 300.0)


def test_an_empty_mass_array_is_rejected_with_its_shape():
    with pytest.raises(ValueError, match=r"masses must be a non-empty \(N,\) array"):
        md._init_velocities(np.zeros((0,)), 300.0)
