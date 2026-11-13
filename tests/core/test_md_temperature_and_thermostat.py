"""biosuite.core.md_simulation: temperature and the Berendsen coupling.

Temperature here is a derived quantity, so these tests check the arithmetic that
turns velocities into kelvin and then turns a target into a scale factor — the
factor is mass-independent by construction, and a coupling time shorter than the
timestep overshoots rather than converging.
"""
import numpy as np
import pytest

from biosuite.core import md_simulation as md


def temperature_from_the_formula(velocities, masses):
    """T = 2 KE / (3 N kB) with KE converted out of amu*(A/fs)^2."""
    ke = 0.5 * float(np.sum(masses[:, np.newaxis] * velocities ** 2))
    return 2.0 * (ke * md.amu_to_kg * 1e10) / (3.0 * len(masses) * md.BOLTZMANN_K)


def test_the_reported_temperature_is_the_one_the_formula_gives():
    masses = np.array([12.011, 1.008, 15.999])
    vel = np.array([[0.05, -0.02, 0.0], [0.0, 0.11, 0.03], [-0.07, 0.0, 0.02]])
    assert md.compute_temperature(vel, masses) == pytest.approx(
        temperature_from_the_formula(vel, masses), rel=1e-12)


def test_a_still_system_has_no_temperature():
    assert md.compute_temperature(np.zeros((3, 3)), np.full(3, 12.011)) == 0.0


def test_no_particles_has_no_temperature_rather_than_a_division_by_zero():
    assert md.compute_temperature(np.zeros((0, 3)), np.zeros(0)) == 0.0


def test_temperature_scales_with_the_square_of_the_speed():
    masses = np.array([12.011])
    single = md.compute_temperature(np.array([[0.01, 0.0, 0.0]]), masses)
    doubled = md.compute_temperature(np.array([[0.02, 0.0, 0.0]]), masses)
    assert doubled == pytest.approx(single * 4.0, rel=1e-12)


def test_a_heavier_atom_at_the_same_speed_is_hotter():
    vel = np.array([[0.0, 0.0, 0.01]])
    assert md.compute_temperature(vel, np.array([32.06])) > \
        md.compute_temperature(vel, np.array([1.008]))


def test_the_thermostat_moves_the_temperature_toward_the_target_by_lambda_squared():
    """lam = sqrt(1 + (dt/tau)(T_target/T - 1)) so T_new = T * lam^2 exactly."""
    masses = np.full(20, 12.011)
    rng = np.random.default_rng(8)
    vel = md._init_velocities(masses, 300.0, rng=rng)
    start = md.compute_temperature(vel, masses)
    scaled = md.berendsen_thermostat(vel, start, 600.0, tau=1.0, dt=0.5)
    lam_sq = 1.0 + 0.5 * (600.0 / start - 1.0)
    assert md.compute_temperature(scaled, masses) == pytest.approx(start * lam_sq,
                                                                  rel=1e-12)
    assert start < md.compute_temperature(scaled, masses) < 600.0


def test_the_thermostat_cools_a_hot_sample():
    vel = np.array([[0.1, 0.0, 0.0]])
    hot = md.compute_temperature(vel, np.array([12.011]))
    cooled = md.berendsen_thermostat(vel, hot, 1.0, tau=10.0, dt=1.0)
    assert md.compute_temperature(cooled, np.array([12.011])) < hot


def test_scaling_is_the_same_for_every_atom_regardless_of_mass():
    masses = np.array([1.008, 12.011, 200.59])
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    scaled = md.berendsen_thermostat(vel, 300.0, 450.0, tau=1.0, dt=1.0)
    ratios = scaled / vel
    assert ratios[0][0] == pytest.approx(ratios[1][1]) == pytest.approx(ratios[2][2])


def test_a_target_at_the_current_temperature_changes_nothing():
    vel = np.array([[0.03, 0.0, 0.0]])
    current = md.compute_temperature(vel, np.array([12.011]))
    np.testing.assert_allclose(
        md.berendsen_thermostat(vel, current, current, tau=1.0, dt=0.1), vel)


def test_a_still_sample_is_left_alone_instead_of_being_rescaled_by_infinity():
    vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        md.berendsen_thermostat(vel, 0.0, 300.0, tau=1.0, dt=1.0), vel)


def test_a_coupling_shorter_than_the_timestep_clamps_at_a_full_quench():
    """dt/tau = 10 toward 0 K makes lam^2 = -9; the documented answer is zero motion."""
    vel = np.array([[0.05, 0.0, 0.0]])
    quenched = md.berendsen_thermostat(vel, 300.0, 0.0, tau=0.1, dt=1.0)
    assert np.all(quenched == 0.0)


@pytest.mark.parametrize("tau", [0, 0.0, -0.1])
def test_a_non_positive_coupling_time_is_refused(tau):
    with pytest.raises(ValueError, match="tau must be a positive coupling time"):
        md.berendsen_thermostat(np.array([[1.0, 0.0, 0.0]]), 300.0, 300.0,
                                tau=tau, dt=1.0)


@pytest.mark.parametrize("dt", [0, 0.0, -1.0])
def test_a_non_positive_timestep_is_refused(dt):
    with pytest.raises(ValueError, match="dt must be a positive timestep"):
        md.berendsen_thermostat(np.array([[1.0, 0.0, 0.0]]), 300.0, 300.0,
                                tau=1.0, dt=dt)


def test_a_negative_target_is_refused():
    with pytest.raises(ValueError, match="target temperature cannot be negative"):
        md.berendsen_thermostat(np.array([[1.0, 0.0, 0.0]]), 300.0, -5.0,
                                tau=1.0, dt=1.0)


def test_a_mass_array_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match=r"masses has 2 entries but the "
                                         r"coordinate set has 3"):
        md.compute_temperature(np.zeros((3, 3)), np.array([12.0, 12.0]))
