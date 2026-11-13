"""biosuite.core.md_simulation: one Velocity Verlet step.

The step is the reason the module exists, and it is where unit mistakes hide:
forces are kcal/mol/A, masses are amu and time is femtoseconds, so the
conversion factor has to be exactly 4.184e-4 A/fs^2 per (kcal/mol/A)/amu.  These
tests recompute the arithmetic independently instead of trusting the run.
"""
import numpy as np
import pytest

from biosuite.core import md_simulation as md

KCAL_PER_MOL_TO_NEWTON_PER_ANGSTROM = 4184.0 / md.AVOGADRO / 1e-10
FORCE_TO_ACCELERATION = KCAL_PER_MOL_TO_NEWTON_PER_ANGSTROM / (
    md.amu_to_kg * 1e20)


def far_apart_lj(n=2):
    """Parameters with no interacting pairs, so the force is exactly zero."""
    return md.LJParameters(np.full(n, 3.5), np.full(n, 0.066))


def test_the_unit_conversion_is_the_documented_one():
    assert FORCE_TO_ACCELERATION == pytest.approx(4.184e-4, rel=1e-4)


def test_a_force_moves_a_light_atom_further_than_a_heavy_one():
    coords = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    lj = far_apart_lj()
    light, _v, _f, _e = md.velocity_verlet_step(
        coords, np.zeros((2, 3)), forces, np.array([1.0, 1.0]), lj, [], [],
        dt=1.0, target_temp=0.0, tau=1.0)
    heavy, _v2, _f2, _e2 = md.velocity_verlet_step(
        coords, np.zeros((2, 3)), forces, np.array([4.0, 4.0]), lj, [], [],
        dt=1.0, target_temp=0.0, tau=1.0)
    assert light[0][0] == pytest.approx(0.5 * FORCE_TO_ACCELERATION / 1.0, rel=1e-9)
    assert heavy[0][0] == pytest.approx(light[0][0] / 4.0, rel=1e-9)


def test_the_position_update_uses_the_half_step_kinematics():
    """x(t+dt) = x + v*dt + a*dt^2/2, which is Verlet's position half-step sum."""
    coords = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    velocities = np.array([[0.002, 0.0, 0.0], [-0.002, 0.0, 0.0]])
    forces = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    masses = np.array([12.011, 12.011])
    lj = far_apart_lj()
    dt = 0.5
    new_coords, _new_vel, _f, energy = md.velocity_verlet_step(
        coords, velocities, forces, masses, lj, [], [], dt=dt,
        target_temp=0.0, tau=1e9)
    acc = forces * FORCE_TO_ACCELERATION / masses[:, np.newaxis]
    expected_coords = coords + velocities * dt + 0.5 * acc * dt * dt
    np.testing.assert_allclose(new_coords, expected_coords, atol=1e-14)
    assert new_coords.shape == (2, 3)
    assert energy == pytest.approx(md.lj_cutoff_correction(lj.sigma, lj.epsilon,
                                                           10.0, 2), rel=1e-12)


def test_the_thermostat_is_applied_after_the_second_half_step():
    """A target of 0 K with a short coupling time quenches the new velocities."""
    coords = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    velocities = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    lj = far_apart_lj()
    _c, new_vel, _f, _e = md.velocity_verlet_step(
        coords, velocities, np.zeros((2, 3)), np.array([12.011, 12.011]), lj,
        [], [], dt=1.0, target_temp=0.0, tau=0.1)
    assert np.all(new_vel == 0.0)


def test_a_step_at_the_target_temperature_leaves_the_sample_broadly_alone():
    masses = np.full(50, 12.011)
    coords = np.array([[i * 4.5, 0.0, 0.0] for i in range(50)])
    lj = far_apart_lj(50)
    velocities = md._init_velocities(masses, 300.0, rng=np.random.default_rng(6))
    _e, forces = md.compute_forces(coords, lj, [], [])
    _c, new_vel, _f, _e2 = md.velocity_verlet_step(
        coords, velocities, forces, masses, lj, [], [], dt=0.5,
        target_temp=300.0, tau=100.0)
    assert md.compute_temperature(new_vel, masses) == pytest.approx(300.0, rel=0.05)


def test_the_returned_forces_are_the_ones_at_the_new_positions():
    coords = np.array([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]])
    lj = far_apart_lj()
    new_coords, _v, new_forces, _e = md.velocity_verlet_step(
        coords, np.zeros((2, 3)), np.zeros((2, 3)), np.array([12.011, 12.011]),
        lj, [], [], dt=1.0, target_temp=0.0, tau=1.0)
    expected = md.compute_forces(new_coords, lj, [], [])[1]
    np.testing.assert_allclose(new_forces, expected, atol=1e-12)


@pytest.mark.parametrize("dt", [0, 0.0, -1.0])
def test_a_timestep_that_moves_nothing_is_refused(dt):
    """dt = 0 used to skip the integration but still rescale the velocities."""
    with pytest.raises(ValueError, match="dt must be a positive timestep"):
        md.velocity_verlet_step(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)),
                                np.array([12.0, 12.0]), far_apart_lj(), [], [], dt=dt)


def test_a_zero_mass_is_refused_before_it_divides():
    with pytest.raises(ValueError, match="mass of atom 0 is 0.0 amu"):
        md.velocity_verlet_step(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)),
                                np.array([0.0, 12.0]), far_apart_lj(), [], [])


def test_arrays_that_describe_different_systems_are_named():
    args = dict(lj=far_apart_lj(), bonds=[], angles=[], dt=1.0)
    with pytest.raises(ValueError, match=r"masses has 2 entries"):
        md.velocity_verlet_step(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)),
                                np.array([12.0, 12.0]), **args)
    with pytest.raises(ValueError, match=r"velocities has 2 entries"):
        md.velocity_verlet_step(np.zeros((3, 3)), np.zeros((2, 3)), np.zeros((3, 3)),
                                np.array([12.0, 12.0, 12.0]), **args)
    with pytest.raises(ValueError, match=r"forces has 3 entries but the coordinate "
                                         r"set has 2"):
        md.velocity_verlet_step(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 3)),
                                np.array([12.0, 12.0]), **args)


def test_a_negative_thermostat_target_is_refused():
    with pytest.raises(ValueError, match="target temperature cannot be negative"):
        md.velocity_verlet_step(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)),
                                np.array([12.0, 12.0]), far_apart_lj(), [], [],
                                target_temp=-1.0)


def test_integer_inputs_are_promoted_not_truncated():
    coords = np.array([[0, 0, 0], [50, 0, 0]])
    new_coords, _v, _f, _e = md.velocity_verlet_step(
        coords, np.zeros((2, 3)), np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([12, 12]), far_apart_lj(), [], [], dt=1.0, target_temp=0.0, tau=1.0)
    assert new_coords.dtype == np.float64
    assert new_coords[0][0] > 0
