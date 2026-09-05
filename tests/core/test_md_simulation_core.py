"""Physical-plausibility tests for the pure-Python MD simulator."""
import numpy as np
import pytest

from biosuite.core import md_simulation as md


PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.200   1.400   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       3.400   1.500   0.000  1.00 10.00           O
ATOM      5  CB  ALA A   1       1.300  -0.900   1.300  1.00 10.00           C
END
"""


@pytest.fixture()
def pdb_struct(tmp_path):
    p = tmp_path / 'ala.pdb'
    p.write_text(PDB)
    return md.parse_pdb(str(p))


def test_parse_pdb_atoms_and_elements(pdb_struct):
    assert len(pdb_struct.atoms) == 5
    assert md._guess_element('CA') == 'C'
    assert md._guess_element(' O ') == 'O'


def test_lj_params_and_cutoff_correction(pdb_struct):
    lj = md.assign_lj_parameters(pdb_struct.atoms)
    assert len(lj.sigma) == 5
    corr = md.lj_cutoff_correction(lj.sigma, lj.epsilon, cutoff=10.0, n=100)
    assert np.isfinite(corr)


def test_lj_energy_repel_at_contact(pdb_struct):
    lj = md.assign_lj_parameters(pdb_struct.atoms[:2])
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    energy, forces = md.compute_lj_energy(coords, lj)
    assert np.isfinite(energy)
    close = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    e2, _ = md.compute_lj_energy(close, lj)
    assert e2 > energy                     # overlap strongly repulsive


def test_bond_and_angle_energy(pdb_struct):
    coords = np.array([[a.x, a.y, a.z] for a in pdb_struct.atoms])
    bonds = md._build_bond_list(pdb_struct)
    assert len(bonds) >= 3
    for i, j, r0 in bonds:
        assert np.isfinite(r0)
    e_b, _ = md.compute_bond_energy(coords, bonds)
    assert np.isfinite(e_b)
    angles = md._build_angle_list(bonds)
    if angles:
        e_a, _ = md.compute_angle_energy(coords, angles)
        assert np.isfinite(e_a)


def test_temperature_and_thermostat():
    masses = np.array([12.0, 1.0, 16.0])
    v = np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 80.0]])
    t = md.compute_temperature(v, masses)
    assert t > 0
    scaled = md.berendsen_thermostat(v.copy(), masses, current_temp=t,
                                     target_temp=2 * t, dt=0.5, tau=1.0)
    t2 = md.compute_temperature(scaled, masses)
    assert t2 > t                          # heats toward hotter target


def test_velocity_verlet_and_rmsd(pdb_struct):
    coords = np.array([[a.x, a.y, a.z] for a in pdb_struct.atoms])
    lj = md.assign_lj_parameters(pdb_struct.atoms)
    v0 = md._init_velocities(pdb_struct.masses, 300.0)
    _, forces = md.compute_forces(coords, lj, bonds=[], angles=[])
    c2, v2, f2, temp = md.velocity_verlet_step(coords, v0, forces,
                                               pdb_struct.masses, lj,
                                               bonds=[], angles=[], dt=0.1)
    assert c2.shape == coords.shape and np.isfinite(c2).all()
    assert md.compute_rmsd(coords, coords) == pytest.approx(0.0)
    perturbed = coords + np.random.default_rng(0).normal(0, 0.1, coords.shape)
    assert md.compute_rmsd(perturbed, coords) > 0


def test_minimize_energy_returns_relaxed_coords(pdb_struct):
    coords = np.array([[a.x, a.y, a.z] for a in pdb_struct.atoms])
    coords_bad = coords + np.tile([0.2, 0.0, 0.0], (5, 1))
    lj = md.assign_lj_parameters(pdb_struct.atoms)
    relaxed, _history = md.minimize_energy(coords_bad, lj, [], [], max_steps=50)
    assert relaxed.shape == coords.shape
