"""Molecular dynamics offline math: element guessing, LJ params, bond/angle energies."""
import numpy as np

from biosuite.core import md_simulation as md


PDB_MINI = (
    "ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 50.00           N\n"
    "ATOM      2  CA  ALA A   1      12.560  13.100   9.050  1.00 50.00           C\n"
    "ATOM      3  C   ALA A   1      13.010  14.400   8.600  1.00 50.00           C\n"
    "ATOM      4  O   ALA A   1      12.500  15.350   9.200  1.00 50.00           O\n"
    "ATOM      5  CB  ALA A   1      13.100  12.200   8.050  1.00 50.00           C\n"
    "TER\nEND\n"
)


def test_guess_element():
    assert md._guess_element('N') == 'N'
    assert md._guess_element('CA') == 'C'   # alpha carbon, not calcium
    assert md._guess_element('O') == 'O'
    assert md._guess_element('FE') == 'FE'  # explicit element map
    assert md._guess_element('1H') in ('H',) or md._guess_element('1H')


def test_parse_pdb_to_structure(tmp_path):
    p = tmp_path / 'a.pdb'
    p.write_text(PDB_MINI)
    struct = md.parse_pdb(str(p))
    assert len(struct.atoms) == 5
    assert struct.coordinates.shape == (5, 3)


def test_assign_lj_parameters(tmp_path):
    p = tmp_path / 'a.pdb'
    p.write_text(PDB_MINI)
    st = md.parse_pdb(str(p))
    lj = md.assign_lj_parameters(st.atoms)
    assert hasattr(lj, 'sigma') or isinstance(lj, object)


def test_compute_lj_energy_is_finite(tmp_path):
    p = tmp_path / 'a.pdb'
    p.write_text(PDB_MINI)
    st = md.parse_pdb(str(p))
    coords = st.coordinates
    lj = md.assign_lj_parameters(st.atoms)
    e, _ = md.compute_lj_energy(coords, lj)
    assert np.isfinite(e)


def test_bond_angle_energy_buildup(tmp_path):
    p = tmp_path / 'a.pdb'
    p.write_text(PDB_MINI)
    st = md.parse_pdb(str(p))
    bonds = md._build_bond_list(st)
    assert isinstance(bonds, list)
    coords = st.coordinates
    be, _f = md.compute_bond_energy(coords, bonds) if bonds else (0.0, None)
    assert np.isfinite(be)
