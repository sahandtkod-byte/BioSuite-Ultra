"""biosuite.core.md_simulation: the atomic masses a simulation runs on.

Mass is not decoration: it sets the Maxwell-Boltzmann velocity draw, the
reported temperature and the radius of gyration, so a wrong mass quietly changes
every observable.  The table used to be keyed by a mix of element symbols and PDB
atom names, which made calcium 12.011 amu and left phosphorus — every nucleic
acid backbone atom — missing entirely.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def atom_line(serial=1, name="CA", res="ALA", chain="A", seq=1, x=0.0, y=0.0, z=0.0,
              element="C", record="ATOM  ", occ=1.0, bfac=0.0):
    return (f"{record:<6s}{serial:>5d} {name:<4s} {res:>3s} {chain}{seq:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{bfac:>6.2f}          {element:>2s}")


def structure(tmp_path, lines):
    path = tmp_path / "s.pdb"
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return md.parse_pdb(str(path))


@pytest.mark.parametrize("element, mass", [
    ("H", 1.008), ("C", 12.011), ("N", 14.007), ("O", 15.999), ("S", 32.06),
    ("P", 30.974), ("SE", 78.971),
    ("NA", 22.990), ("MG", 24.305), ("K", 39.098), ("CA", 40.078),
    ("MN", 54.938), ("FE", 55.845), ("CO", 58.933), ("NI", 58.693),
    ("CU", 63.546), ("ZN", 65.38), ("MO", 95.95), ("CD", 112.414),
    ("CL", 35.45), ("BR", 79.904), ("I", 126.904), ("HG", 200.592),
])
def test_every_element_the_table_claims_has_the_documented_mass(tmp_path, element, mass):
    """The value is the standard atomic weight, not a placeholder."""
    st = structure(tmp_path, [atom_line(1, "XX", element=element, record="HETATM")])
    assert st.masses == pytest.approx([mass])


def test_a_calcium_ion_is_not_an_alpha_carbon(tmp_path):
    st = structure(tmp_path, [
        atom_line(1, "CA", element="CA", record="HETATM"),
        atom_line(2, "CA", element="C"),
    ])
    assert st.masses.tolist() == [40.078, 12.011]


def test_an_unknown_element_is_carbon_and_says_so(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = structure(tmp_path, [
            atom_line(1, "XX", element="U", record="HETATM"),
            atom_line(2, "YY", element="U", record="HETATM"),
            atom_line(3, "C", element="C"),
        ])
    assert st.masses.tolist() == [12.011, 12.011, 12.011]
    assert "'U' (2 atom(s))" in caplog.text
    assert "12.011 amu" in caplog.text, "the substitution is stated as a number"

    caplog.clear()
    structure(tmp_path, [atom_line(1, "CA", element="C")])
    assert caplog.text == "", "a tabulated element must not warn"


def test_the_element_is_matched_case_insensitively_and_ignores_padding(tmp_path):
    st = structure(tmp_path, [atom_line(1, "XX", element="fe", record="HETATM")])
    assert st.masses == pytest.approx([55.845])


def test_masses_are_floats_the_integrator_can_divide_by(tmp_path):
    st = structure(tmp_path, [atom_line(1, "CA"), atom_line(2, "N", element="N")])
    assert st.masses.dtype == np.float64
    assert np.all(st.masses > 0)
