"""biosuite.core.md_simulation: which sigma and epsilon each atom gets.

The table covers the seven names a protein uses.  Everything else — every
phosphorus in a nucleic acid, every metal in a cofactor — falls to a generic
entry, and the module's job is to say so instead of quietly modelling a zinc with
carbon's parameters.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def atom(serial, name, element, res_name="ALA", chain="A", res_seq=1):
    return md.Atom(serial=serial, name=name, alt_loc=" ", res_name=res_name,
                   chain_id=chain, res_seq=res_seq, icode=" ", x=0.0, y=0.0, z=0.0,
                   occupancy=1.0, temp_factor=0.0, element=element)


def test_the_named_atoms_take_their_tabulated_parameters():
    lj = md.assign_lj_parameters([
        atom(1, "CA", "C"), atom(2, "N", "N"), atom(3, "O", "O"),
        atom(4, "SG", "S"), atom(5, "HA", "H"),
    ])
    assert lj.sigma.tolist() == [3.5, 3.25, 3.0, 3.6, 2.5]
    assert lj.epsilon.tolist() == [0.066, 0.17, 0.17, 0.25, 0.03]


def test_an_atom_name_beats_its_element():
    """CA and CB are carbon by name; the name lookup is tried first on purpose."""
    lj = md.assign_lj_parameters([atom(1, "CB", "S")])
    assert (lj.sigma[0], lj.epsilon[0]) == (3.5, 0.066)


def test_an_unnamed_atom_falls_back_to_its_element():
    lj = md.assign_lj_parameters([atom(1, "XX", "O")])
    assert (lj.sigma[0], lj.epsilon[0]) == (3.0, 0.17)


def test_a_type_outside_the_table_gets_the_generic_entry_and_a_warning(caplog):
    atoms = [atom(1, "P", "P"), atom(2, "PG", "P"), atom(3, "ZN", "ZN"),
             atom(4, "OP1", "O")]
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        lj = md.assign_lj_parameters(atoms)
    assert lj.sigma.tolist() == [3.4, 3.4, 3.4, 3.0]
    assert lj.epsilon.tolist() == [0.066, 0.066, 0.066, 0.17]
    assert "'P / P' (1 atom(s))" in caplog.text
    assert "'PG / P' (1 atom(s))" in caplog.text
    assert "'ZN / ZN' (1 atom(s))" in caplog.text
    assert "OP1" not in caplog.text, "an oxygen by element is a tabulated type"
    assert "generic" in caplog.text


def test_a_tabulated_structure_warns_about_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        md.assign_lj_parameters([atom(1, "CA", "C"), atom(2, "N", "N")])
    assert caplog.text == ""


def test_no_atoms_produce_empty_arrays_not_an_index_error():
    lj = md.assign_lj_parameters([])
    assert lj.sigma.shape == (0,) and lj.epsilon.shape == (0,)


def test_the_parameters_are_floats_the_energy_can_use():
    lj = md.assign_lj_parameters([atom(1, "CA", "C")])
    assert lj.sigma.dtype == np.float64 and lj.epsilon.dtype == np.float64
