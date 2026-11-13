"""biosuite.core.md_simulation: which pairs of atoms become bonds.

There is no topology in a bare PDB coordinate list, so connectivity is inferred
from distance inside an open interval.  What the interval excludes matters: an
atom pair closer than 0.5 A is not bonded, it is a duplicate that has just
disappeared from the bond list, and a structure whose atoms are all 3.8 A apart
gets no bonds at all and runs as a box of independent atoms.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def structure(*positions, element="C"):
    atoms = [md.Atom(serial=i + 1, name="CA", alt_loc=" ", res_name="ALA",
                     chain_id="A", res_seq=1, icode=" ", x=x, y=y, z=z,
                     occupancy=1.0, temp_factor=0.0, element=element)
             for i, (x, y, z) in enumerate(positions)]
    return md.PDBStructure(atoms=atoms)


def test_a_distance_inside_the_window_becomes_a_bond_at_that_length():
    st = structure((0.0, 0.0, 0.0), (1.45, 0.0, 0.0))
    assert md._build_bond_list(st) == [(0, 1, 1.45)]


def test_the_equilibrium_length_is_the_distance_the_file_happens_to_hold():
    """The price of bonding without a topology: a clash is kept by its own bond."""
    st = structure((0.0, 0.0, 0.0), (0.9, 0.0, 0.0))
    bonds = md._build_bond_list(st)
    assert bonds == [(0, 1, 0.9)], "0.9 A is stored as the bond's rest length"


@pytest.mark.parametrize("distance", [0.4, 0.5])
def test_the_lower_edge_of_the_window_is_open(distance):
    st = structure((0.0, 0.0, 0.0), (distance, 0.0, 0.0))
    assert md._build_bond_list(st) == []


def test_the_upper_edge_of_the_window_is_open_too():
    st = structure((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert md._build_bond_list(st) == []


def test_a_calcium_alpha_carbon_spacing_is_not_a_bond():
    st = structure((0.0, 0.0, 0.0), (3.8, 0.0, 0.0))
    assert md._build_bond_list(st) == []


def test_bonds_are_indexed_by_atom_order_and_listed_once_per_pair():
    st = structure((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0))
    bonds = md._build_bond_list(st)
    assert bonds == [(0, 1, 1.5), (1, 2, 1.5)], "0-2 is 3.0 A, outside the window"
    assert all(i < j for i, j, _r0 in bonds)


def test_the_window_itself_can_be_widened():
    st = structure((0.0, 0.0, 0.0), (3.8, 0.0, 0.0))
    assert len(md._build_bond_list(st, max_bond_len=4.0)) == 1


@pytest.mark.parametrize("limit", [0, 0.0, -1.0])
def test_a_non_positive_bond_cutoff_is_refused(limit):
    """A zero window silently returns no bonds for every structure."""
    with pytest.raises(ValueError, match="max_bond_len must be a positive distance"):
        md._build_bond_list(structure((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
                            max_bond_len=limit)


def test_a_structure_with_no_bonds_at_all_says_so(caplog):
    """A CA-only trace is 3.8 A between neighbours: no bonds, so no angles either."""
    st = structure(*[(i * 3.8, 0.0, 0.0) for i in range(4)])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        bonds = md._build_bond_list(st)
    assert bonds == []
    assert "no pair of the 4 atom(s) is within 2.00" in caplog.text
    assert "no bond or angle term" in caplog.text


def test_one_atom_is_not_a_bond_and_gets_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        assert md._build_bond_list(structure((0.0, 0.0, 0.0))) == []
        assert md._build_bond_list(structure()) == []
    assert caplog.text == ""


def test_the_bond_lengths_arrive_as_plain_floats():
    st = structure((0.0, 0.0, 0.0), (1.45, 0.0, 0.0))
    _i, _j, r0 = md._build_bond_list(st)[0]
    assert type(r0) is float   # a Python float, not numpy.float64
    assert r0 == 1.45
