"""biosuite.core.md_simulation: which element a bare atom name stands for.

The element column of a PDB file is authoritative; this guesser only runs when a
file has no column to read.  Its whole job is the PDB naming collision between
the alpha carbon and calcium, and between a leucine CD1 and cadmium.
"""
import pytest

from biosuite.core import md_simulation as md


@pytest.mark.parametrize("name, expected", [
    ("N", "N"),
    ("CA", "C"),
    ("CB", "C"),
    ("CD1", "C"),
    ("CE1", "C"),
    ("O", "O"),
    ("OG", "O"),
    ("OH", "O"),
    ("NZ", "N"),
    ("SD", "S"),
    ("SG", "S"),
    ("OP1", "O"),
    ("OP3", "O"),
    (" O ", "O"),
    ("1HB", "H"),
    ("", "C"),
    ("X", "C"),
    ("K", "K"),
    ("SE", "SE"),
])
def test_a_missing_element_column_is_read_from_the_atom_name(name, expected):
    assert md._guess_element(name) == expected


def test_the_alpha_carbon_beats_calcium_and_cadmium_beats_leucine():
    """CA and CD are element symbols that are also ordinary atom names.

    Guessing them would relabel the backbone of every protein as a metal, so
    both are deliberately absent from the ion table and only the exact-ion names
    there are honoured.
    """
    assert md._guess_element("CA") == "C"
    assert md._guess_element("CD") == "C"
    assert "CA" not in md._ION_NAMES
    assert "CD" not in md._ION_NAMES


@pytest.mark.parametrize("name", ["FE", "FE1", "FE2", "ZN", "MG1", "MN3", "CU1"])
def test_a_metal_ion_keeps_its_element(name):
    assert md._guess_element(name) == name[:2]


def test_a_name_with_a_non_digit_suffix_is_not_an_ion():
    """Only 'FE' + digits may borrow a two-letter element, so 'FES' stays fluorine."""
    assert md._guess_element("FES") == "F"


def test_a_phosphate_ester_oxygen_is_oxygen_not_phosphorus():
    """OP1/OP2/OP3 are the oxygens of a phosphate group, named after the letter."""
    assert md._guess_element("OP1") == "O"
    assert md._guess_element("P") == "P"
    assert md._guess_element("PA") == "P"
