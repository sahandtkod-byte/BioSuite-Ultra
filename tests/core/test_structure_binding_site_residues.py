"""Which residues ``find_binding_sites`` counts as part of a binding site, and what it
refuses to search with.

Measured before: with a HEM ligand and three waters in the same shell, the function
returned ``[ALA, HOH, HOH, HOH, GLY]`` - a water shell presented as the binding site,
which on a real deposition outnumbers the protein contacts - and a ``radius`` of ``'8'``
reached Biopython's neighbour search and came back as ``TypeError: must be real number,
not str``, while ``radius=0`` answered "no binding site", a different claim from "that
radius is nonsense".
"""
import re

import numpy as np
import pytest
from Bio.PDB import NeighborSearch

from biosuite.core import structure as S

pytestmark = pytest.mark.skipif(not S.HAS_BIO, reason="biopython not installed")


def _atom(record, serial, name, resname, chain, resseq, coord, element="C"):
    return ("%-6s%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00%6.2f%10s%2s"
            % (record, serial, name, resname, chain, resseq,
               coord[0], coord[1], coord[2], 0.0, "", element))


def _residue(lines, serial, resname, chain, resseq, x):
    """One N/CA/C/O tetramer starting at x, 0.5 A apart along x."""
    for offset, name, element in ((0.0, "N", "N"), (0.5, "CA", "C"),
                                  (1.0, "C", "C"), (1.5, "O", "O")):
        lines.append(_atom("ATOM", serial, name, resname, chain, resseq,
                           np.array([x + offset, 0.0, 0.0]), element=element))
        serial += 1
    return serial


@pytest.fixture
def ligand_with_waters(tmp_path):
    """ALA and GLY flanking a three-atom HEM, with three waters in the same shell."""
    lines, serial = [], 1
    serial = _residue(lines, serial, "ALA", "A", 1, 0.0)
    serial = _residue(lines, serial, "GLY", "A", 2, 12.0)
    for x in (0.0, 6.0, 12.0):
        lines.append(_atom("HETATM", serial, "O", "HOH", "W", 900 + serial,
                           np.array([x, 0.2, 1.5]), element="O"))
        serial += 1
    for x, name, element in ((0.0, "FE", "FE"), (6.0, "N2", "N"), (12.0, "N1", "N")):
        lines.append(_atom("HETATM", serial, name, "HEM", "B", 400,
                           np.array([x, 0.0, 3.0]), element=element))
        serial += 1
    path = tmp_path / "ligand_with_waters.pdb"
    path.write_text("\n".join(lines) + "\nTER\nEND\n", encoding="utf-8")
    structure, error = S.parse_pdb(str(path))
    assert error is None
    return structure


def test_water_is_not_reported_as_a_binding_site_residue(ligand_with_waters):
    found = S.find_binding_sites(ligand_with_waters, radius=8.0)
    assert [entry['residue'] for entry in found] == ['ALA', 'GLY']
    assert [entry['number'] for entry in found] == [1, 2]
    assert {entry['chain'] for entry in found} == {'A'}

    # Positive control: the water shell is genuinely in contact, so it is the filter
    # that removes it and not the geometry.
    atoms = list(ligand_with_waters[0].get_atoms())
    ligand = [a for r in ligand_with_waters[0].get_residues()
              if r.get_resname() == 'HEM' for a in r.get_atoms()]
    contacted = {res.get_resname() for atom in ligand
                 for res in NeighborSearch(atoms).search(atom.coord, 8.0, 'R')}
    assert contacted == {'ALA', 'GLY', 'HEM', 'HOH'}


def test_a_residue_is_listed_once_even_when_several_atoms_contact(ligand_with_waters):
    """The ligand spans 12 A: every ligand atom seeds the search, so a residue
    contacted by two of them must not appear twice (the original bug used only the
    first atom and crashed on ``Atom.vector``)."""
    found = S.find_binding_sites(ligand_with_waters, radius=8.0)
    assert len(found) == len({(entry['chain'], entry['number']) for entry in found})
    assert len(found) == 2


def test_searching_from_a_different_ligand_finds_the_other_side(ligand_with_waters):
    found = S.find_binding_sites(ligand_with_waters, ligand_residue='ALA', radius=4.0)
    assert found == [{'residue': 'HEM', 'chain': 'B', 'number': 400}]


def test_a_ligand_that_is_not_in_the_file_returns_nothing(ligand_with_waters):
    assert S.find_binding_sites(ligand_with_waters, ligand_residue='ZZZ') == []


@pytest.mark.parametrize("radius", [
    pytest.param('8', id="string"),
    pytest.param(None, id="none"),
    pytest.param(True, id="bool"),
    pytest.param(0, id="zero"),
    pytest.param(-5.0, id="negative"),
])
def test_the_search_refuses_a_radius_it_cannot_use(ligand_with_waters, radius):
    with pytest.raises(ValueError, match=re.escape(
            f"radius must be a positive number, got {radius!r}.")):
        S.find_binding_sites(ligand_with_waters, radius=radius)


def test_the_radius_is_checked_before_the_ligand_is_scanned(ligand_with_waters):
    """The refusal must not depend on the file having a ligand in it: the argument is
    wrong either way, and the old code only found out after the residue walk."""
    with pytest.raises(ValueError, match=re.escape(
            "radius must be a positive number, got '8'.")):
        S.find_binding_sites(ligand_with_waters, ligand_residue='ZZZ', radius='8')
