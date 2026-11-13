"""How ``structure.py`` reads a DSSP result.

Biopython's DSSP map yields one **flat 14-element row per residue** - its
``AbstractPropertyMap.__iter__`` walks ``property_list``, which ``DSSP`` fills with the
value tuples themselves - so the ``(key, value)`` unpacking this module used raised
``ValueError: too many values to unpack (expected 2)`` on the first row.  That error was
caught by the fallback handler, which meant the DSSP branch had never once returned DSSP
data: measured on a three-chain structure whose rows say ``H, G, E``, the function
answered the phi/psi heuristic ``{'H': 4, 'E': 4, 'C': 8}`` while the rows said
``{'H': 6, 'E': 6, 'C': 4}``, and warned *"DSSP could not be run … too many values to
unpack"*, blaming the external program for this module's own unpacking.

The tests here drive Biopython's own parser over a hand-written ``.dssp`` file - no
stand-in object - because the defect was exactly a mismatch between this module's
assumption and the library's real row shape.  Nothing calls a binary or opens a socket.
"""
import warnings

import pytest

from biosuite.core import structure as S

pytestmark = pytest.mark.skipif(not S.HAS_BIO, reason="biopython not installed")

TWO_CHAIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.200   1.000   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2      10.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2      11.458   0.000   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2      12.000   1.000   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2      13.200   1.000   0.000  1.00  0.00           O
ATOM      9  N   GLY B   1      20.000   0.000   0.000  1.00  0.00           N
ATOM     10  CA  GLY B   1      21.458   0.000   0.000  1.00  0.00           C
ATOM     11  C   GLY B   1      22.000   1.000   0.000  1.00  0.00           C
ATOM     12  O   GLY B   1      23.200   1.000   0.000  1.00  0.00           O
TER
END
"""


def _dssp_line(index, resseq, chain, aa, ss, acc=100, phi=-60.0, psi=-45.0):
    """A DSSP-format residue line: Biopython reads the structure code at column 16,
    skips a row whose column 9 is blank, and wants angles from columns 103-115."""
    buf = list(" " * 115)

    def put(start, end, text):
        buf[start:end] = list(text.ljust(end - start)[:end - start])

    put(0, 5, f"{index:5d}")
    put(5, 10, f"{resseq:5d}")
    buf[10] = " "
    buf[11] = chain
    buf[13] = aa
    buf[16] = ss
    put(34, 38, f"{acc:4d}")
    for start, end in ((38, 45), (50, 56), (61, 67), (72, 78)):
        put(start, end, "0")
    for start, end in ((46, 50), (57, 61), (68, 72), (79, 83)):
        put(start, end, "0.00")
    put(103, 109, f"{phi:6.1f}")
    put(109, 115, f"{psi:6.1f}")
    return "".join(buf)


@pytest.fixture
def two_chain_pdb(tmp_path):
    """Chain A: ALA 1-2.  Chain B: GLY 1.  Residues sit 8 A apart, so no peptide forms
    and only DSSP can call these rows helix or strand."""
    path = tmp_path / "two_chain.pdb"
    path.write_text(TWO_CHAIN_PDB, encoding="utf-8")
    return str(path)


@pytest.fixture
def two_chain_dssp(tmp_path):
    """The same three residues as DSSP itself would report them: H, G, E."""
    rows = [_dssp_line(1, 1, "A", "A", "H"), _dssp_line(2, 2, "A", "A", "G"),
            _dssp_line(3, 1, "B", "G", "E")]
    path = tmp_path / "two_chain.dssp"
    path.write_text("#\n# RESIDUE\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_real_dssp_rows_are_counted_and_the_heuristic_is_not_reached(two_chain_pdb,
                                                                    two_chain_dssp,
                                                                    monkeypatch):
    structure, error = S.parse_pdb(two_chain_pdb)
    assert error is None
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': True})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert S.compute_secondary_structure(structure, filepath=two_chain_dssp) == {
            'H': 2, 'E': 1, 'C': 0}
    # Positive control: the same structure answered through the heuristic is a
    # different result, so the numbers above can only have come from the DSSP rows.
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': False})
    assert S.compute_secondary_structure(structure, filepath=two_chain_dssp) == {
        'H': 0, 'E': 0, 'C': 0}


def test_dssp_map_shape_is_a_flat_row_per_residue_key(two_chain_pdb, two_chain_dssp):
    """The layout the module now depends on, pinned against the library: keys() are
    (chain, res_id), a row is 14 wide, and index 2 is the secondary structure code."""
    from Bio.PDB import DSSP
    structure, _ = S.parse_pdb(two_chain_pdb)
    dssp = DSSP(structure[0], two_chain_dssp)
    assert len(dssp) == 3
    assert [key[1][1] for key in dssp.keys()] == [1, 2, 1]
    rows = [dssp[key] for key in dssp.keys()]
    assert [row[2] for row in rows] == ['H', 'G', 'E']
    assert {len(row) for row in rows} == {14}
    with pytest.raises(ValueError, match="too many values to unpack"):
        for _key, _row in dssp:
            pass


def test_dssp_that_produces_no_output_falls_back_with_a_warning(two_chain_pdb,
                                                               two_chain_dssp,
                                                               monkeypatch):
    """Biopython raises a *bare* ``Exception`` when the program runs but writes nothing;
    that one failure cannot be enumerated from the outside, so it is treated as a DSSP
    failure and the caller is told."""
    structure, _ = S.parse_pdb(two_chain_pdb)

    def refuse(model, path):
        raise Exception("DSSP failed to produce an output")

    monkeypatch.setattr(S, "DSSP", refuse)
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': True})
    with pytest.warns(RuntimeWarning, match="DSSP failed to produce an output"):
        assert S.compute_secondary_structure(structure, filepath=two_chain_dssp) == {
            'H': 0, 'E': 0, 'C': 0}


def test_a_failure_that_is_not_a_dssp_failure_is_not_downgraded_to_one(two_chain_pdb,
                                                                       two_chain_dssp,
                                                                       monkeypatch):
    """A ``RuntimeError`` from inside the wrapper is a bug, not a missing binary, and it
    must not be paid for with a warning and a different measurement."""
    structure, _ = S.parse_pdb(two_chain_pdb)

    def explode(model, path):
        raise RuntimeError("a bug in here, not a DSSP problem")

    monkeypatch.setattr(S, "DSSP", explode)
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': True})
    with pytest.raises(RuntimeError, match="a bug in here, not a DSSP problem"):
        S.compute_secondary_structure(structure, filepath=two_chain_dssp)


def test_dssp_is_handed_the_model_and_the_path_and_its_absence_falls_back(two_chain_pdb,
                                                                         two_chain_dssp,
                                                                         monkeypatch):
    structure, _ = S.parse_pdb(two_chain_pdb)
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': True})
    called = []
    monkeypatch.setattr(S, "DSSP", lambda model, path: called.append((model, path)))

    assert S.compute_secondary_structure(structure, filepath=None) == {'H': 0, 'E': 0,
                                                                      'C': 0}
    assert called == [], "no path means DSSP is never constructed"

    with pytest.warns(RuntimeWarning, match="DSSP could not be run on"):
        assert S.compute_secondary_structure(structure, filepath=two_chain_dssp) == {
            'H': 0, 'E': 0, 'C': 0}
    assert len(called) == 1
    assert called[0][0] is structure[0]
    assert called[0][1] == two_chain_dssp
