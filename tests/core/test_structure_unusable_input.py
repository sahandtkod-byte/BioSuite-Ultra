"""What ``structure.py`` does with input it cannot measure: a file with no ATOM record, a
path it cannot read, a Biopython object of the wrong kind, and a bug inside its own read.

Measured before this cluster, ``full_analysis`` - the entry point the CLI (option 51) and
the advanced tab call - raised ``KeyError: 0`` on a header-only PDB, ``PermissionError``
on a directory and ``UnicodeDecodeError`` on a binary file, because ``load_pdb`` built its
own parser while ``parse_pdb`` had always returned ``(structure, error)``.  The failing
model lookup also sat inside the DSSP ``try``, so an empty structure was reported as
"DSSP could not be run on None: 0" before crashing anyway.  ``compute_sasa`` ended in
``except Exception: return 0.0``, which made every calculation failure indistinguishable
from a completely buried structure: 0.0 and no note.
"""
import re
import warnings

import pytest

from biosuite.core import structure as S

pytestmark = pytest.mark.skipif(not S.HAS_BIO, reason="biopython not installed")

HEADER_ONLY_PDB = "HEADER    TEST STRUCTURE\nEND\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_file_with_no_atom_record_is_measured_as_empty_rather_than_keyerror(tmp_path):
    """Prose, a FASTA and a truncated download all parse - Biopython simply finds no
    model - and every entry point used to answer ``KeyError: 0``."""
    src = _write(tmp_path, "notes.pdb", "this is not a structure file\njust two lines\n")
    structure, error = S.parse_pdb(src)
    assert error is None, "Biopython accepts the file and finds no ATOM record"

    info = S.get_structure_info(structure, "notes")
    assert info.message == "Structure contains no model"
    assert (info.num_atoms, info.num_residues, info.num_chains) == (0, 0, 0)
    assert info.chains == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert S.compute_secondary_structure(structure, filepath=src) == {}
        assert S.compute_ramachandran(structure) == {}
        assert S.find_binding_sites(structure) == []
        assert S.compute_sasa(structure) == 0.0


def test_full_analysis_of_a_file_with_no_model_returns_a_note(tmp_path):
    src = _write(tmp_path, "header_only.pdb", HEADER_ONLY_PDB)
    info = S.full_analysis(filepath=src)

    assert info.name == "loaded"
    assert info.message == "Structure contains no model"
    assert info.num_atoms == 0 and info.surface_area == 0.0
    assert S.format_structure_report(info).splitlines()[-1] == \
        "Note: Structure contains no model"


def test_load_pdb_reports_a_file_it_cannot_read_instead_of_raising(tmp_path):
    binary = tmp_path / "binary.pdb"
    binary.write_bytes(b"\xff\xfe\x00\x01\x80 not text \x81")
    structure, error = S.load_pdb(filepath=str(binary))
    assert structure is None
    assert "can't decode" in error

    empty = _write(tmp_path, "empty.pdb", "")
    assert S.load_pdb(filepath=empty) == (None, "Empty file.")

    directory = tmp_path / "a_directory"
    directory.mkdir()
    structure, error = S.load_pdb(filepath=str(directory))
    assert structure is None and "Permission denied" in error

    info = S.full_analysis(filepath=str(binary))
    assert info.num_atoms == 0 and "can't decode" in info.message


def test_downloaded_file_that_cannot_be_parsed_is_not_called_a_download_failure(
        tmp_path, monkeypatch):
    """Measured before: the download and the parse shared one ``except``, so a deposition
    that arrived intact but could not be read was reported as a failed download -
    "Could not download PDB 0BAD: Empty file."."""
    import Bio.PDB

    class FakePDBList:
        def retrieve_pdb_file(self, pdb_id, *args, **kwargs):
            path = tmp_path / f"{pdb_id}.pdb"
            path.write_text("", encoding="utf-8")
            return str(path)

    monkeypatch.setattr(Bio.PDB, "PDBList", FakePDBList)
    structure, error = S.load_pdb(pdb_id="0BAD")
    assert structure is None
    assert error == "Empty file."
    assert "Could not download" not in error


def test_parse_pdb_surfaces_a_malformed_record_but_not_an_unexpected_one(tmp_path,
                                                                        monkeypatch):
    """The old ``except Exception`` turned *any* failure into "your file is bad":
    measured raise types are OSError (unusable path), ValueError (empty or undecodable
    bytes) and Biopython's two PDB exception classes (malformed record)."""
    bad = _write(tmp_path, "badcoord.pdb",
                 "ATOM      1  CA  ALA A   1     abcde  fg.000  hijkl  1.00 20.00"
                 "           C\nEND\n")
    structure, error = S.parse_pdb(bad)
    assert structure is None
    assert error == "Invalid or missing coordinate(s) at line 1."

    class ExplodingParser:
        def get_structure(self, *args, **kwargs):
            raise TypeError("a bug in here, not a problem with the file")

    monkeypatch.setattr(S, "PDBParser", lambda QUIET=True: ExplodingParser())
    with pytest.raises(TypeError, match="a bug in here, not a problem with the file"):
        S.parse_pdb(bad)


def test_sasa_failure_warns_and_reports_zero(mini_structure):
    """``except Exception: return 0.0`` turned a calculation failure into the same
    answer as a completely buried structure, and the report then dropped the line."""
    atoms = list(mini_structure[0].get_atoms())
    with pytest.warns(RuntimeWarning, match=re.escape(
            "SASA could not be computed: ValueError: Invalid entity type")) as records:
        assert S.compute_sasa(atoms) == 0.0
    assert len(records) == 1, "the failure is reported once, not once per residue"


def test_a_real_surface_is_measured_without_a_warning(mini_structure):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert S.compute_sasa(mini_structure) == pytest.approx(138.2090783526543)


@pytest.fixture
def mini_structure(tmp_path):
    """One ALA residue, four atoms - enough for the Shrake-Rupley calculator."""
    lines = []
    for serial, (name, x, element) in enumerate(
            (("N", 0.0, "N"), ("CA", 0.5, "C"), ("C", 1.0, "C"), ("O", 1.5, "O")), start=1):
        lines.append("%-6s%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00%6.2f%10s%2s"
                     % ("ATOM", serial, name, "ALA", "A", 1, x, 0.0, 0.0, 0.0, "", element))
    path = tmp_path / "one_residue.pdb"
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    structure, error = S.parse_pdb(str(path))
    assert error is None
    return structure
