"""biosuite.core.md_simulation: reading the fixed columns of a PDB file.

Every defect this file pins is one the 80-column format invites: a record that is
too short, a serial written as text, coordinates that a truncated download
left half-written.  The parser's promise is that a bad record costs one atom and
says so, not that it costs the run in silence.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def atom_line(serial=1, name="CA", alt=" ", res="ALA", chain="A", seq=1, icode=" ",
              x=0.0, y=0.0, z=0.0, occ=1.0, bfac=0.0, element="C", record="ATOM  "):
    return (f"{record:<6s}{serial:>5d} {name:<4s}{alt}{res:>3s} {chain}{seq:>4d}{icode}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{bfac:>6.2f}          {element:>2s}")


def poke(line, column, text):
    """Overwrite PDB columns (1-based) with literal text, width preserved."""
    start = column - 1
    return line[:start] + text + line[start + len(text):]


@pytest.fixture
def pdb_file(tmp_path):
    def write(lines, name="s.pdb"):
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
        return str(path)
    return write


def test_a_clean_file_yields_atoms_with_their_metadata(pdb_file):
    path = pdb_file([
        atom_line(1, "N", res="ALA", chain="A", seq=1, x=1.5, y=-2.5, z=3.0,
                  occ=0.75, bfac=12.5, element="N"),
        atom_line(2, "SG", res="CYS", chain="B", seq=42, x=0.0, y=1.0, z=2.0,
                  element="S"),
    ])
    st = md.parse_pdb(path)
    assert st.n_atoms == 2
    first, second = st.atoms
    assert (first.serial, first.name, first.res_name, first.chain_id, first.res_seq) == \
        (1, "N", "ALA", "A", 1)
    assert (second.serial, second.name, second.res_name, second.chain_id, second.res_seq) == \
        (2, "SG", "CYS", "B", 42)
    assert first.occupancy == pytest.approx(0.75)
    assert first.temp_factor == pytest.approx(12.5)
    assert first.element == "N" and second.element == "S"
    np.testing.assert_allclose(st.coordinates, [[1.5, -2.5, 3.0], [0.0, 1.0, 2.0]])
    assert st.atom_names == ["N", "SG"]
    assert st.residue_names == ["ALA", "CYS"]
    assert st.chain_ids == ["A", "B"]


def test_title_records_are_joined_and_blank_fields_defaulted(pdb_file):
    path = pdb_file([
        "TITLE     FIRST PART",
        "TITLE     SECOND PART",
        "REMARK   2 RESOLUTION    1.80 A",
        atom_line(1, "C", element=" "),
    ])
    st = md.parse_pdb(path)
    assert st.title == "FIRST PART SECOND PART"
    assert len(st.remarks) == 1
    assert st.remarks[0].startswith("REMARK   2")
    assert st.atoms[0].element == "C", "a blank element column falls back to the name"
    assert st.atoms[0].chain_id == "A"


def test_a_non_numeric_serial_is_counted_and_reported(pdb_file, caplog):
    path = pdb_file([
        poke(atom_line(1, "CA"), 7, "XXXXX"),
        atom_line(2, "CA", x=3.0),
    ])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(path)
    assert st.n_atoms == 2
    assert [a.serial for a in st.atoms] == [1, 2], "the row number stands in"
    assert "1 atom record(s) had an unreadable serial number" in caplog.text
    assert "unreadable residue number" not in caplog.text


def test_a_non_numeric_residue_number_becomes_zero_and_is_reported(pdb_file, caplog):
    path = pdb_file([poke(atom_line(1, "CA", seq=7), 23, "  AB")])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(path)
    assert st.atoms[0].res_seq == 0
    assert st.atoms[0].serial == 1, "a readable serial is left alone"
    assert "1 atom record(s) had an unreadable residue number" in caplog.text
    assert "unreadable serial number" not in caplog.text


def test_unparseable_coordinates_drop_the_record_and_say_which_line(pdb_file, caplog):
    broken = poke(atom_line(2, "N", x=1.0, element="N"), 31, "      XX")
    path = pdb_file([atom_line(1, "CA"), broken, atom_line(3, "C", x=2.0)])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(path)
    assert st.n_atoms == 2
    assert [a.serial for a in st.atoms] == [1, 3]
    assert "1 atom record(s) skipped" in caplog.text
    assert "first at line 2" in caplog.text


def test_the_element_list_is_one_per_atom_in_the_same_order(pdb_file):
    st = md.parse_pdb(pdb_file([
        atom_line(1, "CA", element="C"), atom_line(2, "SG", res="CYS", element="S"),
        atom_line(3, "N", element=" "),
    ]))
    assert st.elements == ["C", "S", "N"], "the third record has no element column"


def test_several_bad_records_are_counted_and_the_first_is_named(pdb_file, caplog):
    """The report says how many were lost and where the loss started."""
    path = pdb_file([
        atom_line(1, "CA"),
        poke(atom_line(2, "N", element="N"), 31, "      XX"),
        poke(atom_line(3, "O", element="O"), 39, "      YY"),
        atom_line(4, "CB", x=3.0),
    ])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(path)
    assert st.n_atoms == 2
    assert [a.serial for a in st.atoms] == [1, 4]
    assert "2 atom record(s) skipped" in caplog.text
    assert "first at line 2" in caplog.text


def test_a_truncated_line_loses_only_itself(pdb_file, caplog):
    path = pdb_file([atom_line(1, "CA"), atom_line(2, "CA")[:34]])
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(path)
    assert st.n_atoms == 1
    assert "1 atom record(s) skipped" in caplog.text


def test_blank_occupancy_and_bfactor_columns_take_the_pdb_defaults(pdb_file):
    line = atom_line(1, "CA", occ=1.0, bfac=0.0)
    line = line[:54] + "      " + line[66:]
    st = md.parse_pdb(pdb_file([line]))
    assert (st.atoms[0].occupancy, st.atoms[0].temp_factor) == (1.0, 0.0)


def test_hetatm_records_are_parsed_like_atoms(pdb_file):
    st = md.parse_pdb(pdb_file([atom_line(1, "ZN", res="ZN", element="ZN",
                                          record="HETATM")]))
    assert st.n_atoms == 1
    assert st.atoms[0].element == "ZN"
    assert st.masses == pytest.approx([65.38])


def test_a_file_of_only_headers_has_no_atoms(pdb_file):
    st = md.parse_pdb(pdb_file(["HEADER    SOME TITLE", "TITLE     NOTHING",
                                "TER", "END"]))
    assert st.n_atoms == 0
    assert st.coordinates.shape == (0, 3)
    assert st.chain_ids == []


def test_the_element_column_wins_over_the_name_guess(pdb_file):
    """A calcium record named 'CA' with 'CA' in the element columns is calcium."""
    st = md.parse_pdb(pdb_file([atom_line(1, "CA", res="CA", element="CA",
                                         record="HETATM")]))
    assert st.atoms[0].element == "CA"


def test_reading_a_missing_file_raises_the_os_error_not_a_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        md.parse_pdb(str(tmp_path / "absent.pdb"))


def test_coordinate_columns_are_read_as_floats_even_for_negative_numbers(pdb_file):
    st = md.parse_pdb(pdb_file([atom_line(1, "CA", x=-123.456, y=-0.5, z=999.999)]))
    np.testing.assert_allclose(st.coordinates[0], [-123.456, -0.5, 999.999], atol=1e-6)


def test_masses_and_lj_parameters_describe_the_same_atoms(tmp_path):
    """A parse is only useful if its arrays line up with each other."""
    path = tmp_path / "a.pdb"
    path.write_text("\n".join([atom_line(1, "CA"), atom_line(2, "N", element="N"),
                               atom_line(3, "SG", res="CYS", element="S")]) + "\nEND\n",
                    encoding="utf-8")
    st = md.parse_pdb(str(path))
    lj = md.assign_lj_parameters(st.atoms)
    assert len(lj.sigma) == len(st.masses) == st.n_atoms == 3
    assert np.asarray(st.masses, dtype=np.float64).sum() == pytest.approx(12.011 + 14.007 + 32.06)
    assert isinstance(lj, md.LJParameters)
