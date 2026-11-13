"""Reading ATOM/HETATM records out of a fixed-width PDB file.

``_parse_pdb_atoms`` is the only thing that turns a receptor on disk into the
coordinates every score in this module is computed from, so the columns it reads
and the records it throws away are both part of the contract.  A record it cannot
read must be counted and reported, never dropped in silence.
"""
import pytest

from biosuite.core import docking as dk


def atom_line(serial=1, name="CA", res="GLY", chain="A", x=0.0, y=0.0, z=0.0,
              record="ATOM"):
    """One coordinate record with the fields in their PDB columns."""
    return (f"{record:<6}{serial:>5} {name:<4} {res:>3} {chain}{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")


@pytest.fixture
def receptor(tmp_path):
    path = tmp_path / "rec.pdb"
    path.write_text("HEADER    test\n"
                    + atom_line(1, chain="A", x=0.0, y=0.0, z=0.0)
                    + atom_line(2, chain="A", x=4.0, y=0.0, z=0.0)
                    + atom_line(3, chain="B", x=100.0, y=100.0, z=100.0),
                    encoding="utf-8")
    return str(path)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_chain_filter_keeps_only_the_chain_it_was_named(receptor):
    everything = dk._parse_pdb_atoms(receptor)
    chain_a = dk._parse_pdb_atoms(receptor, chain='A')

    assert len(everything) == 3, "no chain asked for means every chain read"
    assert [a['chain'] for a in chain_a] == ['A', 'A'], chain_a
    assert chain_a[1]['x'] == pytest.approx(4.0), "column 31-38 is x"
    assert chain_a[0]['name'] == 'CA' and chain_a[0]['res'] == 'GLY'


def test_the_three_coordinate_columns_are_read_from_their_own_offsets(tmp_path):
    path = _write(tmp_path, "x.pdb", atom_line(1, x=-12.5, y=0.25, z=999.875))

    atom, = dk._parse_pdb_atoms(path)

    assert (atom['x'], atom['y'], atom['z']) == (-12.5, 0.25, 999.875), atom


def test_hetatm_records_are_atoms_too(tmp_path):
    path = _write(tmp_path, "lig.pdb",
                  atom_line(1, name="C1", res="LIG", chain="B", x=8.0, record="HETATM"))

    atoms = dk._parse_pdb_atoms(path)

    assert [a['res'] for a in atoms] == ['LIG'], atoms
    assert atoms[0]['chain'] == 'B' and atoms[0]['name'] == 'C1'


def test_a_record_whose_coordinate_columns_are_not_numbers_is_reported(tmp_path, caplog):
    path = _write(tmp_path, "broken.pdb",
                  atom_line(1)
                  + "ATOM      2  CA  GLY A   1      XXXXX   2.000   3.000  1.00  0.00\n")

    with caplog.at_level("WARNING", logger="biosuite.core.docking"):
        atoms = dk._parse_pdb_atoms(path)

    assert len(atoms) == 1, atoms
    assert "skipped 1 ATOM/HETATM record" in caplog.text, caplog.text


def test_a_record_cut_off_before_the_chain_column_is_counted_as_malformed(tmp_path, caplog):
    """'ATOM' plus a serial number is not a coordinate record.

    The line is too short for column 22, so ``line[21]`` — not a coordinate —
    is what fails, and it has to reach the same counter.
    """
    path = _write(tmp_path, "short.pdb", atom_line(1) + "ATOM      2\n")

    with caplog.at_level("WARNING", logger="biosuite.core.docking"):
        atoms = dk._parse_pdb_atoms(path)

    assert len(atoms) == 1, atoms
    assert "skipped 1 ATOM/HETATM record" in caplog.text, caplog.text


def test_a_bad_record_in_a_chain_that_was_not_asked_for_is_not_counted(tmp_path, caplog):
    """The warning counts records the caller would have received.

    Chain B's broken atom is not a loss in a chain-A run, and reporting it would
    point the user at a file region that never mattered.
    """
    path = _write(tmp_path, "two_chain.pdb",
                  atom_line(1, chain="A")
                  + "ATOM      2  CA  GLY B   1      XXXXX   2.000   3.000  1.00  0.00\n")

    with caplog.at_level("WARNING", logger="biosuite.core.docking"):
        atoms = dk._parse_pdb_atoms(path, chain='A')

    assert len(atoms) == 1, atoms
    assert "skipped" not in caplog.text, caplog.text


def test_a_file_of_headers_and_record_names_yields_no_atoms(tmp_path):
    path = _write(tmp_path, "empty.pdb", "HEADER only\nTER\nEND\n")

    assert dk._parse_pdb_atoms(path) == []


def test_an_absent_chain_yields_no_atoms(receptor):
    """A chain that is not in the file is an empty result, not an error."""
    assert dk._parse_pdb_atoms(receptor, chain='Z') == []


def test_xyz_answers_none_for_a_line_that_is_not_a_coordinate_record():
    assert dk._xyz("ATOM      1  CA  GLY A   1      1.000\n") is None
    assert dk._xyz("ATOM\n") is None
    assert dk._xyz(atom_line(1, x=1.0, y=2.0, z=3.0)) == (1.0, 2.0, 3.0)


def test_the_centroid_of_no_atoms_is_nan_not_the_origin():
    """(0, 0, 0) is a real place in a PDB file, so it cannot mean "unknown"."""
    import math

    empty = dk._centroid([])
    assert len(empty) == 3 and all(math.isnan(value) for value in empty), empty
    assert dk._centroid([(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)]) == [1.0, 2.0, 3.0]


def test_a_receptor_without_atoms_has_no_centroid(tmp_path):
    path = _write(tmp_path, "headers.pdb", "HEADER only\n")

    assert dk._receptor_centroid(path) is None
