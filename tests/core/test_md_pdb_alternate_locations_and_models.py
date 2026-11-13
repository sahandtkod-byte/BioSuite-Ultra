"""biosuite.core.md_simulation: alternate locations and multi-model PDB files.

Both are about how many copies of an atom a structure should hold.  An alternate
location is a second position for the *same* atom in the same residue of the same
chain; a MODEL is a whole copy of the structure.  Keyed too loosely or read
indiscriminately, one of them silently doubles the system and the other throws a
chain away.
"""
import logging

import pytest

from biosuite.core import md_simulation as md


def atom_line(serial=1, name="CA", alt=" ", res="ALA", chain="A", seq=1, icode=" ",
              x=0.0, y=0.0, z=0.0, element="C", record="ATOM  "):
    return (f"{record:<6s}{serial:>5d} {name:<4s}{alt}{res:>3s} {chain}{seq:>4d}{icode}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2s}")


@pytest.fixture
def pdb_file(tmp_path):
    def write(lines, name="s.pdb"):
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
        return str(path)
    return write


ALT_LINES = [
    atom_line(1, "OD1", alt="A", seq=5, x=0.0, element="O"),
    atom_line(2, "OD1", alt="B", seq=5, x=1.0, element="O"),
    atom_line(3, "CG", seq=5, x=2.0, element="C"),
]


def test_first_keeps_the_first_conformer(pdb_file):
    st = md.parse_pdb(pdb_file(ALT_LINES), alt_loc_strategy="first")
    assert [a.serial for a in st.atoms] == [1, 3]
    assert st.coordinates[:, 0].tolist() == [0.0, 2.0]


def test_last_keeps_the_last_conformer(pdb_file):
    st = md.parse_pdb(pdb_file(ALT_LINES), alt_loc_strategy="last")
    assert [a.serial for a in st.atoms] == [2, 3]
    assert st.coordinates[:, 0].tolist() == [1.0, 2.0]


def test_all_keeps_every_conformer(pdb_file):
    st = md.parse_pdb(pdb_file(ALT_LINES), alt_loc_strategy="all")
    assert [a.serial for a in st.atoms] == [1, 2, 3]


def test_the_default_strategy_is_first(pdb_file):
    assert md.parse_pdb(pdb_file(ALT_LINES)).n_atoms == 2


@pytest.mark.parametrize("strategy", ["First", "FIRST", "newest", "", None, "all "])
def test_an_undocumented_strategy_is_refused_rather_than_silently_reading_everything(
        pdb_file, strategy):
    """A typo used to select no branch at all, which behaved like 'all'."""
    with pytest.raises(ValueError, match=r"alt_loc_strategy must be one of "
                                         r"\('first', 'last', 'all'\)"):
        md.parse_pdb(pdb_file(ALT_LINES), alt_loc_strategy=strategy)


def test_conformers_are_matched_within_one_residue_only(pdb_file):
    """Same atom name and number in two chains are two atoms, not two conformers.

    The lookup key used to be (name, res_seq); a homodimer whose chains both
    number their residues from 1 therefore lost every atom of the second chain.
    """
    st = md.parse_pdb(pdb_file([
        atom_line(1, "CA", chain="A", seq=1, alt="A", x=0.0),
        atom_line(2, "CA", chain="B", seq=1, alt="A", x=9.0),
    ]))
    assert st.n_atoms == 2
    assert st.coordinates[:, 0].tolist() == [0.0, 9.0]


def test_the_same_atom_in_different_residues_is_not_an_alternate_location(pdb_file):
    st = md.parse_pdb(pdb_file([
        atom_line(1, "CA", seq=1, alt="A", x=0.0),
        atom_line(2, "CA", seq=2, alt="A", x=3.8),
    ]))
    assert st.n_atoms == 2


def test_residue_and_insertion_code_are_part_of_the_identity(pdb_file):
    st = md.parse_pdb(pdb_file([
        atom_line(1, "CA", res="ALA", seq=1, alt="A", x=0.0),
        atom_line(2, "CA", res="GLY", seq=1, alt="A", x=1.0),
        atom_line(3, "CA", seq=1, icode="A", alt="A", x=2.0),
        atom_line(4, "CA", seq=1, icode="B", alt="A", x=3.0),
    ]))
    assert st.n_atoms == 4, "different residues or insertion codes are not conformers"


def test_an_insertion_code_conformer_is_still_matched_within_its_residue(pdb_file):
    st = md.parse_pdb(pdb_file([
        atom_line(1, "CA", seq=1, icode="A", alt="A", x=0.0),
        atom_line(2, "CA", seq=1, icode="A", alt="B", x=1.0),
    ]))
    assert st.n_atoms == 1
    assert st.atoms[0].serial == 1


def test_atoms_without_an_alternate_location_column_are_all_kept(pdb_file):
    st = md.parse_pdb(pdb_file([
        atom_line(1, "CA", x=0.0), atom_line(2, "N", x=1.4, element="N"),
    ]))
    assert st.n_atoms == 2


def test_one_model_of_a_multi_model_file_is_one_structure(pdb_file):
    """Concatenating models made a 20-model NMR entry into one 20x bigger molecule."""
    path = pdb_file([
        "MODEL        1", atom_line(1, "CA", x=0.0), atom_line(2, "N", x=1.4, element="N"),
        "TER", "ENDMDL",
        "MODEL        2", atom_line(1, "CA", x=50.0), atom_line(2, "N", x=51.4, element="N"),
        "TER", "ENDMDL",
        "MODEL        3", atom_line(1, "CA", x=90.0), atom_line(2, "N", x=91.4, element="N"),
        "TER", "ENDMDL",
    ], name="multi.pdb")
    st = md.parse_pdb(path)
    assert st.n_atoms == 2, "only the first model is a structure"
    assert st.coordinates[:, 0].tolist() == [0.0, 1.4]
    assert st.num_models == 3


def test_the_multi_model_choice_is_reported(pdb_file, caplog):
    path = pdb_file([
        "MODEL        1", atom_line(1, "CA", x=0.0), "ENDMDL",
        "MODEL        2", atom_line(1, "CA", x=50.0), "ENDMDL",
    ], name="two.pdb")
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(path)
    assert "holds 2 models" in caplog.text
    assert "1 atom(s), 1 record(s) from the later models ignored" in caplog.text
    assert st.num_models == 2


def test_a_single_model_file_claims_one_model(pdb_file, caplog):
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        st = md.parse_pdb(pdb_file([atom_line(1, "CA")]))
    assert st.num_models == 1
    assert "models" not in caplog.text


def test_an_empty_model_block_still_counts(pdb_file):
    st = md.parse_pdb(pdb_file(["MODEL        1", "ENDMDL", "MODEL        2", "ENDMDL"],
                               name="empty.pdb"))
    assert st.n_atoms == 0
    assert st.num_models == 2
