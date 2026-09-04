"""Regression tests for structure.py review fixes."""
import textwrap
import pytest
from biosuite.core import structure as S

pytestmark = pytest.mark.skipif(not S.HAS_BIO, reason="biopython not installed")

# Minimal valid PDB: 12-helix-ish fragment with standard ATOM records
PDB_TEXT = textwrap.dedent("""\
    HEADER    TEST STRUCTURE                            01-JAN-00   0000
    TITLE     MINIMAL TEST
    ATOM      1  N   ALA A   1     -10.000   0.000   0.000  1.00 20.00           N
    ATOM      2  CA  ALA A   1      -9.000   0.500   0.000  1.00 20.00           C
    ATOM      3  C   ALA A   1      -8.200  -0.500   0.800  1.00 20.00           C
    ATOM      4  O   ALA A   1      -8.200  -1.700   0.700  1.00 20.00           O
    ATOM      5  N   GLY A   2      -7.400   0.100   1.700  1.00 20.00           N
    ATOM      6  CA  GLY A   2      -6.400  -0.400   2.500  1.00 20.00           C
    ATOM      7  C   GLY A   2      -5.200   0.200   2.400  1.00 20.00           C
    ATOM      8  O   GLY A   2      -5.000   1.200   3.000  1.00 20.00           O
    ATOM      9  N   ALA A   3      -4.400  -0.300   1.500  1.00 20.00           N
    ATOM     10  CA  ALA A   3      -3.200   0.300   1.400  1.00 20.00           C
    ATOM     11  C   ALA A   3      -2.100  -0.300   2.200  1.00 20.00           C
    ATOM     12  O   ALA A   3      -1.000   0.200   2.200  1.00 20.00           O
    END
    """)


@pytest.fixture
def pdb_file(tmp_path):
    p = tmp_path / 'mini.pdb'
    p.write_text(PDB_TEXT)
    return str(p)


def test_parse_and_counts(pdb_file):
    st, err = S.parse_pdb(pdb_file)
    assert err is None
    info = S.get_structure_info(st, 'mini')
    assert info.num_residues == 3
    assert info.num_chains == 1
    assert info.chains == ['A']


def test_ramachandran_angles_collected(pdb_file):
    st, _ = S.parse_pdb(pdb_file)
    ang = S.compute_ramachandran(st)
    assert set(ang) == {'phi', 'psi'}
    assert len(ang['phi']) >= 1  # internal residues only


def test_secondary_structure_fallback_counts(pdb_file, monkeypatch):
    st, _ = S.parse_pdb(pdb_file)
    monkeypatch.setattr(S, 'check_structure_tools', lambda: {'dssp': False})
    ss = S.compute_secondary_structure(st, filepath=pdb_file)
    assert set(ss) == {'H', 'E', 'C'}
    assert sum(ss.values()) == 3


def test_full_analysis_with_filepath(pdb_file):
    info = S.full_analysis(filepath=pdb_file)
    assert info.num_residues == 3
    assert isinstance(info.secondary_structure, dict)


def test_dssp_needs_file(monkeypatch, pdb_file):
    """DSSP branch must be skipped when no file path is available."""
    st, _ = S.parse_pdb(pdb_file)
    monkeypatch.setattr(S, 'check_structure_tools', lambda: {'dssp': True})
    called = []
    monkeypatch.setattr(S, 'DSSP', lambda m, f: called.append((m, f)))
    S.compute_secondary_structure(st, filepath=None)   # no path -> heuristic
    assert not called
    S.compute_secondary_structure(st, filepath=pdb_file)
    assert called and called[0][1] == pdb_file
