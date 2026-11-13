"""A hand-written 12-atom backbone (N/CA/C/O for three residues) is the smallest
input that exercises every stage of ``structure.py``: the fixed-width parse, the
header fields, one peptide, and the composed report.  Every number below was read
off the running package on this text.  The generated geometries in
``test_structure_pdb_reads_and_binding_sites.py`` cover the multi-chain cases; this
file is the literal-record one, and it is the file that shows what a PDB without a
``REMARK 2`` resolution line produces.
"""
import textwrap

import pytest

from biosuite.core import structure as S

pytestmark = pytest.mark.skipif(not S.HAS_BIO, reason="biopython not installed")

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
    path = tmp_path / 'mini.pdb'
    path.write_text(PDB_TEXT, encoding="utf-8")
    return str(path)


def test_minimal_backbone_parses_with_every_header_field(pdb_file):
    structure, error = S.parse_pdb(pdb_file)
    assert error is None
    info = S.get_structure_info(structure, 'mini')
    assert (info.num_atoms, info.num_residues, info.num_chains) == (12, 3, 1)
    assert info.chains == ['A']
    # HEADER without REMARK 2: Biopython stores None for the resolution, and the
    # float-declared field must not carry it out.
    assert info.resolution == 0.0
    assert info.method == 'unknown'
    assert info.message == ""


def test_ramachandran_keeps_only_the_residue_that_has_both_angles(pdb_file):
    """Residue 1 has no phi and residue 3 no psi, so exactly one pair survives."""
    structure, _ = S.parse_pdb(pdb_file)
    angles = S.compute_ramachandran(structure)
    assert angles['phi'] == pytest.approx([125.0728287826985])
    assert angles['psi'] == pytest.approx([-89.61422325409868])
    assert len(angles['phi']) == len(angles['psi']) == 1


def test_secondary_structure_fallback_counts_every_residue_once(pdb_file, monkeypatch):
    structure, _ = S.parse_pdb(pdb_file)
    monkeypatch.setattr(S, 'check_structure_tools', lambda: {'dssp': False})
    assert S.compute_secondary_structure(structure, filepath=pdb_file) == {
        'H': 0, 'E': 0, 'C': 3}


def test_dssp_is_handed_the_model_and_the_path_and_its_absence_falls_back(pdb_file,
                                                                         monkeypatch):
    """A DSSP stand-in that returns nothing at all still yields the heuristic, and
    the caller is told which of the two answered."""
    structure, _ = S.parse_pdb(pdb_file)
    monkeypatch.setattr(S, 'check_structure_tools', lambda: {'dssp': True})
    called = []
    monkeypatch.setattr(S, 'DSSP', lambda model, path: called.append((model, path)))

    assert S.compute_secondary_structure(structure, filepath=None) == {
        'H': 0, 'E': 0, 'C': 3}
    assert called == [], "no path means DSSP is never constructed"

    with pytest.warns(RuntimeWarning, match="DSSP could not be run on"):
        assert S.compute_secondary_structure(structure, filepath=pdb_file) == {
            'H': 0, 'E': 0, 'C': 3}
    assert len(called) == 1
    assert called[0][0] is structure[0]
    assert called[0][1] == pdb_file


def test_full_analysis_composes_the_minimal_report(pdb_file):
    info = S.full_analysis(filepath=pdb_file)
    assert (info.name, info.num_atoms, info.num_residues) == ('loaded', 12, 3)
    assert info.secondary_structure == {'H': 0, 'E': 0, 'C': 3}
    assert len(info.ramachandran['phi']) == 1
    assert info.surface_area == pytest.approx(339.4487343619999)
    assert info.message == ""
    assert S.format_structure_report(info).splitlines() == [
        "=== Protein Structure Report ===",
        "Name: loaded",
        "Atoms: 12",
        "Residues: 3",
        "Chains: 1 (A)",
        "Method: unknown",
        "Surface area: 339 Å²",
        "Secondary structure: 0% helix, 0% sheet, 100% coil",
        "Ramachandran: 1 angles (phi: 125.1±0.0°)",
    ]
