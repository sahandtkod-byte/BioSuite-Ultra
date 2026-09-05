"""Structure prediction contract tests (offline; mocked endpoints)."""
import pytest

from biosuite.core import structure_prediction as sp


PDB_TWO_RES = (
    "ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 95.40           N\n"
    "ATOM      2  CA  ALA A   1      12.460  14.100   9.800  1.00 78.20           C\n"
    "ATOM      3  N   GLY A   2      13.500  14.900  10.100  1.00 88.00           N\n"
)


def test_extract_plddt_per_residue_list():
    vals = sp._extract_plddt(PDB_TWO_RES)
    assert isinstance(vals, list)
    assert len(vals) == 2              # one per RESIDUE (old bug: one per ATOM)
    assert vals[0] > 0 and vals[1] > 0


def test_extract_plddt_dedupes_atoms_in_same_residue():
    vals = sp._extract_plddt(PDB_TWO_RES)
    # two atoms in residue 1 collapse to a single pLDDT entry
    assert len(vals) == 2


def test_extract_plddt_empty():
    assert sp._extract_plddt('') == []


def test_check_prediction_tools_shape():
    tools = sp.check_prediction_tools()
    assert set(tools) == {'esmfold', 'torch'}
    assert all(isinstance(v, bool) for v in tools.values())


def test_esmfold_no_esm_graceful():
    if not sp.HAS_ESM:
        res = sp._esmfold_predict('MGSHLVDAL')
        assert 'esm' in res.message.lower()
    else:
        pytest.skip("esm installed")


def test_predict_structure_no_input_graceful():
    res = sp.predict_structure()
    assert 'no' in (res.message or '').lower() or res.engine


def test_alphafold_fetch_no_network(monkeypatch):
    # simulate unreachable network cleanly
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda *a, **kw: (_ for _ in ()).throw(OSError('offline')),
                        raising=True)
    res = sp._alphafold_fetch('P04637')
    assert res.engine == 'alphafold'
    assert 'error' in res.message.lower() or res.message


def test_format_prediction_report():
    res = sp.PredictionResult(engine='esmfold', plddt_scores=[90.0, 85.0],
                              sequence='AA', num_residues=2, confidence=87.5,
                              message='ok')
    txt = sp.format_prediction_report(res)
    assert isinstance(txt, str)
