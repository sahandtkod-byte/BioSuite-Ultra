"""Regression tests for structure_prediction.py review fixes."""
import pytest

from biosuite.core import structure_prediction as S


def _atom(chain, resseq, plddt, atom="CA"):
    # columns: chain=22, resSeq=23-26, B-factor=61-66 (1-based)
    return (f"ATOM  {1:5d} {atom:<4} {'GLY':>3} {chain}{resseq:4d}    "
            f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.00:6.2f}{plddt:6.2f}"
            f"          {'C':>2}")


def test_plddt_per_residue_not_per_atom():
    pdb = "\n".join([
        _atom('A', 1, 92.0, "N"), _atom('A', 1, 92.0, "CA"), _atom('A', 1, 92.0, "C"),
        _atom('A', 2, 85.5, "N"), _atom('A', 2, 85.5, "CA"),
        _atom('A', 3, 40.0, "CA"),
    ])
    scores = S._extract_plddt(pdb)
    assert len(scores) == 3                     # NOT 6 (old per-atom bug)
    assert scores == [92.0, 85.5, 40.0]


def test_plddt_multi_chain():
    pdb = "\n".join([
        _atom('A', 1, 90.0), _atom('B', 1, 80.0), _atom('A', 2, 70.0),
    ])
    scores = S._extract_plddt(pdb)
    assert len(scores) == 3  # (A,1) and (B,1) are distinct residues


def test_report_counts_match_residues():
    pdb = "\n".join([_atom('A', i, 95.0) for i in range(1, 11)])
    scores = S._extract_plddt(pdb)
    result = S.PredictionResult(engine='x', pdb_string=pdb, plddt_scores=scores,
                                num_residues=len(scores), confidence=95.0)
    text = S.format_prediction_report(result)
    assert "Confident (>90): 10" in text


def test_no_input_returns_clear_message():
    res = S.predict_structure()
    assert res.engine == 'none'
    assert 'No sequence' in res.message


def test_failed_fetch_surfaces_engine_message(monkeypatch):
    monkeypatch.setattr(S, '_alphafold_fetch',
                        lambda *a, **k: S.PredictionResult(
                            engine='alphafold', message='AlphaFold API error: 404'))
    res = S.predict_structure(uniprot_id='BADID')
    assert 'AlphaFold API error' in res.message  # old code said "No sequence ..."
