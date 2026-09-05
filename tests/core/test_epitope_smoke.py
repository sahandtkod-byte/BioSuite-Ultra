"""API-stability smoke tests for epitope.py (reviewed: heuristics only, no
bugs found — these pin the matrix-driven API against future regressions)."""
import pytest

from biosuite.core.epitope import (
    predict_t_cell_epitopes, predict_b_cell_epitopes, predict_linear_epitopes,
    cleavage_site_prediction, iedb_to_table, format_epitope_report,
)

SEQ = "MAVPKRRTSRKEFVGIPSRLMNHQFSDLLSLFMESSGSTSPSINSTQIPSRKNHTWDEQQRGQIR" * 2


def test_t_cell_results_wellformed():
    res = predict_t_cell_epitopes(SEQ)
    assert 0 < len(res) <= 50
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)
    for r in res:
        assert SEQ[r.start:r.end] == r.peptide
        assert 0 <= r.score <= 1.0


def test_t_cell_anchor_prefers_hydrophobic_p9():
    res = predict_t_cell_epitopes(SEQ)
    top = res[0]
    assert top.peptide[-1] in set('VILMFW')  # HLA-A*02:01 C-term anchor


def test_b_cell_scores_bounded():
    res = predict_b_cell_epitopes(SEQ)
    assert all(0 <= r.score for r in res)


def test_linear_and_cleavage_smoke():
    lin = predict_linear_epitopes(SEQ)
    assert all(len(r.peptide) in (10, 12, 15) for r in lin)
    cl = cleavage_site_prediction("MAVPKRRTSRKE")
    assert all('position' in c and 'preference' in c for c in cl)


def test_reports_format():
    t = predict_t_cell_epitopes(SEQ)[:5]
    b = predict_b_cell_epitopes(SEQ)[:5]
    txt = format_epitope_report(t, b)
    assert "T-cell" in txt and "B-cell" in txt
    tbl = iedb_to_table(t)
    assert tbl.count("Peptide") == 1
