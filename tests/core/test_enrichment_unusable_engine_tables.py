"""A gseapy result table the module cannot read is an error, not an empty analysis.

GSEA and KEGG turn `res2d` rows into `EnrichmentResult` objects by name.  When a column
was missing or held text instead of a number, the old code either died with a numpy
complaint that was reported as the study's result, or invented one: a missing p-value
became 1.0 and a missing gene list became a set containing the empty string.
"""
import logging
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.core import enrichment as en

GENES = ["672", "7157", "2016"]
SCORES = [2.0, 1.0, 0.0]


class FakeGseapy:
    def __init__(self, prerank_result=None, enrich_result=None):
        self.prerank_result = prerank_result
        self.enrich_result = enrich_result
        self.calls = []

    def prerank(self, **kwargs):
        self.calls.append(("prerank", kwargs))
        return self.prerank_result

    def enrich(self, **kwargs):
        self.calls.append(("enrich", kwargs))
        return self.enrich_result


def _gsea(monkeypatch, frame):
    engine = FakeGseapy(prerank_result=types.SimpleNamespace(res2d=frame))
    monkeypatch.setattr(en, "HAS_GSEAPY", True)
    monkeypatch.setattr(en, "gp", engine, raising=False)
    return engine, en.run_gsea(GENES, gene_scores=SCORES)


def _kegg(monkeypatch, frame):
    engine = FakeGseapy(enrich_result=types.SimpleNamespace(res2d=frame))
    monkeypatch.setattr(en, "HAS_GSEAPY", True)
    monkeypatch.setattr(en, "gp", engine, raising=False)
    return engine, en.run_kegg_enrichment(GENES)


# ── a table without the columns the reader needs ─────────────────────────────

def test_gsea_refuses_a_table_with_no_term_column(monkeypatch):
    """Measured before the fix: two rows of `term_id=''`, `p_value=1.0` and no message."""
    _, report = _gsea(monkeypatch, pd.DataFrame([{"id": "x"}, {"id": "y"}]))

    assert report.results == []
    assert report.num_significant == 0
    assert report.message.startswith("GSEA returned a result table without a 'Term'"), \
        report.message
    assert "it carries id." in report.message, report.message


def test_gsea_refuses_a_term_table_that_carries_no_columns_at_all(monkeypatch):
    _, report = _gsea(monkeypatch, pd.DataFrame())

    assert report.message.endswith("carries no columns."), report.message


def test_kegg_refuses_a_table_with_no_p_value_column(monkeypatch):
    _, report = _kegg(monkeypatch, pd.DataFrame({"Term": ["hsa00010"]}))

    assert report.results == []
    assert report.message == ("KEGG enrichment returned a table without a 'Term' (or "
                              "'Description') and 'P-value' column; it carries Term."), \
        report.message


def test_kegg_accepts_the_description_column_as_the_name_it_documents(monkeypatch):
    _, report = _kegg(monkeypatch, pd.DataFrame({
        "Description": ["KEGG_PATHWAY hsa00010"], "P-value": [0.01],
        "Adjusted P-value": [0.04], "Overlap": ["3/12"]}))

    assert report.results[0].term_name == "KEGG_PATHWAY hsa00010"
    assert report.results[0].term_id == ""
    assert report.message == "", report.message


# ── cells that hold text where a number is read ──────────────────────────────

def test_a_string_typed_p_column_no_longer_aborts_the_whole_run(monkeypatch):
    """Measured before the fix: "GSEA error: ufunc 'isnan' not supported for the
    input types…" for a table whose numbers arrived as text."""
    _, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA", "SetB"], "NES": ["1.85", "-1.42"],
        "NOM p-val": ["0.001", "0.4"], "FDR q-val": ["0.02", "0.4"],
        "Genes": ["G1;G2", "G3"]}))

    assert not report.message.startswith("GSEA error"), report.message
    assert [r.term_name for r in report.results] == ["SetA", "SetB"]
    assert [r.adjusted_p_value for r in report.results] == pytest.approx([0.02, 0.4])
    assert report.results[0].enrichment_score == pytest.approx(1.85)
    assert report.num_significant == 1


def test_a_number_that_is_not_a_number_at_all_is_reported_as_missing(monkeypatch, caplog):
    _, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA"], "NES": ["n/a"], "NOM p-val": ["0.001"],
        "FDR q-val": ["0.02"], "Genes": ["G1"]}))

    import math
    assert math.isnan(report.results[0].enrichment_score)
    assert report.results[0].adjusted_p_value == pytest.approx(0.02)
    assert "SetA: NES is not a number" in caplog.text, caplog.text


def test_a_missing_p_value_stays_missing_instead_of_becoming_one(monkeypatch):
    """`np.nan` used to be turned into 1.0, a number that says "measured, not
    significant" about something the engine never measured."""
    _, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA", "SetB"], "NES": [1.85, -1.42],
        "NOM p-val": [0.001, np.nan], "FDR q-val": [0.02, 0.40],
        "Genes": ["G1;G2", "G3"]}))

    a, b = report.results
    assert a.p_value == pytest.approx(0.001)
    assert b.p_value != b.p_value, "NaN, not 1.0"
    assert "1 term(s) had no 'NOM p-val'" in report.message, report.message


def test_a_missing_fdr_is_counted_and_kept_out_of_the_significant_total(monkeypatch):
    _, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA", "SetB"], "NES": [1.85, -1.42],
        "NOM p-val": [0.001, 0.4], "FDR q-val": [0.02, np.nan],
        "Genes": ["G1;G2", "G3"]}))

    assert report.num_significant == 1
    assert "1 no 'FDR q-val'" in report.message, report.message
    assert [r.term_name for r in report.results] == ["SetA", "SetB"], "missing sorts last"


def test_a_kegg_row_with_a_nan_p_is_not_silently_read_as_not_measured_number(monkeypatch):
    _, report = _kegg(monkeypatch, pd.DataFrame({
        "Term": ["hsa00010", "hsa04110"], "P-value": [0.01, np.nan],
        "Adjusted P-value": [0.04, np.nan], "Overlap": ["3/12", "5/20"]}))

    assert report.num_significant == 1
    assert report.message == ("1 pathway(s) carried no adjusted p-value from the "
                              "engine"), report.message


# ── the gene list of a set ───────────────────────────────────────────────────

def test_a_set_with_no_genes_column_holds_no_genes(monkeypatch):
    """`''.split(';')` is `['']`, which used to be a membership list of one."""
    _, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA"], "NES": [1.85], "NOM p-val": [0.001], "FDR q-val": [0.02]}))

    assert report.results[0].genes == []
    assert report.results[0].gene_count == 0
    assert report.message.endswith("carried no member genes "
                                   "(the engine's 'Genes' cell was empty)"), report.message


def test_gene_names_are_trimmed_and_blank_entries_dropped(monkeypatch):
    _, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA"], "NES": [1.85], "NOM p-val": [0.001], "FDR q-val": [0.02],
        "Genes": ["G1; G2;; G3 "]}))

    assert report.results[0].genes == ["G1", "G2", "G3"]
    assert report.results[0].gene_count == 3


def test_kegg_reads_its_own_genes_column_when_the_engine_provides_one(monkeypatch):
    _, report = _kegg(monkeypatch, pd.DataFrame({
        "Term": ["hsa00010"], "P-value": [0.01], "Adjusted P-value": [0.04],
        "Overlap": ["3/12"], "Genes": ["672;7157;2016"]}))

    assert report.results[0].genes == ["672", "7157", "2016"]
    assert report.results[0].gene_count == 3, "the overlap cell still wins the count"


# ── the engine is still told what the caller asked ───────────────────────────

def test_gsea_forwards_the_validated_limits_and_the_ranking(monkeypatch):
    engine, report = _gsea(monkeypatch, pd.DataFrame({
        "Term": ["SetA"], "NES": [1.85], "NOM p-val": [0.001], "FDR q-val": [0.02],
        "Genes": ["672"]}))
    kwargs = engine.calls[0][1]

    assert kwargs["min_size"] == 15 and kwargs["max_size"] == 500
    assert kwargs["permutation_num"] == 1000 and kwargs["seed"] == 42
    assert list(kwargs["rnk"].index) == GENES[:3]
    assert report.engine == "gseapy"


def test_an_engine_exception_is_logged_with_its_traceback(monkeypatch, caplog):
    class Boom:
        def prerank(self, **kwargs):
            raise ValueError("gmt file unreadable")

        def enrich(self, **kwargs):
            raise ValueError("gmt file unreadable")

    monkeypatch.setattr(en, "HAS_GSEAPY", True)
    monkeypatch.setattr(en, "gp", Boom(), raising=False)

    with caplog.at_level(logging.ERROR, logger="biosuite.core.enrichment"):
        report = en.run_gsea(GENES, gene_scores=SCORES)

    assert report.message == "GSEA error: gmt file unreadable", report.message
    assert "Traceback" in caplog.text, caplog.text
    assert "GSEA failed" in caplog.text, caplog.text
