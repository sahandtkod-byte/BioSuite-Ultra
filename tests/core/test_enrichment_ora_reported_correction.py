"""Which correction an ORA term is reported under, and what happens when it is absent.

`run_ora` takes a `method` argument and then read `p_fdr_bh` first whatever the caller
asked, so `method='bonferroni'` (the default) and `method='fdr_bh'` produced the same
numbers; a term significant under the requested correction was dropped because the
*other* column missed the cutoff.  The report header also promised `FDR < 0.05` even
when the caller had passed `cutoff=0.001`.
"""
import types

import pytest

from biosuite.core import enrichment as en

GENES = ["672", "7157"]
BACKGROUND = ["672", "7157", "2016"]


def _record(**fields):
    return types.SimpleNamespace(**fields)


def _study(monkeypatch, records):
    """Wire a goatools stand-in that hands back exactly *records*."""
    class Study:
        last = None

        def __init__(self, population, associations, obodag, methods=None, cutoff=None):
            Study.last = {"population": list(population), "methods": list(methods),
                          "cutoff": cutoff}

        def run_study(self, study_genes):
            return list(records)

    class Dag(dict):
        def __contains__(self, go_id):
            return True

        def __getitem__(self, go_id):
            return types.SimpleNamespace(depth=3)

    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "GOEnrichmentStudy", Study, raising=False)
    monkeypatch.setattr(en, "GODag", lambda path: Dag(), raising=False)
    monkeypatch.setattr(en, "_download_go_obo", lambda: "cached/go-basic.obo")
    return Study


def _associations(tmp_path):
    path = tmp_path / "assoc.tsv"
    path.write_text("672\tGO:0006915\n7157\tGO:0006915\n2016\tGO:0006915\n",
                    encoding="utf-8")
    return str(path)


def run(monkeypatch, records, tmp_path, **kwargs):
    _study(monkeypatch, records)
    return en.run_ora(GENES, associations_file=_associations(tmp_path), **kwargs)


def test_the_correction_the_caller_named_is_the_number_the_term_carries(tmp_path, monkeypatch):
    records = [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9,
                       p_fdr_bh=0.5, p_bonferroni=1e-6, study_count=2, study_items=["672", "7157"])]

    bonferroni = run(monkeypatch, records, tmp_path, method="bonferroni")
    fdr = run(monkeypatch, records, tmp_path, method="fdr_bh")

    assert bonferroni.num_significant == 1, "Bonferroni says significant, BH says not"
    assert bonferroni.results[0].adjusted_p_value == pytest.approx(1e-6)
    assert fdr.num_significant == 0, "the same term under the other correction"
    assert [r.term_id for r in fdr.results] == []


def test_the_default_is_still_bonferroni_and_says_so(tmp_path, monkeypatch):
    records = [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9,
                       p_fdr_bh=0.5, p_bonferroni=1e-6, study_count=2, study_items=["672", "7157"])]

    report = run(monkeypatch, records, tmp_path)

    assert report.correction == "Bonferroni"
    assert report.threshold == 0.05
    header = en.format_enrichment_report(report).splitlines()[2]
    assert header == "Significant terms (Bonferroni < 0.05): 1", header


def test_the_nominal_p_is_kept_beside_whichever_correction_is_reported(
        tmp_path, monkeypatch):
    records = [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9,
                       p_fdr_bh=2e-9, p_bonferroni=7e-9, study_count=2, study_items=["672", "7157"])]

    for method, expected in (("bonferroni", 7e-9), ("fdr_bh", 2e-9)):
        report = run(monkeypatch, records, tmp_path, method=method)
        assert report.results[0].p_value == pytest.approx(1e-9)
        assert report.results[0].adjusted_p_value == pytest.approx(expected)


def test_a_missing_requested_correction_falls_back_and_is_admitted(tmp_path, monkeypatch):
    """A record without p_bonferroni used to be reported as if nothing had changed."""
    records = [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9, p_fdr_bh=1e-4, study_count=2, study_items=["672", "7157"])]

    report = run(monkeypatch, records, tmp_path, method="bonferroni")

    assert report.num_significant == 1
    assert report.results[0].adjusted_p_value == pytest.approx(1e-4)
    assert report.message == ("1 term(s) carried no p_bonferroni from the engine "
                              "and are reported under a fallback correction"), report.message


def test_a_method_the_engine_can_produce_is_read_from_its_own_column(tmp_path, monkeypatch):
    """goatools computes p_<method>, so a caller-named correction needs no whitelist."""
    records = [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9,
                       p_fdr_bh=0.4, p_bonferroni=0.4, p_sidak=1e-6, study_count=2, study_items=["672", "7157"])]

    report = run(monkeypatch, records, tmp_path, method="sidak")

    assert report.correction == "sidak"
    assert report.results[0].adjusted_p_value == pytest.approx(1e-6)
    assert report.message == "", report.message


def test_the_engine_is_asked_for_the_named_correction_and_both_known_ones(
        tmp_path, monkeypatch):
    Study = _study(monkeypatch, [])
    en.run_ora(GENES, associations_file=_associations(tmp_path), method="sidak")

    assert Study.last["methods"] == ["bonferroni", "fdr_bh", "sidak"], Study.last
    assert Study.last["cutoff"] == 0.05


def test_the_cutoff_decides_instead_of_a_hardcoded_five_percent(tmp_path, monkeypatch):
    records = [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9,
                       p_fdr_bh=0.02, p_bonferroni=0.02, study_count=2, study_items=["672", "7157"])]

    loose = run(monkeypatch, records, tmp_path, method="fdr_bh", cutoff=0.05)
    tight = run(monkeypatch, records, tmp_path, method="fdr_bh", cutoff=0.01)

    assert loose.num_significant == 1 and tight.num_significant == 0
    assert tight.threshold == 0.01
    assert en.format_enrichment_report(tight).splitlines()[2] == \
        "Significant terms (FDR < 0.01): 0"


def test_genes_the_background_does_not_have_are_counted_not_dropped_in_silence(
        tmp_path, monkeypatch):
    _study(monkeypatch, [_record(GO="GO:0006915", name="apoptosis", p_value=1e-9,
                                 p_fdr_bh=1e-4, p_bonferroni=1e-4,
                                 study_count=2, study_items=["672", "7157"])])

    report = en.run_ora(["672", "7157", "999999"],
                        associations_file=_associations(tmp_path))

    assert report.num_input_genes == 3
    assert report.background_count == 3
    assert report.message == ("1 of 3 input identifier(s) are not in the 3-gene "
                              "background and were left out"), report.message


def test_a_run_with_nothing_to_report_about_still_names_its_correction(
        tmp_path, monkeypatch):
    _study(monkeypatch, [])

    report = en.run_ora(GENES, associations_file=_associations(tmp_path))

    assert report.correction == "Bonferroni"
    assert report.threshold == 0.05
    assert report.message == ""
