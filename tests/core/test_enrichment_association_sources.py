"""Reading a gene/GO-term association source, and saying which part failed.

`run_ora` had one message for every empty association set — "Could not load gene
associations for {organism}" — even when the caller had passed a file themselves, so a
typo in a path was reported as an organism problem and a space-separated table read as
an empty one.  Both are now named, and a row the reader cannot use is counted.
"""
import logging

import pytest

from biosuite.core import enrichment as en

GOOD = "672\tGO:0006915\n7157\tGO:0006915\n2016\tGO:0007049\n"


def write(tmp_path, text, name="assoc.tsv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _ora_with(monkeypatch, tmp_path, associations_file, records=()):
    class Study:
        def __init__(self, *args, **kwargs):
            pass

        def run_study(self, genes):
            return list(records)

    class Dag(dict):
        def __contains__(self, go_id):
            return True

        def __getitem__(self, go_id):
            import types
            return types.SimpleNamespace(depth=3)

    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "GOEnrichmentStudy", Study, raising=False)
    monkeypatch.setattr(en, "GODag", lambda path: Dag(), raising=False)
    monkeypatch.setattr(en, "_download_go_obo", lambda: "cached/go-basic.obo")
    return en.run_ora(["672"], associations_file=associations_file)


# ── the file reader itself ───────────────────────────────────────────────────

def test_a_two_column_tab_table_is_read_as_gene_to_terms(tmp_path):
    loaded = en._load_associations_file(write(tmp_path, GOOD))

    assert loaded == {"672": {"GO:0006915"}, "7157": {"GO:0006915"},
                      "2016": {"GO:0007049"}}


def test_a_repeated_pair_is_one_term_not_two(tmp_path):
    loaded = en._load_associations_file(write(tmp_path, GOOD + GOOD))

    assert loaded["672"] == {"GO:0006915"}


def test_rows_that_are_not_gene_slash_term_pairs_are_counted_and_said(
        tmp_path, caplog):
    """A space-separated file used to answer an empty dict in total silence."""
    path = write(tmp_path, "672 GO:0006915\n7157 GO:0006915\njunk\n")

    with caplog.at_level(logging.WARNING, logger="biosuite.core.enrichment"):
        loaded = en._load_associations_file(path)

    assert loaded == {}
    assert "skipped 3 row(s) that were not a tab-separated" in caplog.text, caplog.text


def test_a_blank_column_is_a_malformed_row_not_an_empty_identifier(tmp_path):
    loaded = en._load_associations_file(write(tmp_path, "672\t\n\tGO:0006915\n"))

    assert loaded == {}


def test_a_missing_file_is_an_empty_answer_named_by_path(tmp_path, caplog):
    absent = str(tmp_path / "absent.tsv")

    with caplog.at_level(logging.ERROR, logger="biosuite.core.enrichment"):
        assert en._load_associations_file(absent) == {}

    assert absent in caplog.text, caplog.text


# ── what run_ora says about them ─────────────────────────────────────────────

def test_an_unreadable_file_names_the_file_not_the_organism(monkeypatch, tmp_path):
    """Measured before the fix: "Could not load gene associations for human." for a
    path the caller had typed themselves."""
    report = _ora_with(monkeypatch, tmp_path, str(tmp_path / "nope.tsv"))

    assert report.message == (
        f"Could not load gene associations: association file "
        f"{str(tmp_path / 'nope.tsv')!r} yielded no gene/GO-term rows."), report.message
    assert "human" not in report.message


def test_an_organism_with_no_data_available_names_the_organism(monkeypatch, tmp_path):
    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "GODag", lambda path: {}, raising=False)
    monkeypatch.setattr(en, "_download_go_obo", lambda: "cached/go-basic.obo")
    monkeypatch.setattr(en, "_get_associations", lambda organism: {})

    report = en.run_ora(["672"], organism="yeast")

    assert report.message == ("Could not load gene associations: no gene/GO-term "
                              "associations are available for organism 'yeast'."), \
        report.message


def test_the_study_is_given_only_the_genes_the_background_knows(monkeypatch, tmp_path):
    asked = []

    class Study:
        def __init__(self, population, associations, obodag, **kwargs):
            pass

        def run_study(self, genes):
            asked.append(sorted(genes))
            return []

    class Dag(dict):
        def __contains__(self, go_id):
            return False

    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "GOEnrichmentStudy", Study, raising=False)
    monkeypatch.setattr(en, "GODag", lambda path: Dag(), raising=False)
    monkeypatch.setattr(en, "_download_go_obo", lambda: "cached/go-basic.obo")

    report = en.run_ora(["672", "999999"], associations_file=write(tmp_path, GOOD))

    assert asked == [["672"]], asked
    assert report.message == ("1 of 2 input identifier(s) are not in the 3-gene "
                              "background and were left out"), report.message
    assert report.background_count == 3


def test_lowercase_identifiers_are_not_rescued_into_a_false_negative(monkeypatch, tmp_path):
    """Entrez ids are digits, so case cannot save a symbol — but a mixed-case file
    must still match itself, which is what the key space guarantees."""
    report = _ora_with(monkeypatch, tmp_path, write(tmp_path, "672\tGO:0006915\n"))

    assert report.num_significant == 0
    assert report.message == "", report.message
