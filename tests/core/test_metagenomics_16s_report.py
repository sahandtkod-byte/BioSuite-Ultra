"""`format_16s_report()` — the text the 16S pipeline actually shows a user.

Two silences lived here: the result's own message (including "File not found")
was never printed, and reads that the scorer could not judge were counted under
the same "low confidence" heading as reads it had measured as weak.
"""
import pandas as pd
import pytest

from biosuite.core import metagenomics as mg

ECOLI = mg.SILVA_16S_DB["Escherichia coli"]


def test_the_header_counts_the_sequences_it_was_given():
    res = mg.classify_16s_rna([("q1", ECOLI), ("q2", ECOLI)])
    lines = mg.format_16s_report(res).splitlines()
    assert lines[0] == "=== 16S rRNA Classification Report ==="
    assert lines[1] == "Engine: 16S"
    assert lines[2] == "Sequences read: 2"


def test_the_composition_block_draws_a_bar_per_two_percent():
    res = mg.TaxonomyResult(
        engine="16S",
        classifications=[{"name": "a", "taxonomy": "E. coli", "status": "classified"},
                         {"name": "b", "taxonomy": "Unclassified",
                          "status": "low_confidence"}],
        abundance_table=pd.DataFrame({"taxon": ["E. coli", "Unclassified"],
                                      "count": [1, 1],
                                      "relative_abundance": [50.0, 50.0]}))
    text = mg.format_16s_report(res)
    assert "\nTaxonomic Composition:" in text
    assert "50.0% " + "#" * 25 in text


def test_a_full_abundance_is_fifty_bars():
    res = mg.classify_16s_rna([("q", ECOLI)])
    text = mg.format_16s_report(res)
    assert "100.0%" in text and "#" * 50 in text


def test_low_confidence_reads_are_counted_in_the_warning():
    res = mg.classify_16s_rna([("q1", ECOLI), ("q2", "N" * len(ECOLI))])
    text = mg.format_16s_report(res)
    assert "Warning: 1 sequences not confident enough to call" in text


def test_too_short_reads_are_counted_and_explained():
    """A 4-base read is not 'low confidence because it scored 30 %'; it is
    unjudgeable by a 20-base window, and the report now says which."""
    res = mg.classify_16s_rna([("q", ECOLI[:4])])
    text = mg.format_16s_report(res)
    assert "Warning: 1 sequences not confident enough to call" in text
    assert "shorter than the 20-base scoring window" in text
    assert "prefix comparison" in text


def test_everything_confident_keeps_the_report_quiet():
    res = mg.classify_16s_rna([("q", ECOLI)])
    text = mg.format_16s_report(res)
    assert "not confident" not in text
    assert "too short" not in text


def test_the_message_is_printed_as_a_note(tmp_path):
    missing = str(tmp_path / "absent.fa")
    res = mg.classify_16s_rna(missing)
    text = mg.format_16s_report(res)
    assert f"Note: File not found: {missing}" in text
    assert "Sequences read: 0" in text


def test_a_classified_run_reports_its_count_as_a_note():
    res = mg.classify_16s_rna([("q", ECOLI)])
    assert "Note: Classified 1 16S sequences" in mg.format_16s_report(res)


def test_nothing_to_show_prints_only_the_three_header_lines():
    """Both optional blocks are conditional; with no reads the report must not
    invent a composition table."""
    res = mg.TaxonomyResult(engine="16S")
    assert mg.format_16s_report(res) == (
        "=== 16S rRNA Classification Report ===\nEngine: 16S\nSequences read: 0")


def test_the_taxon_column_is_printed_left_aligned_to_thirty_columns():
    res = mg.TaxonomyResult(
        engine="16S",
        classifications=[{"name": "a", "taxonomy": "Longus bacterium",
                          "status": "classified"}],
        abundance_table=pd.DataFrame({"taxon": ["Longus bacterium"], "count": [1],
                                      "relative_abundance": [100.0]}))
    line = [l for l in mg.format_16s_report(res).splitlines()
            if l.startswith("  Longus")][0]
    assert line == f"  {'Longus bacterium':<30} {100.0:5.1f}% " + "#" * 50
