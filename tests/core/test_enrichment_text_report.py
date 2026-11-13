"""The enrichment text report: what `format_enrichment_report` may claim.

The report is what the CLI prints and what a user reads as the study's result, so
its header line is the only place the cutoff and the correction are stated, and the
trailing count is the only place the reader learns that rows were left out.  Both
lines used to be arithmetic nobody checked: the header said ``FDR < 0.05`` whatever
the caller had passed, and the trailing count subtracted `max_terms` from
`num_significant` while the table had omitted `len(results) - shown`.
"""
import pytest

from biosuite.core import enrichment as en


def _result(name, term_id, p, adjusted, score=0.0, genes=0, kind="ORA"):
    return en.EnrichmentResult(kind, name, term_id, p, adjusted,
                               enrichment_score=score, gene_count=genes)


def test_the_report_says_so_when_there_is_nothing_to_format():
    assert en.format_enrichment_report(None) == "No enrichment results available."


def test_a_note_and_a_table_are_both_printed_when_a_run_had_something_to_say():
    """A note used to short-circuit the table.

    "2 terms were dropped for GO depth" arrives beside a real result list, and the
    old code printed the note and returned, hiding every term the analysis found.
    """
    report = en.EnrichmentReport("ORA", 5, 1, results=[_result("apoptosis", "GO:1", 1e-6, 1e-4)],
                                 message="1 term was dropped for depth")
    text = en.format_enrichment_report(report)

    assert "Note: 1 term was dropped for depth" in text, text
    assert "GO:1" in text and "apoptosis" in text, text
    assert text.index("Note:") < text.index("1    GO:1"), text


def test_a_note_alone_is_still_printed_when_there_is_no_table():
    report = en.EnrichmentReport("ORA", 5, 0,
                                 message="goatools is not usable. Run: pip install goatools")
    text = en.format_enrichment_report(report)

    assert text.endswith("Note: goatools is not usable. Run: pip install goatools"), text
    assert "#" not in text.splitlines()[3], text


def test_a_report_with_no_results_and_no_note_says_nothing_was_enriched():
    report = en.EnrichmentReport("GSEA", 120, 0)
    assert en.format_enrichment_report(report).endswith("No significant enriched terms found.")


def test_the_header_names_the_correction_and_the_cutoff_the_analysis_applied():
    """Measured before the fix: ``cutoff=0.001`` was still announced as ``< 0.05``."""
    tight = en.EnrichmentReport("ORA", 200, 2, threshold=0.001, correction="Bonferroni")
    loose = en.EnrichmentReport("GSEA", 200, 2, threshold=0.05, correction="FDR")

    assert en.format_enrichment_report(tight).splitlines()[2] == \
        "Significant terms (Bonferroni < 0.001): 2"
    assert en.format_enrichment_report(loose).splitlines()[2] == \
        "Significant terms (FDR < 0.05): 2"


def test_a_hand_built_report_falls_back_to_a_neutral_correction_name():
    text = en.format_enrichment_report(en.EnrichmentReport("ORA", 5, 0))
    assert text.splitlines()[2] == "Significant terms (adjusted p < 0.05): 0", text


def test_the_report_prints_one_aligned_row_per_term_with_the_term_id_visible():
    results = [_result("apoptosis", "GO:0006915", 1e-6, 0.001, score=1.85, genes=4),
               _result("ribosome biogenesis", "GO:0042252", 0.02, 0.04, score=0.0, genes=7)]
    report = en.EnrichmentReport("ORA", 200, 2, results, correction="FDR")
    lines = en.format_enrichment_report(report).splitlines()

    assert lines[0] == "=== ORA Enrichment Analysis ==="
    assert lines[1] == "Input genes: 200"
    assert lines[2] == "Significant terms (FDR < 0.05): 2"
    assert lines[3] == ""
    assert lines[4].replace(" ", "") == "#IDTermAdjpNESGenes", lines[4]
    assert "Adj p" in lines[4] and "NES" in lines[4] and "Genes" in lines[4], lines[4]
    assert lines[5] == "-" * 88
    assert lines[6].split() == ["1", "GO:0006915", "apoptosis", "1.00e-03", "N/A", "4"]
    assert lines[7].split() == ["2", "GO:0042252", "ribosome", "biogenesis",
                                "4.00e-02", "N/A", "7"]


def test_the_nes_column_is_only_filled_for_the_analysis_that_measures_it():
    """A normalized enrichment score is a GSEA statistic.

    The column used to be printed whenever the number was not 0.0, which made an
    ORA term carrying a stray score look measured and turned a real GSEA score of
    exactly 0.0 into "N/A".
    """
    gsea = en.EnrichmentReport("GSEA", 20, 2,
                               [_result("zero nes", "SET0", 1e-6, 1e-4, score=0.0, kind="GSEA"),
                                _result("missing nes", "SET1", 1e-6, 1e-4,
                                        score=float('nan'), kind="GSEA")],
                               correction="FDR")
    lines = en.format_enrichment_report(gsea).splitlines()

    assert lines[6].split()[-2] == "0.00", lines[6]
    assert lines[7].split()[-2] == "n/a", lines[7]
    assert "nan" not in "\n".join(lines), "\n".join(lines)


def test_a_missing_adjusted_p_is_printed_as_a_blank_not_as_nan():
    report = en.EnrichmentReport("KEGG", 3, 0,
                                 [_result("no number", "hsa00010", float('nan'),
                                          float('nan'), kind="KEGG")],
                                 correction="adjusted p")
    row = en.format_enrichment_report(report).splitlines()[6]

    assert "n/a" in row, row
    assert "nan" not in row, row


def test_the_report_uses_the_term_id_when_the_name_is_blank():
    report = en.EnrichmentReport("KEGG", 3, 1, [_result("", "hsa00010", 0.001, 0.01,
                                                        kind="KEGG")])
    row = en.format_enrichment_report(report).splitlines()[6]

    assert row.split()[:3] == ["1", "hsa00010", "hsa00010"], row


def test_the_trailing_count_is_the_number_of_rows_the_table_left_out():
    results = [_result(f"pathway {i}", f"hsa{i}", 0.001, 0.01, kind="KEGG")
               for i in range(5)]
    text = en.format_enrichment_report(
        en.EnrichmentReport("KEGG", 50, 5, results), max_terms=2)

    assert text.count("pathway") == 2
    assert text.endswith("\n... and 3 more term(s) not shown."), text


def test_the_trailing_count_does_not_use_the_significant_term_population():
    """GSEA lists every term the engine returned and counts the significant ones separately.

    Measured on the old arithmetic: 200 rows with 5 shown and 3 significant printed
    no omission note at all, and with 100 significant it claimed "95 more" — the
    table had left out 195 both times.
    """
    results = [_result(f"set {i}", f"T{i}", 1e-3, 1e-2 * i, kind="GSEA")
               for i in range(200)]
    few_sig = en.EnrichmentReport("GSEA", 500, 3, results, correction="FDR")
    many_sig = en.EnrichmentReport("GSEA", 500, 100, results, correction="FDR")

    for report in (few_sig, many_sig):
        text = en.format_enrichment_report(report, max_terms=5)
        assert text.count("set ") == 5, text
        assert text.endswith("\n... and 195 more term(s) not shown."), text


def test_no_truncation_note_when_every_term_fit():
    results = [_result(f"pathway {i}", f"hsa{i}", 0.001, 0.01, kind="KEGG")
               for i in range(3)]
    text = en.format_enrichment_report(en.EnrichmentReport("KEGG", 50, 3, results),
                                       max_terms=3)

    assert "more term(s)" not in text
    assert text.count("pathway") == 3


def test_max_terms_zero_prints_the_header_and_no_rows():
    results = [_result(f"pathway {i}", f"hsa{i}", 0.001, 0.01, kind="KEGG")
               for i in range(3)]
    text = en.format_enrichment_report(en.EnrichmentReport("KEGG", 50, 3, results),
                                       max_terms=0)

    assert text.count("pathway") == 0
    assert "... and 3 more term(s) not shown." in text, text


@pytest.mark.parametrize("max_terms", [-1, -20, 2.5, "10", True, None])
def test_an_unusable_max_terms_is_refused(max_terms):
    """``max_terms=-1`` printed four rows and announced "and 31 more terms"."""
    report = en.EnrichmentReport("KEGG", 5, 5,
                                 [_result(f"p{i}", f"h{i}", 0.001, 0.01, kind="KEGG")
                                  for i in range(5)])
    with pytest.raises(ValueError, match="max_terms"):
        en.format_enrichment_report(report, max_terms=max_terms)


def test_the_report_reads_the_limit_through_the_same_guard_as_the_report_container():
    results = [_result(f"p{i}", f"h{i}", 0.001, 0.01, kind="KEGG") for i in range(4)]
    report = en.EnrichmentReport("KEGG", 4, 4, results)

    with pytest.raises(ValueError, match="top_terms"):
        report.top_terms(-1)
    with pytest.raises(ValueError, match="top_terms"):
        report.top_terms(1.5)
    assert len(report.top_terms(2)) == 2
    assert len(report.top_terms(None)) == 4
    assert len(report.top_terms(99)) == 4
