"""The two text views of an epitope prediction: the IEDB table and the report.

Both are printed straight to a terminal by the CLI, so a count that disagrees with
the rows underneath it is a wrong answer rather than a layout problem.
"""
from biosuite.core import epitope as ep

UPPER = "MAVPKRRTSRKEFVGIPSRLMNHQFSDLLSLFMESSGSTSPSINSTQIPSRKNHTWDEQQRGQIR"

def results(n, peptide="GLPLSLCTF", start=4, score=0.8):
    return [ep.EpitopeResult(peptide, start, start + len(peptide),
                             score - 0.01 * i, "T-cell", "MHC-A0201-binding")
            for i in range(n)]


def test_the_table_names_its_columns_once_and_counts_the_rows_it_printed():
    table = ep.iedb_to_table(results(3))
    lines = table.splitlines()

    assert lines[0].split() == ['Peptide', 'Type', 'Start', 'End', 'Score', 'Method'], lines[0]
    assert lines[1] == "-" * 80, lines[1]
    assert table.count("Peptide") == 1, table
    assert lines[2].startswith("GLPLSLCTF") and "T-cell" in lines[2], lines[2]
    assert lines[-1] == "Total epitopes: 3", lines


def test_the_score_is_printed_to_three_places():
    table = ep.iedb_to_table([ep.EpitopeResult("AAAA", 0, 4, 0.123456)])

    assert "0.123" in table, table


def test_an_empty_prediction_still_prints_its_header_and_a_zero_count():
    table = ep.iedb_to_table([])
    lines = table.splitlines()

    assert "Peptide" in table, table
    assert lines[-1] == "Total epitopes: 0", lines
    assert len(lines) == 4, lines


def test_a_long_peptide_is_not_truncated_in_the_table():
    long = "A" * 34
    table = ep.iedb_to_table([ep.EpitopeResult(long, 0, 34, 0.5)])

    assert long in table, table


def test_the_report_counts_both_cell_types_and_lists_at_most_five_of_each():
    t_cell, b_cell = results(9), results(7, peptide="AAA", score=0.6)

    report = ep.format_epitope_report(t_cell, b_cell, "input protein")
    lines = report.splitlines()

    assert lines[0] == "Epitope Prediction Report — input protein", lines[0]
    assert lines[1] == "=" * 50, lines[1]
    assert "T-cell epitopes predicted: 9" in report, report
    assert "B-cell epitopes predicted: 7" in report, report
    assert sum(1 for line in lines if line.startswith("  GLPLSLCTF")) == 5, report
    assert sum(1 for line in lines if line.startswith("  AAA ")) == 5, report


def test_the_report_gives_each_listed_position_as_the_window_it_came_from():
    entry = ep.EpitopeResult("GLPLSLCTF", 4, 13, 0.81234)

    assert "pos=4-13" in ep.format_epitope_report([entry], []), entry


def test_a_prediction_with_nothing_in_it_says_zero_rather_than_printing_nothing():
    report = ep.format_epitope_report([], [], "empty")

    assert "T-cell epitopes predicted: 0" in report, report
    assert "B-cell epitopes predicted: 0" in report, report
    assert "Top 5 T-cell epitopes:" in report and "Top 5 B-cell epitopes:" in report, report
    assert not any(line.startswith("  ") and "score=" in line for line in
                   report.splitlines()), report


def test_the_report_names_the_sequence_it_was_given():
    assert ep.format_epitope_report([], [], "SARS-CoV-2 S").splitlines()[0].endswith(
        "SARS-CoV-2 S")


def test_the_defaults_name_the_protein_and_the_method_that_produced_the_hit():
    entry = ep.predict_t_cell_epitopes(UPPER)[0]
    report = ep.format_epitope_report([entry], [])

    assert report.startswith("Epitope Prediction Report — protein"), report
    assert f"{entry.peptide} (score={entry.score:.3f}" in report, report
