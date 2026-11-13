"""`format_go_results`: the table the CLI and the GUI print for a GO answer.

The definition column is the only part of a term a user actually reads, so a row
that truncates it silently or a header that lies about the columns is a wrong
answer even when the search behind it was right.
"""
import pytest

from biosuite.core.go_browser import GOBrowser, GOTerm, format_go_results


def _row(term):
    return format_go_results([term]).splitlines()[2]


def test_no_terms_is_said_as_no_terms_rather_than_a_header_only():
    assert format_go_results([]) == "No GO terms found."


def test_the_header_names_the_four_columns_the_rows_carry():
    header = format_go_results([GOTerm("GO:1", "x", "BP")]).splitlines()[0]
    assert header.split() == ["GO", "ID", "Name", "Namespace", "Definition"]
    assert format_go_results([GOTerm("GO:1", "x", "BP")]).splitlines()[1] == "-" * 100


def test_a_short_row_keeps_its_id_name_namespace_and_definition():
    row = _row(GOTerm("GO:0006915", "apoptotic process", "BP", definition="Programmed "
                                                                        "cell death."))
    assert row.startswith("GO:0006915")
    assert "apoptotic process" in row and "BP" in row
    assert row.endswith("Programmed cell death.")


def test_a_definition_longer_than_forty_characters_is_cut_with_a_visible_mark():
    long_definition = "A" * 55
    row = _row(GOTerm("GO:1", "term", "MF", definition=long_definition))
    assert "A" * 40 + "..." in row
    assert "A" * 41 not in row


def test_a_definition_of_exactly_the_limit_is_left_alone():
    exact = "B" * 40
    row = _row(GOTerm("GO:1", "term", "CC", definition=exact))
    assert row.endswith(exact) and "..." not in row


def test_a_term_without_a_definition_leaves_the_column_empty_not_padded_with_noise():
    row = _row(GOTerm("GO:1", "bare", "BP"))
    assert row.rstrip() == row.rstrip() and "None" not in row
    assert row.split() == ["GO:1", "bare", "BP"]


def test_the_count_line_says_how_many_terms_are_above_it():
    terms = [GOTerm(f"GO:{i}", f"term {i}", "BP") for i in range(3)]
    lines = format_go_results(terms).splitlines()
    assert lines[-1] == "Found 3 terms"
    assert lines[-2] == ""
    assert len([line for line in lines[2:-2]]) == 3


def test_every_row_of_a_real_search_is_well_formed():
    hits = GOBrowser().search("kinase")
    assert len(hits) == 2
    lines = format_go_results(hits).splitlines()
    assert [line.split()[0] for line in lines[2:4]] == ["GO:0004672", "GO:0004674"]
    assert lines[-1] == "Found 2 terms"


def test_a_long_name_is_not_truncated_because_it_is_the_thing_being_read():
    """The width budget is spent on the definition, not on the term's name."""
    name = "very long pathway name that goes on and on"
    row = _row(GOTerm("GO:1", name, "BP", definition="short"))
    assert name in row


def test_the_namespace_column_holds_what_the_record_carries():
    for namespace in ("BP", "MF", "CC"):
        assert _row(GOTerm("GO:1", "x", namespace)).split()[2] == namespace


@pytest.mark.parametrize("term", [
    GOTerm("GO:2", "", "BP", definition="no name at all"),
    GOTerm("", "unnamed id", "BP"),
])
def test_missing_pieces_of_a_record_do_not_break_the_table(term):
    lines = format_go_results([term]).splitlines()
    assert lines[-1] == "Found 1 terms"
    assert lines[2].startswith(term.go_id)
