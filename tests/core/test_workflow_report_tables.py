"""biosuite.core.workflow.report: the table fragment of an HTML report.

A long table used to be cut at ``max_rows`` and presented as if it were the whole
dataset; ``max_rows=0`` produced a header over no rows; and an object without
``to_html`` answered a report question with a pandas ``AttributeError``.
"""
import pandas as pd
import pytest

from biosuite.core.workflow.report import HTMLReport, _df_to_html_table


def frame(rows=100):
    return pd.DataFrame({"a": range(rows), "b": range(rows)})


def rows_of(table):
    """Data rows only: pandas marks its header row with a style attribute."""
    return table.count("\n    <tr>")


def test_a_table_that_fits_is_emitted_without_a_comment():
    table = _df_to_html_table(frame(10))
    assert table.count("<tr") == 11, "header plus ten rows"
    assert rows_of(table) == 10
    assert "Showing" not in table


def test_a_truncated_table_says_how_much_of_it_was_shown():
    table = _df_to_html_table(frame(100), max_rows=50)
    assert rows_of(table) == 50
    assert "Showing 50 of 100 rows" in table
    assert "max_rows=50" in table


def test_the_note_uses_the_limit_the_caller_asked_for():
    assert "Showing 5 of 100 rows" in _df_to_html_table(frame(), max_rows=5)


@pytest.mark.parametrize("max_rows", [0, -1, 1.5, "50", None])
def test_a_row_limit_that_cannot_select_rows_is_refused(max_rows):
    with pytest.raises(ValueError, match="max_rows must be a positive integer"):
        _df_to_html_table(frame(), max_rows=max_rows)


def test_an_empty_frame_and_none_both_say_there_is_no_data():
    assert _df_to_html_table(None) == "<p>No data</p>"
    assert _df_to_html_table(pd.DataFrame()) == "<p>No data</p>"


def test_a_list_is_named_as_the_wrong_kind_of_input():
    """It used to reach pandas and come back as 'list' object has no attribute."""
    with pytest.raises(TypeError, match="expects a pandas DataFrame, got list"):
        _df_to_html_table([1, 2, 3])


def test_a_duck_typed_table_is_still_emitted_as_it_stands():
    class TableOnly:
        def __init__(self):
            self.kwargs = None

        def to_html(self, **kwargs):
            self.kwargs = kwargs
            return "<table>duck</table>"

    table = TableOnly()
    assert _df_to_html_table(table) == "<table>duck</table>"
    assert table.kwargs == {"classes": "data-table", "index": False, "border": 0}


def test_add_table_hands_the_row_limit_through_to_the_fragment(tmp_path):
    report = HTMLReport("T")
    report.add_table(frame(), title="Rows", max_rows=3)
    text = report.to_html()
    assert "Showing 3 of 100 rows" in text
    assert rows_of(text) == 3
