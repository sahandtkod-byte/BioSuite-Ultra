"""The dashboard's statistics panel says what it counted and what it could not.

Two sentences of that panel used to be quietly misleading.  The counts were A, T,
G and C only, so a record that was a third N showed four balanced percentages and
never mentioned the third.  And "ORFs: 45" was the count *after* the scanner's
10-aa floor — the same sequence read by another tool with a lower floor returns a
different number, and nothing on the figure explained the difference.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

#: 100 bases of perfect ACGT balance plus 50 ambiguous ones.
PADDED = "ATGC" * 25 + "N" * 50


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _stats(fig):
    return [t.get_text() for t in fig.axes[0].texts][1]


def test_bases_outside_acgt_are_counted_and_named():
    fig = sv.create_sequence_overview(PADDED)

    text = _stats(fig)
    assert "other:  50  (33.3%)  [N]" in text, text
    assert "A:  25  (16.7%)" in text, text
    assert "GC: 33.3%" in text, "the N's dilute GC and the panel says so"


def test_a_record_with_nothing_to_report_does_not_invent_the_line():
    """The positive control: a clean ACGT sequence gets no 'other' row."""
    fig = sv.create_sequence_overview("ATGC" * 30)

    text = _stats(fig)
    assert "other" not in text, text
    assert "A:  30  (25.0%)" in text and "C:  30  (25.0%)" in text, text


def test_the_orf_count_carries_the_minimum_that_produced_it():
    fig = sv.create_sequence_overview(PADDED)

    line = [ln for ln in _stats(fig).splitlines() if ln.startswith("ORFs")][0]
    assert line == "ORFs: 6 (10 aa or more), 6 not stop-terminated", line


def test_the_minimum_is_the_scanners_own_default():
    """The number on the panel and the number in the code are one number."""
    fig = sv.create_sequence_overview(PADDED)

    line = [ln for ln in _stats(fig).splitlines() if ln.startswith("ORFs")][0]
    assert f"({sv._MIN_ORF_AA} aa or more)" in line, line
    dna = sv._dna(PADDED)
    assert len(sv._find_orfs_forward(dna)) + len(sv._find_orfs_reverse(dna)) == 6


def test_a_two_letter_record_still_gets_its_length_and_an_honest_orf_row():
    fig = sv.create_sequence_overview("AC")

    text = _stats(fig)
    assert text.startswith("Length:  2 bp\n"), text
    assert "ORFs: 0 (10 aa or more)" in text, text
    assert "not stop-terminated" not in text, "nothing to qualify"
