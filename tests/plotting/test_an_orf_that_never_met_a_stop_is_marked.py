"""An ORF that ran out of sequence is not an ORF that met a stop codon.

Both scanners report a span from ATG to the first stop, and when no stop exists
they report the span to the end of the sequence and say nothing.  On a finished
contig that is normal (the gene continues past the read); on a dashboard it is a
different claim, and the record had no field to tell them apart — the count and
the colour scale both treated the two the same.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.core.sequence import reverse_complement
from biosuite.plotting import sequence_viewer as sv

#: ATG plus twenty glycine codons and no stop anywhere.
OPEN = "ATG" + "GGG" * 20
#: The same ORF with its stop codon.
CLOSED = OPEN + "TAG"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_scan_that_never_met_a_stop_codon_says_it_did_not():
    assert sv._find_orfs_forward(OPEN, min_aa=5) == [
        {"frame": 1, "start": 0, "end": 63, "length": 21, "strand": "+",
         "terminated": False}]
    assert sv._find_orfs_forward(CLOSED, min_aa=5) == [
        {"frame": 1, "start": 0, "end": 63, "length": 21, "strand": "+",
         "terminated": True}], "the stop codon itself is not part of the span"


def test_the_reverse_scan_marks_its_open_ends_the_same_way():
    """A reverse ORF with no stop is built by giving the complement one."""
    dna = reverse_complement("ATG" + "AAA" * 20)

    found = sv._find_orfs_reverse(dna, min_aa=5)

    assert found == [{"frame": -1, "start": 0, "end": 63, "length": 21,
                      "strand": "-", "terminated": False}], found
    assert sv._find_orfs_reverse(OPEN, min_aa=5) == [], \
        "this sequence has no reverse ORF to mislabel"


def test_the_map_writes_the_question_mark_the_record_earned():
    fig = sv.draw_orf_map(OPEN, orfs=sv._find_orfs_forward(OPEN, min_aa=5))

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts if t.get_text().endswith("aa?")] == ["21aa?"]
    assert ax.patches[0].get_linestyle() == "--", "the edge says it too"
    assert "1 run to the end without a stop codon" in ax.get_title(), ax.get_title()


def test_a_closed_orf_gets_no_mark_at_all():
    """The positive control: the question mark is not decoration on every bar."""
    fig = sv.draw_orf_map(CLOSED, orfs=sv._find_orfs_forward(CLOSED, min_aa=5))

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts if t.get_text().endswith("aa")] == ["21aa"]
    assert ax.patches[0].get_linestyle() in ("-", "solid")
    assert "without a stop codon" not in ax.get_title()


def test_records_from_before_this_field_existed_are_read_as_closed():
    """A caller that built its own dicts is not suddenly reporting open ORFs."""
    fig = sv.draw_orf_map("A" * 600, orfs=[
        {"frame": 1, "start": 0, "end": 300, "length": 100}])

    assert [t.get_text() for t in fig.axes[0].texts if t.get_text().endswith("aa")] == \
        ["100aa"]
