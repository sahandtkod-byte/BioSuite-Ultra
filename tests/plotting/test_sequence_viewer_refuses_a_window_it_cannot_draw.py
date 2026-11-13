"""The window a caller asks ``draw_sequence_view`` for has to describe a range.

It used to slice with whatever it was given: ``start=-5`` drew the five last
bases and labelled them ``-5``, ``end`` before ``start`` produced eight tracks of
nothing, a ``start`` past the sequence produced the same eight, and a fractional
position was rounded into a window nobody asked for.  Every one of those came
back as a plausible figure.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

#: 56 bases: a start codon, three full codons and a tail of C's.
SEQ = "ATGAAACCCGGGTTTA" + "C" * 40


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _refusal(*args, **kwargs):
    """The message from a refused call, plus proof that nothing was drawn."""
    before = set(plt.get_fignums())
    with pytest.raises(ValueError) as excinfo:
        sv.draw_sequence_view(*args, **kwargs)
    assert set(plt.get_fignums()) == before, "a refused view opened a figure anyway"
    return str(excinfo.value)


def test_a_negative_start_is_named_instead_of_read_as_the_tail():
    text = _refusal(SEQ, start=-5)

    assert "start must be 0 or more, got -5" in text, text


def test_an_end_before_its_start_names_both_numbers():
    text = _refusal(SEQ, start=5, end=3)

    assert "end (3) must be after start (5)" in text, text


def test_a_window_past_the_end_says_how_long_the_sequence_actually_is():
    text = _refusal(SEQ, start=100)

    assert "past the end of a 56 bp sequence" in text, text


def test_a_fractional_position_is_not_rounded_into_a_window_nobody_asked_for():
    text = _refusal(SEQ, start=4.5)

    assert "start must be a base position, got 4.5" in text, text


def test_an_empty_sequence_is_refused_before_the_tracks_exist():
    text = _refusal("")

    assert "must not be empty" in text, text


def test_a_pasted_fasta_header_is_not_a_sequence():
    text = _refusal(">NM_001.2 fake header")

    assert "not IUPAC nucleotides" in text, text


def test_the_ruler_keeps_labelling_absolute_positions_after_all_of_that():
    """The positive control: a legal window still draws, and still says where it is."""
    fig = sv.draw_sequence_view(SEQ, start=10, end=20)

    track = fig.axes[1]
    assert [t.get_text() for t in track.texts] == list(SEQ[10:20])
    assert [t.get_text() for t in fig.axes[0].texts] == ["10"]
    assert track.get_title(loc="left") == "Sequence", "nothing had to be left out"
