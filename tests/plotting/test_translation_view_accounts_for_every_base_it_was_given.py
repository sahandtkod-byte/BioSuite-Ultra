"""Every base a translation view is given ends up drawn, or said to be left out.

Two things used to disappear without a word: the one or two bases at the 3' end
that do not complete a codon (a 7 bp view shows 6 and looks like a 2-codon
sequence), and, past that, whole codons — the panel has no budget, so the
dashboard's fourth panel drew every codon of a 3 kb record and cost two seconds
and five thousand artists to get there.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _dna_title(fig):
    return fig.axes[0].get_title(loc="left")


def _protein_title(fig):
    return fig.axes[1].get_title(loc="left")


def test_a_trailing_base_is_counted_instead_of_vanishing():
    fig = sv.draw_translation_view("ATGAAAC")

    assert _dna_title(fig).startswith("DNA  (frame 1)")
    assert "1 trailing base in no codon" in _dna_title(fig), _dna_title(fig)
    assert [t.get_text() for t in fig.axes[1].texts] == ["M", "K"]


def test_two_trailing_bases_are_plural_and_a_multiple_of_three_is_silent():
    """The positive control: the note only appears when bases really are left."""
    two = sv.draw_translation_view("ATGAA")
    exact = sv.draw_translation_view("ATGAAACCC")

    assert "2 trailing bases in no codon" in _dna_title(two), _dna_title(two)
    assert _dna_title(exact) == "DNA  (frame 1)", _dna_title(exact)
    plt.close(two); plt.close(exact)


def test_the_view_stops_at_the_codon_budget_and_says_where():
    seq = "ATG" + "GGG" * 1_200
    before = set(plt.get_fignums())

    fig = sv.draw_translation_view(seq)

    assert len(set(plt.get_fignums()) - before) == 1, "one figure, not a leak"
    assert len(fig.axes[1].patches) == sv._MAX_VIEW_CODONS
    assert "first 500 of 1,201 codons" in _dna_title(fig), _dna_title(fig)
    assert "first 500 of 1,201 codons" in _protein_title(fig), _protein_title(fig)


def test_the_limit_is_on_artists_and_not_on_the_reading_of_the_first_codons():
    """What is drawn is still the right answer, just not all of it."""
    fig = sv.draw_translation_view("ATG" + "GGG" * 1_200)

    assert [t.get_text() for t in fig.axes[1].texts][:3] == ["M", "G", "G"]
    assert len([t for t in fig.axes[0].texts if t.get_text() in "ACGT"]) == \
        3 * sv._MAX_VIEW_CODONS
