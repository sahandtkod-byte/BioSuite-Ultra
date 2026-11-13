"""Motif enrichment checks the motifs it was given.

A motif is a pattern, so an empty one matches every sequence, a repeated one
draws the same bar twice, and a letter outside the alphabet can only be a typo.
The old ``alphabet`` parameter was ignored entirely; it is now the set of
letters a motif may use, and a wider alphabet is the documented way to count an
ambiguity code.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.conservation_plots import plot_motif_enrichment

SEQUENCES = ["ATGA", "ATGA", "CGCCC"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_an_empty_motif_is_refused():
    with pytest.raises(ValueError, match="motif 1 is not a non-empty string: ''"):
        plot_motif_enrichment(SEQUENCES, [""])


def test_an_empty_motif_list_is_refused():
    with pytest.raises(ValueError, match="at least one motif"):
        plot_motif_enrichment(SEQUENCES, [])


def test_a_bare_motif_string_is_not_a_list():
    with pytest.raises(ValueError, match="not a single string"):
        plot_motif_enrichment(SEQUENCES, "ATG")


def test_a_motif_that_is_not_a_string_is_reported_by_position():
    with pytest.raises(ValueError, match="motif 1 is not a non-empty string: 7"):
        plot_motif_enrichment(SEQUENCES, [7])


def test_a_motif_repeated_in_another_case_is_still_the_same_motif():
    with pytest.raises(ValueError, match="motifs repeat: \\['ATG'\\]"):
        plot_motif_enrichment(SEQUENCES, ["ATG", "atg"])


def test_a_motif_letter_outside_the_alphabet_is_refused():
    with pytest.raises(ValueError, match="not in alphabet 'ACGT'"):
        plot_motif_enrichment(["ATNG"], ["ANG"])


def test_a_wider_alphabet_allows_an_ambiguity_code():
    fig = plot_motif_enrichment(["ATANG"], ["ANG"], alphabet="ACGTN")

    assert [bar.get_width() for bar in fig.axes[0].patches] == [1]


def test_a_lowercase_motif_counts_uppercase_sequences():
    fig = plot_motif_enrichment(SEQUENCES, ["atg"])

    assert [bar.get_width() for bar in fig.axes[0].patches] == [2]
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["ATG"]


def test_a_motif_counts_lowercase_sequences_the_same_way():
    fig = plot_motif_enrichment([s.lower() for s in SEQUENCES], ["ATG", "CCC"])

    assert [bar.get_width() for bar in fig.axes[0].patches] == [2, 1]


def test_an_empty_sequence_list_is_refused():
    with pytest.raises(ValueError, match="motif enrichment needs at least one sequence"):
        plot_motif_enrichment([], ["ATG"])


def test_an_alignment_of_empty_sequences_is_refused():
    with pytest.raises(ValueError, match="at least one non-empty sequence"):
        plot_motif_enrichment(["", ""], ["ATG"])
