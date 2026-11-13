"""Conservation edges: entropy of an empty bin, ragged alignments, custom colours.

``compute_logo_heights`` walks a rectangle of ``max_len`` columns, so sequences of
different lengths, or a column of gap characters, reach the "nothing to count here"
guards.  A column's height is its information content minus a small-sample
correction and never falls below zero, so these tests assert the *shape* of the
result — which columns are zero, which beat which, and how depth moves them —
rather than a constant copied out of the formula.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.conservation_plots import (
    _normalize_frequencies,
    _shannon_entropy,
    compute_conservation_scores,
    compute_logo_heights,
    plot_motif_enrichment,
    plot_sequence_logo,
)

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_entropy_of_a_certain_base_is_zero_bits():
    assert _shannon_entropy({'A': 1.0}) == pytest.approx(0.0)


def test_entropy_ignores_a_probability_of_zero():
    """0 · log2(0) is not a number, so an absent letter must contribute nothing."""
    assert _shannon_entropy({'A': 0.5, 'C': 0.5, 'G': 0.0, 'T': 0.0}) == pytest.approx(1.0)


def test_counts_that_add_up_to_nothing_normalise_to_nothing():
    assert _normalize_frequencies({}) == {}
    assert _normalize_frequencies({'A': 0, 'C': 0}) == {}
    assert _normalize_frequencies({'A': 2, 'C': 2}) == {'A': 0.5, 'C': 0.5}


def test_a_fully_mixed_column_carries_no_information():
    freqs = _normalize_frequencies({'A': 1, 'C': 1, 'G': 1, 'T': 1})
    assert _shannon_entropy(freqs) == pytest.approx(2.0), "four equal letters fill DNA's 2 bits"

    _, _, total = compute_logo_heights(["A", "C", "G", "T"])

    assert total[0] == pytest.approx(0.0), "a column anyone can read carries nothing to show"


def test_ragged_alignment_scores_only_the_columns_it_has_data_for():
    positions, heights, total = compute_logo_heights(["ACGT", "AC"])

    assert positions == [0, 1, 2, 3], "one slot per column of the longest sequence"
    assert total[0] == pytest.approx(total[1]), "both columns hold A twice"
    assert total[2] == 0.0 and total[3] == 0.0, "no sequence reaches column 2, so it holds nothing"


def test_a_gap_column_is_skipped_and_not_counted_as_a_letter():
    positions, heights, total = compute_logo_heights(["A-C", "A-G", "A-"])

    assert total[1] == 0.0, "a column of gap characters has nothing to count"
    assert sum(heights[c][1] for c in "ACGT") == 0.0
    assert total[0] > total[1], "the aligned A column is still conserved"


def test_a_conserved_column_beats_a_mixed_one_of_the_same_depth():
    _, _, conserved = compute_logo_heights(["A"] * 8)
    _, _, mixed = compute_logo_heights(["A"] * 6 + ["C"] * 2)

    assert conserved[0] > mixed[0] > 0.0, "six of eight A beats two thirds A, but neither is zero"
    assert conserved[0] < 2.0, "a DNA column never exceeds its two bits"


def test_a_deeper_alignment_scores_the_same_column_higher():
    """The small-sample correction shrinks with depth, so identical data at n=8 must
    be read as more conserved than the same data at n=2."""
    _, _, shallow = compute_logo_heights(["A"] * 2)
    _, _, deep = compute_logo_heights(["A"] * 8)

    assert deep[0] > shallow[0] > 0.0


def _rgb_of(hex_colour):
    return tuple(round(int(hex_colour[i:i + 2], 16) / 255, 3) for i in (1, 3, 5))


def test_the_logo_uses_the_colour_scheme_it_was_handed():
    sequences = ["ACGT", "ACGT", "ACGA"]
    scheme = {'A': '#101010', 'C': '#202020', 'G': '#303030', 'T': '#404040'}

    fig = plot_sequence_logo(sequences, color_scheme=scheme)

    filled = [p for p in fig.axes[0].patches if p.get_height() > 0]
    assert filled, "the logo drew the stacked letters"
    expected = {_rgb_of(s) for s in scheme.values()}
    seen = {tuple(round(float(value), 3) for value in p.get_facecolor()[:3]) for p in filled}
    assert seen <= expected, "a colour outside the caller's scheme appeared"


def test_the_default_logo_colours_each_base_differently():
    """One column per base plus an all-A column, so all four letters must be drawn."""
    fig = plot_sequence_logo(["AAAA"] * 7 + ["ACGT"])

    filled = [p for p in fig.axes[0].patches if p.get_height() > 0]
    assert len(filled) == 7, "one bar for the pure column and two for each mixed one"
    seen = {tuple(round(float(value), 3) for value in p.get_facecolor()[:3]) for p in filled}
    assert len(seen) == 4, f"A, C, G and T are meant to be distinguishable, saw {sorted(seen)}"


def test_conservation_scores_cover_every_column():
    scores = compute_conservation_scores(["ACGT", "ACGT"])

    assert [pos for pos, _value in scores] == [0, 1, 2, 3]
    assert all(value > 0 for _pos, value in scores), "an identical alignment is fully conserved"


def test_motif_enrichment_counts_the_sequences_carrying_each_motif():
    sequences = ["ATGATG", "TTATGTT", "ATGCGT", "CGCGCG"]

    fig = plot_motif_enrichment(sequences, ["ATG", "CCC"])

    bars = fig.axes[0].patches
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["ATG", "CCC"]
    counts = [bar.get_width() for bar in bars]
    assert counts == [3, 0], f"ATG occurs in three of the four rows: {counts}"
    assert [t.get_text() for t in fig.axes[0].texts] == ["3", "0"], "each bar is annotated with its own count"
