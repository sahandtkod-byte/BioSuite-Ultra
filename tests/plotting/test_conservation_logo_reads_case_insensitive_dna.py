"""The logo reads its alignment once, and reads DNA case-insensitively.

A generator handed to ``compute_logo_heights`` used to be consumed by the
longest-sequence scan and then scored as an empty alignment, and lowercase
bases were ignored as if they were gaps.  Both spellings now describe the same
alignment, and an alphabet keeps the letters it was written with.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np

from biosuite.plotting.conservation_plots import (
    compute_conservation_scores,
    compute_logo_heights,
    plot_logo_with_conservation,
)

SEQUENCES = ["ACGTACGT", "ACGTACGT", "ACGAACGT"]


def test_a_generator_scores_the_same_as_the_list_it_came_from():
    from_generator = compute_logo_heights(s for s in SEQUENCES)
    from_list = compute_logo_heights(SEQUENCES)

    assert from_generator[0] == from_list[0]
    assert all(np.array_equal(from_generator[1][c], from_list[1][c]) for c in "ACGT")
    assert np.array_equal(from_generator[2], from_list[2])


def test_lowercase_bases_are_the_same_alignment_as_uppercase():
    upper = compute_logo_heights(["ACGTACGT"] * 8)
    lower = compute_logo_heights(["acgtacgt"] * 8)

    assert np.array_equal(upper[2], lower[2])
    assert all(np.array_equal(upper[1][c], lower[1][c]) for c in "ACGT")


def test_case_can_change_between_rows_of_one_alignment():
    mixed = compute_logo_heights(["ACGT", "acgt", "AcGt", "ACGT"])
    upper = compute_logo_heights(["ACGT"] * 4)

    assert np.array_equal(mixed[2], upper[2])


def test_a_lowercase_alphabet_names_the_same_letters():
    upper = compute_conservation_scores(["acgt"] * 8, alphabet="ACGT")
    lower = compute_conservation_scores(["acgt"] * 8, alphabet="acgt")

    assert lower == upper
    _positions, heights, _total = compute_logo_heights(["ACGT"] * 8, alphabet="acgt")
    assert set(heights) == set("ACGT")


def test_the_combined_plot_reads_a_generator_once():
    fig = plot_logo_with_conservation(s for s in ["ACGTACGT"] * 8)

    logo, bar = fig.axes
    assert len(logo.patches) == 8, "one letter per column"
    assert len(bar.patches) == 8, "the bar kept all eight columns"

