"""A pseudocount is a prior, not a decoration.

``compute_logo_heights`` accepted ``pseudocount`` and ignored it.  The prior now
adds that much weight to every alphabet letter at a column that has data, so a
conserved column loses some of its information content and a letter nobody wrote
still gets a share.  A column with no data stays empty.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from biosuite.plotting.conservation_plots import compute_logo_heights

CONSERVED = ["AAAA"] * 8


def test_zero_pseudocount_is_the_default():
    with_zero = compute_logo_heights(CONSERVED, pseudocount=0)
    with_default = compute_logo_heights(CONSERVED)

    assert np.array_equal(with_zero[2], with_default[2])


def test_a_prior_lowers_a_conserved_column():
    without = compute_logo_heights(CONSERVED)[2]
    _positions, _heights, with_prior = compute_logo_heights(CONSERVED, pseudocount=1)

    assert 0 < with_prior[0] < without[0]


def test_a_prior_gives_a_letter_nobody_wrote_a_share():
    _positions, heights, total = compute_logo_heights(CONSERVED, pseudocount=1)

    assert heights["C"][0] > 0
    assert sum(heights[c][0] for c in "ACGT") == pytest.approx(total[0])


def test_a_prior_that_swamps_the_data_flattens_the_column():
    _positions, _heights, total = compute_logo_heights(CONSERVED, pseudocount=1000)

    assert total[0] == 0.0


def test_a_prior_does_not_invent_letters_in_a_gap_column():
    _positions, heights, total = compute_logo_heights(["A-C", "A-G"], pseudocount=1)

    assert total[1] == 0.0
    assert heights["A"][1] == 0.0


def test_a_prior_leaves_the_column_count_alone():
    positions, _heights, _total = compute_logo_heights(["ACGTACGT", "AC"], pseudocount=2)

    assert positions == list(range(8))
