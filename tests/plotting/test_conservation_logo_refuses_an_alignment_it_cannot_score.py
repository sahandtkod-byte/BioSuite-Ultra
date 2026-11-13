"""The conservation plots refuse an alignment or alphabet they cannot score.

An empty alignment has no columns, a one-letter alphabet has nothing to compare,
and a negative pseudocount is not a prior.  Each is reported before any figure is
created, so a failed call leaves no window behind.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.conservation_plots import (
    compute_conservation_scores,
    compute_logo_heights,
    plot_conservation_bar,
    plot_logo_with_conservation,
    plot_sequence_logo,
)

DRAWERS = [
    plot_sequence_logo,
    plot_conservation_bar,
    plot_logo_with_conservation,
    compute_conservation_scores,
]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("entry_point", DRAWERS)
def test_every_entry_point_refuses_an_empty_alignment(entry_point):
    with pytest.raises(ValueError, match="at least one sequence"):
        entry_point([])
    assert plt.get_fignums() == [], "a refused call must not leave a figure open"


def test_a_bare_string_is_not_an_alignment():
    with pytest.raises(ValueError, match="not a single string"):
        compute_logo_heights("ACGT")


def test_something_that_is_not_iterable_is_named():
    with pytest.raises(ValueError, match="got NoneType"):
        compute_logo_heights(None)


def test_a_sequence_that_is_not_a_string_is_reported_by_position():
    with pytest.raises(ValueError, match="sequence 2 is not a string: None"):
        compute_logo_heights(["ACGT", None])


def test_an_alignment_of_empty_sequences_has_nothing_to_score():
    with pytest.raises(ValueError, match="at least one non-empty sequence"):
        compute_logo_heights(["", ""])


def test_an_alphabet_that_is_not_a_string_is_named():
    with pytest.raises(ValueError, match="alphabet must be a string of symbols, got list"):
        compute_logo_heights(["ACGT"], alphabet=["A", "C", "G", "T"])


@pytest.mark.parametrize("alphabet", ["A", ""])
def test_an_alphabet_needs_two_symbols(alphabet):
    with pytest.raises(ValueError, match="at least two symbols"):
        compute_logo_heights(["ACGT"], alphabet=alphabet)


def test_a_repeated_alphabet_symbol_is_refused():
    with pytest.raises(ValueError, match="repeats a symbol"):
        compute_logo_heights(["ACGT"], alphabet="AACG")


@pytest.mark.parametrize("pseudocount", [-1, float("nan"), float("inf")])
def test_a_pseudocount_that_is_not_a_finite_prior_is_refused(pseudocount):
    with pytest.raises(ValueError, match="non-negative finite number"):
        compute_logo_heights(["ACGT"], pseudocount=pseudocount)


@pytest.mark.parametrize("pseudocount", [True, "1"])
def test_a_pseudocount_that_is_not_a_number_is_named(pseudocount):
    with pytest.raises(ValueError, match="non-negative number"):
        compute_logo_heights(["ACGT"], pseudocount=pseudocount)


@pytest.mark.parametrize("entry_point", DRAWERS)
def test_every_entry_point_checks_the_alphabet(entry_point):
    with pytest.raises(ValueError, match="at least two symbols"):
        entry_point(["ACGT"], alphabet="A")
    assert plt.get_fignums() == []
