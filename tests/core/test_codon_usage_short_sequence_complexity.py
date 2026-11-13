"""`sequence_complexity` on a molecule shorter than one window.

With `window=20` the loop range `range(0, len(seq) - window + 1, window // 2)` is
empty for any sequence under 20 bases, so nothing was measured and the function
still answered `average_complexity 0.0, is_low_complexity True` — 'ACGTACGT', which
carries 4 distinct dinucleotides in 7 windows, was reported as a low-complexity
artifact.
"""
import pytest

from biosuite.core.codon_usage import sequence_complexity


def test_a_short_but_diverse_sequence_is_measured_not_declared_artifact():
    got = sequence_complexity("ACGTACGT", window=20)
    assert got["regions"] == [{"position": 0, "complexity": 0.5714}]
    assert got["average_complexity"] == pytest.approx(0.5714)
    assert got["is_low_complexity"] is False


def test_a_three_base_sequence_scores_all_of_its_dinucleotides():
    assert sequence_complexity("ACG", window=20)["regions"] == [
        {"position": 0, "complexity": 1.0}]


def test_a_homopolymer_of_the_same_short_length_is_still_low_complexity():
    """Positive control: the clamp must not make everything look diverse.  Eight
    'A' bases hold one distinct dinucleotide across seven windows -> 1/7."""
    got = sequence_complexity("A" * 8, window=20)
    assert got["regions"] == [{"position": 0, "complexity": 0.1429}]
    assert got["is_low_complexity"] is True


def test_a_window_that_fits_is_unchanged():
    twenty = "ACGTACGTACGTACGTACGT"
    assert sequence_complexity(twenty, window=20) == {
        "regions": [{"position": 0, "complexity": 0.25}],
        "average_complexity": 0.25, "is_low_complexity": True}


def test_sliding_regions_step_by_half_a_window():
    got = sequence_complexity("ACGTACGTAC", window=4)
    assert [r["position"] for r in got["regions"]] == [0, 2, 4, 6]
    assert all(r["complexity"] == 1.0 for r in got["regions"])


@pytest.mark.parametrize("seq", ["", "A"])
def test_a_sequence_with_no_dinucleotide_has_no_regions_and_scores_zero(seq):
    """Fewer than two characters hold no dinucleotide at all, so 0.0 here is the
    defined value rather than a missed measurement."""
    got = sequence_complexity(seq, window=20)
    assert got == {"regions": [], "average_complexity": 0.0, "is_low_complexity": True}


@pytest.mark.parametrize("window", [0, 1, -3, 2.5, "20", True])
def test_a_useless_window_is_refused(window):
    with pytest.raises(ValueError,
                       match=r"window must be an integer >= 2, got "):
        sequence_complexity("ACGTACGT", window=window)
