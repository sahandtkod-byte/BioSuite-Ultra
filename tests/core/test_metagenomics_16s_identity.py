"""The sliding-window identity scorer of metagenomics' 16S pipeline.

`_compute_identity` reports the best-matching window pair, which is a local
similarity, not an alignment identity — the tests below say exactly what it can
and cannot answer, including the two degenerate sides (a read shorter than the
window, and a window that is not a window at all).
"""
import pytest

from biosuite.core import metagenomics as mg

REF = mg.SILVA_16S_DB["Escherichia coli"]


def test_an_identical_pair_scores_one():
    assert mg._compute_identity(REF, REF) == 1.0


def test_a_query_with_no_base_in_common_with_any_window_scores_zero():
    assert mg._compute_identity("N" * len(REF), REF, window=20) == 0.0


def test_the_best_offset_inside_a_longer_query_is_found():
    """A 20-base perfect match buried in noise must be reported, or the whole
    point of a sliding window is lost."""
    query = "NNNNNNNNN" + REF[:20] + "NNNNNNNNN"
    assert mg._compute_identity(query, REF, window=20) == 1.0


def test_a_partial_window_is_reported_as_its_fraction():
    """19 shared bases followed only by N: no 20-window can do better than
    19/20, and every later window of the query is N's."""
    altered = REF[:19] + "N" * (len(REF) - 19)
    assert mg._compute_identity(altered, REF, window=20) == pytest.approx(19 / 20)


def test_the_score_is_symmetric_for_two_long_sequences():
    a = mg.SILVA_16S_DB["Escherichia coli"]
    b = mg.SILVA_16S_DB["Bacillus subtilis"]
    assert mg._compute_identity(a, b) == pytest.approx(mg._compute_identity(b, a))


def test_different_genera_do_not_score_as_a_match():
    a = mg.SILVA_16S_DB["Escherichia coli"]
    b = mg.SILVA_16S_DB["Helicobacter pylori"]
    assert mg._compute_identity(a, b, window=20) < 0.9


# ── below the window ────────────────────────────────────────────────────────

def test_a_pair_shorter_than_the_window_is_compared_as_a_prefix():
    assert mg._compute_identity("ACGT", "ACGA") == pytest.approx(0.75)
    assert mg._compute_identity("AC", "ACGT") == pytest.approx(1.0)


def test_the_shorter_of_the_two_sets_how_many_bases_are_compared():
    """The prefix path compares min(len) bases, which is why a 2-base read can
    reach 1.0 — classify_16s_rna() is the place that turns this into 'too_short'."""
    assert mg._compute_identity("AA", "ACGTACGT" * 4) == pytest.approx(0.5)


def test_empty_sequences_score_zero_rather_than_dividing_by_zero():
    assert mg._compute_identity("", "") == 0
    assert mg._compute_identity("A", "") == 0
    assert mg._compute_identity("", "ACGT") == 0


# ── the window argument ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_window", [0, -1, 1.5, "20", True, None])
def test_a_window_that_is_not_a_base_count_is_refused(bad_window):
    """window=0 divided the empty chunk comparison by zero."""
    with pytest.raises((ValueError, TypeError)):
        mg._compute_identity(REF, REF, window=bad_window)


def test_a_window_of_one_finds_any_shared_base():
    """With a one-base window the score is 'do these two sequences share a base
    at all' — which is why the default window is 20, not 1."""
    assert mg._compute_identity("ACG", "TTTT", window=1) == pytest.approx(0.0)
    assert mg._compute_identity("ACGT", "ACGA", window=1) == pytest.approx(1.0)


def test_the_default_window_is_the_one_the_pipeline_reports():
    assert mg.IDENTITY_WINDOW == 20
    assert mg._compute_identity("A" * 20, "A" * 19 + "C") == pytest.approx(19 / 20)
