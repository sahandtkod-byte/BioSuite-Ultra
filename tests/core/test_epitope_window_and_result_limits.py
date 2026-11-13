"""Window sizes and result counts: the two numbers a caller passes to a scan.

A window of 0 was the divisor of every property average, so each predictor failed
in its own way — and a negative `max_results` never failed at all, because
`results[:-1]` keeps every epitope but the strongest and looks like a full run.
"""
import pytest

from biosuite.core import epitope as ep

UPPER = "MAVPKRRTSRKEFVGIPSRLMNHQFSDLLSLFMESSGSTSPSINSTQIPSRKNHTWDEQQRGQIR"


# ── window sizes ────────────────────────────────────────────────────────────

def test_a_zero_window_is_refused_before_it_is_divided_by():
    """ZeroDivisionError was the old answer for `window_size=0`."""
    with pytest.raises(ValueError, match=r"window_size must be window sizes >= 1"):
        ep.predict_b_cell_epitopes(UPPER, window_size=0)

    with pytest.raises(ValueError, match=r"window_sizes must be window sizes >= 1"):
        ep.predict_linear_epitopes(UPPER, window_sizes=(0,))


def test_a_zero_window_does_not_answer_with_an_index_error_either():
    """The T-cell scan read `peptide[1]` of an empty slice; three crash, one reason."""
    with pytest.raises(ValueError, match="window_sizes"):
        ep.predict_t_cell_epitopes(UPPER, window_sizes=(0,))


def test_the_t_cell_scan_needs_two_residues_because_it_scores_p2_and_p9():
    """A 1-mer would be its own P2 *and* P9, and both anchor terms would fire.

    Measured on the old code: `window_sizes=(1,)` returned 32 "epitopes".
    """
    with pytest.raises(ValueError, match=r">= 2"):
        ep.predict_t_cell_epitopes(UPPER, window_sizes=(1,))

    assert ep.predict_t_cell_epitopes(UPPER, window_sizes=(2,)) != []


@pytest.mark.parametrize("bad", [2.5, "8", None, [8], True])
def test_a_window_size_must_be_a_whole_number(bad):
    with pytest.raises(ValueError, match="integer window sizes"):
        ep.predict_t_cell_epitopes(UPPER, window_sizes=(bad,))


def test_the_argument_as_a_whole_must_be_a_list_of_them():
    with pytest.raises(ValueError, match="sequence of window sizes"):
        ep.predict_t_cell_epitopes(UPPER, window_sizes=8)


def test_an_empty_window_list_is_a_question_with_no_windows():
    with pytest.raises(ValueError, match="at least one window size"):
        ep.predict_t_cell_epitopes(UPPER, window_sizes=())


def test_a_negative_window_is_refused_rather_than_reading_the_sequence_backwards():
    with pytest.raises(ValueError, match=r"window_size must be window sizes >= 1"):
        ep.predict_b_cell_epitopes(UPPER, window_size=-3)


def test_a_window_larger_than_the_sequence_answers_no_epitopes_instead_of_failing():
    assert ep.predict_b_cell_epitopes("AAAA", window_size=9) == []
    assert ep.predict_t_cell_epitopes("AAAA", window_sizes=(8, 9)) == []


# ── result counts ───────────────────────────────────────────────────────────

def test_all_results_are_reachable_and_a_negative_count_is_not():
    everything = ep.predict_t_cell_epitopes(UPPER, max_results=10 ** 6)

    assert len(everything) == 41, len(everything)
    with pytest.raises(ValueError, match=r"max_results must be >= 0, got -1"):
        ep.predict_t_cell_epitopes(UPPER, max_results=-1)


def test_the_negative_count_that_used_to_pass_dropped_the_strongest_hit():
    """`results[:-1]` kept 40 of 41 epitopes and reported them as if none were cut."""
    everything = ep.predict_t_cell_epitopes(UPPER, max_results=10 ** 6)

    assert len(everything) - 1 == 40, "the size of the lie this used to be"


@pytest.mark.parametrize("predictor", [
    ep.predict_t_cell_epitopes, ep.predict_b_cell_epitopes, ep.predict_linear_epitopes,
])
def test_a_limit_of_zero_answers_an_empty_list(predictor):
    assert predictor(UPPER, max_results=0) == []


@pytest.mark.parametrize("bad", [None, 2.5, "5", True])
def test_a_result_count_must_be_a_whole_number(bad):
    with pytest.raises(ValueError, match="max_results must be an integer"):
        ep.predict_t_cell_epitopes(UPPER, max_results=bad)


def test_a_limit_smaller_than_the_result_set_is_honoured_exactly():
    results = ep.predict_t_cell_epitopes(UPPER, max_results=7)

    assert len(results) == 7
    assert [round(e.score, 12) for e in results] == sorted(
        (round(e.score, 12) for e in results), reverse=True)


def test_the_limit_counts_the_sorted_list_not_the_first_windows_scanned():
    everything = ep.predict_t_cell_epitopes(UPPER, max_results=10 ** 6)

    top_three = ep.predict_t_cell_epitopes(UPPER, max_results=3)
    assert [e.peptide for e in top_three] == [e.peptide for e in everything[:3]]
