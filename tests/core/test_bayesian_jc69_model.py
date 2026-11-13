"""The Jukes-Cantor 69 model: what a transition probability is, and its floor.

`JC69Model` is the only substitution model the built-in engine offers, so every
likelihood the sampler reports is a product of these two numbers.  The clamping of a
non-positive branch is the model's own, separate from the pruning floor in
`test_bayesian_pruning_and_prior.py`: `prob`/`matrix` clamp at 1e-10, while a child
with no branch length at all is scored at `PRUNING_BRANCH_FLOOR` = 0.001.
"""
import numpy as np
import pytest

from biosuite.core.bayesian_phylogeny import JC69Model


@pytest.mark.parametrize("t", [0.01, 0.1, 0.5, 1.0, 10.0, 500.0])
def test_every_row_of_the_transition_matrix_is_a_probability_distribution(t):
    matrix = JC69Model.matrix(t)
    assert len(matrix) == 4 and all(len(row) == 4 for row in matrix)
    for row in matrix:
        assert sum(row) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in row)
    assert matrix[0][0] == matrix[3][3], "JC69 is symmetric across the four states"
    assert matrix[0][1] == matrix[0][2] == matrix[0][3], "all three changes share one rate"


def test_a_long_enough_branch_saturates_the_model_to_a_uniform_choice():
    """exp(-4·500/3) underflows to 0.0, so staying and switching both become 0.25 and
    the site stops carrying information about the state — the correct limit, and the
    one the sampler's Exponential(10) prior exists to keep the chain away from."""
    assert JC69Model.matrix(500.0) == [[0.25] * 4 for _ in range(4)]
    assert JC69Model.prob(True, 500.0) == pytest.approx(0.25)


@pytest.mark.parametrize("t", [0.01, 0.1, 0.5, 1.0, 10.0])
def test_staying_beats_switching_for_every_branch_short_enough_to_resolve(t):
    matrix = JC69Model.matrix(t)
    assert matrix[0][0] > matrix[0][1]


@pytest.mark.parametrize("t, expected_same", [
    pytest.param(0.0, 0.25 + 0.75, id="zero-branch-is-identity"),
    pytest.param(0.75, 0.25 + 0.75 * np.exp(-1.0), id="quoted-from-the-closed-form"),
])
def test_the_matrix_follows_the_closed_form_it_documents(t, expected_same):
    """P(same|t) = 1/4 + 3/4·exp(-4t/3), which is what the class docstring promises."""
    assert JC69Model.matrix(t)[0][0] == pytest.approx(expected_same)


def test_staying_is_more_likely_than_switching_and_the_two_add_up():
    assert JC69Model.prob(True, 0.5) > JC69Model.prob(False, 0.5)
    assert JC69Model.prob(True, 0.5) + 3 * JC69Model.prob(False, 0.5) == pytest.approx(1.0)
    assert JC69Model.prob(False, 0.5) == pytest.approx(0.25 - 0.25 * np.exp(-2.0 / 3.0))
    assert JC69Model.prob(True, 0.5) == pytest.approx(0.25 + 0.75 * np.exp(-2.0 / 3.0))


def test_a_negative_branch_is_clamped_to_the_floor_rather_than_exponenting_upward():
    """`exp(-4t/3)` with t = -5 is exp(6.67) = 785, so P(same) would be 589.0 — the
    clamp is what keeps a matrix a distribution."""
    assert JC69Model.prob(True, -5.0) == pytest.approx(JC69Model.prob(True, 1e-10))
    assert JC69Model.matrix(-1.0)[0][0] == pytest.approx(1.0)
    assert sum(JC69Model.matrix(-1.0)[0]) == pytest.approx(1.0)
