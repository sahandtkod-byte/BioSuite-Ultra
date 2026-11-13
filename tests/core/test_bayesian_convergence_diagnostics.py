"""The two convergence estimators the Bayesian report prints: ESS and PSRF.

Pinned against hand-computable cases rather than against the module's own arithmetic:
a chain with no variance has nothing to thin, an anti-correlated chain breaks the
initial-positive-sequence window at lag 1, and two chains that disagree by a known
amount give a known ratio.  The 5,000-sample iid chain is the one case that reaches the
`max_lag = min(n // 2, 500)` cap; the short traces all close the window earlier.
"""
import numpy as np
import pytest

from biosuite.core.bayesian_phylogeny import _compute_ess, _compute_psrf

#: A log-likelihood trace that wanders slowly: every autocovariance up to the
#: initial-positive-sequence window (n // 2) stays positive, so the window closes on
#: max_lag instead of on the first non-positive lag.  Values are random.Random(459)
#: gauss(0, 1) increments, cumulative, rounded to 3 dp.
WANDERING_TRACE = [1.655, 1.222, 0.919, -0.046, -0.067, -0.681, -2.581, -1.856, -2.627,
                   -0.6, -0.274, -1.465, -2.462, -3.379, -5.193, -4.803, -3.542, -3.481,
                   -3.506, -3.695]


@pytest.mark.parametrize("samples, expected", [
    pytest.param([1.0] * 19, 19.0, id="short-chain-counts-samples"),
    pytest.param([1.0], 1.0, id="single-sample-chain"),
    pytest.param([5.0] * 100, 100.0, id="zero-variance-chain"),
    pytest.param(WANDERING_TRACE, 4.19430332119379, id="window-closes-at-max-lag"),
    pytest.param([-2.0 * (-1) ** k for k in range(40)], 40.0,
                 id="anti-correlated-breaks-at-lag-one"),
])
def test_ess_matches_the_initial_positive_sequence_estimator(samples, expected):
    assert _compute_ess(samples) == pytest.approx(expected)


def test_an_independent_chain_gets_most_of_its_samples_back():
    """A 5,000-draw iid chain is the case where the lag window runs to its 500 cap.
    Measured: the first autocorrelation is already non-positive, so tau stays 1.0 and
    ESS is the sample count — which is the right answer for independent draws."""
    iid = list(np.random.default_rng(1).normal(0, 1, 5000))
    assert _compute_ess(iid) == pytest.approx(5000.0)


@pytest.mark.parametrize("means, variances, n_per, expected", [
    pytest.param([-120.0], [1.0], 100, 1.0, id="single-chain-is-converged-by-definition"),
    pytest.param([], [], 100, 1.0, id="no-chain-at-all"),
    pytest.param([0.0, 1.0], [0.0, 0.0], 100, 1.0, id="zero-within-chain-variance"),
    pytest.param([0.0, 0.0], [1.0, 1.0], 1000, 0.999499874937461, id="identical-chains"),
    pytest.param([0.0, 1.0], [1.0, 1.0], 100, 1.2206555615733703, id="two-disagreeing-chains"),
])
def test_psrf_only_computes_a_ratio_when_two_chains_vary(means, variances, n_per, expected):
    assert _compute_psrf(means, variances, n_per) == pytest.approx(expected)


def test_psrf_grows_with_the_spread_between_chains_and_not_with_the_noise_inside():
    """Same within-chain variance, means moved apart: V+ rises, so the factor rises.
    The 1.1 threshold the docstring quotes as "acceptable" has to move this way.
    Measured: 0.9975 for means 0.0/0.1 and 3.6729 for means 0.0/5.0 — and note the
    first is below 1.0, because V+ weights W by (n-1)/n before adding B/n."""
    close = _compute_psrf([0.0, 0.1], [1.0, 1.0], 100)
    far = _compute_psrf([0.0, 5.0], [1.0, 1.0], 100)
    assert close == pytest.approx(0.9974968671630001)
    assert far == pytest.approx(3.6728735344413916)
    assert close < 1.0 < 1.1 < far
