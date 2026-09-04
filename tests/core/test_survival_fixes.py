"""Regression tests for survival.py review fixes."""
import numpy as np
import pytest

from biosuite.core.survival import (
    kaplan_meier, log_rank_test, cox_ph_summary, KaplanMeierResult,
)


def test_km_hand_computed_curve():
    r = kaplan_meier([3, 4, 5, 6, 7], [1, 0, 1, 0, 0])
    assert r.times == [0.0, 3.0, 5.0]
    assert r.survival_probs == pytest.approx([1.0, 0.8, 0.5333], abs=1e-3)
    assert r.n_at_risk == [5, 5, 3]      # previously always []
    assert r.number_events == 2
    assert r.median_survival == float('inf')  # curve stays above 0.5


def test_km_median_and_full_event_case():
    r = kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    # S: .75 .5 .25 0 -> first time S <= 0.5 is t=2
    assert r.median_survival == 2.0
    assert r.survival_probs[-1] == 0.0


def test_km_greenwood_ci_grows_monotone():
    rng = np.random.default_rng(2)
    times = np.sort(rng.uniform(1, 50, 40))
    events = (rng.random(40) > 0.3).astype(int)
    r = kaplan_meier(times, events)
    widths = np.array(r.confidence_upper) - np.array(r.confidence_lower)
    # Greenwood accumulates: the widest CI must come later than the first
    # event (old code recomputed per-step width with no accumulation).
    # (The terminal S=0 point degenerates to width 0 by construction.)
    assert widths[1:-1].max() >= widths[1]
    assert all(0 <= l <= u <= 1 for l, u in
               zip(r.confidence_lower, r.confidence_upper))


def test_log_rank_strong_difference():
    rng = np.random.default_rng(3)
    t1 = rng.exponential(5, 60)     # short survival
    t2 = rng.exponential(30, 60)    # long survival
    e = np.ones(60)
    res = log_rank_test(t1, e, t2, e)
    assert res['p_value'] < 0.001
    assert res['statistic'] > 10


def test_log_rank_identical_nonsig():
    rng = np.random.default_rng(4)
    t1 = rng.exponential(10, 80)
    t2 = rng.exponential(10, 80)
    res = log_rank_test(t1, np.ones(80), t2, np.ones(80))
    assert res['p_value'] > 0.01


def test_cox_summary_median_split_api():
    rng = np.random.default_rng(5)
    cov = rng.uniform(0, 1, 60)
    times = np.where(cov > 0.5, rng.exponential(5, 60), rng.exponential(30, 60))
    out = cox_ph_summary(times, np.ones(60), cov)
    assert {'hazard_ratio', 'p_value', 'note'} <= set(out)


def test_cox_insufficient_covariate():
    out = cox_ph_summary([1, 2, 3], [1, 1, 1], [5, 5, 5])
    assert out['hazard_ratio'] == 1.0
