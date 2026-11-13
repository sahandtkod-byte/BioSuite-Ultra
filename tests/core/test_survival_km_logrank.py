"""The arithmetic of ``survival.py``: hand-checkable Kaplan-Meier curves, an
independently recomputed Greenwood interval, the log-rank statistic on two known
random samples, and the median-split Cox summary on a known cohort.  Refusals and the
censored-median contract live in
``test_survival_input_validation_and_censored_medians.py``; the report text and the
no-scipy fallback live in ``test_survival_km_report_and_missing_scipy.py``.

Every curve, statistic and dict below was read off the running package.
"""
import math

import numpy as np
import pytest

from biosuite.core.survival import cox_ph_summary, kaplan_meier, log_rank_test

Z95 = 1.959963984540054


def _greenwood_reference(times, events):
    """Kaplan-Meier and Greenwood's formula, recomputed with plain arithmetic.

    Deliberately written the long way round — sorted lists, no numpy, the quantile
    spelled out as a constant — so a shared mistake between it and the module is
    unlikely.
    """
    n_total = len(times)
    event_times = sorted({float(t) for t, e in zip(times, events) if e == 1})
    surv, acc = 1.0, 0.0
    steps = []
    for t in event_times:
        at_risk = sum(1 for x, e in zip(times, events) if x >= t)
        d = sum(1 for x, e in zip(times, events) if x == t and e == 1)
        surv *= (1 - d / at_risk)
        if at_risk > d > 0:
            acc += d / (at_risk * (at_risk - d))
        se = surv * math.sqrt(acc)
        steps.append((t, round(surv, 4), round(max(0.0, surv - Z95 * se), 4),
                      round(min(1.0, surv + Z95 * se), 4), at_risk))
    return n_total, steps


def test_km_hand_computed_curve():
    r = kaplan_meier([3, 4, 5, 6, 7], [1, 0, 1, 0, 0])
    assert r.times == [0.0, 3.0, 5.0]
    assert r.survival_probs == [1.0, 0.8, 0.5333]
    assert r.confidence_lower == [1.0, 0.4494, 0.0468]
    assert r.confidence_upper == [1.0, 1.0, 1.0]
    assert r.n_at_risk == [5, 5, 3]
    assert r.number_events == 2
    assert r.median_survival == float('inf'), "the curve stays above 0.5"


def test_km_curve_that_reaches_zero_keeps_its_terminal_interval_at_zero():
    r = kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert r.times == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert r.survival_probs == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert r.confidence_lower == [1.0, 0.3257, 0.01, 0.0, 0.0]
    assert r.confidence_upper == [1.0, 1.0, 0.99, 0.6743, 0.0]
    assert r.n_at_risk == [4, 4, 3, 2, 1]
    assert r.median_survival == 2.0, "the first time with S <= 0.5"
    assert r.number_events == 4


def test_km_steps_once_per_tied_event_time():
    r = kaplan_meier([1, 1, 2, 2], [1, 0, 1, 0])
    assert r.times == [0.0, 1.0, 2.0]
    assert r.survival_probs == [1.0, 0.75, 0.375]
    assert r.confidence_upper == [1.0, 1.0, 0.9364]
    assert r.n_at_risk == [4, 4, 2]
    assert r.median_survival == 2.0
    assert r.number_events == 2


def test_km_greenwood_interval_matches_an_independent_recomputation():
    """A 40-subject sample with 30 events and censoring, compared step by step to
    ``_greenwood_reference``.  The old code recomputed the interval per step without
    accumulating, so none of these numbers matched."""
    rng = np.random.default_rng(2)
    times = np.sort(rng.uniform(1, 50, 40))
    events = (rng.random(40) > 0.3).astype(int)
    r = kaplan_meier(times, events)

    n_total, steps = _greenwood_reference(times, events)
    assert n_total == 40
    assert r.times[1:] == [s[0] for s in steps]
    assert r.survival_probs[1:] == [s[1] for s in steps]
    assert r.confidence_lower[1:] == [s[2] for s in steps]
    assert r.confidence_upper[1:] == [s[3] for s in steps]
    assert r.n_at_risk == [n_total] + [s[4] for s in steps]
    assert r.n_at_risk[:6] == [40, 40, 39, 38, 37, 36]
    assert r.number_events == 30
    assert r.median_survival == pytest.approx(33.795567630151496)

    widths = [hi - lo for lo, hi in zip(r.confidence_lower, r.confidence_upper)]
    assert widths[0] == 0.0 and widths[1] == pytest.approx(0.0734)
    assert widths[1:13] == sorted(widths[1:13]), "the band keeps widening early on"
    assert max(widths) == pytest.approx(0.3456)
    assert widths.index(max(widths)) == 21, "its widest point is late in the follow-up"


def test_the_curve_lists_hold_plain_floats():
    """The result dataclass advertises lists of numbers; the anchors were Python floats
    while every appended value was a ``numpy.float64``, which is the sort of thing a
    serialiser trips over."""
    result = kaplan_meier([1, 2, 3, 4, 5], [1, 1, 0, 1, 1])
    for name in ('times', 'survival_probs', 'confidence_lower', 'confidence_upper'):
        values = getattr(result, name)
        assert values and all(type(value) is float for value in values), name
    assert all(type(value) is int for value in result.n_at_risk)


def test_log_rank_strong_difference():
    rng = np.random.default_rng(3)
    t1 = rng.exponential(5, 60)     # short survival
    t2 = rng.exponential(30, 60)    # long survival
    e = np.ones(60)
    assert log_rank_test(t1, e, t2, e) == {'statistic': 68.6669,
                                           'p_value': 1.16581e-16}


def test_log_rank_two_samples_from_one_distribution():
    rng = np.random.default_rng(4)
    t1 = rng.exponential(10, 80)
    t2 = rng.exponential(10, 80)
    assert log_rank_test(t1, np.ones(80), t2, np.ones(80)) == {
        'statistic': 0.0798681, 'p_value': 0.777476}


def test_log_rank_is_symmetric_in_its_two_groups():
    t1, e1 = [5, 8, 10], [1, 1, 0]
    t2, e2 = [3, 6, 9], [1, 1, 1]
    assert log_rank_test(t1, e1, t2, e2) == log_rank_test(t2, e2, t1, e1)


def test_cox_median_split_reports_the_values_it_used():
    rng = np.random.default_rng(5)
    covariates = rng.uniform(0, 1, 60)
    times = np.where(covariates > 0.5, rng.exponential(5, 60), rng.exponential(30, 60))
    events = np.ones(60)

    out = cox_ph_summary(times, events, covariates)
    assert out['split'] == 'median'
    assert out['hazard_ratio'] == 7.366
    assert out['p_value'] == 1.23191e-11
    assert out['message'] is None

    # Independent of the module: with every subject failing at a distinct time,
    # S = 1 - k/30 crosses 0.5 at k = 15, so the group median is its 15th smallest
    # survival time - not np.median of the group, which is what the code used to
    # report (4.056 and 30.401 here, giving a ratio of 7.496).
    high = covariates > np.median(covariates)
    assert out['median_high_group'] == round(float(np.sort(times[high])[14]), 1)
    assert out['median_low_group'] == round(float(np.sort(times[~high])[14]), 1)
    assert out['median_high_group'] == 4.0
    assert out['median_low_group'] == 29.2


def test_cox_insufficient_covariate_reports_no_estimate():
    """A constant covariate gives nothing to split on.  The old return said
    ``hazard_ratio == 1.0`` - "no effect", measured from no comparison at all -
    with three of the seven keys present."""
    out = cox_ph_summary([1, 2, 3], [1, 1, 1], [5, 5, 5])
    assert out['hazard_ratio'] is None
    assert out['split'] is None
    assert out['p_value'] == 1.0
    assert out['message'].startswith('Insufficient data')


def test_cox_accepts_plain_lists():
    out = cox_ph_summary([10.0, 20.0, 30.0, 40.0], [1, 1, 1, 1], [1.0, 2.0, 2.0, 3.0])
    assert out['hazard_ratio'] == 0.5
    assert out['median_high_group'] == 40.0
    assert out['p_value'] == 0.182279
