"""How ``cox_ph_summary`` splits a cohort on its median and what it is allowed to
divide by.

The group medians are the Kaplan-Meier medians of the two groups, computed by
``kaplan_meier`` itself.  Before this cluster they were ``np.median`` of the *observed
event times*, which ignores censoring: on the cohort below - the high group has one event
at 50 and three subjects censored after it - the old code reported
``median_high_group == 50.0`` and a hazard ratio of ``0.05``, a 20-fold effect, while the
module's own curve says that group's median survival is never reached.  Input columns are
checked in ``test_survival_input_validation.py``.
"""
import numpy as np
import pytest

from biosuite.core.survival import cox_ph_summary, kaplan_meier

pytest.importorskip("scipy", reason="the log-rank p-value needs scipy's chi-square")

TIMES = [1.0, 2.0, 3.0, 4.0, 50.0, 60.0, 70.0, 80.0]
EVENTS = [1, 1, 1, 1, 1, 0, 0, 0]
COVARIATES = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

COX_KEYS = {'hazard_ratio', 'median_high_group', 'median_low_group', 'note',
            'p_value', 'split', 'message'}


def test_cox_medians_are_the_medians_kaplan_meier_itself_reports():
    high = np.asarray(COVARIATES) > np.median(COVARIATES)
    km_high = kaplan_meier(np.asarray(TIMES)[high], np.asarray(EVENTS)[high])
    km_low = kaplan_meier(np.asarray(TIMES)[~high], np.asarray(EVENTS)[~high])
    assert km_high.median_survival == float('inf')

    out = cox_ph_summary(TIMES, EVENTS, COVARIATES)
    assert out['median_high_group'] is None
    assert out['median_low_group'] == round(km_low.median_survival, 1)
    assert out['hazard_ratio'] is None
    assert 'not reached' in out['message']
    assert 0.0 <= out['p_value'] <= 1.0


def test_a_reached_median_still_gives_the_ratio_it_promised():
    out = cox_ph_summary([10.0, 20.0, 30.0, 40.0], [1, 1, 1, 1], [1.0, 2.0, 2.0, 3.0])
    assert out['split'] == 'median'
    assert out['median_high_group'] == 40.0
    assert out['median_low_group'] == 20.0
    assert out['hazard_ratio'] == 0.5
    assert out['message'] is None
    # Independent of the module: 3 subjects at times 10/20/30, every one an event,
    # S steps 2/3 then 1/3 - the first time at or below 0.5 is 20.
    assert kaplan_meier([10.0, 20.0, 30.0], [1, 1, 1]).median_survival == 20.0


def test_a_median_that_equals_the_mode_falls_back_to_a_rank_split():
    """[1,2,2,2]: "greater than the median" leaves the high group empty, and the
    rank split has to answer instead of dividing by an infinite median."""
    out = cox_ph_summary([10.0, 20.0, 30.0, 40.0], [1, 1, 1, 1], [1.0, 2.0, 2.0, 2.0])
    assert out['split'] == 'rank'
    assert out['median_high_group'] == 30.0
    assert out['median_low_group'] == 10.0
    assert out['hazard_ratio'] == 0.333
    assert out['message'] is None


def test_a_high_group_median_of_zero_is_reported_as_undefined():
    out = cox_ph_summary([1.0, 2.0, 0.0, 0.0], [1, 1, 1, 1], [1.0, 1.0, 2.0, 2.0])
    assert out['median_high_group'] == 0.0
    assert out['median_low_group'] == 1.0
    assert out['hazard_ratio'] is None
    assert out['message'] == ('the high group median survival is 0, so the ratio is '
                              'undefined; the log-rank p_value is not')


@pytest.mark.parametrize("case", [
    pytest.param(dict(times=[10.0, 20.0, 30.0, 40.0], events=[1, 1, 1, 1],
                      cov=[1.0, 2.0, 2.0, 3.0]), id="healthy"),
    pytest.param(dict(times=[10.0, 20.0, 30.0, 40.0], events=[1, 1, 1, 1],
                      cov=[1.0, 2.0, 2.0, 2.0]), id="rank-split"),
    pytest.param(dict(times=TIMES, events=EVENTS, cov=COVARIATES), id="median-unreached"),
    pytest.param(dict(times=[1, 2, 3], events=[1, 1, 1], cov=[5, 5, 5]),
                 id="constant-covariate"),
    pytest.param(dict(times=[10.0, 20.0, 30.0, 40.0], events=[0, 0, 0, 0],
                      cov=[1.0, 2.0, 2.0, 3.0]), id="no-events-at-all"),
])
def test_every_cox_path_returns_the_same_keys(case):
    """The insufficient-data return used to carry three of the seven keys, so
    ``out['median_high_group']`` raised KeyError for exactly the callers that most
    needed to see that there was no estimate."""
    out = cox_ph_summary(case['times'], case['events'], case['cov'])
    assert set(out) == COX_KEYS
    if case['cov'] == [5, 5, 5]:
        assert out['hazard_ratio'] is None, "no split means no estimate, not 'no effect'"
        assert out['median_high_group'] is None and out['median_low_group'] is None
        assert out['split'] is None
        assert out['message'].startswith('Insufficient data')
