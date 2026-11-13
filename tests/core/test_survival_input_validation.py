"""The columns ``survival.py`` accepts: the times, the event indicators, the covariate,
and the confidence level it acts on.

The CLI (option 123) types the times and the events as two comma-separated lists and the
GUI reads two CSV columns, so a mismatched pair is the most likely thing a real user does
to this module - and until this cluster it either died inside numpy or, when the event
list was the longer one, was silently cut down to the shorter column's length.  What each
measured failure used to print is written into the test that now refuses it.  What the Cox
median split may divide by is in ``test_survival_cox_median_split.py``.
"""
import re

import pytest

from biosuite.core.survival import cox_ph_summary, kaplan_meier, log_rank_test

pytest.importorskip("scipy", reason="the Greenwood interval needs scipy's normal quantile")

def test_mismatched_time_and_event_columns_are_refused_not_truncated():
    """Measured on the old code: the longer-events case returned a complete-looking
    curve for the first two subjects (``survival_probs == [1.0, 0.5]``,
    ``number_events == 1``) and the longer-times case died with
    ``IndexError: index 2 is out of bounds for axis 0 with size 2``."""
    with pytest.raises(ValueError, match=re.escape(
            "kaplan_meier got 3 times but 2 events; every subject needs one "
            "observation and one indicator.")):
        kaplan_meier([1, 2, 3], [1, 0])

    with pytest.raises(ValueError, match=re.escape(
            "kaplan_meier got 2 times but 3 events")):
        kaplan_meier([1, 2], [1, 0, 1])

    with pytest.raises(ValueError, match=re.escape(
            "log_rank_test got 3 times but 2 events")):
        log_rank_test([1, 2, 3], [1, 0], [1, 2, 3], [1, 1, 1])

    with pytest.raises(ValueError, match=re.escape(
            "cox_ph_summary got 3 times but 2 events")):
        cox_ph_summary([1, 2, 3], [1, 0], [1.0, 2.0, 3.0])


def test_the_covariate_column_is_checked_against_the_subject_count():
    with pytest.raises(ValueError, match=re.escape(
            "cox_ph_summary got 3 covariate values for 4 subjects.")):
        cox_ph_summary([1, 2, 3, 4], [1, 1, 1, 1], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("covariates, error, message", [
    pytest.param(['a', 'b', 'c', 'd'], TypeError,
                 "cox_ph_summary needs a numeric covariate column: could not convert "
                 "string to float: 'a'", id="strings"),
    pytest.param([object(), 1.0, 2.0, 3.0], TypeError,
                 "cox_ph_summary needs a numeric covariate column:", id="objects"),
    pytest.param(None, ValueError,
                 "cox_ph_summary needs a 1-D covariate, got shape ().", id="none"),
    pytest.param([[1.0, 2.0], [3.0, 4.0]], ValueError,
                 "cox_ph_summary needs a 1-D covariate, got shape (2, 2).", id="2-d"),
    pytest.param([float('nan'), 1.0, 2.0, 3.0], ValueError,
                 "cox_ph_summary got 1 non-finite covariate value(s).", id="nan"),
    pytest.param([float('inf'), 1.0, 2.0, 3.0], ValueError,
                 "cox_ph_summary got 1 non-finite covariate value(s).", id="infinity"),
])
def test_a_covariate_column_that_is_not_a_column_of_numbers_is_refused(covariates,
                                                                       error, message):
    with pytest.raises(error, match=re.escape(message)):
        cox_ph_summary([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], covariates)


def test_an_event_column_coded_2_is_refused_not_counted():
    """Measured on the old code with ``events == [2, 2, 0]``: an all-flat curve
    ``[1.0]`` — because nothing equalled 1 — beside ``number_events == 4``, which
    summed the raw column.  Two files describing one analysis, two answers."""
    with pytest.raises(ValueError, match=re.escape(
            "kaplan_meier events must be 0 (censored) or 1 (event), got 2.0.")):
        kaplan_meier([1, 2, 3], [2, 2, 0])

    with pytest.raises(ValueError, match=re.escape("events must be 0 (censored) or 1")):
        log_rank_test([1, 2], [1, 3], [1, 2], [1, 1])

    # Positive control: 0/1 and the boolean spelling of it both still work.
    assert kaplan_meier([1, 2, 3], [1, 0, 1]).number_events == 2
    assert kaplan_meier([1, 2, 3], [True, False, True]).number_events == 2


@pytest.mark.parametrize("times, message", [
    pytest.param([-5.0, 1.0, 2.0],
                 "kaplan_meier got 1 negative survival time(s); time is measured "
                 "from study entry.", id="negative"),
    pytest.param([float('nan'), 1.0, 2.0],
                 "kaplan_meier got 1 non-finite survival time(s).", id="nan"),
    pytest.param([float('inf'), 1.0, 2.0],
                 "kaplan_meier got 1 non-finite survival time(s).", id="infinity"),
])
def test_times_that_are_not_observable_durations_are_refused(times, message):
    """Measured on the old code: ``[-5, 1, 2]`` produced a curve whose time axis was
    ``[0.0, -5.0, 1.0, 2.0]`` — a survival curve that goes backwards past its own
    origin — and a NaN subject simply disappeared from every risk set."""
    with pytest.raises(ValueError, match=re.escape(message)):
        kaplan_meier(times, [1, 1, 1])


@pytest.mark.parametrize("times, events, message", [
    pytest.param(['a', 'b'], [1, 0], "needs numeric times and 0/1 events", id="strings"),
    pytest.param(None, [1],
                 "kaplan_meier needs a sequence of times and 0/1 events, got NoneType "
                 "and list.", id="none-times"),
    pytest.param([[1.0, 2.0], [3.0, 4.0]], [1, 1],
                 "needs 1-D times and events, got shapes (2, 2) and (2,).", id="2-d"),
    pytest.param(5.0, 1, "needs 1-D times and events, got shapes () and ().", id="scalar"),
])
def test_inputs_that_are_not_columns_are_refused(times, events, message):
    with pytest.raises((TypeError, ValueError), match=re.escape(message)):
        kaplan_meier(times, events)


def test_an_empty_cohort_is_refused_rather_than_reported_as_a_one_point_curve():
    """Measured on the old code: ``kaplan_meier([], [])`` answered
    ``times == [0.0], survival_probs == [1.0], median_survival == inf``, and the
    report called that "Time points: 1, Survival at end: 1.000"."""
    with pytest.raises(ValueError, match=re.escape("kaplan_meier got no observations.")):
        kaplan_meier([], [])

    with pytest.raises(ValueError, match=re.escape("log_rank_test got no observations.")):
        log_rank_test([], [], [1.0], [1])

    with pytest.raises(ValueError, match=re.escape("cox_ph_summary got no observations.")):
        cox_ph_summary([], [], [])


@pytest.mark.parametrize("level", [
    pytest.param(0, id="zero"),
    pytest.param(0.0, id="zero-float"),
    pytest.param(1, id="one"),
    pytest.param(1.0, id="one-float"),
    pytest.param(1.5, id="above-one"),
    pytest.param(-0.5, id="negative"),
    pytest.param('0.95', id="string"),
    pytest.param(None, id="none"),
    pytest.param(True, id="bool"),
])
def test_a_confidence_level_that_is_not_a_level_is_refused(level):
    """Measured on the old code: -0.5 returned intervals whose *lower* bound sat
    above the *upper* one (``0.9207`` against ``0.6793``), 1.0 and 1.5 clamped to
    ``[0, 1]`` through a NaN quantile and a ``RuntimeWarning: invalid value
    encountered in scalar multiply``, 0 collapsed the interval onto the curve, and
    '0.95'/None raised ``TypeError: unsupported operand type(s) for -``."""
    with pytest.raises(ValueError, match=re.escape(
            f"confidence_level must be a number strictly between 0 and 1, got {level!r}.")):
        kaplan_meier([1, 2, 3, 4, 5], [1, 1, 0, 1, 1], confidence_level=level)
