"""
Kaplan-Meier reporting and the no-scipy fallback in survival.py.

scipy is installed here, so the ``HAS_SCIPY`` branches only run when the module
is re-imported with the dependency blocked.  The contract that matters is that
the fallback says "no evidence" rather than inventing a result.
"""
import importlib.util
import pathlib
import sys
import warnings

import pytest

from biosuite.core.survival import (KaplanMeierResult, cox_ph_summary, format_km_result,
                                   kaplan_meier)

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "biosuite" / "core" / "survival.py"

TIMES = [1, 2, 3, 4, 5]
EVENTS = [1, 1, 0, 1, 1]


def _without_scipy(monkeypatch):
    """Re-import survival.py with ``import scipy`` made to fail."""
    monkeypatch.setitem(sys.modules, "scipy", None)
    spec = importlib.util.spec_from_file_location("survival_without_scipy", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_format_km_result_summarises_the_curve():
    result = kaplan_meier(TIMES, EVENTS)
    assert format_km_result(result) == (
        "=== Kaplan-Meier Survival Analysis ===\n"
        "Number of events: 4\n"
        "Median survival: 4.0\n"
        "Survival at end: 0.000\n"
        "Time points: 5"
    )


def test_format_km_result_marks_a_curve_that_never_reaches_half():
    result = kaplan_meier([5, 6, 7], [0, 0, 0])
    assert format_km_result(result) == (
        "=== Kaplan-Meier Survival Analysis ===\n"
        "Number of events: 0\n"
        "Median survival: inf\n"
        "Survival at end: 1.000\n"
        "Time points: 1"
    )


def test_greenwood_interval_accumulates_over_the_risk_sets():
    result = kaplan_meier(TIMES, EVENTS)
    assert result.times == [0.0, 1.0, 2.0, 4.0, 5.0]
    assert result.survival_probs == [1.0, 0.8, 0.6, 0.3, 0.0]
    assert result.n_at_risk == [5, 5, 4, 2, 1]
    # 0.1706 = 0.6 - 1.96 * 0.6 * sqrt(1/(5*4) + 1/(4*3)); the single-step
    # approximation this replaced gave 0.26 at the same time point.
    assert [float(v) for v in result.confidence_lower] == [1.0, 0.4494, 0.1706, 0.0, 0.0]
    assert [float(v) for v in result.confidence_upper] == [1.0, 1.0, 1.0, 0.7679, 0.0]
    for prob, lower, upper in zip(result.survival_probs,
                                  result.confidence_lower, result.confidence_upper):
        assert 0.0 <= lower <= prob <= upper <= 1.0


@pytest.mark.parametrize("case", [
    {"times": [1, 2, 3, 4], "events": [1, 1, 1, 1]},
    {"times": [1, 1, 2, 2], "events": [1, 0, 1, 0]},
])
def test_without_scipy_no_test_is_fabricated(monkeypatch, case):
    module = _without_scipy(monkeypatch)
    assert module.HAS_SCIPY is False

    assert module.log_rank_test(case["times"], case["events"],
                                case["times"], case["events"]) == {
        "statistic": 0, "p_value": 1.0}

    covariates = [1.0, 2.0, 3.0, 4.0]
    without = module.cox_ph_summary(case["times"], case["events"], covariates)
    assert without['hazard_ratio'] is None, "no split without scipy, and no 'no effect'"
    assert without['p_value'] == 1.0
    assert without['split'] is None
    assert without['message'].startswith('Insufficient data')
    # The same keys the scipy path returns, so a consumer can read any of them.
    with_scipy = cox_ph_summary(case["times"], case["events"], covariates)
    assert set(without) == set(with_scipy)


def test_format_km_result_handles_a_result_that_has_no_curve():
    """``KaplanMeierResult()`` is a legal object - the dataclass has defaults - and the
    formatter raised ``IndexError: list index out of range`` on its empty
    ``survival_probs``, which is what the GUI then showed as an error popup."""
    assert format_km_result(KaplanMeierResult()) == (
        "=== Kaplan-Meier Survival Analysis ===\n"
        "Number of events: 0\n"
        "Median survival: 0.0\n"
        "Survival at end: no curve was computed\n"
        "Time points: 0"
    )
    assert format_km_result(kaplan_meier(TIMES, EVENTS)) == (
        "=== Kaplan-Meier Survival Analysis ===\n"
        "Number of events: 4\n"
        "Median survival: 4.0\n"
        "Survival at end: 0.000\n"
        "Time points: 5"
    )


def test_without_scipy_a_level_that_is_not_95_percent_says_so(monkeypatch):
    """z = 1.96 is the 95 % quantile; using it for a 90 % request without saying so
    would report an interval that is not the one the caller asked for."""
    module = _without_scipy(monkeypatch)
    with pytest.warns(RuntimeWarning, match="reports a 95% interval, not the requested "
                                            "0.9 level"):
        module.kaplan_meier(TIMES, EVENTS, confidence_level=0.90)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert module.kaplan_meier(TIMES, EVENTS).confidence_lower == [
            1.0, 0.4494, 0.1706, 0.0, 0.0]


def test_without_scipy_the_curve_uses_the_normal_approximation(monkeypatch):
    from biosuite.core.survival import kaplan_meier as with_scipy

    module = _without_scipy(monkeypatch)
    fallback = module.kaplan_meier(TIMES, EVENTS)
    reference = with_scipy(TIMES, EVENTS)

    assert fallback.times == reference.times
    assert fallback.survival_probs == reference.survival_probs
    assert fallback.median_survival == reference.median_survival
    # z = 1.96 replaces scipy's norm.ppf(0.975) = 1.959964, which the four
    # decimal places of the reported interval cannot distinguish
    assert fallback.confidence_lower == reference.confidence_lower
    assert fallback.confidence_upper == reference.confidence_upper
