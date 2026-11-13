"""
Survival analysis: Kaplan-Meier curves, log-rank test, Cox proportional hazards.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class KaplanMeierResult:
    times: list = field(default_factory=list)
    survival_probs: list = field(default_factory=list)
    confidence_lower: list = field(default_factory=list)
    confidence_upper: list = field(default_factory=list)
    n_at_risk: list = field(default_factory=list)
    median_survival: float = 0.0
    number_events: int = 0


def _survival_pair(times, events, caller):
    """Return one (times, events) pair as equal-length 1-D arrays, or refuse.

    Every entry point used to hand the two columns straight to numpy, so a
    mismatched pair either died inside numpy (``IndexError: index 2 is out of
    bounds for axis 0``, ``ValueError: operands could not be broadcast
    together`` — both from the CLI, where the user types the two lists by hand)
    or, when ``events`` was the longer one, was silently truncated.  An event
    column coded 2 = censored produced a flat curve next to an honest
    ``number_events`` summed from that same column.
    """
    if times is None or events is None:
        raise TypeError(f"{caller} needs a sequence of times and 0/1 events, got "
                        f"{type(times).__name__} and {type(events).__name__}.")
    try:
        t = np.asarray(times, dtype=float)
        e = np.asarray(events, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{caller} needs numeric times and 0/1 events: {exc}") from exc
    if t.ndim != 1 or e.ndim != 1:
        raise ValueError(f"{caller} needs 1-D times and events, got shapes "
                         f"{t.shape} and {e.shape}.")
    if t.size != e.size:
        raise ValueError(f"{caller} got {t.size} times but {e.size} events; every "
                         f"subject needs one observation and one indicator.")
    if t.size == 0:
        raise ValueError(f"{caller} got no observations.")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{caller} got {int(np.sum(~np.isfinite(t)))} non-finite "
                         f"survival time(s).")
    if np.any(t < 0):
        raise ValueError(f"{caller} got {int(np.sum(t < 0))} negative survival "
                         f"time(s); time is measured from study entry.")
    other = np.unique(e[~np.isin(e, (0.0, 1.0))])
    if other.size:
        raise ValueError(f"{caller} events must be 0 (censored) or 1 (event), got "
                         f"{', '.join(repr(float(c)) for c in other)}.")
    return t, e.astype(int)


def _normal_quantile(confidence_level):
    """The two-sided normal z for a CI level, computed once per call.

    Without scipy the only z available is 1.96, which is the 95 % value; using it
    for a 90 % or 99 % request is a wrong number, so it says so.
    """
    if HAS_SCIPY:
        return float(sp_stats.norm.ppf(1 - (1 - confidence_level) / 2))
    if confidence_level != 0.95:
        warnings.warn("scipy is not installed, so the Greenwood interval keeps its "
                      f"fixed z=1.96 and reports a 95% interval, not the requested "
                      f"{confidence_level} level.", RuntimeWarning, stacklevel=3)
    return 1.96


def kaplan_meier(times, events, confidence_level: float = 0.95) -> KaplanMeierResult:
    """Compute Kaplan-Meier survival estimate.

    Args:
        times: array of survival/censoring times, one per subject, finite and
            non-negative.
        events: array of event indicators (1=event, 0=censored).
        confidence_level: Greenwood interval level, strictly between 0 and 1.

    Returns:
        KaplanMeierResult with survival curve data.  The curve is anchored at
        (0, 1.0) and steps at each observed event time; a curve that never
        reaches 0.5 reports ``median_survival`` as ``inf``.
    """
    times, events = _survival_pair(times, events, "kaplan_meier")
    if (isinstance(confidence_level, bool)
            or not isinstance(confidence_level, (int, float))
            or not 0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be a number strictly between 0 and "
                         f"1, got {confidence_level!r}.")

    sorted_idx = np.argsort(times, kind='stable')
    times = times[sorted_idx]
    events = events[sorted_idx]

    unique_times = np.unique(times[events == 1])
    survival = 1.0
    km_times = [0.0]
    km_survival = [1.0]
    km_lower = [1.0]
    km_upper = [1.0]
    km_at_risk = [len(times)]
    greenwood_term = 0.0  # cumulative sum d_i / (n_i (n_i - d_i))
    z = _normal_quantile(confidence_level)

    for t in sorted(unique_times):
        # t is an observed event time, so at least the subject that failed at t
        # is still at risk: at_risk >= 1 by construction.
        at_risk = int(np.sum(times >= t))
        event_count = int(np.sum((times == t) & (events == 1)))
        survival *= (1 - event_count / at_risk)

        km_times.append(float(t))
        km_survival.append(round(survival, 4))
        km_at_risk.append(at_risk)

        # Greenwood's formula, CUMULATIVE over risk sets — the old code
        # recomputed a single-step approximation per time point, so the
        # CI width never grew with time as it must.
        if at_risk > event_count > 0:
            greenwood_term += event_count / (at_risk * (at_risk - event_count))
        se = survival * np.sqrt(greenwood_term)
        km_lower.append(float(round(max(0.0, survival - z * se), 4)))
        km_upper.append(float(round(min(1.0, survival + z * se), 4)))

    median_surv = 0.0
    for i, s in enumerate(km_survival):
        if s <= 0.5:
            median_surv = km_times[i]
            break
    if median_surv == 0 and km_survival[-1] > 0.5:
        median_surv = float('inf')

    return KaplanMeierResult(
        times=km_times, survival_probs=km_survival,
        confidence_lower=km_lower, confidence_upper=km_upper,
        n_at_risk=km_at_risk,
        median_survival=median_surv, number_events=int(np.sum(events))
    )


def log_rank_test(times1, events1, times2, events2) -> dict:
    """Perform log-rank test comparing two survival curves.

    Without scipy the test cannot be run and the documented "no evidence" pair
    (statistic 0, p_value 1.0) is returned; the inputs are still checked, because
    a mismatched column is the caller's bug either way.
    """
    t1, e1 = _survival_pair(times1, events1, "log_rank_test")
    t2, e2 = _survival_pair(times2, events2, "log_rank_test")
    if not HAS_SCIPY:
        return {'statistic': 0, 'p_value': 1.0}
    all_times = np.unique(np.concatenate([t1, t2]))
    d1_sum = 0
    v_sum = 0
    for t in all_times:
        d1 = np.sum((t1 == t) & (e1 == 1))
        d2 = np.sum((t2 == t) & (e2 == 1))
        n1_at_risk = np.sum(t1 >= t)
        n2_at_risk = np.sum(t2 >= t)
        n_total = n1_at_risk + n2_at_risk
        d1_sum += d1 - n1_at_risk * (d1 + d2) / n_total
        # Guard n_total == 1: the (n_total - 1) denominator is 0 and the
        # O-E contribution is provably 0 there anyway — the old code let
        # 0/0 produce NaN which poisoned v_sum, silently turning STRONG
        # group differences into p = 1.0 on unpaired-end datasets.
        if n_total > 1 and (d1 + d2) > 0:
            v = (n1_at_risk * n2_at_risk * (d1 + d2) * (n_total - d1 - d2)
                 / (n_total**2 * (n_total - 1)))
            v_sum += v

    statistic = d1_sum**2 / v_sum if v_sum > 0 else 0
    # chi2.sf is the numerically stable form; 1 - cdf underflows to exactly 0.0
    # from about chi2 = 100 upward, and popgen.py already uses sf for the same
    # computation.
    p_value = sp_stats.chi2.sf(statistic, df=1) if HAS_SCIPY and statistic > 0 else 1.0
    # Six decimal places would render a genuine 1e-15 as 0.0, which claims more
    # certainty than the data has; keep significant digits instead.
    return {'statistic': float(f"{statistic:.6g}"), 'p_value': float(f"{p_value:.6g}")}


def cox_ph_summary(times, events, covariates) -> dict:
    """Median-split approximation of a one-covariate survival comparison.

    The two group medians are the Kaplan-Meier medians ``kaplan_meier`` itself
    computes, so they agree with the curve a caller can plot from the same data.
    They used to be the medians of the *observed event times*, which ignores
    censoring entirely: on a 10-patient set whose high group had one event at 50
    and three censored afterwards, the old code called 50 the group's median
    survival and divided by it, while the module's own curve never reaches 0.5 —
    ``median survival not reached``, reported as a hazard ratio of 0.05.

    ``hazard_ratio`` is None whenever either median is unreached or the high
    group's is zero, because a ratio of undefined medians is not an estimate.
    The log-rank p-value is always returned, and every key is present on every
    path: the insufficient-data return used to carry three of the seven.
    """
    times, events = _survival_pair(times, events, "cox_ph_summary")
    try:
        covariates = np.asarray(covariates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cox_ph_summary needs a numeric covariate column: {exc}") from exc
    if covariates.ndim != 1:
        raise ValueError(f"cox_ph_summary needs a 1-D covariate, got shape "
                         f"{covariates.shape}.")
    if covariates.size != times.size:
        raise ValueError(f"cox_ph_summary got {covariates.size} covariate values for "
                         f"{times.size} subjects.")
    if not np.all(np.isfinite(covariates)):
        raise ValueError(f"cox_ph_summary got {int(np.sum(~np.isfinite(covariates)))} "
                         f"non-finite covariate value(s).")

    note = ('median-split approximation: no covariate model fit; the '
            'reported ratio is median_low/median_high, not a fitted Cox '
            'hazard ratio')
    if not HAS_SCIPY or np.unique(covariates).size < 2:
        return {
            'hazard_ratio': None,
            'p_value': 1.0,
            'median_high_group': None,
            'median_low_group': None,
            'split': None,
            'note': note,
            'message': ('Insufficient data: nothing to split the cohort on, so no '
                        'estimate is reported'),
        }

    # "greater than the median" empties the high group whenever the median equals
    # the mode (e.g. [1,2,2,2]), which used to report a hazard ratio of 0.0.
    high = covariates > np.median(covariates)
    split = 'median'
    if not high.any() or high.all():
        order = np.argsort(covariates, kind='stable')
        high = np.zeros(covariates.size, dtype=bool)
        high[order[order.size // 2:]] = True
        split = 'rank'

    t_h, e_h = times[high], events[high]
    t_l, e_l = times[~high], events[~high]
    lr = log_rank_test(t_h, e_h, t_l, e_l)
    med_h = kaplan_meier(t_h, e_h).median_survival
    med_l = kaplan_meier(t_l, e_l).median_survival
    reached = np.isfinite(med_h) and np.isfinite(med_l)
    defined = reached and med_h > 0
    if defined:
        message = None
    elif not reached:
        message = ('median survival not reached in at least one group; the ratio is '
                   'undefined, the log-rank p_value is not')
    else:
        message = ('the high group median survival is 0, so the ratio is undefined; '
                   'the log-rank p_value is not')
    return {
        'hazard_ratio': round(med_l / med_h, 3) if defined else None,
        'p_value': lr['p_value'],
        'median_high_group': round(med_h, 1) if np.isfinite(med_h) else None,
        'median_low_group': round(med_l, 1) if np.isfinite(med_l) else None,
        'split': split,
        'note': note,
        'message': message,
    }


def format_km_result(result):
    lines = [
        "=== Kaplan-Meier Survival Analysis ===",
        f"Number of events: {result.number_events}",
        f"Median survival: {result.median_survival}",
        f"Survival at end: {result.survival_probs[-1]:.3f}" if result.survival_probs
        else "Survival at end: no curve was computed",
        f"Time points: {len(result.times)}"
    ]
    return '\n'.join(lines)
