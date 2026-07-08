"""
Survival analysis: Kaplan-Meier curves, log-rank test, Cox proportional hazards.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

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


def kaplan_meier(times, events, confidence_level: float = 0.95) -> KaplanMeierResult:
    """Compute Kaplan-Meier survival estimate.

    Args:
        times: array of survival/censoring times.
        events: array of event indicators (1=event, 0=censored).
        confidence_level: confidence interval level.

    Returns:
        KaplanMeierResult with survival curve data.
    """
    times = np.array(times)
    events = np.array(events)
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    events = events[sorted_idx]

    unique_times = np.unique(times[events == 1])
    n_at_risk = len(times)
    survival = 1.0
    km_times = [0.0]
    km_survival = [1.0]
    km_lower = [1.0]
    km_upper = [1.0]

    for t in sorted(unique_times):
        at_risk = np.sum(times >= t)
        event_count = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            survival *= (1 - event_count / at_risk)

        km_times.append(float(t))
        km_survival.append(round(survival, 4))

        se = survival * np.sqrt(np.sum([1/((at_risk - i) * (at_risk - i - 1))
            for i in range(event_count) if at_risk - i > 1])) if at_risk > 1 else 0
        z = sp_stats.norm.ppf(1 - (1 - confidence_level) / 2) if HAS_SCIPY else 1.96
        km_lower.append(round(max(0, survival - z * se), 4))
        km_upper.append(round(min(1, survival + z * se), 4))

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
        median_survival=median_surv, number_events=int(np.sum(events))
    )


def log_rank_test(times1, events1, times2, events2) -> dict:
    """Perform log-rank test comparing two survival curves."""
    if not HAS_SCIPY:
        return {'statistic': 0, 'p_value': 1.0}
    t1, e1 = np.array(times1), np.array(events1)
    t2, e2 = np.array(times2), np.array(events2)
    all_times = np.unique(np.concatenate([t1, t2]))
    n1, n2 = len(t1), len(t2)
    d1_sum, d2_sum = 0, 0
    v_sum = 0
    for t in all_times:
        d1 = np.sum((t1 == t) & (e1 == 1))
        d2 = np.sum((t2 == t) & (e2 == 1))
        n1_at_risk = np.sum(t1 >= t)
        n2_at_risk = np.sum(t2 >= t)
        n_total = n1_at_risk + n2_at_risk
        if n_total == 0:
            continue
        d1_sum += d1 - n1_at_risk * (d1 + d2) / n_total
        v = n1_at_risk * n2_at_risk * (d1 + d2) * (n_total - d1 - d2) / (n_total**2 * (n_total - 1))
        v_sum += v

    statistic = d1_sum**2 / v_sum if v_sum > 0 else 0
    p_value = 1 - sp_stats.chi2.cdf(statistic, df=1) if HAS_SCIPY and statistic > 0 else 1.0
    return {'statistic': round(float(statistic), 4), 'p_value': round(float(p_value), 6)}


def cox_ph_summary(times, events, covariates) -> dict:
    """Simplified Cox PH analysis (single covariate).

    Args:
        times: survival times.
        events: event indicators.
        covariates: array of covariate values.

    Returns:
        Dict with hazard ratio and p-value.
    """
    if not HAS_SCIPY or len(np.unique(covariates)) < 2:
        return {'hazard_ratio': 1.0, 'p_value': 1.0, 'message': 'Insufficient data'}
    high = covariates > np.median(covariates)
    t_h, e_h = times[high], events[high]
    t_l, e_l = times[~high], events[~high]
    lr = log_rank_test(t_h, e_h, t_l, e_l)
    median_h = np.median(t_h[e_h == 1]) if np.any(e_h == 1) else np.inf
    median_l = np.median(t_l[e_l == 1]) if np.any(e_l == 1) else np.inf
    hr = median_l / median_h if median_h > 0 else 999
    return {'hazard_ratio': round(float(hr), 3), 'p_value': lr['p_value'],
            'median_high_group': round(float(median_h), 1),
            'median_low_group': round(float(median_l), 1)}


def format_km_result(result):
    lines = [
        "=== Kaplan-Meier Survival Analysis ===",
        f"Number of events: {result.number_events}",
        f"Median survival: {result.median_survival}",
        f"Survival at end: {result.survival_probs[-1]:.3f}",
        f"Time points: {len(result.times)}"
    ]
    return '\n'.join(lines)
