"""Regression tests for differential expression (BSU-005 and the zero-variance
p-value defect found while fixing it).

Correctness is checked against ``scipy.stats`` and against the closed-form
Benjamini-Hochberg definition, never against the module's own output.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from biosuite.core.expression import (
    calculate_fold_change, differential_expression,
)


def _counts(n_genes=20, n1=4, n2=4, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean=4.0, sigma=0.6, size=(n_genes, n1 + n2))
    data[:5, n1:] *= 8.0          # first five genes are genuinely up-regulated
    columns = [f"ctrl{i}" for i in range(n1)] + [f"trt{i}" for i in range(n2)]
    return pd.DataFrame(np.round(data), index=[f"g{i}" for i in range(n_genes)],
                        columns=columns)


# ── BSU-005: silent sample loss ─────────────────────────────────────────────

def test_conditions_shorter_than_columns_is_rejected():
    """`zip()` silently truncated to the shorter list, dropping samples."""
    counts = _counts()
    with pytest.raises(ValueError):
        differential_expression(counts, ["A"] * 4 + ["B"] * 3)


def test_conditions_longer_than_columns_is_rejected():
    counts = _counts()
    with pytest.raises(ValueError):
        differential_expression(counts, ["A"] * 4 + ["B"] * 5)


def test_fold_change_rejects_mismatched_conditions():
    counts = _counts()
    with pytest.raises(ValueError):
        calculate_fold_change(counts, ["A"] * 4 + ["B"] * 3)


def test_matching_lengths_are_accepted():
    counts = _counts()
    result = differential_expression(counts, ["A"] * 4 + ["B"] * 4)
    assert len(result) == len(counts)


def test_single_group_is_rejected():
    counts = _counts()
    with pytest.raises(ValueError):
        differential_expression(counts, ["A"] * 8)


# ── scientific correctness against scipy ────────────────────────────────────

def test_p_values_match_scipy_welch_t_test():
    counts = _counts(seed=3)
    conditions = ["A"] * 4 + ["B"] * 4
    result = differential_expression(counts, conditions, method='ttest',
                                     correction='none')
    group_a = counts.iloc[:, :4].to_numpy(dtype=float)
    group_b = counts.iloc[:, 4:].to_numpy(dtype=float)
    expected = stats.ttest_ind(group_a, group_b, axis=1, equal_var=False).pvalue

    p_column = 'pvalue' 
    assert np.allclose(np.asarray(result[p_column], dtype=float), expected,
                       rtol=1e-9, atol=1e-12, equal_nan=True)


def test_benjamini_hochberg_matches_the_closed_form_definition():
    counts = _counts(seed=5)
    conditions = ["A"] * 4 + ["B"] * 4
    result = differential_expression(counts, conditions, method='ttest',
                                     correction='bh')
    p_column = [c for c in result.columns if c.lower() in ('pvalue', 'p_value', 'p')][0]
    q_column = [c for c in result.columns
                if 'adj' in c.lower() or c.lower().startswith('q')][0]

    p = np.asarray(result[p_column], dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    expected = np.empty(n)
    expected[order] = np.clip(ranked, 0, 1)

    assert np.allclose(np.asarray(result[q_column], dtype=float), expected,
                       rtol=1e-9, atol=1e-12, equal_nan=True)


def test_adjusted_p_values_are_never_smaller_than_raw_ones():
    counts = _counts(seed=6)
    result = differential_expression(counts, ["A"] * 4 + ["B"] * 4,
                                     correction='bh')
    p_column = [c for c in result.columns if c.lower() in ('pvalue', 'p_value', 'p')][0]
    q_column = [c for c in result.columns
                if 'adj' in c.lower() or c.lower().startswith('q')][0]
    p = np.asarray(result[p_column], dtype=float)
    q = np.asarray(result[q_column], dtype=float)
    finite = np.isfinite(p) & np.isfinite(q)
    assert np.all(q[finite] >= p[finite] - 1e-12)


def test_truly_changed_genes_rank_above_unchanged_ones():
    counts = _counts(seed=11)
    result = differential_expression(counts, ["A"] * 4 + ["B"] * 4,
                                     correction='none')
    p_column = [c for c in result.columns if c.lower() in ('pvalue', 'p_value', 'p')][0]
    ranked = result.sort_values(p_column)['gene'].tolist()
    spiked = {f"g{i}" for i in range(5)}
    assert spiked.issubset(set(ranked[:8]))


# ── zero within-group variance ──────────────────────────────────────────────

def test_identical_constant_groups_give_p_equal_to_one():
    counts = pd.DataFrame(
        [[5, 5, 5, 5]], index=["g0"],
        columns=["a1", "a2", "b1", "b2"])
    result = differential_expression(counts, ["A", "A", "B", "B"],
                                     correction='none')
    p_column = [c for c in result.columns if c.lower() in ('pvalue', 'p_value', 'p')][0]
    assert float(result[p_column].iloc[0]) == pytest.approx(1.0)


def test_different_constant_groups_do_not_get_a_fabricated_p_value():
    """Zero within-group variance makes the t statistic undefined.

    The old code returned ``p = 1.0`` here, i.e. "no evidence of a
    difference", for two groups that differ by a factor of ten with no
    overlap - a fabricated statistic, and in the anti-conservative direction
    for anyone filtering on p.  scipy returns NaN for the same input.
    """
    counts = pd.DataFrame(
        [[1, 1, 10, 10]], index=["g0"],
        columns=["a1", "a2", "b1", "b2"])
    result = differential_expression(counts, ["A", "A", "B", "B"],
                                     correction='none')
    p_column = [c for c in result.columns if c.lower() in ('pvalue', 'p_value', 'p')][0]
    value = float(result[p_column].iloc[0])
    scipy_value = stats.ttest_ind([1.0, 1.0], [10.0, 10.0], equal_var=False).pvalue
    assert np.isnan(value), f"expected NaN like scipy ({scipy_value}), got {value}"


def test_fold_change_is_computed_for_undefined_p_values():
    """A NaN p-value must not hide a real fold change."""
    counts = pd.DataFrame(
        [[1, 1, 10, 10]], index=["g0"],
        columns=["a1", "a2", "b1", "b2"])
    result = differential_expression(counts, ["A", "A", "B", "B"],
                                     correction='none')
    fc_column = [c for c in result.columns if 'fold' in c.lower() or 'fc' in c.lower()][0]
    assert float(result[fc_column].iloc[0]) > 1.0


# ── gene identity ───────────────────────────────────────────────────────────

def test_gene_names_in_the_index_are_preserved():
    """Index labels used to be replaced by row numbers in the result."""
    counts = _counts(n_genes=6)
    result = differential_expression(counts, ["A"] * 4 + ["B"] * 4)
    assert result['gene'].tolist() == counts.index.tolist()


def test_explicit_gene_column_still_wins():
    counts = _counts(n_genes=6).reset_index().rename(columns={'index': 'gene'})
    result = differential_expression(counts, ["A"] * 4 + ["B"] * 4)
    assert result['gene'].tolist() == counts['gene'].tolist()


def test_positional_labels_when_there_are_no_gene_names():
    counts = _counts(n_genes=6).reset_index(drop=True)
    result = differential_expression(counts, ["A"] * 4 + ["B"] * 4)
    assert result['gene'].tolist() == list(range(6))
