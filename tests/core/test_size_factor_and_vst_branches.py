"""Size factors and the variance-stabilising transform: both estimator branches.

`deseq2_normalization` and `variance_stabilizing_transformation` switch between
median-of-ratios and plain library-size scaling on one condition — whether at
least five genes are positive in every sample.  Only the library-size side had
ever run, so the median-of-ratios loop, its 1e-8 floor and the 'log' transform
were untested.  Every expectation here is recomputed from the definition with
numpy, not copied out of the module.
"""
import re

import numpy as np
import pandas as pd
import pytest

from biosuite.core.expression import (
    _benjamini_hochberg,
    deseq2_normalization,
    variance_stabilizing_transformation as vst,
)


def median_of_ratios(counts):
    """DESeq2's estimator, written out here as the independent reference."""
    positive = (counts > 0).all(axis=1)
    if positive.sum() < 5:
        lib = counts.sum(axis=0)
        lib = np.where(lib == 0, 1.0, lib)
        return lib / lib.mean()
    kept = counts[positive]
    ref = np.exp(np.log(kept).mean(axis=1))
    return np.array([max(np.median(kept[:, j] / ref), 1e-8)
                     for j in range(counts.shape[1])])


@pytest.fixture()
def fully_positive():
    """Six genes, all positive in every sample, with sample b exactly twice a."""
    counts = pd.DataFrame({
        "gene": [f"g{i}" for i in range(6)],
        "a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "b": [20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
        "c": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
    })
    return counts, counts[["a", "b", "c"]].values


# ── deseq2_normalization ─────────────────────────────────────────────────────

def test_median_of_ratios_branch_is_used_when_enough_genes_are_positive(fully_positive):
    counts, values = fully_positive
    assert (values > 0).all(axis=1).sum() == 6
    norm = deseq2_normalization(counts)
    assert norm.attrs["size_factors"] == pytest.approx(median_of_ratios(values))
    # A library that is exactly twice as deep gets exactly twice the factor.
    factors = norm.attrs["size_factors"]
    assert factors[1] == pytest.approx(2 * factors[0])
    assert norm["gene"].tolist() == counts["gene"].tolist()


def test_normalised_counts_are_counts_divided_by_their_factor(fully_positive):
    counts, values = fully_positive
    norm = deseq2_normalization(counts)
    factors = norm.attrs["size_factors"]
    assert norm[["a", "b", "c"]].values == pytest.approx(values / factors)


def test_the_ratio_of_a_deep_library_is_corrected_not_the_gene_order(fully_positive):
    """Scaling must not reorder genes or invent values: the first row divided by
    its factor is what comes back."""
    counts, values = fully_positive
    norm = deseq2_normalization(counts)
    assert norm["a"].iloc[0] == pytest.approx(values[0, 0] / norm.attrs["size_factors"][0])
    assert norm["c"].iloc[-1] == pytest.approx(values[-1, 2] / norm.attrs["size_factors"][2])


def test_library_size_fallback_below_five_fully_positive_genes():
    counts = pd.DataFrame({"gene": ["a", "b", "c"],
                           "s1": [10.0, 0.0, 5.0],
                           "s2": [20.0, 0.0, 15.0],
                           "s3": [30.0, 0.0, 25.0]})
    values = counts[["s1", "s2", "s3"]].values
    assert (values > 0).all(axis=1).sum() == 2
    norm = deseq2_normalization(counts)
    lib = values.sum(axis=0)
    assert norm.attrs["size_factors"] == pytest.approx(lib / lib.mean())
    assert norm.attrs["size_factors"] == pytest.approx(median_of_ratios(values))


def test_an_all_zero_library_does_not_divide_by_zero():
    """A sample nobody sequenced gets the library-size guard's 1.0 stand-in, so the
    other columns keep their scale instead of turning into inf."""
    counts = pd.DataFrame({"gene": ["a", "b"], "s1": [0.0, 0.0], "s2": [1.0, 2.0]})
    norm = deseq2_normalization(counts)
    factors = norm.attrs["size_factors"]
    lib = counts[["s1", "s2"]].values.sum(axis=0)
    assert lib.tolist() == [0.0, 3.0]
    guarded = np.where(lib == 0, 1.0, lib)
    assert np.isfinite(factors).all()
    assert factors == pytest.approx(guarded / guarded.mean())
    assert norm["s1"].tolist() == [0.0, 0.0]
    assert norm["s2"].tolist() == pytest.approx([1.0 / 1.5, 2.0 / 1.5])


def test_a_constant_matrix_gets_unit_size_factors():
    """Every sample identical: every ratio is 1, so the median is 1 and nothing
    is rescaled."""
    counts = pd.DataFrame({"gene": [f"g{i}" for i in range(5)],
                           "s1": [10.0] * 5, "s2": [10.0] * 5, "s3": [10.0] * 5})
    norm = deseq2_normalization(counts)
    assert norm.attrs["size_factors"] == pytest.approx([1.0, 1.0, 1.0])
    assert norm[["s1", "s2", "s3"]].values == pytest.approx(counts[["s1", "s2", "s3"]].values)


def test_non_numeric_columns_are_left_alone(fully_positive):
    counts, values = fully_positive
    counts["note"] = ["x"] * 6
    norm = deseq2_normalization(counts)
    assert norm["note"].tolist() == ["x"] * 6
    assert norm.attrs["size_factors"] == pytest.approx(median_of_ratios(values))


# ── variance_stabilizing_transformation ─────────────────────────────────────

def test_vst_default_transform_uses_the_asinh_formula(fully_positive):
    counts, values = fully_positive
    factors = median_of_ratios(values)
    out = vst(counts)
    x = values / factors
    assert out[["a", "b", "c"]].values == pytest.approx(np.log2(np.sqrt(x) + np.sqrt(x + 1)))


def test_vst_log_method_is_the_shifted_log2(fully_positive):
    counts, values = fully_positive
    factors = median_of_ratios(values)
    out = vst(counts, method="log")
    assert out[["a", "b", "c"]].values == pytest.approx(np.log2(values / factors + 0.5))
    # The two transforms differ by design, so a swapped branch shows up here.
    assert not np.allclose(out[["a", "b", "c"]].values, vst(counts)[["a", "b", "c"]].values)


def test_vst_without_factors_scales_the_way_deseq2_does(fully_positive):
    counts, values = fully_positive
    assert vst(counts).attrs.get("size_factors") is None
    manual = np.log2(np.sqrt(values / median_of_ratios(values))
                     + np.sqrt(values / median_of_ratios(values) + 1))
    assert vst(counts)[["a", "b", "c"]].values == pytest.approx(manual)


def test_vst_with_explicit_factors_divides_by_them():
    """Two libraries holding the same counts but sequenced to different depths:
    the factor the caller supplies decides which one keeps its signal."""
    counts = pd.DataFrame({"gene": ["a", "b"], "s1": [10.0, 20.0], "s2": [10.0, 20.0]})
    out = vst(counts, size_factors=np.array([2.0, 5.0]))
    x = counts[["s1", "s2"]].values / np.array([2.0, 5.0])
    assert out[["s1", "s2"]].values == pytest.approx(np.log2(np.sqrt(x) + np.sqrt(x + 1)))
    assert (out["s1"] > out["s2"]).all(), "the sample divided by 5 must sit lower"


@pytest.mark.parametrize("method", ["vstt", "LOG", "log2", "", None, "asinh"])
def test_an_unrecognised_transform_method_is_refused(method):
    """Anything that was not 'vst' or 'log' silently ran the asinh transform, so
    method='log2' returned a different statistic than the caller asked for."""
    counts = pd.DataFrame({"gene": ["a"], "s1": [10.0]})
    with pytest.raises(ValueError,
                       match=rf"^Unknown method: {re.escape(str(method))}\. "
                              rf"Use 'vst' or 'log'\.$"):
        vst(counts, method=method)


def test_both_valid_transform_methods_still_run():
    """Positive control for the refusal above.  One row falls back to library-size
    factors, so both samples normalise to 25 counts and the two transforms are
    log2(sqrt(25)+sqrt(26)) and log2(25.5)."""
    counts = pd.DataFrame({"gene": ["a"], "s1": [10.0], "s2": [40.0]})
    assert vst(counts, method="vst")["s1"].iloc[0] == pytest.approx(3.3361433273)
    assert vst(counts, method="log")["s1"].iloc[0] == pytest.approx(4.6724253420)


def test_vst_falls_back_to_library_size_below_five_positive_genes():
    counts = pd.DataFrame({"gene": ["a", "b", "c"],
                           "s1": [10.0, 0.0, 5.0], "s2": [20.0, 0.0, 15.0]})
    values = counts[["s1", "s2"]].values
    assert (values > 0).all(axis=1).sum() == 2
    lib = values.sum(axis=0)
    # The reference itself selects the fallback, so this also pins which branch ran.
    assert median_of_ratios(values) == pytest.approx(lib / lib.mean())
    out = vst(counts)
    x = values / (lib / lib.mean())
    assert out[["s1", "s2"]].values == pytest.approx(np.log2(np.sqrt(x) + np.sqrt(x + 1)))


def test_vst_log_method_on_a_zero_gene_is_finite():
    counts = pd.DataFrame({"gene": ["off"], "s1": [0.0], "s2": [0.0]})
    out = vst(counts, method="log", size_factors=np.array([1.0, 1.0]))
    assert np.isfinite(out[["s1", "s2"]].values).all()
    assert out["s1"].iloc[0] == pytest.approx(np.log2(0.5))


def test_vst_keeps_the_gene_column_and_row_order(fully_positive):
    counts, _v = fully_positive
    out = vst(counts)
    assert out["gene"].tolist() == counts["gene"].tolist()
    assert list(out.columns) == list(counts.columns)


# ── Benjamini-Hochberg on degenerate input ──────────────────────────────────

def test_bh_of_an_all_nan_vector_returns_nan_without_raising():
    """A batch in which every gene was degenerate used to reach the sorting code
    with an empty valid set; the corrected path returns NaNs, not a crash and not
    fabricated q-values."""
    out = _benjamini_hochberg(np.array([np.nan, np.nan, np.nan]))
    assert np.isnan(out).all()
    assert out.shape == (3,)


def test_bh_of_an_empty_vector_is_empty():
    assert _benjamini_hochberg(np.array([])).shape == (0,)


def test_bh_keeps_nan_positions_and_adjusts_only_the_rest():
    p = np.array([0.01, np.nan, 0.4])
    out = _benjamini_hochberg(p)
    assert np.isnan(out[1])
    assert out[0] == pytest.approx(0.02)   # 0.01 * 2 / 1
    assert out[2] == pytest.approx(0.4)     # 0.40 * 2 / 2, capped at 1
    assert np.isnan(out).tolist() == [False, True, False]
