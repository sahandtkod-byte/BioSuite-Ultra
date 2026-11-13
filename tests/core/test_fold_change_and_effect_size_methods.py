"""`calculate_fold_change` and `calculate_effect_size` — method branches,
shrinkage, bands and the refusals.

Both functions are reachable only from the API/notebook layer, and before this
file neither had run its non-default branches: the median and trimmed fold
changes, the apeglm-style shrinkage, and the rank-biserial effect size were all
dead lines.  The references below are computed from the standard definitions
(numpy / scipy) rather than copied from the implementation, so swapping the
median and trimmed branches would be caught.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, trim_mean

from biosuite.core.expression import calculate_effect_size, calculate_fold_change

CTRL = ["ctrl"] * 12 + ["trt"] * 12


@pytest.fixture()
def outlier_matrix():
    """Twelve replicates a group so that trim_mean(0.1) really cuts one value
    from each end (int(0.1 * 12) = 1), with a wild outlier in each group pulling
    the arithmetic mean away from both the median and the trimmed mean."""
    ctrl = np.array([
        [500.0, 10.0, 11.0, 9.0, 12.0, 10.0, 11.0, 9.0, 10.0, 12.0, 11.0, 10.0],
        [20.0, 21.0, 19.0, 20.0, 21.0, 19.0, 20.0, 21.0, 19.0, 20.0, 21.0, 19.0],
    ])
    trt = np.array([
        [14.0, 15.0, 13.0, 14.0, 15.0, 13.0, 14.0, 15.0, 13.0, 14.0, 15.0, 13.0],
        [60.0, 61.0, 59.0, 60.0, 61.0, 59.0, 60.0, 61.0, 59.0, 60.0, 61.0, 1000.0],
    ])
    df = pd.DataFrame(np.hstack([ctrl, trt]),
                      columns=[f"c{i}" for i in range(12)] + [f"t{i}" for i in range(12)])
    df.insert(0, "gene", ["outlier_in_ctrl", "outlier_in_trt"])
    return df, ctrl, trt


# ── the three fold-change methods ────────────────────────────────────────────

def test_mean_method_and_base_mean_match_the_definitions(outlier_matrix):
    df, ctrl, trt = outlier_matrix
    fc = calculate_fold_change(df, CTRL)
    assert fc["log2FC"].to_numpy() == pytest.approx(
        np.log2((trt.mean(axis=1) + 1) / (ctrl.mean(axis=1) + 1)))
    assert fc["baseMean"].to_numpy() == pytest.approx(
        np.mean(np.hstack([ctrl, trt]), axis=1))
    assert fc["gene"].tolist() == ["outlier_in_ctrl", "outlier_in_trt"]


def test_median_method_survives_the_outlier_the_mean_cannot(outlier_matrix):
    df, ctrl, trt = outlier_matrix
    fc = calculate_fold_change(df, CTRL, method="median")
    assert fc["log2FC"].to_numpy() == pytest.approx(
        np.log2((np.median(trt, axis=1) + 1) / (np.median(ctrl, axis=1) + 1)))
    plain = calculate_fold_change(df, CTRL)
    # The outlier flips the sign of the mean-based estimate on the first gene;
    # the median stays with the bulk of the replicates.
    assert plain["log2FC"][0] < 0 < fc["log2FC"][0]


def test_trimmed_method_cuts_exactly_one_value_per_end(outlier_matrix):
    df, ctrl, trt = outlier_matrix
    fc = calculate_fold_change(df, CTRL, method="trimmed")
    assert fc["log2FC"].to_numpy() == pytest.approx(
        np.log2((trim_mean(trt, 0.1, axis=1) + 1)
                / (trim_mean(ctrl, 0.1, axis=1) + 1)))
    assert fc["log2FC"][0] != pytest.approx(calculate_fold_change(df, CTRL)["log2FC"][0])


def test_the_three_methods_give_three_different_answers(outlier_matrix):
    """A fixture where every branch coincides would not tell them apart."""
    df, _c, _t = outlier_matrix
    values = [calculate_fold_change(df, CTRL, method=m)["log2FC"].to_numpy()
              for m in ("mean", "median", "trimmed")]
    for row in range(2):
        assert len({round(v[row], 9) for v in values}) == 3


def test_pseudocount_is_the_configurable_one(outlier_matrix):
    df, ctrl, trt = outlier_matrix
    fc = calculate_fold_change(df, CTRL, pseudocount=0.0)
    assert fc["log2FC"].to_numpy() == pytest.approx(
        np.log2(trt.mean(axis=1) / ctrl.mean(axis=1)))


# ── shrinkage ────────────────────────────────────────────────────────────────

def test_shrinkage_matches_its_posterior_formula(outlier_matrix):
    df, _c, _t = outlier_matrix
    plain = calculate_fold_change(df, CTRL)["log2FC"].to_numpy()
    base = np.mean(np.hstack([df.iloc[:, 1:13].values, df.iloc[:, 13:].values]), axis=1)
    shrunk = calculate_fold_change(df, CTRL, shrinkage=True)["log2FC"].to_numpy()
    assert shrunk == pytest.approx(plain * np.clip(base / (base + 2.0), 0.01, 1.0))


def test_shrinkage_never_flattens_a_well_measured_gene():
    """Regression on the direction, which was inverted once: with baseMean used as
    the precision, a 7000-count gene kept only 1% of its log2FC while a 5-count
    gene kept 29%.  The estimate whose variance is actually known should survive.
    """
    df = pd.DataFrame({"gene": ["low", "mid", "high"],
                       "c1": [1.0, 50.0, 5000.0], "c2": [1.0, 50.0, 5000.0],
                       "t1": [9.0, 90.0, 9000.0], "t2": [9.0, 90.0, 9000.0]})
    cond = ["ctrl", "ctrl", "trt", "trt"]
    plain = calculate_fold_change(df, cond)["log2FC"].to_numpy()
    shrunk = calculate_fold_change(df, cond, shrinkage=True)["log2FC"].to_numpy()
    retained = shrunk / plain
    assert retained == pytest.approx([0.714286, 0.972222, 0.999714], rel=1e-6)
    assert retained[0] < retained[1] < retained[2]


def test_shrinkage_only_ever_pulls_toward_zero(outlier_matrix):
    df, _c, _t = outlier_matrix
    plain = calculate_fold_change(df, CTRL)["log2FC"].to_numpy()
    shrunk = calculate_fold_change(df, CTRL, shrinkage=True)["log2FC"].to_numpy()
    assert np.all(np.abs(shrunk) <= np.abs(plain) + 1e-12)
    assert np.all(np.abs(shrunk) < np.abs(plain)), "a 10x gene must shrink"


def test_a_zero_expression_gene_shrinks_to_a_finite_zero():
    df = pd.DataFrame({"gene": ["never"], "c1": [0.0], "c2": [0.0],
                       "t1": [0.0], "t2": [0.0]})
    fc = calculate_fold_change(df, ["ctrl", "ctrl", "trt", "trt"], shrinkage=True)
    assert fc["log2FC"].tolist() == [0.0]
    assert fc["significant"].tolist() == [False]


def test_shrinkage_absorbs_a_missing_count_instead_of_spreading_nan():
    """A half-filled matrix is a real thing to hand this function.  Without
    shrinkage the NaN propagates (the plain estimate simply is not defined), but
    the shrinkage path maps it to 0 so one broken row cannot poison a downstream
    sort of the whole table."""
    df = pd.DataFrame({"gene": ["broken", "fine"], "c1": [np.nan, 4.0], "c2": [1.0, 4.0],
                       "t1": [3.0, 9.0], "t2": [5.0, 9.0]})
    plain = calculate_fold_change(df, ["ctrl", "ctrl", "trt", "trt"])
    assert np.isnan(plain["log2FC"].iloc[0])
    shrunk = calculate_fold_change(df, ["ctrl", "ctrl", "trt", "trt"], shrinkage=True)
    assert shrunk["log2FC"].iloc[0] == 0.0
    assert np.isfinite(shrunk["log2FC"]).all()


# ── the significance flag ────────────────────────────────────────────────────

def test_significant_flag_is_a_strict_cut_at_one_log2_unit():
    df = pd.DataFrame({"gene": ["under", "exactly", "over"],
                       "c": [4.0, 4.0, 4.0], "t": [8.5, 9.0, 9.5]})
    fc = calculate_fold_change(df, ["ctrl", "trt"])
    assert fc["log2FC"].to_numpy() == pytest.approx([0.9259987, 1.0, 1.0703896])
    assert fc["significant"].tolist() == [False, False, True], "|log2FC| == 1 is not > 1"


def test_an_unrecognised_fold_change_method_is_refused(outlier_matrix):
    """'geometric' used to fall through to the arithmetic mean and report a
    different statistic than the caller asked for, without saying so."""
    df, ctrl, trt = outlier_matrix
    with pytest.raises(ValueError,
                       match=r"Unknown method: geometric\. Use 'mean', 'median' or 'trimmed'\."):
        calculate_fold_change(df, CTRL, method="geometric")


def test_gene_labels_come_from_the_index_without_a_gene_column():
    df = pd.DataFrame([[1.0, 2.0, 9.0, 10.0], [2.0, 1.0, 8.0, 7.0]],
                      index=["BRCA1", "TP53"], columns=["c1", "c2", "t1", "t2"])
    assert calculate_fold_change(df, ["ctrl", "ctrl", "trt", "trt"])["gene"].tolist() \
        == ["BRCA1", "TP53"]
    assert calculate_effect_size(df, ["ctrl", "ctrl", "trt", "trt"])["gene"].tolist() \
        == ["BRCA1", "TP53"]


# ── refusals ─────────────────────────────────────────────────────────────────

def test_fold_change_refuses_a_single_group():
    df = pd.DataFrame({"gene": ["a"], "c1": [1.0], "c2": [2.0]})
    with pytest.raises(ValueError, match="Exactly two groups required"):
        calculate_fold_change(df, ["ctrl", "ctrl"])


def test_fold_change_refuses_a_matrix_without_numbers():
    df = pd.DataFrame({"gene": ["a", "b"], "note": ["x", "y"]})
    with pytest.raises(ValueError, match="No numeric columns found."):
        calculate_fold_change(df, ["ctrl", "trt"])


def test_fold_change_refuses_a_label_count_that_does_not_match_the_columns():
    df = pd.DataFrame({"gene": ["a"], "c1": [1.0], "c2": [2.0], "t1": [3.0]})
    with pytest.raises(ValueError, match="they must match"):
        calculate_fold_change(df, ["ctrl", "trt"])


def test_effect_size_refusals():
    df = pd.DataFrame({"gene": ["a"], "c1": [1.0], "t1": [2.0]})
    with pytest.raises(ValueError, match="Exactly two groups required."):
        calculate_effect_size(df, ["ctrl", "ctrl"])
    with pytest.raises(ValueError, match="No numeric columns found."):
        calculate_effect_size(pd.DataFrame({"gene": ["a"], "note": ["x"]}), ["ctrl", "trt"])
    with pytest.raises(ValueError,
                       match=r"Unknown method: bogus\. Use 'cohens_d' or 'rank_biserial'\."):
        calculate_effect_size(df, ["ctrl", "trt"], method="bogus")


# ── effect size ──────────────────────────────────────────────────────────────

@pytest.fixture()
def two_by_two():
    df = pd.DataFrame({"gene": ["up", "flat"],
                       "c1": [1.0, 5.0], "c2": [2.0, 5.0],
                       "t1": [9.0, 5.0], "t2": [11.0, 5.0]})
    return df, ["ctrl", "ctrl", "trt", "trt"]


def test_cohens_d_applies_the_hedges_correction(two_by_two):
    df, cond = two_by_two
    out = calculate_effect_size(df, cond, method="cohens_d")
    mean1 = df[["c1", "c2"]].values.mean(axis=1)
    mean2 = df[["t1", "t2"]].values.mean(axis=1)
    var1 = df[["c1", "c2"]].values.var(axis=1, ddof=1)
    var2 = df[["t1", "t2"]].values.var(axis=1, ddof=1)
    pooled = np.sqrt(np.where(((1 * var1 + 1 * var2) / 2) > 0, (1 * var1 + 1 * var2) / 2, 1.0))
    j = 1 - 3 / (4 * 4 - 9)
    assert out["effect_size"].to_numpy() == pytest.approx((mean2 - mean1) / pooled * j)
    assert out["effect_size"].iloc[0] == pytest.approx(4.344361, rel=1e-6)
    assert out["interpretation"].tolist() == ["large", "negligible"]


def test_rank_biserial_is_scored_from_group_two(two_by_two):
    df, cond = two_by_two
    out = calculate_effect_size(df, cond, method="rank_biserial")
    expected = []
    for i in range(len(df)):
        row = np.concatenate([df.iloc[i, 1:3].values, df.iloc[i, 3:5].values])
        ranks = rankdata(row)
        expected.append(2 * (2 * (4 + 1) / 2 - ranks[:2].sum()) / (2 * 2))
    assert out["effect_size"].to_numpy() == pytest.approx(expected)
    assert out["effect_size"].tolist() == [1.0, 0.0]


def test_the_two_effect_sizes_agree_on_sign(two_by_two):
    """An up-regulated gene is positive under both methods.  The rank-biserial
    branch used to be built from group 1's ranks, so the same gene read +4.34 as
    Cohen's d and -1.0 as rank-biserial, and a caller comparing the two columns
    saw a contradiction."""
    df, cond = two_by_two
    d = calculate_effect_size(df, cond, method="cohens_d")["effect_size"].iloc[0]
    r = calculate_effect_size(df, cond, method="rank_biserial")["effect_size"].iloc[0]
    assert d > 0 and r > 0


def test_cohen_bands_and_their_boundaries():
    """Each row lands on a chosen |g| either side of 0.2 / 0.5 / 0.8.  The probes
    sit a ten-thousandth away from each cut because the comparisons are strict, so
    the boundary value itself belongs to the upper band.  The rows carry a real
    within-group variance (ctrl [0, 2], trt [d, d+2], pooled sd sqrt(2)) rather than
    the constant rows used before, which are no longer measurable as an effect."""
    j = 1 - 3 / (4 * 4 - 9)
    probes = [0.1999, 0.2001, 0.4999, 0.5001, 0.7999, 0.8001]
    expected = ["negligible", "small", "small", "medium", "medium", "large"]
    deltas = [v * np.sqrt(2.0) / j for v in probes]
    df = pd.DataFrame({"gene": [f"t{v}" for v in probes],
                       "c1": [0.0] * len(probes), "c2": [2.0] * len(probes),
                       "t1": deltas, "t2": [d + 2.0 for d in deltas]})
    out = calculate_effect_size(df, ["ctrl", "ctrl", "trt", "trt"])
    assert out["effect_size"].to_numpy() == pytest.approx(probes, abs=1e-9)
    assert out["interpretation"].tolist() == expected


def test_constant_groups_are_only_measurable_when_they_agree():
    """Two constant groups that differ have no finite standardised effect: the
    pooled sd is 0.  The fallback that used to stand in 1.0 there turned 0,0 versus
    5,5 into d = 5 * J = 2.857 and called it 'large', and 10 versus 100 at N = 2
    into 360.  Now the equal case is 0.0 'negligible' and the unequal one is NaN
    'undefined', which no sorting or plotting step can mistake for a number."""
    df = pd.DataFrame({"gene": ["split", "same"], "c1": [0.0, 5.0], "c2": [0.0, 5.0],
                       "t1": [5.0, 5.0], "t2": [5.0, 5.0]})
    out = calculate_effect_size(df, ["ctrl", "ctrl", "trt", "trt"])
    assert np.isnan(out["effect_size"].iloc[0])
    assert out["interpretation"].iloc[0] == "undefined"
    assert out["effect_size"].iloc[1] == 0.0
    assert out["interpretation"].iloc[1] == "negligible"


def test_rank_biserial_bands_and_their_boundaries():
    """r = 2*(E[R1] - R1)/(n1*n2) with n1 = n2 = 2 takes only -1, -0.5, 0,
    0.5 and 1, so a separation this complete is 'large' and an interleaved gene
    is exactly 'negligible'.  Positive means group 2 is the higher one."""
    cases = {   # ctrl pair, trt pair, expected r, expected band
        "complete separation": ((1.0, 2.0), (9.0, 10.0), 1.0, "large"),
        "interleaved": ((1.0, 4.0), (2.0, 3.0), 0.0, "negligible"),
        "reversed": ((9.0, 10.0), (1.0, 2.0), -1.0, "large"),
    }
    for label, (g1, g2, expected, band) in cases.items():
        df = pd.DataFrame({"gene": [label], "c1": [g1[0]], "c2": [g1[1]],
                           "t1": [g2[0]], "t2": [g2[1]]})
        out = calculate_effect_size(df, ["ctrl", "ctrl", "trt", "trt"],
                                    method="rank_biserial")
        assert out["effect_size"].iloc[0] == pytest.approx(expected), label
        assert out["interpretation"].iloc[0] == band, label


def test_a_constant_gene_is_negligible_under_both_methods():
    df = pd.DataFrame({"gene": ["flat", "also_flat"], "c1": [5.0, 2.0], "c2": [5.0, 2.0],
                       "t1": [5.0, 2.0], "t2": [5.0, 2.0]})
    cond = ["ctrl", "ctrl", "trt", "trt"]
    for method in ("cohens_d", "rank_biserial"):
        out = calculate_effect_size(df, cond, method=method)
        assert out["effect_size"].tolist() == [0.0, 0.0], method
        assert out["interpretation"].tolist() == ["negligible", "negligible"], method


def test_cohens_d_refuses_a_group_without_replicates():
    """J = 1 - 3/(4N - 9) is 4.0 at N = 2 and 0.0 at N = 3, outside the domain of
    the approximation, and a ddof=1 variance does not exist on one observation.
    Before the fix 1 vs 1 measured 360.0 'large' for a 10-to-100 change, every
    gene at N = 3 came back 0.0 'negligible', and numpy leaked a
    "Degrees of freedom <= 0 for slice" warning into the caller's console."""
    one_vs_one = pd.DataFrame({"gene": ["a"], "c": [1.0], "t": [9.0]})
    with pytest.raises(ValueError,
                       match=r"needs at least two replicates per group, got 1 vs 1\."):
        calculate_effect_size(one_vs_one, ["ctrl", "trt"])
    one_vs_two = pd.DataFrame({"gene": ["a"], "c": [1.0], "t1": [9.0], "t2": [11.0]})
    with pytest.raises(ValueError, match=r"got 1 vs 2\."):
        calculate_effect_size(one_vs_two, ["ctrl", "trt", "trt"])
    # rank_biserial still works at 1 vs 1: it needs ranks, not variances.
    assert calculate_effect_size(one_vs_one, ["ctrl", "trt"],
                                method="rank_biserial")["effect_size"].tolist() == [1.0]


def test_effect_size_refuses_a_label_count_that_does_not_match_the_columns():
    """Fewer labels than numeric columns used to analyse only the leading columns
    and report a two-column answer as if it were the whole matrix."""
    df = pd.DataFrame({"gene": ["a"], "c1": [1.0], "c2": [2.0], "t1": [3.0]})
    with pytest.raises(ValueError, match="they must match"):
        calculate_effect_size(df, ["ctrl", "trt"])
    with pytest.raises(ValueError, match="they must match"):
        calculate_effect_size(df, ["ctrl", "ctrl", "trt", "trt"])
