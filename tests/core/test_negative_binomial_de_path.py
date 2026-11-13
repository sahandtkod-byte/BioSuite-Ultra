"""`_negative_binomial_test` and `differential_expression(method='nb')`.

The NB path is the one the CLI's "j" and the API expose for count data, and it
had never been executed by a test.  It branches on the observed dispersion:
a gene whose variance does not exceed its mean is treated as Poisson-like and
sent to Fisher's exact test, an over-dispersed gene gets the Wald statistic on
the log2 fold-change, and a gene with no counts at all is left at p = 1.
Each branch is asserted here against an independent reference computation, not
against a constant copied out of the implementation.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact, norm

from biosuite.core.expression import _negative_binomial_test, differential_expression


def wald_reference(vals1, vals2):
    """The documented NB/Wald arithmetic, recomputed straight from the formulas."""
    counts = np.concatenate([vals1[0], vals2[0]])
    mean, var = counts.mean(), counts.var()
    size = max(mean ** 2 / (var - mean), 0.1)
    m1 = vals1[0].mean() + 0.5
    m2 = vals2[0].mean() + 0.5
    log2fc = np.log2(m2 / m1)
    v1 = m1 + m1 ** 2 / size
    v2 = m2 + m2 ** 2 / size
    se = np.sqrt(v1 / (vals1.shape[1] * m1 ** 2) + v2 / (vals2.shape[1] * m2 ** 2))
    return 2 * (1 - norm.cdf(abs(log2fc / se)))


def fisher_reference(vals1, vals2):
    """Fisher's exact p on the median-split table, with the implementation's
    floor of 1 in every cell."""
    counts = np.concatenate([vals1[0], vals2[0]])
    median = np.median(counts)
    above1 = int((vals1[0] >= median).sum())
    above2 = int((vals2[0] >= median).sum())
    table = [[max(above1, 1), max(vals1.shape[1] - above1, 1)],
             [max(above2, 1), max(vals2.shape[1] - above2, 1)]]
    return table, fisher_exact(table)[1]


# ── branch selection ─────────────────────────────────────────────────────────

def test_a_gene_with_no_counts_anywhere_is_reported_as_no_evidence():
    p = _negative_binomial_test(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]))
    assert p == [1.0]


def test_under_dispersed_gene_goes_to_fisher_not_the_wald_test():
    """var <= mean is Poisson-like, which is exactly where the NB/Wald model is
    the wrong tool; the reference table and p are recomputed here."""
    v1 = np.array([[4.0, 5.0, 6.0]])
    v2 = np.array([[6.0, 7.0, 8.0]])
    counts = np.concatenate([v1[0], v2[0]])
    assert counts.var() <= counts.mean()
    table, expected = fisher_reference(v1, v2)
    assert table == [[1, 2], [3, 1]]
    assert _negative_binomial_test(v1, v2) == pytest.approx([expected])
    assert expected == pytest.approx(0.4857142857, abs=1e-9)


def test_identical_groups_cannot_be_called_significant():
    v1 = np.array([[5.0, 5.0, 5.0]])
    v2 = np.array([[5.0, 5.0, 5.0]])
    assert _negative_binomial_test(v1, v2) == [1.0]


def test_over_dispersed_gene_uses_the_wald_statistic():
    v1 = np.array([[2.0, 3.0, 2.0]])
    v2 = np.array([[50.0, 60.0, 55.0]])
    counts = np.concatenate([v1[0], v2[0]])
    assert counts.var() > counts.mean()
    p = _negative_binomial_test(v1, v2)
    assert p == pytest.approx([wald_reference(v1, v2)], rel=1e-12)
    assert p[0] < 1e-5, "a 2 -> 55 jump must be significant"


def test_a_zero_replicate_group_still_produces_a_finite_p():
    """One group entirely unsequenced: the pseudocounts keep the Wald statistic
    finite rather than dividing by a zero mean."""
    v1 = np.array([[0.0, 0.0]])
    v2 = np.array([[7.0, 9.0]])
    p = _negative_binomial_test(v1, v2)
    assert np.isfinite(p).all()
    assert p == pytest.approx([wald_reference(v1, v2)], rel=1e-12)


def test_genes_are_tested_independently_and_in_order():
    v1 = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 3.0]])
    v2 = np.array([[0.0, 0.0], [6.0, 7.0], [50.0, 60.0]])
    p = _negative_binomial_test(v1, v2)
    assert p.shape == (3,)
    single = [_negative_binomial_test(v1[i:i + 1], v2[i:i + 1])[0] for i in range(3)]
    assert p == pytest.approx(single)
    assert p[0] == 1.0
    assert p[2] < p[1], "the stronger separation must get the smaller p"


def test_a_matrix_that_is_not_counts_cannot_claim_significance():
    """Negative values are not counts.  They drive the standard error to NaN, and
    the fallback must report p = 1 — no evidence — instead of letting a NaN reach
    the multiple-testing correction."""
    v1 = np.array([[-3.0, -2.0]])
    v2 = np.array([[-1.0, 0.0]])
    with np.errstate(all="ignore"):
        p = _negative_binomial_test(v1, v2)
    assert np.isfinite(p).all()
    assert p == [1.0]


def test_pseudocounts_are_added_before_the_log_ratio():
    """A 0 -> 7 gene would be log2(inf) without them; the reference recomputes
    the same quantity from the definition."""
    v1 = np.array([[0.0, 0.0]])
    v2 = np.array([[7.0, 9.0]])
    m1, m2 = v1[0].mean() + 0.5, v2[0].mean() + 0.5
    assert np.log2(m2 / m1) == pytest.approx(4.0874, abs=1e-4)
    assert np.isfinite(_negative_binomial_test(v1, v2)).all()


# ── through the public entry point ──────────────────────────────────────────

@pytest.fixture()
def mixed_matrix():
    """Four genes, three replicates a group, one gene per NB branch."""
    return pd.DataFrame({
        "gene": ["zero", "eq", "over", "mid"],
        "c1": [0, 4, 2, 30], "c2": [0, 5, 3, 32], "c3": [0, 6, 2, 28],
        "t1": [0, 6, 50, 31], "t2": [0, 7, 60, 33], "t3": [0, 8, 55, 29],
    }), ["ctrl"] * 3 + ["trt"] * 3


def test_nb_method_delegates_to_the_nb_test(mixed_matrix):
    df, conditions = mixed_matrix
    de = differential_expression(df, conditions, method="nb", correction="none")
    direct = _negative_binomial_test(df.iloc[:, 1:4].values.astype(float),
                                     df.iloc[:, 4:7].values.astype(float))
    assert de["pvalue"].to_numpy() == pytest.approx(direct)
    assert de["padj"].to_numpy() == pytest.approx(direct), "correction='none' is a copy"


def test_nb_method_applies_bh_by_default(mixed_matrix):
    df, conditions = mixed_matrix
    de = differential_expression(df, conditions, method="nb")
    assert np.all(de["padj"].to_numpy() >= de["pvalue"].to_numpy() - 1e-12)
    assert de["padj"].max() <= 1.0
    assert de["gene"].tolist() == ["zero", "eq", "over", "mid"]
    assert de.loc[de["gene"] == "over", "padj"].iloc[0] < 0.05
    assert de.loc[de["gene"] == "zero", "padj"].iloc[0] == 1.0


def test_fold_change_column_is_the_same_whichever_test_is_used(mixed_matrix):
    """log2FC comes from the means with a +1 pseudocount and does not depend on
    which statistical test was asked for."""
    df, conditions = mixed_matrix
    nb = differential_expression(df, conditions, method="nb")
    tt = differential_expression(df, conditions, method="ttest")
    assert nb["log2FC"].to_numpy() == pytest.approx(tt["log2FC"].to_numpy())
    manual = np.log2(np.mean(df.iloc[:, 4:7].values + 1, axis=1)
                     / np.mean(df.iloc[:, 1:4].values + 1, axis=1))
    assert nb["log2FC"].to_numpy() == pytest.approx(manual)


def test_nb_path_reports_no_nan_pvalues(mixed_matrix):
    """The t-test path returns NaN for degenerate genes on purpose; the NB path
    fills every gene, so a caller filtering on padj sees one row per gene."""
    df, conditions = mixed_matrix
    assert np.isfinite(differential_expression(df, conditions, method="nb")["pvalue"]).all()


def test_unknown_method_is_refused_naming_the_two_valid_ones():
    df = pd.DataFrame({"gene": ["a"], "c": [1.0], "t": [2.0]})
    with pytest.raises(ValueError, match=r"Unknown method: nope\. Use 'ttest' or 'nb'\."):
        differential_expression(df, ["ctrl", "trt"], method="nope")


def test_a_flat_gene_is_not_called_significant_by_either_method():
    """A gene that does not move at all: the NB branch (equi-dispersed, so
    Fisher) and the Welch branch must both refuse to call it differential."""
    df = pd.DataFrame({"gene": ["flat"], "c1": [5.0], "c2": [5.0], "c3": [5.0],
                       "t1": [5.0], "t2": [5.0], "t3": [5.0]})
    cond = ["ctrl"] * 3 + ["trt"] * 3
    for method in ("nb", "ttest"):
        de = differential_expression(df, cond, method=method, correction="none")
        assert de["pvalue"].iloc[0] > 0.05, method
        assert de["log2FC"].iloc[0] == pytest.approx(0.0)
