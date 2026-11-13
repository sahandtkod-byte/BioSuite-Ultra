"""What each ``plot_api`` matplotlib plot really puts on its axes.

Every function here used to be tested with ``assert fig is not None``, which a blank
figure satisfies.  These tests read the artists: how many collections a volcano
plot splits its genes into and *which* genes went into each, the two boxes a
boxplot draws, the heights behind a barplot's bars, the array a heatmap coloured,
the label each series brought to a time series.  The input data is fixed by a
seeded generator, so an assertion that fails means the drawing changed, not that
the random numbers did.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from biosuite.plotting import plot_api as pa

RNG = np.random.default_rng
LOG2FC = RNG(0).normal(0, 2, 100)
PVALUES = 10 ** RNG(0).uniform(-6, -0.5, 100)
GENES = [f"G{i}" for i in range(100)]
FEATURES = RNG(1).normal(size=(20, 5))
CHROMS = RNG(2).integers(1, 4, 200)
POSITIONS = RNG(2).integers(1, 10_000, 200)
GWAS_P = 10 ** RNG(2).uniform(-8, -1, 200)
MEAN_EXPR = RNG(3).uniform(100, 10_000, 200)
LOG_FC = RNG(3).normal(0, 1.5, 200)
MATRIX = np.arange(25).reshape(5, 5) / 10


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _points(collection):
    return np.asarray(collection.get_offsets())


def _all_points(ax):
    return np.vstack([_points(c) for c in ax.collections if len(c.get_offsets())])


# ── differential expression ──────────────────────────────────────────────────

def test_the_volcano_split_is_the_cutoff_the_caller_asked_for():
    fig = pa.volcano(LOG2FC, PVALUES, gene_names=GENES, fc_thresh=1.0, p_thresh=0.05)
    ax = fig.axes[0]

    significant, other = [len(_points(c)) for c in ax.collections]
    expected = int(np.sum((np.abs(LOG2FC) >= 1.0) & (PVALUES < 0.05)))

    assert significant + other == 100, "every gene is drawn in one of the two groups"
    assert max(significant, other) == 100 - expected, "the significant count is the cutoff's own"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Not sig", "Significant"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Log2 Fold Change", "-log10(P-value)")
    assert len(ax.lines) == 3, "one significance line and the two fold-change walls"


def test_the_volcano_puts_each_point_at_its_own_fold_change_and_probability():
    fig = pa.volcano(LOG2FC, PVALUES, gene_names=GENES)
    drawn = _all_points(fig.axes[0])

    assert sorted(drawn[:, 0].tolist()) == pytest.approx(sorted(LOG2FC.tolist())), \
        "the x coordinate is the fold change, not an index"
    heights = -np.log10(PVALUES)
    assert sorted(drawn[:, 1].tolist()) == pytest.approx(sorted(heights.tolist()))


def test_the_ma_plot_separates_on_the_fold_change_it_was_given():
    fig = pa.ma(MEAN_EXPR, LOG_FC, fc_thresh=1.0)
    ax = fig.axes[0]

    sizes = sorted(len(_points(c)) for c in ax.collections)
    significant = int(np.sum(np.abs(LOG_FC) >= 1.0))

    assert sum(sizes) == 200
    assert sizes == sorted([significant, 200 - significant]), \
        "the two groups are exactly the fold-change split, whichever is smaller"
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.0), "the fold-change zero line"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Mean Expression (log)", "Log2 Fold Change")


# ── dimensionality reduction ─────────────────────────────────────────────────

def test_the_pca_axes_quote_the_variance_the_same_decomposition_explains():
    frame = pd.DataFrame(FEATURES, columns=[f"f{i}" for i in range(5)])
    expected = PCA(n_components=2).fit_transform(frame.values)
    ratios = PCA(n_components=2).fit(frame.values).explained_variance_ratio_

    fig = pa.pca(frame, labels=[f"S{i}" for i in range(20)])
    ax = fig.axes[0]

    assert len(ax.collections) == 1, "a frame with no grouping column draws one cloud"
    assert len(_points(ax.collections[0])) == 20
    assert ax.get_xlabel() == f"PC1 ({ratios[0] * 100:.1f}%)"
    assert ax.get_ylabel() == f"PC2 ({ratios[1] * 100:.1f}%)"
    assert np.allclose(np.sort(_points(ax.collections[0]), axis=0),
                       np.sort(expected, axis=0))


# ── genomics ─────────────────────────────────────────────────────────────────

def test_manhattan_lays_one_block_of_points_per_chromosome_in_order():
    fig = pa.manhattan(CHROMS, POSITIONS, GWAS_P)
    ax = fig.axes[0]

    blocks = [np.asarray(_points(c))[:, 0] for c in ax.collections]
    assert len(blocks) == len(set(CHROMS.tolist())), "one collection per chromosome"
    assert all(later.min() > earlier.max() for earlier, later in zip(blocks, blocks[1:])), \
        "chromosome blocks do not overlap on the axis"
    assert sum(len(b) for b in blocks) == 200
    assert ax.get_ylabel() == "-log10(p)"


def test_the_qq_plot_is_a_normal_quantile_plot_of_the_p_values_themselves():
    """Expected and observed are both on the probit scale, so small p-values sit
    far below zero — this is a normality check, not a -log10 Manhattan scale."""
    from scipy import stats as sp_stats

    pvalues = sorted(10 ** RNG(5).uniform(-6, -0.5, 100))
    fig = pa.qqplot(pvalues)
    drawn = np.asarray(_points(fig.axes[0].collections[0]))

    assert drawn.shape == (len(pvalues), 2)
    assert drawn[:, 1] == pytest.approx(sp_stats.norm.ppf(
        np.clip(pvalues, 1e-16, 1 - 1e-16)))
    assert drawn[:, 0] == pytest.approx(
        sp_stats.norm.ppf(np.arange(1, len(pvalues) + 1) / (len(pvalues) + 1)))
    assert np.all(np.diff(drawn[:, 1]) >= 0), "observed keeps the order of the input"
    assert len(fig.axes[0].lines) == 1, "the diagonal reference"


# ── tables and groups ────────────────────────────────────────────────────────

def test_the_heatmap_colours_the_matrix_it_was_handed_and_labels_both_sides():
    rows = [f"r{i}" for i in range(5)]
    cols = [f"c{i}" for i in range(5)]

    fig = pa.heatmap(MATRIX, row_labels=rows, col_labels=cols)

    ax = fig.axes[0]
    assert len(fig.axes) == 2, "the colourbar comes with it"
    assert np.array_equal(ax.images[0].get_array().reshape(5, 5), MATRIX)
    assert [t.get_text() for t in ax.get_yticklabels()] == rows
    assert [t.get_text() for t in ax.get_xticklabels()] == cols


def test_the_boxplot_draws_one_box_per_group_in_the_order_of_the_dict():
    fig = pa.boxplot({"a": [1.0, 2.0, 3.0, 2.5], "b": [2.0, 3.0, 4.0, 2.2]})

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    assert ax.get_ylabel() == "Value"
    assert ax.get_ylim()[1] >= 4.0, "the tallest whisker has to fit"


def test_the_barplot_bars_carry_the_values_and_the_error_caps_appear():
    plain = pa.barplot(["a", "b", "c"], [3.0, 7.0, 2.0])
    with_caps = pa.barplot(["a", "b", "c"], [3.0, 7.0, 2.0], errors=[0.2, 0.4, 0.2])

    bars = [p for p in with_caps.axes[0].patches if type(p).__name__ == "Rectangle"]
    assert [round(float(b.get_height()), 3) for b in bars] == [3.0, 7.0, 2.0]
    assert len(with_caps.axes[0].lines) == len(plain.axes[0].lines) + 2, \
        "the caps are the only difference the errors make"
    assert with_caps.axes[0].get_ylabel() == "Value"


def test_the_violin_draws_one_shape_per_group_named_on_the_axis():
    groups = {"g1": RNG(4).normal(0, 1, 50).tolist(),
              "g2": RNG(4).normal(1.5, 1, 50).tolist()}

    fig = pa.violin(groups)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["g1", "g2"]
    assert len(ax.collections) >= 2, "each group needs a body to fill"
    assert ax.get_ylabel() == "Value"


def test_a_scatter_annotates_the_fit_of_a_line_the_points_actually_follow():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]

    # Four points, four labels: the fixture used to hand over three names for
    # four points, which the matplotlib path accepted because it never reads
    # labels at all — the mismatch only mattered for the hover text.
    fig = pa.scatter(x, y, labels=["a", "b", "c", "d"],
                     xlabel="Dose", ylabel="Response")

    ax = fig.axes[0]
    assert np.array_equal(_points(ax.collections[0]), np.column_stack([x, y]))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "R\u00b2=1.000, p=1.00e-20"], \
        "a perfect line is reported as R\u00b2 = 1.000 (the p is scipy's floor, not zero)"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Dose", "Response")


def test_a_timeseries_gives_every_series_its_own_name_and_values():
    series = [np.arange(10) * i for i in (1, 2, 3)]

    fig = pa.timeseries(np.arange(10), series, names=["s1", "s2", "s3"])

    ax = fig.axes[0]
    assert [line.get_label() for line in ax.lines[:3]] == ["s1", "s2", "s3"]
    for line, scaled in zip(ax.lines[:3], series):
        assert np.array_equal(np.asarray(line.get_ydata()), scaled)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["s1", "s2", "s3"]


def test_a_two_set_venn_labels_every_region_it_was_given():
    fig = pa.venn([50, 40, 15], set_names=["X", "Y"])

    ax = fig.axes[0]
    assert len(ax.patches) == 2, "two circles"
    texts = [t.get_text() for t in ax.texts]
    assert "50" in texts and "40" in texts and "15" in texts, texts
    assert "X" in texts and "Y" in texts, texts
