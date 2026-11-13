"""Biological demo plots: default-data and typed-input flows, judged on artefacts.

Each demo in ``biological_plots`` catches its own exceptions, so a run that
failed part-way raises nothing; proof that a flow worked is the PNG it wrote and
the figure it released (see ``conftest.run_demo``).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import biological_plots as bp

#: Demos that open with "Load data from file?" and fall back to synthetic data.
SYNTHETIC_DEMOS = [
    ("volcano_plot", "volcano.png"),
    ("pca_plot", "pca.png"),
    ("manhattan_plot", "manhattan.png"),
    ("ma_plot", "maplot.png"),
    ("barplot_custom", "barplot.png"),
    ("boxplot_custom", "boxplot.png"),
    ("heatmap_custom", "heatmap.png"),
    ("scatter_custom", "scatter.png"),
    ("timeseries_plot", "timeseries.png"),
    ("qq_plot", "qqplot.png"),
    ("clustered_heatmap", "clustered_heatmap.png"),
    ("violin_plot", "violin.png"),
    ("raincloud_plot", "raincloud.png"),
    ("ridge_plot", "ridge.png"),
    ("dot_plot", "dotplot.png"),
]


def _assert_saved(report, filename):
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == [filename], report["stdout"]
    assert report["empty_images"] == []
    assert report["figures_open"] == 0, f"{filename} left its figure pinned"


@pytest.mark.parametrize("name,filename", SYNTHETIC_DEMOS)
def test_synthetic_data_flow_saves_plot(run_demo, name, filename):
    report = run_demo(getattr(bp, name), answers={"load data from file": "n"})
    _assert_saved(report, filename)


@pytest.mark.parametrize("n_sets", ["2", "3"])
def test_venn_diagram_saves_plot(run_demo, n_sets):
    report = run_demo(bp.venn_diagram, answers={"number of sets": n_sets})
    _assert_saved(report, "venn.png")


def test_venn_diagram_refuses_other_set_counts(run_demo):
    report = run_demo(bp.venn_diagram, answers={"number of sets": "4"})
    assert "Only 2 or 3 sets supported." in report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0


def test_circos_default_data_saves_plot(run_demo):
    report = run_demo(bp.circos_plot, answers={"use default data": "y"})
    _assert_saved(report, "circos.png")


def test_circos_typed_sectors_and_links_save_plot(run_demo):
    report = run_demo(bp.circos_plot, answers={
        "use default data": "n",
        "sector 1 name": "Alpha",
        "sector 2 name": "Beta",
        "sector 3 name": "Gamma",
        "link 1 from sector": "0",
        "link 2 from sector": "1",
        "to sector": "1",
    })
    _assert_saved(report, "circos.png")


def test_alignment_viewer_default_alignment_saves_plot(run_demo):
    report = run_demo(bp.alignment_viewer, answers={"use default alignment": "y"})
    _assert_saved(report, "alignment.png")


def test_alignment_viewer_typed_sequences_save_plot(run_demo):
    report = run_demo(bp.alignment_viewer, answers={
        "use default alignment": "n",
        "<blank>": ["ACGT", "ACGA", "ACGT", ""],
    })
    _assert_saved(report, "alignment.png")


@pytest.fixture
def saved_figures(monkeypatch):
    """Grab each figure a demo hands to ``ask_save_plot``.

    ``plt.close`` only unregisters a figure, so the artists keep their data and
    can be inspected after the demo has finished and released them.
    """
    figures = []
    real = bp.ask_save_plot

    def spy(default_name, save_format, dpi, pdf=None, story=None):
        figures.append(plt.gcf())
        return real(default_name, save_format, dpi, pdf, story)

    monkeypatch.setattr(bp, "ask_save_plot", spy)
    return figures


def test_ma_plot_unreadable_file_falls_back_to_synthetic_data(run_demo):
    report = run_demo(bp.ma_plot, answers={
        "load data from file": "y",
        "file path": "does-not-exist.csv",
    })
    _assert_saved(report, "maplot.png")
    assert "Could not read the file" in report["stdout"]


def test_ma_plot_uses_the_chosen_expression_columns(run_demo, saved_figures, tmp_path):
    m1 = np.array([1.0, 2.0, 3.0, 4.0])
    m2 = np.array([4.0, 3.0, 2.0, 1.0])
    csv = tmp_path / "expr.csv"
    pd.DataFrame({"m1": m1, "m2": m2}).to_csv(csv, index=False)
    report = run_demo(bp.ma_plot, answers={
        "load data from file": "y",
        "file path": str(csv),
        "first expression column": "m1",
        "second expression column": "m2",
    })

    assert report["images"] == ["maplot.png"]
    # A is the mean log intensity of the two columns, so the file — not the
    # synthetic default — decides where the points land.
    points = np.asarray(saved_figures[0].axes[0].collections[0].get_offsets())
    assert np.allclose(points[:, 0], (np.log2(m1 + 1) + np.log2(m2 + 1)) / 2)
    assert np.allclose(points[:, 1], np.log2(m2 + 1) - np.log2(m1 + 1))


# ── drawing helpers: non-interactive, so assert on what lands on the axes ─────

def test_draw_venn2_labels_regions_and_sets():
    ax = bp.draw_venn2([10, 15, 4], set_labels=("Set A", "Set B"))
    rendered = [t.get_text() for t in ax.texts]
    assert len(ax.patches) == 2
    assert ["10", "15", "4"] == rendered[:3]
    assert ["Set A", "Set B"] == rendered[3:]


def test_draw_venn3_labels_regions_and_sets():
    ax = bp.draw_venn3([8, 8, 8, 3, 3, 3, 1], set_labels=("X", "Y", "Z"))
    rendered = [t.get_text() for t in ax.texts]
    assert len(ax.patches) == 3
    assert rendered[:7] == ["8", "8", "8", "3", "3", "3", "1"]
    assert rendered[7:] == ["X", "Y", "Z"]


def test_draw_motif_logo_renders_one_block_per_base():
    seqs = ["AAGT", "AAGT", "CAGT", "ACGT"]
    ax = bp.draw_motif_logo(seqs)
    assert len(ax.patches) == len(seqs) * len(seqs[0])
    assert {type(p).__name__ for p in ax.patches} == {"Rectangle"}


def test_draw_sankey_renders_nodes_and_links():
    from matplotlib.patches import ConnectionPatch

    ax = bp.draw_sankey(["A", "B", "C"], [0, 1], [1, 2], [10, 20])
    links = sum(isinstance(p, ConnectionPatch) for p in ax.patches)
    assert links == 2
    # A node is boxed once per column it appears in, so B (target of A, source
    # of C) is drawn twice: 3 distinct labels give 4 boxes.
    assert len(ax.patches) - links == 4
    assert {t.get_text() for t in ax.texts} == {"A", "B", "C"}
