"""The demos' other input: a real CSV, not the synthetic curve.

Every demo in ``biological_plots`` can read the user's own table, and that branch is
where the data-integrity bugs live.  ``volcano``, ``scatter``, ``barplot`` and
``timeseries`` each cleaned their two columns with an independent ``dropna()``, so a
single blank cell dropped one value from one column only and paired every remaining
point with its neighbour's; ``timeseries`` then compared the two lengths and, when
they differed, drew the demo curve without saying so.  ``ridge_plot`` aborted
entirely on a group with one value, because a kernel density needs two.

The oracle is the figure itself: ``ask_save_plot`` is spied on at the moment each
demo is about to save, so the tests read the coordinates, labels and titles that
were actually drawn instead of trusting a printed message.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import biological_plots as bp

#: Four complete genes and one row with no fold change.
VOLCANO_CSV = """log2fc,pvalue
1.5,0.001
-2.0,0.02
0.1,0.9
,0.5
2.5,0.0001
"""
VOLCANO_PAIRS = [(1.5, 0.001), (-2.0, 0.02), (0.1, 0.9), (2.5, 0.0001)]

#: Six rows, one with no y — the pair, not the column, has to go.
SCATTER_CSV = """peak,trough
1,2
2,4
3,6
4,
5,10
6,12
"""

#: ``het`` has a single value and ``mut`` one blank row.
GROUPED_CSV = """group,value
wt,1
wt,2
mut,5
mut,
mut,7
het,3
"""


@pytest.fixture
def saved_figure(monkeypatch):
    """Capture the figure each demo is about to save, before the saver closes it."""
    captured = {}
    real = bp.ask_save_plot

    def spy(default_name, save_format, dpi, pdf=None, story=None):
        captured["name"] = default_name
        captured["fig"] = plt.gcf()
        return real(default_name, save_format, dpi, pdf, story)

    monkeypatch.setattr(bp, "ask_save_plot", spy)
    return captured


def _drawn_points(figure):
    """Every (x, y) point the scatter-style calls placed on the first axes."""
    arrays = [np.asarray(c.get_offsets()) for c in figure.axes[0].collections]
    arrays = [a for a in arrays if a.size]
    return np.vstack(arrays) if arrays else np.empty((0, 2))


def _tick_labels(axes, which):
    return [t.get_text() for t in getattr(axes, f"get_{which}ticklabels")() if t.get_text()]


def _bars(axes):
    """(slot, height) for each seaborn bar; other patches are error-bar marks."""
    return [(round(p.get_x() + p.get_width() / 2), round(p.get_height(), 3))
            for p in axes.patches if p.get_width() < 0.9]


# ── volcano ──────────────────────────────────────────────────────────────────

def test_volcano_keeps_each_genes_pair_when_a_value_is_missing(run_demo, tmp_path,
                                                              saved_figure):
    (tmp_path / "de.csv").write_text(VOLCANO_CSV, encoding="utf-8")

    report = run_demo(bp.volcano_plot, answers={
        "load data from file": "y", "file path": "de.csv",
        "log2fc column": "log2fc", "p-value column": "pvalue",
        "fold-change threshold": "", "p-value threshold": "",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    points = _drawn_points(saved_figure["fig"])
    assert len(points) == len(VOLCANO_PAIRS), \
        f"the four complete rows must survive, {len(points)} points were drawn"
    for (x, p), (dx, dy) in zip(sorted(VOLCANO_PAIRS), sorted(points.tolist())):
        assert dx == pytest.approx(x)
        assert dy == pytest.approx(-np.log10(p + 1e-300), abs=1e-6)
    assert report["images"] == ["volcano.png"]


def test_volcano_falls_back_when_no_column_matches(run_demo, tmp_path):
    (tmp_path / "de.csv").write_text(VOLCANO_CSV, encoding="utf-8")

    report = run_demo(bp.volcano_plot, answers={
        "load data from file": "y", "file path": "de.csv",
        "log2fc column": "nope", "p-value column": "pvalue",
        "fold-change threshold": "", "p-value threshold": "",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["volcano.png"], "the synthetic fallback still plots"


# ── scatter ──────────────────────────────────────────────────────────────────

def test_scatter_pairs_x_and_y_row_by_row(run_demo, tmp_path, saved_figure):
    (tmp_path / "xy.csv").write_text(SCATTER_CSV, encoding="utf-8")

    report = run_demo(bp.scatter_custom, answers={
        "load data from file": "y", "correlation type": "pearson",
        "file path": "xy.csv", "x column": "peak", "y column": "trough",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    points = _drawn_points(saved_figure["fig"])
    assert sorted(p[0] for p in points) == [1.0, 2.0, 3.0, 5.0, 6.0]
    for x, y in points:
        assert y == pytest.approx(2 * x), f"({x}, {y}) is not the row the file has"
    assert "Pearson r = 1.000" in report["stdout"]


def test_scatter_reports_a_spearman_rho_when_asked(run_demo, tmp_path):
    (tmp_path / "xy.csv").write_text(SCATTER_CSV, encoding="utf-8")

    report = run_demo(bp.scatter_custom, answers={
        "load data from file": "y", "correlation type": "spearman",
        "file path": "xy.csv", "x column": "peak", "y column": "trough",
    })

    assert "Spearman rho = 1.000" in report["stdout"]


# ── barplot ──────────────────────────────────────────────────────────────────

def test_barplot_keeps_labels_with_their_heights(run_demo, tmp_path, saved_figure):
    (tmp_path / "bars.csv").write_text(
        "sample,reads,err\nctrl,10,\ntrt,20,1.5\nblank,\ntrt2,30,2\n", encoding="utf-8")

    report = run_demo(bp.barplot_custom, answers={
        "load data from file": "y", "file path": "bars.csv",
        "category column": "sample", "value column": "reads", "error column": "err",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    axes = saved_figure["fig"].axes[0]
    assert _tick_labels(axes, "x") == ["ctrl", "trt", "trt2"]
    assert _bars(axes) == [(0, 10.0), (1, 20.0), (2, 30.0)], "label and height came apart"


# ── timeseries ───────────────────────────────────────────────────────────────

def test_timeseries_draws_the_file_not_the_demo_curve(run_demo, tmp_path, saved_figure):
    (tmp_path / "ts.csv").write_text(
        "hour,od\n0,1.0\n1,1.4\n2,\n3,2.1\n4,2.6\n", encoding="utf-8")

    report = run_demo(bp.timeseries_plot, answers={
        "load data from file": "y", "file path": "ts.csv",
        "time column": "hour", "value column": "od",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    line = saved_figure["fig"].axes[0].lines[0]
    assert list(line.get_xdata()) == [0, 1, 3, 4], "a blank row must take its pair with it"
    assert "Using defaults" not in report["stdout"]


def test_timeseries_says_so_when_the_file_has_one_point(run_demo, tmp_path):
    (tmp_path / "ts.csv").write_text("hour,od\n0,\n1,1.4\n", encoding="utf-8")

    report = run_demo(bp.timeseries_plot, answers={
        "load data from file": "y", "file path": "ts.csv",
        "time column": "hour", "value column": "od",
    })

    assert "fewer than two complete time/value pairs" in report["stdout"]
    assert report["images"] == ["timeseries.png"]


# ── manhattan ────────────────────────────────────────────────────────────────

def test_manhattan_uses_the_chromosomes_it_was_given(run_demo, tmp_path, saved_figure):
    rows = "\n".join(f"chr{1 + i % 3},{1000 + 500 * i},{10 ** -(1 + i)}"
                     for i in range(9))
    (tmp_path / "gwas.csv").write_text(f"chromosome,position,p\n{rows}\n", encoding="utf-8")

    report = run_demo(bp.manhattan_plot, answers={
        "load data from file": "y", "file path": "gwas.csv",
        "chromosome column": "chromosome", "position column": "position",
        "p-value column": "p",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert set(_tick_labels(saved_figure["fig"].axes[0], "x")) == {"chr1", "chr2", "chr3"}
    assert len(_drawn_points(saved_figure["fig"])) == 9


def test_manhattan_names_the_columns_it_could_not_use(run_demo, tmp_path):
    (tmp_path / "gwas.csv").write_text("chromosome,position,p\nchr1,10,0.5\n",
                                       encoding="utf-8")

    report = run_demo(bp.manhattan_plot, answers={
        "load data from file": "y", "file path": "gwas.csv",
        "chromosome column": "nope", "position column": "position",
        "p-value column": "p",
    })

    assert "Invalid columns. Using defaults." in report["stdout"]
    assert report["images"] == ["manhattan.png"]


# ── MA, heatmap, QQ ──────────────────────────────────────────────────────────

def test_ma_plot_uses_the_two_expression_columns(run_demo, tmp_path, saved_figure):
    (tmp_path / "two.csv").write_text("ch1,ch2\n10,20\n30,30\n40,20\n", encoding="utf-8")

    report = run_demo(bp.ma_plot, answers={
        "load data from file": "y", "file path": "two.csv",
        "first expression column": "ch1", "second expression column": "ch2",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    points = _drawn_points(saved_figure["fig"])
    assert len(points) == 3
    assert max(points[:, 0]) == pytest.approx(np.log2(31), abs=1e-9)
    # The +1 pseudocount is part of the definition: 20 over 10 is not a clean 2x.
    assert points[:, 1].max() == pytest.approx(np.log2(21) - np.log2(11), abs=1e-9)
    assert points[:, 1].min() == pytest.approx(np.log2(21) - np.log2(41), abs=1e-9)


def test_ma_plot_falls_back_when_the_file_has_no_such_columns(run_demo, tmp_path):
    (tmp_path / "two.csv").write_text("ch1,ch2\n10,20\n", encoding="utf-8")

    report = run_demo(bp.ma_plot, answers={
        "load data from file": "y", "file path": "two.csv",
        "first expression column": "nope", "second expression column": "ch2",
    })

    assert "Invalid columns. Using defaults." in report["stdout"]
    assert "Error:" not in report["stdout"]
    assert report["images"] == ["maplot.png"]


def test_heatmap_titles_a_correlation_matrix_from_the_file(run_demo, tmp_path,
                                                           saved_figure):
    (tmp_path / "mat.csv").write_text("a,b,c\n1,2,3\n4,5,6\n7,8,10\n", encoding="utf-8")

    report = run_demo(bp.heatmap_custom, answers={
        "load data from file": "y", "file path": "mat.csv",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert saved_figure["fig"].axes[0].get_title() == "Correlation Heatmap"
    assert report["images"] == ["heatmap.png"]


def test_qq_plot_tests_the_column_that_was_named(run_demo, tmp_path, saved_figure):
    (tmp_path / "vals.csv").write_text(
        "value,other\n0.1,9\n0.4,8\n0.6,7\n1.2,6\n2.0,5\n", encoding="utf-8")

    report = run_demo(bp.qq_plot, answers={
        "load data from file": "y", "file path": "vals.csv", "column to test": "value",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    sample = saved_figure["fig"].axes[0].lines[0].get_ydata()
    assert len(sample) == 5, "the demo draws 200 points"
    assert sorted(sample) == pytest.approx([0.1, 0.4, 0.6, 1.2, 2.0])


# ── grouped distributions ────────────────────────────────────────────────────

@pytest.mark.parametrize("demo,filename,title", [
    ("boxplot_custom", "boxplot.png", "Boxplot"),
    ("violin_plot", "violin.png", "Violin Plot with Quartiles"),
    ("raincloud_plot", "raincloud.png", "Raincloud Plot"),
])
def test_grouped_demos_plot_the_groups_from_the_file(run_demo, tmp_path, saved_figure,
                                                     demo, filename, title):
    (tmp_path / "groups.csv").write_text(GROUPED_CSV, encoding="utf-8")

    report = run_demo(getattr(bp, demo), answers={
        "load data from file": "y", "file path": "groups.csv",
        "value column": "value", "group column": "group",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    axes = saved_figure["fig"].axes[0]
    assert axes.get_title() == title
    assert _tick_labels(axes, "x") == ["wt", "mut", "het"]
    assert report["images"] == [filename]


def test_ridge_plot_draws_the_groups_a_density_can_fit(run_demo, tmp_path, saved_figure):
    """``het`` has one value: it cannot have a density, but it must not take the
    other two groups down with it — which is what it used to do."""
    (tmp_path / "groups.csv").write_text(GROUPED_CSV, encoding="utf-8")

    report = run_demo(bp.ridge_plot, answers={
        "load data from file": "y", "file path": "groups.csv",
        "value column": "value", "group column": "group",
    })

    out = report["stdout"]
    assert "Error:" not in out, out
    assert "Groups with a single value, no density drawn: het" in out
    axes = saved_figure["fig"].axes[0]
    assert axes.get_xlabel() == "value"
    assert [c.get_label() for c in axes.collections] == ["wt", "mut"]
    assert report["images"] == ["ridge.png"]


def test_ridge_plot_says_so_when_no_group_has_two_values(run_demo, tmp_path):
    (tmp_path / "one.csv").write_text("group,value\nwt,1\nmut,2\n", encoding="utf-8")

    report = run_demo(bp.ridge_plot, answers={
        "load data from file": "y", "file path": "one.csv",
        "value column": "value", "group column": "group",
    })

    assert "No group has the two values a density needs" in report["stdout"]
    assert "Error:" not in report["stdout"]


# ── clustered heatmap and dot plot ───────────────────────────────────────────

def test_clustered_heatmap_reads_a_numeric_table(run_demo, tmp_path):
    (tmp_path / "mat.csv").write_text(
        "a,b,c\n1,2,3\n4,5,6\n7,8,10\n2,2,9\n", encoding="utf-8")

    report = run_demo(bp.clustered_heatmap, answers={
        "load data from file": "y", "file path": "mat.csv",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert "No numeric columns" not in report["stdout"]
    assert report["images"] == ["clustered_heatmap.png"]


def test_clustered_heatmap_refuses_a_table_of_text(run_demo, tmp_path):
    (tmp_path / "words.csv").write_text("a,b\nx,y\np,q\n", encoding="utf-8")

    report = run_demo(bp.clustered_heatmap, answers={
        "load data from file": "y", "file path": "words.csv",
    })

    assert "No numeric columns found." in report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0


def test_dot_plot_labels_the_genes_it_was_given(run_demo, tmp_path, saved_figure):
    (tmp_path / "single.csv").write_text(
        "gene,cluster,pct,exp\nCd3g,C1,0.8,2.1\nCd3g,C2,0.2,0.4\n"
        "Ptprc,C1,0.5,1.1\nPtprc,C2,0.9,2.8\n", encoding="utf-8")

    report = run_demo(bp.dot_plot, answers={
        "load data from file": "y", "file path": "single.csv",
        "gene column": "gene", "cluster column": "cluster",
        "percent expressed column": "pct", "average expression column": "exp",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    axes = saved_figure["fig"].axes[0]
    assert _tick_labels(axes, "y") == ["Cd3g", "Ptprc"]
    assert _tick_labels(axes, "x") == ["C1", "C2"]
    assert len(axes.collections) == 4


def test_dot_plot_uses_its_demo_when_a_column_is_missing(run_demo, tmp_path):
    (tmp_path / "single.csv").write_text("gene,cluster,pct\nCd3g,C1,0.8\n",
                                         encoding="utf-8")

    report = run_demo(bp.dot_plot, answers={
        "load data from file": "y", "file path": "single.csv",
        "gene column": "gene", "cluster column": "cluster",
        "percent expressed column": "pct", "average expression column": "nope",
    })

    assert "Using default demo data." in report["stdout"]
    assert "Error:" not in report["stdout"]


# ── typed-in sectors and sequences ───────────────────────────────────────────

def test_circos_draws_the_sectors_and_links_that_were_typed(run_demo, tmp_path,
                                                            saved_figure):
    """``quiet=False`` because the counts come from ``safe_int_input``, which answers
    itself with the default when the config is quiet — 3 sectors, not the 2 typed."""
    report = run_demo(bp.circos_plot, quiet=False, answers={
        "use default data": "n", "number of sectors": "2",
        "sector 1 name": "Alpha", "size of alpha": "10",
        "sector 2 name": "Beta", "size of beta": "5",
        "number of links": "2", "link 1 from sector": "Alpha", "to sector": "Beta",
        "link 2 from sector": "Gamma",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    texts = [t.get_text() for t in saved_figure["fig"].axes[0].texts]
    assert "Alpha" in texts and "Beta" in texts
    assert saved_figure["name"] == "circos"


def test_alignment_viewer_draws_every_sequence_that_was_pasted(run_demo, tmp_path,
                                                              saved_figure):
    report = run_demo(bp.alignment_viewer, answers={
        "use default alignment": "n", "<blank>": ["ATCG", "ATCA", ""],
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    axes = saved_figure["fig"].axes[0]
    assert _tick_labels(axes, "y") == ["Seq1", "Seq2"]
    assert axes.get_title() == "Alignment Viewer"


def test_alignment_viewer_falls_back_when_nothing_is_pasted(run_demo, tmp_path,
                                                            saved_figure):
    report = run_demo(bp.alignment_viewer, answers={
        "use default alignment": "n", "<blank>": [""],
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert len(_tick_labels(saved_figure["fig"].axes[0], "y")) == 5
