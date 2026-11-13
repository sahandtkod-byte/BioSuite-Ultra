"""The edges of the demos' file input: remembered paths, missing columns, big tables.

``test_biological_plots_from_files.py`` proves that a good table is drawn correctly.
This file is about what happens around it — the branch where the user answers "y"
and types nothing, where the column they named is not in the file, where the table
is bigger than ``downsample_threshold``, where a file cannot be read at all, and
where an optional dependency is missing at import time.  Each of those branches
either falls back to the demo data or says why; the tests assert which.
"""
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.core import utils as core_utils
from biosuite.plotting import biological_plots as bp

DE_TABLE = """log2fc,pvalue
1.5,0.001
-2.0,0.02
0.1,0.9
2.5,0.0001
"""

SCATTER_TABLE = """x,y
1,2
2,4
3,6
4,8
"""

PCA_TABLE = """group,f1,f2,f3
a,1.0,2.0,3.0
a,1.5,2.5,3.5
b,5.0,6.0,7.0
b,5.5,6.5,7.5
c,9.0,1.0,2.0
c,9.5,1.5,2.5
"""


@pytest.fixture
def table(tmp_path):
    def write(name, text):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return write


@pytest.fixture
def big_table(tmp_path):
    def write(rows):
        lines = ["log2fc,pvalue,x,y"]
        for i in range(rows):
            lines.append(f"{0.1 * i - 1},{0.9 - 0.02 * i},{i},{2 * i}")
        path = tmp_path / "big.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    return write


@pytest.fixture
def quiet_threshold(monkeypatch):
    """A downsample limit the fixtures can cross deliberately."""
    def set_threshold(value):
        monkeypatch.setitem(core_utils.config, 'downsample_threshold', value)

    return set_threshold


# ─── the path the demo remembers ─────────────────────────────────────────────

@pytest.mark.parametrize("demo, key, columns, prompts, filename", [
    (bp.volcano_plot, 'last_volcano_path', DE_TABLE,
     {"log2fc column": "log2fc", "p-value column": "pvalue",
      "fold-change threshold": "", "p-value threshold": ""}, "volcano.png"),
    (bp.pca_plot, 'last_pca_path', PCA_TABLE,
     {"group column": "group"}, "pca.png"),
    (bp.scatter_custom, 'last_scatter_path', SCATTER_TABLE,
     {"x column": "x", "y column": "y", "correlation type": ""}, "scatter.png"),
])
def test_an_empty_path_answer_reuses_the_file_the_demo_remembered(
        run_demo, monkeypatch, table, demo, key, columns, prompts, filename):
    path = table("de.csv", columns)
    monkeypatch.setitem(core_utils.session, key, path)
    answers = {"load data from file": "y", "file path": ""}
    answers.update(prompts)

    report = run_demo(demo, answers=answers)

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == [filename]
    assert core_utils.session[key] == path, "the path is refreshed for the next run"


@pytest.mark.parametrize("demo, prompts, filename", [
    (bp.volcano_plot, {"log2fc column": "log2fc", "p-value column": "pvalue",
                       "fold-change threshold": "", "p-value threshold": ""},
     "volcano.png"),
    (bp.scatter_custom, {"x column": "x", "y column": "y", "correlation type": ""},
     "scatter.png"),
    (bp.pca_plot, {"group column": ""}, "pca.png"),
])
def test_answering_y_without_naming_a_file_keeps_the_remembered_path_unused(
        demo, prompts, filename, run_demo, quiet_threshold):
    quiet_threshold(1000)
    answers = {"load data from file": "y", "file path": ""}
    answers.update(prompts)

    report = run_demo(demo, answers=answers)

    assert report["images"] == [filename], report["stdout"]
    assert "Error:" not in report["stdout"], report["stdout"]


def test_pca_reads_the_features_of_a_real_table(run_demo, table, quiet_threshold):
    quiet_threshold(1000)
    path = table("pca.csv", PCA_TABLE)

    report = run_demo(bp.pca_plot, answers={
        "load data from file": "y", "file path": path, "group column": "group",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["pca.png"]
    assert report["figure_title"].startswith("PCA (var: "), report["figure_title"]
    assert report["legend_title"] == "Group"
    assert report["points_drawn"] == 6, "one point per sample row"


def test_pca_without_two_numeric_columns_falls_back_to_simulated_samples(
        run_demo, tmp_path, quiet_threshold):
    quiet_threshold(1000)
    path = tmp_path / "text.csv"
    path.write_text("label,note\none,two\nthree,four\n", encoding="utf-8")

    report = run_demo(bp.pca_plot, answers={
        "load data from file": "y", "file path": str(path),
    })

    assert report["images"] == ["pca.png"], "the demo still draws, so nothing is printed"
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["points_drawn"] == 30, "the simulated table has 30 samples"


# ─── tables that need downsampling ───────────────────────────────────────────

@pytest.mark.parametrize("demo, prompts, filename", [
    (bp.volcano_plot, {"log2fc column": "log2fc", "p-value column": "pvalue",
                       "fold-change threshold": "", "p-value threshold": ""},
     "volcano.png"),
    (bp.scatter_custom, {"x column": "x", "y column": "y", "correlation type": ""},
     "scatter.png"),
])
def test_a_table_bigger_than_the_downsample_limit_is_cut(
        run_demo, big_table, quiet_threshold, demo, prompts, filename):
    quiet_threshold(10)
    path = big_table(40)
    answers = {"load data from file": "y", "file path": path}
    answers.update(prompts)

    report = run_demo(demo, answers=answers)

    assert report["images"] == [filename]
    assert report["points_drawn"] <= 10,             f"{report['points_drawn']} points survived a limit of 10"


def test_pca_downsamples_more_rows_than_the_threshold_allows(
        run_demo, table, quiet_threshold):
    quiet_threshold(4)
    path = table("pca_many.csv", "\n".join(
        ["group,f1,f2,f3"] + [f"g{i % 3},{i * 1.0},{i * 1.5},{i * 2.0}" for i in range(30)]))

    report = run_demo(bp.pca_plot, answers={
        "load data from file": "y", "file path": path, "group column": "group",
    })

    assert report["images"] == ["pca.png"]
    assert report["points_drawn"] == 4, report["points_drawn"]


def test_a_qq_column_bigger_than_the_limit_is_sampled(
        run_demo, tmp_path, quiet_threshold, figure_at_save):
    quiet_threshold(25)
    path = tmp_path / "values.csv"
    values = np.linspace(-3, 3, 400)
    path.write_text("value\n" + "\n".join(f"{v:.4f}" for v in values), encoding="utf-8")
    captured = figure_at_save(bp)

    report = run_demo(bp.qq_plot, answers={
        "load data from file": "y", "file path": str(path), "column to test": "value",
    })

    assert report["images"] == ["qqplot.png"]
    longest = max(captured["fig"].axes[0].lines, key=lambda line: len(line.get_xdata()))
    assert len(longest.get_xdata()) == 25,             "the QQ shows the sampled points, not the whole table"


# ─── answers that do not match the file ──────────────────────────────────────

NOT_THERE = {
    "boxplot": (bp.boxplot_custom, {"value column": "nope", "group column": "nope"},
                "boxplot.png"),
    "pca": (bp.pca_plot, {"group column": "nope"}, "pca.png"),
    "qq": (bp.qq_plot, {"column to test": "nope"}, "qqplot.png"),
    "scatter": (bp.scatter_custom, {"x column": "nope", "y column": "nope",
                                    "correlation type": ""}, "scatter.png"),
    "timeseries": (bp.timeseries_plot, {"time column": "nope", "value column": "nope"},
                   "timeseries.png"),
    "volcano": (bp.volcano_plot, {"log2fc column": "nope", "p-value column": "nope",
                                  "fold-change threshold": "", "p-value threshold": ""},
                "volcano.png"),
}


@pytest.mark.parametrize("name", sorted(NOT_THERE))
def test_a_named_column_that_the_file_lacks_falls_back_to_the_demo(
        name, run_demo, table, quiet_threshold):
    quiet_threshold(1000)
    demo, prompts, filename = NOT_THERE[name]
    path = table("p.csv", PCA_TABLE)
    answers = {"load data from file": "y", "file path": path}
    answers.update(prompts)

    report = run_demo(demo, answers=answers)

    assert report["images"] == [filename], report["stdout"]
    assert "Error:" not in report["stdout"], report["stdout"]


def test_pca_plots_a_table_whose_group_column_is_missing(run_demo, table, quiet_threshold):
    """A missing group column is not an error — every sample simply shares one label."""
    quiet_threshold(1000)
    path = table("pca.csv", PCA_TABLE)

    report = run_demo(bp.pca_plot, answers={
        "load data from file": "y", "file path": path, "group column": "nope",
    })

    assert report["legend_title"] is None or report["legend_title"] == "Group"
    assert report["points_drawn"] == 6


@pytest.mark.parametrize("demo, filename", [
    (bp.barplot_custom, "barplot.png"),
    (bp.heatmap_custom, "heatmap.png"),
    (bp.timeseries_plot, "timeseries.png"),
])
def test_pressing_y_and_naming_no_file_draws_the_demo_curve(
        demo, filename, run_demo, quiet_threshold):
    quiet_threshold(1000)

    report = run_demo(demo, answers={"load data from file": "y", "file path": ""})

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == [filename], report["stdout"]


def test_a_file_that_cannot_be_read_still_leaves_a_plot_and_says_why(
        run_demo, tmp_path, quiet_threshold):
    quiet_threshold(1000)
    missing = tmp_path / "deleted.csv"

    report = run_demo(bp.qq_plot, answers={
        "load data from file": "y", "file path": str(missing),
    })

    assert "does not exist" not in report["stdout"], "the loader logs, it does not print"
    assert report["images"] == ["qqplot.png"], "the demo still draws after a bad path"
    assert "Error:" not in report["stdout"], report["stdout"]


# ─── the table that has nothing in it ────────────────────────────────────────

def test_manhattan_names_a_table_whose_positions_are_all_missing(run_demo, table):
    """Every row disappears once its blank position is dropped, leaving nothing to
    plot; the demo has to say that instead of drawing an empty axis."""
    path = table("n_pos.csv", "chrom,pos,p\nchr1,,0.5\nchr1,,0.4\n")

    report = run_demo(bp.manhattan_plot, answers={
        "load data from file": "y", "file path": path,
        "chromosome column": "chrom", "position column": "pos", "p-value column": "p",
    })

    assert "No valid data." in report["stdout"], report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0


# ─── optional dependencies and the import-time guards ────────────────────────

def test_the_module_loads_without_tqdm_and_keeps_its_demos(monkeypatch):
    """``HAS_TQDM`` is decided at import time; hiding the library and re-executing
    the module shows what the batch export does without it."""
    monkeypatch.setitem(sys.modules, "tqdm", None)
    spec = importlib.util.spec_from_file_location(
        "biosuite.plotting.biological_plots_without_tqdm", bp.__file__)
    fresh = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(fresh)

    assert fresh.HAS_TQDM is False
    assert fresh.volcano_plot.__name__ == "volcano_plot"
    assert fresh.draw_venn2 is not None, "the rest of the module still loaded"


def test_the_batch_export_wraps_its_loop_in_a_progress_bar_when_tqdm_is_present(
        run_demo, monkeypatch, tmp_path):
    seen = {}

    class FakeBar:
        def __init__(self, iterable=None, **kwargs):
            seen['kwargs'] = kwargs
            self._items = list(iterable or [])

        def __iter__(self):
            return iter(self._items)

    monkeypatch.setattr(bp, "HAS_TQDM", True)
    monkeypatch.setattr(bp, "tqdm", FakeBar)
    monkeypatch.setattr(bp, "_batch_plot_functions", lambda: [])
    folder = tmp_path / "export"

    report = run_demo(bp.export_all_to_folder, str(folder))

    assert seen['kwargs'], "the batch counted its own loop instead of using tqdm"
    assert list(folder.iterdir()) == [], "an empty batch writes nothing"


def test_draw_sankey_places_one_link_per_pair():
    fig, ax = plt.subplots()

    returned = bp.draw_sankey(["A", "B", "C"], [0, 1], [1, 2], [10.0, 5.0], ax=ax)

    links = [p for p in ax.patches if type(p).__name__ == 'ConnectionPatch']
    assert len(links) == 2
    assert returned is ax
    plt.close(fig)


# ─── every reader demo, handed a file that is not there ──────────────────────

#: The demos that can be pointed at the user's own table.
READER_DEMOS = [
    bp.volcano_plot, bp.pca_plot, bp.manhattan_plot, bp.ma_plot, bp.barplot_custom,
    bp.boxplot_custom, bp.heatmap_custom, bp.scatter_custom, bp.timeseries_plot,
    bp.qq_plot, bp.clustered_heatmap, bp.violin_plot, bp.raincloud_plot,
    bp.ridge_plot, bp.dot_plot,
]


@pytest.mark.parametrize("demo", READER_DEMOS,
                         ids=[d.__name__ for d in READER_DEMOS])
def test_a_path_that_leads_nowhere_still_leaves_the_demo_plot(run_demo, demo):
    """``load_dataframe_safe`` answers None for a missing file, and every demo has
    to carry on with its own data instead of ending the run in silence."""
    report = run_demo(demo, answers={
        "load data from file": "y", "file path": "no_such_table.csv",
        "correlation type": "",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"], f"{demo.__name__} saved nothing after a failed read"
    assert report["empty_images"] == []
    assert report["figures_open"] == 0, "the figure was left pinned after the save"


# ─── the columns and groups a real table can be short of ─────────────────────

GROUPED_TABLE = """group,value
ctrl,1.0
ctrl,2.0
ctrl,3.0
trt,5.0
trt,6.0
trt,7.0
"""

ONE_GROUP_TABLE = """group,value
ctrl,1.0
ctrl,2.0
ctrl,4.0
"""

RIDGE_WITH_LONE_GROUP = """group,value
ctrl,1.0
ctrl,2.0
ctrl,3.0
lonely,5.0
"""

BAR_TABLE_WITH_ERRORS = """cat,val,err
a,1.0,0.2
b,2.0,0.3
c,3.0,0.1
d,4.0,0.4
"""

BAR_TABLE_WITHOUT_ERRORS = """cat,val
a,1.0
b,2.0
c,3.0
d,4.0
"""

NO_NUMERIC_TABLE = """gene,note
A,hello
B,world
"""


@pytest.mark.parametrize("demo, filename", [
    (bp.boxplot_custom, "boxplot"), (bp.violin_plot, "violin"),
    (bp.raincloud_plot, "raincloud"),
])
def test_two_groups_get_a_significance_test_and_one_group_does_not(
        run_demo, table, figure_at_save, demo, filename):
    """A t-test across a single group is not a test, it is a number wearing a costume."""
    captured = figure_at_save(bp)
    two = table("two.csv", GROUPED_TABLE)
    one = table("one.csv", ONE_GROUP_TABLE)

    run_demo(demo, answers={"load data from file": "y", "file path": two,
                            "value column": "value", "group column": "group"})
    annotated = [t.get_text() for t in captured["fig"].axes[0].texts]
    assert any("p =" in text for text in annotated), annotated

    plt.close("all")
    run_demo(demo, answers={"load data from file": "y", "file path": one,
                            "value column": "value", "group column": "group"})
    alone = [t.get_text() for t in captured["fig"].axes[0].texts]
    assert not any("p =" in text for text in alone), \
        f"{filename} reported a test over the single group it was given: {alone}"


def test_a_column_the_table_lacks_sends_the_boxplot_back_to_its_own_data(
        run_demo, table, figure_at_save):
    captured = figure_at_save(bp)
    path = table("grouped.csv", GROUPED_TABLE)

    run_demo(bp.boxplot_custom, answers={
        "load data from file": "y", "file path": path,
        "value column": "value", "group column": "genotype",
    })

    groups = sorted({t.get_text() for t in captured["fig"].axes[0].get_xticklabels()})
    assert groups == ["Ctrl", "Treat"], f"the demo's own groups, not the missing column: {groups}"


def test_a_table_of_only_text_is_refused_by_the_heatmap(run_demo, table, figure_at_save):
    """Nothing to correlate means the demo matrix, and the user is told that swap
    happened instead of reading their own numbers into a picture of it."""
    captured = figure_at_save(bp)
    path = table("notes.csv", NO_NUMERIC_TABLE)

    report = run_demo(bp.heatmap_custom, answers={
        "load data from file": "y", "file path": path,
    })

    assert "no numeric columns to correlate" in report["stdout"], report["stdout"]
    assert captured["fig"].axes[0].get_title() == "Gene Correlation", \
        "the fallback is the demo matrix, and it is titled as one"


def test_the_barplot_draws_caps_only_for_a_table_that_carries_them(
        run_demo, table, figure_at_save):
    """``apply_glass_ax`` lays a panel behind every plot, so the bars are the
    Rectangles and the caps are the one line artist the error column buys."""
    captured = figure_at_save(bp)
    with_caps = table("bar3.csv", BAR_TABLE_WITH_ERRORS)
    without_caps = table("bar2.csv", BAR_TABLE_WITHOUT_ERRORS)

    def bars_and_lines():
        ax = captured["fig"].axes[0]
        bars = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
        return [round(float(p.get_height()), 3) for p in bars], len(ax.lines)

    run_demo(bp.barplot_custom, answers={
        "load data from file": "y", "file path": without_caps,
        "category column": "cat", "value column": "val", "error column (optional)": "",
    })
    heights_plain, lines_plain = bars_and_lines()
    assert heights_plain == [1.0, 2.0, 3.0, 4.0], "one bar per row of the table"

    plt.close("all")
    run_demo(bp.barplot_custom, answers={
        "load data from file": "y", "file path": with_caps,
        "category column": "cat", "value column": "val", "error column (optional)": "err",
    })
    heights_caps, lines_caps = bars_and_lines()
    assert heights_caps == heights_plain, "the same bars, with caps added"
    assert lines_caps == lines_plain + 2, "the errorbar container brings two caplines"


def test_a_column_the_table_lacks_sends_the_barplot_back_to_its_own_values(
        run_demo, table, figure_at_save):
    captured = figure_at_save(bp)
    path = table("bar2.csv", BAR_TABLE_WITHOUT_ERRORS)

    run_demo(bp.barplot_custom, answers={
        "load data from file": "y", "file path": path,
        "category column": "strain", "value column": "val", "error column (optional)": "",
    })

    ax = captured["fig"].axes[0]
    bars = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    assert [round(float(p.get_height()), 3) for p in bars] == [5.2, 8.7, 3.4], \
        "a column that is not in the file sends the whole table back to the demo data"


def test_the_ridge_plot_names_the_group_it_cannot_estimate_a_density_for(
        run_demo, table):
    path = table("ridge.csv", RIDGE_WITH_LONE_GROUP)

    report = run_demo(bp.ridge_plot, answers={
        "load data from file": "y", "file path": path,
        "value column": "value", "group column": "group",
    })

    assert "lonely" in report["stdout"], "the group left out of the plot was not mentioned"
    assert report["images"], "the group that could be drawn still got drawn"


# ─── the loops that read more than one answer ────────────────────────────────

def test_an_alignment_loop_closed_midway_falls_back_to_the_sample_alignment(
        run_demo, eof):
    report = run_demo(bp.alignment_viewer, answers={
        "use default alignment": "n", "<blank>": [eof],
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"], "a closed stdin should not end the run without a plot"


def _venn_numbers(figure):
    """The sizes printed on the circles, parsed so int and float rendering agree."""
    drawn = []
    for text in figure.axes[0].texts:
        try:
            drawn.append(float(text.get_text()))
        except ValueError:
            pass
    return sorted(drawn)


@pytest.mark.parametrize("n_sets, typed, expected", [
    ("2", "6,8,3", [3.0, 6.0, 8.0]),
    # Three sets whose regions can actually exist: ABC has to fit inside every
    # pair, which the 11,12,13,4,5,6,7 this test used to type does not (7 > 4).
    ("3", "11,12,13,8,7,6,4", [4.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]),
    ("2", "6,8", [4.0, 10.0, 15.0]),
    ("3", "1,2", [1.0, 3.0, 3.0, 3.0, 8.0, 8.0, 8.0]),
])
def test_the_venn_draws_the_sizes_it_was_given_and_its_own_when_they_do_not_fit(
        run_demo, figure_at_save, n_sets, typed, expected):
    """A list of the wrong length is refused without a word: the demo falls back to
    its own sizes, and the only honest check is the number printed on each region.
    A list of the right length but impossible set relations is now refused by the
    drawer itself — see test_biological_plot_venn_region_counts.py."""
    captured = figure_at_save(bp)

    report = run_demo(bp.venn_diagram, quiet=False,
                      answers={"number of sets": n_sets, "enter sizes": typed})

    assert _venn_numbers(captured["fig"]) == expected, typed
    assert report["images"], "the diagram never reached the disk"


# ─── the batch export ────────────────────────────────────────────────────────

def test_the_batch_export_fills_a_folder_that_is_already_there(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "already"
    folder.mkdir()
    monkeypatch.setattr(bp, "_batch_plot_functions", lambda: [bp.barplot_custom])

    bp.export_all_to_folder(str(folder))

    assert list(folder.iterdir()), "an existing folder was skipped instead of filled"
    assert "FAILED to save" not in capsys.readouterr().out


def test_the_batch_export_names_the_file_it_could_not_write(tmp_path, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bp, "_batch_plot_functions", lambda: [bp.barplot_custom])
    monkeypatch.setattr(plt, "savefig", refuse)

    bp.export_all_to_folder("broken")

    out = capsys.readouterr().out
    assert "FAILED to save" in out and "disk full" in out, out
    assert list((tmp_path / "broken").iterdir()) == [], "a refused write left no file"


def test_a_clustered_heatmap_larger_than_the_limit_is_cut(run_demo, big_table, monkeypatch):
    """``sns.clustermap`` is the slow thing here, so it is the one artist stubbed out:
    what is under test is how many rows the demo hands it."""
    import types

    seen = {}

    def fake_clustermap(data, **kwargs):
        seen["rows"] = len(data)
        fig, ax = plt.subplots()
        seen["fig"] = fig
        return types.SimpleNamespace(ax_heatmap=ax)

    monkeypatch.setattr(bp.sns, "clustermap", fake_clustermap)
    path = big_table(2500)

    report = run_demo(bp.clustered_heatmap, answers={
        "load data from file": "y", "file path": path,
    })

    assert seen["rows"] == 2000, "the demo handed the whole table to the clusterer"
    assert "Downsampled to 2000 rows" in report["stdout"], report["stdout"]


def test_a_ridge_plot_of_groups_it_can_estimate_needs_no_apology(run_demo, table):
    path = table("grouped.csv", GROUPED_TABLE)

    report = run_demo(bp.ridge_plot, answers={
        "load data from file": "y", "file path": path,
        "value column": "value", "group column": "group",
    })

    assert "no density drawn" not in report["stdout"], report["stdout"]
    assert report["images"] == ["ridge.png"], report["stdout"]


@pytest.mark.parametrize("demo, filename", [
    (bp.violin_plot, "violin.png"), (bp.raincloud_plot, "raincloud.png"),
    (bp.ridge_plot, "ridge.png"),
])
def test_a_group_plot_names_a_column_its_table_lacks_by_drawing_its_own_data(
        run_demo, table, figure_at_save, demo, filename):
    """The named column is gone, so the plot that arrives is the demo's — and the
    file's group labels must not be anywhere on it, or the caller reads their own
    numbers into a picture that never held them."""
    captured = figure_at_save(bp)
    path = table("grouped.csv", GROUPED_TABLE)

    report = run_demo(demo, answers={
        "load data from file": "y", "file path": path,
        "value column": "abundance", "group column": "genotype",
    })

    assert report["images"] == [filename], report["stdout"]
    assert "Error:" not in report["stdout"], report["stdout"]
    drawn = [t.get_text() for t in captured["fig"].axes[0].texts]
    drawn += [t.get_text() for t in captured["fig"].axes[0].get_xticklabels()]
    drawn += [t.get_text() for t in captured["fig"].axes[0].get_yticklabels()]
    assert not any("ctrl" in text for text in drawn), \
        f"the file's group names appear on a plot the file could not feed: {drawn}"


def test_a_motif_letter_the_counter_does_not_hold_is_ignored():
    """A padded column is a gap, not a base, and must not be counted as one."""
    fig, ax = plt.subplots()

    bp.draw_motif_logo(["AAA", "GG"], ax=ax)

    assert len(ax.patches) > 0, "the two letters that were counted still drew"
    plt.close(fig)
