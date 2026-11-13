"""The two backends every ``plot_api`` function promises, and the files it writes.

Each function takes ``interactive`` and ``output_html``, and the docstring says it
returns "matplotlib Figure or Plotly Figure" and saves when asked.  Neither promise
held for the whole family: eleven of the twelve matplotlib branches read
``output_html`` and then returned without writing anything, and the whole
``interactive`` half of the module was untested apart from volcano, pca, heatmap,
manhattan, scatter and qq.  ``venn`` was worse still — measured before the fix, its
matplotlib branch drew **no circles at all** for the ``[A, B, AB]`` shape its own
test suite passes, and drew nothing for the 7-region three-set shape too.

``HAS_PLOTLY`` is patched rather than skipped: an install without plotly is the
normal case for ``pip install biosuite-ultra``, and there the request for an
interactive figure has to degrade to a matplotlib one, not fail.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import plot_api as pa

LOG2FC = np.array([2.5, -3.0, 0.1, 1.2, -0.4])
PVALUES = np.array([0.001, 0.002, 0.60, 0.01, 0.40])
GENES = [f"G{i}" for i in range(5)]
MATRIX = np.arange(12, dtype=float).reshape(3, 4)
GROUPS = {"ctrl": [1.0, 2.0, 3.0, 2.5], "trt": [4.0, 5.0, 4.5, 6.0]}
X = np.arange(5.0)
Y = 2 * X + 1.0
CHROMS = np.array(["chr1", "chr1", "chr2", "chr2", "chr3"])
POSITIONS = np.array([100, 900, 200, 500, 300])
MEAN_EXPR = np.array([5.0, 6.0, 7.0])
LOG_FC = np.array([1.0, -1.0, 0.2])

#: One call per public plot, with the arguments each one actually needs.
PLOTS = {
    "volcano": lambda **kw: pa.volcano(LOG2FC, PVALUES, gene_names=GENES, **kw),
    "pca": lambda **kw: pa.pca(np.random.default_rng(0).normal(size=(6, 3)),
                               labels=[f"S{i}" for i in range(6)], **kw),
    "manhattan": lambda **kw: pa.manhattan(CHROMS, POSITIONS, PVALUES, **kw),
    "ma": lambda **kw: pa.ma(MEAN_EXPR, LOG_FC, sig=np.array([True, False, True]), **kw),
    "heatmap": lambda **kw: pa.heatmap(MATRIX, row_labels=["a", "b", "c"],
                                       col_labels=["w", "x", "y", "z"], **kw),
    "boxplot": lambda **kw: pa.boxplot(GROUPS, **kw),
    "scatter": lambda **kw: pa.scatter(X, Y, labels=GENES, **kw),
    "barplot": lambda **kw: pa.barplot(["a", "b", "c"], [3.0, 7.0, 2.0],
                                       errors=[0.2, 0.4, 0.2], **kw),
    "violin": lambda **kw: pa.violin(GROUPS, **kw),
    "timeseries": lambda **kw: pa.timeseries([0, 1, 2, 3, 4], [Y, Y * 0.5],
                                             names=["up", "down"], **kw),
    "qqplot": lambda **kw: pa.qqplot(PVALUES, **kw),
    "venn": lambda **kw: pa.venn([50, 40, 15], set_names=["X", "Y"], **kw),
}

NAMES = sorted(PLOTS)


def _ax(fig):
    return fig.axes[0]


# ── the matplotlib backend ───────────────────────────────────────────────────

@pytest.mark.parametrize("name", NAMES)
def test_the_default_call_returns_a_labelled_matplotlib_figure(name):
    fig = PLOTS[name]()

    assert isinstance(fig, plt.Figure)
    assert fig.axes, "a figure with no axes cannot be labelled or read"
    assert _ax(fig).get_title(), f"{name} left its own title unset"


@pytest.mark.parametrize("name", NAMES)
def test_a_plot_is_drawn_not_just_created(name):
    """An empty figure used to count as success: venn returned one for the shapes
    its callers actually pass."""
    fig = PLOTS[name]()
    ax = _ax(fig)

    drawn = (len(ax.patches) + len(ax.lines) + len(ax.collections) + len(ax.images)
             + len(ax.texts))
    assert drawn, f"{name} drew nothing at all"


@pytest.mark.parametrize("name", NAMES)
def test_output_html_writes_a_file_without_the_plotly_backend(name, tmp_path):
    target = tmp_path / f"{name}.html"

    PLOTS[name](output_html=str(target))

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [f"{name}.png"], (
        f"matplotlib cannot write HTML, so {name} must write the PNG beside it "
        f"(found {written})")
    assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_no_file_is_written_when_none_was_asked(tmp_path):
    PLOTS["volcano"]()

    assert list(tmp_path.iterdir()) == []


# ── the plotly backend ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", NAMES)
def test_interactive_returns_a_plotly_figure_with_traces(name):
    go = pytest.importorskip("plotly.graph_objects")

    fig = PLOTS[name](interactive=True)

    assert isinstance(fig, go.Figure)
    assert fig.data, f"{name} returned a Plotly figure with no traces"
    assert fig.layout.title.text, f"{name} dropped its title on the way to plotly"


@pytest.mark.parametrize("name", NAMES)
def test_interactive_falls_back_when_plotly_is_not_installed(monkeypatch, name):
    """``interactive`` is a preference, not a requirement: without plotly the
    caller still gets a figure they can save."""
    monkeypatch.setattr(pa, "HAS_PLOTLY", False)

    fig = PLOTS[name](interactive=True)

    assert isinstance(fig, plt.Figure), f"{name} promised a figure and returned {fig!r}"
    assert _ax(fig).get_title()


@pytest.mark.parametrize("name", NAMES)
def test_interactive_html_is_written_by_plotly(name, tmp_path):
    go = pytest.importorskip("plotly.graph_objects")
    target = tmp_path / f"{name}.html"

    fig = PLOTS[name](interactive=True, output_html=str(target))

    assert isinstance(fig, go.Figure)
    assert target.exists(), f"{name}'s interactive branch promised an HTML file"
    assert target.stat().st_size > 1000, "an HTML file with no figure in it"
    assert [p.name for p in tmp_path.iterdir()] == [f"{name}.html"], \
        "plotly writes the HTML it was asked for, not a PNG beside it"


# ── the calls the table above never makes ────────────────────────────────────

def test_pca_without_anything_to_group_by_draws_one_cloud():
    pytest.importorskip("plotly.graph_objects")
    data = np.random.default_rng(3).normal(size=(8, 3))

    fig = pa.pca(data, title="Unlabelled", interactive=True)

    assert len(fig.data) == 1, "one scatter of every sample, not one trace per group"
    assert len(fig.data[0].x) == 8


def test_heatmap_reads_its_labels_out_of_a_dataframe():
    frame = pd.DataFrame(MATRIX, index=["a", "b", "c"], columns=["w", "x", "y", "z"])

    fig = pa.heatmap(frame)

    ax = _ax(fig)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["w", "x", "y", "z"]


def test_a_caller_naming_the_rows_of_a_dataframe_overrides_its_index():
    frame = pd.DataFrame(MATRIX, index=["a", "b", "c"], columns=["w", "x", "y", "z"])

    fig = pa.heatmap(frame, row_labels=["R1", "R2", "R3"], col_labels=["C1", "C2", "C3", "C4"])

    ax = _ax(fig)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["R1", "R2", "R3"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C1", "C2", "C3", "C4"]


def test_heatmap_without_labels_leaves_the_axes_as_imshow_set_it():
    fig = pa.heatmap(MATRIX)

    ax = _ax(fig)
    assert ax.get_xlim() == pytest.approx((-0.5, 3.5)), "four columns, numbered by matplotlib"
    assert ax.get_ylim() == pytest.approx((2.5, -0.5)), "three rows, and imshow reads top down"


@pytest.mark.parametrize("interactive", [False, True])
def test_scatter_can_leave_the_regression_line_out(interactive):
    pytest.importorskip("plotly.graph_objects")

    fig = pa.scatter(X, Y, show_regression=False, interactive=interactive)

    if interactive:
        assert len(fig.data) == 1, "the points alone"
    else:
        assert len(_ax(fig).lines) == 0, "a scatter with a fitted line in it is not a plain scatter"


def test_barplot_draws_the_same_bars_with_or_without_error_caps():
    capped = _ax(pa.barplot(["a", "b", "c"], [3.0, 7.0, 2.0], errors=[0.2, 0.4, 0.2]))
    plain = _ax(pa.barplot(["a", "b", "c"], [3.0, 7.0, 2.0]))

    assert [p.get_height() for p in plain.patches] == [3.0, 7.0, 2.0]
    assert len(capped.patches) == len(plain.patches), "the bars do not change with the caps"
    assert len(plain.lines) == 0, "nothing was asked for, so nothing else was drawn"
    assert len(capped.lines) > 0, "the caps are the only thing the errors add"


# ── the module as it loads when plotly is not installed ──────────────────────

def test_the_api_loads_without_plotly_and_answers_with_matplotlib(monkeypatch):
    """``HAS_PLOTLY`` is decided once, at import, so patching the flag only shows
    half the story; re-importing the module shows the whole module falling back."""
    import importlib.util
    import pathlib
    import sys

    spec = importlib.util.spec_from_file_location(
        "biosuite.plotting.plot_api_without_plotly", pathlib.Path(pa.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "plotly", None)
    spec.loader.exec_module(module)

    assert module.HAS_PLOTLY is False
    fig = module.volcano(LOG2FC, PVALUES, interactive=True)

    assert isinstance(fig, plt.Figure), "the caller asked for interactive and got nothing"
    assert _ax(fig).get_title() == "Volcano Plot"


# ── venn shapes ──────────────────────────────────────────────────────────────

def test_venn_two_set_sizes_draw_two_circles_and_three_numbers():
    fig = pa.venn([50, 40, 15], set_names=["X", "Y"])

    ax = _ax(fig)
    assert len(ax.patches) == 2, "two sets need two circles"
    texts = [t.get_text() for t in ax.texts]
    assert "50" in texts and "40" in texts and "15" in texts, texts
    assert "X" in texts and "Y" in texts


def test_venn_three_set_sizes_draw_three_circles_and_seven_numbers():
    fig = pa.venn([10, 12, 8, 4, 3, 2, 1], set_names=["A", "B", "C"])

    ax = _ax(fig)
    assert len(ax.patches) == 3
    numbers = [t.get_text() for t in ax.texts if t.get_text().isdigit()]
    assert numbers == ["10", "12", "8", "4", "3", "2", "1"], numbers


@pytest.mark.parametrize("sizes", [[10, 8], [1, 2, 3, 4], []])
def test_venn_refuses_a_region_list_it_cannot_draw(sizes):
    with pytest.raises(ValueError, match="3 region sizes|7"):
        pa.venn(sizes)


def test_venn_refuses_names_that_do_not_match_the_sets():
    with pytest.raises(ValueError, match="2 sets need 2 names"):
        pa.venn([50, 40, 15], set_names=["X", "Y", "Z"])


def test_venn_interactive_sunburst_draws_each_set_at_its_true_size():
    """The wedge named X has to be |X|, not 'only X'.

    ``branchvalues='total'`` reads a parent's own value as its total, so passing
    the region sizes [50, 40, 15] drew X as 50 wide while the intersection (15)
    sat inside it — the figure claimed |X| = 50 and only-X = 50 at once. With
    only-X = 50, only-Y = 40 and both = 15 the wedges are 65 and 55.
    """
    go = pytest.importorskip("plotly.graph_objects")

    fig = pa.venn([50, 40, 15], set_names=["X", "Y"], interactive=True)

    trace = fig.data[0]
    assert isinstance(trace, go.Sunburst)
    assert list(trace.labels) == ["X", "Y", "Intersection"]
    assert list(trace.values) == [65, 55, 15]
    assert list(trace.parents) == ["", "", "X"]
    assert trace.values[2] <= trace.values[0], "the intersection fits inside X"


def test_venn_defaults_to_numbered_set_names():
    fig = pa.venn([50, 40, 15])

    texts = [t.get_text() for t in _ax(fig).texts]
    assert "Set 1" in texts and "Set 2" in texts


# ── the combined report ──────────────────────────────────────────────────────

def test_export_all_interactive_puts_every_figure_in_one_page(tmp_path):
    pytest.importorskip("plotly")
    report = tmp_path / "report.html"
    figures = {"Volcano": pa.volcano(LOG2FC, PVALUES, interactive=True),
               "Boxplot": pa.boxplot(GROUPS, interactive=True)}

    pa.export_all_interactive(figures, output_path=str(report))

    html = report.read_text(encoding="utf-8")
    assert "<h2>Volcano</h2>" in html and "<h2>Boxplot</h2>" in html
    assert html.count("<h2>") == len(figures)
    assert "BioSuite Interactive Report" in html


def test_export_all_interactive_says_why_it_wrote_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pa, "HAS_PLOTLY", False)
    report = tmp_path / "report.html"

    pa.export_all_interactive({"Volcano": object()}, output_path=str(report))

    assert not report.exists()
    assert "Plotly not installed. Cannot generate interactive report." in \
        capsys.readouterr().out
