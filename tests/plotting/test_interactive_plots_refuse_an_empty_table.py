"""An empty table has no plot in it, and both backends must say so.

Nine of these functions used to hand back an empty plotly figure for empty input
— a test named that "an empty trace rather than an error" — while the matplotlib
fallback for the same call raised: ``interactive_boxplot({})`` died on
"Dimensions of labels and X must be compatible" and ``interactive_violin({})`` on
numpy's zero-size reduction.  Which of the two a user saw depended only on
whether plotly was installed.  ``plot_api``'s entry points already refuse this
input with the reason (cluster 27), so these do too, and the refusal happens
before either backend builds anything.
"""
import importlib.util
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import interactive_plots as ip

#: (id, call, the function that must name itself in the message)
EMPTY_CALLS = [
    ("scatter", lambda m: m.interactive_scatter([], []), "interactive_scatter"),
    ("bar", lambda m: m.interactive_bar([], []), "interactive_bar"),
    ("heatmap", lambda m: m.interactive_heatmap(np.zeros((0, 0))), "interactive_heatmap"),
    ("volcano", lambda m: m.interactive_volcano([], []), "interactive_volcano"),
    ("line", lambda m: m.interactive_line([], [[]]), "interactive_line"),
    ("scatter3d", lambda m: m.interactive_3d_scatter([], [], []), "interactive_3d_scatter"),
    ("pie", lambda m: m.interactive_pie([], []), "interactive_pie"),
    ("boxplot", lambda m: m.interactive_boxplot({}), "interactive_boxplot"),
    ("pca", lambda m: m.interactive_pca([]), "interactive_pca"),
    ("manhattan", lambda m: m.interactive_manhattan([], [], []), "interactive_manhattan"),
    ("ma", lambda m: m.interactive_ma([], []), "interactive_ma"),
    ("qq", lambda m: m.interactive_qq([]), "interactive_qq"),
    ("violin", lambda m: m.interactive_violin({}), "interactive_violin"),
    ("dotplot", lambda m: m.interactive_dotplot([], [], []), "interactive_dotplot"),
    ("upset", lambda m: m.interactive_upset_sunburst({}, {}),
     "interactive_upset_sunburst"),
]
IDS = [case[0] for case in EMPTY_CALLS]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def without_plotly():
    """The same module with the plotly branch decided the other way."""
    spec = importlib.util.spec_from_file_location(
        "biosuite.plotting.interactive_plots_empty_probe", ip.__file__)
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("plotly")
    sys.modules["plotly"] = None
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("plotly", None)
        else:
            sys.modules["plotly"] = saved
    assert module.HAS_PLOTLY is False, "the reload kept the real plotly"
    return module


@pytest.mark.parametrize("name,call,function", EMPTY_CALLS, ids=IDS)
def test_the_plotly_backend_refuses_an_empty_table_by_name(name, call, function):
    before = set(plt.get_fignums())

    with pytest.raises(ValueError) as excinfo:
        call(ip)

    message = str(excinfo.value)
    assert f"{function}()" in message, message
    assert "needs" in message, message
    assert set(plt.get_fignums()) == before, f"{name} opened a figure on the way out"


@pytest.mark.parametrize("name,call,function", EMPTY_CALLS, ids=IDS)
def test_the_matplotlib_backend_refuses_the_same_table(without_plotly, name, call,
                                                       function):
    with pytest.raises(ValueError) as excinfo:
        call(without_plotly)

    assert f"{function}()" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("name,call,function", EMPTY_CALLS, ids=IDS)
def test_one_point_is_all_a_plot_needs(name, call, function):
    """The positive control: the refusal is about emptiness, not about size."""
    single = {
        "scatter": lambda m: m.interactive_scatter([1.0], [2.0]),
        "bar": lambda m: m.interactive_bar(["a"], [1.0]),
        "heatmap": lambda m: m.interactive_heatmap([[1.0]]),
        "volcano": lambda m: m.interactive_volcano([1.0], [0.01]),
        "line": lambda m: m.interactive_line([1.0], [[2.0]]),
        "scatter3d": lambda m: m.interactive_3d_scatter([1.0], [2.0], [3.0]),
        "pie": lambda m: m.interactive_pie(["a"], [1.0]),
        "boxplot": lambda m: m.interactive_boxplot({"a": [1.0]}),
        "pca": lambda m: m.interactive_pca(np.array([[1.0, 2.0]])),
        "manhattan": lambda m: m.interactive_manhattan(["1"], [1], [2.0]),
        "ma": lambda m: m.interactive_ma([1.0], [0.0]),
        "qq": lambda m: m.interactive_qq([0.5]),
        "violin": lambda m: m.interactive_violin({"a": [1.0]}),
        "dotplot": lambda m: m.interactive_dotplot(["c"], ["g"], [1.0]),
        "upset": lambda m: m.interactive_upset_sunburst({"A": 1}, {"A&B": 1}),
    }[name]

    fig = single(ip)
    assert len(fig.data) >= 1  # the one point really lands in a trace


def test_a_report_with_no_figures_is_refused_and_writes_nothing(tmp_path):
    out = tmp_path / "empty.html"

    with pytest.raises(ValueError, match="at least one figure"):
        ip.export_interactive_report({}, str(out))

    assert not out.exists(), "a refused report still wrote a file"


def test_a_report_names_the_entry_that_is_not_a_figure(tmp_path):
    out = tmp_path / "report.html"

    with pytest.raises(ValueError, match="plot 'bad' is not a plotly Figure"):
        ip.export_interactive_report({"bad": object()}, str(out))

    assert not out.exists()
