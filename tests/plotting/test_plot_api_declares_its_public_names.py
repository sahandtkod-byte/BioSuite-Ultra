"""plot_api.py declares the names a star import may take, and nothing else.

Every other plotting module declared ``__all__``; ``plot_api`` was the one module
whose namespace still leaked its imports, so ``from biosuite.plotting.plot_api
import *`` handed the caller ``np``, ``pd``, ``plt`` and even Plotly's ``go``.
The declared list is also the compatibility contract: these are the functions
external callers import (the api, cli, gui and notebook surfaces).
"""
import types

from biosuite.plotting import plot_api


def test_the_declared_surface_is_the_thirteen_public_plots():
    assert sorted(plot_api.__all__) == sorted([
        "volcano", "pca", "manhattan", "ma", "heatmap", "boxplot", "scatter",
        "barplot", "violin", "timeseries", "qqplot", "venn",
        "export_all_interactive"])


def test_every_declared_name_is_a_function_defined_by_this_module():
    for name in plot_api.__all__:
        value = getattr(plot_api, name)
        assert isinstance(value, types.FunctionType), name
        assert value.__module__ == plot_api.__name__, name


def test_a_star_import_takes_the_functions_and_nothing_foreign():
    namespace = {}
    exec("from biosuite.plotting.plot_api import *", namespace)
    exported = {n for n in namespace if not n.startswith("__")}
    assert exported == set(plot_api.__all__)
    for foreign in ("np", "pd", "plt", "go", "matplotlib", "pandas", "numpy",
                    "logging", "html", "numbers", "HAS_PLOTLY"):
        assert foreign not in exported, foreign


def test_no_public_function_drifts_out_of_the_declared_list():
    """The guard against an over-strict ``__all__``: a new public plot function
    added without declaring itself fails here instead of disappearing from the
    star-import surface."""
    stray = [node.__name__ for node in vars(plot_api).values()
             if isinstance(node, types.FunctionType)
             and node.__module__ == plot_api.__name__
             and not node.__name__.startswith("_")
             and node.__name__ not in plot_api.__all__]
    assert stray == []
