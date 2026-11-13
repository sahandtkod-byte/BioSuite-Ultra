"""The ``biosuite.plotting`` package surface.

``biosuite/plotting/__init__.py`` is a list of ``import *`` lines, so a name
defined in two of those submodules resolves to whichever was imported last.
Nothing inside the package uses the re-exported names — every caller imports the
submodule — which is exactly why this surface needs a test rather than a review.
"""
import importlib
import pathlib
import re
import types

import biosuite.plotting as plotting
from biosuite.plotting import biological_plots, math_plots, specialized_plots

DEMOS = [
    (biological_plots, [
        "volcano_plot", "pca_plot", "manhattan_plot", "ma_plot", "venn_diagram",
        "barplot_custom", "boxplot_custom", "heatmap_custom", "scatter_custom",
        "timeseries_plot", "qq_plot", "clustered_heatmap", "circos_plot",
        "alignment_viewer", "violin_plot", "raincloud_plot", "ridge_plot",
        "dot_plot",
    ]),
    (math_plots, [
        "sine_plot", "cosine_plot", "linear_plot", "quadratic_plot", "cubic_plot",
        "exponential_plot", "logistic_plot",
    ]),
    (specialized_plots, ["gsea_plot", "motif_logo", "sankey_diagram", "umap_plot"]),
]


def _star_imported_modules():
    text = pathlib.Path(plotting.__file__).read_text(encoding="utf-8")
    return [importlib.import_module("biosuite.plotting." + name)
            for name in re.findall(r"^from \.(\w+) import \*", text, re.M)]


def test_package_re_exports_every_demo():
    wrong = [f"{module.__name__}.{name}"
             for module, names in DEMOS
             for name in names
             if getattr(plotting, name, None) is not getattr(module, name)]
    assert wrong == []


def test_star_imports_never_bind_one_name_to_two_definitions():
    owners, collisions = {}, []
    for module in _star_imported_modules():
        for name, value in vars(module).items():
            if name.startswith("_") or not isinstance(value, types.FunctionType):
                continue
            if name in owners and owners[name] is not value:
                collisions.append(f"{name}: {owners[name].__module__} vs {value.__module__}")
            owners[name] = value
    assert collisions == []
