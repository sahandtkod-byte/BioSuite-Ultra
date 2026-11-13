"""Every interactive plot demo must survive a closed standard input.

Each demo in ``biosuite.plotting`` ends in a broad ``except Exception`` that prints
one line, and the CLI and GUI call demos directly — so "the handler runs, reports,
and does not raise" is what keeps a menu session alive when a plot fails.  The
realistic failure injected here is standard input closed under the demo:
``core.utils.ask_save_plot`` already guards ``EOFError`` for exactly this reason,
but the demos reach ``input()`` long before it, and ``safe_float_input`` catches
``ValueError``/``TypeError`` only.

Each test asserts the demo said something, wrote nothing, and left no figure
pinned open — an unclosed figure is a process-lifetime leak in a long CLI session
(F2-3), which is what this family of tests guards.
"""
import builtins
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from biosuite.core import utils as core_utils
from biosuite.plotting import biological_plots, math_plots, specialized_plots

#: (module, function) for every demo that reads a prompt and ends in a broad
#: ``except Exception``.  The two batch exporters are deliberately absent: they
#: install their own input responder, so a closed stdin never reaches them (see
#: ``test_biological_plot_batch_export.py``), and ``draw_translation_view`` is a
#: pure renderer that takes its sequence as an argument.
DEMOS = (
    [(biological_plots, name) for name in [
        "volcano_plot", "pca_plot", "manhattan_plot", "ma_plot", "venn_diagram",
        "barplot_custom", "boxplot_custom", "heatmap_custom", "scatter_custom",
        "timeseries_plot", "qq_plot", "clustered_heatmap", "circos_plot",
        "alignment_viewer", "violin_plot", "raincloud_plot", "ridge_plot",
        "dot_plot"]]
    + [(math_plots, name) for name in [
        "sine_plot", "cosine_plot", "linear_plot", "quadratic_plot", "cubic_plot",
        "exponential_plot", "logistic_plot"]]
    + [(specialized_plots, name) for name in [
        "gsea_plot", "motif_logo", "sankey_diagram", "umap_plot"]]
)

IDS = [f"{module.__name__.partition('plotting.')[2]}.{name}" for module, name in DEMOS]


@pytest.fixture(autouse=True)
def closed_stdin(monkeypatch, tmp_path):
    """Make every prompt raise the way a closed pipe does, and run in an empty dir."""
    def ask(prompt=""):
        raise EOFError("EOF when reading a line")

    monkeypatch.setattr(builtins, "input", ask)
    monkeypatch.setitem(core_utils.config, "quiet", False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("module,name", DEMOS, ids=IDS)
def test_a_demo_without_input_reports_instead_of_raising(module, name, capsys):
    figures_before = set(plt.get_fignums())

    result = getattr(module, name)()

    out = capsys.readouterr().out
    assert "Error: EOF when reading a line" in out, \
        f"{name} caught nothing and printed:\n{out}"
    assert "Traceback" not in out
    assert result is None, f"{name} returned {result!r} from its failure path"
    leaked = set(plt.get_fignums()) - figures_before
    assert not leaked, f"{name} left figure(s) {sorted(leaked)} pinned open"
    written = [p.name for p in Path.cwd().iterdir() if p.is_file()]
    assert written == [], f"{name} wrote {written} from a failed run"
