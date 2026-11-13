"""Specialized demos: every interactive flow has to leave a PNG and no figure.

Answers are keyed on prompt text (see ``conftest``).  Assertions are on
artefacts rather than on raised exceptions because these demos catch their own
exceptions: a flow that failed half-way would otherwise look like a pass.  The UMAP
demo has the most branches (a file, an optional group column, downsampling, an
optional legend), so it gets the most cases here — including what the module looks
like when ``umap-learn`` cannot be imported at all.
"""
import importlib.util
import sys

import pandas as pd
import pytest

from biosuite.plotting import specialized_plots as sp


def _assert_saved(report, filename):
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == [filename], report["stdout"]
    assert report["empty_images"] == []
    assert report["figures_open"] == 0, f"{filename} left its figure pinned"


def test_gsea_default_scores_save_plot(run_demo):
    report = run_demo(sp.gsea_plot,
                      answers={"load ranked list from file": "n"})
    _assert_saved(report, "gsea.png")


def test_gsea_scores_from_csv_save_plot(run_demo, tmp_path):
    ranked = tmp_path / "ranked.csv"
    pd.DataFrame({"score": [3, 2, 1, 0.5, -0.2, -1, -2]}).to_csv(ranked, index=False)
    report = run_demo(sp.gsea_plot, answers={
        "load ranked list from file": "y",
        "file path": str(ranked),
        "score column": "score",
    })
    _assert_saved(report, "gsea.png")
    curve = report["result"]
    assert curve is None, "the demo saves rather than returning its figure"


def test_a_score_column_that_is_not_in_the_file_falls_back_to_simulated_scores(
        run_demo, tmp_path):
    """The user asked for a column that does not exist; the demo says so and draws
    the simulated curve it substituted (it used to say nothing at all)."""
    ranked = tmp_path / "ranked.csv"
    pd.DataFrame({"log2FC": [1.0, 2.0, 3.0]}).to_csv(ranked, index=False)

    report = run_demo(sp.gsea_plot, answers={
        "load ranked list from file": "y",
        "file path": str(ranked),
        "score column": "not_a_column",
    })

    _assert_saved(report, "gsea.png")
    assert "Column 'not_a_column' is not in the file (log2FC)" in report["stdout"], \
        report["stdout"]
    assert "simulated ranked list" in report["stdout"], report["stdout"]
    title = report["figure_title"]
    assert title.startswith("GSEA Running Sum (max ES = "), title


def test_motif_logo_default_alignment_saves_plot(run_demo):
    report = run_demo(sp.motif_logo, answers={"use default alignment": "y"})
    _assert_saved(report, "motif_logo.png")


def test_motif_logo_typed_sequences_save_plot(run_demo):
    report = run_demo(sp.motif_logo, answers={
        "use default alignment": "n",
        "<blank>": ["AAGT", "AAGT", "CAGT", "ACGT", ""],
    })
    _assert_saved(report, "motif_logo.png")


def test_motif_logo_stops_reading_sequences_at_end_of_input(run_demo, eof):
    """Piping BioSuite's output to a file closes stdin mid-loop; the demo has to
    finish with what it has, not die on EOFError."""
    report = run_demo(sp.motif_logo, answers={
        "use default alignment": "n",
        "<blank>": [eof],
    })

    _assert_saved(report, "motif_logo.png")
    assert "Error:" not in report["stdout"], report["stdout"]


def test_a_lower_case_and_padded_alignment_is_still_read(run_demo):
    report = run_demo(sp.motif_logo, answers={
        "use default alignment": "n",
        "<blank>": [" aagt ", "aagt", "cagt", ""],
    })
    _assert_saved(report, "motif_logo.png")


@pytest.mark.parametrize("default_data", ["y", "n"])
def test_sankey_diagram_saves_plot(run_demo, default_data):
    answers = {"use default data": default_data}
    if default_data == "n":
        answers.update({"node labels": "A,B,C", "source indices": "0,1",
                        "target indices": "1,2", "values": "10,20"})
    report = run_demo(sp.sankey_diagram, answers=answers)
    _assert_saved(report, "sankey.png")


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_default_data_saves_plot(run_demo):
    report = run_demo(sp.umap_plot, answers={"load data from file": "n"})
    _assert_saved(report, "umap.png")


@pytest.fixture
def umap_csv(tmp_path):
    def write(rows, extra_text_column=True):
        path = tmp_path / "embedding_input.csv"
        frame = pd.DataFrame({f"feature_{i}": [1.0 * r + i for r in range(len(rows))]
                             for i in range(4)})
        frame['x'] = range(len(frame))
        if extra_text_column:
            frame['group'] = ["a"] * len(frame)
        path.write_text(frame.to_csv(index=False), encoding="utf-8")
        return path

    return write


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_reduces_a_loaded_table_and_legends_its_groups(run_demo, umap_csv, monkeypatch):
    monkeypatch.setitem(sp.config, 'downsample_threshold', 1000)
    path = umap_csv(range(20))

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
        "group column": "group",
    })

    _assert_saved(report, "umap.png")
    assert report["figure_title"] == "UMAP Projection"
    assert report["legend_title"] == "Groups", "a table with few groups must be legended"


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_without_a_group_column_still_plots_every_sample(run_demo, umap_csv, monkeypatch):
    monkeypatch.setitem(sp.config, 'downsample_threshold', 1000)
    path = umap_csv(range(12), extra_text_column=False)

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
        "group column": "",
    })

    _assert_saved(report, "umap.png")
    assert report["legend_title"] == "Groups"


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_omits_the_legend_when_there_are_too_many_groups(run_demo, tmp_path, monkeypatch):
    monkeypatch.setitem(sp.config, 'downsample_threshold', 1000)
    frame = pd.DataFrame({f"feature_{i}": [float(r) + i for r in range(14)] for i in range(3)})
    frame['group'] = [f"g{n}" for n in range(14)]
    path = tmp_path / "many_groups.csv"
    path.write_text(frame.to_csv(index=False), encoding="utf-8")

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
        "group column": "group",
    })

    _assert_saved(report, "umap.png")
    assert report["legend_title"] is None, "14 groups would bury the plot in a legend"


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_downsamples_a_table_bigger_than_the_threshold(run_demo, tmp_path, monkeypatch):
    frame = pd.DataFrame({f"feature_{i}": [float(r) + i for r in range(60)] for i in range(3)})
    path = tmp_path / "big.csv"
    path.write_text(frame.to_csv(index=False), encoding="utf-8")
    monkeypatch.setitem(sp.config, 'downsample_threshold', 20)

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
        "group column": "",
    })

    _assert_saved(report, "umap.png")
    assert report["points_drawn"] == 20, "only the downsampled points may be drawn"


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_says_so_for_a_table_without_numbers(run_demo, tmp_path, monkeypatch):
    path = tmp_path / "text_only.csv"
    path.write_text("note,other\nhello,world\nsecond,third\n", encoding="utf-8")
    monkeypatch.setitem(sp.config, 'downsample_threshold', 1000)

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
    })

    assert report["images"] == []
    assert report["figures_open"] == 0
    assert "No numeric columns found." in report["stdout"], report["stdout"]


def test_umap_names_its_two_install_commands_when_the_library_is_missing(run_demo, monkeypatch):
    """Without the library the demo must say what to install and leave no plot."""
    monkeypatch.setattr(sp, "HAS_UMAP", False)

    report = run_demo(sp.umap_plot)

    assert report["images"] == []
    assert report["figures_open"] == 0
    assert "pip install umap-learn" in report["stdout"]
    assert "use PCA plot (option 2)" in report["stdout"]
    assert "Error:" not in report["stdout"]


def test_the_module_loads_without_umap_and_says_the_library_is_absent(tmp_path, monkeypatch):
    """``HAS_UMAP`` is decided at import time, so the guard is exercised by
    re-executing the module's top level with the library hidden."""
    monkeypatch.setitem(sys.modules, "umap.umap_", None)
    spec = importlib.util.spec_from_file_location(
        "biosuite.plotting.specialized_plots_without_umap", sp.__file__)
    fresh = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(fresh)

    assert fresh.HAS_UMAP is False
    assert not hasattr(fresh, "umap") or fresh.umap is None
    assert fresh.gsea_plot.__name__ == "gsea_plot", "the rest of the module still loads"
