"""Specialized demos: declined saves, unreadable files and malformed numbers.

These are the branches a user hits by accident, and the contract is the same in
every one of them: nothing half-written is left on disk and no figure stays
registered, because a pinned figure costs memory for the life of the process.
"""
import pytest

from biosuite.plotting import specialized_plots as sp


def test_gsea_declined_save_writes_nothing_but_releases_figure(run_demo):
    report = run_demo(sp.gsea_plot, save=False,
                      answers={"load ranked list from file": "n"})
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0


def test_gsea_unreadable_file_falls_back_to_synthetic_scores(run_demo):
    report = run_demo(sp.gsea_plot, answers={
        "load ranked list from file": "y",
        "file path": "does-not-exist.csv",
    })
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["gsea.png"]
    assert report["figures_open"] == 0


def test_motif_logo_declined_save_releases_figure(run_demo):
    report = run_demo(sp.motif_logo, save=False,
                      answers={"use default alignment": "y"})
    assert report["images"] == []
    assert report["figures_open"] == 0


def test_motif_logo_without_sequences_falls_back_to_default(run_demo):
    report = run_demo(sp.motif_logo, answers={
        "use default alignment": "n",
        "<blank>": [""],
    })
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["motif_logo.png"]
    assert report["figures_open"] == 0


def test_sankey_malformed_edges_reports_and_leaks_no_figure(run_demo):
    report = run_demo(sp.sankey_diagram, answers={
        "use default data": "n",
        "node labels": "A,B,C",
        "source indices": "not-a-number",
    })
    assert "Error:" in report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_umap_unreadable_file_falls_back_to_synthetic_data(run_demo):
    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": "does-not-exist.csv",
    })
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["umap.png"]
    assert report["figures_open"] == 0
