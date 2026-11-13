"""Batch exporters and the markdown story: one artefact per demo in the batch.

Both exporters run the same demo list, so artefacts are counted against that list
rather than against a literal — when the Venn demo was dropped from the PDF only,
a hard-coded number would have aged into the wrong answer.  The file names are
pinned separately, because writing them was the bug: 16 of the 29 plots used to
be skipped without a word.
"""
import os
import re

import matplotlib.pyplot as plt

from biosuite.plotting import biological_plots as bp
from biosuite.plotting import math_plots as mp
from biosuite.plotting import specialized_plots as sp

#: ``umap_plot`` returns without drawing when umap-learn is absent.
EXPECTED_PLOTS = {
    "alignment.png", "barplot.png", "boxplot.png", "circos.png",
    "clustered_heatmap.png", "cosine.png", "cubic.png", "dotplot.png",
    "exponential.png", "gsea.png", "heatmap.png", "linear.png", "logistic.png",
    "manhattan.png", "maplot.png", "motif_logo.png", "pca.png", "qqplot.png",
    "quadratic.png", "raincloud.png", "ridge.png", "sankey.png", "scatter.png",
    "sine.png", "timeseries.png", "umap.png", "venn.png", "violin.png",
    "volcano.png",
}
EXPECTED_NAMES = EXPECTED_PLOTS if sp.HAS_UMAP else EXPECTED_PLOTS - {"umap.png"}
EXPECTED_DEMOS = len(bp._batch_plot_functions()) - (0 if sp.HAS_UMAP else 1)


def _pdf_pages(path):
    data = path.read_bytes()
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", data)]
    assert counts, "no /Count entry — the PDF page tree is not what we read"
    return max(counts)


def test_export_all_to_folder_writes_every_plot_and_restores_state(tmp_path, monkeypatch):
    """One non-empty PNG per demo, and nothing left patched behind the run.

    The batch swaps ``ask_save_plot`` in three modules and edits the config for
    its own run; a leak there would be worse than a broken export, because every
    later plot in the process would save through its silent saver.
    """
    from biosuite.core.utils import ask_save_plot, config

    folder = tmp_path / "exports"
    monkeypatch.chdir(tmp_path)
    format_before, quiet_before = config["save_format"], config.get("quiet")
    figures_before = set(plt.get_fignums())

    bp.export_all_to_folder(str(folder))

    files = {p.name for p in folder.iterdir()}
    assert files == EXPECTED_NAMES
    assert len(files) == EXPECTED_DEMOS
    assert [p.name for p in folder.iterdir() if p.stat().st_size == 0] == []
    # The batch replaces ask_save_plot with its own saver, which owes the same
    # figure release as the one it stands in for.  Measured as a delta because the
    # pyplot registry is process-global and earlier suites are not always clean.
    assert set(plt.get_fignums()) == figures_before
    for module in (bp, mp, sp):
        assert module.ask_save_plot is ask_save_plot, module.__name__
    assert config["save_format"] == format_before
    assert config.get("quiet") == quiet_before
    assert os.getcwd() == str(tmp_path)


def test_batch_export_to_pdf_writes_a_page_per_demo(tmp_path):
    figures_before = set(plt.get_fignums())
    report = tmp_path / "report.pdf"
    bp.batch_export_to_pdf(str(report))

    assert report.exists()
    assert report.stat().st_size > 1000
    assert _pdf_pages(report) == EXPECTED_DEMOS
    assert set(plt.get_fignums()) == figures_before



def test_generate_markdown_story_lists_every_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = ["volcano_plot", "ma_plot", "dot_plot"]
    bp.generate_markdown_story(names, output_path=str(tmp_path / "story.md"))

    text = (tmp_path / "story.md").read_text(encoding="utf-8")
    assert f"This report contains {len(names)} exported plots." in text
    for name in names:
        assert f"![{name}]({name}.png)" in text
    assert text.count("### ") == len(names)


def test_batch_default_input_drives_demos_to_their_defaults():
    """The shared answer map is what keeps the two exporters in step."""
    assert bp._batch_default_input("Number of sets (2 or 3): ") == '2'
    assert bp._batch_default_input("Load data from file? (y/n): ") == 'n'
    assert bp._batch_default_input("Use default data? (y/n): ") == 'y'
    assert bp._batch_default_input("Amplitude A (default 1): ") == ''


def _failing_demo(name, message):
    """A stand-in demo that raises, with the name the batch reports back."""
    def demo(*args, **kwargs):
        raise RuntimeError(message)

    demo.__name__ = name
    return demo


def test_one_failing_demo_does_not_stop_the_folder_export(tmp_path, monkeypatch, capsys):
    """A batch that dies on the first bad plot tells the user nothing about the
    other 28; a batch that reports and carries on tells them exactly what is
    missing from the folder — and still takes its own patches back.
    """
    from biosuite.core.utils import ask_save_plot, config

    folder = tmp_path / "exports"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bp, "volcano_plot", _failing_demo("volcano_plot", "no data"))
    quiet_before, format_before = config.get("quiet"), config["save_format"]

    bp.export_all_to_folder(str(folder))

    assert "Warning: volcano_plot failed: no data" in capsys.readouterr().out
    assert {p.name for p in folder.iterdir()} == EXPECTED_NAMES - {"volcano.png"}
    assert bp.ask_save_plot is ask_save_plot
    assert config.get("quiet") == quiet_before
    assert config["save_format"] == format_before
    assert os.getcwd() == str(tmp_path)


def test_one_failing_demo_loses_one_page_and_says_so(tmp_path, monkeypatch, capsys):
    """Same contract for the PDF: the report still closes, one page shorter."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mp, "sine_plot", _failing_demo("sine_plot", "bad parameters"))
    report = tmp_path / "report.pdf"
    quiet_before = bp.config.get("quiet")

    bp.batch_export_to_pdf(str(report))

    out = capsys.readouterr().out
    assert "Warning: sine_plot failed: bad parameters" in out
    assert f"PDF saved: {report}" in out
    assert _pdf_pages(report) == EXPECTED_DEMOS - 1
    assert bp.config.get("quiet") == quiet_before
