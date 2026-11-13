"""A UMAP of a file larger than the threshold used to be a UMAP of a random sample.

``np.random.choice(n, threshold, replace=False)`` drew a new subset on every run,
and the demo printed nothing about having cut the table at all, so two runs of the
same file produced two different pictures of it.  ``maybe_downsample`` already
strides evenly (first and last kept) — the same consolidation cluster 28 made in
``biological_plots``.
"""
import inspect
import types

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import specialized_plots as sp

ROWS = 60


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def table(tmp_path):
    frame = pd.DataFrame({f"feature_{i}": [float(r) + i for r in range(ROWS)]
                          for i in range(3)})
    path = tmp_path / "big.csv"
    path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return path


@pytest.fixture
def recorded(monkeypatch):
    """A reducer that records which rows it was asked to embed."""
    seen = []

    class Recording:
        def __init__(self, *a, **kw):
            pass

        def fit_transform(self, data):
            seen.append([tuple(row) for row in data])
            return np.array([[float(i), 0.0] for i in range(len(data))])

    monkeypatch.setattr(sp, "umap", types.SimpleNamespace(UMAP=Recording))
    monkeypatch.setitem(sp.config, "downsample_threshold", 20)
    return seen


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_two_runs_of_one_file_embed_the_same_rows(run_demo, table, recorded):
    for _ in range(2):
        report = run_demo(sp.umap_plot, answers={
            "load data from file": "y",
            "file path": str(table),
            "group column": "",
        })
        assert report["images"] == ["umap.png"], report["stdout"]

    assert len(recorded) == 2
    assert recorded[0] == recorded[1], "the sample changed between two runs"
    assert len(recorded[0]) == 20


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_the_cut_is_announced_with_both_counts(run_demo, table, recorded):
    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(table),
        "group column": "",
    })

    assert "Drawing 20 of 60 samples" in report["stdout"], report["stdout"]


@pytest.mark.skipif(not sp.HAS_UMAP, reason="umap-learn not installed")
def test_the_stride_keeps_the_first_and_last_row(run_demo, table, recorded):
    run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(table),
        "group column": "",
    })

    # feature_0 is the row number, so the sampled rows can be read back.
    sampled = [row[0] for row in recorded[0]]
    assert sampled[0] == 0.0 and sampled[-1] == float(ROWS - 1)
    assert sampled == sorted(sampled), "the stride must keep the table in order"


def test_a_table_under_the_threshold_is_not_cut(run_demo, tmp_path, recorded, monkeypatch):
    """The positive control: no message, and every row embedded."""
    monkeypatch.setitem(sp.config, "downsample_threshold", 1000)
    small = tmp_path / "small.csv"
    frame = pd.DataFrame({f"feature_{i}": [float(r) + i for r in range(10)]
                          for i in range(3)})
    small.write_text(frame.to_csv(index=False), encoding="utf-8")

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(small),
        "group column": "",
    })

    assert "Drawing" not in report["stdout"], report["stdout"]
    assert len(recorded[0]) == 10


def test_the_module_no_longer_draws_a_random_sample():
    source = inspect.getsource(sp.umap_plot)

    assert "np.random.choice" not in source
    assert "maybe_downsample(" in source
