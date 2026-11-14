"""The UMAP demo's branches, exercised without umap-learn installed.

``umap-learn`` is optional and CI does not install it, so the integration tests
in ``test_specialized_plot_flows`` skip the demo's default-data, empty-table,
downsampling and many-groups branches there.  A stub reducer stands in for the
library so every branch runs deterministically in any environment, and the
import-time ``HAS_UMAP = True`` line is reached by re-executing the module with a
fake ``umap`` package injected.
"""
import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import specialized_plots as sp


class _StubReducer:
    """A deterministic stand-in for ``umap.UMAP``: one row in, one point out."""

    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, data):
        return np.array([[float(i), float(i) * 0.5] for i in range(len(data))])


@pytest.fixture
def stub_umap(monkeypatch):
    """Make the demo believe umap-learn is present, without importing it."""
    monkeypatch.setattr(sp, "umap", types.SimpleNamespace(UMAP=_StubReducer),
                        raising=False)
    monkeypatch.setattr(sp, "HAS_UMAP", True)


def _write_csv(tmp_path, rows, groups=None):
    frame = pd.DataFrame({f"feature_{i}": [float(r) + i for r in range(rows)]
                          for i in range(3)})
    if groups is not None:
        frame["group"] = groups
    path = tmp_path / "embedding_input.csv"
    path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return path


def test_the_default_data_branch_draws_a_hundred_points(run_demo, stub_umap):
    report = run_demo(sp.umap_plot, answers={"load data from file": "n"})

    assert report["images"] == ["umap.png"], report["stdout"]
    assert report["figure_title"] == "UMAP (default data)"
    assert report["points_drawn"] == 100
    assert report["figures_open"] == 0


def test_an_unreadable_file_falls_back_to_the_default_data(run_demo, stub_umap):
    """A missing file makes ``load_dataframe_safe`` return None; the demo must
    fall through to the synthetic data, not crash on ``df.select_dtypes``."""
    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": "does-not-exist.csv",
    })

    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["images"] == ["umap.png"], report["stdout"]
    assert report["figure_title"] == "UMAP (default data)"
    assert report["figures_open"] == 0


def test_a_table_without_numeric_columns_says_so_and_draws_nothing(
        run_demo, stub_umap, tmp_path):
    path = tmp_path / "text_only.csv"
    path.write_text("note,other\nhello,world\nsecond,third\n", encoding="utf-8")

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
    })

    assert "No numeric columns found." in report["stdout"], report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0


def test_a_table_over_the_threshold_is_downsampled(run_demo, stub_umap, tmp_path,
                                                   monkeypatch):
    monkeypatch.setitem(sp.config, "downsample_threshold", 20)
    path = _write_csv(tmp_path, 60)

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
        "group column": "",
    })

    assert "Drawing 20 of 60 samples" in report["stdout"], report["stdout"]
    assert report["points_drawn"] == 20
    assert report["images"] == ["umap.png"], report["stdout"]


def test_more_than_ten_groups_omits_the_legend(run_demo, stub_umap, tmp_path,
                                               monkeypatch):
    monkeypatch.setitem(sp.config, "downsample_threshold", 1000)
    path = _write_csv(tmp_path, 14, groups=[f"g{n}" for n in range(14)])

    report = run_demo(sp.umap_plot, answers={
        "load data from file": "y",
        "file path": str(path),
        "group column": "group",
    })

    assert report["figure_title"] == "UMAP Projection"
    assert report["legend_title"] is None, "14 groups would bury the plot in a legend"
    assert report["points_drawn"] == 14
    assert report["images"] == ["umap.png"], report["stdout"]


def test_the_module_sets_has_umap_when_the_library_imports(monkeypatch):
    """``HAS_UMAP = True`` only runs when ``import umap.umap_`` succeeds, which
    never happens in CI; re-executing the module with a fake package reaches it."""
    leaf = types.ModuleType("umap.umap_")
    leaf.UMAP = _StubReducer
    parent = types.ModuleType("umap")
    parent.__path__ = []
    parent.umap_ = leaf
    monkeypatch.setitem(sys.modules, "umap", parent)
    monkeypatch.setitem(sys.modules, "umap.umap_", leaf)

    spec = importlib.util.spec_from_file_location(
        "biosuite.plotting.specialized_plots_with_umap", sp.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    assert fresh.HAS_UMAP is True
    assert fresh.umap is leaf