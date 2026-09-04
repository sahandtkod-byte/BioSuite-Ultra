"""Tests for core/single_cell graceful-missing-deps contract."""
import pytest

from biosuite.core import single_cell as scm


def test_tool_check_shape():
    tools = scm.check_single_cell_tools()
    assert set(tools) == {'scanpy', 'anndata'}
    assert tools['scanpy'] == scm.HAS_SCANPY


def test_graceful_when_scanpy_missing(tmp_path):
    """Functions degrade to (None/input, error-string) without scanpy."""
    if scm.HAS_SCANPY:
        pytest.skip("scanpy installed — loaded path covered by smoke test")
    adata, err = scm.load_count_matrix(str(tmp_path / "x.csv"))
    assert adata is None and "scanpy" in err
    out, msg = scm.qc_filter(None)
    assert "scanpy" in msg
    adata2, report = scm.run_full_pipeline(None)
    assert "scanpy" in report.message
    assert report.num_cells == 0


@pytest.mark.skipif(not scm.HAS_SCANPY, reason="scanpy not installed")
def test_pipeline_smoke_toy_data():
    import anndata as ad
    import numpy as np
    rng = np.random.default_rng(0)
    counts = rng.poisson(3, (60, 300)).astype(float)
    var_names = [f"Gene{i}" for i in range(300)]
    var_names[0] = "MT-CO1"          # mitochondrial marker for QC path
    var_names[1] = "mt-Nd1"
    X = ad.AnnData(counts)
    X.var_names = var_names
    X.obs_names = [f"cell{i}" for i in range(60)]
    adata, report = scm.run_full_pipeline(X, min_genes=1, resolution=0.5)
    assert report.num_cells > 0
    assert report.num_clusters >= 1
    assert "QC" in report.message


def test_format_sc_report_output():
    r = scm.SingleCellReport(num_cells=50, num_genes=1000, num_clusters=3,
                             cluster_counts={'0': 30, '1': 15, '2': 5},
                             message="ok")
    txt = scm.format_sc_report(r)
    assert "Cells: 50" in txt and "Cluster 0: 30 cells" in txt
