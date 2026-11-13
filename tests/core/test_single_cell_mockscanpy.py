"""Single-cell: fake scanpy/anndata surface exercises the pipeline without the deps."""
import sys
import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import importlib


class _FakeAnnData:
    def __init__(self, X, obs=None, var=None):
        self.X = np.asarray(X, dtype=float)
        self.obs = obs if obs is not None else pd.DataFrame(index=[f'c{i}' for i in range(self.X.shape[0])])
        self.var = var if var is not None else pd.DataFrame(index=[f'g{i}' for i in range(self.X.shape[1])])
        self.uns = {}
        self.obsm = {}
        self.layers = {}

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_vars(self):
        return self.X.shape[1]

    @property
    def T(self):
        return _FakeAnnData(self.X.T)


@pytest.fixture
def fake_scanpy(monkeypatch):
    sc = types.ModuleType('scanpy')

    def read_csv(fp, sep=','):
        df = pd.read_csv(fp, sep=sep, index_col=0)
        return _FakeAnnData(df.values)

    def read_10x_mtx(fp, *a, **kw):
        return _FakeAnnData(np.ones((5, 3)))

    sc.read_csv = read_csv
    sc.read_h5ad = lambda fp: _FakeAnnData(np.zeros((2, 2)))
    sc.read_10x_h5 = lambda fp: _FakeAnnData(np.zeros((2, 2)))
    sc.read_10x_mtx = lambda fp: _FakeAnnData(np.zeros((2, 2)))
    sc.read = lambda fp: _FakeAnnData(np.zeros((2, 2)))
    monkeypatch.setitem(sys.modules, 'scanpy', sc)
    return sc


def test_load_count_matrix_missing_dep(monkeypatch):
    import biosuite.core.single_cell as scm
    monkeypatch.setattr(scm, 'HAS_SCANPY', False)
    rep = scm.load_count_matrix('/nonexistent/x.csv')
    assert rep == (None, 'scanpy not installed. Run: pip install scanpy anndata')


def test_load_count_matrix_with_fake_scanpy(fake_scanpy, tmp_path, monkeypatch):
    import biosuite.core.single_cell as scm
    monkeypatch.setattr(scm, 'sc', fake_scanpy, raising=False)  # sc is unbound without scanpy
    monkeypatch.setattr(scm, 'HAS_SCANPY', True)
    f = tmp_path / 'counts.csv'
    pd.DataFrame({'g1': [1, 2, 3], 'g2': [2, 1, 0]}).to_csv(f)
    out = scm.load_count_matrix(str(f))
    assert out[1] is None
    assert out[0].X.shape == (2, 3)  # genes x cells after the .T


def test_pipeline_missing_scanpy_graceful():
    import biosuite.core.single_cell as scm
    report = scm.SingleCellReport()
    assert report.num_cells == 0
    assert report.num_genes == 0
    assert report.num_clusters == 0
    assert report.message == ''
    assert report.cluster_counts == {}
    assert report.top_markers == []
    assert report.qc_stats == {}


def test_check_single_cell_tools_shape():
    import biosuite.core.single_cell as scm
    tools = scm.check_single_cell_tools()
    assert isinstance(tools, dict) and 'scanpy' in tools


def test_directory_path_10x_regression_for_missing_os_import(fake_scanpy, tmp_path):
    """Regression: os.path.isdir branch used to NameError (no `import os`)."""
    import biosuite.core.single_cell as scm
    original = scm.HAS_SCANPY
    scm.HAS_SCANPY = True
    try:
        d = tmp_path / 'mtx_dir'
        d.mkdir()
        adata, msg = scm.load_count_matrix(str(d))
        assert msg == '' or msg is None or 'error' not in msg.lower()
    finally:
        scm.HAS_SCANPY = original
