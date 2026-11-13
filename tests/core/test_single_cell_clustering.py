"""Single-cell pipeline logic driven through an explicit scanpy/anndata stand-in.

scanpy and anndata are optional extras and are NOT installed in this
environment, so ``HAS_SCANPY`` is False and every statement after the guards in
``biosuite/core/single_cell.py`` is dead unless the library boundary is
stubbed.  These tests therefore drive the module through a small stand-in whose
behaviour is the *documented* behaviour of the few scanpy calls the module
makes (``pp.filter_cells/filter_genes/calculate_qc_metrics/normalize_total/
log1p/highly_variable_genes/scale/neighbors`` and ``tl.pca/umap/leiden/louvain/
rank_genes_groups/diffmap/dpt``).

No test here claims a biological result.  Every assertion targets BioSuite's
own code: input routing by extension, the ``min()`` caps on ``n_comps``/
``n_pcs``, the message strings users read, report assembly and cluster-size
ordering.  The missing-scanpy fallback is asserted against the real
environment first, so both sides of every guard are covered by behaviour users
actually get.
"""
import importlib
import importlib.util
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from biosuite.core import single_cell as scm

# The fixture matrix below is written so that every expected number can be
# counted by eye.  Genes:                G1  G2  G3  G4  MT-CO1 mt-Nd1
TOY_COUNTS = np.array([
    [40, 10, 20,  0,  30,  0],    # 30/100  = 30.0% mito -> dropped
    [20, 10,  0, 30,  40,  0],    # 40/100  = 40.0% mito -> dropped
    [10, 20, 30, 40,   0, 10],    # 10/110  ~  9.1% mito -> kept
    [25, 25, 25, 20,   5,  0],    #  5/100  =  5.0% mito -> kept
    [10, 10, 10, 10,   5,  0],    #  5/45   ~ 11.1% mito -> kept
    [35, 10, 10, 10,   0, 10],    # 10/75   ~ 13.3% mito -> kept
    [20, 20, 20, 20,  20,  0],    # 20/100  = 20.0% mito -> NOT < 20 -> dropped
    [45,  5,  5, 40,   5,  5],    # 10/105  ~  9.5% mito -> kept
], dtype=float)
TOY_VAR_NAMES = ['G1', 'G2', 'G3', 'G4', 'MT-CO1', 'mt-Nd1']
TOY_OBS_NAMES = [f'cell{i}' for i in range(len(TOY_COUNTS))]

# Every gene is non-zero in at least 3 cells, so filter_genes(min_cells=3)
# keeps the whole panel; mt-Nd1 sits exactly on that boundary.
SURVIVING_CELLS = ['cell2', 'cell3', 'cell4', 'cell5', 'cell7']


def toy_adata():
    return FakeAnnData(TOY_COUNTS.copy(), list(TOY_OBS_NAMES), list(TOY_VAR_NAMES))


# ── in-memory AnnData stand-in ───────────────────────────────────────────────

def _mask(values, length):
    """Normalise an indexer (slice / bool list / ndarray / Series) to a bool array."""
    if values is None or isinstance(values, slice):
        return np.ones(length, dtype=bool)
    if isinstance(values, pd.Series):
        return np.asarray(values.values, dtype=bool)
    arr = np.asarray(values, dtype=bool)
    if arr.shape != (length,):
        raise IndexError(f"indexer of length {arr.shape} for axis of length {length}")
    return arr


class FakeAnnData:
    """Rows are cells, columns are genes; X is dense, obs/var are DataFrames."""

    def __init__(self, X, obs_names, var_names):
        self.X = np.asarray(X, dtype=float)
        self.obs = pd.DataFrame(index=list(obs_names))
        self.var = pd.DataFrame(index=list(var_names))
        self.uns = {}
        self.obsm = {}
        self.layers = {}

    @classmethod
    def _parts(cls, X, obs, var, uns=None, obsm=None):
        obj = cls(X, obs.index, var.index)
        obj.obs = obs.copy()
        obj.var = var.copy()
        obj.uns = dict(uns or {})
        obj.obsm = dict(obsm or {})
        return obj

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_vars(self):
        return self.X.shape[1]

    @property
    def shape(self):
        return self.X.shape

    @property
    def var_names(self):
        return pd.Index(self.var.index)

    @var_names.setter
    def var_names(self, names):
        self.var.index = pd.Index(list(names))

    @property
    def obs_names(self):
        return pd.Index(self.obs.index)

    def var_names_make_unique(self):
        """anndata's contract: duplicate gene labels get a '#' per earlier copy."""
        names = [str(name) for name in self.var.index]
        seen = {}
        unique = []
        for name in names:
            seen[name] = seen.get(name, 0) + 1
            unique.append(name if seen[name] == 1 else name + '#' * (seen[name] - 1))
        self.var.index = pd.Index(unique)

    def __getitem__(self, key):
        rows, cols = key
        row_mask = _mask(rows, self.n_obs)
        col_mask = _mask(cols, self.n_vars)
        return FakeAnnData._parts(self.X[np.ix_(row_mask, col_mask)],
                                  self.obs[row_mask], self.var[col_mask],
                                  self.uns, self.obsm)

    def copy(self):
        return FakeAnnData._parts(self.X, self.obs, self.var, self.uns, self.obsm)

    @property
    def T(self):
        return FakeAnnData(self.X.T, self.var.index, self.obs.index)

    def _subset_obs(self, mask):
        mask = _mask(mask, self.n_obs)
        self.X = self.X[mask, :]
        self.obs = self.obs[mask]

    def _subset_var(self, mask):
        mask = _mask(mask, self.n_vars)
        self.X = self.X[:, mask]
        self.var = self.var[mask]


class Calls:
    """Ordered log of every stand-in call, with the kwargs the module passed."""

    def __init__(self):
        self.log = []

    def names(self):
        return [name for name, _kwargs in self.log]

    def kwargs_for(self, name):
        return [kwargs for n, kwargs in self.log if n == name]


def make_fake_scanpy(calls, read_error=None):
    sc = ModuleType('scanpy_stub')

    def record(name, **kwargs):
        calls.log.append((name, kwargs))

    class PP:
        @staticmethod
        def filter_cells(adata, min_genes=None, min_counts=None):
            record('filter_cells', min_genes=min_genes, min_counts=min_counts)
            adata._subset_obs((adata.X > 0).sum(axis=1) >= min_genes)

        @staticmethod
        def filter_genes(adata, min_cells=None, min_counts=None):
            record('filter_genes', min_cells=min_cells, min_counts=min_counts)
            adata._subset_var((adata.X > 0).sum(axis=0) >= min_cells)

        @staticmethod
        def calculate_qc_metrics(adata, qc_vars=(), percent_top=None, log1p=False,
                                 inplace=True):
            record('calculate_qc_metrics', qc_vars=list(qc_vars),
                   percent_top=percent_top, log1p=log1p, inplace=inplace)
            totals = adata.X.sum(axis=1)
            adata.obs['total_counts'] = totals
            adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
            for name in qc_vars:
                flag = np.asarray(adata.var[name].values, dtype=bool)
                mito = adata.X[:, flag].sum(axis=1)
                adata.obs[f'pct_counts_{name}'] = np.divide(
                    100.0 * mito, totals, out=np.zeros(adata.n_obs), where=totals > 0)

        @staticmethod
        def normalize_total(adata, target_sum=None):
            record('normalize_total', target_sum=target_sum)
            totals = adata.X.sum(axis=1, keepdims=True)
            adata.X = np.where(totals > 0, adata.X / totals * target_sum, adata.X)

        @staticmethod
        def log1p(adata):
            record('log1p')
            adata.X = np.log1p(adata.X)

        @staticmethod
        def highly_variable_genes(adata, n_top_genes=None, flavor=None, **kwargs):
            record('highly_variable_genes', n_top_genes=n_top_genes, flavor=flavor)
            variances = adata.X.var(axis=0)
            order = np.argsort(-variances, kind='stable')
            chosen = np.zeros(adata.n_vars, dtype=bool)
            chosen[order[:n_top_genes]] = True
            adata.var['highly_variable'] = chosen

        @staticmethod
        def scale(adata, max_value=None, zero_center=True):
            record('scale', max_value=max_value)
            mean = adata.X.mean(axis=0, keepdims=True)
            std = adata.X.std(axis=0, keepdims=True)
            adata.X = (adata.X - mean) / np.where(std == 0, 1.0, std)
            if max_value is not None:
                adata.X = np.clip(adata.X, -max_value, max_value)

        @staticmethod
        def neighbors(adata, n_neighbors=15, n_pcs=None, **kwargs):
            record('neighbors', n_neighbors=n_neighbors, n_pcs=n_pcs)
            adata.uns['neighbors'] = {'n_neighbors': n_neighbors, 'n_pcs': n_pcs}

    class TL:
        @staticmethod
        def pca(adata, svd_solver=None, n_comps=None, **kwargs):
            record('pca', svd_solver=svd_solver, n_comps=n_comps)
            n_comps = int(n_comps)
            # arpack (the solver the module asks for) needs n_comps < min(shape)
            if n_comps < 1 or n_comps >= min(adata.n_obs, adata.n_vars):
                raise ValueError(
                    f"arpack requires 0 < n_comps < min(n_obs, n_vars), got {n_comps}")
            rng = np.random.default_rng(7)
            adata.obsm['X_pca'] = rng.normal(size=(adata.n_obs, n_comps))

        @staticmethod
        def umap(adata, **kwargs):
            record('umap')
            adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()

        @staticmethod
        def leiden(adata, resolution=None, **kwargs):
            record('leiden', resolution=resolution)
            adata.obs['leiden'] = [str(i % 2) for i in range(adata.n_obs)]

        @staticmethod
        def louvain(adata, resolution=None, **kwargs):
            record('louvain', resolution=resolution)
            adata.obs['louvain'] = [str(i % 3) for i in range(adata.n_obs)]

        @staticmethod
        def rank_genes_groups(adata, groupby=None, method=None, **kwargs):
            record('rank_genes_groups', groupby=groupby, method=method)
            labels = list(adata.obs[groupby])
            groups = {}
            for label in pd.unique(pd.Series(labels)):
                idx = [i for i, value in enumerate(labels) if value == label]
                means = adata.X[idx, :].mean(axis=0)
                ranked = np.argsort(-means, kind='stable')
                groups[str(label)] = pd.DataFrame({
                    'names': list(adata.var_names[ranked]),
                    'scores': means[ranked],
                })
            adata.uns['rank_genes_groups'] = {'groups': groups, 'groupby': groupby}

        @staticmethod
        def diffmap(adata, **kwargs):
            record('diffmap')
            adata.obsm['X_diffmap'] = adata.obsm['X_pca'][:, :3].copy()

        @staticmethod
        def dpt(adata, n_dcs=None, **kwargs):
            record('dpt', n_dcs=n_dcs)
            adata.obs['dpt_pseudotime'] = np.linspace(0.0, 1.0, adata.n_obs)

    class Get:
        @staticmethod
        def rank_genes_groups_df(adata, group=None):
            record('rank_genes_groups_df', group=group)
            return adata.uns['rank_genes_groups']['groups'][str(group)]

    def _fail_if_broken():
        if read_error is not None:
            raise OSError(read_error)

    def read_h5ad(filepath, *args, **kwargs):
        record('read_h5ad', filepath=filepath)
        _fail_if_broken()
        return FakeAnnData(np.ones((2, 2)), ['c0', 'c1'], ['g0', 'g1'])

    def read_10x_h5(filepath, *args, **kwargs):
        record('read_10x_h5', filepath=filepath)
        _fail_if_broken()
        return FakeAnnData(np.ones((2, 2)), ['c0', 'c1'], ['g0', 'g1'])

    def read_csv(filepath, sep=',', *args, **kwargs):
        record('read_csv', filepath=filepath, sep=sep)
        _fail_if_broken()
        df = pd.read_csv(filepath, sep=sep, index_col=0)
        return FakeAnnData(df.values, df.index, df.columns)

    def read_10x_mtx(filepath, *args, **kwargs):
        record('read_10x_mtx', filepath=filepath)
        _fail_if_broken()
        return FakeAnnData(np.ones((4, 3)), ['b1', 'b2', 'b3', 'b4'], ['g1', 'g2', 'g3'])

    sc.pp = PP()
    sc.tl = TL()
    sc.get = Get()
    sc.read_h5ad = read_h5ad
    sc.read_10x_h5 = read_10x_h5
    sc.read_csv = read_csv
    sc.read_10x_mtx = read_10x_mtx
    return sc


@pytest.fixture
def scanpy_env(monkeypatch):
    """Turn the availability flag on and hand the module a recording stand-in."""
    calls = Calls()
    sc = make_fake_scanpy(calls)
    monkeypatch.setattr(scm, 'HAS_SCANPY', True, raising=False)
    monkeypatch.setattr(scm, 'sc', sc, raising=False)
    return sc, calls


# ── the environment as it really is: documented no-scanpy fallback ───────────

def test_check_single_cell_tools_matches_actual_importability():
    """Each engine is reported by its own import, not by scanpy's."""
    scanpy_there = importlib.util.find_spec('scanpy') is not None
    anndata_there = importlib.util.find_spec('anndata') is not None
    tools = scm.check_single_cell_tools()
    assert tools == {'scanpy': scanpy_there, 'anndata': anndata_there}
    assert tools['scanpy'] is scm.HAS_SCANPY
    assert tools['anndata'] is scm.HAS_ANNDATA


def test_the_probe_does_not_report_anndata_as_missing_when_only_scanpy_is(monkeypatch):
    """anndata and scanpy were imported in one try block, so an install with
    anndata present and scanpy absent claimed neither was available."""
    monkeypatch.setattr(scm, 'HAS_SCANPY', False, raising=False)
    monkeypatch.setattr(scm, 'HAS_ANNDATA', True, raising=False)
    assert scm.check_single_cell_tools() == {'scanpy': False, 'anndata': True}


def test_every_entry_point_degrades_when_scanpy_is_missing(monkeypatch):
    """What users without the optional extra get, line for line."""
    monkeypatch.setattr(scm, 'HAS_SCANPY', False, raising=False)
    monkeypatch.delattr(scm, 'sc', raising=False)

    adata, err = scm.load_count_matrix('whatever.csv')
    assert adata is None
    assert err == "scanpy not installed. Run: pip install scanpy anndata"

    sentinel = object()
    assert scm.qc_filter(sentinel) == (sentinel, "scanpy not installed")
    assert scm.normalize_and_log(sentinel) is sentinel
    assert scm.find_highly_variable_genes(sentinel) == (sentinel, [])
    assert scm.scale_and_pca(sentinel) is sentinel
    assert scm.compute_neighbors(sentinel) is sentinel
    assert scm.run_umap(sentinel) is sentinel
    assert scm.cluster_leiden(sentinel) == (sentinel, 0)
    assert scm.cluster_louvain(sentinel) == (sentinel, 0)
    assert scm.find_marker_genes(sentinel) == {}
    assert scm.compute_pseudotime(sentinel) is sentinel

    returned, report = scm.run_full_pipeline(sentinel)
    assert returned is sentinel
    assert report.message == "scanpy not installed"
    assert (report.num_cells, report.num_genes, report.num_clusters) == (0, 0, 0)
    assert report.cluster_counts == {}
    assert report.top_markers == []
    assert report.qc_stats == {}


def test_availability_probe_tracks_import_outcome(monkeypatch):
    """single_cell.py:11-16 — HAS_SCANPY is decided by the import, not hardcoded."""
    fake_sc = make_fake_scanpy(Calls())
    monkeypatch.setitem(sys.modules, 'scanpy', fake_sc)
    monkeypatch.setitem(sys.modules, 'anndata', ModuleType('anndata_stub'))
    try:
        importlib.reload(scm)
        assert scm.HAS_SCANPY is True
        assert scm.check_single_cell_tools() == {'scanpy': True, 'anndata': True}
    finally:
        sys.modules.pop('scanpy', None)
        sys.modules.pop('anndata', None)
        importlib.reload(scm)
        assert scm.HAS_SCANPY is False
        # scanpy genuinely cannot be imported here, so the real module has no `sc`.
        monkeypatch.delattr(scm, 'sc', raising=False)
        monkeypatch.delattr(scm, 'ad', raising=False)


# ── load_count_matrix routing ────────────────────────────────────────────────

@pytest.mark.parametrize('name,expected_reader', [
    ('counts.h5ad', 'read_h5ad'),
    ('counts.h5', 'read_10x_h5'),
    ('counts.csv', 'read_csv'),
    ('counts.tsv', 'read_csv'),
    ('counts.txt', 'read_csv'),
])
def test_reader_selected_by_extension(tmp_path, scanpy_env, name, expected_reader):
    sc, calls = scanpy_env
    path = tmp_path / name
    if name.endswith(('.h5ad', '.h5')):
        path.write_bytes(b'\x00')
    else:
        path.write_text("gene,c1,c2\nG1,1,2\nG2,3,4\n")

    adata, err = scm.load_count_matrix(str(path))

    assert err is None
    assert calls.names() == [expected_reader]


@pytest.mark.parametrize('suffix', ['.tsv', '.txt'])
def test_tab_separated_inputs_are_read_with_tab_separator(tmp_path, scanpy_env, suffix):
    sc, calls = scanpy_env
    path = tmp_path / f'matrix{suffix}'
    path.write_text("gene\tc1\tc2\nG1\t1\t2\nG2\t3\t4\n")

    adata, err = scm.load_count_matrix(str(path))

    assert err is None
    assert calls.log[0] == ('read_csv', {'filepath': str(path), 'sep': '\t'})


@pytest.mark.parametrize('suffix', ['.csv', '.tsv', '.txt'])
def test_dense_text_matrices_are_transposed_to_cells_by_genes(tmp_path, scanpy_env, suffix):
    """read_csv(...).T: samples come out as genes, so cells must end up on obs."""
    sc, calls = scanpy_env
    sep = '\t' if suffix != '.csv' else ','
    path = tmp_path / f'matrix{suffix}'
    path.write_text(sep.join(['gene', 'cellA', 'cellB']) + "\n"
                    + sep.join(['G1', '1', '2']) + "\n"
                    + sep.join(['G2', '3', '4']) + "\n")

    adata, err = scm.load_count_matrix(str(path))

    assert err is None
    assert list(adata.obs_names) == ['cellA', 'cellB']
    assert list(adata.var_names) == ['G1', 'G2']
    assert adata.X.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_explicit_h5ad_format_overrides_the_filename_extension(scanpy_env, tmp_path):
    sc, calls = scanpy_env
    path = tmp_path / 'mystery.bgz'
    path.write_bytes(b'\x00')

    adata, err = scm.load_count_matrix(str(path), file_format='h5ad')

    assert err is None
    assert calls.names() == ['read_h5ad']


def test_directory_input_uses_the_tenx_mtx_reader(scanpy_env, tmp_path):
    sc, calls = scanpy_env
    folder = tmp_path / 'scRNA_export'
    folder.mkdir()

    adata, err = scm.load_count_matrix(str(folder))

    assert err is None
    assert calls.names() == ['read_10x_mtx']
    assert (adata.n_obs, adata.n_vars) == (4, 3)


def test_unrecognized_input_returns_the_supported_shape_list(scanpy_env, tmp_path):
    sc, calls = scanpy_env
    path = tmp_path / 'reads.fasta'
    path.write_text(">seq\nACGT\n")

    adata, err = scm.load_count_matrix(str(path))

    assert adata is None
    assert calls.names() == []
    assert err == ("Unrecognized input: expected .h5ad/.h5/.csv/.tsv/.txt file "
                   "or a 10x directory")


def test_reader_failure_surfaces_as_a_message_not_an_exception(monkeypatch, tmp_path):
    calls = Calls()
    sc = make_fake_scanpy(calls, read_error='h5 file is truncated')
    monkeypatch.setattr(scm, 'HAS_SCANPY', True, raising=False)
    monkeypatch.setattr(scm, 'sc', sc, raising=False)
    path = tmp_path / 'counts.h5ad'
    path.write_bytes(b'\x00')

    adata, err = scm.load_count_matrix(str(path))

    assert adata is None
    assert err == 'h5 file is truncated'


# ── qc_filter ────────────────────────────────────────────────────────────────

def test_qc_filter_drops_high_mito_cells_and_reports_the_exact_counts(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    filtered, msg = scm.qc_filter(adata, min_genes=1, min_cells=1, max_pct_mito=20)

    assert list(filtered.obs_names) == SURVIVING_CELLS
    assert filtered.n_obs == 5
    assert msg == "QC: 8 → 5 cells (filtered 3 high-MT cells)"
    # 20.0% is not < 20, so cell6 must be filtered out although it is on the edge
    assert 'cell6' not in list(filtered.obs_names)


def test_qc_filter_flags_both_upper_and_lower_case_mito_prefixes(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    scm.qc_filter(adata, min_genes=1, min_cells=1, max_pct_mito=20)

    assert list(adata.var['mt']) == [False, False, False, False, True, True]


@pytest.mark.parametrize('threshold,expected_kept', [
    (5.0, []),
    (9.6, ['cell2', 'cell3', 'cell7']),
    (11.2, ['cell2', 'cell3', 'cell4', 'cell7']),
    (20.0, SURVIVING_CELLS),
    (100.0, list(TOY_OBS_NAMES)),
])
def test_qc_filter_threshold_is_live_not_inert(scanpy_env, threshold, expected_kept):
    """Mito fractions 5.0 / 9.09 / 9.52 / 11.11 / 13.33 / 20 / 30 / 40 %:
    each gate keeps exactly the cells counted by eye from the fixture."""
    sc, calls = scanpy_env
    adata = toy_adata()

    filtered, msg = scm.qc_filter(adata, min_genes=1, min_cells=1, max_pct_mito=threshold)

    assert list(filtered.obs_names) == expected_kept
    assert msg == f"QC: 8 → {len(expected_kept)} cells (filtered {8 - len(expected_kept)} high-MT cells)"


def test_qc_filter_forwards_thresholds_and_metric_options(scanpy_env):
    sc, calls = scanpy_env
    scm.qc_filter(toy_adata(), min_genes=123, min_cells=7, max_pct_mito=5)

    assert calls.log[0] == ('filter_cells', {'min_genes': 123, 'min_counts': None})
    assert calls.log[1] == ('filter_genes', {'min_cells': 7, 'min_counts': None})
    assert calls.log[2] == ('calculate_qc_metrics',
                            {'qc_vars': ['mt'], 'percent_top': None,
                             'log1p': False, 'inplace': True})


def test_qc_filter_makes_duplicate_gene_names_unique(scanpy_env):
    sc, calls = scanpy_env
    adata = FakeAnnData(np.ones((4, 3)), ['a', 'b', 'c', 'd'], ['G1', 'G1', 'G3'])

    filtered, msg = scm.qc_filter(adata, min_genes=1, min_cells=1, max_pct_mito=50)

    assert len(set(filtered.var_names)) == 3
    assert msg == "QC: 4 → 4 cells (filtered 0 high-MT cells)"


def test_qc_filter_on_flat_input_reports_zero_filtered_without_crashing(scanpy_env):
    """Constant mito fraction: the gate must still count honestly."""
    sc, calls = scanpy_env
    adata = FakeAnnData(np.ones((4, 3)), list('abcd'), ['G1', 'G2', 'G3'])

    filtered, msg = scm.qc_filter(adata, min_genes=1, min_cells=1, max_pct_mito=20)

    assert filtered.n_obs == 4
    assert float(filtered.obs['pct_counts_mt'].iloc[0]) == 0.0
    assert msg == "QC: 4 → 4 cells (filtered 0 high-MT cells)"


# ── normalization / HVG / PCA / neighbours / UMAP ────────────────────────────

def test_normalize_and_log_runs_cpg_then_log1p_in_order(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    out = scm.normalize_and_log(adata)

    assert out is adata
    assert calls.names() == ['normalize_total', 'log1p']
    assert calls.log[0][1] == {'target_sum': 1e4}
    expected = np.log1p(TOY_COUNTS[0] / TOY_COUNTS[0].sum() * 1e4)
    assert out.X[0] == pytest.approx(expected)


def test_find_highly_variable_genes_selects_then_subsets(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    variances = pd.Series(TOY_COUNTS.var(axis=0), index=TOY_VAR_NAMES)
    # variances: MT-CO1 199.6 > G4 185.9 > G1 152.7 > G3 93.8 > G2 42.2 > mt-Nd1 18.4
    top_by_variance = set(variances.nlargest(3).index)
    expected_hvg = [name for name in TOY_VAR_NAMES if name in top_by_variance]

    subset, hvg = scm.find_highly_variable_genes(adata, n_top_genes=3)

    assert calls.log[0] == ('highly_variable_genes', {'n_top_genes': 3, 'flavor': 'seurat'})
    assert hvg == expected_hvg == ['G1', 'G4', 'MT-CO1']
    assert subset.n_vars == 3
    assert list(subset.var_names) == expected_hvg
    # the two flattest genes must never make a top-3 cut
    assert 'G2' not in hvg and 'mt-Nd1' not in hvg


def test_find_highly_variable_genes_keeps_all_when_request_exceeds_panel(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    subset, hvg = scm.find_highly_variable_genes(adata, n_top_genes=2000)

    assert calls.log[0][1]['n_top_genes'] == 2000
    assert len(hvg) == 6
    assert subset.n_vars == 6


def test_scale_and_pca_caps_components_at_gene_dimension(scanpy_env):
    sc, calls = scanpy_env
    adata = FakeAnnData(TOY_COUNTS[:, :4], list(TOY_OBS_NAMES), ['G1', 'G2', 'G3', 'G4'])

    out = scm.scale_and_pca(adata, n_pcs=50)

    assert out is adata
    assert calls.log[0] == ('scale', {'max_value': 10})
    assert calls.log[1] == ('pca', {'svd_solver': 'arpack', 'n_comps': 3})
    assert out.obsm['X_pca'].shape == (8, 3)


def test_scale_and_pca_uses_requested_count_when_panel_is_wide(scanpy_env):
    sc, calls = scanpy_env
    rng = np.random.default_rng(3)
    X = rng.poisson(5, (20, 12)).astype(float)
    adata = FakeAnnData(X, [f'c{i}' for i in range(20)], [f'g{i}' for i in range(12)])

    scm.scale_and_pca(adata, n_pcs=5)

    assert calls.log[1][1]['n_comps'] == 5


@pytest.mark.parametrize('n_pcs,available,expected', [(30, 3, 3), (8, 12, 8), (30, 30, 30)])
def test_compute_neighbors_caps_pcs_at_available_components(scanpy_env, n_pcs,
                                                            available, expected):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obsm['X_pca'] = np.zeros((8, available))

    out = scm.compute_neighbors(adata, n_pcs=n_pcs)

    assert out is adata
    assert calls.log[0] == ('neighbors', {'n_neighbors': 15, 'n_pcs': expected})
    assert adata.uns['neighbors']['n_pcs'] == expected


def test_run_umap_stores_two_dimensional_embedding(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obsm['X_pca'] = np.arange(16.0).reshape(8, 2)

    out = scm.run_umap(adata)

    assert out is adata
    assert calls.names() == ['umap']
    assert out.obsm['X_umap'].shape == (8, 2)
    assert np.array_equal(out.obsm['X_umap'], np.arange(16.0).reshape(8, 2)[:, :2])


# ── clustering ───────────────────────────────────────────────────────────────

def test_cluster_leiden_counts_labels_and_forwards_resolution(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    out, n_clusters = scm.cluster_leiden(adata, resolution=1.2)

    assert calls.log[0] == ('leiden', {'resolution': 1.2})
    assert list(out.obs['leiden']) == ['0', '1', '0', '1', '0', '1', '0', '1']
    assert n_clusters == 2


def test_cluster_leiden_of_one_group_reports_one_cluster(monkeypatch):
    calls = Calls()
    sc = make_fake_scanpy(calls)
    sc.tl.leiden = staticmethod(lambda adata, resolution=None:
                               (calls.log.append(('leiden', {'resolution': resolution})),
                                adata.obs.__setitem__('leiden', ['4'] * adata.n_obs))[0])
    monkeypatch.setattr(scm, 'HAS_SCANPY', True, raising=False)
    monkeypatch.setattr(scm, 'sc', sc, raising=False)
    adata = toy_adata()

    out, n_clusters = scm.cluster_leiden(adata, resolution=0.2)

    assert n_clusters == 1
    assert out.obs['leiden'].nunique() == 1
    assert calls.log[0] == ('leiden', {'resolution': 0.2})


def test_cluster_louvain_counts_three_way_split(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    out, n_clusters = scm.cluster_louvain(adata, resolution=0.9)

    assert calls.log[0] == ('louvain', {'resolution': 0.9})
    assert list(out.obs['louvain']) == ['0', '1', '2', '0', '1', '2', '0', '1']
    assert n_clusters == 3


def test_cluster_louvain_default_resolution_is_half(scanpy_env):
    sc, calls = scanpy_env
    adata = FakeAnnData(np.ones((2, 3)), ['a', 'b'], ['g1', 'g2', 'g3'])

    out, n_clusters = scm.cluster_louvain(adata)

    assert calls.log[0] == ('louvain', {'resolution': 0.5})
    assert n_clusters == 2


# ── markers / pseudotime ─────────────────────────────────────────────────────

def test_find_marker_genes_keys_by_cluster_and_honours_n_genes(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obs['leiden'] = ['0', '0'] + ['1'] * 6

    markers = scm.find_marker_genes(adata, groupby='leiden', n_genes=2)

    assert calls.log[0] == ('rank_genes_groups', {'groupby': 'leiden', 'method': 'wilcoxon'})
    assert calls.kwargs_for('rank_genes_groups_df') == [{'group': '0'}, {'group': '1'}]
    assert sorted(markers) == ['0', '1']
    for label in ('0', '1'):
        rows = [i for i, value in enumerate(adata.obs['leiden']) if value == label]
        means = TOY_COUNTS[rows, :].mean(axis=0)
        assert markers[label] == list(np.array(TOY_VAR_NAMES)[np.argsort(-means, kind='stable')][:2])
    # group 0 = cells0-1 (mean MT-CO1 35, mean G1 30); group 1 = cells2-7 (G1 24.2, G4 23.3)
    assert markers['0'] == ['MT-CO1', 'G1']
    assert markers['1'] == ['G1', 'G4']


def test_find_marker_genes_truncates_at_gene_panel_size(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obs['leiden'] = ['0'] * 8

    markers = scm.find_marker_genes(adata, groupby='leiden', n_genes=99)

    assert list(markers.keys()) == ['0']
    assert len(markers['0']) == 6
    assert sorted(markers['0']) == sorted(TOY_VAR_NAMES)


def test_compute_pseudotime_runs_diffmap_then_dpt_with_n_nodes(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obsm['X_pca'] = np.random.default_rng(1).normal(size=(8, 4))

    out = scm.compute_pseudotime(adata, n_nodes=4)

    assert calls.names() == ['diffmap', 'dpt']
    assert calls.log[1] == ('dpt', {'n_dcs': 4})
    assert out.obs['dpt_pseudotime'].tolist() == pytest.approx(np.linspace(0.0, 1.0, 8))


def test_compute_pseudotime_default_node_count_is_ten(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obsm['X_pca'] = np.zeros((8, 3))

    scm.compute_pseudotime(adata)

    assert calls.log[1] == ('dpt', {'n_dcs': 10})


def test_pseudotime_roots_itself_when_the_caller_did_not(scanpy_env):
    """`dpt()` reads adata.uns['iroot'] and raises KeyError without it, so the
    function only ever worked on an object the caller had rooted by hand."""
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obsm['X_pca'] = np.arange(8, dtype=float).reshape(8, 1)
    assert 'iroot' not in adata.uns

    out = scm.compute_pseudotime(adata, n_nodes=4)

    assert out.uns['iroot'] == 7, "the cell furthest along PC1"
    assert calls.names() == ['diffmap', 'dpt']


def test_pseudotime_keeps_a_root_the_caller_chose(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.obsm['X_pca'] = np.arange(8, dtype=float).reshape(8, 1)
    adata.uns['iroot'] = 2

    assert scm.compute_pseudotime(adata).uns['iroot'] == 2


def _quiet_pseudotime_steps(sc, calls):
    """Replace the stand-in's diffmap/dpt, which insist on an X_pca the way the
    real pipeline leaves it, so the root choice can be tested on its own."""
    sc.tl.diffmap = lambda a: calls.log.append(('diffmap', {}))
    sc.tl.dpt = lambda a, n_dcs=None: calls.log.append(('dpt', {'n_dcs': n_dcs}))


def test_pseudotime_without_a_pca_starts_at_the_first_cell(scanpy_env):
    sc, calls = scanpy_env
    _quiet_pseudotime_steps(sc, calls)
    adata = toy_adata()
    assert 'X_pca' not in adata.obsm

    assert scm.compute_pseudotime(adata).uns['iroot'] == 0
    assert calls.names() == ['diffmap', 'dpt']


def test_pseudotime_with_an_empty_pca_starts_at_the_first_cell(scanpy_env):
    sc, calls = scanpy_env
    _quiet_pseudotime_steps(sc, calls)
    adata = toy_adata()
    adata.obsm['X_pca'] = np.zeros((0, 0))

    assert scm.compute_pseudotime(adata).uns['iroot'] == 0


def test_scale_and_pca_also_caps_components_at_the_cell_count(scanpy_env):
    """arpack needs n_comps below the *smaller* dimension. Capping by genes alone
    asked for 11 components from a 4-cell sample and crashed inside scipy."""
    sc, calls = scanpy_env
    rng = np.random.default_rng(3)
    X = rng.poisson(5, (4, 12)).astype(float)
    adata = FakeAnnData(X, [f'c{i}' for i in range(4)], [f'g{i}' for i in range(12)])

    scm.scale_and_pca(adata, n_pcs=50)

    assert calls.log[1] == ('pca', {'svd_solver': 'arpack', 'n_comps': 3})


def test_scale_and_pca_refuses_an_object_too_small_to_reduce(scanpy_env):
    """One cell has no variance to decompose; the request used to reach scipy,
    which answered 'arpack requires 0 < n_comps < min(n_obs, n_vars)'."""
    sc, calls = scanpy_env
    adata = FakeAnnData(np.array([[1.0, 2.0, 3.0]]), ['c1'], ['g1', 'g2', 'g3'])

    with pytest.raises(ValueError,
                       match=r"PCA needs at least two cells and two genes, got "
                             r"1 cells and 3 genes\."):
        scm.scale_and_pca(adata, n_pcs=50)

    assert calls.names() == [], "the refusal comes before any scaling"


# ── full pipeline + report formatting ────────────────────────────────────────

def test_run_full_pipeline_message_counts_and_cluster_sizes(scanpy_env):
    sc, calls = scanpy_env
    adata = toy_adata()

    out, report = scm.run_full_pipeline(adata, min_genes=1, max_pct_mito=20,
                                        n_top_genes=4, resolution=0.5)

    assert calls.names() == [
        'filter_cells', 'filter_genes', 'calculate_qc_metrics',
        'normalize_total', 'log1p', 'highly_variable_genes',
        'scale', 'pca', 'neighbors', 'umap', 'leiden',
        'rank_genes_groups', 'rank_genes_groups_df', 'rank_genes_groups_df',
    ]
    assert list(out.obs_names) == SURVIVING_CELLS
    assert report.num_cells == 5
    assert report.num_genes == out.n_vars == 4
    assert report.num_clusters == 2
    assert list(out.obs['leiden']) == ['0', '1', '0', '1', '0']
    assert sorted(report.cluster_counts.items()) == [('0', 3), ('1', 2)]
    assert report.qc_stats == {'hvg_count': 4}
    assert sorted(report.top_markers) == ['0', '1']
    assert all(len(v) == 4 for v in report.top_markers.values())

    parts = report.message.split(' | ')
    assert parts == [
        "QC: 8 → 5 cells (filtered 3 high-MT cells)",
        "Normalized and log-transformed",
        "Found 4 highly variable genes",
        "PCA computed",
        "Neighbors computed",
        "UMAP computed",
        "Leiden clustering: 2 clusters",
        "Marker genes found for 2 clusters",
    ]
    assert calls.kwargs_for('highly_variable_genes') == [{'n_top_genes': 4, 'flavor': 'seurat'}]
    # 5 surviving cells x 4 genes: the module caps at n_vars-1 = 3, which arpack accepts
    assert calls.kwargs_for('pca') == [{'svd_solver': 'arpack', 'n_comps': 3}]
    assert calls.kwargs_for('neighbors') == [{'n_neighbors': 15, 'n_pcs': 3}]


def test_run_full_pipeline_drops_sparse_genes_before_the_mito_gate(scanpy_env):
    """A gene non-zero in only 2 cells is filtered out; counts stay honest."""
    sc, calls = scanpy_env
    adata = toy_adata()
    adata.X[:, TOY_VAR_NAMES.index('mt-Nd1')] = [0, 0, 0, 10, 0, 0, 0, 5]

    out, report = scm.run_full_pipeline(adata, min_genes=1, max_pct_mito=20,
                                        n_top_genes=4, resolution=0.5)

    assert 'mt-Nd1' not in list(out.var_names)
    assert report.num_genes == 4
    assert report.num_cells == 5
    assert report.message.split(' | ')[0] == "QC: 8 → 5 cells (filtered 3 high-MT cells)"


def test_format_sc_report_lists_clusters_in_numeric_order():
    report = scm.SingleCellReport(
        num_cells=13, num_genes=40, num_clusters=3,
        cluster_counts={'10': 4, '2': 5, '1': 4},
        message='step one | step two')

    text = scm.format_sc_report(report)
    lines = text.splitlines()

    assert lines[:4] == ["=== Single-Cell RNA-seq Report ===",
                         "Cells: 13", "Genes: 40", "Clusters: 3"]
    assert lines[4:] == ["", "Cluster sizes:",
                         "  Cluster 1: 4 cells",
                         "  Cluster 2: 5 cells",
                         "  Cluster 10: 4 cells",
                         "", "Pipeline: step one | step two"]
    # string sorting would put 10 before 2; numeric sorting must not
    assert lines.index("  Cluster 2: 5 cells") < lines.index("  Cluster 10: 4 cells")


def test_format_sc_report_omits_absent_sections():
    report = scm.SingleCellReport(num_cells=2, num_genes=3, num_clusters=0)

    text = scm.format_sc_report(report)

    assert text == ("=== Single-Cell RNA-seq Report ===\n"
                    "Cells: 2\nGenes: 3\nClusters: 0")
    assert "Cluster sizes" not in text
    assert "Pipeline" not in text
