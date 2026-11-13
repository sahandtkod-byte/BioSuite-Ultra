"""biosuite.core.single_cell without scanpy installed.

The result record, every entry point's answer when the single-cell engine is
missing, and the report text — the branches that stay reachable on a machine
without scanpy.
"""
import numpy as np
import pytest


def _has_scanpy():
    try:
        import scanpy
        return True
    except ImportError:
        return False


class TestSingleCellDataclass:
    def test_report_defaults(self):
        from biosuite.core.single_cell import SingleCellReport
        r = SingleCellReport()
        assert r.num_cells == 0
        assert r.num_genes == 0
        assert r.num_clusters == 0
        assert r.cluster_counts == {}
        assert r.top_markers == []


class TestSingleCellNoDeps:
    """Tests that don't require scanpy."""

    def test_check_single_cell_tools(self):
        from biosuite.core.single_cell import check_single_cell_tools
        tools = check_single_cell_tools()
        assert 'scanpy' in tools
        assert 'anndata' in tools

    @pytest.mark.skipif(
        not _has_scanpy(), reason="scanpy not installed"
    )
    def test_load_count_matrix_csv(self, tmp_path):
        from biosuite.core.single_cell import load_count_matrix
        csv = tmp_path / "counts.csv"
        # Transposed CSV: genes in rows, cells in columns
        csv.write_text("gene1,gene2,gene3\ncell1,10,20,30\ncell2,15,25,35\n")
        adata, err = load_count_matrix(str(csv))
        assert err is None
        assert adata is not None

    def test_load_count_matrix_missing_file(self):
        from biosuite.core.single_cell import load_count_matrix
        adata, err = load_count_matrix("/nonexistent/file.h5ad")
        # Without scanpy, should return None with error message
        if not _has_scanpy():
            assert adata is None
            assert err is not None

    @pytest.mark.skipif(
        not _has_scanpy(), reason="scanpy not installed"
    )
    def test_normalize_and_log(self):
        import anndata as ad

        from biosuite.core.single_cell import normalize_and_log
        X = np.array([[10, 20, 0], [30, 0, 5], [0, 10, 40]])
        adata = ad.AnnData(X=X)
        adata = normalize_and_log(adata)
        # After normalization + log1p, values should be >= 0
        assert np.all(adata.X >= 0)

    @pytest.mark.skipif(
        not _has_scanpy(), reason="scanpy not installed"
    )
    def test_find_highly_variable_genes(self):
        import anndata as ad
        import scanpy as sc

        from biosuite.core.single_cell import find_highly_variable_genes
        np.random.seed(42)
        X = np.random.poisson(10, (50, 100))
        adata = ad.AnnData(X=X)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata_hvg, hvg = find_highly_variable_genes(adata, n_top_genes=10)
        assert isinstance(hvg, list)
        assert len(hvg) == 10
        assert all(isinstance(g, str) for g in hvg)
        assert len(set(hvg)) == 10  # no duplicates in the top-10

    @pytest.mark.skipif(
        not _has_scanpy(), reason="scanpy not installed"
    )
    def test_scale_and_pca(self):
        import anndata as ad

        from biosuite.core.single_cell import scale_and_pca
        np.random.seed(42)
        X = np.random.normal(5, 1, (30, 50))
        adata = ad.AnnData(X=X.astype(float))
        adata = scale_and_pca(adata, n_pcs=10)
        assert 'X_pca' in adata.obsm
        assert adata.obsm['X_pca'].shape[1] == 10

    @pytest.mark.skipif(
        not _has_scanpy(), reason="scanpy not installed"
    )
    def test_run_umap(self):
        import anndata as ad
        import scanpy as sc

        from biosuite.core.single_cell import run_umap
        np.random.seed(42)
        X = np.random.normal(5, 1, (30, 20))
        adata = ad.AnnData(X=X.astype(float))
        sc.pp.scale(adata)
        sc.tl.pca(adata, n_comps=10)
        sc.pp.neighbors(adata, n_neighbors=5, n_pcs=10)
        adata = run_umap(adata)
        assert 'X_umap' in adata.obsm

    @pytest.mark.skipif(
        not _has_scanpy(), reason="scanpy not installed"
    )
    def test_cluster_cells(self):
        import anndata as ad
        import scanpy as sc

        from biosuite.core.single_cell import cluster_leiden
        np.random.seed(42)
        X = np.random.normal(5, 1, (50, 20))
        adata = ad.AnnData(X=X.astype(float))
        sc.pp.scale(adata)
        sc.tl.pca(adata, n_comps=10)
        sc.pp.neighbors(adata, n_neighbors=5, n_pcs=10)
        adata, n_clusters = cluster_leiden(adata, resolution=0.5)
        assert n_clusters >= 1
        assert 'leiden' in adata.obs.columns


class TestSingleCellFormatReport:
    def test_format_sc_report(self):
        from biosuite.core.single_cell import SingleCellReport, format_sc_report
        r = SingleCellReport(num_cells=1000, num_genes=2000, num_clusters=5,
                             cluster_counts={'0': 300, '1': 200, '2': 250, '3': 150, '4': 100},
                             message="QC done | PCA computed")
        text = format_sc_report(r)
        assert "Single-Cell RNA-seq Report" in text
        assert "1000" in text
        assert "Cluster 0" in text
