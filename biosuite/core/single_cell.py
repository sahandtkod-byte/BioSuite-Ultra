"""
Single-cell RNA-seq analysis using scanpy.

Provides a complete scRNA-seq pipeline: QC, normalization, dimensionality
reduction, clustering, marker detection, and visualization.
Requires: scanpy, anndata (pip installable).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


@dataclass
class SingleCellReport:
    num_cells: int = 0
    num_genes: int = 0
    num_clusters: int = 0
    cluster_counts: dict = field(default_factory=dict)
    top_markers: list = field(default_factory=list)
    qc_stats: dict = field(default_factory=dict)
    message: str = ""


def check_single_cell_tools():
    return {'scanpy': HAS_SCANPY, 'anndata': HAS_SCANPY}


def load_count_matrix(filepath, file_format='auto'):
    if not HAS_SCANPY:
        return None, "scanpy not installed. Run: pip install scanpy anndata"

    try:
        if file_format == 'h5ad' or filepath.endswith('.h5ad'):
            adata = sc.read_h5ad(filepath)
        elif filepath.endswith('.h5'):
            adata = sc.read_10x_h5(filepath)
        elif filepath.endswith('.csv'):
            adata = sc.read_csv(filepath).T
        elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
            adata = sc.read_csv(filepath, sep='\t').T
        else:
            adata = sc.read_10x_mtx(filepath)
        return adata, None
    except Exception as e:
        return None, str(e)


def qc_filter(adata, min_genes=200, min_cells=3, max_pct_mito=20):
    if not HAS_SCANPY:
        return adata, "scanpy not installed"

    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    before = adata.n_obs
    adata = adata[adata.obs.pct_counts_mt < max_pct_mito, :].copy()
    after = adata.n_obs

    return adata, f"QC: {before} → {after} cells (filtered {before - after} high-MT cells)"


def normalize_and_log(adata):
    if not HAS_SCANPY:
        return adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def find_highly_variable_genes(adata, n_top_genes=2000):
    if not HAS_SCANPY:
        return adata, []
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    hvg = adata.var_names[adata.var.highly_variable].tolist()
    adata = adata[:, adata.var.highly_variable].copy()
    return adata, hvg


def scale_and_pca(adata, n_pcs=50):
    if not HAS_SCANPY:
        return adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(n_pcs, adata.n_vars - 1))
    return adata


def compute_neighbors(adata, n_pcs=30):
    if not HAS_SCANPY:
        return adata
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(n_pcs, adata.obsm['X_pca'].shape[1]))
    return adata


def run_umap(adata):
    if not HAS_SCANPY:
        return adata
    sc.tl.umap(adata)
    return adata


def cluster_leiden(adata, resolution=0.5):
    if not HAS_SCANPY:
        return adata, 0
    sc.tl.leiden(adata, resolution=resolution)
    n_clusters = adata.obs['leiden'].nunique()
    return adata, n_clusters


def cluster_louvain(adata, resolution=0.5):
    if not HAS_SCANPY:
        return adata, 0
    sc.tl.louvain(adata, resolution=resolution)
    n_clusters = adata.obs['louvain'].nunique()
    return adata, n_clusters


def find_marker_genes(adata, groupby='leiden', n_genes=10):
    if not HAS_SCANPY:
        return {}
    sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon')
    markers = {}
    for cluster in adata.obs[groupby].unique():
        genes = sc.get.rank_genes_groups_df(adata, group=str(cluster))
        markers[cluster] = genes.head(n_genes)['names'].tolist()
    return markers


def compute_pseudotime(adata, n_nodes=10):
    if not HAS_SCANPY:
        return adata
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata, n_dcs=n_nodes)
    return adata


def run_full_pipeline(adata, min_genes=200, max_pct_mito=20,
                      n_top_genes=2000, resolution=0.5):
    if not HAS_SCANPY:
        return adata, SingleCellReport(message="scanpy not installed")

    log_messages = []

    adata, msg = qc_filter(adata, min_genes=min_genes, max_pct_mito=max_pct_mito)
    log_messages.append(msg)

    adata = normalize_and_log(adata)
    log_messages.append("Normalized and log-transformed")

    adata, hvg = find_highly_variable_genes(adata, n_top_genes=n_top_genes)
    log_messages.append(f"Found {len(hvg)} highly variable genes")

    adata = scale_and_pca(adata)
    log_messages.append("PCA computed")

    adata = compute_neighbors(adata)
    log_messages.append("Neighbors computed")

    adata = run_umap(adata)
    log_messages.append("UMAP computed")

    adata, n_clusters = cluster_leiden(adata, resolution=resolution)
    log_messages.append(f"Leiden clustering: {n_clusters} clusters")

    markers = find_marker_genes(adata)
    log_messages.append(f"Marker genes found for {len(markers)} clusters")

    cluster_counts = adata.obs['leiden'].value_counts().to_dict()

    report = SingleCellReport(
        num_cells=adata.n_obs,
        num_genes=adata.n_vars,
        num_clusters=n_clusters,
        cluster_counts=cluster_counts,
        top_markers=markers,
        qc_stats={'hvg_count': len(hvg)},
        message=' | '.join(log_messages)
    )

    return adata, report


def format_sc_report(report):
    lines = [
        "=== Single-Cell RNA-seq Report ===",
        f"Cells: {report.num_cells}",
        f"Genes: {report.num_genes}",
        f"Clusters: {report.num_clusters}",
    ]
    if report.cluster_counts:
        lines.append("\nCluster sizes:")
        for cl, count in sorted(report.cluster_counts.items(), key=lambda x: int(x[0])):
            lines.append(f"  Cluster {cl}: {count} cells")
    if report.message:
        lines.append(f"\nPipeline: {report.message}")
    return '\n'.join(lines)
