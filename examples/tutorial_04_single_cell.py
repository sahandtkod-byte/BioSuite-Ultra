"""
BioSuite Tutorial 4: Single-Cell RNA-seq Analysis

This notebook demonstrates single-cell analysis using scanpy:
- Quality control
- Normalization
- Dimensionality reduction
- Clustering
- Marker gene detection
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

print("BioSuite Single-Cell Analysis Tutorial")
print("=" * 50)

# %%
# ## 1. Check if scanpy is available

try:
    import scanpy as sc
    import anndata as ad
    print(f"Scanpy version: {sc.__version__}")
    print(f"AnnData version: {ad.__version__}")
    HAS_SCANPY = True
except ImportError:
    print("Scanpy not installed. Install with: pip install scanpy anndata")
    HAS_SCANPY = False

# %%
# ## 2. Create Synthetic Single-Cell Data

if HAS_SCANPY:
    # Generate synthetic scRNA-seq data
    np.random.seed(42)
    n_cells = 500
    n_genes = 1000

    # Create count matrix
    counts = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))

    # Add some differentially expressed genes for clusters
    # Cluster 0: genes 0-49 high
    counts[:100, :50] *= 5
    # Cluster 1: genes 50-99 high
    counts[100:200, 50:100] *= 5
    # Cluster 2: genes 100-149 high
    counts[200:300, 100:150] *= 5

    # Create AnnData object
    adata = sc.AnnData(X=counts.astype(np.float32))
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Add some metadata
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], n_cells)

    print(f"Created AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")

# %%
# ## 3. Quality Control

if HAS_SCANPY:
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('Gene_') & (
        adata.var_names.isin([f'Gene_{i}' for i in range(10)])
    )
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    print(f"Before QC: {adata.n_obs} cells")
    print(f"  Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.0f}")
    print(f"  Mean UMI per cell: {adata.obs['total_counts'].mean():.0f}")

    # Filter cells
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")

# %%
# ## 4. Normalization

if HAS_SCANPY:
    # Normalize to 10,000 reads per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Log transform
    sc.pp.log1p(adata)

    print("Normalized and log-transformed")

# %%
# ## 5. Find Highly Variable Genes

if HAS_SCANPY:
    # Find HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=200, flavor='seurat')

    n_hvg = adata.var['highly_variable'].sum()
    print(f"Found {n_hvg} highly variable genes")

    # Subset to HVGs
    adata_hvg = adata[:, adata.var.highly_variable].copy()

# %%
# ## 6. Scale and PCA

if HAS_SCANPY:
    # Scale data
    sc.pp.scale(adata_hvg, max_value=10)

    # PCA
    sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=50)

    print(f"PCA explained variance ratio (first 5): "
          f"{adata_hvg.uns['pca']['variance_ratio'][:5].round(3)}")

# %%
# ## 7. Compute Neighbors and UMAP

if HAS_SCANPY:
    # Compute neighbor graph
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)

    # UMAP
    sc.tl.umap(adata_hvg)

    print("Computed neighbor graph and UMAP embedding")

# %%
# ## 8. Clustering

if HAS_SCANPY:
    # Leiden clustering
    sc.tl.leiden(adata_hvg, resolution=0.5)

    n_clusters = adata_hvg.obs['leiden'].nunique()
    print(f"Found {n_clusters} clusters")

    # Cluster sizes
    print("\nCluster sizes:")
    for cluster in sorted(adata_hvg.obs['leiden'].unique()):
        n = (adata_hvg.obs['leiden'] == cluster).sum()
        print(f"  Cluster {cluster}: {n} cells")

# %%
# ## 9. Find Marker Genes

if HAS_SCANPY:
    # Find markers for each cluster
    sc.tl.rank_genes_groups(adata_hvg, groupby='leiden', method='wilcoxon')

    print("Top marker genes per cluster:")
    for cluster in sorted(adata_hvg.obs['leiden'].unique()):
        genes = sc.get.rank_genes_groups_df(adata_hvg, group=str(cluster))
        top_genes = genes.head(3)['names'].tolist()
        print(f"  Cluster {cluster}: {', '.join(top_genes)}")

# %%
# ## 10. Summary Report

if HAS_SCANPY:
    print("\n" + "=" * 50)
    print("Single-Cell Analysis Summary")
    print("=" * 50)
    print(f"Cells: {adata_hvg.n_obs}")
    print(f"Genes (HVG): {adata_hvg.n_vars}")
    print(f"Clusters: {n_clusters}")
    print(f"UMAP computed: True")
    print(f"Marker genes found: True")

# %%
# ## Summary
#
# This tutorial covered:
# 1. Creating synthetic scRNA-seq data
# 2. Quality control
# 3. Normalization
# 4. Highly variable gene selection
# 5. PCA
# 6. UMAP embedding
# 7. Leiden clustering
# 8. Marker gene detection
#
# Next: Tutorial 5 - CRISPR Guide RNA Design
