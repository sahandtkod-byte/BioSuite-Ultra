"""
BioSuite Tutorial 3: Differential Expression Analysis

This notebook demonstrates RNA-seq analysis:
- Reading count matrices
- CPM/TPM normalization
- Differential expression testing
- Volcano plot visualization
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from biosuite.core.expression import (
    cpm_normalization, tpm_normalization, differential_expression,
    deseq2_normalization, variance_stabilizing_transformation
)
from biosuite.plotting.plot_api import volcano, scatter, heatmap

print("BioSuite Differential Expression Tutorial")
print("=" * 50)

# %%
# ## 1. Create Sample Count Matrix

np.random.seed(42)
n_genes = 1000
n_samples = 6

# Generate count data
gene_names = [f"Gene_{i}" for i in range(n_genes)]
sample_names = [f"Sample_{i}" for i in range(n_samples)]

# Base expression levels
base_expression = np.random.negative_binomial(5, 0.3, (n_genes, n_samples))

# Add differential expression for some genes
# Genes 0-49: upregulated in treatment
base_expression[0:50, 3:6] *= 3
# Genes 50-99: downregulated in treatment
base_expression[50:100, 3:6] //= 3

# Create DataFrame
counts_df = pd.DataFrame(base_expression, columns=sample_names)
counts_df.insert(0, 'gene', gene_names)

print("Sample count matrix:")
print(counts_df.head(10))
print(f"\nShape: {counts_df.shape}")

# %%
# ## 2. CPM Normalization

cpm_df = cpm_normalization(counts_df)
print("\nCPM-normalized data (first 5 genes):")
print(cpm_df.head())

# %%
# ## 3. TPM Normalization

# Generate random gene lengths (in kilobases)
gene_lengths = np.random.uniform(0.5, 5.0, n_genes)
tpm_df = tpm_normalization(counts_df, gene_lengths)
print("\nTPM-normalized data (first 5 genes):")
print(tpm_df.head())

# %%
# ## 4. DESeq2-style Normalization

deseq2_df = deseq2_normalization(counts_df)
print("\nDESeq2-normalized data (first 5 genes):")
print(deseq2_df.head())

# %%
# ## 5. Variance Stabilizing Transformation

vst_df = variance_stabilizing_transformation(counts_df)
print("\nVST-transformed data (first 5 genes):")
print(vst_df.head())

# %%
# ## 6. Differential Expression Analysis

conditions = ['ctrl'] * 3 + ['treat'] * 3

# Using t-test
de_results_ttest = differential_expression(counts_df, conditions, method='ttest')
print("\nDifferential Expression (t-test):")
print(de_results_ttest.head(10))

# Using negative binomial (DESeq2-equivalent)
de_results_nb = differential_expression(counts_df, conditions, method='nb')
print("\nDifferential Expression (Negative Binomial):")
print(de_results_nb.head(10))

# %%
# ## 7. Summary Statistics

n_up = ((de_results_ttest['log2FC'] > 1) & (de_results_ttest['padj'] < 0.05)).sum()
n_down = ((de_results_ttest['log2FC'] < -1) & (de_results_ttest['padj'] < 0.05)).sum()
print(f"\nSummary:")
print(f"  Total genes: {len(de_results_ttest)}")
print(f"  Upregulated (log2FC > 1, padj < 0.05): {n_up}")
print(f"  Downregulated (log2FC < -1, padj < 0.05): {n_down}")

# %%
# ## 8. Volcano Plot

fig = volcano(
    log2fc=de_results_ttest['log2FC'].values,
    pvalues=de_results_ttest['pvalue'].values,
    gene_names=de_results_ttest['gene'].values,
    fc_thresh=1.0,
    p_thresh=0.05,
    title="Differential Expression Volcano Plot",
    interactive=True
)
print("Volcano plot created!")

# %%
# ## 9. Visualize Top DE Genes

# Get top 20 significant genes
top_genes = de_results_ttest.nsmallest(20, 'padj')
top_gene_names = top_genes['gene'].tolist()

# Get VST values for these genes
vst_values = vst_df.loc[vst_df.index.isin(top_gene_names)].values

# Create heatmap
fig = heatmap(
    data=vst_values,
    row_labels=top_gene_names,
    col_labels=sample_names,
    title="Top 20 DE Genes (VST-transformed)",
    interactive=True
)
print("Heatmap of top DE genes created!")

# %%
# ## 10. Compare Methods

# Scatter plot comparing t-test vs NB p-values
fig = scatter(
    x=-np.log10(de_results_ttest['pvalue'].values + 1e-300),
    y=-np.log10(de_results_nb['pvalue'].values + 1e-300),
    title="t-test vs Negative Binomial P-values",
    xlabel="-log10(p-value) [t-test]",
    ylabel="-log10(p-value) [NB]",
    show_regression=True
)
print("Method comparison plot created!")

# %%
# ## Summary
#
# This tutorial covered:
# 1. Creating sample count data
# 2. CPM normalization
# 3. TPM normalization
# 4. DESeq2-style normalization
# 5. Variance stabilizing transformation
# 6. Differential expression (t-test and NB)
# 7. Volcano plot visualization
# 8. Heatmap of top DE genes
# 9. Method comparison
#
# Next: Tutorial 4 - Single-Cell Analysis
