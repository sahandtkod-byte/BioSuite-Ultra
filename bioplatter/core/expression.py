"""
Expression analysis: count matrix, normalization, differential expression.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

def read_counts_matrix(filepath):
    """Read featureCounts/HTSeq output (gene id and counts)."""
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', comment='#')
        if df.shape[1] < 2:
            df = pd.read_csv(filepath, sep='\t', comment='#', header=None)
        df.columns = ['gene'] + [f'Sample_{i}' for i in range(1, df.shape[1])]
        return df
    except Exception as e:
        print(f"Error reading counts matrix: {e}")
        return None

def cpm_normalization(counts_df):
    """Convert raw counts to Counts Per Million (CPM)."""
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns
    lib_sizes = counts_df[numeric_cols].sum(axis=0)
    cpm = counts_df[numeric_cols].div(lib_sizes, axis=1) * 1e6
    return cpm

def tpm_normalization(counts_df, gene_lengths):
    """TPM normalization (requires gene lengths)."""
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns
    rpk = counts_df[numeric_cols].div(gene_lengths, axis=0) * 1000
    scaling_factor = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scaling_factor, axis=1)
    return tpm

def differential_expression(counts_df, conditions, method='ttest'):
    """
    Simple t-test or fold-change between two groups.
    conditions: list of group labels for each column.
    Returns DataFrame with gene, log2FC, pvalue.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Only two groups supported for t-test.")
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in counts matrix.")
    group1 = [i for i, c in enumerate(conditions) if c == conditions[0]]
    group2 = [i for i, c in enumerate(conditions) if c != conditions[0]]
    vals1 = counts_df[numeric_cols].iloc[:, group1].values.astype(float)
    vals2 = counts_df[numeric_cols].iloc[:, group2].values.astype(float)
    mean1 = np.mean(vals1 + 1, axis=1)
    mean2 = np.mean(vals2 + 1, axis=1)
    log2fc = np.log2(mean2 / mean1)
    if method == 'ttest':
        std1 = np.std(vals1, axis=1)
        std2 = np.std(vals2, axis=1)
        both_zero = (std1 == 0) & (std2 == 0)
        pvals = np.ones(len(counts_df))
        if not np.all(both_zero):
            from scipy.stats import ttest_ind
            _, p = ttest_ind(vals1, vals2, axis=1)
            pvals[~both_zero] = p[~both_zero]
    else:
        pvals = np.ones(len(counts_df))
    genes = counts_df['gene'].values if 'gene' in counts_df.columns else range(len(counts_df))
    return pd.DataFrame({'gene': genes, 'log2FC': log2fc, 'pvalue': pvals})
