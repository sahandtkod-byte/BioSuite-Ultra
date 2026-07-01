"""
Expression analysis: count matrix normalization and differential expression.

Provides tools for RNA-seq count data analysis including CPM/TPM normalization,
two-group differential expression testing with multiple testing correction,
and I/O for common count matrix formats (featureCounts, HTSeq).
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def read_counts_matrix(filepath: str) -> pd.DataFrame | None:
    """Read a gene count matrix (featureCounts/HTSeq output).

    Parses tab- or comma-separated files where the first column is gene IDs
    and subsequent columns are sample counts. Handles commented header lines.

    Args:
        filepath: Path to CSV, TSV, or space-separated count file.

    Returns:
        DataFrame with 'gene' column and sample count columns, or None on error.
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', comment='#')
        if df.shape[1] < 2:
            df = pd.read_csv(filepath, sep='\t', comment='#', header=None)
        df.columns = ['gene'] + [f'Sample_{i}' for i in range(1, df.shape[1])]
        return df
    except Exception as e:
        print(f"Error reading counts matrix: {e}")
        return None


def cpm_normalization(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to Counts Per Million (CPM).

    Divides each count by its column library size (sum of all counts
    in that sample) and multiplies by 1e6. Useful for comparing
    expression levels across samples with different sequencing depths.

    Args:
        counts_df: DataFrame with numeric count columns.

    Returns:
        DataFrame of CPM values (numeric columns only).
    """
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns
    lib_sizes = counts_df[numeric_cols].sum(axis=0)
    cpm = counts_df[numeric_cols].div(lib_sizes, axis=1) * 1e6
    return cpm


def tpm_normalization(counts_df: pd.DataFrame, gene_lengths) -> pd.DataFrame:
    """Transcripts Per Million (TPM) normalization.

    First computes reads per kilobase (RPK) by dividing counts by gene
    length, then normalizes so each sample sums to 1e6. TPM accounts for
    both sequencing depth and gene length, making it ideal for comparing
    expression of genes of different lengths within a sample.

    Args:
        counts_df: DataFrame with numeric count columns.
        gene_lengths: Array-like of gene lengths in kilobases (one per row).

    Returns:
        DataFrame of TPM values (numeric columns only).
    """
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns
    rpk = counts_df[numeric_cols].div(gene_lengths, axis=0) * 1000
    scaling_factor = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scaling_factor, axis=1)
    return tpm


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Controls the false discovery rate when performing multiple hypothesis
    tests simultaneously (e.g., testing 20,000 genes for differential
    expression). Without this correction, ~5% of non-significant genes
    would appear significant by chance at p < 0.05.

    Reference: Benjamini & Hochberg (1995) J. Royal Statistical Society B.

    Args:
        pvalues: Array of raw p-values from statistical tests.

    Returns:
        Array of adjusted p-values (q-values), same shape as input.
    """
    n = len(pvalues)
    if n == 0:
        return pvalues
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    adjusted = np.zeros(n)
    # BH step-up procedure
    for rank, p in enumerate(sorted_p, 1):
        adjusted[rank - 1] = p * n / rank
    # Enforce monotonicity (cumulative minimum from bottom)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    # Restore original order
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


def differential_expression(
    counts_df: pd.DataFrame,
    conditions: list[str],
    method: str = 'ttest',
    correction: str = 'bh'
) -> pd.DataFrame:
    """Perform differential expression analysis between two groups.

    Computes log2 fold-change and p-values for each gene comparing
    two experimental conditions (e.g., control vs treated). Optionally
    applies Benjamini-Hochberg multiple testing correction.

    The fold-change uses pseudocount +1 to avoid log(0):
        log2FC = log2(mean(treat+1) / mean(control+1))

    Args:
        counts_df: DataFrame with 'gene' column and numeric count columns.
        conditions: List of group labels, one per numeric column.
                    Must contain exactly 2 unique groups.
        method: Statistical test ('ttest' for Welch's t-test).
        correction: Multiple testing correction ('bh' for Benjamini-Hochberg,
                     'none' for raw p-values).

    Returns:
        DataFrame with columns: gene, log2FC, pvalue, padj (adjusted p-value).

    Raises:
        ValueError: If conditions doesn't have exactly 2 groups or no numeric data.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Only two groups supported for differential expression.")
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in counts matrix.")

    group1 = [i for i, c in enumerate(conditions) if c == conditions[0]]
    group2 = [i for i, c in enumerate(conditions) if c != conditions[0]]
    vals1 = counts_df[numeric_cols].iloc[:, group1].values.astype(float)
    vals2 = counts_df[numeric_cols].iloc[:, group2].values.astype(float)

    # Log2 fold-change with pseudocount to avoid log(0)
    mean1 = np.mean(vals1 + 1, axis=1)
    mean2 = np.mean(vals2 + 1, axis=1)
    log2fc = np.log2(mean2 / mean1)

    # Statistical test
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

    # Multiple testing correction
    if correction == 'bh':
        padj = _benjamini_hochberg(pvals)
    else:
        padj = pvals.copy()

    genes = counts_df['gene'].values if 'gene' in counts_df.columns else range(len(counts_df))
    return pd.DataFrame({
        'gene': genes,
        'log2FC': log2fc,
        'pvalue': pvals,
        'padj': padj
    })
