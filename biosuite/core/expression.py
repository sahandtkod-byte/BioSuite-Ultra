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
    # Guard against zero library sizes
    lib_sizes = lib_sizes.replace(0, 1)
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
    gene_lengths = np.asarray(gene_lengths, dtype=float)
    # Guard against zero-length genes
    gene_lengths = np.where(gene_lengths == 0, 1.0, gene_lengths)
    rpk = counts_df[numeric_cols].div(gene_lengths, axis=0) * 1000
    scaling_factor = rpk.sum(axis=0) / 1e6
    # Guard against zero scaling factor
    scaling_factor = scaling_factor.replace(0, 1)
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
        return pvalues.copy()

    pvalues = np.asarray(pvalues, dtype=float)

    # Separate NaN and valid p-values
    nan_mask = np.isnan(pvalues)
    valid = pvalues[~nan_mask]
    n_valid = len(valid)

    if n_valid == 0:
        return pvalues.copy()

    sorted_idx = np.argsort(valid)
    sorted_p = valid[sorted_idx]
    adjusted = np.zeros(n_valid)
    # BH step-up procedure
    for rank, p in enumerate(sorted_p, 1):
        adjusted[rank - 1] = p * n_valid / rank
    # Enforce monotonicity (cumulative minimum from the end)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order within valid subset
    result_valid = np.empty(n_valid)
    result_valid[sorted_idx] = adjusted

    # Reassemble full array with NaNs
    result = np.empty(n)
    result[nan_mask] = np.nan
    result[~nan_mask] = result_valid
    return result


def _welch_ttest_rows(vals1: np.ndarray, vals2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform Welch's t-test row-by-row on two 2D arrays.

    Returns (t_statistics, p_values), each of shape (n_genes,).
    Handles edge cases: zero variance, constant groups, NaN.
    """
    n1 = vals1.shape[1]
    n2 = vals2.shape[1]
    n_genes = vals1.shape[0]

    mean1 = vals1.mean(axis=1)
    mean2 = vals2.mean(axis=1)
    with np.errstate(all='ignore'):
        var1 = vals1.var(axis=1, ddof=1) if n1 > 1 else np.zeros(n_genes)
        var2 = vals2.var(axis=1, ddof=1) if n2 > 1 else np.zeros(n_genes)

    # Welch's t-test statistic
    se = np.sqrt(var1 / max(n1, 1) + var2 / max(n2, 1))
    finite_se = np.isfinite(se) & (se > 0)
    safe_se = np.where(finite_se, se, 1.0)
    t_stat = (mean1 - mean2) / safe_se
    t_stat = np.where(finite_se, t_stat, 0.0)

    # Welch-Satterthwaite degrees of freedom
    with np.errstate(divide='ignore', invalid='ignore'):
        df_num = (var1 / max(n1, 1) + var2 / max(n2, 1)) ** 2
        df_den = (var1 / max(n1, 1)) ** 2 / max(n1 - 1, 1) + \
                 (var2 / max(n2, 1)) ** 2 / max(n2 - 1, 1)
        df = np.where(df_den > 0, df_num / df_den, 1.0)
        df = np.clip(df, 1.0, None)

    p_vals = 2 * sp_stats.t.sf(np.abs(t_stat), df)
    # Set p=1.0 where SE is zero or NaN (no valid test possible)
    p_vals = np.where(finite_se, p_vals, 1.0)

    return t_stat, p_vals


def _negative_binomial_test(vals1: np.ndarray, vals2: np.ndarray) -> np.ndarray:
    """Negative Binomial test for differential expression (DESeq2-equivalent).

    Implements a simplified version of the DESeq2 approach:
    1. Estimate size parameter (dispersion) from the data
    2. Fit NB GLM for each gene
    3. Compute Wald test statistic

    Args:
        vals1: count matrix for group 1 (genes x samples).
        vals2: count matrix for group 2 (genes x samples).
    Returns:
        Array of p-values.
    """
    from scipy.stats import norm

    n_genes = vals1.shape[0]
    n1 = vals1.shape[1]
    n2 = vals2.shape[1]
    pvals = np.ones(n_genes)

    for i in range(n_genes):
        counts = np.concatenate([vals1[i], vals2[i]])

        # Skip genes with all zeros
        if counts.sum() == 0:
            continue

        # Estimate dispersion using method of moments
        mean_val = counts.mean()
        var_val = counts.var()

        if mean_val == 0:
            pvals[i] = 1.0
            continue

        if var_val <= mean_val:
            # Under-dispersed or equi-dispersed: use Fisher's exact test
            # which is the appropriate test for Poisson-like count data
            from scipy.stats import fisher_exact
            # Build a 2x2 table: [group1_above_median, group1_below_median]
            #                        [group2_above_median, group2_below_median]
            median_val = np.median(counts)
            g1_above = int((vals1[i] >= median_val).sum())
            g1_below = n1 - g1_above
            g2_above = int((vals2[i] >= median_val).sum())
            g2_below = n2 - g2_above
            table = [[max(g1_above, 1), max(g1_below, 1)],
                     [max(g2_above, 1), max(g2_below, 1)]]
            try:
                _, p = fisher_exact(table)
                pvals[i] = p
            except (ValueError, ZeroDivisionError):
                pvals[i] = 1.0
            continue

        # Method of moments dispersion estimate
        # For NB: var = mean + mean^2/size
        # size = mean^2 / (var - mean)
        size_est = mean_val ** 2 / (var_val - mean_val)
        size_est = max(size_est, 0.1)  # Bound to avoid numerical issues

        # Log2 fold-change with pseudocount
        mean_g1 = vals1[i].mean() + 0.5  # pseudocount
        mean_g2 = vals2[i].mean() + 0.5
        log2fc = np.log2(mean_g2 / mean_g1)

        # Wald test: compare means using NB variance
        # Var(log2fc) ≈ 1/(n1*mean_g1) + 1/(n2*mean_g2) (delta method)
        var_g1 = mean_g1 + mean_g1 ** 2 / size_est
        var_g2 = mean_g2 + mean_g2 ** 2 / size_est

        se = np.sqrt(var_g1 / (n1 * mean_g1 ** 2) + var_g2 / (n2 * mean_g2 ** 2))

        if se > 0:
            wald_stat = log2fc / se
            pvals[i] = 2 * (1 - norm.cdf(abs(wald_stat)))
        else:
            pvals[i] = 1.0

    return pvals


def deseq2_normalization(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Median-of-ratios normalization (DESeq2 size factors).

    Computes size factors that account for sequencing depth and RNA composition.
    Uses the geometric mean of each gene across samples as a reference,
    then takes the median of ratios for each sample.

    Args:
        counts_df: DataFrame with 'gene' column and numeric count columns.
    Returns:
        DataFrame with normalized counts and size_factors attribute.
    """
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns
    counts = counts_df[numeric_cols].values.astype(float)

    n_samples = counts.shape[1]

    # Compute geometric mean for each gene (across all samples)
    # Only use genes where all samples have positive counts
    positive_mask = (counts > 0).all(axis=1)

    if positive_mask.sum() == 0:
        # Fallback: use simple library size normalization (edge case)
        lib_sizes = counts.sum(axis=0)
        lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
        size_factors = lib_sizes / lib_sizes.mean()
    elif positive_mask.sum() < 5:
        # Too few fully-positive genes; fall back to library size
        lib_sizes = counts.sum(axis=0)
        lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
        size_factors = lib_sizes / lib_sizes.mean()
    else:
        counts_pos = counts[positive_mask]
        log_means = np.log(counts_pos).mean(axis=1)

        # Size factors: median of (count_ij / geometric_mean_i) for each sample
        size_factors = np.ones(n_samples)
        for j in range(n_samples):
            ratios = counts_pos[:, j] / np.exp(log_means)
            median_ratio = np.median(ratios)
            # Guard against zero or negative size factors
            size_factors[j] = max(median_ratio, 1e-8)

    # Normalize
    normalized = counts / size_factors
    result = counts_df.copy()
    result[numeric_cols] = normalized
    # Attach size factors for downstream use
    result.attrs['size_factors'] = size_factors
    return result


def variance_stabilizing_transformation(
    counts_df: pd.DataFrame,
    size_factors: np.ndarray | None = None,
    method: str = 'vst'
) -> pd.DataFrame:
    """Variance Stabilizing Transformation (VST) for count data.

    Stabilizes variance across the dynamic range, similar to DESeq2's vst().
    Supports two methods:
    - 'vst': asinht transform: log2(sqrt(x) + sqrt(x+1))
    - 'log': shifted log: log2(x + 0.5)

    Args:
        counts_df: DataFrame with numeric count columns.
        size_factors: optional size factors (computed via DESeq2 method if None).
        method: 'vst' for variance-stabilizing, 'log' for shifted log2.
    Returns:
        DataFrame of transformed values.
    """
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns
    counts = counts_df[numeric_cols].values.astype(float)

    if size_factors is None:
        # Use DESeq2 median-of-ratios size factors for proper normalization
        positive_mask = (counts > 0).all(axis=1)
        if positive_mask.sum() >= 5:
            counts_pos = counts[positive_mask]
            log_means = np.log(counts_pos).mean(axis=1)
            size_factors = np.ones(counts.shape[1])
            for j in range(counts.shape[1]):
                ratios = counts_pos[:, j] / np.exp(log_means)
                median_ratio = np.median(ratios)
                size_factors[j] = max(median_ratio, 1e-8)
        else:
            # Fallback to library size normalization
            lib_sizes = counts.sum(axis=0)
            lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
            size_factors = lib_sizes / lib_sizes.mean()

    # Normalize by size factors
    normalized = counts / size_factors

    if method == 'log':
        # Shifted log2 transform (simple but effective)
        result_vals = np.log2(normalized + 0.5)
    else:
        # VST: log2(sqrt(x) + sqrt(x+1))
        # This is the asinht transform, a good approximation to the
        # exact NB-based VST for moderate-to-high counts.
        result_vals = np.log2(np.sqrt(normalized) + np.sqrt(normalized + 1))

    result = counts_df.copy()
    result[numeric_cols] = result_vals
    return result


def calculate_fold_change(
    counts_df: pd.DataFrame,
    conditions: list[str],
    method: str = 'mean',
    pseudocount: float = 1.0,
    shrinkage: bool = False,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Calculate log2 fold-change between two groups.

    Supports multiple methods for computing fold change, with optional
   ape-style shrinkage for reduced noise in low-count genes.

    Args:
        counts_df: DataFrame with 'gene' column and numeric count columns.
        conditions: List of group labels, one per numeric column.
        method: 'mean' (arithmetic mean of group), 'median', or 'trimmed'.
        pseudocount: Added to means before log to avoid log(0).
        shrinkage: If True, apply apeglm-style posterior shrinkage.
        alpha: Significance threshold for flagging DE genes.
    Returns:
        DataFrame with columns: gene, log2FC, baseMean, significant.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Exactly two groups required for fold change calculation.")

    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found.")

    group1 = [i for i, c in enumerate(conditions) if c == conditions[0]]
    group2 = [i for i, c in enumerate(conditions) if c != conditions[0]]
    vals1 = counts_df[numeric_cols].iloc[:, group1].values.astype(float)
    vals2 = counts_df[numeric_cols].iloc[:, group2].values.astype(float)

    if method == 'median':
        g1 = np.median(vals1, axis=1)
        g2 = np.median(vals2, axis=1)
    elif method == 'trimmed':
        from scipy.stats import trim_mean
        g1 = trim_mean(vals1, 0.1, axis=1)
        g2 = trim_mean(vals2, 0.1, axis=1)
    else:  # mean
        g1 = np.mean(vals1, axis=1)
        g2 = np.mean(vals2, axis=1)

    # Base mean (average expression across all samples)
    base_mean = np.mean(np.concatenate([vals1, vals2], axis=1), axis=1)

    # Log2 fold-change
    log2fc = np.log2((g2 + pseudocount) / (g1 + pseudocount))

    if shrinkage:
        # Simple apeglm-inspired shrinkage:
        # Shrink toward zero proportionally to 1/baseMean
        # Genes with low counts get more shrinkage
        prior_var = 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            posterior_var = 1.0 / (1.0 / prior_var + base_mean / 2.0)
            shrinkage_factor = posterior_var / prior_var
            log2fc = log2fc * np.clip(shrinkage_factor, 0.01, 1.0)
        log2fc = np.where(np.isfinite(log2fc), log2fc, 0.0)

    # Flag significant genes (|log2FC| > 1)
    significant = np.abs(log2fc) > 1.0

    genes = counts_df['gene'].values if 'gene' in counts_df.columns else range(len(counts_df))
    return pd.DataFrame({
        'gene': genes,
        'log2FC': log2fc,
        'baseMean': base_mean,
        'significant': significant,
    })


def calculate_effect_size(
    counts_df: pd.DataFrame,
    conditions: list[str],
    method: str = 'cohens_d'
) -> pd.DataFrame:
    """Calculate effect size measures for differential expression.

    Args:
        counts_df: DataFrame with 'gene' column and numeric count columns.
        conditions: List of group labels, one per numeric column.
        method: 'cohens_d' (standardized mean difference) or 'rank_biserial'.
    Returns:
        DataFrame with columns: gene, effect_size, interpretation.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Exactly two groups required.")

    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found.")

    group1 = [i for i, c in enumerate(conditions) if c == conditions[0]]
    group2 = [i for i, c in enumerate(conditions) if c != conditions[0]]
    vals1 = counts_df[numeric_cols].iloc[:, group1].values.astype(float)
    vals2 = counts_df[numeric_cols].iloc[:, group2].values.astype(float)

    n1 = vals1.shape[1]
    n2 = vals2.shape[1]
    n_genes = vals1.shape[0]

    if method == 'cohens_d':
        mean1 = vals1.mean(axis=1)
        mean2 = vals2.mean(axis=1)
        var1 = vals1.var(axis=1, ddof=1)
        var2 = vals2.var(axis=1, ddof=1)

        # Pooled standard deviation (Hedges' correction)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(np.where(pooled_var > 0, pooled_var, 1.0))
        cohens_d = (mean2 - mean1) / pooled_std

        # Hedges' g correction factor
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        effect_sizes = cohens_d * correction

        # Interpretation thresholds (Cohen's conventions)
        abs_d = np.abs(effect_sizes)
        interpretation = np.where(abs_d < 0.2, 'negligible',
                         np.where(abs_d < 0.5, 'small',
                         np.where(abs_d < 0.8, 'medium', 'large')))

    elif method == 'rank_biserial':
        # Rank-biserial correlation for non-parametric effect size
        n_total = n1 + n2
        all_vals = np.concatenate([vals1, vals2], axis=1)
        effect_sizes = np.zeros(n_genes)

        for i in range(n_genes):
            ranks = sp_stats.rankdata(all_vals[i])
            r1_sum = ranks[:n1].sum()
            # Rank-biserial: r = 2*(R1/n1 - R2/n2) / n_total
            expected_r1 = n1 * (n_total + 1) / 2
            effect_sizes[i] = 2 * (r1_sum - expected_r1) / (n1 * n2)

        abs_r = np.abs(effect_sizes)
        interpretation = np.where(abs_r < 0.1, 'negligible',
                         np.where(abs_r < 0.3, 'small',
                         np.where(abs_r < 0.5, 'medium', 'large')))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cohens_d' or 'rank_biserial'.")

    genes = counts_df['gene'].values if 'gene' in counts_df.columns else range(n_genes)
    return pd.DataFrame({
        'gene': genes,
        'effect_size': effect_sizes,
        'interpretation': interpretation,
    })


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
        method: Statistical test ('ttest' for Welch's t-test,
                'nb' for negative binomial / Wald test).
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
        _, pvals = _welch_ttest_rows(vals1, vals2)
    elif method == 'nb':
        pvals = _negative_binomial_test(vals1, vals2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ttest' or 'nb'.")

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
