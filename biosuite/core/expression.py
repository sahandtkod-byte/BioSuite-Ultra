"""
Expression analysis: count matrix normalization and differential expression.

Provides tools for RNA-seq count data analysis including CPM/TPM normalization,
two-group differential expression testing with multiple testing correction,
and I/O for common count matrix formats (featureCounts, HTSeq).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .log import get_logger

logger = get_logger(__name__)



def read_counts_matrix(filepath: str) -> pd.DataFrame | None:
    """Read a gene count matrix (featureCounts/HTSeq output).

    Parses tab-, comma- or space-separated files where the first column is gene
    IDs and subsequent columns are sample counts. Handles commented header lines.

    Args:
        filepath: Path to CSV, TSV, or space-separated count file.
    Returns:
        DataFrame with 'gene' column and sample count columns, or None when the
        file is missing, unparsable, or carries no count column at all.
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', comment='#')
        if df.shape[1] < 2:
            df = pd.read_csv(filepath, sep='\t', comment='#', header=None)

        names = list(df.columns)
        # A header that starts with the separator while the data rows carry no
        # row-name field — write.table(col.names = NA, row.names = FALSE) writes
        # exactly that — shifts every column name one place right and pads an
        # all-NaN sample at the end, so a gene length would be analysed as
        # expression. Dropping the phantom column restores the alignment.
        if (len(names) > 1 and str(names[0]).startswith('Unnamed: 0')
                and df.iloc[:, -1].isna().all()):
            df = df.iloc[:, :-1].copy()

        # A column of only NaN is float64, so dtypes alone cannot tell a count
        # matrix from a bare gene list; require a sample column holding a number.
        if df.empty or not df.iloc[:, 1:].select_dtypes(
                include='number').notna().any().any():
            logger.error(f"No count columns found in {filepath}")
            return None

        df.columns = ['gene'] + [f'Sample_{i}' for i in range(1, df.shape[1])]
        return df
    except Exception as e:
        logger.error(f"Error reading counts matrix: {e}")
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


def _gene_labels(counts_df: pd.DataFrame) -> np.ndarray:
    """Best available gene identifiers for a count matrix.

    Order of preference: an explicit ``gene`` column, then the DataFrame index
    when it carries labels, then row positions.

    Gene names held in the index used to be discarded and replaced with row
    numbers, so a caller who filtered or reordered their matrix could not map
    the result rows back to genes.
    """
    if 'gene' in counts_df.columns:
        return np.asarray(counts_df['gene'].values)
    index = counts_df.index
    if not isinstance(index, pd.RangeIndex):
        return np.asarray(index.tolist(), dtype=object)
    return np.arange(len(counts_df))


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
    # Degenerate cases: the standard error is zero or non-finite, so no t-test
    # exists.  Returning p = 1.0 unconditionally used to assert "no evidence of
    # a difference" even for genes that jump 10 -> 500 with zero within-group
    # variance, which is a positive claim the data cannot support.  Report an
    # honest NaN when the group means differ (undefined test) and 1.0 only when
    # the groups are genuinely identical.
    identical = np.isclose(mean1, mean2, rtol=0, atol=0)
    p_vals = np.where(finite_se, p_vals, np.where(identical, 1.0, np.nan))

    return t_stat, p_vals


def _negative_binomial_test(vals1: np.ndarray, vals2: np.ndarray) -> np.ndarray:
    """Negative Binomial test for differential expression (approximate, not DESeq2).

    A reduced stand-in for DESeq2 rather than an equivalent of it: dispersion is
    estimated per gene from the observed counts, with no shrinkage toward a fitted
    mean-dispersion trend and no size-factor normalisation inside the test.  The
    labels in the output therefore should not be read as DESeq2 results.

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
            # Every cell is floored at 1 above, so the contingency table is always
            # a valid 2x2 of positive integers and fisher_exact cannot raise on it.
            # (A zero total was skipped with the sum check, and a zero mean is
            # impossible once the sum is non-zero, so the guards that used to sit
            # around this call were unreachable.)
            _, p = fisher_exact(table)
            pvals[i] = p
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

    if positive_mask.sum() < 5:
        # Too few (or no) fully-positive genes to build a geometric-mean
        # reference, so fall back to library size normalization.  The
        # separate `== 0` branch this replaces ran byte-for-byte the same
        # statements, because 0 < 5 already covers it.
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
    - 'vst': asinh transform: log2(sqrt(x) + sqrt(x+1))
    - 'log': shifted log: log2(x + 0.5)

    Args:
        counts_df: DataFrame with numeric count columns.
        size_factors: optional size factors (computed via DESeq2 method if None).
        method: 'vst' for variance-stabilizing, 'log' for shifted log2.
    Returns:
        DataFrame of transformed values.
    Raises:
        ValueError: If ``method`` is neither 'vst' nor 'log'.  An unknown label
            used to be transformed as 'vst' without saying so.
    """
    if method not in ('vst', 'log'):
        raise ValueError(f"Unknown method: {method}. Use 'vst' or 'log'.")

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
    shrinkage: bool = False
) -> pd.DataFrame:
    """Calculate log2 fold-change between two groups.

    Supports multiple methods for computing fold change, with optional
    apeglm-style shrinkage for reduced noise in low-count genes.

    Args:
        counts_df: DataFrame with 'gene' column and numeric count columns.
        conditions: List of group labels, one per numeric column.
        method: 'mean' (arithmetic mean of group), 'median', or 'trimmed'.
        pseudocount: Added to means before log to avoid log(0).
        shrinkage: If True, apply apeglm-style posterior shrinkage.
    Returns:
        DataFrame with columns: gene, log2FC, baseMean, significant.  The flag is
        the conventional |log2FC| > 1 cut; this function computes no p-value, so
        there is nothing for a significance level to threshold.
    Raises:
        ValueError: If the groups are not exactly two, if the label count does
            not match the numeric column count, if there is no numeric data, or
            if ``method`` is not one of 'mean', 'median', 'trimmed'.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Exactly two groups required for fold change calculation.")

    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found.")
    if len(conditions) != len(numeric_cols):
        raise ValueError(
            f"conditions has {len(conditions)} labels but the counts matrix has "
            f"{len(numeric_cols)} numeric columns ({numeric_cols}); they must match "
            "one-to-one.")
    if method not in ('mean', 'median', 'trimmed'):
        # An unrecognised label used to fall through to the arithmetic mean, so
        # method='geometric' quietly reported a different statistic than the one
        # the caller asked for.
        raise ValueError(
            f"Unknown method: {method}. Use 'mean', 'median' or 'trimmed'.")

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
        # apeglm-inspired posterior mean under a Normal prior of variance 1.0 on
        # log2FC with observation variance 2.0/baseMean (delta method): the weight
        # kept is prior/(prior+var) = baseMean/(baseMean+2), so the noisier
        # low-count estimate is pulled toward zero and a well-measured gene keeps
        # its fold-change.  baseMean was used as the *precision* here before,
        # which did the opposite: a 7000-count gene was shrunk to 1% of its
        # log2FC while a 5-count gene kept 29%.
        shrinkage_factor = np.clip(base_mean / (base_mean + 2.0), 0.01, 1.0)
        log2fc = np.where(np.isfinite(log2fc), log2fc * shrinkage_factor, 0.0)

    # Flag significant genes (|log2FC| > 1)
    significant = np.abs(log2fc) > 1.0

    genes = _gene_labels(counts_df)
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
        DataFrame with columns: gene, effect_size, interpretation.  Both methods
        share one sign convention: positive means the second group is higher.
        A gene whose pooled standard deviation is zero while its group means
        differ has no defined standardized effect, so it is reported as NaN with
        interpretation 'undefined'.
    Raises:
        ValueError: If the groups are not exactly two, if there is no numeric
            data, if ``method`` is unrecognised, or if ``method='cohens_d'`` is
            asked of a group with fewer than two replicates.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Exactly two groups required.")

    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found.")
    if len(conditions) != len(numeric_cols):
        # Too few labels used to analyse only the leading columns and too many
        # raised inside pandas, both without saying which input was wrong.
        raise ValueError(
            f"conditions has {len(conditions)} labels but the counts matrix has "
            f"{len(numeric_cols)} numeric columns ({numeric_cols}); they must match "
            "one-to-one.")
    if method not in ('cohens_d', 'rank_biserial'):
        # Checked here rather than in a trailing else-branch so the replicate
        # check below cannot mask it.
        raise ValueError(f"Unknown method: {method}. Use 'cohens_d' or 'rank_biserial'.")

    group1 = [i for i, c in enumerate(conditions) if c == conditions[0]]
    group2 = [i for i, c in enumerate(conditions) if c != conditions[0]]
    vals1 = counts_df[numeric_cols].iloc[:, group1].values.astype(float)
    vals2 = counts_df[numeric_cols].iloc[:, group2].values.astype(float)

    n1 = vals1.shape[1]
    n2 = vals2.shape[1]
    n_genes = vals1.shape[0]

    if method == 'cohens_d':
        if n1 < 2 or n2 < 2:
            # ddof=1 variance needs two observations per group, and below N=4 the
            # Hedges factor leaves its own domain: 1 - 3/(4N - 9) measures 4.0 at
            # N=2 (inflating a real difference fourfold) and 0.0 at N=3 (reporting
            # every gene as having no effect at all).
            raise ValueError(
                "Cohen's d needs at least two replicates per group, got "
                f"{n1} vs {n2}.")

        mean1 = vals1.mean(axis=1)
        mean2 = vals2.mean(axis=1)
        var1 = vals1.var(axis=1, ddof=1)
        var2 = vals2.var(axis=1, ddof=1)

        # Pooled standard deviation (Hedges' correction)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_d = (mean2 - mean1) / pooled_std
        # A zero pooled standard deviation means both groups are constant.  The
        # old fallback of std=1.0 then reported the raw count difference as if it
        # were a standardized effect (10 vs 100 became d=360).  Equal constant
        # groups really do have no effect; unequal ones have no finite one.
        cohens_d = np.where(pooled_std > 0, raw_d,
                            np.where(mean1 == mean2, 0.0, np.nan))

        # Hedges' g correction factor (Hedges & Olkin 1985), valid for N >= 4,
        # which the replicate check above now guarantees.
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        effect_sizes = cohens_d * correction

        # Interpretation thresholds (Cohen's conventions)
        abs_d = np.abs(effect_sizes)
        interpretation = np.where(np.isnan(effect_sizes), 'undefined',
                         np.where(abs_d < 0.2, 'negligible',
                         np.where(abs_d < 0.5, 'small',
                         np.where(abs_d < 0.8, 'medium', 'large'))))

    else:
        # rank_biserial — the only remaining validated method.  Written as an
        # else rather than `elif method == 'rank_biserial'` because the check at
        # the top of the function already rules anything else out, so the elif's
        # false edge could never be taken.
        n_total = n1 + n2
        all_vals = np.concatenate([vals1, vals2], axis=1)
        effect_sizes = np.zeros(n_genes)

        for i in range(n_genes):
            ranks = sp_stats.rankdata(all_vals[i])
            r1_sum = ranks[:n1].sum()
            # r = 2*(E[R1] - R1)/(n1*n2) with E[R1] = n1*(N+1)/2, which equals
            # 2*U1/(n1*n2) - 1 for group 2.  Flipped from group 1's ranks so an
            # up-regulated gene is positive here exactly as it is in 'cohens_d'.
            expected_r1 = n1 * (n_total + 1) / 2
            effect_sizes[i] = 2 * (expected_r1 - r1_sum) / (n1 * n2)

        abs_r = np.abs(effect_sizes)
        interpretation = np.where(abs_r < 0.1, 'negligible',
                         np.where(abs_r < 0.3, 'small',
                         np.where(abs_r < 0.5, 'medium', 'large')))

    genes = _gene_labels(counts_df)
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

    The 'ttest' method compares log1p(counts) rather than raw counts, because
    count variance grows with the mean and a raw-scale t-test discards most of
    its power on highly expressed genes.

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
        ValueError: If conditions doesn't have exactly 2 groups, if the number
            of labels does not match the number of numeric columns, or if there
            is no numeric data.
    """
    if len(set(conditions)) != 2:
        raise ValueError("Only two groups supported for differential expression.")
    numeric_cols = counts_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in counts matrix.")
    if len(conditions) != len(numeric_cols):
        # Zipping labels against columns without this check silently analysed a
        # subset of the samples (too few labels) or raised IndexError deep in
        # pandas (too many), both of which produced misleading statistics.
        raise ValueError(
            f"conditions has {len(conditions)} labels but the counts matrix has "
            f"{len(numeric_cols)} numeric columns ({numeric_cols}); they must match "
            "one-to-one.")

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
        # Raw counts carry mean-dependent variance, so a t-test on them loses most of
        # its power on highly expressed genes; the audit report's Pass 5 experiment
        # measured 56/100 planted genes recovered raw versus ~100/100 on log1p.
        _, pvals = _welch_ttest_rows(np.log1p(vals1), np.log1p(vals2))
    elif method == 'nb':
        pvals = _negative_binomial_test(vals1, vals2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ttest' or 'nb'.")

    # Multiple testing correction
    if correction == 'bh':
        padj = _benjamini_hochberg(pvals)
    else:
        padj = pvals.copy()

    genes = _gene_labels(counts_df)
    return pd.DataFrame({
        'gene': genes,
        'log2FC': log2fc,
        'pvalue': pvals,
        'padj': padj
    })
