"""
Population genetics analysis.

Pure Python implementations of Hardy-Weinberg, FST, nucleotide diversity,
Tajima's D, PCA, and linkage disequilibrium. All pip-installable dependencies.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats as sp_stats
from collections import Counter


@dataclass
class PopGenReport:
    num_sites: int = 0
    num_populations: int = 0
    fst: dict = field(default_factory=dict)
    nucleotide_diversity: float = 0.0
    tajima_d: float = 0.0
    hw_test: dict = field(default_factory=dict)
    ld: dict = field(default_factory=dict)
    message: str = ""


def hardy_weinberg_test(genotype_counts):
    """Test Hardy-Weinberg equilibrium for biallelic locus.

    Args:
        genotype_counts: dict with keys 'AA', 'Aa', 'aa' and integer counts.

    Returns:
        Dict with chi2 statistic, p-value, and expected frequencies.
    """
    n_AA = genotype_counts.get('AA', 0)
    n_Aa = genotype_counts.get('Aa', 0)
    n_aa = genotype_counts.get('aa', 0)
    n = n_AA + n_Aa + n_aa

    if n == 0:
        return {'chi2': 0, 'p_value': 1.0, 'message': 'No data'}

    p = (2 * n_AA + n_Aa) / (2 * n)
    q = 1 - p

    exp_AA = p**2 * n
    exp_Aa = 2 * p * q * n
    exp_aa = q**2 * n

    observed = np.array([n_AA, n_Aa, n_aa])
    expected = np.array([exp_AA, exp_Aa, exp_aa])

    expected = np.maximum(expected, 1e-10)
    chi2 = np.sum((observed - expected)**2 / expected)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)

    return {
        'chi2': round(chi2, 4),
        'p_value': round(p_value, 6),
        'allele_freq_p': round(p, 4),
        'allele_freq_q': round(q, 4),
        'expected': {'AA': round(exp_AA, 1), 'Aa': round(exp_Aa, 1), 'aa': round(exp_aa, 1)},
        'in_hwe': p_value > 0.05
    }


def calculate_fst(population_genotype_matrices):
    """Calculate pairwise FST between populations (Weir-Cockerham简化版).

    Args:
        population_genotype_matrices: list of numpy arrays (n_samples x n_sites).
            Values: 0=ref/ref, 1=ref/alt, 2=alt/alt.

    Returns:
        Dict mapping (pop_i, pop_j) to FST value.
    """
    n_pops = len(population_genotype_matrices)
    fst_pairs = {}

    for i in range(n_pops):
        for j in range(i + 1, n_pops):
            gt_i = population_genotype_matrices[i]
            gt_j = population_genotype_matrices[j]

            freqs_i = gt_i.mean(axis=0) / 2
            freqs_j = gt_j.mean(axis=0) / 2

            n_i = gt_i.shape[0]
            n_j = gt_j.shape[0]

            p_bar = (freqs_i * n_i + freqs_j * n_j) / (n_i + n_j)
            var_between = ((freqs_i - p_bar)**2 * n_i + (freqs_j - p_bar)**2 * n_j) / (n_i + n_j)

            het_i = 2 * freqs_i * (1 - freqs_i)
            het_j = 2 * freqs_j * (1 - freqs_j)
            var_within = (het_i * n_i + het_j * n_j) / (n_i + n_j)

            total_var = var_between + var_within / 2
            with np.errstate(divide='ignore', invalid='ignore'):
                fst = np.where(total_var > 0, var_between / total_var, 0)

            avg_fst = float(np.mean(fst))
            fst_pairs[(i, j)] = round(avg_fst, 4)

    return fst_pairs


def nucleotide_diversity(genotype_matrix):
    """Calculate nucleotide diversity (pi) from genotype matrix.

    Args:
        genotype_matrix: numpy array (n_samples x n_sites).
            Values: 0=ref/ref, 1=ref/alt, 2=alt/alt.

    Returns:
        Per-site and average nucleotide diversity.
    """
    n = genotype_matrix.shape[0]
    if n < 2:
        return 0.0

    allele_freqs = genotype_matrix.mean(axis=0) / 2
    per_site_pi = 2 * allele_freqs * (1 - allele_freqs) * n / (n - 1)
    return float(np.mean(per_site_pi))


def tajimas_d(genotype_matrix):
    """Calculate Tajima's D from genotype matrix.

    Args:
        genotype_matrix: numpy array (n_samples x n_sites).

    Returns:
        Tajima's D statistic.
    """
    n = genotype_matrix.shape[0]
    n_sites = genotype_matrix.shape[1]

    if n < 2 or n_sites < 2:
        return 0.0

    # Count segregating sites
    seg_sites = 0
    allele_counts = []
    for j in range(n_sites):
        col = genotype_matrix[:, j]
        unique = np.unique(col)
        if len(unique) > 1:
            seg_sites += 1
            alt_count = int(col.sum())
            allele_counts.append((alt_count, n - alt_count))

    if seg_sites == 0:
        return 0.0

    # Theta from number of segregating sites
    a1 = sum(1.0 / i for i in range(1, n))
    theta_w = seg_sites / a1 if a1 > 0 else 0

    # Theta from average pairwise differences
    pi_avg = nucleotide_diversity(genotype_matrix) * n_sites
    theta_pi = pi_avg / (n * (n - 1) / 2) if n > 1 else 0

    # Tajima's D
    a2 = sum(1.0 / i**2 for i in range(1, n))
    var_d = (a1 * (n - 1) / (n * (n - 1))) * (
        n * (n + 1) / (n - 1) / 3 - (n + 1) / a1 / (n - 1) + a2 / a1**2
    )

    if var_d <= 0:
        return 0.0

    d = (theta_pi - theta_w) / np.sqrt(var_d)
    return round(float(d), 4)


def linkage_disequilibrium(genotype_matrix, max_pairs=1000):
    """Calculate pairwise LD (r-squared) between sites.

    Args:
        genotype_matrix: numpy array (n_samples x n_sites).
        max_pairs: maximum number of site pairs to compute.

    Returns:
        Dict of (site_i, site_j) -> r2.
    """
    n_sites = genotype_matrix.shape[1]
    ld = {}

    pairs = []
    step = max(1, n_sites // int(np.sqrt(max_pairs)))
    for i in range(0, n_sites, step):
        for j in range(i + 1, min(i + 20, n_sites)):
            pairs.append((i, j))
    pairs = pairs[:max_pairs]

    for i, j in pairs:
        a = genotype_matrix[:, i]
        b = genotype_matrix[:, j]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        ld[(i, j)] = round(r**2, 4)

    return ld


def pca_genotypes(genotype_matrix, n_components=2):
    """PCA of genotype data.

    Args:
        genotype_matrix: numpy array (n_samples x n_sites).
        n_components: number of principal components.

    Returns:
        Transformed coordinates (n_samples x n_components).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for PCA. "
            "Install with: pip install scikit-learn"
        )
    centered = genotype_matrix - genotype_matrix.mean(axis=0)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(centered)
    return coords, pca.explained_variance_ratio_


def full_analysis(genotype_matrix, population_labels=None):
    """Run complete population genetics analysis.

    Args:
        genotype_matrix: numpy array (n_samples x n_sites).
        population_labels: optional list of population IDs per sample.

    Returns:
        PopGenReport with all statistics.
    """
    report = PopGenReport(
        num_sites=genotype_matrix.shape[1],
        num_populations=len(np.unique(population_labels)) if population_labels is not None else 1
    )

    report.nucleotide_diversity = nucleotide_diversity(genotype_matrix)
    report.tajima_d = tajimas_d(genotype_matrix)

    # HWE for first site
    genotypes = genotype_matrix[:, 0]
    n_ref = (genotypes == 0).sum() * 2 + (genotypes == 1).sum()
    n_alt = (genotypes == 2).sum() * 2 + (genotypes == 1).sum()
    report.hw_test = hardy_weinberg_test({
        'AA': int((genotypes == 0).sum()),
        'Aa': int((genotypes == 1).sum()),
        'aa': int((genotypes == 2).sum())
    })

    # LD
    report.ld = linkage_disequilibrium(genotype_matrix, max_pairs=100)

    return report


def format_popgen_report(report):
    lines = [
        "=== Population Genetics Report ===",
        f"Sites: {report.num_sites}",
        f"Populations: {report.num_populations}",
        f"Nucleotide diversity (pi): {report.nucleotide_diversity:.6f}",
        f"Tajima's D: {report.tajima_d:.4f}",
    ]
    if report.hw_test:
        lines.append(f"HWE (site 0): chi2={report.hw_test.get('chi2', 0):.2f}, p={report.hw_test.get('p_value', 1):.4f}")
        lines.append(f"  In HWE: {'Yes' if report.hw_test.get('in_hwe') else 'No'}")
    if report.ld:
        avg_ld = np.mean(list(report.ld.values()))
        lines.append(f"Avg LD (r2): {avg_ld:.4f} across {len(report.ld)} pairs")
    if report.message:
        lines.append(f"Note: {report.message}")
    return '\n'.join(lines)
