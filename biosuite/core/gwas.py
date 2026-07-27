"""
GWAS analysis — genome-wide association studies.
Chi-squared test, Manhattan/QQ plots, lead SNP detection.
Pure Python/numpy/scipy implementation.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2


def gwas_chi_squared(controls_alt_count: int, cases_alt_count: int, controls_total: int, cases_total: int) -> dict:
    """Perform chi-squared test for one SNP.

    Args:
        controls_alt_count: number of alt alleles in controls.
        cases_alt_count: number of alt alleles in cases.
        controls_total: total allele count in controls.
        cases_total: total allele count in cases.

    Returns:
        dict with odds_ratio, p_value, chi2_stat.
    """
    ref_controls = controls_total - controls_alt_count
    ref_cases = cases_total - cases_alt_count
    table = [[controls_alt_count, ref_controls],
             [cases_alt_count, ref_cases]]
    chi2_stat, p_val, dof, expected = chi2_contingency(table, correction=True)
    odds_ratio = (controls_alt_count * ref_cases) / max(ref_controls * cases_alt_count, 1)
    return {
        "chi2_stat": chi2_stat,
        "p_value": p_val,
        "odds_ratio": odds_ratio,
    }


def run_gwas(snp_data, case_col: str = "case", control_col: str = "control") -> pd.DataFrame:
    """Run GWAS on a DataFrame of SNP data.

    Args:
        snp_data: DataFrame with columns [chrom, pos, snp_id, case_alt, case_ref, ctrl_alt, ctrl_ref]
        case_col: prefix for case columns
        control_col: prefix for control columns

    Returns:
        DataFrame with GWAS results.
    """
    results = []
    for _, row in snp_data.iterrows():
        case_alt = row.get(f"{case_col}_alt", row.get("case_alt", 0))
        case_ref = row.get(f"{case_col}_ref", row.get("case_ref", 0))
        ctrl_alt = row.get(f"{control_col}_alt", row.get("ctrl_alt", 0))
        ctrl_ref = row.get(f"{control_col}_ref", row.get("ctrl_ref", 0))

        case_total = case_alt + case_ref
        ctrl_total = ctrl_alt + ctrl_ref

        if case_total == 0 or ctrl_total == 0:
            continue

        res = gwas_chi_squared(ctrl_alt, case_alt, ctrl_total, case_total)
        results.append({
            "chrom": row.get("chrom", ""),
            "pos": row.get("pos", 0),
            "snp_id": row.get("snp_id", ""),
            "case_alt": case_alt,
            "case_ref": case_ref,
            "ctrl_alt": ctrl_alt,
            "ctrl_ref": ctrl_ref,
            "odds_ratio": res["odds_ratio"],
            "chi2_stat": res["chi2_stat"],
            "p_value": res["p_value"],
            "neg_log10": -np.log10(res["p_value"] + 1e-300),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("p_value")
        df["p_adjusted"] = _benjamini_hochberg(df["p_value"].values)
    return df


def _benjamini_hochberg(p_values):
    n = len(p_values)
    ranked = np.argsort(p_values)
    adjusted = np.zeros(n)
    for i, r in enumerate(ranked):
        adjusted[r] = p_values[r] * n / (i + 1)
    adjusted = np.minimum(adjusted, 1.0)
    # Enforce monotonicity
    order = np.argsort(p_values)
    for i in range(n - 2, -1, -1):
        idx = order[i]
        next_idx = order[i + 1]
        adjusted[idx] = min(adjusted[idx], adjusted[next_idx])
    return adjusted


def detect_lead_snps(gwas_results, p_threshold: float = 5e-8, window_kb: int = 500) -> pd.DataFrame:
    """Detect lead (top) SNPs per locus.

    Args:
        gwas_results: DataFrame with chrom, pos, p_value columns.
        p_threshold: genome-wide significance threshold.
        window_kb: window size to define independent loci.

    Returns:
        DataFrame of lead SNPs.
    """
    sig = gwas_results[gwas_results["p_value"] < p_threshold].copy()
    if sig.empty:
        return pd.DataFrame()

    lead_snps = []
    for chrom, group in sig.groupby("chrom"):
        group = group.sort_values("pos")
        used = set()
        for _, row in group.iterrows():
            if row["snp_id"] in used:
                continue
            lead_snps.append(row)
            nearby = group[(group["pos"] >= row["pos"] - window_kb * 1000) &
                           (group["pos"] <= row["pos"] + window_kb * 1000)]
            used.update(nearby["snp_id"].tolist())

    return pd.DataFrame(lead_snps) if lead_snps else pd.DataFrame()


def generate_gwas_data(n_snps=5000, n_chromosomes=22, seed=42):
    """Generate simulated GWAS data for testing/demo.

    Returns:
        DataFrame with chrom, pos, snp_id, case_alt, case_ref, ctrl_alt, ctrl_ref.
    """
    np.random.seed(seed)
    data = []
    snp_idx = 0
    for chrom in range(1, n_chromosomes + 1):
        n_snps_chr = n_snps // n_chromosomes
        for i in range(n_snps_chr):
            snp_idx += 1
            pos = np.random.randint(1, 250_000_000)
            ctrl_alt = np.random.randint(5, 200)
            ctrl_ref = np.random.randint(5, 200)
            # Add some signals
            if chrom == 6 and 25_000_000 < pos < 35_000_000:
                case_alt = ctrl_alt + np.random.randint(20, 60)
            else:
                case_alt = ctrl_alt + np.random.randint(-10, 10)
            case_ref = max(5, ctrl_ref + np.random.randint(-10, 10))
            data.append({
                "chrom": f"chr{chrom}", "pos": pos,
                "snp_id": f"rs{snp_idx}",
                "case_alt": max(0, case_alt), "case_ref": max(0, case_ref),
                "ctrl_alt": ctrl_alt, "ctrl_ref": ctrl_ref,
            })
    return pd.DataFrame(data)


def format_gwas_report(gwas_results, lead_snps=None):
    """Format GWAS results as text report."""
    n_total = len(gwas_results)
    n_sig = len(gwas_results[gwas_results["p_value"] < 5e-8])
    n_suggestive = len(gwas_results[gwas_results["p_value"] < 1e-5])
    lines = [
        "GWAS Analysis Report",
        "=" * 40,
        f"Total SNPs tested: {n_total}",
        f"Genome-wide significant (p < 5e-8): {n_sig}",
        f"Suggestive (p < 1e-5): {n_suggestive}",
        "",
    ]
    if lead_snps is not None and not lead_snps.empty:
        lines.append("Lead SNPs:")
        for _, row in lead_snps.iterrows():
            lines.append(f"  {row['snp_id']} ({row['chrom']}:{row['pos']}) "
                         f"p={row['p_value']:.2e} OR={row['odds_ratio']:.2f}")
    else:
        lines.append("No significant loci detected.")
    return "\n".join(lines)
