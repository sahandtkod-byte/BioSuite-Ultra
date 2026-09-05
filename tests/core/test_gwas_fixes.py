"""Regression tests for gwas.py fixes (OR inversion, lead-SNP choice)."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core.gwas import (
    gwas_chi_squared, run_gwas, detect_lead_snps, generate_gwas_data,
    _benjamini_hochberg,
)


def test_odds_ratio_direction_risk():
    # Alt allele enriched in cases -> OR > 1 (before the fix it was inverted)
    res = gwas_chi_squared(controls_alt_count=100, cases_alt_count=200,
                           controls_total=1000, cases_total=1000)
    assert res["odds_ratio"] > 1.0
    # hand value: (200*900)/(800*100) = 2.25
    assert res["odds_ratio"] == pytest.approx(2.25)


def test_odds_ratio_direction_protective():
    res = gwas_chi_squared(controls_alt_count=200, cases_alt_count=100,
                           controls_total=1000, cases_total=1000)
    assert res["odds_ratio"] < 1.0


def test_lead_snp_is_most_significant_not_leftmost():
    df = pd.DataFrame([
        {"chrom": "chr1", "pos": 1000, "snp_id": "left", "p_value": 1e-9},
        {"chrom": "chr1", "pos": 2000, "snp_id": "strong", "p_value": 1e-40},
        {"chrom": "chr1", "pos": 3000, "snp_id": "also", "p_value": 1e-10},
    ])
    leads = detect_lead_snps(df, p_threshold=1e-8, window_kb=1)
    assert list(leads["snp_id"]) == ["strong"]  # was "left" before the fix


def test_bh_monotonic_and_bounds():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    adj = _benjamini_hochberg(p)
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj <= 1.0)
    assert adj[0] == pytest.approx(0.005)  # 0.001*5/1


def test_generate_gwas_data_no_global_rng_side_effect():
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    generate_gwas_data(n_snps=100, n_chromosomes=2, seed=7)
    after = np.random.random()
    assert before == pytest.approx(after)  # global RNG state preserved


def test_run_gwas_end_to_end():
    data = generate_gwas_data(n_snps=804, n_chromosomes=2, seed=5)
    res = run_gwas(data)
    assert {"p_value", "p_adjusted", "odds_ratio"}.issubset(res.columns)
    # signal SNPs on the chr6 enriched region should mostly have OR>1
    sig = res[res["chrom"] == "chr6"]
    if len(sig):
        assert (sig["odds_ratio"] > 1).mean() > 0.8
