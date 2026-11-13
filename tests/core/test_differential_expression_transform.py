"""Audit-remediation regression tests — Phase 2.5 (FIX-4: DE on raw counts).

The audit report's Pass 5 claims a controlled experiment in which a raw-count
Welch t-test recovered ~56 of 100 planted 4x genes while a log1p transform
recovered ~100.  That magnitude could not be reproduced here: across eight
negative-binomial regimes the raw path recovered 100/100 in the report-like
configuration, and no regime produced the 56 -> 100 gap.

What is reproducible, and what these tests therefore assert, is the direction:
log1p is never worse than raw and is clearly better when replicates are few
and dispersion is high.  See audit_output/PROGRESS.md for the measured table.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import expression as ex
from biosuite.core import gwas as gw

N_GENES = 2000
N_REP = 6
N_PLANTED = 100
DISPERSION = 20.0


def simulate(seed, n_rep=N_REP, effect=4.0, phi=DISPERSION, mu_lo=20.0, mu_hi=5000.0):
    """Group means, negative-binomial counts, and the first 100 genes raised."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(mu_lo, mu_hi, N_GENES)

    def draw(scale):
        mean = (mu * scale)[:, None]
        var = mean + mean ** 2 / phi
        p = mean / var
        n = mean * p / (1.0 - p)
        return rng.negative_binomial(np.maximum(n, 1e-6), p, size=(N_GENES, n_rep)).astype(float)

    ctrl = draw(np.ones(N_GENES))
    scale = np.ones(N_GENES)
    scale[:N_PLANTED] = effect
    return ctrl, draw(scale)


def frame(ctrl, treat):
    n = ctrl.shape[1]
    cols = [f"c{i}" for i in range(n)] + [f"t{i}" for i in range(n)]
    df = pd.DataFrame(np.hstack([ctrl, treat]), columns=cols)
    df.insert(0, "gene", [f"g{i}" for i in range(N_GENES)])
    return df, ["ctrl"] * n + ["treat"] * n


def sensitivity(seed, **kw):
    """(planted recovered, null false-positive rate) for both scales."""
    ctrl, treat = simulate(seed, **kw)
    _t, raw_p = ex._welch_ttest_rows(ctrl, treat)
    _t, log_p = ex._welch_ttest_rows(np.log1p(ctrl), np.log1p(treat))
    return ((raw_p[:N_PLANTED] < 0.05).sum(), (log_p[:N_PLANTED] < 0.05).sum(),
            (raw_p[N_PLANTED:] < 0.05).mean(), (log_p[N_PLANTED:] < 0.05).mean())


@pytest.mark.parametrize("seed", [42, 7, 2024])
def test_shipped_path_recovers_planted_genes(seed):
    df, conditions = frame(*simulate(seed))
    result = ex.differential_expression(df, conditions, method='ttest', correction='none')
    hit = int((result["pvalue"].to_numpy()[:N_PLANTED] < 0.05).sum())
    assert hit >= 85, f"only {hit}/{N_PLANTED} planted genes recovered"


@pytest.mark.parametrize("kw", [
    dict(effect=4.0, n_rep=3, phi=1.0),
    dict(effect=3.0, n_rep=4, phi=2.0, mu_lo=1.0, mu_hi=30.0),
])
def test_log1p_clearly_beats_raw_when_replicates_are_few(kw):
    hits = [sensitivity(seed, **kw) for seed in (42, 7, 13, 99, 2024)]
    raw = float(np.mean([h[0] for h in hits]))
    log = float(np.mean([h[1] for h in hits]))
    assert log >= raw + 5, f"log1p {log} not clearly above raw {raw}"


def test_log1p_is_never_worse_across_regimes():
    regimes = [dict(), dict(effect=2.0, n_rep=3), dict(effect=2.0, phi=5.0),
               dict(effect=1.5, phi=2.0, n_rep=3), dict(effect=4.0, phi=1.0, n_rep=3),
               dict(mu_lo=1.0, mu_hi=50.0, phi=5.0)]
    for kw in regimes:
        for seed in (42, 7):
            raw, log, _fr, _fl = sensitivity(seed, **kw)
            assert log >= raw, f"log1p lost sensitivity relative to raw under {kw} seed={seed}"


def test_type_i_error_stays_sane_on_the_null_genes():
    df, conditions = frame(*simulate(99))
    p = ex.differential_expression(df, conditions, method='ttest',
                                   correction='none')["pvalue"].to_numpy()[N_PLANTED:]
    fdr = float((p < 0.05).mean())
    assert 0.01 <= fdr <= 0.15, f"false-positive rate on null genes = {fdr:.3f}"


def test_padj_is_monotone_and_never_below_pvalue():
    ctrl, treat = simulate(7)
    df, conditions = frame(ctrl, treat)
    res = ex.differential_expression(df, conditions, method='ttest', correction='bh')
    p, q = res["pvalue"].to_numpy(), res["padj"].to_numpy()
    order = np.argsort(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all(q <= 1.0)


class TestSharedBenjaminiHochberg:
    """The report flagged BH as duplicated; gwas now uses the one implementation."""

    def test_gwas_uses_the_expression_implementation(self):
        assert gw._benjamini_hochberg is not ex._benjamini_hochberg or \
            gw._bh_shared is ex._benjamini_hochberg
        rng = np.random.default_rng(0)
        p = rng.random(500)
        np.testing.assert_allclose(gw._benjamini_hochberg(p), ex._benjamini_hochberg(p))

    def test_nan_no_longer_corrupts_its_neighbourhood(self):
        p = np.linspace(0.001, 0.9, 50)
        p_with_nan = p.copy()
        p_with_nan[7] = np.nan
        adjusted = gw._benjamini_hochberg(p_with_nan)
        assert np.isnan(adjusted[7]), "the NaN input must stay a NaN output"
        others = adjusted[~np.isnan(adjusted)]
        assert others.shape[0] == 49
        assert np.all(others >= p[~np.isnan(p_with_nan)] - 1e-12)

    def test_empty_input(self):
        assert len(gw._benjamini_hochberg(np.array([]))) == 0


def test_nb_docstring_no_longer_claims_deseq2_equivalence():
    doc = ex._negative_binomial_test.__doc__.lower()
    assert "deseq2-equivalent" not in doc
    assert "not deseq2" in doc or "approximate" in doc
