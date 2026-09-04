"""Regression tests for metabolism/metabolomics/epigenomics review fixes."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import metabolism, metabolomics, epigenomics


# ── metabolism ───────────────────────────────────────────────────────────────
def test_builtin_fba_maximises_not_minimises():
    # B produced by R1, consumed by R2; maximize R1 flux with bound 5.
    S = np.array([[1.0, -1.0]])
    res = metabolism._builtin_fba(stoich_matrix=S, flux_bounds=(0, 5))
    assert res.objective_value == pytest.approx(5.0)  # old sign gave 0.0/-0.0


def test_create_stoichiometric_matrix_signs():
    S = metabolism.create_stoichiometric_matrix(
        {'R1': {'substrates': [('A', 2)], 'products': [('B', 1)]}},
        ['A', 'B'],
    )
    assert S[0, 0] == -2 and S[1, 0] == 1


def test_knockout_without_cobra_returns_empty():
    if not metabolism.HAS_COBRA:
        assert metabolism.knockout_analysis("/nonexistent.sbml", ['g1']) == []


# ── metabolomics ─────────────────────────────────────────────────────────────
def test_detect_peaks_snr_semantics():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.5, 0.05, 400)
    noise[150] = 30.0  # one huge peak; old height=min_snr caught nothing <3 abs
    feats = metabolomics.detect_peaks(noise, min_snr=3.0, min_peak_width=1)
    assert any(abs(f.rt - 150) < 2 for f in feats)


def test_detect_peaks_area_is_area_not_width():
    rng = np.random.default_rng(1)
    arr = np.zeros(200) + 1.0
    arr[95:105] = 10.0
    arr += rng.normal(0, 0.01, 200)
    feats = metabolomics.detect_peaks(arr, min_snr=3.0, min_peak_width=3)
    assert feats, "no peaks detected"
    f = max(feats, key=lambda x: x.intensity)
    # triangle approx: width_pts (~10) * intensity (~10) / 2 ~ 50; must
    # not equal the raw width (~10) like before
    assert f.peak_area > 2 * f.intensity


def test_align_features_smoke():
    df1 = pd.DataFrame({'mz': [100.0, 200.0], 'rt': [10, 20], 'intensity': [5, 7]})
    df2 = pd.DataFrame({'mz': [100.001, 200.0], 'rt': [12, 22], 'intensity': [6, 8]})
    out = metabolomics.align_features([df1, df2], mz_tolerance=0.01, rt_tolerance=30)
    assert out.shape[1] == 2


# ── epigenomics ──────────────────────────────────────────────────────────────
def test_parse_and_levels(tmp_path):
    bed = tmp_path / "b.bed"
    bed.write_text("chr1\t100\t101\t18\t20\tCpG\nchr1\t200\t201\t2\t20\tCHH\n")
    sites = epigenomics.parse_bisulfite_bed(str(bed))
    assert len(sites) == 2 and abs(sites[0].methylation_level - 0.9) < 1e-9
    rep = epigenomics.calculate_methylation_levels(sites)
    assert rep.cpg_methylation == pytest.approx(0.9)
    assert rep.chh_methylation == pytest.approx(0.1)
    assert isinstance(rep.avg_methylation, float)


def test_dmr_fisher_significant():
    g1 = [epigenomics.MethylationSite('c1', 100, 'CpG', 0.9, 20, 18)]
    g2 = [epigenomics.MethylationSite('c1', 100, 'CpG', 0.2, 20, 4)]
    dmrs = epigenomics.find_dmrs(g1, g2, min_coverage=5)
    assert len(dmrs) == 1 and dmrs[0]['p_value'] < 0.05
    assert dmrs[0]['delta_methylation'] == pytest.approx(0.7)


def test_dmr_min_delta_filter():
    g1 = [epigenomics.MethylationSite('c1', 100, 'CpG', 0.55, 20, 11)]
    g2 = [epigenomics.MethylationSite('c1', 100, 'CpG', 0.45, 20, 9)]
    dmrs = epigenomics.find_dmrs(g1, g2, min_coverage=5, min_delta=0.2)
    assert dmrs == []


def test_atac_peaks_stats(tmp_path):
    p = tmp_path / "a.bed"
    p.write_text("chr1\t10\t50\nchr1\t100\t130\nchr2\t5\t25\n")
    peaks = epigenomics.parse_atac_peaks(str(p))
    stats = epigenomics.atac_peak_stats(peaks)
    assert stats['total_peaks'] == 3 and stats['median_length'] == 30
