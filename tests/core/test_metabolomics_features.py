"""The metabolomics feature surface of biosuite.core.metabolomics.

Feature and report records, peak detection in a chromatogram, features from an
intensity matrix, cross-sample feature alignment, the group statistics the
differential test runs, and the text the report prints — plus the one-call
pipeline that strings them together.
"""
import os
import numpy as np
import pandas as pd
import pytest


class TestMetabolomicsDataclasses:
    """MetaboliteFeature and MetabolomicsReport construction."""

    def test_metabolite_feature_defaults(self):
        from biosuite.core.metabolomics import MetaboliteFeature
        f = MetaboliteFeature(mz=100.5, rt=12.3, intensity=1e6)
        assert f.mz == 100.5
        assert f.rt == 12.3
        assert f.intensity == 1e6
        assert f.peak_area == 0.0
        assert f.snr == 0.0
        assert f.annotation == ""

    def test_metabolite_feature_custom(self):
        from biosuite.core.metabolomics import MetaboliteFeature
        f = MetaboliteFeature(mz=200.0, rt=5.0, intensity=500, peak_area=10, snr=7.5, annotation="glucose")
        assert f.annotation == "glucose"
        assert f.snr == 7.5

    def test_metabolomics_report_defaults(self):
        from biosuite.core.metabolomics import MetabolomicsReport
        r = MetabolomicsReport()
        assert r.total_features == 0
        assert r.detected_peaks == 0
        assert r.aligned_features is None
        assert r.message == ""


class TestDetectPeaks:
    """detect_peaks function."""

    def test_single_prominent_peak(self):
        from biosuite.core.metabolomics import detect_peaks
        arr = np.zeros(200)
        arr[100:110] = 1000  # sharp peak
        features = detect_peaks(arr, min_snr=3, min_peak_width=3, prominence=10)
        assert isinstance(features, list)
        # At least one peak should be found near index 105
        assert len(features) >= 1

    def test_no_peaks_flat_signal(self):
        from biosuite.core.metabolomics import detect_peaks
        arr = np.ones(200) * 5.0
        features = detect_peaks(arr, min_snr=3, min_peak_width=5, prominence=100)
        # Very flat data with high prominence threshold → no peaks
        assert features == []

    def test_peak_snr_positive(self):
        from biosuite.core.metabolomics import detect_peaks
        arr = np.random.normal(0, 1, 500)
        arr[250:260] = 200
        features = detect_peaks(arr, min_snr=3, min_peak_width=3, prominence=1)
        for f in features:
            assert f.snr > 0

    def test_return_type_metabolite_feature(self):
        from biosuite.core.metabolomics import MetaboliteFeature, detect_peaks
        arr = np.zeros(200)
        arr[95:110] = 500
        features = detect_peaks(arr, min_peak_width=3, prominence=1)
        assert len(features) == 1
        assert isinstance(features[0], MetaboliteFeature)
        assert features[0].snr > 0


class TestDetectFeaturesFromMatrix:
    def test_basic_matrix(self):
        from biosuite.core.metabolomics import detect_features_from_matrix
        mat = np.random.normal(0, 1, (5, 200))
        mat[:, 100:105] = 500
        df = detect_features_from_matrix(mat, min_snr=1)
        assert isinstance(df, pd.DataFrame)
        # Columns should include expected fields
        for col in ['sample', 'mz', 'rt', 'intensity', 'snr']:
            assert col in df.columns

    def test_with_mz_values(self):
        from biosuite.core.metabolomics import detect_features_from_matrix
        mat = np.random.default_rng(1).normal(0, 1, (3, 100))
        mat[:, 50:55] = 800
        mz = list(np.linspace(50, 500, 100))
        df = detect_features_from_matrix(mat, mz_values=mz, min_snr=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # one feature per sample
        assert (df['mz'] >= 50).all() and (df['mz'] <= 500).all()


class TestAlignFeatures:
    def test_empty_input(self):
        from biosuite.core.metabolomics import align_features
        result = align_features([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_basic_alignment(self):
        from biosuite.core.metabolomics import align_features
        df1 = pd.DataFrame({'mz': [100, 200], 'rt': [10, 20], 'intensity': [1e5, 2e5]})
        df2 = pd.DataFrame({'mz': [100, 250], 'rt': [10, 30], 'intensity': [1.1e5, 3e5]})
        result = align_features([df1, df2], mz_tolerance=1, rt_tolerance=1)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestAnovaTest:
    def test_two_group_comparison(self):
        from biosuite.core.metabolomics import anova_test
        np.random.seed(42)
        X = np.vstack([
            np.random.normal(5, 1, (10, 5)),
            np.random.normal(10, 1, (10, 5)),
        ])
        labels = ['A'] * 10 + ['B'] * 10
        result = anova_test(X, labels)
        assert isinstance(result, pd.DataFrame)
        assert 'f_statistic' in result.columns
        assert 'p_value' in result.columns
        assert len(result) == 5

    def test_single_group(self):
        from biosuite.core.metabolomics import anova_test
        X = np.random.normal(5, 1, (5, 3))
        labels = ['A'] * 5
        result = anova_test(X, labels)
        # Single group → p_value should be 1.0
        assert all(result['p_value'] == 1.0)


class TestPCAFeatureMatrix:
    def test_basic_pca(self):
        from biosuite.core.metabolomics import pca_feature_matrix
        X = np.random.normal(0, 1, (20, 10))
        coords, var = pca_feature_matrix(X, n_components=2)
        assert coords.shape == (20, 2)
        assert len(var) == 2
        assert sum(var) <= 1.0 + 1e-6


class TestFormatMetabolomicsReport:
    def test_basic_format(self):
        from biosuite.core.metabolomics import (
            MetabolomicsReport,
            format_metabolomics_report,
        )
        report = MetabolomicsReport(total_features=100, detected_peaks=42, message="test note")
        text = format_metabolomics_report(report)
        assert "Metabolomics Report" in text
        assert "100" in text
        assert "test note" in text

    def test_with_anova_results(self):
        from biosuite.core.metabolomics import (
            MetabolomicsReport,
            format_metabolomics_report,
        )
        anova_df = pd.DataFrame({'feature': [0, 1], 'f_statistic': [5.0, 1.0], 'p_value': [0.01, 0.2]})
        report = MetabolomicsReport(total_features=10, anova_results=anova_df)
        text = format_metabolomics_report(report)
        assert "Significant features" in text


class TestMetabolomicsPipeline:
    """End-to-end metabolomics workflow."""

    def test_peak_detection_to_anova(self):
        from biosuite.core.metabolomics import (
            anova_test,
            pca_feature_matrix,
        )
        np.random.seed(42)
        # Simulate 2 groups of 5 samples each, 10 features
        group_a = np.random.normal(5, 1, (5, 10))
        group_b = np.random.normal(8, 1, (5, 10))
        X = np.vstack([group_a, group_b])
        labels = ['A'] * 5 + ['B'] * 5

        # ANOVA
        anova = anova_test(X, labels)
        assert len(anova) == 10
        # PCA
        coords, var = pca_feature_matrix(X, n_components=2)
        assert coords.shape == (10, 2)
