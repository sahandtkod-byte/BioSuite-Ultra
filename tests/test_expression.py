"""Tests for biosuite.core.expression module (normalization functions)."""
import pytest
import numpy as np


@pytest.fixture
def counts_df():
    """Small count matrix for expression tests."""
    import pandas as pd
    np.random.seed(42)
    return pd.DataFrame(
        np.random.poisson(100, (10, 4)),
        columns=["sample1", "sample2", "sample3", "sample4"],
        index=[f"gene_{i}" for i in range(10)],
    )


class TestCpmNormalization:
    """Tests for cpm_normalization()."""

    def test_cpm_output_shape(self, counts_df):
        from biosuite.core.expression import cpm_normalization
        result = cpm_normalization(counts_df)
        assert result.shape == counts_df.shape

    def test_cpm_columns_preserved(self, counts_df):
        from biosuite.core.expression import cpm_normalization
        result = cpm_normalization(counts_df)
        assert list(result.columns) == list(counts_df.columns)

    def test_cpm_non_negative(self, counts_df):
        from biosuite.core.expression import cpm_normalization
        result = cpm_normalization(counts_df)
        assert (result >= 0).all().all()


class TestTpmNormalization:
    """Tests for tpm_normalization()."""

    def test_tpm_sums_to_million(self, counts_df):
        from biosuite.core.expression import tpm_normalization
        gene_lengths = np.ones(10) * 1000  # all genes same length
        result = tpm_normalization(counts_df, gene_lengths)
        # Each sample should sum to ~1e6
        for col in result.columns:
            assert abs(result[col].sum() - 1e6) < 1.0


class TestBenjaminiHochberg:
    """Tests for _benjamini_hochberg()."""

    def test_bh_output_shape(self):
        from biosuite.core.expression import _benjamini_hochberg
        pvals = np.array([0.01, 0.05, 0.1, 0.5, 0.9])
        result = _benjamini_hochberg(pvals)
        assert result.shape == pvals.shape

    def test_bh_monotone(self):
        """Adjusted p-values should be non-decreasing when input is sorted."""
        from biosuite.core.expression import _benjamini_hochberg
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        result = _benjamini_hochberg(pvals)
        # All should be >= original p-value
        assert np.all(result >= pvals - 1e-10)

    def test_bh_extreme_values(self):
        from biosuite.core.expression import _benjamini_hochberg
        pvals = np.array([0.0, 1.0])
        result = _benjamini_hochberg(pvals)
        assert len(result) == 2
