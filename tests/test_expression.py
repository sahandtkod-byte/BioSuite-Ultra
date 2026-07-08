"""
Unit tests for biosuite.core.expression module.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.expression import (
    read_counts_matrix, cpm_normalization, tpm_normalization,
    differential_expression, _benjamini_hochberg
)


# ─── read_counts_matrix ──────────────────────────────────────────────────────

class TestReadCountsMatrix:
    def test_csv(self, tmp_path):
        f = tmp_path / "counts.csv"
        f.write_text("gene,S1,S2\nG1,10,20\nG2,30,40\n")
        df = read_counts_matrix(str(f))
        assert df is not None
        assert df.shape[0] == 2
        assert 'gene' in df.columns

    def test_tsv(self, tmp_path):
        f = tmp_path / "counts.tsv"
        f.write_text("gene\tS1\tS2\nG1\t10\t20\nG2\t30\t40\n")
        df = read_counts_matrix(str(f))
        assert df is not None
        assert df.shape[0] == 2

    def test_nonexistent(self):
        assert read_counts_matrix("nonexistent.csv") is None


# ─── cpm_normalization ───────────────────────────────────────────────────────

class TestCPMNormalization:
    def test_basic(self):
        df = pd.DataFrame({'gene': ['G1', 'G2'], 'S1': [10, 30], 'S2': [20, 40]})
        cpm = cpm_normalization(df)
        # Column sums: S1=40, S2=60
        # G1/S1 = 10/40 * 1e6 = 250000
        assert cpm.loc[0, 'S1'] == pytest.approx(250000.0)
        # G1/S2 = 20/60 * 1e6 = 333333.33
        assert cpm.loc[0, 'S2'] == pytest.approx(333333.33, rel=1e-4)

    def test_column_sums_to_1e6(self):
        df = pd.DataFrame({'gene': ['G1', 'G2', 'G3'],
                           'S1': [100, 200, 300], 'S2': [50, 100, 150]})
        cpm = cpm_normalization(df)
        # Each column should sum to 1e6
        assert cpm['S1'].sum() == pytest.approx(1e6, rel=1e-6)
        assert cpm['S2'].sum() == pytest.approx(1e6, rel=1e-6)

    def test_preserves_gene_column(self):
        df = pd.DataFrame({'gene': ['G1', 'G2'], 'S1': [10, 20], 'S2': [30, 40]})
        cpm = cpm_normalization(df)
        # Gene column should not be in CPM output (it's not numeric)
        assert 'gene' not in cpm.columns


# ─── tpm_normalization ───────────────────────────────────────────────────────

class TestTPMNormalization:
    def test_basic(self):
        df = pd.DataFrame({'gene': ['G1', 'G2'], 'S1': [10, 20], 'S2': [30, 40]})
        lengths = pd.Series([1000, 2000])
        tpm = tpm_normalization(df, lengths)
        # RPK: G1/S1 = 10/1000*1000 = 10, G2/S1 = 20/2000*1000 = 10
        # Sum RPK S1 = 20, scaling = 20/1e6 = 0.00002
        # TPM G1/S1 = 10/0.00002 = 500000
        assert tpm.loc[0, 'S1'] == pytest.approx(500000.0)

    def test_tpm_sums_to_1e6(self):
        df = pd.DataFrame({'gene': ['G1', 'G2'], 'S1': [100, 200], 'S2': [50, 100]})
        lengths = pd.Series([1000, 1000])
        tpm = tpm_normalization(df, lengths)
        assert tpm['S1'].sum() == pytest.approx(1e6, rel=1e-6)
        assert tpm['S2'].sum() == pytest.approx(1e6, rel=1e-6)


# ─── differential_expression ─────────────────────────────────────────────────

class TestDifferentialExpression:
    def _make_df(self, n_genes=100, seed=42):
        np.random.seed(seed)
        return pd.DataFrame({
            'gene': [f'G{i}' for i in range(n_genes)],
            'ctrl1': np.random.randint(100, 1000, n_genes),
            'ctrl2': np.random.randint(100, 1000, n_genes),
            'treat1': np.random.randint(100, 1000, n_genes),
            'treat2': np.random.randint(100, 1000, n_genes),
        })

    def test_basic_output(self):
        df = self._make_df()
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert isinstance(result, pd.DataFrame)
        assert 'gene' in result.columns
        assert 'log2FC' in result.columns
        assert 'pvalue' in result.columns
        assert len(result) == 100

    def test_pvalues_between_0_and_1(self):
        df = self._make_df()
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert (result['pvalue'] >= 0).all()
        assert (result['pvalue'] <= 1).all()

    def test_log2fc_finite(self):
        df = self._make_df()
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert np.all(np.isfinite(result['log2FC']))

    def test_three_groups_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="Only two groups"):
            differential_expression(df, ['A', 'B', 'C', 'A'])

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({'gene': ['G1', 'G2'], 'name': ['a', 'b']})
        with pytest.raises(ValueError, match="No numeric columns"):
            differential_expression(df, ['A', 'B'])

    def test_vectorized_speed(self):
        """DE on 2000 genes should complete in under 1 second."""
        df = self._make_df(n_genes=2000)
        import time
        start = time.time()
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        elapsed = time.time() - start
        assert elapsed < 1.0
        assert len(result) == 2000

    def test_gene_column_preserved(self):
        df = self._make_df(n_genes=10)
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert list(result['gene']) == [f'G{i}' for i in range(10)]

    def test_has_padj_column(self):
        df = self._make_df(n_genes=10)
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert 'padj' in result.columns

    def test_padj_between_0_and_1(self):
        df = self._make_df(n_genes=50)
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert (result['padj'] >= 0).all()
        assert (result['padj'] <= 1).all()

    def test_padj_greater_or_equal_pvalue(self):
        """BH correction should produce adjusted p >= raw p."""
        df = self._make_df(n_genes=100)
        result = differential_expression(df, ['ctrl', 'ctrl', 'treat', 'treat'])
        assert (result['padj'] >= result['pvalue'] - 1e-10).all()


# ─── Benjamini-Hochberg ──────────────────────────────────────────────────────

class TestBenjaminiHochberg:
    def test_all_zeros(self):
        pvals = np.array([0.0, 0.0, 0.0])
        result = _benjamini_hochberg(pvals)
        assert np.all(result == 0.0)

    def test_all_ones(self):
        pvals = np.array([0.5, 0.8, 1.0])
        result = _benjamini_hochberg(pvals)
        assert np.all(result <= 1.0)

    def test_monotonicity(self):
        """Adjusted p-values should not increase when sorted."""
        pvals = np.array([0.01, 0.05, 0.1, 0.5])
        result = _benjamini_hochberg(pvals)
        sorted_result = np.sort(result)
        assert np.allclose(result, sorted_result)

    def test_length_preserved(self):
        pvals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = _benjamini_hochberg(pvals)
        assert len(result) == len(pvals)

    def test_empty(self):
        result = _benjamini_hochberg(np.array([]))
        assert len(result) == 0
