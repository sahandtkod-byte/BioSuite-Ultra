"""Tests for biosuite.core.codon_usage and bio_ml modules."""
import pytest
import numpy as np


class TestCodonUsage:
    """Tests for codon_usage.py functions."""

    def test_codon_usage_table(self):
        from biosuite.core.codon_usage import codon_usage_table
        result = codon_usage_table("ATGAAATTTTAA")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_kmer_composition(self):
        from biosuite.core.codon_usage import kmer_composition
        result = kmer_composition("ATCGATCG", k=2)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_sequence_complexity(self):
        from biosuite.core.codon_usage import sequence_complexity
        result = sequence_complexity("ATCGATCGATCG", window=4)
        assert isinstance(result, dict)
        assert "average_complexity" in result


class TestBioML:
    """Tests for bio_ml.py functions."""

    def test_roc_curve(self):
        from biosuite.core.bio_ml import compute_roc_curve
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85, 0.95, 0.15])
        result = compute_roc_curve(y_true, y_prob)
        assert result is not None


class TestValidators:
    """Tests for validators.py."""

    def test_validate_range_decorator(self):
        from biosuite.core.validators import validate_range
        @validate_range(min_val=0, max_val=100)
        def set_value(x):
            return x
        assert set_value(50) == 50
        with pytest.raises(ValueError):
            set_value(-1)
        with pytest.raises(ValueError):
            set_value(101)

    def test_retry_on_error(self):
        from biosuite.core.validators import retry_on_error
        call_count = 0
        @retry_on_error(max_retries=2, delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        result = flaky()
        assert result == "success"
        assert call_count == 3

    def test_safe_execute(self):
        from biosuite.core.validators import safe_execute
        result, err = safe_execute(lambda: 42)
        assert result == 42
        assert err is None

        result, err = safe_execute(lambda: 1/0, _default="error")
        assert result == "error"
        assert err is not None
