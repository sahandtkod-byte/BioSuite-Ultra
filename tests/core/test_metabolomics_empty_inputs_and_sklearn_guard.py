"""
Empty feature tables and the optional scikit-learn backend in metabolomics.py.

``align_features`` is called with whatever the detection step produced, which
includes lists of feature tables that contain no rows at all; ``pca_feature_matrix``
imports scikit-learn inside the function and must say what is missing when that
extra is not installed.
"""
import sys

import numpy as np
import pandas as pd
import pytest

from biosuite.core.metabolomics import align_features, pca_feature_matrix

EMPTY_TABLE = pd.DataFrame({'mz': [], 'rt': [], 'intensity': []})


def test_alignment_of_rowless_feature_tables_keeps_one_zeroed_column_per_sample():
    aligned = align_features([EMPTY_TABLE, EMPTY_TABLE])
    assert aligned.empty
    assert list(aligned.columns) == [0, 1], "two samples in, two columns out"
    assert list(aligned.dtypes) == ["float64", "float64"]
    assert align_features([]).empty and list(align_features([]).columns) == []


def test_alignment_keeps_a_column_for_a_sample_that_found_nothing():
    """A sample with no rows contributed no value to pivot on and used to vanish,
    so aligning five samples could return four columns and leave the surviving ones
    labelled by the wrong sample index."""
    one = pd.DataFrame({'mz': [100.0], 'rt': [1.0], 'intensity': [1e5]})
    aligned = align_features([one, EMPTY_TABLE])
    assert list(aligned.columns) == [0, 1]
    assert list(aligned.index) == [(10000, 0)]   # mz / mz_tolerance, rt / rt_tolerance
    assert aligned.iloc[0, 0] == pytest.approx(1e5)
    assert aligned.iloc[0, 1] == 0.0
    assert aligned.dtypes.iloc[1] == np.dtype('float64'), "the filled column stays float"


def test_features_missing_from_a_sample_are_zero_filled():
    one = pd.DataFrame({'mz': [100.0], 'rt': [1.0], 'intensity': [1e5]})
    two = pd.DataFrame({'mz': [100.0, 100.4], 'rt': [1.0, 2.0], 'intensity': [1e5, 2e5]})
    aligned = align_features([one, two], mz_tolerance=0.5, rt_tolerance=1)
    assert list(aligned.columns) == [0, 1]
    assert aligned.values.tolist() == [[1e5, 1e5], [0.0, 2e5]]


def test_pca_names_the_missing_extra_and_keeps_its_cause(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
    with pytest.raises(ImportError) as exc:
        pca_feature_matrix(np.zeros((4, 3)))
    assert "scikit-learn is required for PCA" in str(exc.value)
    assert "pip install scikit-learn" in str(exc.value)
    assert isinstance(exc.value.__cause__, ImportError)


def test_pca_reduces_a_real_matrix(monkeypatch):
    """Positive control: the guard above must not disable the working path."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    coords, explained = pca_feature_matrix(matrix, n_components=1)
    assert coords.shape == (4, 1)
    assert explained.shape == (1,)
    assert explained[0] == pytest.approx(1.0)
