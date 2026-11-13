"""`detect_features_from_matrix` axis mapping and `align_features` input guards.

The detection step hands the alignment step whatever it found, so both are driven
here with real peak-shaped arrays and with the malformed inputs a GUI or notebook
can produce.  Every number below was read off the running code.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core.metabolomics import align_features, detect_features_from_matrix

X_AXIS = np.arange(40)
# A broad Gaussian apex at column 20, wide enough to clear min_peak_width=5.
SIGNAL = 100.0 * np.exp(-0.5 * ((X_AXIS - 20) / 3.0) ** 2)
ONE_SAMPLE = SIGNAL.reshape(1, -1)
RT_BY_COLUMN = [50.0 + i * 0.1 for i in range(40)]
MZ_BY_COLUMN = [1000 + i for i in range(40)]


def test_a_named_retention_time_replaces_the_column_index():
    """`rt_values` was accepted and never read: a peak at column 20 came back as
    rt=20.0 while rt_values[20] said 52.0."""
    found = detect_features_from_matrix(ONE_SAMPLE, mz_values=MZ_BY_COLUMN,
                                        rt_values=RT_BY_COLUMN)
    assert found[['sample', 'mz', 'rt']].values.tolist() == [[0, 1020.0, 52.0]]
    assert found['intensity'].iloc[0] == pytest.approx(100.0, rel=1e-6)
    assert found['snr'].iloc[0] > 3.0


def test_without_rt_values_the_column_index_is_reported_as_rt():
    """Positive control for the branch above: detect_peaks runs at sampling_rate
    1.0, so its peak position is the column number."""
    found = detect_features_from_matrix(ONE_SAMPLE, mz_values=MZ_BY_COLUMN)
    assert found[['mz', 'rt']].values.tolist() == [[1020.0, 20.0]]


def test_an_array_of_axis_values_is_accepted():
    """The guard was `if mz_values and ...`, and bool(numpy_array) raises
    'The truth value of an array with more than one element is ambiguous'."""
    found = detect_features_from_matrix(ONE_SAMPLE, mz_values=np.arange(40) + 1000,
                                       rt_values=np.arange(40) * 0.5)
    assert found[['mz', 'rt']].values.tolist() == [[1020.0, 10.0]]


def test_a_flat_matrix_with_no_peak_yields_no_rows():
    found = detect_features_from_matrix(np.zeros((2, 40)), mz_values=MZ_BY_COLUMN,
                                        rt_values=RT_BY_COLUMN)
    assert found.empty and list(found.columns) == []


@pytest.mark.parametrize("name", ["mz_values", "rt_values"])
def test_an_axis_shorter_than_the_matrix_is_refused(name):
    """Out-of-range peaks used to be written as mz=0 rather than reported, which
    turned a mismatched axis into a column of plausible-looking zeros."""
    values = [1.0, 2.0, 3.0] if name == "mz_values" else [1.0, 2.0, 3.0]
    with pytest.raises(ValueError) as exc:
        detect_features_from_matrix(ONE_SAMPLE, **{name: values})
    assert str(exc.value) == (
        f"{name} has 3 entries but the intensity matrix has 40 columns; "
        "they must match one-to-one.")


def test_a_one_dimensional_matrix_is_named_as_the_wrong_shape():
    with pytest.raises(ValueError,
                       match=r"intensity_matrix must be 2-D \(samples x columns\), got 1-D\."):
        detect_features_from_matrix(np.arange(10.0))


def test_a_list_of_rows_is_read_as_a_matrix():
    found = detect_features_from_matrix([list(SIGNAL)], mz_values=MZ_BY_COLUMN,
                                        rt_values=RT_BY_COLUMN)
    assert found[['mz', 'rt']].values.tolist() == [[1020.0, 52.0]]


@pytest.mark.parametrize("name, value", [
    ("mz_tolerance", 0), ("mz_tolerance", -0.5),
    ("rt_tolerance", 0), ("rt_tolerance", -1),
])
def test_a_non_positive_tolerance_is_refused_before_it_becomes_a_nan_bin(name, value):
    """Dividing by zero produced NaN bins and then an opaque
    'Cannot convert non-finite values (NA or inf) to integer' from pandas."""
    table = pd.DataFrame({'mz': [100.0], 'rt': [1.0], 'intensity': [1e5]})
    with pytest.raises(ValueError,
                       match=rf"{name} must be positive, got {value}\."):
        align_features([table], **{name: value})
