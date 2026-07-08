"""
Metabolomics analysis: peak detection, feature alignment, statistics.

Pure Python implementations using numpy/pandas/scipy.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import signal as sp_signal
from scipy import stats as sp_stats


@dataclass
class MetaboliteFeature:
    mz: float
    rt: float
    intensity: float
    peak_area: float = 0.0
    snr: float = 0.0
    annotation: str = ""


@dataclass
class MetabolomicsReport:
    total_features: int = 0
    detected_peaks: int = 0
    aligned_features: pd.DataFrame = None
    pca_coords: np.ndarray = None
    anova_results: pd.DataFrame = None
    message: str = ""


def detect_peaks(intensity_array, sampling_rate=1.0, min_snr=3.0,
                 min_peak_width=5, prominence=None):
    """Detect peaks in a chromatogram or mass spectrum.

    Args:
        intensity_array: 1D numpy array of intensity values.
        sampling_rate: data points per unit time.
        min_snr: minimum signal-to-noise ratio.
        min_peak_width: minimum peak width in data points.
        prominence: minimum peak prominence (auto-calculated if None).

    Returns:
        List of MetaboliteFeature objects.
    """
    if prominence is None:
        prominence = np.std(intensity_array) * 1.5

    peaks, properties = sp_signal.find_peaks(
        intensity_array,
        prominence=prominence,
        width=min_peak_width,
        height=min_snr
    )

    features = []
    noise_level = np.median(np.abs(np.diff(intensity_array)))
    for i, peak_idx in enumerate(peaks):
        snr_val = intensity_array[peak_idx] / max(noise_level, 1e-10)
        features.append(MetaboliteFeature(
            mz=0.0,
            rt=float(peak_idx) / sampling_rate,
            intensity=float(intensity_array[peak_idx]),
            peak_area=float(properties.get('widths', [1])[i] if 'widths' in properties else 1),
            snr=round(float(snr_val), 1)
        ))

    return features


def detect_features_from_matrix(intensity_matrix, mz_values=None, rt_values=None,
                                 min_snr=3.0):
    """Detect features across multiple samples from a 2D intensity matrix.

    Args:
        intensity_matrix: 2D numpy array (samples x mz_bins or samples x timepoints).
        mz_values: optional mz values for each column.
        rt_values: optional retention time values for each column.
        min_snr: minimum signal-to-noise ratio.

    Returns:
        DataFrame of detected features.
    """
    all_features = []
    for sample_idx in range(intensity_matrix.shape[0]):
        sample = intensity_matrix[sample_idx]
        peaks = detect_peaks(sample, min_snr=min_snr)
        for p in peaks:
            p.mz = mz_values[int(p.rt)] if mz_values and int(p.rt) < len(mz_values) else 0
            all_features.append({
                'sample': sample_idx,
                'mz': p.mz,
                'rt': p.rt,
                'intensity': p.intensity,
                'snr': p.snr
            })

    return pd.DataFrame(all_features)


def align_features(feature_dfs, mz_tolerance=0.01, rt_tolerance=30):
    """Align detected features across samples.

    Args:
        feature_dfs: list of DataFrames, one per sample.
        mz_tolerance: mz tolerance for matching.
        rt_tolerance: RT tolerance for matching.

    Returns:
        Aligned feature matrix (samples x features).
    """
    if not feature_dfs:
        return pd.DataFrame()

    all_features = []
    for i, df in enumerate(feature_dfs):
        for _, row in df.iterrows():
            all_features.append({'sample': i, 'mz': row['mz'], 'rt': row['rt'],
                                'intensity': row['intensity']})

    if not all_features:
        return pd.DataFrame()

    all_df = pd.DataFrame(all_features)

    # Simple binning alignment
    all_df['mz_bin'] = (all_df['mz'] / mz_tolerance).round().astype(int)
    all_df['rt_bin'] = (all_df['rt'] / rt_tolerance).round().astype(int)

    pivot = all_df.pivot_table(values='intensity', index=['mz_bin', 'rt_bin'],
                               columns='sample', aggfunc='max', fill_value=0)
    return pivot


def anova_test(feature_matrix, group_labels):
    """Perform ANOVA test across groups for each feature.

    Args:
        feature_matrix: numpy array (samples x features).
        group_labels: list of group identifiers per sample.

    Returns:
        DataFrame with F-statistic and p-value per feature.
    """
    unique_groups = list(set(group_labels))
    n_features = feature_matrix.shape[1]
    results = []

    for j in range(n_features):
        groups = [feature_matrix[np.array(group_labels) == g, j] for g in unique_groups]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            results.append({'feature': j, 'f_statistic': 0, 'p_value': 1.0})
            continue
        f_stat, p_val = sp_stats.f_oneway(*groups)
        results.append({'feature': j, 'f_statistic': f_stat, 'p_value': p_val})

    return pd.DataFrame(results)


def pca_feature_matrix(feature_matrix, n_components=2):
    """PCA of metabolomics feature matrix.

    Args:
        feature_matrix: numpy array (samples x features).
        n_components: number of PCs.

    Returns:
        PC coordinates and explained variance.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for PCA. "
            "Install with: pip install scikit-learn"
        )
    centered = feature_matrix - feature_matrix.mean(axis=0)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(centered)
    return coords, pca.explained_variance_ratio_


def format_metabolomics_report(report):
    lines = [
        "=== Metabolomics Report ===",
        f"Features detected: {report.total_features}",
        f"Detected peaks: {report.detected_peaks}",
    ]
    if report.anova_results is not None and not report.anova_results.empty:
        sig = (report.anova_results['p_value'] < 0.05).sum()
        lines.append(f"Significant features (ANOVA p<0.05): {sig}")
    if report.message:
        lines.append(f"Note: {report.message}")
    return '\n'.join(lines)
