"""The Metabolomics tab: peak detection on a real intensity matrix, and the demo.

The handlers read the entries, call ``biosuite.core.metabolomics`` and write the
result into the tab's box.  Every expectation here is recomputed from the matrix
the tab itself parses (``_matrix``), so the tab is checked against the module it
wraps rather than against a remembered string, and the SNR and width entries are
set to thresholds the data cannot clear, which is how the tests know those values
reached ``detect_peaks`` instead of being decoration.
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest

from biosuite.core.metabolomics import (
    align_features,
    anova_test,
    detect_features_from_matrix,
    detect_peaks,
    format_metabolomics_report,
)

POINTS = 48
BUMP = slice(20, 28)


@pytest.fixture(autouse=True)
def _default_thresholds(ui):
    """The app is shared for the whole session; each test starts at the defaults."""
    ui.set("meta_snr", "3.0")
    ui.set("meta_width", "5")
    ui.set("meta_mz_tol", "0.01")
    ui.set("meta_rt_tol", "30")
    ui.clear("meta_groups")


def _row(rng, scale=1.0):
    """A noisy baseline with one chromatographic peak, so SNR and width mean
    something: a perfectly flat baseline makes the noise estimate zero and the
    SNR gate can then never bind."""
    values = 10.0 + rng.normal(0.0, 4.0, POINTS)
    values[BUMP] += 900.0 * scale
    return values


@pytest.fixture
def intensities_csv(tmp_path):
    """Two samples as rows and scan times as columns, written as plain decimals.

    ``repr()`` of a numpy scalar is ``np.float64(12.3)`` on numpy 2, which the tab
    cannot read back as a number, so the values are formatted explicitly.
    """
    rng = np.random.default_rng(7)
    rows = [_row(rng, 1.0), _row(rng, 0.5)]
    path = tmp_path / "intensities.csv"
    header = ",".join(f"t{i}" for i in range(POINTS))
    body = "\n".join(",".join(f"{v:.6f}" for v in row) for row in rows)
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")
    return path


def _matrix(path):
    """The tab's own view of the file: its numeric columns, one row per sample."""
    return pd.read_csv(str(path)).select_dtypes(include=[np.number]).to_numpy()


def _peaks(matrix, min_snr, min_width):
    found = []
    for row in matrix:
        found.extend(detect_peaks(row, min_snr=min_snr, min_peak_width=min_width))
    return found


def test_load_stores_the_chosen_matrix(ui, intensities_csv):
    ui.open_path = str(intensities_csv)

    ui.app._meta_load()

    assert ui.app.meta_path.get() == str(intensities_csv)
    assert ui.status() == f"Loaded: {intensities_csv.name}"


def test_load_keeps_the_entry_when_the_dialog_is_cancelled(ui):
    ui.set("meta_path", "keep-me.csv")
    before = ui.status()
    ui.open_path = ""

    ui.app._meta_load()

    assert ui.text("meta_path") == "keep-me.csv"
    assert ui.status() == before


def test_detect_without_a_matrix_asks_for_one(ui):
    ui.clear("meta_path")

    ui.app._meta_detect()

    assert ui.messages_of("warning") == ["Please load a CSV intensity matrix."]


def test_detect_counts_the_peaks_the_core_finds(ui, intensities_csv):
    matrix = _matrix(intensities_csv)
    assert matrix.shape == (2, POINTS), "the tab reads samples from rows"
    expected = _peaks(matrix, min_snr=3.0, min_width=5)
    assert len(expected) == 2,             f"both samples carry a peak at the default thresholds, got {len(expected)}"

    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_snr", "3.0")
    ui.set("meta_width", "5")
    ui.app._meta_detect()

    text = ui.text("meta_result")
    assert f"Detected {len(expected)} peaks across 2 samples" in text, text
    for feature in expected:
        assert f"RT={feature.rt:.1f}" in text, "the detected features should be listed"
    assert ui.messages == []
    assert ui.status() == f"Detected {len(expected)} peaks"


def test_the_snr_entry_reaches_detect_peaks(ui, intensities_csv):
    """A threshold the data cannot clear must drop every peak, or the entry is only
    decoration and the tab ignores what the user typed."""
    matrix = _matrix(intensities_csv)
    assert len(_peaks(matrix, min_snr=3.0, min_width=5)) == 2
    assert _peaks(matrix, min_snr=1e6, min_width=5) == []

    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_snr", "1000000")
    ui.set("meta_width", "5")
    ui.app._meta_detect()

    assert "Detected 0 peaks across 2 samples" in ui.text("meta_result")
    assert ui.status() == "Detected 0 peaks"


def test_the_width_entry_reaches_detect_peaks(ui, intensities_csv):
    """An eight-point peak cannot satisfy a sixty-point width requirement."""
    matrix = _matrix(intensities_csv)
    assert len(_peaks(matrix, min_snr=3.0, min_width=5)) == 2
    assert _peaks(matrix, min_snr=3.0, min_width=60) == []

    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_snr", "3.0")
    ui.set("meta_width", "60")
    ui.app._meta_detect()

    assert ui.status() == "Detected 0 peaks"


def test_empty_threshold_entries_fall_back_to_the_defaults(ui, intensities_csv):
    """``float(self.meta_snr.get() or 3.0)``: cleared entries must mean defaults,
    not a crash on an empty string."""
    expected = _peaks(_matrix(intensities_csv), min_snr=3.0, min_width=5)

    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_snr", "")
    ui.set("meta_width", "")
    ui.app._meta_detect()

    assert ui.status() == f"Detected {len(expected)} peaks"
    assert ui.messages == []


@pytest.mark.parametrize("snr,width,message", [
    ("abc", "5", "Min SNR must be a number and Min Peak Width a whole number."),
    ("3.0", "5.5", "Min SNR must be a number and Min Peak Width a whole number."),
    ("nan", "5", "Min SNR must be a non-negative finite number."),
    ("-1", "5", "Min SNR must be a non-negative finite number."),
    ("inf", "5", "Min SNR must be a non-negative finite number."),
    ("3.0", "0", "Min Peak Width must be at least 1."),
])
def test_a_threshold_that_is_not_a_usable_number_is_named(
        ui, intensities_csv, snr, width, message):
    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_snr", snr)
    ui.set("meta_width", width)
    ui.clear("meta_result")

    ui.app._meta_detect()

    assert ui.messages_of("error") == [message]
    assert ui.text("meta_result").strip() == ""


def test_detect_reports_a_file_it_cannot_read(ui, tmp_path):
    ui.set("meta_path", str(tmp_path / "absent.csv"))

    ui.app._meta_detect()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and "absent.csv" in errors[0], errors


def test_detect_names_a_file_without_numeric_columns(ui, tmp_path):
    """A file the tab cannot measure must be refused by name: reporting
    "Detected 0 peaks across 1 samples" would claim a scan that never happened."""
    ui.clear("meta_result")
    path = tmp_path / "notes.csv"
    path.write_text("note\nnothing measurable here\n", encoding="utf-8")
    ui.set("meta_path", str(path))

    ui.app._meta_detect()

    assert ui.messages_of("error") == ["The CSV holds no numeric intensity columns."]
    assert ui.text("meta_result").strip() == ""


def test_detect_names_a_header_only_matrix(ui, tmp_path):
    ui.clear("meta_result")
    path = tmp_path / "empty.csv"
    path.write_text("t0,t1,t2\n", encoding="utf-8")
    ui.set("meta_path", str(path))

    ui.app._meta_detect()

    assert ui.messages_of("error") == ["The CSV holds no numeric intensity columns."]
    assert ui.text("meta_result").strip() == ""


def test_the_demo_analysis_writes_its_report(ui):
    ui.clear("meta_result")

    ui.app._meta_demo()

    assert ui.text("meta_result").strip()
    assert ui.status() == "Demo metabolomics analysis complete"
    assert ui.messages == []


def test_the_demo_report_matches_the_core_formatter(ui, monkeypatch):
    """The demo builds a MetabolomicsReport itself; the box must show exactly what
    the module's own formatter returns for it."""
    from biosuite.core.metabolomics import MetabolomicsReport

    captured = {}
    real = format_metabolomics_report

    def spy(report):
        captured["report"] = report
        return real(report)

    monkeypatch.setattr("biosuite.core.metabolomics.format_metabolomics_report", spy)

    ui.app._meta_demo()

    assert isinstance(captured["report"], MetabolomicsReport)
    assert captured["report"].total_features == captured["report"].detected_peaks > 0
    assert ui.text("meta_result").strip() == real(captured["report"]).strip()


def test_the_demo_shows_a_core_failure_instead_of_a_stale_report(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("scipy is unwell")

    monkeypatch.setattr("biosuite.core.metabolomics.detect_peaks", broken)
    ui.clear("meta_result")

    ui.app._meta_demo()

    assert ui.messages_of("error") == ["scipy is unwell"]
    assert ui.text("meta_result").strip() == ""


@pytest.mark.parametrize("handler", ["_meta_demo", "_meta_pca"])
def test_the_demo_handlers_leave_the_global_random_state_alone(ui, handler):
    before = np.random.get_state()
    getattr(ui.app, handler)()
    after = np.random.get_state()

    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_pca_records_and_shows_a_figure(ui):
    before = len(ui.app._plot_history)

    ui.app._meta_pca()

    assert ui.plots, "the tab never asked for its figure to be displayed"
    title, fig = ui.plots[-1]
    assert title == "Metabolomics PCA", f"the window is titled after another tab: {title!r}"
    assert fig.axes[0].get_title() == "Metabolomics PCA"
    assert len(ui.app._plot_history) == before + 1
    assert ui.status() == "PCA plot displayed"


def test_the_pca_button_reports_a_core_failure(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no eigenvalues")

    monkeypatch.setattr("biosuite.core.metabolomics.pca_feature_matrix", broken)

    ui.app._meta_pca()

    assert ui.messages_of("error") == ["no eigenvalues"]
    assert ui.plots == [], "a failed PCA must not display a figure"


def test_pca_still_leaves_the_backend_alone(ui):
    """The tab used to run ``matplotlib.use("TkAgg")`` here, which switched the
    process-wide backend from a worker thread and froze the GUI."""
    before = matplotlib.get_backend()

    ui.app._meta_pca()

    assert matplotlib.get_backend() == before


def test_the_pca_points_use_the_active_theme_accent(ui):
    accent = ui.app.T['accent']

    ui.app._meta_pca()

    _title, fig = ui.plots[-1]
    face = fig.axes[0].collections[0].get_facecolor()[0]
    expected = tuple(round(int(accent[i:i + 2], 16) / 255, 3) for i in (1, 3, 5))
    assert tuple(round(float(value), 3) for value in face[:3]) == expected


@pytest.fixture
def flat_csv(tmp_path):
    path = tmp_path / "flat.csv"
    path.write_text("t0,t1,t2,t3\n0,0,0,0\n0,0,0,0\n", encoding="utf-8")
    return path


@pytest.fixture
def shifted_csv(tmp_path):
    """Two samples whose peaks sit far apart, so alignment tolerances bind."""
    rng = np.random.default_rng(3)
    row_a = 10.0 + rng.normal(0.0, 4.0, POINTS)
    row_a[10:18] += 900.0
    row_b = 10.0 + rng.normal(0.0, 4.0, POINTS)
    row_b[30:38] += 900.0
    path = tmp_path / "shifted.csv"
    header = ",".join(f"t{i}" for i in range(POINTS))
    body = "\n".join(",".join(f"{v:.6f}" for v in row) for row in (row_a, row_b))
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture
def four_samples_csv(tmp_path):
    rng = np.random.default_rng(11)
    rows = [_row(rng, scale) for scale in (1.0, 1.1, 0.5, 0.6)]
    path = tmp_path / "four.csv"
    header = ",".join(f"t{i}" for i in range(POINTS))
    body = "\n".join(",".join(f"{v:.6f}" for v in row) for row in rows)
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")
    return path


def _per_sample_features(matrix):
    return [detect_features_from_matrix(matrix[i:i + 1]) for i in range(matrix.shape[0])]


def test_features_report_matches_the_core(ui, intensities_csv):
    matrix = _matrix(intensities_csv)
    expected = detect_features_from_matrix(matrix)
    assert not expected.empty, "the fixture must carry detectable peaks"
    lines = [f"Detected {len(expected)} features across {matrix.shape[0]} samples",
             "", expected.head(10).to_string(index=False)]

    ui.set("meta_path", str(intensities_csv))
    ui.app._meta_features()

    assert ui.text("meta_result").strip() == "\n".join(lines)
    assert ui.status() == f"Detected {len(expected)} features"
    assert ui.messages == []


def test_features_say_when_no_peaks_pass(ui, flat_csv):
    ui.set("meta_path", str(flat_csv))
    ui.app._meta_features()

    assert "No peaks passed the detection thresholds." in ui.text("meta_result")
    assert ui.status() == "Detected 0 features"
    assert ui.messages == []


def test_features_without_a_matrix_ask_for_one(ui):
    ui.clear("meta_path")

    ui.app._meta_features()

    assert ui.messages_of("warning") == ["Please load a CSV intensity matrix."]


def test_features_report_a_file_they_cannot_read(ui, tmp_path):
    ui.set("meta_path", str(tmp_path / "absent.csv"))

    ui.app._meta_features()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and "absent.csv" in errors[0], errors


def test_features_name_a_file_without_numeric_columns(ui, tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("note\nnothing measurable here\n", encoding="utf-8")
    ui.set("meta_path", str(path))

    ui.app._meta_features()

    assert ui.messages_of("error") == ["The CSV holds no numeric intensity columns."]


def test_features_report_a_core_failure(ui, intensities_csv, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("matrix is unwell")

    monkeypatch.setattr("biosuite.core.metabolomics.detect_features_from_matrix", broken)
    ui.set("meta_path", str(intensities_csv))
    ui.clear("meta_result")

    ui.app._meta_features()

    assert ui.messages_of("error") == ["matrix is unwell"]
    assert ui.text("meta_result").strip() == ""


def test_align_report_matches_the_core(ui, intensities_csv):
    matrix = _matrix(intensities_csv)
    expected = align_features(_per_sample_features(matrix))
    assert not expected.empty, "the fixture must carry alignable peaks"
    lines = [f"Aligned matrix: {expected.shape[0]} features x {expected.shape[1]} samples",
             "", expected.head(10).to_string()]

    ui.set("meta_path", str(intensities_csv))
    ui.app._meta_align()

    assert ui.text("meta_result").strip() == "\n".join(lines)
    assert ui.status() == f"Aligned {expected.shape[0]} features"
    assert ui.messages == []


def test_the_tolerance_entries_reach_align_features(ui, shifted_csv):
    matrix = _matrix(shifted_csv)
    per_sample = _per_sample_features(matrix)
    tight = align_features(per_sample, rt_tolerance=0.1)
    wide = align_features(per_sample, rt_tolerance=1000.0)
    assert tight.shape[0] == 2, "far-apart peaks must stay separate at a tight tolerance"
    assert wide.shape[0] == 1, "a wide tolerance must merge them into one bin"

    ui.set("meta_path", str(shifted_csv))
    ui.set("meta_rt_tol", "1000")
    ui.app._meta_align()
    assert ui.status() == "Aligned 1 features"

    ui.set("meta_rt_tol", "0.1")
    ui.app._meta_align()
    assert ui.status() == "Aligned 2 features"
    assert ui.messages == []


def test_empty_tolerance_entries_fall_back_to_the_defaults(ui, intensities_csv):
    matrix = _matrix(intensities_csv)
    expected = align_features(_per_sample_features(matrix))

    ui.set("meta_path", str(intensities_csv))
    ui.clear("meta_mz_tol")
    ui.clear("meta_rt_tol")
    ui.app._meta_align()

    assert ui.status() == f"Aligned {expected.shape[0]} features"
    assert ui.messages == []


def test_a_tolerance_that_is_not_a_number_is_named(ui, intensities_csv):
    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_mz_tol", "abc")
    ui.clear("meta_result")

    ui.app._meta_align()

    assert ui.messages_of("error") == ["Tolerances must be numbers."]
    assert ui.text("meta_result").strip() == ""


def test_a_non_positive_tolerance_is_named_by_the_core(ui, intensities_csv):
    ui.set("meta_path", str(intensities_csv))
    ui.set("meta_mz_tol", "-1")
    ui.clear("meta_result")

    ui.app._meta_align()

    assert ui.messages_of("error") == ["mz_tolerance must be positive, got -1.0."]
    assert ui.text("meta_result").strip() == ""


def test_align_of_a_flat_matrix_reports_zero_features(ui, flat_csv):
    ui.set("meta_path", str(flat_csv))
    ui.app._meta_align()

    assert ui.text("meta_result").strip() == "Aligned matrix: 0 features x 2 samples"
    assert ui.status() == "Aligned 0 features"
    assert ui.messages == []


def test_align_without_a_matrix_asks_for_one(ui):
    ui.clear("meta_path")

    ui.app._meta_align()

    assert ui.messages_of("warning") == ["Please load a CSV intensity matrix."]


def test_anova_report_matches_the_core(ui, four_samples_csv):
    matrix = _matrix(four_samples_csv)
    labels = ["A", "A", "B", "B"]
    expected = anova_test(matrix, labels)
    significant = int((expected['p_value'] < 0.05).sum())
    lines = [
        f"ANOVA across 2 groups, {matrix.shape[1]} features",
        f"Significant features (p<0.05): {significant}",
        "",
        expected.to_string(index=False),
    ]

    ui.set("meta_path", str(four_samples_csv))
    ui.set("meta_groups", "A,A,B,B")
    ui.app._meta_anova()

    assert ui.text("meta_result").strip() == "\n".join(lines)
    assert ui.status() == f"ANOVA: {significant} significant features"
    assert ui.messages == []


@pytest.mark.parametrize("typed", ["", " , "])
def test_anova_without_labels_asks_for_them(ui, four_samples_csv, typed):
    ui.set("meta_path", str(four_samples_csv))
    ui.set("meta_groups", typed)
    ui.clear("meta_result")

    ui.app._meta_anova()

    assert ui.messages_of("warning") == [
        "Enter one group label per sample row, comma-separated."]
    assert ui.text("meta_result").strip() == ""


def test_anova_names_a_label_count_mismatch(ui, four_samples_csv):
    ui.set("meta_path", str(four_samples_csv))
    ui.set("meta_groups", "A,B")
    ui.clear("meta_result")

    ui.app._meta_anova()

    assert ui.messages_of("error") == ["Got 2 group labels for 4 samples."]
    assert ui.text("meta_result").strip() == ""


def test_anova_reports_a_core_failure(ui, four_samples_csv, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("groups are unwell")

    monkeypatch.setattr("biosuite.core.metabolomics.anova_test", broken)
    ui.set("meta_path", str(four_samples_csv))
    ui.set("meta_groups", "A,A,B,B")
    ui.clear("meta_result")

    ui.app._meta_anova()

    assert ui.messages_of("error") == ["groups are unwell"]
    assert ui.text("meta_result").strip() == ""


def test_anova_without_a_matrix_asks_for_one(ui):
    ui.clear("meta_path")

    ui.app._meta_anova()

    assert ui.messages_of("warning") == ["Please load a CSV intensity matrix."]
