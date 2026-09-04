"""Regression tests for bio_ml honest-metrics overhaul."""
import numpy as np
import pytest

ml = pytest.importorskip("biosuite.core.bio_ml")


def _toy_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)   # learnable signal
    return X, y


def test_rf_uses_holdout_not_memorisation():
    """Held-out accuracy is reported — the API accepted test_size but
    silently ignored it and reported memorised-resubstitution metrics."""
    out = ml.train_random_forest(*_toy_dataset(), test_size=0.5)
    assert 0.5 <= out.accuracy <= 1.0
    assert "holdout" in out.message
    assert len(out.cv_scores) >= 2


def test_classification_report_with_numeric_labels():
    """Numeric class labels used to break target_names=le.classes_ (ints)."""
    out = ml.train_random_forest(*_toy_dataset())
    assert "0" in out.classification_rep and "1" in out.classification_rep


def test_svm_holdout_smoke():
    out = ml.train_svm(*_toy_dataset(), kernel='linear', test_size=0.4)
    assert 0.4 <= out.accuracy <= 1.0
    assert "holdout" in out.message


def test_rf_regressor_holdout():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] * 3 + rng.normal(0, 0.1, 80)
    out = ml.train_random_forest_regressor(X, y)
    assert out.accuracy > 0.5   # R² on holdout with strong signal
    assert "holdout" in out.message


def test_roc_curve_and_feature_selection():
    X, y = _toy_dataset()
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    roc = ml.compute_roc_curve(y, m.predict_proba(X)[:, 1])
    assert 0.9 <= roc['auc'] <= 1.0
    idx, imp = ml.select_features(X, y, n_features=3)
    assert len(idx) == 3 and len(imp) == 3
