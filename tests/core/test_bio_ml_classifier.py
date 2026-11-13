"""biosuite.core.bio_ml's classifier and its report.

Random-forest training through sklearn, the parameters it refuses, and the text the
report prints — skipped when sklearn is not installed.
"""
import numpy as np
import pytest


def _has_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        return True
    except ImportError:
        return False


class TestBioMLDataclass:
    def test_ml_result_defaults(self):
        from biosuite.core.bio_ml import MLResult
        r = MLResult(model_type='rf', engine='sklearn')
        assert r.accuracy == 0.0
        assert r.auc == 0.0
        assert r.cv_scores == []
        assert r.feature_importances == {}


class TestBioMLTrainClassifier:
    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_random_forest_basic(self):
        from biosuite.core.bio_ml import train_random_forest
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 10))
        y = np.array(["classA"] * 50 + ["classB"] * 50)
        result = train_random_forest(X, y, n_estimators=10)
        assert result.engine == 'sklearn'
        assert result.accuracy > 0
        assert len(result.cv_scores) > 0
        assert result.feature_importances != {}

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_random_forest_multiclass(self):
        from biosuite.core.bio_ml import train_random_forest
        np.random.seed(42)
        X = np.random.normal(0, 1, (90, 5))
        y = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
        result = train_random_forest(X, y, n_estimators=10)
        assert result.accuracy > 0

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_svm_basic(self):
        from biosuite.core.bio_ml import train_svm
        np.random.seed(42)
        X = np.random.normal(0, 1, (60, 5))
        y = np.array([0] * 30 + [1] * 30)
        result = train_svm(X, y, kernel='linear')
        assert result.engine == 'sklearn'
        assert result.accuracy > 0

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_random_forest_regressor(self):
        from biosuite.core.bio_ml import train_random_forest_regressor
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 5))
        y = X @ np.array([1, 2, 0, 0, 0]) + np.random.normal(0, 0.1, 50)
        result = train_random_forest_regressor(X, y, n_estimators=10)
        assert result.engine == 'sklearn'
        assert result.accuracy > 0  # R² score

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_compute_roc_curve(self):
        from biosuite.core.bio_ml import compute_roc_curve
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.4, 0.6, 0.9]
        roc = compute_roc_curve(y_true, y_prob)
        assert 'fpr' in roc
        assert 'tpr' in roc
        assert 'auc' in roc
        assert roc['auc'] > 0.5

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_select_features_importance(self):
        from biosuite.core.bio_ml import select_features
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 20))
        y = np.array([0] * 50 + [1] * 50)
        indices, importances = select_features(X, y, n_features=5, method='importance')
        assert len(indices) == 5
        assert len(importances) == 5

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_select_features_variance(self):
        from biosuite.core.bio_ml import select_features
        X = np.random.normal(0, 1, (50, 10))
        y = np.array([0] * 25 + [1] * 25)
        indices, importances = select_features(X, y, n_features=3, method='variance')
        assert len(indices) == 3


class TestBioMLFormatReport:
    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_format_report_basic(self):
        from biosuite.core.bio_ml import MLResult, format_ml_report
        r = MLResult(model_type='random_forest', engine='sklearn',
                     accuracy=0.95, auc=0.98, cv_scores=[0.93, 0.94, 0.95, 0.96, 0.97],
                     feature_importances={0: 0.5, 1: 0.3}, message="done")
        text = format_ml_report(r)
        assert "Machine Learning Report" in text
        assert "0.9500" in text
        assert "0.9800" in text
        assert "Feature 0" in text


class TestBioMLPipeline:
    """End-to-end ML workflow."""

    @pytest.mark.skipif(
        not _has_sklearn(), reason="scikit-learn not installed"
    )
    def test_train_and_report(self):
        from biosuite.core.bio_ml import format_ml_report, train_random_forest
        np.random.seed(42)
        X = np.random.normal(0, 1, (80, 10))
        y = np.array(["neg"] * 40 + ["pos"] * 40)
        result = train_random_forest(X, y, n_estimators=10)
        text = format_ml_report(result)
        assert "random_forest" in text
        assert "Accuracy" in text
